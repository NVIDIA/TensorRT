#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import abc
import argparse
import gc
import json
import os
import pathlib
import sys
from abc import ABC, abstractmethod
from typing import Any, List

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import nvtx
import torch
from cuda.bindings import runtime as cudart
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DDPMWuerstchenScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from torch.utils.data import DataLoader

import demo_diffusion.engine as engine_module
import demo_diffusion.image as image_module
from demo_diffusion.model import (
    make_scheduler,
    merge_loras,
    unload_torch_model,
)
from demo_diffusion.pipeline.calibrate import load_calib_prompts
from demo_diffusion.pipeline.model_memory_manager import ModelMemoryManager
from demo_diffusion.pipeline.type import PIPELINE_TYPE
from demo_diffusion.utils_modelopt import (
    SD_FP8_BF16_FLUX_MMDIT_BMM2_FP8_OUTPUT_CONFIG,
    SD_FP8_FP16_DEFAULT_CONFIG,
    SD_FP8_FP32_DEFAULT_CONFIG,
    PromptImageDataset,
    SameSizeSampler,
    check_lora,
    custom_collate,
    filter_func,
    filter_func_no_proj_out,
    fp8_mha_disable,
    generate_fp8_scales,
    get_int8_config,
    infinite_dataloader,
    quantize_lvl,
    set_fmha,
    set_quant_precision,
)


class DiffusionPipeline(ABC):
    """
    Application showcasing the acceleration of Stable Diffusion pipelines using NVidia TensorRT.
    """
    VALID_DIFFUSION_PIPELINES = (
        "1.4",
        "1.5",
        "dreamshaper-7",
        "2.0-base",
        "2.0",
        "2.1-base",
        "2.1",
        "xl-1.0",
        "xl-turbo",
        "svd-xt-1.1",
        "sd3",
        "3.5-medium",
        "3.5-large",
        "cascade",
        "flux.1-dev",
        "flux.1-dev-canny",
        "flux.1-dev-depth",
        "flux.1-schnell",
        "flux.1-kontext-dev",
        "cosmos-predict2-2b-text2image",
        "cosmos-predict2-14b-text2image",
        "cosmos-predict2-2b-video2world",
        "cosmos-predict2-14b-video2world",
    )
    SCHEDULER_DEFAULTS = {
        "1.4": "PNDM",
        "1.5": "PNDM",
        "dreamshaper-7": "PNDM",
        "2.0-base": "DDIM",
        "2.0": "DDIM",
        "2.1-base": "PNDM",
        "2.1": "DDIM",
        "xl-1.0": "Euler",
        "xl-turbo": "EulerA",
        "3.5-large": "FlowMatchEuler",
        "3.5-medium": "FlowMatchEuler",
        "svd-xt-1.1": "Euler",
        "cascade": "DDPMWuerstchen",
        "flux.1-dev": "FlowMatchEuler",
        "flux.1-dev-canny": "FlowMatchEuler",
        "flux.1-dev-depth": "FlowMatchEuler",
        "flux.1-schnell": "FlowMatchEuler",
        "flux.1-kontext-dev": "FlowMatchEuler",
        "cosmos-predict2-2b-text2image": "FlowMatchEuler",
        "cosmos-predict2-14b-text2image": "FlowMatchEuler",
        "cosmos-predict2-2b-video2world": "FlowMatchEuler",
        "cosmos-predict2-14b-video2world": "FlowMatchEuler",
    }

    def __init__(
        self,
        dd_path,
        version="1.5",
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        bf16=False,
        max_batch_size=16,
        denoising_steps=30,
        scheduler=None,
        device="cuda",
        output_dir=".",
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
        framework_model_dir="pytorch_model",
        return_latents=False,
        low_vram=False,
        torch_inference="",
        torch_fallback=None,
        weight_streaming=False,
        text_encoder_weight_streaming_budget_percentage=None,
        denoiser_weight_streaming_budget_percentage=None,
        controlnet=None,
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            dd_path (load_module.DDPath): DDPath object that contains all paths used in DemoDiffusion.
            version (str):
                The version of the pipeline. Should be one of the values listed in DiffusionPipeline.VALID_DIFFUSION_PIPELINES.
            pipeline_type (PIPELINE_TYPE):
                Task performed by the current pipeline. Should be one of PIPELINE_TYPE.__members__.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            bf16 (`bool`, defaults to False):
                Whether to run the pipeline in BFloat16 precision.
            denoising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the values listed in DiffusionPipeline.SCHEDULER_DEFAULTS.values().
            device (str):
                PyTorch device to run inference. Default: 'cuda'.
            output_dir (str):
                Output directory for log files and image artifacts.
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference.
            framework_model_dir (str):
                cache directory for framework checkpoints.
            return_latents (bool):
                Skip decoding the image and return latents instead.
            low_vram (bool):
                [FLUX only] Optimize for low VRAM usage, possibly at the expense of inference performance. Disabled by default.
            torch_inference (str):
                Run inference with PyTorch (using specified compilation mode) instead of TensorRT. The compilation mode specified should be one of ['eager', 'reduce-overhead', 'max-autotune'].
            torch_fallback (str):
                [FLUX only] Comma separated list of models to be inferenced using PyTorch instead of TRT. For example --torch-fallback t5,transformer. If --torch-inference set, this parameter will be ignored.
            weight_streaming (`bool`, defaults to False):
                Whether to enable weight streaming during TensorRT engine build.
            text_encoder_ws_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the text encoder model.
            denoiser_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the denoiser model.
            controlnet (str, defaults to None):
                Type of ControlNet to use for the pipeline.
        """
        self.bf16 = bf16
        self.dd_path = dd_path

        self.denoising_steps = denoising_steps
        self.max_batch_size = max_batch_size

        self.framework_model_dir = framework_model_dir
        self.output_dir = output_dir
        for directory in [self.framework_model_dir, self.output_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        self.version = version
        self.pipeline_type = pipeline_type
        self.return_latents = return_latents

        self.low_vram = low_vram
        self.weight_streaming = weight_streaming
        self.text_encoder_weight_streaming_budget_percentage = text_encoder_weight_streaming_budget_percentage
        self.denoiser_weight_streaming_budget_percentage = denoiser_weight_streaming_budget_percentage

        self.stages = self.get_model_names(self.pipeline_type, controlnet)
        # config to store additional info
        self.config = {}
        if torch_fallback:
            assert type(torch_fallback) is list
            for model_name in torch_fallback:
                if model_name not in self.stages:
                    raise ValueError(f'Model "{model_name}" set in --torch-fallback does not exist')
                self.config[model_name.replace("-", "_") + "_torch_fallback"] = True
                print(f"[I] Setting torch_fallback for {model_name} model.")

        if not scheduler:
            scheduler = 'UniPC' if self.pipeline_type.is_controlnet() and not self.version == "3.5-large" else self.SCHEDULER_DEFAULTS.get(version, 'DDIM')
            print(f"[I] Autoselected scheduler: {scheduler}")

        scheduler_class_map = {
            "DDIM" : DDIMScheduler,
            "DDPM" : DDPMScheduler,
            "EulerA" : EulerAncestralDiscreteScheduler,
            "Euler" : EulerDiscreteScheduler,
            "LCM" : LCMScheduler,
            "LMSD" : LMSDiscreteScheduler,
            "PNDM" : PNDMScheduler,
            "UniPC" : UniPCMultistepScheduler,
            "DDPMWuerstchen" : DDPMWuerstchenScheduler,
            "FlowMatchEuler": FlowMatchEulerDiscreteScheduler
        }
        try:
            scheduler_class = scheduler_class_map[scheduler]
        except KeyError:
            raise ValueError(
                f"Unsupported scheduler {scheduler}.  Should be one of {list(scheduler_class_map.keys())}."
            )
        self.scheduler = make_scheduler(scheduler_class, version, pipeline_type, hf_token, framework_model_dir)

        self.torch_inference = torch_inference
        if self.torch_inference:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
        self.use_cuda_graph = use_cuda_graph

        # initialized in load_engines()
        self.models = {}
        self.torch_models = {}
        self.engine = {}
        self.shape_dicts = {}
        self.shared_device_memory = None
        self.lora_loader = None

        # initialized in load_resources()
        self.events = {}
        self.generator = None
        self.markers = {}
        self.seed = None
        self.stream = None
        self.tokenizer = None

    def model_memory_manager(self, model_names, low_vram=False):
        return ModelMemoryManager(self, model_names, low_vram)

    @classmethod
    @abc.abstractmethod
    def FromArgs(cls, args: argparse.Namespace, pipeline_type: PIPELINE_TYPE) -> DiffusionPipeline:
        """Factory method to construct a concrete pipeline object from parsed arguments."""
        raise NotImplementedError("FromArgs cannot be called from the abstract base class.")

    @classmethod
    @abc.abstractmethod
    def get_model_names(cls, pipeline_type: PIPELINE_TYPE, controlnet_type: str = None) -> List[str]:
        """Return a list of model names used by this pipeline."""
        raise NotImplementedError("get_model_names cannot be called from the abstract base class.")

    @classmethod
    def _get_pipeline_uid(cls, version: str) -> str:
        """Return the unique ID of this pipeline.

        This is typically used to determine the default path for things like engine files, artifacts caches, etc.
        """
        return f"{cls.__name__}_{version}"

    def profile_start(self, name, color="blue", domain=None):
        if self.nvtx_profile:
            self.markers[name] = nvtx.start_range(message=name, color=color, domain=domain)
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][0], 0)

    def profile_stop(self, name):
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][1], 0)
        if self.nvtx_profile:
            nvtx.end_range(self.markers[name])

    def load_resources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        if seed is not None:
            self.seed = seed
            self.generator = torch.Generator(device="cuda").manual_seed(seed)

        # Create CUDA events and stream
        for stage in self.stages:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        # Allocate TensorRT I/O buffers
        if not self.torch_inference:
            for model_name, obj in self.models.items():
                if self.torch_fallback[model_name]:
                    continue
                self.shape_dicts[model_name] = obj.get_shape_dict(batch_size, image_height, image_width)
                if not self.low_vram:
                    self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)

    @abstractmethod
    def _initialize_models(self, *args, **kwargs):
        raise NotImplementedError("Please Implement the _initialize_models method")

    def _prepare_model_configs(
        self,
        enable_refit,
        int8,
        fp8,
        fp4
    ):
        model_names = self.models.keys()
        self.torch_fallback = dict(zip(model_names, [self.torch_inference or self.config.get(model_name.replace('-','_')+'_torch_fallback', False) for model_name in model_names]))

        configs = {}
        for model_name in model_names:
            # Initialize config
            do_engine_refit = enable_refit and not self.pipeline_type.is_sd_xl_refiner() and any(model_name.startswith(prefix) for prefix in ("unet", "transformer"))
            do_lora_merge = not enable_refit and self.lora_loader and any(model_name.startswith(prefix) for prefix in ("unet", "transformer"))

            config = {
                "do_engine_refit": do_engine_refit,
                "do_lora_merge": do_lora_merge,
                "use_int8": False,
                "use_fp8": False,
                'use_fp4': False,
            }

            # TODO: Move this to when arguments are first being validated in dd_argparse.py
            # 8-bit/4-bit precision inference
            if int8:
                assert self.pipeline_type.is_sd_xl_base() or self.version in [
                    "1.5",
                    "2.1",
                    "2.1-base",
                ], "int8 quantization only supported for SDXL, SD1.5 and SD2.1 pipeline"
                if (self.pipeline_type.is_sd_xl() and model_name == "unetxl") or (model_name == "unet"):
                    config["use_int8"] = True

            elif fp8:
                assert (
                    self.pipeline_type.is_sd_xl()
                    or self.version in ["1.5", "2.1", "2.1-base"]
                    or self.version.startswith("flux.1")
                    or self.version.startswith("3.5-large")
                ), "fp8 quantization only supported for SDXL, SD1.5, SD2.1, SD3.5-large and FLUX pipelines"
                if (
                    (self.pipeline_type.is_sd_xl() and model_name == "unetxl")
                    or (self.version.startswith("flux.1") and model_name == "transformer")
                    or (
                        self.version.startswith("3.5-large")
                        and ("transformer" in model_name or "controlnet" in model_name)
                    )
                    or (model_name == "unet")
                ):
                    config["use_fp8"] = True
            elif fp4:
                config['use_fp4'] = True

            # Setup paths
            config["onnx_path"] = self.dd_path.model_name_to_unoptimized_onnx_path[model_name]
            config["onnx_opt_path"] = self.dd_path.model_name_to_optimized_onnx_path[model_name]
            config["engine_path"] = self.dd_path.model_name_to_engine_path[model_name]
            config["weights_map_path"] = (
                self.dd_path.model_name_to_weights_map_path[model_name] if config["do_engine_refit"] else None
            )
            config["state_dict_path"] = self.dd_path.model_name_to_quantized_model_state_dict_path[model_name]
            config["refit_weights_path"] = self.dd_path.model_name_to_refit_weights_path[model_name]

            configs[model_name] = config

        return configs

    def _calibrate_and_save_model(
            self,
            pipeline,
            model,
            model_config,
            quantization_level,
            quantization_percentile,
            quantization_alpha,
            calibration_size,
            calib_batch_size,
            enable_lora_merge = False,
            **kwargs):
        print(f"[I] Calibrated weights not found, generating {model_config['state_dict_path']}")

        # TODO check size > calibration_size
        def do_calibrate(pipeline, calibration_prompts, **kwargs):
            for i_th, prompts in enumerate(calibration_prompts):
                if i_th >= kwargs["calib_size"]:
                    return
                if kwargs["model_id"] in ("flux.1-dev", "flux.1-schnell"):
                    common_args = {
                        "prompt": prompts,
                        "prompt_2": prompts,
                        "num_inference_steps": kwargs["n_steps"],
                        "height": kwargs.get("height", 1024),
                        "width": kwargs.get("width", 1024),
                        "guidance_scale": 3.5,
                        "max_sequence_length": 512 if kwargs["model_id"] == "flux.1-dev" else 256,
                    }
                else:
                    common_args = {
                        "prompt": prompts,
                        "num_inference_steps": kwargs["n_steps"],
                        "negative_prompt": ["normal quality, low quality, worst quality, low res, blurry, nsfw, nude"]
                        * len(prompts),
                    }

                pipeline(**common_args).images

        def do_calibrate_img2img(pipeline, dataloader, **kwargs):
            for i_th, (img_conds, prompts) in enumerate(dataloader):
                if i_th >= kwargs["calib_size"]:
                    return

                common_args = {
                    "prompt": list(prompts),
                    "control_image": img_conds,
                    "num_inference_steps": kwargs["n_steps"],
                    "height": img_conds.size(2),
                    "width": img_conds.size(3),
                    "generator": torch.Generator().manual_seed(42),
                    "guidance_scale": 3.5,
                    "max_sequence_length": 512,
                }
                pipeline(**common_args).images

        if self.version in ("flux.1-dev-depth", "flux.1-dev-canny"):
            dataset = PromptImageDataset(
                root_dir=self.calibration_dataset,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=calib_batch_size,
                shuffle=False,
                num_workers=0,
                sampler=SameSizeSampler(dataset=dataset, batch_size=calib_batch_size),
                collate_fn=custom_collate,
            )
        else:
            root_dir = os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__))
            calibration_file = os.path.join(root_dir, "calibration_data", "calibration-prompts.txt")
            calibration_prompts = load_calib_prompts(calib_batch_size, calibration_file)

        def forward_loop(model):
            if self.version not in ("sd3", "flux.1-dev", "flux.1-schnell", "flux.1-dev-depth", "flux.1-dev-canny"):
                pipeline.unet = model
            else:
                pipeline.transformer = model

            if self.version in ("flux.1-dev-depth", "flux.1-dev-canny"):
                do_calibrate_img2img(
                    pipeline=pipeline,
                    dataloader=infinite_dataloader(dataloader),
                    calib_size=calibration_size // calib_batch_size,
                    n_steps=self.denoising_steps,
                    model_id=self.version,
                )
            else:
                do_calibrate(
                    pipeline=pipeline,
                    calibration_prompts=calibration_prompts,
                    calib_size=calibration_size // calib_batch_size,
                    n_steps=self.denoising_steps,
                    model_id=self.version,
                    **kwargs
                )

        print(f"[I] Performing calibration for {calibration_size} steps.")
        if model_config['use_int8']:
            quant_config = get_int8_config(
                model,
                quantization_level,
                quantization_alpha,
                quantization_percentile,
                self.denoising_steps
            )
        elif model_config['use_fp8']:
            if self.version.startswith("flux.1"):
                quant_config = SD_FP8_BF16_FLUX_MMDIT_BMM2_FP8_OUTPUT_CONFIG
            elif self.version == "2.1":
                quant_config = SD_FP8_FP32_DEFAULT_CONFIG
            else:
                quant_config = SD_FP8_FP16_DEFAULT_CONFIG

        # Handle LoRA
        if enable_lora_merge:
            assert self.lora_loader is not None
            model = merge_loras(model, self.lora_loader)

        check_lora(model)

        if self.version.startswith("flux.1"):
            set_quant_precision(quant_config, "BFloat16")
        mtq.quantize(model, quant_config, forward_loop)
        mto.save(model, model_config['state_dict_path'])

    def _get_quantized_model(
            self,
            obj,
            model_config,
            quantization_level,
            quantization_percentile,
            quantization_alpha,
            calibration_size,
            calib_batch_size,
            enable_lora_merge = False,
            **kwargs):
        pipeline = obj.get_pipeline()
        is_flux = self.version.startswith("flux.1")
        model = pipeline.unet if self.version not in ("sd3", "flux.1-dev", "flux.1-schnell", "flux.1-dev-depth", "flux.1-dev-canny") else pipeline.transformer
        if model_config['use_fp8'] and quantization_level == 4.0:
            set_fmha(model, is_flux=is_flux)

        if not os.path.exists(model_config['state_dict_path']):
            self._calibrate_and_save_model(
                pipeline,
                model,
                model_config,
                quantization_level,
                quantization_percentile,
                quantization_alpha,
                calibration_size,
                calib_batch_size,
                enable_lora_merge,
                **kwargs)
        else:
            mto.restore(model, model_config['state_dict_path'])

        if not os.path.exists(model_config['onnx_path']):
            quantize_lvl(self.version, model, quantization_level)
            if self.version.startswith("flux.1"):
                mtq.disable_quantizer(model, filter_func_no_proj_out)
            else:
                mtq.disable_quantizer(model, filter_func)
            if model_config['use_fp8'] and not self.version.startswith("flux.1"):
                generate_fp8_scales(model)
            if quantization_level == 4.0:
                fp8_mha_disable(model, quantized_mha_output=False) # Remove Q/DQ after BMM2 in MHA
        else:
            model = None

        return model

    @abstractmethod
    def download_onnx_models(self, model_name: str, model_config: dict[str, Any]) -> None:
        """Download pre-exported ONNX Models"""
        raise NotImplementedError("Please Implement the download_onnx_models method")

    def is_native_export_supported(self, model_config: dict[str, Any]) -> bool:
        """Check if pipeline supports native ONNX export"""
        # Native export is supported by default
        return True

    def _export_onnx(
        self,
        obj,
        model_name,
        model_config,
        opt_image_height,
        opt_image_width,
        static_shape,
        onnx_opset,
        quantization_level,
        quantization_percentile,
        quantization_alpha,
        calibration_size,
        calib_batch_size,
        onnx_export_only,
        download_onnx_models,
    ):
        # With onnx_export_only True, the export still happens even if the TRT engine exists. However, it will not re-run the export if the onnx exists.
        do_export_onnx = (not os.path.exists(model_config['engine_path']) or onnx_export_only) and not os.path.exists(model_config['onnx_opt_path'])
        do_export_weights_map = model_config['weights_map_path'] and not os.path.exists(model_config['weights_map_path'])

        # If ONNX export is required, either download ONNX models or check if the pipeline supports native ONNX export
        if do_export_onnx:
            if download_onnx_models:
                self.download_onnx_models(model_name, model_config)
                do_export_onnx = False
            else:
                self.is_native_export_supported(model_config)

        dynamo = True if self.pipeline_type.is_video2world() and model_name == "transformer" else False
        if do_export_onnx or do_export_weights_map:
            if not model_config['use_int8'] and not model_config['use_fp8']:
                obj.export_onnx(
                    model_config["onnx_path"],
                    model_config["onnx_opt_path"],
                    onnx_opset,
                    opt_image_height,
                    opt_image_width,
                    enable_lora_merge=model_config["do_lora_merge"],
                    static_shape=static_shape,
                    lora_loader=self.lora_loader,
                    dynamo=dynamo,
                )
            else:
                print(f"[I] Generating quantized ONNX model: {model_config['onnx_path']}")
                quantized_model = self._get_quantized_model(
                    obj,
                    model_config,
                    quantization_level,
                    quantization_percentile,
                    quantization_alpha,
                    calibration_size,
                    calib_batch_size,
                    height=opt_image_width,
                    width=opt_image_width,
                    enable_lora_merge=model_config["do_lora_merge"],
                )
                obj.export_onnx(
                    model_config["onnx_path"],
                    model_config["onnx_opt_path"],
                    onnx_opset,
                    opt_image_height,
                    opt_image_width,
                    custom_model=quantized_model,
                    static_shape=static_shape,
                    dynamo=dynamo,
                )

        # FIXME do_export_weights_map needs ONNX graph
        if do_export_weights_map:
            print(f"[I] Saving weights map: {model_config['weights_map_path']}")
            obj.export_weights_map(model_config['onnx_opt_path'], model_config['weights_map_path'])

    def _build_engine(self, obj, engine, model_config, opt_batch_size, opt_image_height, opt_image_width, optimization_level, static_batch, static_shape, enable_all_tactics, timing_cache):
        update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
        fp16amp = False if (model_config['use_fp8'] or getattr(obj, 'build_strongly_typed', False)) else obj.fp16
        tf32amp = obj.tf32
        bf16amp = False if (model_config['use_fp8'] or getattr(obj, 'build_strongly_typed', False)) else obj.bf16
        strongly_typed = True if (model_config['use_fp8'] or getattr(obj, 'build_strongly_typed', False)) else False
        weight_streaming = getattr(obj, 'weight_streaming', False)
        int8amp = model_config.get('use_int8', False)
        precision_constraints = 'prefer' if int8amp else 'none'
        engine.build(
            model_config["onnx_opt_path"],
            strongly_typed=strongly_typed,
            fp16=fp16amp,
            tf32=tf32amp,
            bf16=bf16amp,
            int8=int8amp,
            input_profile=obj.get_input_profile(
                opt_batch_size, opt_image_height, opt_image_width, static_batch=static_batch, static_shape=static_shape
            ),
            enable_refit=model_config["do_engine_refit"],
            enable_all_tactics=enable_all_tactics,
            timing_cache=timing_cache,
            update_output_names=update_output_names,
            weight_streaming=weight_streaming,
            verbose=self.verbose,
            builder_optimization_level=optimization_level,
            precision_constraints=precision_constraints,
        )

    def _refit_engine(self, obj, model_name, model_config):
        assert model_config['weights_map_path']
        with open(model_config['weights_map_path'], 'r') as fp_wts:
            print(f"[I] Loading weights map: {model_config['weights_map_path']} ")
            [weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)

            if not os.path.exists(model_config['refit_weights_path']):
                model = merge_loras(obj.get_model(), self.lora_loader)
                refit_weights, updated_weight_names = engine_module.get_refit_weights(
                    model.state_dict(), model_config["onnx_opt_path"], weights_name_mapping, weights_shape_mapping
                )
                print(f"[I] Saving refit weights: {model_config['refit_weights_path']}")
                torch.save((refit_weights, updated_weight_names), model_config["refit_weights_path"])
                unload_torch_model(model)
            else:
                print(f"[I] Loading refit weights: {model_config['refit_weights_path']}")
                refit_weights, updated_weight_names = torch.load(model_config['refit_weights_path'])
            self.engine[model_name].refit(refit_weights, updated_weight_names)

    def _load_torch_models(self):
        # Load torch models
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                self.torch_models[model_name] = obj.get_model(torch_inference=self.torch_inference)
                if self.low_vram:
                    self.torch_models[model_name] = self.torch_models[model_name].to('cpu')
                    torch.cuda.empty_cache()

    def load_engines(
        self,
        framework_model_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        optimization_level=3,
        static_batch=False,
        static_shape=True,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        int8=False,
        fp8=False,
        fp4=False,
        quantization_level=2.5,
        quantization_percentile=1.0,
        quantization_alpha=0.8,
        calibration_size=32,
        calib_batch_size=2,
        onnx_export_only=False,
        download_onnx_models=False,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            framework_model_dir (str):
                Directory to store the framework model ckpt.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            optimization_level (int):
                Optimization level to build the TensorRT engine with.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_refit (bool):
                Build engines with refit option enabled.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to speed up TensorRT build.
            int8 (bool):
                Whether to quantize to int8 format or not (SDXL, SD15 and SD21 only).
            fp8 (bool):
                Whether to quantize to fp8 format or not (SDXL, SD15 and SD21 only).
            quantization_level (float):
                Controls which layers to quantize. 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC
            quantization_percentile (float):
                Control quantization scaling factors (amax) collecting range, where the minimum amax in
                range(n_steps * percentile) will be collected. Recommendation: 1.0
            quantization_alpha (float):
                The alpha parameter for SmoothQuant quantization used for linear layers.
                Recommendation: 0.8 for SDXL
            calibration_size (int):
                The number of steps to use for calibrating the model for quantization.
                Recommendation: 32, 64, 128 for SDXL
            calib_batch_size (int):
                The batch size to use for calibration. Defaults to 2.
            onnx_export_only (bool):
                Whether only export onnx without building the TRT engine.
            download_onnx_models (bool):
                Download pre-exported ONNX models
        """
        self._initialize_models(framework_model_dir, int8, fp8, fp4)

        model_configs = self._prepare_model_configs(enable_refit, int8, fp8, fp4)

        # Export models to ONNX
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            self._export_onnx(
                obj,
                model_name,
                model_configs[model_name],
                opt_image_height,
                opt_image_width,
                static_shape,
                onnx_opset,
                quantization_level,
                quantization_percentile,
                quantization_alpha,
                calibration_size,
                calib_batch_size,
                onnx_export_only,
                download_onnx_models,
            )

            # Release temp GPU memory during onnx export to avoid OOM.
            gc.collect()
            torch.cuda.empty_cache()

        if onnx_export_only:
            return

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue

            model_config = model_configs[model_name]
            engine = engine_module.Engine(model_config["engine_path"])
            if not os.path.exists(model_config['engine_path']):
                self._build_engine(obj, engine, model_config, opt_batch_size, opt_image_height, opt_image_width, optimization_level, static_batch, static_shape, enable_all_tactics, timing_cache)
            self.engine[model_name] = engine

        # Load and refit TensorRT engines
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            model_config = model_configs[model_name]

            # For non low_vram case, the engines will remain in GPU memory from now on.
            assert self.engine[model_name].engine is None
            if not self.low_vram:
                weight_streaming = getattr(obj, 'weight_streaming', False)
                weight_streaming_budget_percentage = getattr(obj, 'weight_streaming_budget_percentage', None)
                self.engine[model_name].load(weight_streaming, weight_streaming_budget_percentage)

            if model_config['do_engine_refit'] and self.lora_loader:
                # For low_vram, using on-demand load and unload for refit.
                if self.low_vram:
                    assert self.engine[model_name].engine is None
                    self.engine[model_name].load()
                self._refit_engine(obj, model_name, model_config)
                if self.low_vram:
                    self.engine[model_name].unload()

        # Load PyTorch models if torch-inference mode is enabled
        self._load_torch_models()

        # Reclaim GPU memory from torch cache
        torch.cuda.empty_cache()

    def calculate_max_device_memory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            if self.low_vram:
                engine.load()
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
            if self.low_vram:
                engine.unload()
        return max_device_memory

    def get_device_memory_sizes(self):
        device_memory_sizes = {}
        for model_name, engine in self.engine.items():
            engine.load()
            device_memory_sizes[model_name] = engine.engine.device_memory_size
            engine.unload()
        return device_memory_sizes

    def activate_engines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculate_max_device_memory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        if not self.low_vram:
            for engine in self.engine.values():
                engine.activate(device_memory=self.shared_device_memory)

    def run_engine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        # CUDA graphs should be disabled when low_vram is enabled.
        if self.low_vram:
            assert self.use_cuda_graph == False
        return engine.infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e[0])
            cudart.cudaEventDestroy(e[1])

        for engine in self.engine.values():
            engine.deallocate_buffers()
            engine.deactivate()
            engine.unload(verbose=False)
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        for torch_model in self.torch_models.values():
            torch_model.to("cpu")
            del torch_model

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

        gc.collect()
        torch.cuda.empty_cache()

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width, latents_dtype=torch.float32):
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def save_image(self, images, pipeline, prompt, seed):
        # Save image
        prompt_prefix = ''.join(set([prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))
        image_name_prefix = '-'.join([pipeline, prompt_prefix, str(seed)])
        image_name_suffix = 'torch' if self.torch_inference else 'trt'
        image_module.save_image(images, self.output_dir, image_name_prefix, image_name_suffix)

    @abstractmethod
    def print_summary(self):
        """Print a summary of the pipeline's configuration."""
        raise NotImplementedError("Please Implement the print_summary method")

    @abstractmethod
    def infer(self):
        """Perform inference using the pipeline."""
        raise NotImplementedError("Please Implement the infer method")

    @abstractmethod
    def run(self):
        """Run the pipeline."""
        raise NotImplementedError("Please Implement the run method")
