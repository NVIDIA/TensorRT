
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from cuda import cudart
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler, LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    DDPMWuerstchenScheduler,
    FlowMatchEulerDiscreteScheduler
)
from hashlib import md5
from models import make_scheduler, LoraLoader
import nvtx
import json
import os
import pathlib
import tensorrt as trt
import torch
from utilities import (
    PIPELINE_TYPE,
    Engine,
    get_refit_weights,
    load_calib_prompts,
    merge_loras,
    unload_model,
    save_image
)
from typing import Optional, List
from utils_modelopt import (
    filter_func,
    filter_func_no_proj_out,
    quantize_lvl,
    get_int8_config,
    check_lora,
    set_fmha,
    set_quant_precision,
    generate_fp8_scales,
    SD_FP8_BF16_DEFAULT_CONFIG,
    SD_FP8_FP16_DEFAULT_CONFIG,
    SD_FP8_FP32_DEFAULT_CONFIG,
)
import gc

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
        "cascade",
        "flux.1-dev",
        "flux.1-schnell"
    )
    SCHEDULER_DEFAULTS = {
        "1.4": "PNDM",
        "1.5": "PNDM",
        "dreamshaper-7": "PNDM",
        "2.0-base": "DDIM",
        "2.0": "DDIM",
        "2.1-base": "PNDM",
        "2.1": "DDIM",
        "xl-1.0" : "Euler",
        "xl-turbo": "EulerA",
        "svd-xt-1.1": "Euler",
        "cascade": "DDPMWuerstchen",
        "flux.1-dev": "FlowMatchEuler",
        "flux.1-schnell": "FlowMatchEuler"
    }

    def __init__(
        self,
        version='1.5',
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        max_batch_size=16,
        denoising_steps=30,
        scheduler=None,
        lora_scale: float = 1.0,
        lora_weight: Optional[List[float]] = None,
        lora_path: Optional[List[str]] = None,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
        framework_model_dir='pytorch_model',
        return_latents=False,
        torch_inference='',
        weight_streaming=False,
        text_encoder_weight_streaming_budget_percentage=None,
        denoiser_weight_streaming_budget_percentage=None,
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of the values listed in DiffusionPipeline.VALID_DIFFUSION_PIPELINES.
            pipeline_type (PIPELINE_TYPE):
                Task performed by the current pipeline. Should be one of PIPELINE_TYPE.__members__.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            denoising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the values listed in DiffusionPipeline.SCHEDULER_DEFAULTS.values().
            lora_scale (float):
                Controls how much to influence the outputs with the LoRA parameters. (must between 0 and 1).
            lora_weight (float):
                The LoRA adapter(s) weights to use with the UNet. (must between 0 and 1).
            lora_path (str):
                Path to LoRA adaptor. Ex: 'latent-consistency/lcm-lora-sdv1-5'.
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
            torch_inference (str):
                Run inference with PyTorch (using specified compilation mode) instead of TensorRT.
            weight_streaming (`bool`, defaults to False):
                Whether to enable weight streaming during TensorRT engine build.
            text_encoder_ws_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the text encoder model.
            denoiser_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the denoiser model.
        """
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

        self.weight_streaming = weight_streaming
        self.text_encoder_weight_streaming_budget_percentage = text_encoder_weight_streaming_budget_percentage
        self.denoiser_weight_streaming_budget_percentage = denoiser_weight_streaming_budget_percentage

        if not scheduler:
            scheduler = 'UniPC' if self.pipeline_type.is_controlnet() else self.SCHEDULER_DEFAULTS.get(version, 'DDIM')
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
            raise ValueError(f"Unsupported scheduler {scheduler}.  Should be one of {list(scheduler_class.keys())}.")
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

        # initialize lora loader and scales
        self.lora_loader = None
        self.lora_weights = dict()
        if lora_path:
            self.lora_loader = LoraLoader(lora_path, lora_weight, lora_scale)
            assert len(lora_path) == len(lora_weight)
            for i, path in enumerate(lora_path):
                self.lora_weights[path] = lora_weight[i]

        # initialized in load_resources()
        self.events = {}
        self.generator = None
        self.markers = {}
        self.seed = None
        self.stream = None
        self.tokenizer = None

        # config to store additional info
        self.config = {}

    def profile_start(self, name, color='blue'):
        if self.nvtx_profile:
            self.markers[name] = nvtx.start_range(message=name, color=color)
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

    def _create_directories(self, engine_dir, onnx_dir):
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

    def _cached_model_name(self, model_name):
        if self.pipeline_type.is_inpaint():
            model_name += '_inpaint'
        return model_name

    def _get_onnx_path(self, model_name, onnx_dir, opt=True, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self._cached_model_name(model_name)+suffix+('.opt' if opt else ''))
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'model.onnx')

    def _get_engine_path(self, model_name, engine_dir, enable_refit=False, suffix=''):
        return os.path.join(engine_dir, self._cached_model_name(model_name)+suffix+('.refit' if enable_refit else '')+'.trt'+trt.__version__+'.plan')

    def _get_weights_map_path(self, model_name, onnx_dir):
        onnx_model_dir = os.path.join(onnx_dir, self._cached_model_name(model_name)+'.opt')
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'weights_map.json')

    def _get_refit_nodes_path(self, model_name, onnx_dir, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self._cached_model_name(model_name)+'.opt')
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'refit'+suffix+'.json')

    def _get_state_dict_path(self, model_name, onnx_dir, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self._cached_model_name(model_name)+suffix)
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'state_dict.pt')

    @abstractmethod
    def _initialize_models(self):
        raise NotImplementedError("Please Implement the _initialize_models method")

    def _get_lora_suffix(self):
        if self.lora_loader:
            return '-' + '-'.join([str(md5(path.encode('utf-8')).hexdigest()) + '-' + ('%.2f' % self.lora_weights[path]) + '-' + ('%.2f' % self.lora_loader.scale) for path in sorted(self.lora_loader.paths)])
        return ''

    def _prepare_model_configs(self, onnx_dir, engine_dir, enable_refit, int8, fp8, quantization_level, quantization_percentile, quantization_alpha, calibration_size):
        model_names = self.models.keys()
        lora_suffix = self._get_lora_suffix()
        self.torch_fallback = dict(zip(model_names, [self.torch_inference or self.config.get(model_name.replace('-','_')+'_torch_fallback', False) for model_name in model_names]))

        configs = {}
        for model_name in model_names:
            config = {
                'do_engine_refit': not self.pipeline_type.is_sd_xl_refiner() and enable_refit and model_name.startswith('unet'),
                'do_lora_merge': not enable_refit and self.lora_loader and model_name.startswith('unet'),
                'use_int8': False,
                'use_fp8': False,
            }
            config['model_suffix'] = lora_suffix if config['do_lora_merge'] else ''

            if int8:
                assert self.pipeline_type.is_sd_xl_base() or self.version in ["1.5", "2.1", "2.1-base"], "int8 quantization only supported for SDXL, SD1.5 and SD2.1 pipeline"
                if (self.pipeline_type.is_sd_xl() and model_name == 'unetxl') or \
                    (model_name == 'unet'):
                    config['use_int8'] = True
                    config['model_suffix'] += f"-int8.l{quantization_level}.bs2.s{self.denoising_steps}.c{calibration_size}.p{quantization_percentile}.a{quantization_alpha}"
            elif fp8:
                assert self.pipeline_type.is_sd_xl() or self.version in ["1.5", "2.1", "2.1-base", "flux.1-dev", "flux.1-schnell"], "fp8 quantization only supported for SDXL, SD1.5, SD2.1 and FLUX pipeline"
                if (self.pipeline_type.is_sd_xl() and model_name == 'unetxl') or \
                    ((self.version in ("flux.1-dev", "flux.1-schnell")) and model_name == 'transformer') or \
                    (model_name == 'unet'):
                    config['use_fp8'] = True
                    config['model_suffix'] += f"-fp8.l{quantization_level}.bs2.s{self.denoising_steps}.c{calibration_size}.p{quantization_percentile}.a{quantization_alpha}"

            config['onnx_path'] = self._get_onnx_path(model_name, onnx_dir, opt=False, suffix=config['model_suffix'])
            config['onnx_opt_path'] = self._get_onnx_path(model_name, onnx_dir, suffix=config['model_suffix'])
            config['engine_path'] = self._get_engine_path(model_name, engine_dir, config['do_engine_refit'], suffix=config['model_suffix'])
            config['weights_map_path'] = self._get_weights_map_path(model_name, onnx_dir) if config['do_engine_refit'] else None
            config['state_dict_path'] = self._get_state_dict_path(model_name, onnx_dir, suffix=config['model_suffix'])
            config['refit_weights_path'] = self._get_refit_nodes_path(model_name, onnx_dir, suffix=lora_suffix)

            configs[model_name] = config

        return configs

    def _calibrate_and_save_model(self, pipeline, model, model_config, quantization_level, quantization_percentile, quantization_alpha, calibration_size, calib_batch_size, **kwargs):
        print(f"[I] Calibrated weights not found, generating {model_config['state_dict_path']}")
        calibration_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'calibration-prompts.txt')
        calibration_prompts = load_calib_prompts(calib_batch_size, calibration_file)

        # TODO check size > calibration_size
        def do_calibrate(pipeline, calibration_prompts, **kwargs):
            for i_th, prompts in enumerate(calibration_prompts):
                if i_th >= kwargs["calib_size"]:
                    return
                if kwargs["model_id"] in ("flux.1-dev", "flux.1-schnell"):
                    max_seq_len = 512 if kwargs["model_id"] == "flux.1-dev" else 256
                    height = kwargs.get("height", 1024)
                    width = kwargs.get("width", 1024)
                    pipeline(
                        prompt=prompts,
                        prompt_2=prompts,
                        num_inference_steps=kwargs["n_steps"],
                        height=height,
                        width=width,
                        guidance_scale=3.5,
                        max_sequence_length=max_seq_len
                    ).images
                else:
                    pipeline(
                        prompt=prompts,
                        num_inference_steps=kwargs["n_steps"],
                        negative_prompt=[
                            "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
                        ]
                        * len(prompts),
                    ).images

        def forward_loop(model):
            if self.version not in ("sd3", "flux.1-dev", "flux.1-schnell"):
                pipeline.unet = model
            else:
                pipeline.transformer = model

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
            if self.version in ("flux.1-dev", "flux.1-schnell"):
                quant_config = SD_FP8_BF16_DEFAULT_CONFIG
            elif self.version == "2.1":
                quant_config = SD_FP8_FP32_DEFAULT_CONFIG
            else:
                quant_config = SD_FP8_FP16_DEFAULT_CONFIG

        check_lora(model)
        if self.version in ("flux.1-dev", "flux.1-schnell"):
            set_quant_precision(quant_config, "BFloat16")
        mtq.quantize(model, quant_config, forward_loop)
        mto.save(model, model_config['state_dict_path'])

    def _get_quantized_model(self, obj, model_config, quantization_level, quantization_percentile, quantization_alpha, calibration_size, calib_batch_size, **kwargs):
        pipeline = obj.get_pipeline()
        model = pipeline.unet if self.version not in ("sd3", "flux.1-dev", "flux.1-schnell") else pipeline.transformer
        if model_config['use_fp8'] and quantization_level == 4.0:
            set_fmha(model)

        if not os.path.exists(model_config['state_dict_path']):
            self._calibrate_and_save_model(pipeline, model, model_config, quantization_level, quantization_percentile, quantization_alpha, calibration_size, calib_batch_size, **kwargs)
        else:
            mto.restore(model, model_config['state_dict_path'])

        if not os.path.exists(model_config['onnx_path']):
            quantize_lvl(model, quantization_level)
            if self.version in ("flux.1-dev", "flux.1-schnell"):
                mtq.disable_quantizer(model, filter_func_no_proj_out)
            else:
                mtq.disable_quantizer(model, filter_func)
            if model_config['use_fp8']:
                generate_fp8_scales(model)
        else:
            model = None

        return model

    def _export_onnx(self, obj, model_config, opt_image_height, opt_image_width, static_shape, onnx_opset, quantization_level, quantization_percentile, quantization_alpha, calibration_size, calib_batch_size):
        do_export_onnx = not os.path.exists(model_config['engine_path']) and not os.path.exists(model_config['onnx_opt_path'])
        do_export_weights_map = model_config['weights_map_path'] and not os.path.exists(model_config['weights_map_path'])

        if do_export_onnx or do_export_weights_map:
            if not model_config['use_int8'] and not model_config['use_fp8']:
                obj.export_onnx(model_config['onnx_path'], model_config['onnx_opt_path'], onnx_opset, opt_image_height, opt_image_width, enable_lora_merge=model_config['do_lora_merge'], static_shape=static_shape, lora_loader=self.lora_loader)
            else:
                print(f"[I] Generating quantized ONNX model: {model_config['onnx_path']}")
                quantized_model = self._get_quantized_model(obj, model_config, quantization_level, quantization_percentile, quantization_alpha, calibration_size, calib_batch_size, height=opt_image_width, width=opt_image_width)
                obj.export_onnx(model_config['onnx_path'], model_config['onnx_opt_path'], onnx_opset, opt_image_height, opt_image_width, custom_model=quantized_model, static_shape=static_shape)

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
        extra_build_args = {'verbose': self.verbose}
        extra_build_args['builder_optimization_level'] = optimization_level
        if model_config['use_int8']:
            extra_build_args['int8'] = True
            extra_build_args['precision_constraints'] = 'prefer'
        engine.build(model_config['onnx_opt_path'],
            strongly_typed=strongly_typed,
            fp16=fp16amp,
            tf32=tf32amp,
            bf16=bf16amp,
            input_profile=obj.get_input_profile(
                opt_batch_size, opt_image_height, opt_image_width,
                static_batch=static_batch, static_shape=static_shape
            ),
            enable_refit=model_config['do_engine_refit'],
            enable_all_tactics=enable_all_tactics,
            timing_cache=timing_cache,
            update_output_names=update_output_names,
            weight_streaming=weight_streaming,
            **extra_build_args)

    def _refit_engine(self, obj, model_name, model_config):
        assert model_config['weights_map_path']
        with open(model_config['weights_map_path'], 'r') as fp_wts:
            print(f"[I] Loading weights map: {model_config['weights_map_path']} ")
            [weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)

            if not os.path.exists(model_config['refit_weights_path']):
                print(f"[I] Saving refit weights: {model_config['refit_weights_path']}")
                model = merge_loras(obj.get_model(), self.lora_loader)
                refit_weights, updated_weight_names = get_refit_weights(model.state_dict(), model_config['onnx_opt_path'], weights_name_mapping, weights_shape_mapping)
                torch.save((refit_weights, updated_weight_names), model_config['refit_weights_path'])
                unload_model(model)
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
        engine_dir,
        framework_model_dir,
        onnx_dir,
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
        quantization_level=2.5,
        quantization_percentile=1.0,
        quantization_alpha=0.8,
        calibration_size=32,
        calib_batch_size=2,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to store the TensorRT engines.
            framework_model_dir (str):
                Directory to store the framework model ckpt.
            onnx_dir (str):
                Directory to store the ONNX models.
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
        """
        self._create_directories(engine_dir, onnx_dir)
        self._initialize_models(framework_model_dir, int8, fp8)

        model_configs = self._prepare_model_configs(onnx_dir, engine_dir, enable_refit, int8, fp8, quantization_level, quantization_percentile, quantization_alpha, calibration_size)

        # Export models to ONNX
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            self._export_onnx(obj, model_configs[model_name], opt_image_height, opt_image_width, static_shape, onnx_opset, quantization_level, quantization_percentile, quantization_alpha, calibration_size, calib_batch_size)

            # Release temp GPU memory during onnx export to avoid OOM.
            gc.collect()
            torch.cuda.empty_cache()

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue

            model_config = model_configs[model_name]
            engine = Engine(model_config['engine_path'])
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
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        for torch_model in self.torch_models.values():
            del torch_model

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width, latents_dtype=torch.float32):
        latents_dtype = latents_dtype # text_embeddings.dtype
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
        save_image(images, self.output_dir, image_name_prefix, image_name_suffix)

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
