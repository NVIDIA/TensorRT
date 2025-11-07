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
import gc
import inspect
import json
import os
import pathlib
import sys
import time
from hashlib import md5
from typing import List, Optional

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import numpy as np
import nvtx
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DDPMWuerstchenScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

import demo_diffusion.engine as engine_module
import demo_diffusion.image as image_module
from demo_diffusion.model import (
    CLIPModel,
    CLIPWithProjModel,
    SDLoraLoader,
    UNet2DConditionControlNetModel,
    UNetModel,
    UNetXLModel,
    UNetXLModelControlNet,
    VAEEncoderModel,
    VAEModel,
    get_clip_embedding_dim,
    make_scheduler,
    make_tokenizer,
    merge_loras,
    unload_torch_model,
)
from demo_diffusion.pipeline.calibrate import load_calib_prompts
from demo_diffusion.pipeline.type import PIPELINE_TYPE
from demo_diffusion.utils_modelopt import (
    SD_FP8_FP16_DEFAULT_CONFIG,
    SD_FP8_FP32_DEFAULT_CONFIG,
    check_lora,
    filter_func,
    generate_fp8_scales,
    get_int8_config,
    quantize_lvl,
    set_fmha,
)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class StableDiffusionPipeline:
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
        "cascade": "DDPMWuerstchen"
    }
    """
    Application showcasing the acceleration of Stable Diffusion pipelines using NVidia TensorRT.
    """
    def __init__(
        self,
        version='1.5',
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        max_batch_size=16,
        denoising_steps=30,
        scheduler=None,
        guidance_scale=7.5,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
        vae_scaling_factor=0.18215,
        framework_model_dir='pytorch_model',
        controlnets=None,
        lora_scale: float = 1.0,
        lora_weight: Optional[List[float]] = None,
        lora_path: Optional[List[str]] = None,
        return_latents=False,
        torch_inference='',
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of [1.4, 1.5, 2.0, 2.0-base, 2.1, 2.1-base]
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            denoising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of [DDIM, DPM, EulerA, Euler, LCM, LMSD, PNDM].
            guidance_scale (float):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
            vae_scaling_factor (float):
                VAE scaling factor
            framework_model_dir (str):
                cache directory for framework checkpoints
            controlnets (str):
                Which ControlNet/ControlNets to use.
            return_latents (bool):
                Skip decoding the image and return latents instead.
            torch_inference (str):
                Run inference with PyTorch (using specified compilation mode) instead of TensorRT.
        """

        self.denoising_steps = denoising_steps
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = (guidance_scale > 1.0)
        self.vae_scaling_factor = vae_scaling_factor

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
        self.controlnets = controlnets

        # Pipeline type
        self.pipeline_type = pipeline_type
        if self.pipeline_type.is_txt2img():
            self.stages = ['clip','unet','vae']
        elif self.pipeline_type.is_img2img() or self.pipeline_type.is_inpaint():
            self.stages = ['vae_encoder', 'clip','unet','vae']
        elif self.pipeline_type.is_sd_xl_base():
            self.stages = ['clip', 'clip2', 'unetxl']
            if not return_latents:
                self.stages.append('vae')
        elif self.pipeline_type.is_sd_xl_refiner():
            self.stages = ['clip2', 'unetxl', 'vae']
        elif self.pipeline_type.is_img2vid():
            self.stages = ['clip-vis', 'clip-imgfe', 'unet-temp', 'vae-temp']
        elif self.pipeline_type.is_cascade_prior():
            self.stages = ['clip', 'unet']
        elif self.pipeline_type.is_cascade_decoder():
            self.stages = ['clip', 'unet', 'vqgan']
        else:
            raise ValueError(f"Unsupported pipeline {self.pipeline_type.name}.")
        self.return_latents = return_latents

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
        }
        try:
            scheduler_class = scheduler_class_map[scheduler]
        except KeyError:
            raise ValueError(f"Unsupported scheduler {scheduler}.  Should be one of {list(scheduler_class.keys())}.")
        self.scheduler = make_scheduler(scheduler_class, version, pipeline_type, hf_token, framework_model_dir)

        self.config = {}
        if self.pipeline_type.is_sd_xl():
            self.config['clip_hidden_states'] = True
        self.torch_inference = torch_inference
        if self.torch_inference:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
        self.use_cuda_graph = use_cuda_graph

        # initialized in loadEngines()
        self.models = {}
        self.torch_models = {}
        self.engine = {}
        self.shared_device_memory = None

        # initialize lora loader and scales
        self.lora_loader = None
        self.lora_weights = dict()
        if lora_path:
            self.lora_loader = SDLoraLoader(lora_path, lora_weight, lora_scale)
            assert len(lora_path) == len(lora_weight)
            for i, path in enumerate(lora_path):
                self.lora_weights[path] = lora_weight[i]

        # initialized in loadResources()
        self.events = {}
        self.generator = None
        self.markers = {}
        self.seed = None
        self.stream = None
        self.tokenizer = None

    def loadResources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        if seed:
            self.seed = seed
            self.generator = torch.Generator(device="cuda").manual_seed(seed)

        # Create CUDA events and stream
        for stage in self.stages:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        # Allocate TensorRT I/O buffers
        if not self.torch_inference:
            for model_name, obj in self.models.items():
                self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)

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

    def cachedModelName(self, model_name):
        if self.pipeline_type.is_inpaint():
            model_name += '_inpaint'
        return model_name

    def getOnnxPath(self, model_name, onnx_dir, opt=True, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+suffix+('.opt' if opt else ''))
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'model.onnx')

    def getEnginePath(self, model_name, engine_dir, enable_refit=False, suffix=''):
        return os.path.join(engine_dir, self.cachedModelName(model_name)+suffix+('.refit' if enable_refit else '')+'.trt'+trt.__version__+'.plan')

    def getWeightsMapPath(self, model_name, onnx_dir):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+'.opt')
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'weights_map.json')

    def getRefitNodesPath(self, model_name, onnx_dir, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+'.opt')
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'refit'+suffix+'.json')

    def getStateDictPath(self, model_name, onnx_dir, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+suffix)
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'state_dict.pt')

    def initializeModels(self, framework_model_dir, int8, fp8):
        # Load text tokenizer(s)
        if not self.pipeline_type.is_sd_xl_refiner():
            self.tokenizer = make_tokenizer(self.version, self.pipeline_type, self.hf_token, framework_model_dir)
        if self.pipeline_type.is_sd_xl():
            self.tokenizer2 = make_tokenizer(self.version, self.pipeline_type, self.hf_token, framework_model_dir, subfolder='tokenizer_2')

        # Load pipeline models
        models_args = {'version': self.version, 'pipeline': self.pipeline_type, 'device': self.device,
            'hf_token': self.hf_token, 'verbose': self.verbose, 'framework_model_dir': framework_model_dir,
            'max_batch_size': self.max_batch_size}

        if 'clip' in self.stages:
            subfolder = 'text_encoder'
            self.models['clip'] = CLIPModel(**models_args, fp16=True, embedding_dim=get_clip_embedding_dim(self.version, self.pipeline_type), output_hidden_states=self.config.get('clip_hidden_states', False), subfolder=subfolder)

        if 'clip2' in self.stages:
            subfolder = 'text_encoder_2'
            self.models['clip2'] = CLIPWithProjModel(**models_args, fp16=True, output_hidden_states=self.config.get('clip_hidden_states', False), subfolder=subfolder)

        if 'unet' in self.stages:
            self.models['unet'] = UNetModel(**models_args, fp16=True, int8=int8, fp8=fp8, controlnets=self.controlnets, do_classifier_free_guidance=self.do_classifier_free_guidance)

        if 'unetxl' in self.stages:
            if not self.controlnets:
                self.models["unetxl"] = UNetXLModel(
                    **models_args,
                    fp16=True,
                    int8=int8,
                    fp8=fp8,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )
            else:
                self.models["unetxl"] = UNetXLModelControlNet(
                    **models_args,
                    fp16=True,
                    int8=int8,
                    fp8=fp8,
                    controlnets=self.controlnets,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )

        vae_fp16 = not self.pipeline_type.is_sd_xl()

        if 'vae' in self.stages:
            self.models['vae'] = VAEModel(**models_args, fp16=vae_fp16, tf32=True)

        if 'vae_encoder' in self.stages:
            self.models['vae_encoder'] = VAEEncoderModel(**models_args, fp16=vae_fp16)

    def loadEngines(
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
        **_kwargs,
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
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        # Initialize models
        self.initializeModels(framework_model_dir, int8, fp8)

        # Configure pipeline models to load
        model_names = self.models.keys()
        lora_suffix = '-'+'-'.join([str(md5(path.encode('utf-8')).hexdigest())+'-'+('%.2f' % self.lora_weights[path])+'-'+('%.2f' % self.lora_loader.scale) for path in sorted(self.lora_loader.paths)]) if self.lora_loader else ''
        # Enable refit and LoRA merging only for UNet & UNetXL for now
        do_engine_refit = dict(zip(model_names, [not self.pipeline_type.is_sd_xl_refiner() and enable_refit and model_name.startswith('unet') for model_name in model_names]))
        do_lora_merge = dict(zip(model_names, [not enable_refit and self.lora_loader and model_name.startswith('unet') for model_name in model_names]))
        # Torch fallback for VAE if specified
        torch_fallback = dict(zip(model_names, [self.torch_inference for model_name in model_names]))
        model_suffix = dict(zip(model_names, [lora_suffix if do_lora_merge[model_name] else '' for model_name in model_names]))
        use_int8 = dict.fromkeys(model_names, False)
        use_fp8 = dict.fromkeys(model_names, False)
        if int8:
            assert self.pipeline_type.is_sd_xl_base() or self.version in ["1.4", "1.5", "2.1", "2.1-base"], "int8 quantization only supported for SDXL, SD1.4, SD1.5 and SD2.1 pipeline"
            model_name = 'unetxl' if self.pipeline_type.is_sd_xl() else 'unet'
            use_int8[model_name] = True
            model_suffix[model_name] += f"-int8.l{quantization_level}.bs2.s{self.denoising_steps}.c{calibration_size}.p{quantization_percentile}.a{quantization_alpha}"
        elif fp8:
            assert self.pipeline_type.is_sd_xl() or self.version in ["1.4", "1.5", "2.1", "2.1-base"], "fp8 quantization only supported for SDXL, SD1.4, SD1.5 and SD2.1 pipeline"
            model_name = 'unetxl' if self.pipeline_type.is_sd_xl() else 'unet'
            use_fp8[model_name] = True
            model_suffix[model_name] += f"-fp8.l{quantization_level}.bs2.s{self.denoising_steps}.c{calibration_size}.p{quantization_percentile}.a{quantization_alpha}"
        onnx_path = dict(zip(model_names, [self.getOnnxPath(model_name, onnx_dir, opt=False, suffix=model_suffix[model_name]) for model_name in model_names]))
        onnx_opt_path = dict(zip(model_names, [self.getOnnxPath(model_name, onnx_dir, suffix=model_suffix[model_name]) for model_name in model_names]))
        engine_path = dict(zip(model_names, [self.getEnginePath(model_name, engine_dir, do_engine_refit[model_name], suffix=model_suffix[model_name]) for model_name in model_names]))
        weights_map_path = dict(zip(model_names, [(self.getWeightsMapPath(model_name, onnx_dir) if do_engine_refit[model_name] else None) for model_name in model_names]))

        for model_name, obj in self.models.items():
            if torch_fallback[model_name]:
                continue
            # Export models to ONNX and save weights name mapping
            do_export_onnx = not os.path.exists(engine_path[model_name]) and not os.path.exists(onnx_opt_path[model_name])
            do_export_weights_map = weights_map_path[model_name] and not os.path.exists(weights_map_path[model_name])
            if do_export_onnx or do_export_weights_map:
                # Non-quantized ONNX export
                if not use_int8[model_name] and not use_fp8[model_name]:
                    obj.export_onnx(onnx_path[model_name], onnx_opt_path[model_name], onnx_opset, opt_image_height, opt_image_width, enable_lora_merge=do_lora_merge[model_name], static_shape=static_shape, lora_loader=self.lora_loader)
                else:
                    pipeline = obj.get_pipeline()
                    if self.pipeline_type.is_controlnet():
                        model = UNet2DConditionControlNetModel(pipeline.unet, pipeline.controlnet.nets)
                    else:
                        model = pipeline.unet
                    if use_fp8[model_name] and quantization_level == 4.0:
                        set_fmha(model)

                    state_dict_path = self.getStateDictPath(model_name, onnx_dir, suffix=model_suffix[model_name])
                    if not os.path.exists(state_dict_path):
                        print(f"[I] Calibrated weights not found, generating {state_dict_path}")
                        root_dir = os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__))
                        calibration_file = os.path.join(root_dir, "calibration_data", "calibration-prompts.txt")
                        calibration_prompts = load_calib_prompts(calib_batch_size, calibration_file)
                        if self.pipeline_type.is_controlnet():
                            calibration_image_canny = image_module.download_image(
                                "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/out_bird.png"
                            )
                            # "out_bird.png" has 5 images combined in a row. We pick the first image which is the input image.
                            calibration_image_canny = calibration_image_canny.crop(
                                (0, 0, calibration_image_canny.width / 5, calibration_image_canny.height)
                            )
                            calibration_images = [calibration_image_canny]

                        # TODO check size > calibration_size
                        def do_calibrate(pipeline, calibration_prompts, **kwargs):
                            for i_th, prompts in enumerate(calibration_prompts):
                                if i_th >= kwargs["calib_size"]:
                                    return
                                pipeline_call_kwargs = {
                                    "prompt": prompts,
                                    "num_inference_steps": kwargs["n_steps"],
                                    "negative_prompt": [
                                        "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
                                    ]
                                    * len(prompts),
                                }
                                if self.pipeline_type.is_controlnet():
                                    pipeline_call_kwargs["image"] = calibration_images
                                pipeline(**pipeline_call_kwargs).images

                        def forward_loop(model):
                            if self.pipeline_type.is_controlnet():
                                pipeline.unet = model.unet
                                pipeline.controlnet.nets = model.controlnets
                            else:
                                pipeline.unet = model
                            do_calibrate(
                                pipeline=pipeline,
                                calibration_prompts=calibration_prompts,
                                calib_size=calibration_size // calib_batch_size,
                                n_steps=self.denoising_steps,
                            )

                        print(f"[I] Performing calibration for {calibration_size} steps.")
                        if use_int8[model_name]:
                            quant_config = get_int8_config(
                                model,
                                quantization_level,
                                quantization_alpha,
                                quantization_percentile,
                                self.denoising_steps
                            )
                        elif use_fp8[model_name]:
                            quant_config = SD_FP8_FP32_DEFAULT_CONFIG if self.version == "2.1" else SD_FP8_FP16_DEFAULT_CONFIG

                        # Handle LoRA
                        if do_lora_merge[model_name]:
                            assert self.lora_loader is not None
                            model = merge_loras(model, self.lora_loader)
                        check_lora(model)
                        mtq.quantize(model, quant_config, forward_loop)
                        mto.save(model, state_dict_path)
                    else:
                        mto.restore(model, state_dict_path)

                    print(f"[I] Generating quantized ONNX model: {onnx_opt_path[model_name]}")
                    if not os.path.exists(onnx_path[model_name]):
                        quantize_lvl(self.version, model, quantization_level)
                        mtq.disable_quantizer(model, filter_func)
                        if use_fp8[model_name]:
                            generate_fp8_scales(model)
                    else:
                        model = None
                    obj.export_onnx(onnx_path[model_name], onnx_opt_path[model_name], onnx_opset, opt_image_height, opt_image_width, custom_model=model, static_shape=static_shape)

            # FIXME do_export_weights_map needs ONNX graph
            if do_export_weights_map:
                print(f"[I] Saving weights map: {weights_map_path[model_name]}")
                obj.export_weights_map(onnx_opt_path[model_name], weights_map_path[model_name])

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            if torch_fallback[model_name]:
                continue
            engine = engine_module.Engine(engine_path[model_name])
            if not os.path.exists(engine_path[model_name]):
                update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
                fp16amp = obj.fp16 if not use_fp8[model_name] else False
                bf16amp = obj.bf16 if not use_fp8[model_name] else False
                # TF32 can be enabled for all precisions (including INT8/FP8)
                tf32amp = obj.tf32
                strongly_typed = False if not use_fp8[model_name] else True
                int8amp = use_int8.get('model_name', False)
                precision_constraints = 'prefer' if int8amp else 'none'
                engine.build(onnx_opt_path[model_name],
                    strongly_typed=strongly_typed,
                    fp16=fp16amp,
                    bf16=bf16amp,
                    tf32=tf32amp,
                    int8=int8amp,
                    input_profile=obj.get_input_profile(
                        opt_batch_size, opt_image_height, opt_image_width,
                        static_batch=static_batch, static_shape=static_shape
                    ),
                    enable_refit=do_engine_refit[model_name],
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    update_output_names=update_output_names,
                    verbose=self.verbose,
                    builder_optimization_level=optimization_level,
                    precision_constraints=precision_constraints,
                )
            self.engine[model_name] = engine

        # Load TensorRT engines
        for model_name, obj in self.models.items():
            if torch_fallback[model_name]:
                continue
            self.engine[model_name].load()
            if do_engine_refit[model_name] and self.lora_loader:
                assert weights_map_path[model_name]
                with open(weights_map_path[model_name], 'r') as fp_wts:
                    print(f"[I] Loading weights map: {weights_map_path[model_name]} ")
                    [weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)
                    refit_weights_path = self.getRefitNodesPath(model_name, engine_dir, suffix=lora_suffix)
                    if not os.path.exists(refit_weights_path):
                        print(f"[I] Saving refit weights: {refit_weights_path}")
                        model = merge_loras(obj.get_model(), self.lora_loader)
                        refit_weights, updated_weight_names = engine_module.get_refit_weights(
                            model.state_dict(), onnx_opt_path[model_name], weights_name_mapping, weights_shape_mapping
                        )
                        torch.save((refit_weights, updated_weight_names), refit_weights_path)
                        unload_torch_model(model)
                    else:
                        print(f"[I] Loading refit weights: {refit_weights_path}")
                        refit_weights, updated_weight_names = torch.load(refit_weights_path)
                    self.engine[model_name].refit(refit_weights, updated_weight_names)

        # Load torch models
        for model_name, obj in self.models.items():
            if torch_fallback[model_name]:
                self.torch_models[model_name] = obj.get_model(torch_inference=self.torch_inference)

        # Release temp GPU memory during onnx export to avoid OOM.
        gc.collect()
        torch.cuda.empty_cache()

    def calculateMaxDeviceMemory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activateEngines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculateMaxDeviceMemory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engine.values():
            engine.activate(device_memory=self.shared_device_memory)

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width, latents_dtype=torch.float32):
        latents_dtype = latents_dtype # text_embeddings.dtype
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

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

    def preprocess_images(self, batch_size, images=()):
        if not images:
            return ()
        self.profile_start('preprocess', color='pink')
        input_images=[]
        for image in images:
            image = image.to(self.device).float()
            if image.shape[0] != batch_size:
                image = image.repeat(batch_size, 1, 1, 1)
            input_images.append(image)
        self.profile_stop('preprocess')
        return tuple(input_images)

    def preprocess_controlnet_images(self, batch_size, images=None):
        '''
        images: List of PIL.Image.Image
        '''
        if images is None:
            return None
        self.profile_start('preprocess', color='pink')
        images = [(np.array(i.convert("RGB")).astype(np.float32) / 255.0)[..., None].transpose(3, 2, 0, 1).repeat(batch_size, axis=0) for i in images]
        # do_classifier_free_guidance
        images = [torch.cat([torch.from_numpy(i).to(self.device).float()] * 2) for i in images]
        images = torch.cat([image[None, ...] for image in images], dim=0)
        self.profile_stop('preprocess')
        return images

    def encode_prompt(self, prompt, negative_prompt, encoder='clip', pooled_outputs=False, output_hidden_states=False):
        self.profile_start(encoder, color='green')

        tokenizer = self.tokenizer2 if encoder == 'clip2' else self.tokenizer

        def tokenize(prompt, output_hidden_states):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(self.device)

            text_hidden_states = None
            if self.torch_inference:
                outputs = self.torch_models[encoder](text_input_ids, output_hidden_states=output_hidden_states)
                text_embeddings = outputs[0].clone()
                if output_hidden_states:
                    text_hidden_states = outputs['hidden_states'][-2].clone()
            else:
                # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
                outputs = self.runEngine(encoder, {'input_ids': text_input_ids})
                text_embeddings = outputs['text_embeddings'].clone()
                if output_hidden_states:
                    text_hidden_states = outputs['hidden_states'].clone()
            return text_embeddings, text_hidden_states

        # Tokenize prompt
        text_embeddings, text_hidden_states = tokenize(prompt, output_hidden_states)

        if self.do_classifier_free_guidance:
            # Tokenize negative prompt
            uncond_embeddings, uncond_hidden_states = tokenize(negative_prompt, output_hidden_states)

            # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        if pooled_outputs:
            pooled_output = text_embeddings

        if output_hidden_states:
            text_embeddings = torch.cat([uncond_hidden_states, text_hidden_states]).to(dtype=torch.float16) if self.do_classifier_free_guidance else text_hidden_states

        self.profile_stop(encoder)
        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings

    # from diffusers (get_timesteps)
    def get_timesteps(self, num_inference_steps, strength, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps

        return timesteps, num_inference_steps - t_start

    def denoise_latent(self,
        latents,
        text_embeddings,
        denoiser='unet',
        timesteps=None,
        step_offset=0,
        mask=None,
        masked_image_latents=None,
        image_guidance=1.5,
        controlnet_imgs=None,
        controlnet_scales=None,
        text_embeds=None,
        time_ids=None):

        assert image_guidance > 1.0, "Image guidance has to be > 1.0"

        controlnet_imgs = self.preprocess_controlnet_images(latents.shape[0], controlnet_imgs)

        do_autocast = self.torch_inference != '' and self.models[denoiser].fp16
        with torch.autocast('cuda', enabled=do_autocast):
            self.profile_start(denoiser, color='blue')
            for step_index, timestep in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
                if isinstance(mask, torch.Tensor):
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # Predict the noise residual
                if self.torch_inference:
                    params = {"sample": latent_model_input, "timestep": timestep, "encoder_hidden_states": text_embeddings}
                    if controlnet_imgs is not None:
                        params.update({"images": controlnet_imgs, "controlnet_scales": controlnet_scales})
                    added_cond_kwargs = {}
                    if text_embeds != None:
                        added_cond_kwargs.update({'text_embeds': text_embeds})
                    if time_ids != None:
                        added_cond_kwargs.update({'time_ids': time_ids})
                    if text_embeds != None or time_ids != None:
                        params.update({'added_cond_kwargs': added_cond_kwargs})
                    noise_pred = self.torch_models[denoiser](**params)["sample"]
                else:
                    timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

                    params = {"sample": latent_model_input, "timestep": timestep_float, "encoder_hidden_states": text_embeddings}
                    if controlnet_imgs is not None:
                        params.update({"images": controlnet_imgs, "controlnet_scales": controlnet_scales})
                    if text_embeds != None:
                        params.update({'text_embeds': text_embeds})
                    if time_ids != None:
                        params.update({'time_ids': time_ids})
                    noise_pred = self.runEngine(denoiser, params)['latent']

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # from diffusers (prepare_extra_step_kwargs)
                extra_step_kwargs = {}
                if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                    # TODO: configurable eta
                    eta = 0.0
                    extra_step_kwargs["eta"] = eta
                if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                    extra_step_kwargs["generator"] = self.generator

                latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

            latents = 1. / self.vae_scaling_factor * latents
            latents = latents.to(dtype=torch.float32)

        self.profile_stop(denoiser)
        return latents

    def encode_image(self, input_image):
        self.profile_start('vae_encoder', color='red')
        cast_to = torch.float16 if self.models['vae_encoder'].fp16 else torch.bfloat16 if self.models['vae_encoder'].bf16 else torch.float32
        input_image = input_image.to(dtype=cast_to)
        if self.torch_inference:
            image_latents = self.torch_models['vae_encoder'](input_image)
        else:
            image_latents = self.runEngine('vae_encoder', {'images': input_image})['latent']
        image_latents = self.vae_scaling_factor * image_latents
        self.profile_stop('vae_encoder')
        return image_latents

    def decode_latent(self, latents):
        self.profile_start('vae', color='red')
        cast_to = torch.float16 if self.models['vae'].fp16 else torch.bfloat16 if self.models['vae'].bf16 else torch.float32
        latents = latents.to(dtype=cast_to)

        if self.torch_inference:
            images = self.torch_models['vae'](latents, return_dict=False)[0]
        else:
            images = self.runEngine('vae', {'latent': latents})['images']
        self.profile_stop('vae')
        return images

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print('|-----------------|--------------|')
        print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
        print('|-----------------|--------------|')
        for stage in self.stages:
            stage_name = stage
            if "unet" in stage:
                if self.pipeline_type.is_controlnet():
                    stage_name += '+cnet'
                stage_name += ' x ' + str(denoising_steps)
            print(
                "| {:^15} | {:>9.2f} ms |".format(
                    stage_name, cudart.cudaEventElapsedTime(self.events[stage][0], self.events[stage][1])[1],
                )
            )
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', walltime_ms))
        print('|-----------------|--------------|')
        print('Throughput: {:.5f} image/s'.format(batch_size*1000./walltime_ms))

    def save_image(self, images, pipeline, prompt, seed):
        # Save image
        image_name_prefix = pipeline+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'+str(seed)+'-'
        image_name_suffix = 'torch' if self.torch_inference else 'trt'
        image_module.save_image(images, self.output_dir, image_name_prefix, image_name_suffix)

    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        input_image=None,
        image_strength=0.75,
        mask_image=None,
        controlnet_scales=None,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        warmup=False,
        verbose=False,
        save_image=True,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            input_image (image):
                Input image used to initialize the latents or to be inpainted.
            image_strength (float):
                Strength of transformation applied to input_image. Must be between 0 and 1.
            mask_image (image):
                Mask image containg the region to be inpainted.
            controlnet_scales (torch.Tensor)
                A tensor which containes ControlNet scales, essential for multi ControlNet.
                Must be equal to number of Controlnets.
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
            save_image (bool):
                Save the generated image (if applicable)
        """
        assert len(prompt) == len(negative_prompt)
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        if self.generator and self.seed:
            self.generator.manual_seed(self.seed)

        num_inference_steps = self.denoising_steps

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # TODO: support custom timesteps
            timesteps = None
            if timesteps is not None:
                if "timesteps" not in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys()):
                    raise ValueError(
                        f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                        f" timestep schedules. Please check whether you are using the correct scheduler."
                    )
                self.scheduler.set_timesteps(timesteps=timesteps, device=self.device)
                assert self.denoising_steps == len(self.scheduler.timesteps)
            else:
                self.scheduler.set_timesteps(self.denoising_steps, device=self.device)
            timesteps = self.scheduler.timesteps.to(self.device)

            denoise_kwargs = {}
            if not (self.pipeline_type.is_img2img() or self.pipeline_type.is_sd_xl_refiner()):
                # Initialize latents
                latents = self.initialize_latents(batch_size=batch_size,
                    unet_channels=4,
                    latent_height=latent_height,
                    latent_width=latent_width)
            if self.pipeline_type.is_controlnet():
                denoise_kwargs.update({'controlnet_imgs': input_image, 'controlnet_scales': controlnet_scales})

            # Pre-process and VAE encode input image
            if self.pipeline_type.is_img2img() or self.pipeline_type.is_inpaint() or self.pipeline_type.is_sd_xl_refiner():
                assert input_image != None
                # Initialize timesteps and pre-process input image
                timesteps, num_inference_steps = self.get_timesteps(self.denoising_steps, image_strength)
            denoise_kwargs.update({'timesteps': timesteps})
            if self.pipeline_type.is_img2img() or self.pipeline_type.is_sd_xl_refiner():
                latent_timestep = timesteps[:1].repeat(batch_size)
                input_image = self.preprocess_images(batch_size, (input_image,))[0]
                # Encode if not a latent
                image_latents = input_image if input_image.shape[1] == 4 else self.encode_image(input_image)
                # Add noise to latents using timesteps
                noise = torch.randn(image_latents.shape, generator=self.generator, device=self.device, dtype=torch.float32)
                latents = self.scheduler.add_noise(image_latents, noise, latent_timestep)
            elif self.pipeline_type.is_inpaint():
                mask, mask_image = self.preprocess_images(
                    batch_size, image_module.prepare_mask_and_masked_image(input_image, mask_image)
                )
                mask = torch.nn.functional.interpolate(mask, size=(latent_height, latent_width))
                mask = torch.cat([mask] * 2)
                masked_image_latents = self.encode_image(mask_image)
                masked_image_latents = torch.cat([masked_image_latents] * 2)
                denoise_kwargs.update({'mask': mask, 'masked_image_latents': masked_image_latents})

            # CLIP text encoder(s)
            if self.pipeline_type.is_sd_xl():
                text_embeddings2, pooled_embeddings2 = self.encode_prompt(prompt, negative_prompt,
                        encoder='clip2', pooled_outputs=True, output_hidden_states=True)

                # Merge text embeddings
                if self.pipeline_type.is_sd_xl_base():
                    text_embeddings = self.encode_prompt(prompt, negative_prompt, output_hidden_states=True)
                    text_embeddings = torch.cat([text_embeddings, text_embeddings2], dim=-1)
                else:
                    text_embeddings = text_embeddings2

                # Time embeddings
                def _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype, aesthetic_score=None, negative_aesthetic_score=None):
                    if self.pipeline_type.is_sd_xl_refiner(): #self.requires_aesthetics_score:
                        add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
                        if self.do_classifier_free_guidance:
                            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
                    else:
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        if self.do_classifier_free_guidance:
                            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=self.device)
                    if self.do_classifier_free_guidance:
                        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype, device=self.device)
                        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
                    return add_time_ids

                original_size = (image_height, image_width)
                crops_coords_top_left = (0, 0)
                target_size = (image_height, image_width)
                if self.pipeline_type.is_sd_xl_refiner():
                    add_time_ids = _get_add_time_ids(
                        original_size, crops_coords_top_left, target_size, dtype=text_embeddings.dtype, aesthetic_score=aesthetic_score, negative_aesthetic_score=negative_aesthetic_score
                    )
                else:
                    add_time_ids = _get_add_time_ids(
                        original_size, crops_coords_top_left, target_size, dtype=text_embeddings.dtype
                    )
                add_time_ids = add_time_ids.repeat(batch_size, 1)
                denoise_kwargs.update({'text_embeds': pooled_embeddings2, 'time_ids': add_time_ids})
            else:
                text_embeddings = self.encode_prompt(prompt, negative_prompt)

            # UNet denoiser + (optional) ControlNet(s)
            denoiser = 'unetxl' if self.pipeline_type.is_sd_xl() else 'unet'
            latents = self.denoise_latent(latents, text_embeddings, denoiser=denoiser, **denoise_kwargs)

            # VAE decode latent (if applicable)
            if self.return_latents:
                latents = latents * self.vae_scaling_factor
            else:
                images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.
        if not warmup:
            self.print_summary(num_inference_steps, walltime_ms, batch_size)
            if not self.return_latents and save_image:
                # post-process images
                images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
                self.save_image(images, self.pipeline_type.name.lower(), prompt, self.seed)

        return (latents, walltime_ms) if self.return_latents else (images, walltime_ms)

    def run(self, prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, use_cuda_graph, **kwargs):
        # Process prompt
        if not isinstance(prompt, list):
            raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
        prompt = prompt * batch_size

        if not isinstance(negative_prompt, list):
            raise ValueError(f"`--negative-prompt` must be of type `str` list, but is {type(negative_prompt)}")
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * batch_size

        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                self.infer(prompt, negative_prompt, height, width, warmup=True, **kwargs)

        for _ in range(batch_count):
            print("[I] Running StableDiffusion pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            self.infer(prompt, negative_prompt, height, width, warmup=False, **kwargs)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()
