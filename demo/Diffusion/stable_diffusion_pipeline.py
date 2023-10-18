#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import pathlib
import time

import numpy as np
import nvtx
import onnx
import tensorrt as trt
import torch
from cuda import cudart

from models import (
    make_CLIP,
    make_CLIPWithProj,
    make_tokenizer,
    make_UNet,
    make_UNetXL,
    make_VAE,
    make_VAEEncoder
)
from utilities import (
    PIPELINE_TYPE,
    TRT_LOGGER,
    DDIMScheduler,
    DPMScheduler,
    Engine,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    prepare_mask_and_masked_image,
    save_image
)


class StableDiffusionPipeline:
    """
    Application showcasing the acceleration of Stable Diffusion pipelines using NVidia TensorRT.
    """
    def __init__(
        self,
        version='1.5',
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        max_batch_size=16,
        denoising_steps=50,
        scheduler="DDIM",
        guidance_scale=7.5,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
        vae_scaling_factor=0.18215,
        framework_model_dir='pytorch_model',
        controlnet=None,
        lora_scale=1,
        lora_weights=None,
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
                The scheduler to guide the denoising process. Must be one of [DDIM, DPM, EulerA, LMSD, PNDM].
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
            controlnet (str):
                Which ControlNet/ControlNets to use. 
            return_latents (bool):
                Skip decoding the image and return latents instead.
            torch_inference (str):
                Run inference with PyTorch (using specified compilation mode) instead of TensorRT.
        """

        self.denoising_steps = denoising_steps
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale
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
        self.controlnet = controlnet

        # Schedule options
        sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        if self.version in ("2.0", "2.1"):
            sched_opts['prediction_type'] = 'v_prediction'
        else:
            sched_opts['prediction_type'] = 'epsilon'

        if pipeline_type.is_inpaint() and scheduler != "PNDM":
            raise ValueError(f"Inpainting only supports PNDM scheduler. Specified {scheduler}.")

        if scheduler == "DDIM":
            self.scheduler = DDIMScheduler(device=self.device, **sched_opts)
        elif scheduler == "DPM":
            self.scheduler = DPMScheduler(device=self.device, **sched_opts)
        elif scheduler == "EulerA":
            self.scheduler = EulerAncestralDiscreteScheduler(device=self.device, **sched_opts)
        elif scheduler == "LMSD":
            self.scheduler = LMSDiscreteScheduler(device=self.device, **sched_opts)
        elif scheduler == "PNDM":
            sched_opts["steps_offset"] = 1
            self.scheduler = PNDMScheduler(device=self.device, **sched_opts)
        elif scheduler == "UniPCMultistepScheduler":
            self.scheduler = UniPCMultistepScheduler(device=self.device)
        else:
            raise ValueError(f"Unsupported scheduler {scheduler}. Should be either DDIM, DPM, EulerA, LMSD, PNDM, or UniPCMultistepScheduler.")

        self.pipeline_type = pipeline_type
        if self.pipeline_type.is_txt2img() or self.pipeline_type.is_controlnet():
            self.stages = ['clip','unet','vae']
        elif self.pipeline_type.is_img2img() or self.pipeline_type.is_inpaint():
            self.stages = ['vae_encoder', 'clip','unet','vae']
        elif self.pipeline_type.is_sd_xl_base():
            self.stages = ['clip', 'clip2', 'unetxl']
            if not return_latents:
                self.stages.append('vae')
        elif self.pipeline_type.is_sd_xl_refiner():
            self.stages = ['clip2', 'unetxl', 'vae']
        else:
            raise ValueError(f"Unsupported pipeline {self.pipeline_type.name}.")
        self.return_latents = return_latents

        self.config = {}
        if self.pipeline_type.is_sd_xl():
            self.config['vae_torch_fallback'] = True
            self.config['clip_hidden_states'] = True
        self.torch_inference = torch_inference
        self.use_cuda_graph = use_cuda_graph

        # initialized in loadResources()
        self.stream = None
        self.tokenizer = None
        # initialized in loadEngines()
        self.models = {}
        self.torch_models = {}
        self.engine = {}
        self.shared_device_memory = None
        self.lora_scale=lora_scale
        self.lora_weights=lora_weights

    def loadResources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        self.generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps)
        self.scheduler.configure()

        # Create CUDA events and stream
        self.events = {}
        self.markers = {}
        for stage in ['clip', 'denoise', 'vae', 'vae_encoder']:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        # Skip allocation TensorRT resources for torch inference
        if self.torch_inference:
            return

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            if model_name == 'vae' and self.config.get('vae_torch_fallback', False):
                continue
            self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e[0])
            cudart.cudaEventDestroy(e[1])

        for engine in self.engine.values():
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def cachedModelName(self, model_name):
        if self.pipeline_type.is_inpaint():
            model_name += '_inpaint'
        return model_name

    def getOnnxPath(self, model_name, onnx_dir, opt=True):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+('.opt' if opt else ''))
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'model.onnx')

    def getEnginePath(self, model_name, engine_dir, enable_refit):
        return os.path.join(engine_dir, self.cachedModelName(model_name)+('.refit' if enable_refit else '')+'.trt'+trt.__version__+'.plan')

    def loadEngines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        static_batch=False,
        static_shape=True,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        onnx_refit_dir=None,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
            framework_model_dir (str):
                Directory to write the framework model ckpt.
            onnx_dir (str):
                Directory to write the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            force_export (bool):
                Force re-exporting the ONNX models.
            force_optimize (bool):
                Force re-optimizing the ONNX models.
            force_build (bool):
                Force re-building the TensorRT engine.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_refit (bool):
                Build engines with refit option enabled.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to accelerate build or None
            onnx_refit_dir (str):
                Directory containing refit ONNX models.
        """
        # Create directory
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        # Load text tokenizer(s)
        if not self.pipeline_type.is_sd_xl_refiner():
            self.tokenizer = make_tokenizer(self.version, self.pipeline_type, self.hf_token, framework_model_dir)
        if self.pipeline_type.is_sd_xl():
            self.tokenizer2 = make_tokenizer(self.version, self.pipeline_type, self.hf_token, framework_model_dir, subfolder='tokenizer_2')

        # Load pipeline models
        models_args = {'version': self.version, 'pipeline': self.pipeline_type,
            'hf_token': self.hf_token, 'device': self.device,
            'verbose': self.verbose, 'max_batch_size': self.max_batch_size}

        if 'vae_encoder' in self.stages:
            self.models['vae_encoder'] = make_VAEEncoder(**models_args)
        if 'clip' in self.stages:
            self.models['clip'] = make_CLIP(output_hidden_states=self.config.get('clip_hidden_states', False), **models_args)
        if 'clip2' in self.stages:
            self.models['clip2'] = make_CLIPWithProj(output_hidden_states=self.config.get('clip_hidden_states', False), **models_args)
        if 'unet' in self.stages:
            models_args['lora_scale'] = self.lora_scale
            models_args['lora_weights'] = self.lora_weights
            self.models['unet'] = make_UNet(controlnet=self.controlnet, **models_args)
            models_args.pop('lora_scale')
            models_args.pop('lora_weights')
        if 'unetxl' in self.stages:
            models_args['lora_scale'] = self.lora_scale
            models_args['lora_weights'] = self.lora_weights
            self.models['unetxl'] = make_UNetXL(**models_args)
            models_args.pop('lora_scale')
            models_args.pop('lora_weights')
        if 'vae' in self.stages:
            self.models['vae'] = make_VAE(**models_args)

        if self.torch_inference:
            for k, v in self.models.items():
                self.torch_models[k] = v.get_model(framework_model_dir, torch_inference=self.torch_inference)
            return
        elif 'vae' in self.stages and self.config.get('vae_torch_fallback', False):
            self.torch_models['vae'] = self.models['vae'].get_model(framework_model_dir)

        # setup models to export, optimize and refit.
        force_export_models = []
        force_optimize_models = []
        enable_refit_models = []
        refit_pattern_list = []

        all_models = ['vae_encoder', 'clip', 'clip2', 'unet', 'unetxl', 'vae']
        unet_models = ['unet', 'unetxl']

        # when lora weights is given.
        if self.lora_weights and self.lora_weights != "":
            # build engine with refit enabled, and also refit the weigths.
            onnx_refit_dir = onnx_dir
            force_export_models = unet_models
            force_optimize_models = unet_models
            enable_refit_models = unet_models
            refit_pattern_list = ['onnx::MatMul']

        if force_export:
            force_export_models = all_models 
        if force_optimize:
            force_optimize_models = all_models
        if enable_refit:
            enable_refit_models = all_models

        # Export models to ONNX
        for model_name, obj in self.models.items():
            if model_name == 'vae' and self.config.get('vae_torch_fallback', False):
                continue
            enable_refit = model_name in enable_refit_models
            engine_path = self.getEnginePath(model_name, engine_dir, enable_refit)
            force_export = model_name in force_export_models
            if force_export or force_build or not os.path.exists(engine_path):
                onnx_path = self.getOnnxPath(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.getOnnxPath(model_name, onnx_dir)
                if force_export or not os.path.exists(onnx_opt_path):
                    if force_export or not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = obj.get_model(framework_model_dir)
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = obj.get_sample_input(1, opt_image_height, opt_image_width)
                            torch.onnx.export(model,
                                    inputs,
                                    onnx_path,
                                    export_params=True,
                                    opset_version=onnx_opset,
                                    do_constant_folding=True,
                                    input_names=obj.get_input_names(),
                                    output_names=obj.get_output_names(),
                                    dynamic_axes=obj.get_dynamic_axes(),
                            )
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        print(f"Found cached model: {onnx_path}")

                    # Optimize onnx
                    if model_name in force_optimize_models or not os.path.exists(onnx_opt_path):
                        print(f"Generating optimizing model: {onnx_opt_path}")
                        onnx_opt_graph = obj.optimize(onnx.load(onnx_path))
                        if onnx_opt_graph.ByteSize() > 2147483648:
                            onnx.save_model(
                                onnx_opt_graph,
                                onnx_opt_path,
                                save_as_external_data=True, 
                                all_tensors_to_one_file=True,
                                convert_attribute=False)
                        else:
                            onnx.save(onnx_opt_graph, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            if model_name == 'vae' and self.config.get('vae_torch_fallback', False):
                continue
            enable_refit = model_name in enable_refit_models
            engine_path = self.getEnginePath(model_name, engine_dir, enable_refit)
            engine = Engine(engine_path)
            onnx_path = self.getOnnxPath(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.getOnnxPath(model_name, onnx_dir)

            if force_build or not os.path.exists(engine.engine_path):
                update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
                engine.build(onnx_opt_path,
                    fp16=True,
                    input_profile=obj.get_input_profile(
                        opt_batch_size, opt_image_height, opt_image_width,
                        static_batch=static_batch, static_shape=static_shape
                    ),
                    enable_refit=enable_refit,
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    update_output_names=update_output_names)
            self.engine[model_name] = engine

        # Load TensorRT engines
        for model_name, obj in self.models.items():
            if model_name == 'vae' and self.config.get('vae_torch_fallback', False):
                continue
            self.engine[model_name].load()
            if onnx_refit_dir and model_name in enable_refit_models:
                onnx_refit_path = self.getOnnxPath(model_name, onnx_refit_dir)
                if os.path.exists(onnx_refit_path):
                    onnx_opt_path = self.getOnnxPath(model_name, onnx_dir)
                    self.engine[model_name].refit(onnx_opt_path, onnx_refit_path, refit_pattern_list)

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
            engine.activate(reuse_device_memory=self.shared_device_memory)

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width):
        latents_dtype = torch.float32 # text_embeddings.dtype
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def initialize_timesteps(self, timesteps, strength):
        self.scheduler.set_timesteps(timesteps)
        offset = self.scheduler.steps_offset if hasattr(self.scheduler, "steps_offset") else 0
        init_timestep = int(timesteps * strength) + offset
        init_timestep = min(init_timestep, timesteps)
        t_start = max(timesteps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        return timesteps, t_start

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
        self.profile_start('clip', color='green')

        tokenizer = self.tokenizer2 if encoder == 'clip2' else self.tokenizer

        def tokenize(prompt):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(self.device)

            text_hidden_states = None
            if self.torch_inference:
                outputs = self.torch_models[encoder](text_input_ids)
                text_embeddings = outputs[0].clone()
                if output_hidden_states:
                    text_hidden_states = outputs['last_hidden_state'].clone()
            else:
                # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
                outputs = self.runEngine(encoder, {'input_ids': text_input_ids})
                text_embeddings = outputs['text_embeddings'].clone()
                if output_hidden_states:
                    text_hidden_states = outputs['hidden_states'].clone()
            return text_embeddings, text_hidden_states

        # Tokenize prompt
        text_embeddings, text_hidden_states = tokenize(prompt)

        # Tokenize negative prompt
        uncond_embeddings, uncond_hidden_states = tokenize(negative_prompt)

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        if pooled_outputs:
            pooled_output = text_embeddings

        if output_hidden_states:
            text_embeddings = torch.cat([uncond_hidden_states, text_hidden_states]).to(dtype=torch.float16)

        self.profile_stop('clip')
        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings

    def denoise_latent(self,
        latents,
        text_embeddings,
        denoiser='unet',
        timesteps=None,
        step_offset=0,
        mask=None,
        masked_image_latents=None,
        guidance=7.5,
        image_guidance=1.5,
        controlnet_imgs=None,
        controlnet_scales=None,
        text_embeds=None,
        time_ids=None):

        assert guidance > 1.0, "Guidance has to be > 1.0"
        assert image_guidance > 1.0, "Image guidance has to be > 1.0"

        controlnet_imgs = self.preprocess_controlnet_images(latents.shape[0], controlnet_imgs)

        self.profile_start('denoise', color='blue')
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = self.scheduler.scale_model_input(torch.cat([latents] * 2), step_offset + step_index, timestep)
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
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            if type(self.scheduler) == UniPCMultistepScheduler:
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            else:
                latents = self.scheduler.step(noise_pred, latents, step_offset + step_index, timestep)

        latents = 1. / self.vae_scaling_factor * latents

        self.profile_stop('denoise')
        return latents

    def encode_image(self, input_image):
        self.profile_start('vae_encoder', color='red')
        if self.torch_inference:
            image_latents = self.torch_models['vae_encoder'](input_image)
        else:
            image_latents = self.runEngine('vae_encoder', {'images': input_image})['latent']
        image_latents = self.vae_scaling_factor * image_latents
        self.profile_stop('vae_encoder')
        return image_latents

    def decode_latent(self, latents):
        self.profile_start('vae', color='red')
        if self.torch_inference:
            images = self.torch_models['vae'](latents)['sample']
        elif self.config.get('vae_torch_fallback', False):
            latents = latents.to(dtype=torch.float32)
            self.torch_models["vae"] = self.torch_models["vae"].to(dtype=torch.float32)
            images = self.torch_models['vae'](latents)['sample']
        else:
            images = self.runEngine('vae', {'latent': latents})['images']
        self.profile_stop('vae')
        return images

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print('|-----------------|--------------|')
        print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
        print('|-----------------|--------------|')
        if 'vae_encoder' in self.stages:
            print('| {:^15} | {:>9.2f} ms |'.format('VAE-Enc', cudart.cudaEventElapsedTime(self.events['vae_encoder'][0], self.events['vae_encoder'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('CLIP', cudart.cudaEventElapsedTime(self.events['clip'][0], self.events['clip'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('UNet'+('+CNet' if self.pipeline_type.is_controlnet() else '')+' x '+str(denoising_steps), cudart.cudaEventElapsedTime(self.events['denoise'][0], self.events['denoise'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('VAE-Dec', cudart.cudaEventElapsedTime(self.events['vae'][0], self.events['vae'][1])[1]))
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', walltime_ms))
        print('|-----------------|--------------|')
        print('Throughput: {:.2f} image/s'.format(batch_size*1000./walltime_ms))

    def save_image(self, images, pipeline, prompt):
        # Save image
        image_name_prefix = pipeline+'-fp16'+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'
        save_image(images, self.output_dir, image_name_prefix)

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

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            denoise_kwargs = {}

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

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
                timesteps, t_start = self.initialize_timesteps(self.denoising_steps, image_strength)
                denoise_kwargs.update({'timesteps': timesteps, 'step_offset': t_start})
            if self.pipeline_type.is_img2img() or self.pipeline_type.is_sd_xl_refiner():
                latent_timestep = timesteps[:1].repeat(batch_size)
                input_image = self.preprocess_images(batch_size, (input_image,))[0]
                # Encode if not a latent
                image_latents = input_image if input_image.shape[1] == 4 else self.encode_image(input_image)
                # Add noise to latents using timesteps
                noise = torch.randn(image_latents.shape, generator=self.generator, device=self.device, dtype=torch.float32)
                if type(self.scheduler) == UniPCMultistepScheduler:
                    latents = self.scheduler.add_noise(image_latents, noise, latent_timestep)
                else:
                    latents = self.scheduler.add_noise(image_latents, noise, t_start, latent_timestep)
            elif self.pipeline_type.is_inpaint():
                mask, mask_image = self.preprocess_images(batch_size, prepare_mask_and_masked_image(input_image, mask_image))
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
                        add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
                    else:
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
                    add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)
                    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0).to(device=self.device)
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
            latents = self.denoise_latent(latents, text_embeddings, denoiser=denoiser, guidance=self.guidance_scale, **denoise_kwargs)

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            # VAE decode latent (if applicable)
            if self.return_latents:
                latents = latents * self.vae_scaling_factor
            else:
                images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.
        if not warmup:
            self.print_summary(self.denoising_steps, walltime_ms, batch_size)
            if not self.return_latents and save_image:
                self.save_image(images, self.pipeline_type.name.lower(), prompt)

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


