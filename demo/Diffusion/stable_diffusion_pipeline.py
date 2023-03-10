#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cuda import cudart
import gc
from models import make_CLIP, make_tokenizer, make_UNet, make_VAE, make_VAEEncoder
import numpy as np
import nvtx
import os
import onnx
from polygraphy import cuda
import torch
from utilities import Engine, device_view, save_image
from utilities import DPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler

class StableDiffusionPipeline:
    """
    Application showcasing the acceleration of Stable Diffusion Txt2Img v1.4, v1.5, v2.0-base, v2.0, v2.1, v2.1-base pipeline using NVidia TensorRT w/ Plugins.
    """
    def __init__(
        self,
        version="2.1",
        inpaint=False,
        stages=['clip','unet','vae'],
        max_batch_size=16,
        denoising_steps=50,
        scheduler="DDIM",
        guidance_scale=7.5,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of [1.4, 1.5, 2.0, 2.0-base, 2.1, 2.1-base]
            inpaint (bool):
                True if inpainting pipeline.
            stages (list):
                Ordered sequence of stages. Options: ['vae_encoder', 'clip','unet','vae']
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
        """

        self.denoising_steps = denoising_steps
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale

        self.max_batch_size = max_batch_size

        # Limit the workspace size for systems with GPU memory larger
        # than 6 GiB to silence OOM warnings from TensorRT optimizer.
        _, free_mem, _ = cudart.cudaMemGetInfo()
        GiB = 2 ** 30
        if free_mem > 6*GiB:
            activation_carveout = 4*GiB
            self.max_workspace_size = free_mem - activation_carveout
        else:
            self.max_workspace_size = 0

        self.output_dir = output_dir
        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        self.version = version

        # Schedule options
        sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        if self.version in ("2.0", "2.1"):
            sched_opts['prediction_type'] = 'v_prediction'
        else:
            sched_opts['prediction_type'] = 'epsilon'

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
        else:
            raise ValueError(f"Scheduler should be either DDIM, DPM, EulerA, LMSD or PNDM")

        self.stages = stages
        self.inpaint = inpaint

        self.stream = None # loaded in loadResources()
        self.tokenizer = None # loaded in loadResources()
        self.models = {} # loaded in loadEngines()
        self.engine = {} # loaded in loadEngines()

    def loadResources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        self.generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps)
        self.scheduler.configure()

        # Create CUDA events and stream
        self.events = {}
        for stage in ['clip', 'denoise', 'vae', 'vae_encoder']:
            for marker in ['start', 'stop']:
                self.events[stage+'-'+marker] = cudart.cudaEventCreate()[1]
        self.stream = cuda.Stream()

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        for engine in self.engine.values():
            del engine

        self.stream.free()
        del self.stream

    def cachedModelName(self, model_name):
        if self.inpaint:
            model_name += '_inpaint'
        return model_name

    def getOnnxPath(self, model_name, onnx_dir, opt=True):
        return os.path.join(onnx_dir, self.cachedModelName(model_name)+('.opt' if opt else '')+'.onnx')

    def getEnginePath(self, model_name, engine_dir):
        return os.path.join(engine_dir, self.cachedModelName(model_name)+'.plan')

    def loadEngines(
        self,
        engine_dir,
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
        enable_preview=False,
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
            enable_preview (bool):
                Enable TensorRT preview features.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to accelerate build or None
            onnx_refit_dir (str):
                Directory containing refit ONNX models.
        """
        # Load text tokenizer
        self.tokenizer = make_tokenizer(self.version, self.hf_token)

        # Load pipeline models
        models_args = {'version': self.version, 'hf_token': self.hf_token, 'device': self.device, \
            'verbose': self.verbose, 'max_batch_size': self.max_batch_size}
        if 'vae_encoder' in self.stages:
            self.models['vae_encoder'] = make_VAEEncoder(inpaint=self.inpaint, **models_args)
        if 'clip' in self.stages:
            self.models['clip'] = make_CLIP(inpaint=self.inpaint, **models_args)
        if 'unet' in self.stages:
            self.models['unet'] = make_UNet(inpaint=self.inpaint, **models_args)
        if 'vae' in self.stages:
            self.models['vae'] = make_VAE(inpaint=self.inpaint, **models_args)

        # Export models to ONNX
        for model_name, obj in self.models.items():
            engine_path = self.getEnginePath(model_name, engine_dir)
            if force_export or force_build or not os.path.exists(engine_path):
                onnx_path = self.getOnnxPath(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.getOnnxPath(model_name, onnx_dir)
                if force_export or not os.path.exists(onnx_opt_path):
                    if force_export or not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = obj.get_model()
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
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
                    if force_optimize or not os.path.exists(onnx_opt_path):
                        print(f"Generating optimizing model: {onnx_opt_path}")
                        onnx_opt_graph = obj.optimize(onnx.load(onnx_path))
                        onnx.save(onnx_opt_graph, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            engine_path = self.getEnginePath(model_name, engine_dir)
            engine = Engine(engine_path)
            onnx_path = self.getOnnxPath(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.getOnnxPath(model_name, onnx_dir)

            if force_build or not os.path.exists(engine.engine_path):
                engine.build(onnx_opt_path,
                    fp16=True,
                    input_profile=obj.get_input_profile(
                        opt_batch_size, opt_image_height, opt_image_width,
                        static_batch=static_batch, static_shape=static_shape
                    ),
                    enable_refit=enable_refit,
                    enable_preview=enable_preview,
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    workspace_size=self.max_workspace_size)
            self.engine[model_name] = engine

        # Load and activate TensorRT engines
        for model_name, obj in self.models.items():
            engine = self.engine[model_name]
            engine.load()
            if onnx_refit_dir:
                onnx_refit_path = self.getOnnxPath(model_name, onnx_refit_dir)
                if os.path.exists(onnx_refit_path):
                    engine.refit(onnx_opt_path, onnx_refit_path)
            engine.activate()

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

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

    def preprocess_images(self, batch_size, images=()):
        if self.nvtx_profile:
            nvtx_image_preprocess = nvtx.start_range(message='image_preprocess', color='pink')
        init_images=[]
        for image in images:
            image = image.to(self.device).float()
            image = image.repeat(batch_size, 1, 1, 1)
            init_images .append(image)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_image_preprocess)
        return tuple(init_images)

    def encode_prompt(self, prompt, negative_prompt):
        if self.nvtx_profile:
            nvtx_clip = nvtx.start_range(message='clip', color='green')
        cudart.cudaEventRecord(self.events['clip-start'], 0)

        # Tokenize prompt
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)

        text_input_ids_inp = device_view(text_input_ids)
        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings = self.runEngine('clip', {"input_ids": text_input_ids_inp})['text_embeddings'].clone()

        # Tokenize negative prompt
        uncond_input_ids = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        uncond_input_ids_inp = device_view(uncond_input_ids)
        uncond_embeddings = self.runEngine('clip', {"input_ids": uncond_input_ids_inp})['text_embeddings']

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        cudart.cudaEventRecord(self.events['clip-stop'], 0)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_clip)

        return text_embeddings

    def denoise_latent(self, latents, text_embeddings, timesteps=None, step_offset=0, mask=None, masked_image_latents=None):
        cudart.cudaEventRecord(self.events['denoise-start'], 0)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step_index, timestep in enumerate(timesteps):
            if self.nvtx_profile:
                nvtx_latent_scale = nvtx.start_range(message='latent_scale', color='pink')

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_offset + step_index, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_latent_scale)

            # Predict the noise residual
            if self.nvtx_profile:
                nvtx_unet = nvtx.start_range(message='unet', color='blue')

            embeddings_dtype = np.float16
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            noise_pred = self.runEngine('unet', {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp})['latent']
            if self.nvtx_profile:
                nvtx.end_range(nvtx_unet)

            if self.nvtx_profile:
                nvtx_latent_step = nvtx.start_range(message='latent_step', color='pink')

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, latents, step_offset + step_index, timestep)

            if self.nvtx_profile:
                nvtx.end_range(nvtx_latent_step)

        latents = 1. / 0.18215 * latents
        cudart.cudaEventRecord(self.events['denoise-stop'], 0)
        return latents

    def encode_image(self, init_image):
        if self.nvtx_profile:
            nvtx_vae = nvtx.start_range(message='vae_encoder', color='red')
        cudart.cudaEventRecord(self.events['vae_encoder-start'], 0)
        init_latents = self.runEngine('vae_encoder', {"images": device_view(init_image)})['latent']
        cudart.cudaEventRecord(self.events['vae_encoder-stop'], 0)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_vae)

        init_latents = 0.18215 * init_latents
        return init_latents

    def decode_latent(self, latents):
        if self.nvtx_profile:
            nvtx_vae = nvtx.start_range(message='vae', color='red')
        cudart.cudaEventRecord(self.events['vae-start'], 0)
        images = self.runEngine('vae', {"latent": device_view(latents)})['images']
        cudart.cudaEventRecord(self.events['vae-stop'], 0)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_vae)
        return images

    def print_summary(self, denoising_steps, tic, toc, vae_enc=False):
            print('|------------|--------------|')
            print('| {:^10} | {:^12} |'.format('Module', 'Latency'))
            print('|------------|--------------|')
            if vae_enc:
                print('| {:^10} | {:>9.2f} ms |'.format('VAE-Enc', cudart.cudaEventElapsedTime(self.events['vae_encoder-start'], self.events['vae_encoder-stop'])[1]))
            print('| {:^10} | {:>9.2f} ms |'.format('CLIP', cudart.cudaEventElapsedTime(self.events['clip-start'], self.events['clip-stop'])[1]))
            print('| {:^10} | {:>9.2f} ms |'.format('UNet x '+str(denoising_steps), cudart.cudaEventElapsedTime(self.events['denoise-start'], self.events['denoise-stop'])[1]))
            print('| {:^10} | {:>9.2f} ms |'.format('VAE-Dec', cudart.cudaEventElapsedTime(self.events['vae-start'], self.events['vae-stop'])[1]))
            print('|------------|--------------|')
            print('| {:^10} | {:>9.2f} ms |'.format('Pipeline', (toc - tic)*1000.))
            print('|------------|--------------|')

    def save_image(self, images, pipeline, prompt):
            # Save image
            image_name_prefix = pipeline+'-fp16'+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'
            save_image(images, self.output_dir, image_name_prefix)
