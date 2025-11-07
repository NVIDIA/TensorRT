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

import inspect
import time

import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from diffusers import DDPMWuerstchenScheduler

from demo_diffusion.model import (
    CLIPWithProjModel,
    UNetCascadeModel,
    VQGANModel,
    make_tokenizer,
)
from demo_diffusion.pipeline.stable_diffusion_pipeline import StableDiffusionPipeline
from demo_diffusion.pipeline.type import PIPELINE_TYPE

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class StableCascadePipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Cascade pipelines using NVidia TensorRT.
    """
    def __init__(
        self,
        version='cascade',
        pipeline_type=PIPELINE_TYPE.CASCADE_PRIOR,
        latent_dim_scale=10.67,
        lite=False,
        **kwargs
    ):
        """
        Initializes the Stable Cascade pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of [cascade]
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline.
            latent_dim_scale (float):
                Multiplier to determine the VQ latent space size from the image embeddings. If the image embeddings are
                height=24 and width=24, the VQ latent shape needs to be height=int(24*10.67)=256 and
                width=int(24*10.67)=256 in order to match the training conditions.
            lite (bool):
                Boolean indicating if the Lite Version of the Stage B and Stage C models is to be used
        """
        super().__init__(
            version=version,
            pipeline_type=pipeline_type,
            **kwargs
        )
        self.config['clip_hidden_states'] = True
        # from Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade.py#L91C9-L91C41
        self.latent_dim_scale = latent_dim_scale
        self.lite = lite

    def initializeModels(self, framework_model_dir, int8, fp8):
        # Load text tokenizer(s)
        self.tokenizer = make_tokenizer(self.version, self.pipeline_type, self.hf_token, framework_model_dir)

        # Load pipeline models
        models_args = {'version': self.version, 'pipeline': self.pipeline_type, 'device': self.device,
            'hf_token': self.hf_token, 'verbose': self.verbose, 'framework_model_dir': framework_model_dir,
            'max_batch_size': self.max_batch_size}

        self.fp16 = False # TODO: enable FP16 mode for decoder model (requires strongly typed engine)
        self.bf16 = True
        if 'clip' in self.stages:
            self.models['clip'] = CLIPWithProjModel(**models_args, fp16=self.fp16, bf16=self.bf16, output_hidden_states=self.config.get('clip_hidden_states', False), subfolder='text_encoder')

        if 'unet' in self.stages:
            self.models['unet'] = UNetCascadeModel(**models_args, fp16=self.fp16, bf16=self.bf16, lite=self.lite, do_classifier_free_guidance=self.do_classifier_free_guidance)

        if 'vqgan' in self.stages:
            self.models['vqgan'] = VQGANModel(**models_args, fp16=self.fp16, bf16=self.bf16, latent_dim_scale = self.latent_dim_scale)

    def encode_prompt(self, prompt, negative_prompt, encoder='clip', pooled_outputs=False, output_hidden_states=False):
        self.profile_start(encoder, color='green')

        tokenizer = self.tokenizer

        def tokenize(prompt, output_hidden_states):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.type(torch.int32).to(self.device)
            attention_mask = text_inputs.attention_mask.type(torch.int32).to(self.device)

            text_hidden_states = None
            if self.torch_inference:
                outputs = self.torch_models[encoder](text_input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
                text_embeddings = outputs[0].clone()
                if output_hidden_states:
                    hidden_state_layer = -1
                    text_hidden_states = outputs['hidden_states'][hidden_state_layer].clone()
            else:
                # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
                outputs = self.runEngine(encoder, {'input_ids': text_input_ids, 'attention_mask': attention_mask})
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
            text_embeddings = torch.cat([text_embeddings, uncond_embeddings])

        if pooled_outputs:
            pooled_output = text_embeddings

        if output_hidden_states:
            text_embeddings = torch.cat([text_hidden_states, uncond_hidden_states]) if self.do_classifier_free_guidance else text_hidden_states

        self.profile_stop(encoder)
        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings

    def denoise_latent(self,
        latents,
        pooled_embeddings,
        text_embeddings=None,
        image_embeds=None,
        effnet=None,
        denoiser='unet',
        timesteps=None,
    ):

        do_autocast = False
        with torch.autocast('cuda', enabled=do_autocast):
            self.profile_start(denoiser, color='blue')
            for step_index, timestep in enumerate(timesteps):
                # ratio input required for stable cascade prior
                timestep_ratio = timestep.expand(latents.size(0)).to(latents.dtype)
                # Expand the latents and timestep_ratio if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep_ratio_input = torch.cat([timestep_ratio] * 2) if self.do_classifier_free_guidance else timestep_ratio

                params = {"sample": latent_model_input, "timestep_ratio": timestep_ratio_input, "clip_text_pooled": pooled_embeddings}
                if text_embeddings is not None:
                    params.update({'clip_text': text_embeddings})
                if image_embeds is not None:
                    params.update({'clip_img': image_embeds})
                if effnet is not None:
                    params.update({'effnet': effnet})

                # Predict the noise residual
                if self.torch_inference:
                    noise_pred = self.torch_models[denoiser](**params)['sample']
                else:
                    noise_pred = self.runEngine(denoiser, params)['latent']

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # from diffusers (prepare_extra_step_kwargs)
                extra_step_kwargs = {}
                if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                    # TODO: configurable eta
                    eta = 0.0
                    extra_step_kwargs["eta"] = eta
                if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                    extra_step_kwargs["generator"] = self.generator

                latents = self.scheduler.step(noise_pred, timestep_ratio, latents, **extra_step_kwargs, return_dict=False)[0]

            latents = latents.to(dtype=torch.bfloat16 if self.bf16 else torch.float32)

        self.profile_stop(denoiser)
        return latents

    def decode_latent(self, latents, model_name='vqgan'):
        self.profile_start(model_name, color='red')
        latents = self.models[model_name].scale_factor * latents
        if self.torch_inference:
            images = self.torch_models[model_name](latents)['sample']
        else:
            images = self.runEngine(model_name, {'latent': latents})['images']
        self.profile_stop(model_name)
        return images

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print('|-----------------|--------------|')
        print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
        print('|-----------------|--------------|')
        for stage in self.stages:
            stage_name = stage + ' x ' + str(denoising_steps) if stage == 'unet' else stage
            print(
                "| {:^15} | {:>9.2f} ms |".format(
                    stage_name, cudart.cudaEventElapsedTime(self.events[stage][0], self.events[stage][1])[1],
                )
            )
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', walltime_ms))
        print('|-----------------|--------------|')
        print('Throughput: {:.5f} image/s'.format(batch_size*1000./walltime_ms))

    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        image_embeddings=None,
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
            image_embeddings (`torch.FloatTensor` or `List[torch.FloatTensor]`):
                Image Embeddings either extracted from an image or generated by a Prior Model.
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
            save_image (bool):
                Save the generated image (if applicable)
        """
        if self.pipeline_type.is_cascade_decoder():
            assert image_embeddings is not None, "Image Embeddings are required to run the decoder. Provided None"
        assert len(prompt) == len(negative_prompt)
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 42
        latent_width = image_width // 42

        if image_embeddings is not None:
            assert latent_height == image_embeddings.shape[-2]
            assert latent_width == image_embeddings.shape[-1]

        if self.generator and self.seed:
            self.generator.manual_seed(self.seed)

        num_inference_steps = self.denoising_steps

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            denoise_kwargs = {}
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
            if isinstance(self.scheduler, DDPMWuerstchenScheduler):
                timesteps = timesteps[:-1]
            denoise_kwargs.update({'timesteps': timesteps})

            # Initialize latents
            latents_dtpye = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
            latents = self.initialize_latents(
                batch_size=batch_size,
                unet_channels=16 if self.pipeline_type.is_cascade_prior() else 4, # TODO: can we query "in_channels" from config
                latent_height=latent_height if self.pipeline_type.is_cascade_prior() else int(latent_height * self.latent_dim_scale),
                latent_width=latent_width if self.pipeline_type.is_cascade_prior() else int(latent_width * self.latent_dim_scale),
                latents_dtype=latents_dtpye
            )

            # CLIP text encoder(s)
            text_embeddings, pooled_embeddings = self.encode_prompt(prompt, negative_prompt,
                        encoder='clip', pooled_outputs=True, output_hidden_states=True)

            if self.pipeline_type.is_cascade_prior():
                denoise_kwargs.update({'text_embeddings': text_embeddings})

                # image embeds
                image_embeds_pooled = torch.zeros(batch_size, 1, 768, device=self.device, dtype=latents_dtpye)
                image_embeds = (torch.cat([image_embeds_pooled, torch.zeros_like(image_embeds_pooled)]) if self.do_classifier_free_guidance else image_embeddings)
                denoise_kwargs.update({'image_embeds': image_embeds})
            else:
                effnet = (torch.cat([image_embeddings, torch.zeros_like(image_embeddings)]) if self.do_classifier_free_guidance else image_embeddings)
                denoise_kwargs.update({'effnet': effnet})

            # UNet denoiser
            latents = self.denoise_latent(latents, pooled_embeddings.unsqueeze(1), denoiser='unet', **denoise_kwargs)

            if not self.return_latents:
                images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.
        if not warmup:
            self.print_summary(num_inference_steps, walltime_ms, batch_size)
            if not self.return_latents and save_image:
                # post-process images
                images = ((images) * 255).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
                self.save_image(images, self.pipeline_type.name.lower(), prompt, self.seed)

        return (latents, walltime_ms) if self.return_latents else (images, walltime_ms)
