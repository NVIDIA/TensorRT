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

import numpy as np
import nvtx
import time
import torch
import tensorrt as trt
from utilities import TRT_LOGGER
from stable_diffusion_pipeline import StableDiffusionPipeline

class Img2ImgPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Diffusion Img2Img v1.4, v1.5, v2.0-base, v2.0, v2.1-base, v2.1 pipeline using NVidia TensorRT.
    """
    def __init__(
        self,
        scheduler="DDIM",
        *args, **kwargs
    ):
        """
        Initializes the Img2Img Diffusion pipeline.

        Args:
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the [EulerA, DDIM, DPM, LMSD, PNDM].
        """
        super(Img2ImgPipeline, self).__init__(*args, **kwargs, \
            scheduler=scheduler, stages=['vae_encoder', 'clip', 'unet', 'vae'])

    def infer(
        self,
        prompt,
        negative_prompt,
        init_image,
        image_height,
        image_width,
        seed=None,
        strength=0.75,
        warmup=False,
        verbose=False
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            init_image (image):
                Input image to be used as input.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            seed (int):
                Seed for the random generator
            strength (float):
                How much to transform the input image. Must be between 0 and 1
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
        """
        batch_size = len(prompt)
        assert len(prompt) == len(negative_prompt)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Initialize timesteps
            timesteps, t_start = self.initialize_timesteps(self.denoising_steps, strength)
            latent_timestep = timesteps[:1].repeat(batch_size)

            # Pre-process input image
            init_image = self.preprocess_images(batch_size, (init_image,))[0]

            # VAE encode init image
            init_latents = self.encode_image(init_image)

            # CLIP text encoder
            text_embeddings = self.encode_prompt(prompt, negative_prompt)

            # Add noise to latents using timesteps
            noise = torch.randn(init_latents.shape, generator=self.generator, device=self.device, dtype=torch.float32)
            latents = self.scheduler.add_noise(init_latents, noise, t_start, latent_timestep)

            # UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, timesteps=timesteps, step_offset=t_start)

            # VAE decode latent
            images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            if not warmup:
                self.print_summary(self.denoising_steps, e2e_tic, e2e_toc, batch_size, vae_enc=True)
                self.save_image(images, 'img2img', prompt)
