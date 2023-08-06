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
from utilities import prepare_mask_and_masked_image, TRT_LOGGER, PIPELINE_TYPE
from stable_diffusion_pipeline import StableDiffusionPipeline

class InpaintPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Diffusion Inpainting v1.5, v2.0 pipeline using NVidia TensorRT.
    """
    def __init__(
        self,
        scheduler="PNDM",
        *args, **kwargs
    ):
        """
        Initializes the Inpainting Diffusion pipeline.

        Args:
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the [PNDM].
        """

        if scheduler != "PNDM":
            raise ValueError(f"Inpainting only supports PNDM scheduler")

        super(InpaintPipeline, self).__init__(*args, **kwargs, \
            pipeline_type=PIPELINE_TYPE.INPAINT, scheduler=scheduler, stages=[ 'vae_encoder', 'clip', 'unet', 'vae'])

    def infer(
        self,
        prompt,
        negative_prompt,
        input_image,
        mask_image,
        image_height,
        image_width,
        seed=None,
        strength=0.75,
        warmup = False,
        verbose = False,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            input_image (image):
                Input image to be inpainted.
            mask_image (image):
                Mask image containg the region to be inpainted.
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
                Enable verbose logging.
        """
        batch_size = len(prompt)
        assert len(prompt) == len(negative_prompt)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # Pre-initialize latents
            # TODO: unet_channels = 9?
            latents = self.initialize_latents( \
                batch_size=batch_size, \
                unet_channels=4, \
                latent_height=latent_height, \
                latent_width=latent_width
            )

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Pre-process input images
            mask, masked_image = self.preprocess_images(batch_size, prepare_mask_and_masked_image(input_image, mask_image))
            mask = torch.nn.functional.interpolate(mask, size=(latent_height, latent_width))
            mask = torch.cat([mask] * 2)

            # Initialize timesteps
            timesteps, t_start = self.initialize_timesteps(self.denoising_steps, strength)

            # VAE encode masked image
            masked_latents = self.encode_image(masked_image)
            masked_latents = torch.cat([masked_latents] * 2)

            # CLIP text encoder
            text_embeddings = self.encode_prompt(prompt, negative_prompt)

            # UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, timesteps=timesteps, \
                step_offset=t_start, mask=mask, masked_image_latents=masked_latents)

            # VAE decode latent
            images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            if not warmup:
                self.print_summary(self.denoising_steps, e2e_tic, e2e_toc, batch_size, vae_enc=True)
                self.save_image(images, 'inpaint', prompt)

