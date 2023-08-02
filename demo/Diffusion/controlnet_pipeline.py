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

class ControlNetPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of ControlNet v1.4, v1.5 pipeline using NVidia TensorRT.
    """
    def __init__(
        self,
        scheduler="UniPCMultistepScheduler",
        *args, **kwargs
    ):
        """
        Initializes the ControlNet pipeline.

        Args:
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the [EulerA, DDIM, DPM, LMSD, PNDM, UniPCMultistepScheduler].
        """
        super(ControlNetPipeline, self).__init__(*args, **kwargs, \
            scheduler=scheduler, stages=['clip', 'unet', 'vae'])
    
    def infer(
        self,
        prompt,
        negative_prompt,
        init_image,
        image_height,
        image_width,
        controlnet_scales,
        seed=None,
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
            controlnet_scales (torch.Tensor)
                A tensor which containes ControlNet scales, essential for multi ControlNet. 
                Must be equal to number of Controlnets. 
            seed (int):
                Seed for the random generator
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
        """
        assert len(prompt) == len(negative_prompt)
        batch_size = len(prompt)
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # Pre-initialize latents
            latents = self.initialize_latents( \
                batch_size=batch_size, \
                unet_channels=4, \
                latent_height=(image_height // 8), \
                latent_width=(image_width // 8)
            )

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # CLIP text encoder
            text_embeddings = self.encode_prompt(prompt, negative_prompt)            

            # ControlNet + UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, controlnet_imgs=init_image, controlnet_scales=controlnet_scales)

            # VAE decode latent
            images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            if not warmup:
                self.print_summary(self.denoising_steps, e2e_tic, e2e_toc, batch_size, controlnet=True)
                self.save_image(images, 'controlnet', prompt)