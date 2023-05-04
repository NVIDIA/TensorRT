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
import contextlib
import tensorrt as trt
from utilities import TRT_LOGGER, PIPELINE_TYPE
from stable_diffusion_pipeline import StableDiffusionPipeline
from models import make_tokenizer

class Txt2ImgXLPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Diffusion Txt2Img XL v1.4, v1.5, v2.0, v2.0-base, v2.1, v2.1-base pipeline using NVidia TensorRT.
    """
    def __init__(
        self,
        scheduler="DDIM",
        refiner=False,
        *args, **kwargs
    ):
        """
        Initializes the Txt2Img XL Diffusion pipeline.

        Args:
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the [DPM, LMSD, DDIM, EulerA, PNDM].
        """
        super(Txt2ImgXLPipeline, self).__init__(*args, **kwargs, \
            scheduler=scheduler, stages=['clip', 'clip2', 'unetxl'],
            pipeline_type=PIPELINE_TYPE.SD_XL_BASE, vae_scaling_factor=0.13025)
        
        # Load additional text tokenizer
        self.tokenizer2 = make_tokenizer(self.version, self.pipeline_type, self.hf_token, self.framework_model_dir, subfolder="tokenizer_2")

        self.refiner = refiner

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        seed=None,
        warmup=False,
        verbose=False,
        return_type="latents"
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
            seed (int):
                Seed for the random generator
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
        """
        assert len(prompt) == len(negative_prompt)

        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        guidance = 5.0
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
            text_embeddings = self.encode_prompt(prompt, negative_prompt, 
                                                 encoder='clip', tokenizer=self.tokenizer, 
                                                 output_hidden_states=True)
            # CLIP text encoder 2
            text_embeddings2, pooled_embeddings2 = self.encode_prompt(prompt, negative_prompt, 
                                                                      encoder='clip2', tokenizer=self.tokenizer2, 
                                                                      pooled_outputs=True, output_hidden_states=True)

            # Merged text embeddings
            text_embeddings = torch.cat([text_embeddings, text_embeddings2], dim=-1)

            # Time embeddings
            add_time_ids = self._get_add_time_ids(
                original_size, crops_coords_top_left, target_size, dtype=text_embeddings.dtype
            )
            add_time_ids = add_time_ids.repeat(batch_size, 1)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)

            add_kwargs = {'text_embeds': pooled_embeddings2, 'time_ids': add_time_ids}

            # UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, denoiser='unetxl', guidance=guidance, add_kwargs=add_kwargs)

        # FIXME - SDXL/VAE torch fallback
        with torch.inference_mode():
            # VAE decode latent
            if return_type == "latents":
                images = latents * self.vae_scaling_factor
            else:
                images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            if not warmup:
                print("SD-XL Base Pipeline")
                self.print_summary(self.denoising_steps, e2e_tic, e2e_toc, batch_size)
                if return_type == "image":
                    self.save_image(images, 'txt2img-xl', prompt)

        return images, (e2e_toc - e2e_tic)*1000.
