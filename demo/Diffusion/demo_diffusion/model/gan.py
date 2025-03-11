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

import torch
from diffusers.pipelines.wuerstchen import PaellaVQModel

from demo_diffusion.model import base_model, load, optimizer


class VQGANModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        bf16=False,
        max_batch_size=16,
        compression_factor=42,
        latent_dim_scale=10.67,
        scale_factor=0.3764,
    ):
        super(VQGANModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            bf16=bf16,
            max_batch_size=max_batch_size,
            compression_factor=compression_factor,
        )
        self.subfolder = "vqgan"
        self.latent_dim_scale = latent_dim_scale
        self.scale_factor = scale_factor

    def get_model(self, torch_inference=""):
        model_opts = {"variant": "bf16", "torch_dtype": torch.bfloat16} if self.bf16 else {}
        vqgan_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not load.is_model_cached(vqgan_model_dir, model_opts, self.hf_safetensor, model_name="model"):
            model = PaellaVQModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(vqgan_model_dir, **model_opts)
        else:
            print(f"[I] Load VQGAN pytorch model from: {vqgan_model_dir}")
            model = PaellaVQModel.from_pretrained(vqgan_model_dir, **model_opts).to(self.device)
        model.forward = model.decode
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=dtype, device=self.device)

    def check_dims(self, batch_size, image_height, image_width):
        latent_height, latent_width = super().check_dims(batch_size, image_height, image_width)
        latent_height = int(latent_height * self.latent_dim_scale)
        latent_width = int(latent_width * self.latent_dim_scale)
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = super().get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        min_latent_height = int(min_latent_height * self.latent_dim_scale)
        min_latent_width = int(min_latent_width * self.latent_dim_scale)
        max_latent_height = int(max_latent_height * self.latent_dim_scale)
        max_latent_width = int(max_latent_width * self.latent_dim_scale)
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )
