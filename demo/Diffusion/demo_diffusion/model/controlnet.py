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


import os

import torch

from demo_diffusion.dynamic_import import import_from_diffusers
from demo_diffusion.model import base_model, load, optimizer

# List of models to import from diffusers.models
models_to_import = ["SD3Transformer2DModel", "SD3ControlNetModel"]
for model in models_to_import:
    globals()[model] = import_from_diffusers(model, "diffusers.models")


class SD3ControlNetWrapper(torch.nn.Module):
    def __init__(self, controlnet):
        super().__init__()
        self.controlnet = controlnet

    def forward(self, hidden_states, controlnet_cond, conditioning_scale, pooled_projections, timestep):
        params = {
            "hidden_states": hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": conditioning_scale,
        }
        out = self.controlnet(**params)["controlnet_block_samples"]
        return torch.stack(out, dim=0)


class SD3ControlNet(base_model.BaseModel):

    def __init__(
        self,
        version,
        controlnet,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        tf32=False,
        bf16=False,
        int8=False,
        fp8=False,
        max_batch_size=16,
        build_strongly_typed=False,
        do_classifier_free_guidance=False,
    ):
        super(SD3ControlNet, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            int8=int8,
            fp8=fp8,
            max_batch_size=max_batch_size,
        )
        self.path = load.get_path(version, pipeline, controlnet)
        self.subfolder = "controlnet_{}".format(controlnet)
        self.controlnet_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        self.transformer_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, "transformer"
        )
        if not os.path.exists(self.controlnet_model_dir):
            self.config = SD3ControlNetModel.load_config(self.path, token=self.hf_token)
        else:
            print(f"[I] Load SD3ControlNetModel config from: {self.controlnet_model_dir}")
            self.config = SD3ControlNetModel.load_config(self.controlnet_model_dir)
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier
        self.build_strongly_typed = build_strongly_typed

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.controlnet_model_dir, model_opts, self.hf_safetensor):
            model = SD3ControlNetModel.from_pretrained(self.path, **model_opts, use_safetensors=self.hf_safetensor).to(
                self.device
            )
            model.save_pretrained(self.controlnet_model_dir, **model_opts)
        else:
            print(f"[I] Load SD3ControlNetModel model from: {self.controlnet_model_dir}")
            model = SD3ControlNetModel.from_pretrained(self.controlnet_model_dir, **model_opts).to(self.device)

        # Load transformer model for pos_embed
        transformer = SD3Transformer2DModel.from_pretrained(self.transformer_model_dir, **model_opts).to(self.device)

        if hasattr(model.config, "use_pos_embed") and model.config.use_pos_embed is False:
            pos_embed = model._get_pos_embed_from_transformer(transformer)
            model.pos_embed = pos_embed.to(model.dtype).to(model.device)
        # Free transformer model
        del transformer

        model = optimizer.optimize_checkpoint(model, torch_inference)
        model = SD3ControlNetWrapper(model)
        return model

    def get_input_names(self):
        return ["hidden_states", "controlnet_cond", "conditioning_scale", "pooled_projections", "timestep"]

    def get_output_names(self):
        return ["controlnet_block_samples"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        dynamic_axes = {
            "hidden_states": {0: xB, 2: "H", 3: "W"},
            "controlnet_cond": {0: xB, 2: "H", 3: "W"},
            "pooled_projections": {0: xB},
            "timestep": {0: xB},
        }
        return dynamic_axes

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        input_profile = {
            "hidden_states": [
                (self.xB * min_batch, self.config["in_channels"], min_latent_height, min_latent_width),
                (self.xB * batch_size, self.config["in_channels"], latent_height, latent_width),
                (self.xB * max_batch, self.config["in_channels"], max_latent_height, max_latent_width),
            ],
            "timestep": [(self.xB * min_batch,), (self.xB * batch_size,), (self.xB * max_batch,)],
            "pooled_projections": [
                (self.xB * min_batch, self.config["pooled_projection_dim"]),
                (self.xB * batch_size, self.config["pooled_projection_dim"]),
                (self.xB * max_batch, self.config["pooled_projection_dim"]),
            ],
            "controlnet_cond": [
                (self.xB * min_batch, self.config["in_channels"], min_latent_height, min_latent_width),
                (self.xB * batch_size, self.config["in_channels"], latent_height, latent_width),
                (self.xB * max_batch, self.config["in_channels"], max_latent_height, max_latent_width),
            ],
        }
        return input_profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "hidden_states": (self.xB * batch_size, self.config["in_channels"], latent_height, latent_width),
            "timestep": (self.xB * batch_size,),
            "pooled_projections": (self.xB * batch_size, self.config["pooled_projection_dim"]),
            "controlnet_cond": (self.xB * batch_size, self.config["in_channels"], latent_height, latent_width),
            "conditioning_scale": (1,),
            "controlnet_block_samples": (
                self.config["num_layers"],
                self.xB * batch_size,
                latent_height // 2 * latent_width // 2,
                self.config["num_attention_heads"] * self.config["attention_head_dim"],
            ),
        }
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        sample_input = (
            torch.randn(
                self.xB * batch_size,
                self.config["in_channels"],
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device,
            ),
            torch.randn(
                self.xB * batch_size,
                self.config["in_channels"],
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device,
            ),
            torch.tensor(1.0, dtype=dtype, device=self.device),
            torch.randn(self.xB * batch_size, self.config["pooled_projection_dim"], dtype=dtype, device=self.device),
            torch.randn(self.xB * batch_size, dtype=torch.float32, device=self.device),
        )

        return sample_input
