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
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from demo_diffusion.dynamic_import import import_from_diffusers
from demo_diffusion.model import base_model, load, optimizer
from demo_diffusion.utils_sd3.other_impls import load_into
from demo_diffusion.utils_sd3.sd3_impls import BaseModel as BaseModelSD3

# List of models to import from diffusers.models
models_to_import = ["FluxTransformer2DModel", "SD3Transformer2DModel", "CosmosTransformer3DModel"]
for model in models_to_import:
    globals()[model] = import_from_diffusers(model, "diffusers.models")

# Import FluxKontextUtil from pipeline module
# Using a deferred import to avoid circular dependencies
def _get_flux_kontext_util():
    from demo_diffusion.pipeline.flux_pipeline import FluxKontextUtil
    return FluxKontextUtil

class SD3_MMDiTModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        shift=1.0,
        fp16=False,
        max_batch_size=16,
        text_maxlen=77,
    ):

        super(SD3_MMDiTModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
        )
        self.subfolder = "sd3"
        self.mmdit_dim = 16
        self.shift = shift
        self.xB = 2

    def get_model(self, torch_inference=""):
        sd3_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        sd3_filename = "sd3_medium.safetensors"
        sd3_model_path = f"{sd3_model_dir}/{sd3_filename}"
        if not os.path.exists(sd3_model_path):
            hf_hub_download(repo_id=self.path, filename=sd3_filename, local_dir=sd3_model_dir)
        with safe_open(sd3_model_path, framework="pt", device=self.device) as f:
            model = BaseModelSD3(
                shift=self.shift, file=f, prefix="model.diffusion_model.", device=self.device, dtype=torch.float16
            ).eval()
            load_into(f, model, "model.", self.device, torch.float16)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["sample", "sigma", "c_crossattn", "y"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {
            "sample": {0: xB, 2: "H", 3: "W"},
            "sigma": {0: xB},
            "c_crossattn": {0: xB},
            "y": {0: xB},
            "latent": {0: xB, 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )
        return {
            "sample": [
                (self.xB * min_batch, self.mmdit_dim, min_latent_height, min_latent_width),
                (self.xB * batch_size, self.mmdit_dim, latent_height, latent_width),
                (self.xB * max_batch, self.mmdit_dim, max_latent_height, max_latent_width),
            ],
            "sigma": [(self.xB * min_batch,), (self.xB * batch_size,), (self.xB * max_batch,)],
            "c_crossattn": [
                (self.xB * min_batch, 154, 4096),
                (self.xB * batch_size, 154, 4096),
                (self.xB * max_batch, 154, 4096),
            ],
            "y": [(self.xB * min_batch, 2048), (self.xB * batch_size, 2048), (self.xB * max_batch, 2048)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (self.xB * batch_size, self.mmdit_dim, latent_height, latent_width),
            "sigma": (self.xB * batch_size,),
            "c_crossattn": (self.xB * batch_size, 154, 4096),
            "y": (self.xB * batch_size, 2048),
            "latent": (self.xB * batch_size, self.mmdit_dim, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(batch_size, self.mmdit_dim, latent_height, latent_width, dtype=dtype, device=self.device),
            torch.randn(batch_size, dtype=dtype, device=self.device),
            {
                "c_crossattn": torch.randn(batch_size, 154, 4096, dtype=dtype, device=self.device),
                "y": torch.randn(batch_size, 2048, dtype=dtype, device=self.device),
            },
        )


class FluxTransformerModel(base_model.BaseModel):

    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        tf32=False,
        int8=False,
        fp8=False,
        bf16=False,
        max_batch_size=16,
        text_maxlen=77,
        build_strongly_typed=False,
        weight_streaming=False,
        weight_streaming_budget_percentage=None,
        kontext_resolution=None,
    ):
        super(FluxTransformerModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            int8=int8,
            fp8=fp8,
            bf16=bf16,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
        )
        self.subfolder = "transformer"
        self.transformer_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.transformer_model_dir):
            self.config = FluxTransformer2DModel.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load FluxTransformer2DModel config from: {self.transformer_model_dir}")
            self.config = FluxTransformer2DModel.load_config(self.transformer_model_dir)
        self.build_strongly_typed = build_strongly_typed
        self.weight_streaming = weight_streaming
        self.weight_streaming_budget_percentage = weight_streaming_budget_percentage
        self.out_channels = self.config.get("out_channels") or self.config["in_channels"]
        self.kontext_resolution = kontext_resolution

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.transformer_model_dir, model_opts, self.hf_safetensor):
            model = FluxTransformer2DModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(self.transformer_model_dir, **model_opts)
        else:
            print(f"[I] Load FluxTransformer2DModel model from: {self.transformer_model_dir}")
            model = FluxTransformer2DModel.from_pretrained(self.transformer_model_dir, **model_opts).to(self.device)
        if torch_inference:
            model.to(memory_format=torch.channels_last)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
        ]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "hidden_states": {0: "B", 1: "latent_dim"},
            "encoder_hidden_states": {0: "B"},
            "pooled_projections": {0: "B"},
            "timestep": {0: "B"},
            "img_ids": {0: "latent_dim"},
        }
        if self.config["guidance_embeds"]:
            dynamic_axes["guidance"] = {0: "B"}

        return dynamic_axes

    def get_context_latent_dim(self, static_shape=False):
        FluxKontextUtil = _get_flux_kontext_util()
        return FluxKontextUtil.get_context_latent_dim(
            version=self.version,
            kontext_resolution=self.kontext_resolution,
            compression_factor=self.compression_factor,
            static_shape=static_shape,
        )

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
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
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        min_context_latent_dim, context_latent_dim, max_context_latent_dim = self.get_context_latent_dim(static_shape)

        input_profile = {
            "hidden_states": [
                (
                    min_batch,
                    (min_latent_height // 2) * (min_latent_width // 2) + min_context_latent_dim,
                    self.config["in_channels"],
                ),
                (
                    batch_size,
                    (latent_height // 2) * (latent_width // 2) + context_latent_dim,
                    self.config["in_channels"],
                ),
                (
                    max_batch,
                    (max_latent_height // 2) * (max_latent_width // 2) + max_context_latent_dim,
                    self.config["in_channels"],
                ),
            ],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.config["joint_attention_dim"]),
                (batch_size, self.text_maxlen, self.config["joint_attention_dim"]),
                (max_batch, self.text_maxlen, self.config["joint_attention_dim"]),
            ],
            "pooled_projections": [
                (min_batch, self.config["pooled_projection_dim"]),
                (batch_size, self.config["pooled_projection_dim"]),
                (max_batch, self.config["pooled_projection_dim"]),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "img_ids": [
                ((min_latent_height // 2) * (min_latent_width // 2) + min_context_latent_dim, 3),
                ((latent_height // 2) * (latent_width // 2) + context_latent_dim, 3),
                ((max_latent_height // 2) * (max_latent_width // 2) + max_context_latent_dim, 3),
            ],
            "txt_ids": [(self.text_maxlen, 3), (self.text_maxlen, 3), (self.text_maxlen, 3)],
        }
        if self.config["guidance_embeds"]:
            input_profile["guidance"] = [(min_batch,), (batch_size,), (max_batch,)]
        return input_profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        _, context_latent_dim, _ = self.get_context_latent_dim()
        shape_dict = {
            "hidden_states": (
                batch_size,
                (latent_height // 2) * (latent_width // 2) + context_latent_dim,
                self.config["in_channels"],
            ),
            "encoder_hidden_states": (batch_size, self.text_maxlen, self.config["joint_attention_dim"]),
            "pooled_projections": (batch_size, self.config["pooled_projection_dim"]),
            "timestep": (batch_size,),
            "img_ids": ((latent_height // 2) * (latent_width // 2) + context_latent_dim, 3),
            "txt_ids": (self.text_maxlen, 3),
            "latent": (batch_size, (latent_height // 2) * (latent_width // 2) + context_latent_dim, self.out_channels),
        }
        if self.config["guidance_embeds"]:
            shape_dict["guidance"] = (batch_size,)
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float32
        assert not (self.fp16 and self.bf16), "fp16 and bf16 cannot be enabled simultaneously"
        tensor_dtype = torch.bfloat16 if self.bf16 else (torch.float16 if self.fp16 else torch.float32)

        sample_input = (
            torch.randn(
                batch_size,
                (latent_height // 2) * (latent_width // 2),
                self.config["in_channels"],
                dtype=tensor_dtype,
                device=self.device,
            ),
            torch.randn(
                batch_size, self.text_maxlen, self.config["joint_attention_dim"], dtype=tensor_dtype, device=self.device
            ),
            torch.randn(batch_size, self.config["pooled_projection_dim"], dtype=tensor_dtype, device=self.device),
            torch.tensor([1.0] * batch_size, dtype=tensor_dtype, device=self.device),
            torch.randn((latent_height // 2) * (latent_width // 2), 3, dtype=dtype, device=self.device),
            torch.randn(self.text_maxlen, 3, dtype=dtype, device=self.device),
            {},
        )
        if self.config["guidance_embeds"]:
            sample_input[-1]["guidance"] = torch.tensor([1.0] * batch_size, dtype=dtype, device=self.device)
        return sample_input

    def optimize(self, onnx_graph):
        if self.fp8:
            return super().optimize(onnx_graph)
        if self.int8:
            return super().optimize(onnx_graph, fuse_mha_qkv_int8=True)
        return super().optimize(onnx_graph)


class UpcastLayer(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Module, upcast_to: torch.dtype):
        super().__init__()
        self.output_dtype = next(base_layer.parameters()).dtype
        self.upcast_to = upcast_to
        self.context_pre_only = base_layer.context_pre_only

        base_layer = base_layer.to(dtype=self.upcast_to)
        self.base_layer = base_layer

    def forward(self, *inputs, **kwargs):
        casted_inputs = tuple(
            in_val.to(self.upcast_to) if isinstance(in_val, torch.Tensor) else in_val for in_val in inputs
        )

        kwarg_casted = {}
        for name, val in kwargs.items():
            kwarg_casted[name] = val.to(dtype=self.upcast_to) if isinstance(val, torch.Tensor) else val

        output = self.base_layer(*casted_inputs, **kwarg_casted)
        if isinstance(output, tuple):
            output = tuple(out.to(self.output_dtype) if isinstance(out, torch.Tensor) else out for out in output)
        else:
            output = output.to(dtype=self.output_dtype)
        return output


class SD3TransformerModel(base_model.BaseModel):

    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        tf32=False,
        bf16=False,
        fp8=False,
        int8=False,
        fp4=False,
        max_batch_size=16,
        text_maxlen=256,
        build_strongly_typed=False,
        weight_streaming=False,
        weight_streaming_budget_percentage=None,
        do_classifier_free_guidance=False,
    ):
        super(SD3TransformerModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            fp8=fp8,
            int8=int8,
            fp4=fp4,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
        )
        self.subfolder = "transformer"
        self.transformer_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.transformer_model_dir):
            self.config = SD3Transformer2DModel.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load SD3Transformer2DModel config from: {self.transformer_model_dir}")
            self.config = SD3Transformer2DModel.load_config(self.transformer_model_dir)
        self.build_strongly_typed = build_strongly_typed
        self.weight_streaming = weight_streaming
        self.weight_streaming_budget_percentage = weight_streaming_budget_percentage
        self.out_channels = self.config.get("out_channels")
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier
        self.num_controlnet_layers = 19  # Can be queried from the ControlNet model config

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.transformer_model_dir, model_opts, self.hf_safetensor):
            model = SD3Transformer2DModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(self.transformer_model_dir, **model_opts)
        else:
            print(f"[I] Load SD3Transformer2DModel model from: {self.transformer_model_dir}")
            model = SD3Transformer2DModel.from_pretrained(self.transformer_model_dir, **model_opts).to(self.device)

        if self.version == "3.5-large":
            model.transformer_blocks[35] = UpcastLayer(model.transformer_blocks[35], torch.float32)

        if torch_inference:
            model.to(memory_format=torch.channels_last)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        input_names = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "block_controlnet_hidden_states"
        ]
        return input_names

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        dynamic_axes = {
            "hidden_states": {0: xB, 2: "H", 3: "W"},
            "encoder_hidden_states": {0: xB},
            "pooled_projections": {0: xB},
            "timestep": {0: xB},
            "latent": {0: xB, 2: "H", 3: "W"},
            "block_controlnet_hidden_states": {1: xB, 2: "latent_dim"}
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
            "encoder_hidden_states": [
                (self.xB * min_batch, self.text_maxlen, self.config["joint_attention_dim"]),
                (self.xB * batch_size, self.text_maxlen, self.config["joint_attention_dim"]),
                (self.xB * max_batch, self.text_maxlen, self.config["joint_attention_dim"]),
            ],
            "pooled_projections": [
                (self.xB * min_batch, self.config["pooled_projection_dim"]),
                (self.xB * batch_size, self.config["pooled_projection_dim"]),
                (self.xB * max_batch, self.config["pooled_projection_dim"]),
            ],
            "timestep": [(self.xB * min_batch,), (self.xB * batch_size,), (self.xB * max_batch,)],
            "block_controlnet_hidden_states":  [
                (
                    self.num_controlnet_layers,
                    self.xB * min_batch,
                    min_latent_height // self.config["patch_size"] * min_latent_width // self.config["patch_size"],
                    self.config["num_attention_heads"] * self.config["attention_head_dim"],
                ),
                (
                    self.num_controlnet_layers,
                    self.xB * batch_size,
                    latent_height // self.config["patch_size"] * latent_width // self.config["patch_size"],
                    self.config["num_attention_heads"] * self.config["attention_head_dim"],
                ),
                (
                    self.num_controlnet_layers,
                    self.xB * max_batch,
                    max_latent_height // self.config["patch_size"] * max_latent_width // self.config["patch_size"],
                    self.config["num_attention_heads"] * self.config["attention_head_dim"],
                ),
            ]
        }

        return input_profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "hidden_states": (self.xB * batch_size, self.config["in_channels"], latent_height, latent_width),
            "encoder_hidden_states": (self.xB * batch_size, self.text_maxlen, self.config["joint_attention_dim"]),
            "pooled_projections": (self.xB * batch_size, self.config["pooled_projection_dim"]),
            "timestep": (self.xB * batch_size,),
            "latent": (self.xB * batch_size, self.out_channels, latent_height, latent_width),
            "block_controlnet_hidden_states": (
                self.num_controlnet_layers,
                self.xB * batch_size,
                latent_height // self.config["patch_size"] * latent_width // self.config["patch_size"],
                self.config["num_attention_heads"] * self.config["attention_head_dim"],
            )
        }
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        assert not (self.fp16 and self.bf16), "fp16 and bf16 cannot be enabled simultaneously"
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
                self.text_maxlen,
                self.config["joint_attention_dim"],
                dtype=dtype,
                device=self.device,
            ),
            torch.randn(self.xB * batch_size, self.config["pooled_projection_dim"], dtype=dtype, device=self.device),
            torch.randn(self.xB * batch_size, dtype=torch.float32, device=self.device),
            {
                "block_controlnet_hidden_states": torch.randn(
                    self.num_controlnet_layers,
                    self.xB * batch_size,
                    latent_height // self.config["patch_size"] * latent_width // self.config["patch_size"],
                    self.config["num_attention_heads"] * self.config["attention_head_dim"],
                    dtype=dtype,
                device=self.device,
                ),
            }
        )

        return sample_input


class CosmosTransformerModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        tf32=False,
        int8=False,
        fp8=False,
        bf16=False,
        max_batch_size=16,
        text_maxlen=77,
        build_strongly_typed=False,
        weight_streaming=False,
        weight_streaming_budget_percentage=None,
    ):
        super(CosmosTransformerModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            int8=int8,
            fp8=fp8,
            bf16=bf16,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
        )
        self.subfolder = "transformer"
        self.transformer_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.transformer_model_dir):
            self.config = CosmosTransformer3DModel.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load CosmosTransformer3DModel config from: {self.transformer_model_dir}")
            self.config = CosmosTransformer3DModel.load_config(self.transformer_model_dir)
        self.build_strongly_typed = build_strongly_typed
        self.weight_streaming = weight_streaming
        self.weight_streaming_budget_percentage = weight_streaming_budget_percentage

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.transformer_model_dir, model_opts, self.hf_safetensor):
            model = CosmosTransformer3DModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(self.transformer_model_dir, **model_opts)
        else:
            print(f"[I] Load CosmosTransformer3DModel model from: {self.transformer_model_dir}")
            model = CosmosTransformer3DModel.from_pretrained(self.transformer_model_dir, **model_opts).to(self.device)
        if torch_inference:
            model.to(memory_format=torch.channels_last)
        if self.fp16:
            model.transformer_blocks[6].attn1.norm_q.float().to(self.device)

        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model.to(self.device)

    def get_input_names(self):
        input_names = [
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "padding_mask",
        ]
        if self.pipeline_type.is_video2world():
            input_names.append("fps")
            input_names.append("condition_mask")
        return input_names

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "hidden_states": {0: "B", 2: "latent_frames", 3: "latent_H", 4: "latent_W"},
            "timestep": {0: "B"},
            "encoder_hidden_states": {0: "B"},
            "padding_mask": {0: "B", 2: "H", 3: "W"},
        }
        if self.pipeline_type.is_video2world():
            dynamic_axes["fps"] = {0: "B"}
            dynamic_axes["condition_mask"] = {0: "B", 2: "latent_frames", 3: "latent_H", 4: "latent_W"}

        return dynamic_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
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
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        latent_frames = 24 if self.pipeline_type.is_video2world() else 1
        latent_channels = (
            self.config["in_channels"] - 1 if self.pipeline_type.is_video2world() else self.config["in_channels"]
        )
        input_profile = {
            "hidden_states": [
                (min_batch, latent_channels, latent_frames, min_latent_height, min_latent_width),
                (batch_size, latent_channels, latent_frames, latent_height, latent_width),
                (max_batch, latent_channels, latent_frames, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.config["text_embed_dim"]),
                (batch_size, self.text_maxlen, self.config["text_embed_dim"]),
                (max_batch, self.text_maxlen, self.config["text_embed_dim"]),
            ],
            "padding_mask": [
                (1, 1, min_image_height, min_image_width),
                (1, 1, image_height, image_width),
                (1, 1, max_image_height, max_image_width),
            ],
        }
        if self.pipeline_type.is_video2world():
            input_profile["fps"] = [(min_batch,), (batch_size,), (max_batch,)]
            input_profile["condition_mask"] = [
                (min_batch, 1, latent_frames, min_latent_height, min_latent_width),
                (batch_size, 1, latent_frames, latent_height, latent_width),
                (max_batch, 1, latent_frames, max_latent_height, max_latent_width),
            ]
        return input_profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        # TODO: get latent_frames from infer call
        latent_frames = 24 if self.pipeline_type.is_video2world() else 1
        latent_channels = (
            self.config["in_channels"] - 1 if self.pipeline_type.is_video2world() else self.config["in_channels"]
        )
        shape_dict = {
            "hidden_states": (batch_size, latent_channels, latent_frames, latent_height, latent_width),
            "timestep": (batch_size,),
            "encoder_hidden_states": (batch_size, self.text_maxlen, self.config["text_embed_dim"]),
            "padding_mask": (1, 1, image_height, image_width),
            "latent": (batch_size, self.config["in_channels"], latent_frames, latent_height, latent_width),
        }

        if self.pipeline_type.is_video2world():
            shape_dict["fps"] = (batch_size,)
            shape_dict["condition_mask"] = (batch_size, 1, latent_frames, latent_height, latent_width)
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float32
        assert not (self.fp16 and self.bf16), "fp16 and bf16 cannot be enabled simultaneously"
        tensor_dtype = torch.bfloat16 if self.bf16 else (torch.float16 if self.fp16 else torch.float32)
        latent_frames = 1
        latent_channels = (
            self.config["in_channels"] - 1 if self.pipeline_type.is_video2world() else self.config["in_channels"]
        )
        sample_input = (
            {
                "hidden_states": torch.randn(
                    batch_size,
                    latent_channels,
                    latent_frames,
                    latent_height,
                    latent_width,
                    dtype=tensor_dtype,
                    device=self.device,
                ),
                "timestep": torch.tensor([1.0] * batch_size, dtype=tensor_dtype, device=self.device),
                "encoder_hidden_states": torch.randn(
                    batch_size, self.text_maxlen, self.config["text_embed_dim"], dtype=tensor_dtype, device=self.device
                ),
                "padding_mask": torch.ones(
                    batch_size, 1, image_height, image_width, dtype=tensor_dtype, device=self.device
                ),
            },
        )
        if self.pipeline_type.is_video2world():
            sample_input[-1]["fps"] = torch.tensor([30] * batch_size, dtype=dtype, device=self.device)
            sample_input[-1]["condition_mask"] = torch.randn(
                batch_size,
                1,
                latent_frames,
                latent_height,
                latent_width,
                dtype=tensor_dtype,
                device=self.device,
            )

        return sample_input
