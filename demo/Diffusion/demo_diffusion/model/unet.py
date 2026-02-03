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

"""
Model definitions for UNet models.
"""

import torch

from demo_diffusion.dynamic_import import import_from_diffusers
from demo_diffusion.model import base_model, load, optimizer
from diffusers import StableDiffusionXLControlNetPipeline

# List of models to import from diffusers.models
models_to_import = [
    "ControlNetModel",
    "UNet2DConditionModel",
    "UNetSpatioTemporalConditionModel",
    "StableCascadeUNet",
]
for model in models_to_import:
    globals()[model] = import_from_diffusers(model, "diffusers.models")


def get_unet_embedding_dim(version, pipeline):
    if version in ("1.4", "dreamshaper-7"):
        return 768
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_base():
        return 2048
    elif version in ("cascade"):
        return 1280
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_refiner():
        return 1280
    elif pipeline.is_img2vid():
        return 1024
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets) -> None:
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets

    def forward(self, sample, timestep, encoder_hidden_states, images, controlnet_scales, added_cond_kwargs=None):
        for i, (image, conditioning_scale, controlnet) in enumerate(zip(images, controlnet_scales, self.controlnets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                return_dict=False,
                added_cond_kwargs=added_cond_kwargs,
            )

            down_samples = [down_sample * conditioning_scale for down_sample in down_samples]
            mid_sample *= conditioning_scale

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
        )
        return noise_pred


class UNetModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        int8=False,
        fp8=False,
        max_batch_size=16,
        text_maxlen=77,
        controlnets=None,
        do_classifier_free_guidance=False,
    ):

        super(UNetModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            int8=int8,
            fp8=fp8,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
            embedding_dim=get_unet_embedding_dim(version, pipeline),
        )
        self.subfolder = "unet"
        self.controlnets = load.get_path(version, pipeline, controlnets) if controlnets else None
        self.unet_dim = 4
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_model(self, torch_inference=""):
        model_opts = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        if self.controlnets:
            unet_model = UNet2DConditionModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            cnet_model_opts = {"torch_dtype": torch.float16} if self.fp16 else {}
            controlnets = torch.nn.ModuleList(
                [ControlNetModel.from_pretrained(path, **cnet_model_opts).to(self.device) for path in self.controlnets]
            )
            # FIXME - cache UNet2DConditionControlNetModel
            model = UNet2DConditionControlNetModel(unet_model, controlnets)
        else:
            unet_model_dir = load.get_checkpoint_dir(
                self.framework_model_dir, self.version, self.pipeline, self.subfolder
            )
            if not load.is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
                model = UNet2DConditionModel.from_pretrained(
                    self.path,
                    subfolder=self.subfolder,
                    use_safetensors=self.hf_safetensor,
                    token=self.hf_token,
                    **model_opts,
                ).to(self.device)
                model.save_pretrained(unet_model_dir, **model_opts)
            else:
                print(f"[I] Load UNet2DConditionModel  model from: {unet_model_dir}")
                model = UNet2DConditionModel.from_pretrained(unet_model_dir, **model_opts).to(self.device)
            if torch_inference:
                model.to(memory_format=torch.channels_last)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        if self.controlnets is None:
            return ["sample", "timestep", "encoder_hidden_states"]
        else:
            return ["sample", "timestep", "encoder_hidden_states", "images", "controlnet_scales"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        if self.controlnets is None:
            return {
                "sample": {0: xB, 2: "H", 3: "W"},
                "encoder_hidden_states": {0: xB},
                "latent": {0: xB, 2: "H", 3: "W"},
            }
        else:
            return {
                "sample": {0: xB, 2: "H", 3: "W"},
                "encoder_hidden_states": {0: xB},
                "images": {1: xB, 3: "8H", 4: "8W"},
                "latent": {0: xB, 2: "H", 3: "W"},
            }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
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
        if self.controlnets is None:
            return {
                "sample": [
                    (self.xB * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                    (self.xB * batch_size, self.unet_dim, latent_height, latent_width),
                    (self.xB * max_batch, self.unet_dim, max_latent_height, max_latent_width),
                ],
                "encoder_hidden_states": [
                    (self.xB * min_batch, self.text_maxlen, self.embedding_dim),
                    (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
                    (self.xB * max_batch, self.text_maxlen, self.embedding_dim),
                ],
            }
        else:
            return {
                "sample": [
                    (self.xB * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                    (self.xB * batch_size, self.unet_dim, latent_height, latent_width),
                    (self.xB * max_batch, self.unet_dim, max_latent_height, max_latent_width),
                ],
                "encoder_hidden_states": [
                    (self.xB * min_batch, self.text_maxlen, self.embedding_dim),
                    (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
                    (self.xB * max_batch, self.text_maxlen, self.embedding_dim),
                ],
                "images": [
                    (len(self.controlnets), self.xB * min_batch, 3, min_image_height, min_image_width),
                    (len(self.controlnets), self.xB * batch_size, 3, image_height, image_width),
                    (len(self.controlnets), self.xB * max_batch, 3, max_image_height, max_image_width),
                ],
            }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        if self.controlnets is None:
            return {
                "sample": (self.xB * batch_size, self.unet_dim, latent_height, latent_width),
                "encoder_hidden_states": (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
                "latent": (self.xB * batch_size, 4, latent_height, latent_width),
            }
        else:
            return {
                "sample": (self.xB * batch_size, self.unet_dim, latent_height, latent_width),
                "encoder_hidden_states": (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
                "images": (len(self.controlnets), self.xB * batch_size, 3, image_height, image_width),
                "latent": (self.xB * batch_size, 4, latent_height, latent_width),
            }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        if self.controlnets is None:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device),
                torch.tensor([1.0], dtype=dtype, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            )
        else:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device),
                torch.tensor(999, dtype=dtype, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
                torch.randn(
                    len(self.controlnets), batch_size, 3, image_height, image_width, dtype=dtype, device=self.device
                ),
                torch.randn(len(self.controlnets), dtype=dtype, device=self.device),
            )

    def optimize(self, onnx_graph):
        if self.fp8:
            return super().optimize(onnx_graph, modify_fp8_graph=True)
        if self.int8:
            return super().optimize(onnx_graph, fuse_mha_qkv_int8=True)
        return super().optimize(onnx_graph)


class UNetXLModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        int8=False,
        fp8=False,
        max_batch_size=16,
        text_maxlen=77,
        do_classifier_free_guidance=False,
    ):
        super(UNetXLModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            int8=int8,
            fp8=fp8,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
            embedding_dim=get_unet_embedding_dim(version, pipeline),
        )
        self.subfolder = "unet"
        self.unet_dim = 4
        self.time_dim = 5 if pipeline.is_sd_xl_refiner() else 6
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_model(self, torch_inference=""):
        model_opts = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        unet_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not load.is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
            model = UNet2DConditionModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            # Use default attention processor for ONNX export
            if not torch_inference:
                model.set_default_attn_processor()
            model.save_pretrained(unet_model_dir, **model_opts)
        else:
            print(f"[I] Load UNet2DConditionModel model from: {unet_model_dir}")
            model = UNet2DConditionModel.from_pretrained(unet_model_dir, **model_opts).to(self.device)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {
            "sample": {0: xB, 2: "H", 3: "W"},
            "encoder_hidden_states": {0: xB},
            "latent": {0: xB, 2: "H", 3: "W"},
            "text_embeds": {0: xB},
            "time_ids": {0: xB},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )
        return {
            "sample": [
                (self.xB * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (self.xB * batch_size, self.unet_dim, latent_height, latent_width),
                (self.xB * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (self.xB * min_batch, self.text_maxlen, self.embedding_dim),
                (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
                (self.xB * max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "text_embeds": [(self.xB * min_batch, 1280), (self.xB * batch_size, 1280), (self.xB * max_batch, 1280)],
            "time_ids": [
                (self.xB * min_batch, self.time_dim),
                (self.xB * batch_size, self.time_dim),
                (self.xB * max_batch, self.time_dim),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (self.xB * batch_size, self.unet_dim, latent_height, latent_width),
            "encoder_hidden_states": (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (self.xB * batch_size, 4, latent_height, latent_width),
            "text_embeds": (self.xB * batch_size, 1280),
            "time_ids": (self.xB * batch_size, self.time_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                self.xB * batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device
            ),
            torch.tensor([1.0], dtype=dtype, device=self.device),
            torch.randn(self.xB * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            {
                "added_cond_kwargs": {
                    "text_embeds": torch.randn(self.xB * batch_size, 1280, dtype=dtype, device=self.device),
                    "time_ids": torch.randn(self.xB * batch_size, self.time_dim, dtype=dtype, device=self.device),
                }
            },
        )

    def optimize(self, onnx_graph):
        if self.fp8:
            return super().optimize(onnx_graph, modify_fp8_graph=True)
        if self.int8:
            return super().optimize(onnx_graph, fuse_mha_qkv_int8=True)
        return super().optimize(onnx_graph)


class UNetXLModelControlNet(UNetXLModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        int8=False,
        fp8=False,
        max_batch_size=16,
        text_maxlen=77,
        controlnets=None,
        do_classifier_free_guidance=False,
    ):
        super().__init__(
            version=version,
            pipeline=pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            int8=int8,
            fp8=fp8,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        self.controlnets = load.get_path(version, pipeline, controlnets) if controlnets else None

    def get_pipeline(self):
        cnet_model_opts = {"torch_dtype": torch.float16} if self.fp16 else {}
        controlnets = [
            ControlNetModel.from_pretrained(path, **cnet_model_opts).to(self.device) for path in self.controlnets
        ]
        if self.bf16:
            model_opts = {"torch_dtype": torch.bfloat16}
        elif self.fp16:
            model_opts = {"variant": "fp16", "torch_dtype": torch.float16}
        else:
            model_opts = {}
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.path,
            use_safetensors=self.hf_safetensor,
            token=self.hf_token,
            controlnet=controlnets,
            **model_opts,
        ).to(self.device)
        return pipeline

    def get_model(self, torch_inference=""):
        model_opts = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        unet_model = UNet2DConditionModel.from_pretrained(
            self.path,
            subfolder=self.subfolder,
            use_safetensors=self.hf_safetensor,
            token=self.hf_token,
            **model_opts,
        ).to(self.device)
        cnet_model_opts = {"torch_dtype": torch.float16} if self.fp16 else {}
        controlnets = torch.nn.ModuleList(
            [ControlNetModel.from_pretrained(path, **cnet_model_opts).to(self.device) for path in self.controlnets]
        )
        # FIXME - cache UNet2DConditionControlNetModel
        model = UNet2DConditionControlNetModel(unet_model, controlnets)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "images", "controlnet_scales", "text_embeds", "time_ids"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        result = super().get_dynamic_axes()
        result["images"] = {1: xB, 3: "8H", 4: "8W"}
        return result

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )
        result = super().get_input_profile(batch_size, image_height, image_width, static_batch, static_shape)
        result["images"] = [
            (len(self.controlnets), self.xB * min_batch, 3, min_image_height, min_image_width),
            (len(self.controlnets), self.xB * batch_size, 3, image_height, image_width),
            (len(self.controlnets), self.xB * max_batch, 3, max_image_height, max_image_width),
        ]
        return result

    def get_shape_dict(self, batch_size, image_height, image_width):
        result = super().get_shape_dict(batch_size, image_height, image_width)
        result["images"] = (len(self.controlnets), self.xB * batch_size, 3, image_height, image_width)
        return result

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        dtype = torch.float16 if self.fp16 else torch.float32
        result = super().get_sample_input(batch_size, image_height, image_width, static_shape)
        result = (
            result[:-1]
            + (
                torch.randn(
                    len(self.controlnets),
                    self.xB * batch_size,
                    3,
                    image_height,
                    image_width,
                    dtype=dtype,
                    device=self.device,
                ),  # images
                torch.randn(len(self.controlnets), dtype=dtype, device=self.device),  # controlnet_scales
            )
            + result[-1:]
        )
        return result


class UNetTemporalModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        fp8=False,
        max_batch_size=16,
        num_frames=14,
        do_classifier_free_guidance=True,
    ):
        super(UNetTemporalModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            fp8=fp8,
            max_batch_size=max_batch_size,
            embedding_dim=get_unet_embedding_dim(version, pipeline),
        )
        self.subfolder = "unet"
        self.unet_dim = 4
        self.num_frames = num_frames
        self.out_channels = 4
        self.cross_attention_dim = 1024
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_model(self, torch_inference=""):
        model_opts = {"torch_dtype": torch.float16} if self.fp16 else {}
        unet_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not load.is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
            model = UNetSpatioTemporalConditionModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(unet_model_dir, **model_opts)
        else:
            print(f"[I] Load UNetSpatioTemporalConditionModel model from: {unet_model_dir}")
            model = UNetSpatioTemporalConditionModel.from_pretrained(unet_model_dir, **model_opts).to(self.device)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "added_time_ids"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        xB = str(self.xB) + "B"
        return {
            "sample": {0: xB, 1: "num_frames", 3: "H", 4: "W"},
            "encoder_hidden_states": {0: xB},
            "added_time_ids": {0: xB},
        }

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
        return {
            "sample": [
                (self.xB * min_batch, self.num_frames, 2 * self.out_channels, min_latent_height, min_latent_width),
                (self.xB * batch_size, self.num_frames, 2 * self.out_channels, latent_height, latent_width),
                (self.xB * max_batch, self.num_frames, 2 * self.out_channels, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (self.xB * min_batch, 1, self.cross_attention_dim),
                (self.xB * batch_size, 1, self.cross_attention_dim),
                (self.xB * max_batch, 1, self.cross_attention_dim),
            ],
            "added_time_ids": [(self.xB * min_batch, 3), (self.xB * batch_size, 3), (self.xB * max_batch, 3)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (self.xB * batch_size, self.num_frames, 2 * self.out_channels, latent_height, latent_width),
            "timestep": (1,),
            "encoder_hidden_states": (self.xB * batch_size, 1, self.cross_attention_dim),
            "added_time_ids": (self.xB * batch_size, 3),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        # TODO chunk_size if forward_chunking is used
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)

        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                self.xB * batch_size,
                self.num_frames,
                2 * self.out_channels,
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device,
            ),
            torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.randn(self.xB * batch_size, 1, self.cross_attention_dim, dtype=dtype, device=self.device),
            torch.randn(self.xB * batch_size, 3, dtype=dtype, device=self.device),
        )

    def optimize(self, onnx_graph):
        return super().optimize(onnx_graph, modify_fp8_graph=self.fp8)


class UNetCascadeModel(base_model.BaseModel):
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
        text_maxlen=77,
        do_classifier_free_guidance=False,
        compression_factor=42,
        latent_dim_scale=10.67,
        image_embedding_dim=768,
        lite=False,
    ):
        super(UNetCascadeModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            bf16=bf16,
            max_batch_size=max_batch_size,
            text_maxlen=text_maxlen,
            embedding_dim=get_unet_embedding_dim(version, pipeline),
            compression_factor=compression_factor,
        )
        self.is_prior = True if pipeline.is_cascade_prior() else False
        self.subfolder = "prior" if self.is_prior else "decoder"
        if lite:
            self.subfolder += "_lite"
        self.prior_dim = 16
        self.decoder_dim = 4
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier
        self.latent_dim_scale = latent_dim_scale
        self.min_latent_shape = self.min_image_shape // self.compression_factor
        self.max_latent_shape = self.max_image_shape // self.compression_factor
        self.do_constant_folding = False
        self.image_embedding_dim = image_embedding_dim

    def get_model(self, torch_inference=""):
        # FP16 variant doesn't exist
        model_opts = {"torch_dtype": torch.float16} if self.fp16 else {}
        model_opts = {"variant": "bf16", "torch_dtype": torch.bfloat16} if self.bf16 else model_opts
        unet_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not load.is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
            model = StableCascadeUNet.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(unet_model_dir, **model_opts)
        else:
            print(f"[I] Load Stable Cascade UNet pytorch model from: {unet_model_dir}")
            model = StableCascadeUNet.from_pretrained(unet_model_dir, **model_opts).to(self.device)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        if self.is_prior:
            return ["sample", "timestep_ratio", "clip_text_pooled", "clip_text", "clip_img"]
        else:
            return ["sample", "timestep_ratio", "clip_text_pooled", "effnet"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        if self.is_prior:
            return {
                "sample": {0: xB, 2: "H", 3: "W"},
                "timestep_ratio": {0: xB},
                "clip_text_pooled": {0: xB},
                "clip_text": {0: xB},
                "clip_img": {0: xB},
                "latent": {0: xB, 2: "H", 3: "W"},
            }
        else:
            return {
                "sample": {0: xB, 2: "H", 3: "W"},
                "timestep_ratio": {0: xB},
                "clip_text_pooled": {0: xB},
                "effnet": {0: xB, 2: "H_effnet", 3: "W_effnet"},
                "latent": {0: xB, 2: "H", 3: "W"},
            }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )
        if self.is_prior:
            return {
                "sample": [
                    (self.xB * min_batch, self.prior_dim, min_latent_height, min_latent_width),
                    (self.xB * batch_size, self.prior_dim, latent_height, latent_width),
                    (self.xB * max_batch, self.prior_dim, max_latent_height, max_latent_width),
                ],
                "timestep_ratio": [(self.xB * min_batch,), (self.xB * batch_size,), (self.xB * max_batch,)],
                "clip_text_pooled": [
                    (self.xB * min_batch, 1, self.embedding_dim),
                    (self.xB * batch_size, 1, self.embedding_dim),
                    (self.xB * max_batch, 1, self.embedding_dim),
                ],
                "clip_text": [
                    (self.xB * min_batch, self.text_maxlen, self.embedding_dim),
                    (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
                    (self.xB * max_batch, self.text_maxlen, self.embedding_dim),
                ],
                "clip_img": [
                    (self.xB * min_batch, 1, self.image_embedding_dim),
                    (self.xB * batch_size, 1, self.image_embedding_dim),
                    (self.xB * max_batch, 1, self.image_embedding_dim),
                ],
            }
        else:
            return {
                "sample": [
                    (
                        self.xB * min_batch,
                        self.decoder_dim,
                        int(min_latent_height * self.latent_dim_scale),
                        int(min_latent_width * self.latent_dim_scale),
                    ),
                    (
                        self.xB * batch_size,
                        self.decoder_dim,
                        int(latent_height * self.latent_dim_scale),
                        int(latent_width * self.latent_dim_scale),
                    ),
                    (
                        self.xB * max_batch,
                        self.decoder_dim,
                        int(max_latent_height * self.latent_dim_scale),
                        int(max_latent_width * self.latent_dim_scale),
                    ),
                ],
                "timestep_ratio": [(self.xB * min_batch,), (self.xB * batch_size,), (self.xB * max_batch,)],
                "clip_text_pooled": [
                    (self.xB * min_batch, 1, self.embedding_dim),
                    (self.xB * batch_size, 1, self.embedding_dim),
                    (self.xB * max_batch, 1, self.embedding_dim),
                ],
                "effnet": [
                    (self.xB * min_batch, self.prior_dim, min_latent_height, min_latent_width),
                    (self.xB * batch_size, self.prior_dim, latent_height, latent_width),
                    (self.xB * max_batch, self.prior_dim, max_latent_height, max_latent_width),
                ],
            }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        if self.is_prior:
            return {
                "sample": (self.xB * batch_size, self.prior_dim, latent_height, latent_width),
                "timestep_ratio": (self.xB * batch_size,),
                "clip_text_pooled": (self.xB * batch_size, 1, self.embedding_dim),
                "clip_text": (self.xB * batch_size, self.text_maxlen, self.embedding_dim),
                "clip_img": (self.xB * batch_size, 1, self.image_embedding_dim),
                "latent": (self.xB * batch_size, self.prior_dim, latent_height, latent_width),
            }
        else:
            return {
                "sample": (
                    self.xB * batch_size,
                    self.decoder_dim,
                    int(latent_height * self.latent_dim_scale),
                    int(latent_width * self.latent_dim_scale),
                ),
                "timestep_ratio": (self.xB * batch_size,),
                "clip_text_pooled": (self.xB * batch_size, 1, self.embedding_dim),
                "effnet": (self.xB * batch_size, self.prior_dim, latent_height, latent_width),
                "latent": (
                    self.xB * batch_size,
                    self.decoder_dim,
                    int(latent_height * self.latent_dim_scale),
                    int(latent_width * self.latent_dim_scale),
                ),
            }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        if self.is_prior:
            return (
                torch.randn(batch_size, self.prior_dim, latent_height, latent_width, dtype=dtype, device=self.device),
                torch.tensor([1.0] * batch_size, dtype=dtype, device=self.device),
                torch.randn(batch_size, 1, self.embedding_dim, dtype=dtype, device=self.device),
                {
                    "clip_text": torch.randn(
                        batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device
                    ),
                    "clip_img": torch.randn(batch_size, 1, self.image_embedding_dim, dtype=dtype, device=self.device),
                },
            )
        else:
            return (
                torch.randn(
                    batch_size,
                    self.decoder_dim,
                    int(latent_height * self.latent_dim_scale),
                    int(latent_width * self.latent_dim_scale),
                    dtype=dtype,
                    device=self.device,
                ),
                torch.tensor([1.0] * batch_size, dtype=dtype, device=self.device),
                torch.randn(batch_size, 1, self.embedding_dim, dtype=dtype, device=self.device),
                {
                    "effnet": torch.randn(
                        batch_size, self.prior_dim, latent_height, latent_width, dtype=dtype, device=self.device
                    ),
                },
            )
