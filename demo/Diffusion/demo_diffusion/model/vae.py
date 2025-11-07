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
from demo_diffusion.utils_sd3.sd3_impls import SDVAE

# List of models to import from diffusers.models
models_to_import = ["AutoencoderKL", "AutoencoderKLTemporalDecoder", "AutoencoderKLWan"]
for model in models_to_import:
    globals()[model] = import_from_diffusers(model, "diffusers.models")

# Import FluxKontextUtil from pipeline module
# Using a deferred import to avoid circular dependencies
def _get_flux_kontext_util():
    from demo_diffusion.pipeline.flux_pipeline import FluxKontextUtil
    return FluxKontextUtil


class VAEModel(base_model.BaseModel):
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
        max_batch_size=16,
    ):
        super(VAEModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch_size=max_batch_size,
        )
        self.subfolder = "vae"
        self.vae_decoder_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.vae_decoder_model_dir):
            self.config = AutoencoderKL.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load AutoencoderKL (decoder) config from: {self.vae_decoder_model_dir}")
            self.config = AutoencoderKL.load_config(self.vae_decoder_model_dir)

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.vae_decoder_model_dir, model_opts, self.hf_safetensor):
            model = AutoencoderKL.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(self.vae_decoder_model_dir, **model_opts)
        else:
            print(f"[I] Load AutoencoderKL (decoder) model from: {self.vae_decoder_model_dir}")
            model = AutoencoderKL.from_pretrained(self.vae_decoder_model_dir, **model_opts).to(self.device)
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
                (min_batch, self.config["latent_channels"], min_latent_height, min_latent_width),
                (batch_size, self.config["latent_channels"], latent_height, latent_width),
                (max_batch, self.config["latent_channels"], max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, self.config["latent_channels"], latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        return torch.randn(
            batch_size, self.config["latent_channels"], latent_height, latent_width, dtype=dtype, device=self.device
        )


class SD3_VAEDecoderModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        fp16=False,
    ):
        super(SD3_VAEDecoderModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            max_batch_size=max_batch_size,
        )
        self.subfolder = "sd3"

    def get_model(self, torch_inference=""):
        dtype = torch.float16 if self.fp16 else torch.float32
        sd3_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        sd3_filename = "sd3_medium.safetensors"
        sd3_model_path = f"{sd3_model_dir}/{sd3_filename}"
        if not os.path.exists(sd3_model_path):
            hf_hub_download(repo_id=self.path, filename=sd3_filename, local_dir=sd3_model_dir)
        with safe_open(sd3_model_path, framework="pt", device=self.device) as f:
            model = SDVAE(device=self.device, dtype=dtype).eval().cuda()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, model, prefix, self.device, dtype)
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
                (min_batch, 16, min_latent_height, min_latent_width),
                (batch_size, 16, latent_height, latent_width),
                (max_batch, 16, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 16, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return torch.randn(batch_size, 16, latent_height, latent_width, dtype=dtype, device=self.device)


class VAEDecTemporalModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=16,
        decode_chunk_size=14,
    ):
        super(VAEDecTemporalModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            max_batch_size=max_batch_size,
        )
        self.subfolder = "vae"
        self.decode_chunk_size = decode_chunk_size

    def get_model(self, torch_inference=""):
        vae_decoder_model_path = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(vae_decoder_model_path):
            model = AutoencoderKLTemporalDecoder.from_pretrained(
                self.path, subfolder=self.subfolder, use_safetensors=self.hf_safetensor, token=self.hf_token
            ).to(self.device)
            model.save_pretrained(vae_decoder_model_path)
        else:
            print(f"[I] Load AutoencoderKLTemporalDecoder model from: {vae_decoder_model_path}")
            model = AutoencoderKLTemporalDecoder.from_pretrained(vae_decoder_model_path).to(self.device)
        model.forward = model.decode
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["latent", "num_frames_in"]

    def get_output_names(self):
        return ["frames"]

    def get_dynamic_axes(self):
        return {"latent": {0: "num_frames_in", 2: "H", 3: "W"}, "frames": {0: "num_frames_in", 2: "8H", 3: "8W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        assert batch_size == 1
        _, _, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )
        return {
            "latent": [
                (1, 4, min_latent_height, min_latent_width),
                (self.decode_chunk_size, 4, latent_height, latent_width),
                (self.decode_chunk_size, 4, max_latent_height, max_latent_width),
            ],
            "num_frames_in": [(1,), (1,), (1,)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        assert batch_size == 1
        return {
            "latent": (self.decode_chunk_size, 4, latent_height, latent_width),
            #'num_frames_in': (1,),
            "frames": (self.decode_chunk_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        assert batch_size == 1
        return (
            torch.randn(
                self.decode_chunk_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            self.decode_chunk_size,
        )


class TorchVAEEncoder(torch.nn.Module):
    def __init__(
        self,
        version,
        pipeline,
        hf_token,
        device,
        path,
        framework_model_dir,
        subfolder,
        fp16=False,
        bf16=False,
        hf_safetensor=False,
    ):
        super().__init__()
        model_opts = {"torch_dtype": torch.float16} if fp16 else {"torch_dtype": torch.bfloat16} if bf16 else {}
        vae_encoder_model_dir = load.get_checkpoint_dir(framework_model_dir, version, pipeline, subfolder)
        if not load.is_model_cached(vae_encoder_model_dir, model_opts, hf_safetensor):
            self.vae_encoder = AutoencoderKL.from_pretrained(
                path, subfolder="vae", use_safetensors=hf_safetensor, token=hf_token, **model_opts
            ).to(device)
            self.vae_encoder.save_pretrained(vae_encoder_model_dir, **model_opts)
        else:
            print(f"[I] Load AutoencoderKL (encoder) model from: {vae_encoder_model_dir}")
            self.vae_encoder = AutoencoderKL.from_pretrained(vae_encoder_model_dir, **model_opts).to(device)

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoderModel(base_model.BaseModel):
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
        max_batch_size=16,
        do_classifier_free_guidance=False,
        kontext_resolution=None,
    ):
        super(VAEEncoderModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch_size=max_batch_size,
        )
        self.kontext_resolution = kontext_resolution
        self.subfolder = "vae"
        self.vae_encoder_model_dir = load.get_checkpoint_dir(
            framework_model_dir, version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.vae_encoder_model_dir):
            self.config = AutoencoderKL.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load AutoencoderKL (encoder) config from: {self.vae_encoder_model_dir}")
            self.config = AutoencoderKL.load_config(self.vae_encoder_model_dir)
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_model(self, torch_inference=""):
        vae_encoder = TorchVAEEncoder(
            self.version,
            self.pipeline,
            self.hf_token,
            self.device,
            self.path,
            self.framework_model_dir,
            self.subfolder,
            self.fp16,
            self.bf16,
            hf_safetensor=self.hf_safetensor,
        )
        return vae_encoder

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {"images": {0: xB, 2: "8H", 3: "8W"}, "latent": {0: xB, 2: "H", 3: "W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )

        if self.version == "flux.1-kontext-dev":
            FluxKontextUtil = _get_flux_kontext_util()
            min_latent_dim, max_latent_dim = FluxKontextUtil.get_min_max_kontext_dimensions()
            return {
                "images": [
                    (self.xB * min_batch, 3, min_latent_dim[1], min_latent_dim[0]),
                    (self.xB * batch_size, 3, self.kontext_resolution[1], self.kontext_resolution[0]),
                    (self.xB * max_batch, 3, max_latent_dim[1], max_latent_dim[0]),
                ],
            }
        return {
            "images": [
                (self.xB * min_batch, 3, min_image_height, min_image_width),
                (self.xB * batch_size, 3, image_height, image_width),
                (self.xB * max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        # Determine dimensions based on version
        if self.version == "flux.1-kontext-dev":
            img_h, img_w = self.kontext_resolution[1], self.kontext_resolution[0]
        else:
            img_h, img_w = image_height, image_width
        latent_height, latent_width = self.check_dims(batch_size, img_h, img_w)

        return {
            "images": (self.xB * batch_size, 3, img_h, img_w),
            "latent": (self.xB * batch_size, self.config["latent_channels"], latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        return torch.randn(self.xB * batch_size, 3, image_height, image_width, dtype=dtype, device=self.device)


class SD3_VAEEncoderModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        fp16=False,
    ):
        super(SD3_VAEEncoderModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            max_batch_size=max_batch_size,
        )
        self.subfolder = "sd3"

    def get_model(self, torch_inference=""):
        dtype = torch.float16 if self.fp16 else torch.float32
        sd3_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        sd3_filename = "sd3_medium.safetensors"
        sd3_model_path = f"{sd3_model_dir}/{sd3_filename}"
        if not os.path.exists(sd3_model_path):
            hf_hub_download(repo_id=self.path, filename=sd3_filename, local_dir=sd3_model_dir)
        with safe_open(sd3_model_path, framework="pt", device=self.device) as f:
            model = SDVAE(device=self.device, dtype=dtype).eval().cuda()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, model, prefix, self.device, dtype)
        model.forward = model.encode
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {"images": {0: "B", 2: "8H", 3: "8W"}, "latent": {0: "B", 2: "H", 3: "W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "images": [
                (min_batch, 3, image_height, image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, image_height, image_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 16, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        dtype = torch.float16 if self.fp16 else torch.float32
        return torch.randn(batch_size, 3, image_height, image_width, dtype=dtype, device=self.device)


class AutoencoderKLWanModel(base_model.BaseModel):

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
        max_batch_size=16,
    ):
        super(AutoencoderKLWanModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch_size=max_batch_size,
        )
        self.subfolder = "vae"
        self.vae_decoder_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.vae_decoder_model_dir):
            self.config = AutoencoderKLWan.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load AutoencoderKLWan (decoder) config from: {self.vae_decoder_model_dir}")
            self.config = AutoencoderKLWan.load_config(self.vae_decoder_model_dir)

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.vae_decoder_model_dir, model_opts, self.hf_safetensor):
            model = AutoencoderKLWan.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(self.vae_decoder_model_dir, **model_opts)
        else:
            print(f"[I] Load AutoencoderKLWan (decoder) model from: {self.vae_decoder_model_dir}")
            model = AutoencoderKLWan.from_pretrained(self.vae_decoder_model_dir, **model_opts).to(self.device)
        model.forward = model.decode
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 3: "H", 4: "W"}, "images": {0: "B", 3: "8H", 4: "8W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = (
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        )
        return {
            "latent": [
                (min_batch, self.config["z_dim"], 1, min_latent_height, min_latent_width),
                (batch_size, self.config["z_dim"], 1, latent_height, latent_width),
                (max_batch, self.config["z_dim"], 1, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, self.config["z_dim"], 1, latent_height, latent_width),
            "images": (batch_size, 3, 1, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        return torch.randn(
            batch_size, self.config["z_dim"], 1, latent_height, latent_width, dtype=dtype, device=self.device
        )


class AutoencoderKLWanEncoderModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode(x).latent_dist.sample()


class AutoencoderKLWanEncoderModel(base_model.BaseModel):
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
        max_batch_size=16,
    ):
        super(AutoencoderKLWanEncoderModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch_size=max_batch_size,
        )
        self.subfolder = "vae"
        self.vae_encoder_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.vae_encoder_model_dir):
            self.config = AutoencoderKLWan.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load AutoencoderKLWan (encoder) config from: {self.vae_encoder_model_dir}")
            self.config = AutoencoderKLWan.load_config(self.vae_encoder_model_dir)

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.vae_encoder_model_dir, model_opts, self.hf_safetensor):
            model = AutoencoderKLWan.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(self.vae_encoder_model_dir, **model_opts)
        else:
            print(f"[I] Load AutoencoderKLWan (encoder) model from: {self.vae_encoder_model_dir}")
            model = AutoencoderKLWan.from_pretrained(self.vae_encoder_model_dir, **model_opts).to(self.device)
        model = AutoencoderKLWanEncoderModelWrapper(model)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model
