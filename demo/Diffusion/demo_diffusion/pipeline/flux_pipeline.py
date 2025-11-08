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
from __future__ import annotations

import argparse
import inspect
import os
import time
import warnings
from typing import Any, List, Optional

import numpy as np
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from diffusers.image_processor import VaeImageProcessor
from flux.content_filters import PixtralContentFilter
from huggingface_hub import snapshot_download

from demo_diffusion import path as path_module
from demo_diffusion.model import (
    CLIPModel,
    FLUXLoraLoader,
    FluxTransformerModel,
    T5Model,
    VAEEncoderModel,
    VAEModel,
    get_clip_embedding_dim,
    load,
    make_tokenizer,
)
from demo_diffusion.pipeline.diffusion_pipeline import DiffusionPipeline
from demo_diffusion.pipeline.type import PIPELINE_TYPE

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class FluxKontextUtil:
    """
    Utility class for Flux Kontext-related operations including context dimension calculations
    and resolution handling.
    """

    @staticmethod
    def _get_context_dim(
        image_height: int,
        image_width: int,
        compression_factor: int,
    ) -> int:
        """
        Calculate the context dimension based on image dimensions and compression factor.

        Args:
            image_height: Height of the image
            image_width: Width of the image
            compression_factor: Compression factor applied to the image

        Returns:
            The calculated sequence length for the context
        """
        seq_len = (image_height // (2 * compression_factor)) * (image_width // (2 * compression_factor))
        return seq_len

    @staticmethod
    def get_context_latent_dim(
        version: str,
        kontext_resolution: tuple = None,
        compression_factor: int = 8,
        static_shape: bool = False,
    ):
        """
        Get the context latent dimensions for Flux Kontext models.

        Args:
            version: Model version string
            kontext_resolution: Tuple of (width, height) for kontext resolution
            compression_factor: Compression factor for the model
            static_shape: Whether to use static shapes

        Returns:
            Tuple of (min_context_latent_dim, context_latent_dim, max_context_latent_dim)
        """
        min_context_latent_dim, context_latent_dim, max_context_latent_dim = 0, 0, 0

        if version == "flux.1-kontext-dev":
            assert kontext_resolution is not None, "kontext_resolution must be provided for flux.1-kontext-dev"

            # get opt context size
            context_latent_dim = FluxKontextUtil._get_context_dim(
                image_height=kontext_resolution[1],
                image_width=kontext_resolution[0],
                compression_factor=compression_factor,
            )

            # get min context size
            _, min_context_width, min_context_height = min((w * h, w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
            min_context_latent_dim = (
                context_latent_dim
                if static_shape
                else FluxKontextUtil._get_context_dim(
                    image_height=min_context_height,
                    image_width=min_context_width,
                    compression_factor=compression_factor,
                )
            )

            # get max context size
            _, max_context_width, max_context_height = max((w * h, w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
            max_context_latent_dim = (
                context_latent_dim
                if static_shape
                else FluxKontextUtil._get_context_dim(
                    image_height=max_context_height,
                    image_width=max_context_width,
                    compression_factor=compression_factor,
                )
            )

        return min_context_latent_dim, context_latent_dim, max_context_latent_dim

    @staticmethod
    def get_preferred_resolutions():
        """
        Get the list of preferred Kontext resolutions.

        Returns:
            List of (width, height) tuples representing preferred resolutions
        """
        return PREFERRED_KONTEXT_RESOLUTIONS.copy()

    @staticmethod
    def get_min_max_kontext_dimensions():
        """
        Get the resolution tuples with minimum and maximum aspect ratios from preferred Kontext resolutions.

        Returns:
            Tuple of ((min_aspect_width, min_aspect_height), (max_aspect_width, max_aspect_height))
        """
        widths = [w for w, h in PREFERRED_KONTEXT_RESOLUTIONS]
        heights = [h for w, h in PREFERRED_KONTEXT_RESOLUTIONS]

        min_width = min(widths)
        max_width = max(widths)
        min_height = min(heights)
        max_height = max(heights)

        return (min_width, min_height), (max_width, max_height)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FluxPipeline(DiffusionPipeline):
    """
    Application showcasing the acceleration of Flux pipelines using Nvidia TensorRT.
    """

    def __init__(
        self,
        version="flux.1-dev",
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        guidance_scale=3.5,
        max_sequence_length=512,
        calibration_dataset=None,
        t5_weight_streaming_budget_percentage=None,
        transformer_weight_streaming_budget_percentage=None,
        lora_scale: float = 1.0,
        lora_weight: Optional[List[float]] = None,
        lora_path: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initializes the Flux pipeline.

        Args:
            version (`str`, defaults to `flux.1-dev`)
                Version of the underlying Flux model.
            guidance_scale (`float`, defaults to 3.5):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            max_sequence_length (`int`, defaults to 512):
                Maximum sequence length to use with the `prompt`.
            t5_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the T5 model.
            transformer_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the transformer model.
        """
        super().__init__(
            version=version,
            pipeline_type=pipeline_type,
            text_encoder_weight_streaming_budget_percentage=t5_weight_streaming_budget_percentage,
            denoiser_weight_streaming_budget_percentage=transformer_weight_streaming_budget_percentage,
            **kwargs,
        )
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        self.calibration_dataset = calibration_dataset  # Currently supported for Flux ControlNet pipelines only

        # Initialize LoRA
        self.lora_loader = None
        if lora_path:
            self.lora_weights = dict()
            self.lora_loader = FLUXLoraLoader(lora_path, lora_weight, lora_scale)
            assert len(lora_path) == len(lora_weight)
            for i, path in enumerate(lora_path):
                self.lora_weights[path] = lora_weight[i]

    @classmethod
    def FromArgs(cls, args: argparse.Namespace, pipeline_type: PIPELINE_TYPE) -> FluxPipeline:
        """Factory method to construct a `FluxPipeline` object from parsed arguments.

        Overrides:
            DiffusionPipeline.FromArgs
        """
        MAX_BATCH_SIZE = 4
        DEVICE = "cuda"
        DO_RETURN_LATENTS = False

        # Resolve all paths.
        dd_path = path_module.resolve_path(
            cls.get_model_names(pipeline_type), args, pipeline_type, cls._get_pipeline_uid(args.version)
        )

        return cls(
            dd_path=dd_path,
            version=args.version,
            pipeline_type=pipeline_type,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            bf16=args.bf16,
            calibration_dataset=args.calibration_dataset if hasattr(args, "calibration_dataset") else None,
            low_vram=args.low_vram,
            torch_fallback=args.torch_fallback,
            weight_streaming=args.ws,
            t5_weight_streaming_budget_percentage=args.t5_ws_percentage,
            transformer_weight_streaming_budget_percentage=args.transformer_ws_percentage,
            max_batch_size=MAX_BATCH_SIZE,
            denoising_steps=args.denoising_steps,
            scheduler=args.scheduler,
            lora_scale=args.lora_scale,
            lora_weight=args.lora_weight,
            lora_path=args.lora_path,
            device=DEVICE,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            verbose=args.verbose,
            nvtx_profile=args.nvtx_profile,
            use_cuda_graph=args.use_cuda_graph,
            framework_model_dir=args.framework_model_dir,
            return_latents=DO_RETURN_LATENTS,
            torch_inference=args.torch_inference,
        )

    @classmethod
    def get_model_names(cls, pipeline_type: PIPELINE_TYPE, controlnet_type: str = None) -> List[str]:
        """Return a list of model names used by this pipeline.

        Overrides:
            DiffusionPipeline.get_model_names
        """
        if pipeline_type.is_img2img():
            return ["clip", "t5", "transformer", "vae", "vae_encoder"]
        else:
            return ["clip", "t5", "transformer", "vae"]

    def download_onnx_models(self, model_name: str, model_config: dict[str, Any]) -> None:
        if self.fp16:
            raise ValueError(
                "ONNX models can be downloaded only for the following precisions: BF16, FP8, FP4. This pipeline is running in FP16."
            )

        hf_download_path = "-".join([load.get_path(self.version, self.pipeline_type.name), "onnx"])
        model_path = model_config["onnx_opt_path"]
        base_dir = os.path.dirname(os.path.dirname(model_config["onnx_opt_path"]))

        if not os.path.exists(model_path):
            if model_name == "clip":
                dirname = "clip.opt"
            elif model_name == "t5":
                dirname = "t5.opt"
            elif model_name == "transformer":
                if model_config["use_fp4"]:
                    dirname = "transformer.opt/fp4"
                    if self.version == "flux.1-kontext-dev":
                        dirname = "_".join([dirname, "svd32"])
                elif model_config["use_fp8"]:
                    dirname = "transformer.opt/fp8"
                elif self.bf16:
                    dirname = "transformer.opt/bf16"
            elif model_name == "vae":
                dirname = "vae.opt"
            elif model_name == "vae_encoder":
                dirname = "vae_encoder.opt"
            else:
                raise ValueError(f"{model_name} not found in {self.stages}")

            snapshot_download(
                repo_id=hf_download_path,
                allow_patterns=os.path.join(dirname, "*"),
                local_dir=base_dir,
                token=self.hf_token,
            )
            # Rename directory from <model_name>.opt to <model_name>
            saved_dir = os.path.join(base_dir, dirname)
            model_dir = os.path.dirname(model_path)
            os.rename(saved_dir, model_dir)
            # Rename model from model.onnx to model_optimized.onnx
            os.rename(os.path.join(model_dir, "model.onnx"), model_path)

    def is_native_export_supported(self, model_config: dict[str, Any]) -> bool:
        if self.version.startswith("flux.1") and model_config["use_fp4"]:
            # Native export not supported for FP4.
            raise ValueError(
                f"Native FP4 quantization is not supported. No ONNX model found in {model_config['onnx_opt_path']}. Please pass --download-onnx-models."
            )
        if (
            self.version in ["flux.1-dev-canny", "flux.1-dev-depth"]
            and model_config["use_fp8"]
            and not self.calibration_dataset
        ):
            # Native export of FP8 model requires calibration data.
            raise ValueError(
                f"No ONNX model found in {model_config['onnx_opt_path']}. Please pass --download-onnx-models. If you would like to quantize and export natively, please provide calibration data using --calibration-."
            )
        return True

    def _initialize_models(self, framework_model_dir, int8, fp8, fp4):
        # Load text tokenizer(s)
        self.tokenizer = make_tokenizer(
            self.version, self.pipeline_type, self.hf_token, framework_model_dir,
        )
        self.tokenizer2 = make_tokenizer(
            self.version,
            self.pipeline_type,
            self.hf_token,
            framework_model_dir,
            subfolder="tokenizer_2",
            tokenizer_type="t5",
        )

        # Load pipeline models
        models_args = {
            "version": self.version,
            "pipeline": self.pipeline_type,
            "device": self.device,
            "hf_token": self.hf_token,
            "verbose": self.verbose,
            "framework_model_dir": framework_model_dir,
            "max_batch_size": self.max_batch_size,
        }

        self.bf16 = True if int8 or fp8 or fp4 else self.bf16
        self.fp16 = True if not self.bf16 else False
        self.tf32 = True
        if "clip" in self.stages:
            self.models["clip"] = CLIPModel(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                embedding_dim=get_clip_embedding_dim(self.version, self.pipeline_type),
                keep_pooled_output=True,
                subfolder="text_encoder",
            )

        if "t5" in self.stages:
            # Known accuracy issues with FP16
            self.models["t5"] = T5Model(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                subfolder="text_encoder_2",
                text_maxlen=self.max_sequence_length,
                build_strongly_typed=True,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.text_encoder_weight_streaming_budget_percentage,
            )

        if "vae" in self.stages:
            # Accuracy issues with FP16
            self.models["vae"] = VAEModel(**models_args, fp16=False, tf32=self.tf32, bf16=self.bf16)

        self.vae_scale_factor = (
            2 ** (len(self.models["vae"].config["block_out_channels"]))
            if "vae" in self.stages and self.models["vae"] is not None
            else 16
        )
        self.vae_latent_channels = (
            self.models["vae"].config["latent_channels"]
            if "vae" in self.stages and self.models["vae"] is not None
            else 16
        )

        if "vae_encoder" in self.stages:
            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor * 2, vae_latent_channels=self.vae_latent_channels
            )
            # Add kontext_resolution if available (for FluxKontextPipeline)
            vae_encoder_kwargs = {}
            if hasattr(self, "kontext_image") and self.kontext_image is not None:
                self.resize_height, self.resize_width = self._get_resize_dimensions(self.kontext_image)
                vae_encoder_kwargs["kontext_resolution"] = (self.resize_width, self.resize_height)

            self.models["vae_encoder"] = VAEEncoderModel(**models_args, fp16=False, tf32=self.tf32, bf16=self.bf16, **vae_encoder_kwargs)

        if "transformer" in self.stages:
            transformer_kwargs = {
                **models_args,
                "bf16": self.bf16,
                "fp16": self.fp16,
                "int8": int8,
                "fp8": fp8,
                "tf32": self.tf32,
                "text_maxlen": self.max_sequence_length,
                "build_strongly_typed": True,
                "weight_streaming": self.weight_streaming,
                "weight_streaming_budget_percentage": self.denoiser_weight_streaming_budget_percentage,
            }
            if hasattr(self, "kontext_image") and self.kontext_image is not None:
                transformer_kwargs["kontext_resolution"] = (self.resize_width, self.resize_height)
            self.models["transformer"] = FluxTransformerModel(**transformer_kwargs)

    def encode_image(self, input_image, encoder="vae_encoder"):
        self.profile_start(encoder, color='red')
        cast_to = torch.float16 if self.models[encoder].fp16 else torch.bfloat16 if self.models[encoder].bf16 else torch.float32
        input_image = input_image.to(dtype=cast_to)
        if self.torch_inference or self.torch_fallback[encoder]:
            image_latents = self.torch_models[encoder](input_image)
        else:
            image_latents = self.run_engine(encoder, {'images': input_image})['latent']

        image_latents = self.models[encoder].config["scaling_factor"] * (image_latents - self.models[encoder].config["shift_factor"])
        self.profile_stop(encoder)
        return image_latents

    # Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py#L546
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/pipelines/flux/pipeline_flux.py#L436
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """
        Reshapes latents from (B, C, H, W) to (B, H/2, W/2, C*4) as expected by the denoiser
        """
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/pipelines/flux/pipeline_flux.py#L444
    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """
        Reshapes denoised latents to the format (B, C, H, W)
        """
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(
            batch_size, channels // (2 * 2), height * 2, width * 2
        )

        return latents

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/pipelines/flux/pipeline_flux.py#L421
    @staticmethod
    def _prepare_latent_image_ids(height, width, dtype, device):
        """
        Prepares latent image indices
        """
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def initialize_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        latent_timestep=None,
        image_latents=None,
        latents_dtype=torch.float32,
    ):
        latents_dtype = latents_dtype  # text_embeddings.dtype
        latents_shape = (batch_size, num_channels_latents, latent_height, latent_width)
        latents = torch.randn(
            latents_shape,
            device=self.device,
            dtype=latents_dtype,
            generator=self.generator,
        )

        if image_latents is not None:
            image_latents = torch.cat([image_latents], dim=0).to(latents_dtype)
            latents = self.scheduler.scale_noise(image_latents, latent_timestep, latents)

        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, latent_height, latent_width
        )

        latent_image_ids = self._prepare_latent_image_ids(latent_height, latent_width, latents_dtype, self.device)

        return latents, latent_image_ids

    # Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_img2img.py#L416C1
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def encode_prompt(
        self, prompt, encoder="clip", max_sequence_length=None, pooled_output=False
    ):
        self.profile_start(encoder, color="green")

        tokenizer = self.tokenizer2 if encoder == "t5" else self.tokenizer
        max_sequence_length = (
            tokenizer.model_max_length
            if max_sequence_length is None
            else max_sequence_length
        )

        def tokenize(prompt, max_sequence_length):
            text_input_ids = (
                tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_overflowing_tokens=False,
                    return_length=False,
                    return_tensors="pt",
                )
                .input_ids.type(torch.int32)
                .to(self.device)
            )

            untruncated_ids = tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids.type(torch.int32).to(self.device)
            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, max_sequence_length - 1 : -1]
                )
                warnings.warn(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f"{max_sequence_length} tokens: {removed_text}"
                )

            if self.torch_inference or self.torch_fallback[encoder]:
                outputs = self.torch_models[encoder](
                    text_input_ids, output_hidden_states=False
                )
                text_encoder_output = (
                    outputs[0].clone()
                    if pooled_output == False
                    else outputs.pooler_output.clone()
                )
            else:
                # NOTE: output tensor for the encoder must be cloned because it will be overwritten when called again for prompt2
                outputs = self.run_engine(encoder, {"input_ids": text_input_ids})
                output_name = (
                    "text_embeddings" if not pooled_output else "pooled_embeddings"
                )
                text_encoder_output = outputs[output_name].clone()

            return text_encoder_output

        # Tokenize prompt
        text_encoder_output = tokenize(prompt, max_sequence_length)

        self.profile_stop(encoder)
        return text_encoder_output.to(torch.float16) if self.fp16 else text_encoder_output.to(torch.bfloat16) if self.bf16 else text_encoder_output

    def denoise_latent(
        self,
        latents,
        timesteps,
        text_embeddings,
        pooled_embeddings,
        text_ids,
        latent_image_ids,
        denoiser="transformer",
        guidance=None,
        control_latent=None,
    ):
        do_autocast = self.torch_inference != "" and self.models[denoiser].fp16
        with torch.autocast("cuda", enabled=do_autocast):
            self.profile_start(denoiser, color="blue")

            # handle guidance
            if self.models[denoiser].config["guidance_embeds"] and guidance is None:
                guidance = torch.full(
                    [1], self.guidance_scale, device=self.device, dtype=torch.float32
                )
                guidance = guidance.expand(latents.shape[0])

            for step_index, timestep in enumerate(timesteps):
                # Prepare latents
                latents_input = latents if control_latent is None else torch.cat((latents, control_latent), dim=-1)

                # prepare inputs
                timestep_inp = timestep.expand(latents.shape[0]).to(latents_input.dtype)

                params = {
                    "hidden_states": latents_input,
                    "timestep": timestep_inp / 1000,
                    "pooled_projections": pooled_embeddings,
                    "encoder_hidden_states": text_embeddings,
                    "txt_ids": text_ids.float(),
                    "img_ids": latent_image_ids.float(),
                }
                if guidance is not None:
                    params.update({"guidance": guidance})

                # Predict the noise residual
                if self.torch_inference or self.torch_fallback[denoiser]:
                    noise_pred = self.torch_models[denoiser](**params)["sample"]
                else:
                    noise_pred = self.run_engine(denoiser, params)["latent"]

                latents = self.scheduler.step(
                    noise_pred, timestep, latents, return_dict=False
                )[0]

        self.profile_stop(denoiser)
        return latents.to(dtype=torch.bfloat16) if self.bf16 else latents.to(dtype=torch.float32)

    def decode_latent(self, latents, decoder="vae"):
        self.profile_start(decoder, color="red")
        cast_to = torch.float16 if self.models[decoder].fp16 else torch.bfloat16 if self.models[decoder].bf16 else torch.float32
        latents = latents.to(dtype=cast_to)
        if self.torch_inference or self.torch_fallback[decoder]:
            images = self.torch_models[decoder](latents, return_dict=False)[0]
        else:
            images = self.run_engine(decoder, {"latent": latents})["images"]
        self.profile_stop(decoder)
        return images

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print("|-----------------|--------------|")
        print("| {:^15} | {:^12} |".format("Module", "Latency"))
        print("|-----------------|--------------|")
        print(
            "| {:^15} | {:>9.2f} ms |".format(
                "CLIP",
                cudart.cudaEventElapsedTime(
                    self.events["clip"][0], self.events["clip"][1]
                )[1],
            )
        )
        print(
            "| {:^15} | {:>9.2f} ms |".format(
                "T5",
                cudart.cudaEventElapsedTime(self.events["t5"][0], self.events["t5"][1])[
                    1
                ],
            )
        )
        if "vae_encoder" in self.stages:
            print(
                "| {:^15} | {:>9.2f} ms |".format(
                    "VAE-Enc",
                    cudart.cudaEventElapsedTime(
                        self.events["vae_encoder"][0], self.events["vae_encoder"][1]
                    )[1],
                )
            )
        print(
            "| {:^15} | {:>9.2f} ms |".format(
                "Transformer x " + str(denoising_steps),
                cudart.cudaEventElapsedTime(
                    self.events["transformer"][0], self.events["transformer"][1]
                )[1],
            )
        )
        print(
            "| {:^15} | {:>9.2f} ms |".format(
                "VAE-Dec",
                cudart.cudaEventElapsedTime(
                    self.events["vae"][0], self.events["vae"][1]
                )[1],
            )
        )
        print("|-----------------|--------------|")
        print("| {:^15} | {:>9.2f} ms |".format("Pipeline", walltime_ms))
        print("|-----------------|--------------|")
        print("Throughput: {:.5f} image/s".format(batch_size * 1000.0 / walltime_ms))

    def _check_integrity(self, images):
        integrity_checker = PixtralContentFilter(self.device)
        for image in images:
            image_ = np.array(image) / 255.0
            image_ = 2 * image_ - 1
            image_ = torch.from_numpy(image_).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
            if integrity_checker.test_image(image_):
                raise ValueError("Your image has been flagged. Choose another prompt/image or try again.")

    def save_images(self, prompt, images, check_integrity=False):
        if check_integrity:
            self._check_integrity(images)
        for image in images:
            self.save_image(image, self.pipeline_type.name.lower(), prompt, self.seed)

    def infer(
        self,
        prompt,
        prompt2,
        image_height,
        image_width,
        input_image=None,
        image_strength=1.0,
        control_image=None,
        warmup=False,
        save_image=True,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            prompt2 (str):
                The prompt to be sent to the T5 tokenizer and text encoder
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            input_image (PIL.Image.Image):
                `Image` representing an image batch to be used as the starting point.
            image_strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            control_image (PIL.Image.Image):
                The ControlNet input condition to provide guidance to the `transformer` for generation.
            warmup (bool):
                Indicate if this is a warmup run.
            save_image (bool):
                Save the generated image (if applicable)
        """
        assert len(prompt) == len(prompt2)
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        latent_height = 2 * (int(image_height) // self.vae_scale_factor)
        latent_width = 2 * (int(image_width) // self.vae_scale_factor)

        num_inference_steps = self.denoising_steps
        latent_kwargs = {}

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            num_channels_latents = self.models["transformer"].config["in_channels"] // 4
            if control_image:
                num_channels_latents = self.models["transformer"].config["in_channels"] // 8

                # Prepare control latents
                control_image = self.prepare_image(
                    image=control_image,
                    width=image_width,
                    height=image_height,
                    batch_size=batch_size,
                    num_images_per_prompt=1,
                    device=self.device,
                    dtype=torch.float16 if self.models["vae"].fp16 else torch.bfloat16 if self.models["vae"].bf16 else torch.float32,
                )

                if control_image.ndim == 4:
                    with self.model_memory_manager(["vae_encoder"], low_vram=self.low_vram):
                        control_image = self.encode_image(control_image)

                    height_control_image, width_control_image = control_image.shape[2:]
                    control_image = self._pack_latents(
                        control_image,
                        batch_size,
                        num_channels_latents,
                        height_control_image,
                        width_control_image,
                    )

            # CLIP and T5 text encoder(s)
            with self.model_memory_manager(["clip", "t5"], low_vram=self.low_vram):
                pooled_embeddings = self.encode_prompt(prompt, pooled_output=True)
                text_embeddings = self.encode_prompt(
                    prompt2, encoder="t5", max_sequence_length=self.max_sequence_length
                )
                text_ids = torch.zeros(text_embeddings.shape[1], 3).to(
                    device=self.device, dtype=text_embeddings.dtype
                )

            # Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = (latent_height // 2) * (latent_width // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            timesteps = None
            # TODO: support custom timesteps
            if timesteps is not None:
                if (
                    "timesteps"
                    not in inspect.signature(self.scheduler.set_timesteps).parameters
                ):
                    raise ValueError(
                        f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                        f" timestep schedules. Please check whether you are using the correct scheduler."
                    )
                self.scheduler.set_timesteps(timesteps=timesteps, device=self.device)
                assert self.denoising_steps == len(self.scheduler.timesteps)
            else:
                self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)
            timesteps = self.scheduler.timesteps.to(self.device)
            num_inference_steps = len(timesteps)

            # Pre-process input image and timestep for the img2img pipeline
            if input_image:
                input_image = self.image_processor.preprocess(input_image, height=image_height, width=image_width).to(
                    self.device
                )
                with self.model_memory_manager(["vae_encoder"], low_vram=self.low_vram):
                    image_latents = self.encode_image(input_image)

                timesteps, num_inference_steps = self.get_timesteps(self.denoising_steps, image_strength)
                if num_inference_steps < 1:
                    raise ValueError(
                        f"After adjusting the num_inference_steps by strength parameter: {image_strength}, the number of pipeline"
                        f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
                    )
                latent_timestep = timesteps[:1].repeat(batch_size)

                latent_kwargs.update({"image_latents": image_latents, "latent_timestep": latent_timestep})

            # Initialize latents
            latents, latent_image_ids = self.initialize_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                latent_height=latent_height,
                latent_width=latent_width,
                latents_dtype=torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32,
                **latent_kwargs,
            )

            # DiT denoiser
            with self.model_memory_manager(["transformer"], low_vram=self.low_vram):
                latents = self.denoise_latent(
                    latents,
                    timesteps,
                    text_embeddings,
                    pooled_embeddings,
                    text_ids,
                    latent_image_ids,
                    control_latent=control_image,
                )

            # VAE decode latent
            with self.model_memory_manager(["vae"], low_vram=self.low_vram):
                latents = self._unpack_latents(
                    latents, image_height, image_width, self.vae_scale_factor
                )
                latents = (
                    latents / self.models["vae"].config["scaling_factor"]
                ) + self.models["vae"].config["shift_factor"]
                images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.0
        if not warmup:
            self.print_summary(num_inference_steps, walltime_ms, batch_size)
            if not self.return_latents and save_image:
                # post-process images
                images = (
                    ((images + 1) * 255 / 2)
                    .clamp(0, 255)
                    .detach()
                    .permute(0, 2, 3, 1)
                    .round()
                    .type(torch.uint8)
                    .cpu()
                    .numpy()
                )

        return (latents, walltime_ms) if self.return_latents else (images, walltime_ms)

    def run(
        self,
        prompt,
        prompt2,
        height,
        width,
        batch_count,
        num_warmup_runs,
        use_cuda_graph,
        **kwargs,
    ):
        if self.low_vram and self.use_cuda_graph:
            print("[W] Using low_vram, use_cuda_graph will be disabled")
            self.use_cuda_graph = False
        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                self.infer(prompt, prompt2, height, width, warmup=True, **kwargs)

        images = []
        for _ in range(batch_count):
            print("[I] Running Flux pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            image, _ = self.infer(prompt, prompt2, height, width, warmup=False, **kwargs)
            images.append(image)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()
        return images


class FluxKontextPipeline(FluxPipeline):
    """
    Application showcasing the acceleration of Flux Kontext pipelines using Nvidia TensorRT.
    This pipeline handles the specific logic for the Kontext model variant.
    """

    def __init__(
        self,
        kontext_image,
        version="flux.1-kontext-dev",
        pipeline_type=PIPELINE_TYPE.IMG2IMG,
        guidance_scale=3.5,
        max_sequence_length=512,
        **kwargs,
    ):
        """
        Initializes the Flux Kontext pipeline.

        Args:
            kontext_image (`PIL.Image.Image`):
                Kontext Image to be edited.
            version (`str`, defaults to `flux.1-kontext-dev`)
                Version of the underlying Flux Kontext model.
            guidance_scale (`float`, defaults to 3.5):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            max_sequence_length (`int`, defaults to 512):
                Maximum sequence length to use with the `prompt`.
        """
        super().__init__(
            version=version,
            pipeline_type=pipeline_type,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )
        self.kontext_image = kontext_image
        # WAR to avoid RuntimeError: ScalarType ComplexDouble is an unexpected tensor scalar type during CLIP export
        self.config["clip_torch_fallback"] = True

    @classmethod
    def FromArgs(cls, args: argparse.Namespace, pipeline_type: PIPELINE_TYPE) -> FluxKontextPipeline:
        """Factory method to construct a `FluxKontextPipeline` object from parsed arguments.

        Overrides:
            FluxPipeline.FromArgs
        """
        MAX_BATCH_SIZE = 4
        DEVICE = "cuda"
        DO_RETURN_LATENTS = False

        # Resolve all paths.
        dd_path = path_module.resolve_path(
            cls.get_model_names(pipeline_type), args, pipeline_type, cls._get_pipeline_uid(args.version)
        )

        return cls(
            dd_path=dd_path,
            version=args.version,
            pipeline_type=pipeline_type,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            bf16=args.bf16,
            low_vram=args.low_vram,
            torch_fallback=args.torch_fallback,
            weight_streaming=args.ws,
            t5_weight_streaming_budget_percentage=args.t5_ws_percentage,
            transformer_weight_streaming_budget_percentage=args.transformer_ws_percentage,
            max_batch_size=MAX_BATCH_SIZE,
            denoising_steps=args.denoising_steps,
            scheduler=args.scheduler,
            lora_scale=args.lora_scale,
            lora_weight=args.lora_weight,
            lora_path=args.lora_path,
            kontext_image=args.kontext_image if hasattr(args, "kontext_image") else None,
            device=DEVICE,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            verbose=args.verbose,
            nvtx_profile=args.nvtx_profile,
            use_cuda_graph=args.use_cuda_graph,
            framework_model_dir=args.framework_model_dir,
            return_latents=DO_RETURN_LATENTS,
            torch_inference=args.torch_inference,
        )

    def initialize_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        latent_timestep=None,
        image_latents=None,
        latents_dtype=torch.float32,
    ):
        """
        Initialize latents for Kontext pipeline.
        Overrides FluxPipeline.initialize_latents to handle Kontext-specific logic.
        """
        latents_dtype = latents_dtype  # text_embeddings.dtype
        latents_shape = (batch_size, num_channels_latents, latent_height, latent_width)
        latents = torch.randn(
            latents_shape,
            device=self.device,
            dtype=latents_dtype,
            generator=self.generator,
        )

        image_ids = None
        if image_latents is not None:
            image_latents = torch.cat([image_latents], dim=0).to(latents_dtype)
            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(
                image_latent_height, image_latent_width, latents_dtype, self.device
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1

        latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)

        latent_ids = self._prepare_latent_image_ids(latent_height, latent_width, latents_dtype, self.device)

        latent_image_ids = torch.cat([latent_ids, image_ids], dim=0) if image_ids is not None else latent_ids

        return latents, image_latents, latent_image_ids

    def denoise_latent(
        self,
        latents,
        timesteps,
        text_embeddings,
        pooled_embeddings,
        text_ids,
        latent_image_ids,
        image_latents,
        denoiser="transformer",
        guidance=None,
    ):
        """
        Denoise latents for Kontext pipeline.
        Overrides FluxPipeline.denoise_latent to handle Kontext-specific logic.
        """
        do_autocast = self.torch_inference != "" and self.models[denoiser].fp16
        with torch.autocast("cuda", enabled=do_autocast):
            self.profile_start(denoiser, color="blue")

            # handle guidance
            if self.models[denoiser].config["guidance_embeds"] and guidance is None:
                guidance = torch.full([1], self.guidance_scale, device=self.device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])

            for step_index, timestep in enumerate(timesteps):
                # Kontext-specific: concatenate image_latents along dim=1
                latents_input = torch.cat([latents, image_latents], dim=1)

                # prepare inputs
                timestep_inp = timestep.expand(latents.shape[0]).to(latents_input.dtype)

                params = {
                    "hidden_states": latents_input,
                    "timestep": timestep_inp / 1000,
                    "pooled_projections": pooled_embeddings,
                    "encoder_hidden_states": text_embeddings,
                    "txt_ids": text_ids.float(),
                    "img_ids": latent_image_ids.float(),
                }
                if guidance is not None:
                    params.update({"guidance": guidance})

                # Predict the noise residual
                if self.torch_inference or self.torch_fallback[denoiser]:
                    noise_pred = self.torch_models[denoiser](**params)["sample"]
                else:
                    noise_pred = self.run_engine(denoiser, params)["latent"]

                # Kontext-specific: extract only the latent part of the prediction
                noise_pred = noise_pred[:, : latents.size(1)]

                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        self.profile_stop(denoiser)
        return latents.to(dtype=torch.bfloat16) if self.bf16 else latents.to(dtype=torch.float32)

    def _get_resize_dimensions(self, input_image):
        """
        Preprocess input image for Kontext pipeline using preferred resolutions.
        Uses FluxKontextUtil to get the standardized list of preferred resolutions.
        """
        multiple_of = self.vae_scale_factor * 2
        resize_height, resize_width = self.image_processor.get_default_height_width(input_image)
        aspect_ratio = resize_width / resize_height
        # Kontext is trained on specific resolutions, using one of them is recommended
        preferred_resolutions = FluxKontextUtil.get_preferred_resolutions()
        _, resize_width, resize_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in preferred_resolutions
        )
        resize_width = resize_width // multiple_of * multiple_of
        resize_height = resize_height // multiple_of * multiple_of

        return resize_height, resize_width

    def infer(
        self,
        prompt,
        prompt2,
        image_height,
        image_width,
        image_strength=1.0,
        warmup=False,
        save_image=True,
    ):
        """
        Run the Kontext diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            prompt2 (str):
                The prompt to be sent to the T5 tokenizer and text encoder
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            image_strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            warmup (bool):
                Indicate if this is a warmup run.
            save_image (bool):
                Save the generated image (if applicable)
        """
        assert len(prompt) == len(prompt2)
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        latent_height = 2 * (int(image_height) // self.vae_scale_factor)
        latent_width = 2 * (int(image_width) // self.vae_scale_factor)

        num_inference_steps = self.denoising_steps
        latent_kwargs = {}

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            num_channels_latents = self.models["transformer"].config["in_channels"] // 4

            # CLIP and T5 text encoder(s)
            with self.model_memory_manager(["clip", "t5"], low_vram=self.low_vram):
                pooled_embeddings = self.encode_prompt(prompt, pooled_output=True)
                text_embeddings = self.encode_prompt(
                    prompt2, encoder="t5", max_sequence_length=self.max_sequence_length
                )
                text_ids = torch.zeros(text_embeddings.shape[1], 3).to(device=self.device, dtype=text_embeddings.dtype)

            # Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = (latent_height // 2) * (latent_width // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            timesteps = None
            # TODO: support custom timesteps
            if timesteps is not None:
                if "timesteps" not in inspect.signature(self.scheduler.set_timesteps).parameters:
                    raise ValueError(
                        f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                        f" timestep schedules. Please check whether you are using the correct scheduler."
                    )
                self.scheduler.set_timesteps(timesteps=timesteps, device=self.device)
                assert self.denoising_steps == len(self.scheduler.timesteps)
            else:
                self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)
            timesteps = self.scheduler.timesteps.to(self.device)
            num_inference_steps = len(timesteps)

            # Pre-process kontext image and timestep for the img2img pipeline
            if self.kontext_image:
                # Kontext-specific image preprocessing
                kontext_image = self.image_processor.resize(self.kontext_image, self.resize_height, self.resize_width)

                kontext_image = self.image_processor.preprocess(
                    kontext_image, height=self.resize_height, width=self.resize_width
                ).to(self.device)
                with self.model_memory_manager(["vae_encoder"], low_vram=self.low_vram):
                    image_latents = self.encode_image(kontext_image)

                timesteps, num_inference_steps = self.get_timesteps(self.denoising_steps, image_strength)
                if num_inference_steps < 1:
                    raise ValueError(
                        f"After adjusting the num_inference_steps by strength parameter: {image_strength}, the number of pipeline"
                        f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
                    )
                latent_timestep = timesteps[:1].repeat(batch_size)

                latent_kwargs.update({"image_latents": image_latents, "latent_timestep": latent_timestep})

            # Initialize latents
            latents, image_latents, latent_ids = self.initialize_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                latent_height=latent_height,
                latent_width=latent_width,
                latents_dtype=torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32,
                **latent_kwargs,
            )

            # DiT denoiser
            with self.model_memory_manager(["transformer"], low_vram=self.low_vram):
                latents = self.denoise_latent(
                    latents,
                    timesteps,
                    text_embeddings,
                    pooled_embeddings,
                    text_ids,
                    latent_ids,
                    image_latents,
                )

            # VAE decode latent
            with self.model_memory_manager(["vae"], low_vram=self.low_vram):
                latents = self._unpack_latents(latents, image_height, image_width, self.vae_scale_factor)
                latents = (latents / self.models["vae"].config["scaling_factor"]) + self.models["vae"].config[
                    "shift_factor"
                ]
                images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.0
        if not warmup:
            self.print_summary(num_inference_steps, walltime_ms, batch_size)
            if not self.return_latents and save_image:
                # post-process images
                images = (
                    ((images + 1) * 255 / 2)
                    .clamp(0, 255)
                    .detach()
                    .permute(0, 2, 3, 1)
                    .round()
                    .type(torch.uint8)
                    .cpu()
                    .numpy()
                )

        return (latents, walltime_ms) if self.return_latents else (images, walltime_ms)
