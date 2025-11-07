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
import random
import time
import warnings
from typing import Any, List

import numpy as np
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from diffusers.video_processor import VideoProcessor
from flux.content_filters import PixtralContentFilter
from tqdm import tqdm

from demo_diffusion import path as path_module
from demo_diffusion.model import (
    AutoencoderKLWanEncoderModel,
    AutoencoderKLWanModel,
    CosmosTransformerModel,
    T5Model,
    make_tokenizer,
)
from demo_diffusion.pipeline.diffusion_pipeline import DiffusionPipeline
from demo_diffusion.pipeline.type import PIPELINE_TYPE

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class CosmosPipeline(DiffusionPipeline):
    """
    Application showcasing the acceleration of Cosmos pipelines using Nvidia TensorRT.
    """

    def __init__(
        self,
        version="cosmos-predict2-2b",
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        guidance_scale=6.0,
        max_sequence_length=512,
        t5_weight_streaming_budget_percentage=None,
        transformer_weight_streaming_budget_percentage=None,
        **kwargs,
    ):
        """
        Initializes the Cosmos pipeline.

        Args:
            version (`str`, defaults to `cosmos-1.0-7B`)
                Version of the underlying Cosmos model.
            guidance_scale (`float`, defaults to 3.5):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            max_sequence_length (`int`, defaults to 512):
                Maximum sequence length to use with the `prompt`.
            t5_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the T5 model.
            transformer_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the Transformer model.
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
        self.do_classifier_free_guidance = self.guidance_scale > 1

        # WAR ONNX export error: Exporting the operator 'aten::_upsample_nearest_exact2d' to ONNX opset version 19 is not supported
        self.config["vae_torch_fallback"] = True
        self.config["vae_encoder_torch_fallback"] = True

    @classmethod
    def FromArgs(cls, args: argparse.Namespace, pipeline_type: PIPELINE_TYPE) -> CosmosPipeline:
        """Factory method to construct a `CosmosPipeline` object from parsed arguments.

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
            low_vram=args.low_vram,
            torch_fallback=args.torch_fallback,
            weight_streaming=args.ws,
            t5_weight_streaming_budget_percentage=args.t5_ws_percentage,
            transformer_weight_streaming_budget_percentage=args.transformer_ws_percentage,
            max_batch_size=MAX_BATCH_SIZE,
            denoising_steps=args.denoising_steps,
            scheduler=args.scheduler,
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
        if pipeline_type.is_video2world():
            return ["vae_encoder", "t5", "transformer", "vae"]
        return ["t5", "transformer", "vae"]

    def download_onnx_models(self, model_name: str, model_config: dict[str, Any]) -> None:
        raise ValueError("ONNX models download is not supported for the Cosmos Pipeline")

    def _initialize_models(self, framework_model_dir, int8, fp8, fp4):
        # Load text tokenizer(s)
        self.tokenizer = make_tokenizer(
            self.version, self.pipeline_type, self.hf_token, framework_model_dir, tokenizer_type="t5"
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

        self.fp16 = True if not self.bf16 else False
        self.tf32 = True
        if "t5" in self.stages:
            # Known accuracy issues with FP16
            self.models["t5"] = T5Model(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                text_maxlen=self.max_sequence_length,
                build_strongly_typed=True,
                use_attention_mask=True,
            )

        if "transformer" in self.stages:
            self.models["transformer"] = CosmosTransformerModel(
                **models_args,
                bf16=self.bf16,
                fp16=self.fp16,
                int8=int8,
                fp8=fp8,
                tf32=self.tf32,
                text_maxlen=self.max_sequence_length,
                build_strongly_typed=True,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.denoiser_weight_streaming_budget_percentage,
            )

        if "vae" in self.stages:
            self.models["vae"] = AutoencoderKLWanModel(**models_args, fp16=False, tf32=self.tf32, bf16=self.bf16)

        if "vae_encoder" in self.stages:
            self.models["vae_encoder"] = AutoencoderKLWanEncoderModel(
                **models_args, fp16=False, tf32=self.tf32, bf16=self.bf16
            )

        self.vae_scale_factor_temporal = (
            2 ** sum(self.models["vae"].config["temperal_downsample"])
            if "vae" in self.stages and self.models["vae"] is not None
            else 4
        )
        self.vae_scale_factor_spatial = (
            2 ** len(self.models["vae"].config["temperal_downsample"])
            if "vae" in self.stages and self.models["vae"] is not None
            else 8
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def encode_video(self, video):
        self.profile_start("vae_encoder", color="red")
        cast_to = (
            torch.float16
            if self.models["vae_encoder"].fp16
            else torch.bfloat16 if self.models["vae_encoder"].bf16 else torch.float32
        )
        video = video.to(dtype=cast_to)
        if self.torch_inference:
            image_latents = self.torch_models["vae_encoder"](video)
        else:
            image_latents = self.run_engine("vae_encoder", {"images": video})["latent"]
        self.profile_stop("vae_encoder")
        return image_latents

    def initialize_latents_text2image(
        self,
        batch_size,
        num_channels_latents,
        num_latent_frames,
        latent_height,
        latent_width,
        latents_dtype=torch.float32,
    ):
        latents_shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = torch.randn(
            latents_shape,
            device=self.device,
            dtype=latents_dtype,
            generator=self.generator,
        )

        return latents * self.scheduler.config.sigma_max

    def initialize_latents_video2world(
        self,
        video,
        batch_size,
        num_channels_latents,
        num_frames,
        latent_height,
        latent_width,
        latents_dtype=torch.float32,
        do_classifier_free_guidance=False,
    ):
        num_cond_frames = video.size(2)
        if num_cond_frames >= num_frames:
            # Take the last `num_frames` frames for conditioning
            num_cond_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            video = video[:, :, -num_frames:]
        else:
            num_cond_latent_frames = (num_cond_frames - 1) // self.vae_scale_factor_temporal + 1
            num_padding_frames = num_frames - num_cond_frames
            last_frame = video[:, :, -1:]
            padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1)
            video = torch.cat([video, padding], dim=2)

        # Encode video
        with self.model_memory_manager(["vae_encoder"], low_vram=self.low_vram):
            video_latents = self.encode_video(
                video=video,
            )

        latents_mean = (
            torch.tensor(self.models["vae"].config["latents_mean"])
            .view(1, self.models["vae"].config["z_dim"], 1, 1, 1)
            .to(self.device, latents_dtype)
        )
        latents_std = (
            torch.tensor(self.models["vae"].config["latents_std"])
            .view(1, self.models["vae"].config["z_dim"], 1, 1, 1)
            .to(self.device, latents_dtype)
        )
        init_latents = (video_latents - latents_mean) / latents_std * self.scheduler.config.sigma_data

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        latents = torch.randn(
            shape,
            device=self.device,
            dtype=latents_dtype,
            generator=self.generator,
        )

        latents = latents * self.scheduler.config.sigma_max

        padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        uncond_indicator = uncond_mask = None
        if do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding

        return latents, init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask

    # Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_img2img.py#L416C1
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def _duplicate_text_embeddings(self, batch_size, text_embeddings, num_outputs_per_prompt):
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_outputs_per_prompt, 1)
        text_embeddings = text_embeddings.view(batch_size * num_outputs_per_prompt, seq_len, -1)
        return text_embeddings

    def _prepare_timesteps(self, num_inference_steps):
        """Prepare timesteps for the scheduler."""
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        accept_sigmas = "sigmas" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
        if self.scheduler.config.get("final_sigmas_type", "zero") == "sigma_min":
            # Replace the last sigma (which is zero) with the minimum sigma value
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]
        return timesteps, num_inference_steps

    def _encode_text_prompts(self, prompt, negative_prompt, batch_size, num_outputs_per_prompt):
        """Encode text prompts using T5 encoder."""
        with self.model_memory_manager(["t5"], low_vram=self.low_vram):
            text_embeddings = self.encode_prompt(prompt)
            text_embeddings = self._duplicate_text_embeddings(batch_size, text_embeddings, num_outputs_per_prompt)
            negative_text_embeddings = None
            if self.do_classifier_free_guidance:
                negative_text_embeddings = self.encode_prompt(negative_prompt)
                negative_text_embeddings = self._duplicate_text_embeddings(
                    batch_size, negative_text_embeddings, num_outputs_per_prompt
                )
        return text_embeddings, negative_text_embeddings

    def _get_latents_normalization_params(self, device, dtype):
        """Get latents normalization parameters from VAE config."""
        latents_mean = (
            torch.tensor(self.models["vae"].config["latents_mean"])
            .view(1, self.models["vae"].config["z_dim"], 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = (
            torch.tensor(self.models["vae"].config["latents_std"])
            .view(1, self.models["vae"].config["z_dim"], 1, 1, 1)
            .to(device, dtype)
        )
        return latents_mean, latents_std

    def _normalize_and_decode_latents(self, latents, is_video2world=False):
        """Normalize latents and decode using VAE."""
        latents_mean, latents_std = self._get_latents_normalization_params(latents.device, latents.dtype)

        if is_video2world:
            # For video2world: latents * std / sigma_data + mean
            latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean
        else:
            # For text2image: latents / (1/std) / sigma_data + mean
            latents_std_inv = 1.0 / latents_std
            latents = latents / latents_std_inv / self.scheduler.config.sigma_data + latents_mean

        with self.model_memory_manager(["vae"], low_vram=self.low_vram):
            video = self.decode_latent(latents)

        return video

    def encode_prompt(self, prompt, encoder="t5"):
        self.profile_start(encoder, color="green")

        def tokenize(prompt):
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.bool().to(self.device)

            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(self.device)
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.max_sequence_length - 1 : -1])
                warnings.warn(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f"{self.max_sequence_length} tokens: {removed_text}"
                )

            if self.torch_inference or self.torch_fallback[encoder]:
                text_encoder_output = self.torch_models[encoder](
                    text_input_ids, attention_mask=attention_mask
                ).last_hidden_state
            else:
                # NOTE: output tensor for the encoder must be cloned because it will be overwritten when called again for prompt2
                text_encoder_output = self.run_engine(
                    encoder, {"input_ids": text_input_ids, "attention_mask": attention_mask}
                )["text_embeddings"]

            lengths = attention_mask.sum(dim=1).cpu()
            for i, length in enumerate(lengths):
                text_encoder_output[i, length:] = 0
            return text_encoder_output

        # Tokenize prompt
        text_encoder_output = tokenize(prompt)

        self.profile_stop(encoder)
        return (
            text_encoder_output.to(torch.float16)
            if self.fp16
            else text_encoder_output.to(torch.bfloat16) if self.bf16 else text_encoder_output.to(torch.float32)
        )

    def denoise_latent(
        self,
        latents,
        timesteps,
        text_embeddings,
        negative_text_embeddings,
        padding_mask,
        denoiser="transformer",
    ):
        do_autocast = self.torch_inference != "" and self.models[denoiser].fp16
        with torch.autocast("cuda", enabled=do_autocast, dtype=torch.float32):
            self.profile_start(denoiser, color="blue")

            for step_index, timestep in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising"):
                # Prepare latents
                cast_to = (
                    torch.float16
                    if self.models[denoiser].fp16
                    else torch.bfloat16 if self.models[denoiser].bf16 else torch.float32
                )
                current_sigma = self.scheduler.sigmas[step_index]
                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep_inp = current_t.expand(latents.shape[0]).to(cast_to)  # [B, 1, T, 1, 1]
                latents_input = (latents * c_in).to(cast_to)
                # prepare inputs
                params = {
                    "hidden_states": latents_input,
                    "timestep": timestep_inp,
                    "encoder_hidden_states": text_embeddings,
                    "padding_mask": padding_mask,
                }

                if self.torch_inference or self.torch_fallback[denoiser]:
                    noise_pred = self.torch_models[denoiser](**params)["sample"]
                else:
                    noise_pred = self.run_engine(denoiser, params)["latent"].clone()

                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(cast_to)
                if self.do_classifier_free_guidance:
                    params = {
                        "hidden_states": latents_input,
                        "timestep": timestep_inp,
                        "encoder_hidden_states": negative_text_embeddings,
                        "padding_mask": padding_mask,
                    }

                    # Predict the noise residual
                    if self.torch_inference or self.torch_fallback[denoiser]:
                        noise_pred_uncond = self.torch_models[denoiser](**params)["sample"]
                    else:
                        noise_pred_uncond = self.run_engine(denoiser, params)["latent"].clone()

                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(cast_to)
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                noise_pred = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        self.profile_stop(denoiser)
        return latents.to(dtype=torch.bfloat16) if self.bf16 else latents.to(dtype=torch.float32)

    def denoise_latent_video2world(
        self,
        latents,
        timesteps,
        text_embeddings,
        negative_text_embeddings,
        padding_mask,
        fps,
        cond_mask,
        uncond_mask,
        t_conditioning,
        cond_indicator,
        conditioning_latents,
        uncond_indicator,
        unconditioning_latents,
        denoiser="transformer",
    ):
        do_autocast = self.torch_inference != "" and self.models[denoiser].fp16
        with torch.autocast("cuda", enabled=do_autocast, dtype=torch.float32):
            self.profile_start(denoiser, color="blue")

            for step_index, timestep in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising"):
                # Prepare latents
                cast_to = (
                    torch.float16
                    if self.models[denoiser].fp16
                    else torch.bfloat16 if self.models[denoiser].bf16 else torch.float32
                )
                current_sigma = self.scheduler.sigmas[step_index]
                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep_inp = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]
                latents_input = latents * c_in
                latents_input = (cond_indicator * conditioning_latents + (1 - cond_indicator) * latents_input).to(
                    cast_to
                )
                timestep_inp = (cond_indicator * t_conditioning + (1 - cond_indicator) * timestep_inp).to(cast_to)

                # prepare inputs
                params = {
                    "hidden_states": latents_input,
                    "timestep": timestep_inp,
                    "encoder_hidden_states": text_embeddings,
                    "padding_mask": padding_mask,
                    "fps": fps,
                    "condition_mask": cond_mask,
                }

                if self.torch_inference or self.torch_fallback[denoiser]:
                    noise_pred = self.torch_models[denoiser](**params)["sample"]
                else:
                    noise_pred = self.run_engine(denoiser, params)["latent"].clone()

                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(cast_to)
                noise_pred = cond_indicator * conditioning_latents + (1 - cond_indicator) * noise_pred
                if self.do_classifier_free_guidance:
                    latents_input = latents * c_in
                    latents_input = (
                        uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * latents_input
                    ).to(cast_to)
                    timestep_inp = (uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep_inp).to(
                        cast_to
                    )
                    params = {
                        "hidden_states": latents_input,
                        "timestep": timestep_inp,
                        "encoder_hidden_states": negative_text_embeddings,
                        "padding_mask": padding_mask,
                        "fps": fps,
                        "condition_mask": uncond_mask,
                    }

                    # Predict the noise residual
                    if self.torch_inference or self.torch_fallback[denoiser]:
                        noise_pred_uncond = self.torch_models[denoiser](**params)["sample"]
                    else:
                        noise_pred_uncond = self.run_engine(denoiser, params)["latent"].clone()

                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(cast_to)
                    noise_pred_uncond = (
                        uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                    )
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                noise_pred = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        self.profile_stop(denoiser)
        return latents.to(dtype=torch.bfloat16) if self.bf16 else latents.to(dtype=torch.float32)

    def decode_latent(self, latents, decoder="vae"):
        self.profile_start(decoder, color="red")
        cast_to = (
            torch.float16
            if self.models[decoder].fp16
            else torch.bfloat16 if self.models[decoder].bf16 else torch.float32
        )
        latents = latents.to(dtype=cast_to)

        if self.torch_inference or self.torch_fallback[decoder]:
            video = self.torch_models[decoder](latents, return_dict=False)[0]
        else:
            video = self.run_engine(decoder, {"latent": latents})["frames"]

        self.profile_stop(decoder)
        return video

    def post_process_video(self, video):
        # Post-process video
        video = self.video_processor.postprocess_video(video, output_type="np")
        video = (video * 255).astype(np.uint8)
        video_batch = []
        for vid in video:
            # vid = self.safety_checker.check_video_safety(vid)
            video_batch.append(vid)
        video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
        video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
        video = self.video_processor.postprocess_video(video, output_type="pil")

        if self.pipeline_type.is_video2world():
            return video

        image = [batch[0] for batch in video]
        if isinstance(video, torch.Tensor):
            image = torch.stack(image)
        elif isinstance(video, np.ndarray):
            image = np.stack(image)

        return image

    def _finalize_generation(self, video, walltime_ms, num_inference_steps, batch_size, warmup, save_output):
        """Handle post-processing, saving, and performance reporting."""
        if not warmup:
            self.print_summary(num_inference_steps, walltime_ms, batch_size)
            if not self.return_latents and save_output:
                # post-process video
                processed_output = self.post_process_video(video)

                # save output
                if self.pipeline_type.is_video2world():
                    return (processed_output[0], walltime_ms)
                return (np.array(processed_output), walltime_ms)

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

    def save_video(
        self,
        prompt,
        videos,
        check_integrity=False,
    ):
        for frames in videos:
            if check_integrity:
                self._check_integrity([frames])
            prompt_prefix = "".join(set([prompt[i].replace(" ", "_")[:10] for i in range(len(prompt))]))
            video_name_prefix = "-".join(
                [self.pipeline_type.name.lower(), "fp16", str(self.seed), str(random.randint(1000, 9999))]
            )
            video_name_suffix = "torch" if self.torch_inference else "trt"
            video_path = prompt_prefix + "-" + video_name_prefix + "-" + video_name_suffix + ".gif"
            print(f"Saving video to: {video_path}")
            frames[0].save(
                os.path.join(self.output_dir, video_path),
                save_all=True,
                optimize=False,
                append_images=frames[1:],
                loop=0,
            )

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print("|-----------------|--------------|")
        print("| {:^15} | {:^12} |".format("Module", "Latency"))
        print("|-----------------|--------------|")
        for stage in self.stages:
            print(
                "| {:^15} | {:>9.2f} ms |".format(
                    stage + " x " + str(denoising_steps) if stage == "transformer" else stage,
                    cudart.cudaEventElapsedTime(self.events[stage][0], self.events[stage][1])[1],
                )
            )
        print("|-----------------|--------------|")
        print("| {:^15} | {:>9.2f} ms |".format("Pipeline", walltime_ms))
        print("|-----------------|--------------|")
        print("Throughput: {:.2f} image/s".format(batch_size * 1000.0 / walltime_ms))

    def generate_image(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        num_frames=1,
        num_images_per_prompt=1,
        save_image=True,
        warmup=False,
    ):
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = image_height // self.vae_scale_factor_spatial
        latent_width = image_width // self.vae_scale_factor_spatial

        num_inference_steps = self.denoising_steps

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Prepare timesteps
            timesteps, num_inference_steps = self._prepare_timesteps(num_inference_steps)

            # T5 text encoder
            text_embeddings, negative_text_embeddings = self._encode_text_prompts(
                prompt, negative_prompt, batch_size, num_images_per_prompt
            )

            num_channels_latents = self.models["transformer"].config["in_channels"]
            latents_dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32

            # Initialize latents
            latents = self.initialize_latents_text2image(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                num_latent_frames=num_latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                latents_dtype=latents_dtype,
            )
            padding_mask = latents.new_zeros(1, 1, image_height, image_width, dtype=latents_dtype)

            # denoiser
            with self.model_memory_manager(["transformer"], low_vram=self.low_vram):
                latents = self.denoise_latent(
                    latents,
                    timesteps,
                    text_embeddings,
                    negative_text_embeddings,
                    padding_mask,
                )

            # VAE decode latent
            video = self._normalize_and_decode_latents(latents, is_video2world=False)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.0
        return self._finalize_generation(
            video,
            walltime_ms,
            num_inference_steps,
            batch_size,
            warmup,
            save_image,
        )

    def generate_video(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        input_image=None,
        input_video=None,
        num_frames=1,
        fps=16,
        num_videos_per_prompt=1,
        sigma_conditioning=0.0001,
        save_video=True,
        warmup=False,
    ):
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        latent_height = image_height // self.vae_scale_factor_spatial
        latent_width = image_width // self.vae_scale_factor_spatial

        num_inference_steps = self.denoising_steps

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Prepare timesteps
            timesteps, num_inference_steps = self._prepare_timesteps(num_inference_steps)

            # T5 text encoder
            text_embeddings, negative_text_embeddings = self._encode_text_prompts(
                prompt, negative_prompt, batch_size, num_videos_per_prompt
            )

            num_channels_latents = self.models["transformer"].config["in_channels"] - 1
            latents_dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32

            # Process input conditioning
            if input_image is not None:
                video = (
                    self.video_processor.preprocess(input_image, image_height, image_width)
                    .unsqueeze(2)
                    .to(device=self.device, dtype=latents_dtype)
                )
            elif input_video is not None:
                video = self.video_processor.preprocess_video(input_video, image_height, image_width).to(
                    device=self.device, dtype=latents_dtype
                )
            else:
                raise ValueError("Video2world pipeline requires either input_image or input_video to be provided")

            # Initialize latents
            latents, conditioning_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask = (
                self.initialize_latents_video2world(
                    video,
                    batch_size=batch_size,
                    num_channels_latents=num_channels_latents,
                    num_frames=num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    latents_dtype=latents_dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )
            )
            unconditioning_latents = None

            cond_mask = cond_mask.to(latents_dtype)
            if self.do_classifier_free_guidance:
                uncond_mask = uncond_mask.to(latents_dtype)
                unconditioning_latents = conditioning_latents

            padding_mask = latents.new_zeros(1, 1, image_height, image_width, dtype=latents_dtype)
            sigma_conditioning = torch.tensor(sigma_conditioning, dtype=torch.float32, device=self.device)
            t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

            # denoiser
            with self.model_memory_manager(["transformer"], low_vram=self.low_vram):
                latents = self.denoise_latent_video2world(
                    latents,
                    timesteps,
                    text_embeddings,
                    negative_text_embeddings,
                    padding_mask,
                    fps,
                    cond_mask,
                    uncond_mask,
                    t_conditioning,
                    cond_indicator,
                    conditioning_latents,
                    uncond_indicator,
                    unconditioning_latents,
                )

            # VAE decode latent
            video = self._normalize_and_decode_latents(latents, is_video2world=True)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.0
        return self._finalize_generation(
            video,
            walltime_ms,
            num_inference_steps,
            batch_size,
            warmup,
            save_video,
        )

    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        input_image=None,
        input_video=None,
        num_frames=1,
        fps=16,
        num_images_per_prompt=1,
        num_videos_per_prompt=1,
        sigma_conditioning=0.0001,
        warmup=False,
        save_output=True,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            input_image (image):
                Input image used to initialize the latents.
            input_video (video):
                Input video used to initialize the latents.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            num_frames (int):
                The number of frames in the generated video.
            fps (int):
                The frames per second of the generated video.
            num_images_per_prompt (int):
                The number of images to generate per prompt.
            num_videos_per_prompt (int):
                The number of videos to generate per prompt.
            sigma_conditioning (`float`, defaults to `0.0001`):
                The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
                set to a small value close to zero.
            warmup (bool):
                Indicate if this is a warmup run.
            save_output (bool):
                Save the generated image or video (if applicable)
        """
        if self.pipeline_type.is_txt2img():
            return self.generate_image(
                prompt,
                negative_prompt,
                image_height,
                image_width,
                num_frames,
                num_images_per_prompt,
                save_output,
                warmup,
            )
        elif self.pipeline_type.is_video2world():
            return self.generate_video(
                prompt,
                negative_prompt,
                image_height,
                image_width,
                input_image,
                input_video,
                num_frames,
                fps,
                num_videos_per_prompt,
                sigma_conditioning,
                save_output,
                warmup,
            )
        else:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")

    def run(
        self,
        prompt,
        negative_prompt,
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
                self.infer(prompt, negative_prompt, height, width, warmup=True, **kwargs)

        outputs = []
        for _ in range(batch_count):
            print("[I] Running Cosmos pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            output, _ = self.infer(prompt, negative_prompt, height, width, warmup=False, **kwargs)
            outputs.append(output)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()

        return outputs
