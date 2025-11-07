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
from typing import Any, List, Union

import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerBase

from demo_diffusion import path as path_module
from demo_diffusion.model import (
    CLIPWithProjModel,
    SD3ControlNet,
    SD3TransformerModel,
    T5Model,
    VAEEncoderModel,
    VAEModel,
    load,
    make_tokenizer,
)
from demo_diffusion.pipeline.diffusion_pipeline import DiffusionPipeline
from demo_diffusion.pipeline.type import PIPELINE_TYPE

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)
    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image
    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image

class StableDiffusion35Pipeline(DiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Diffusion 3.5 pipelines using Nvidia TensorRT.
    """

    def __init__(
        self,
        version: str,
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        guidance_scale: float = 7.0,
        max_sequence_length: int = 256,
        controlnet=None,
        **kwargs,
    ):
        """
        Initializes the Stable Diffusion 3.5 pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of ['3.5-medium', '3.5-large']
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline.
            guidance_scale (`float`, defaults to 7.0):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            max_sequence_length (`int`, defaults to 256):
                Maximum sequence length to use with the `prompt`.
            controlnet (str):
                Which ControlNet to use.
        """
        super().__init__(version=version, pipeline_type=pipeline_type, controlnet=controlnet, **kwargs)

        self.fp16 = True if not self.bf16 else False

        self.force_weakly_typed_t5 = False
        self.config["clip_g_torch_fallback"] = True
        self.config["clip_l_torch_fallback"] = True
        self.config["clip_hidden_states"] = True
        self.controlnet = controlnet

        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1
        self.max_sequence_length = max_sequence_length

    @classmethod
    def FromArgs(cls, args: argparse.Namespace, pipeline_type: PIPELINE_TYPE) -> StableDiffusion35Pipeline:
        """Factory method to construct a `StableDiffusion35Pipeline` object from parsed arguments.

        Overrides:
            DiffusionPipeline.FromArgs
        """
        MAX_BATCH_SIZE = 4
        DEVICE = "cuda"
        DO_RETURN_LATENTS = False

        # Resolve all paths.
        controlnet_type = args.controlnet_type if "controlnet_type" in args else None
        dd_path = path_module.resolve_path(
            cls.get_model_names(pipeline_type, controlnet_type),
            args,
            pipeline_type,
            cls._get_pipeline_uid(args.version),
        )

        return cls(
            dd_path=dd_path,
            version=args.version,
            pipeline_type=pipeline_type,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            controlnet=controlnet_type,
            bf16=args.bf16,
            low_vram=args.low_vram,
            torch_fallback=args.torch_fallback,
            weight_streaming=args.ws,
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
        if pipeline_type.is_controlnet():
            assert controlnet_type, "ControlNet type must be specified for ControlNet pipelines"
            return ["clip_l", "clip_g", "t5", "transformer", "vae", "vae_encoder", f"controlnet_{controlnet_type}"]
        return ["clip_l", "clip_g", "t5", "transformer", "vae"]

    def download_onnx_models(self, model_name: str, model_config: dict[str, Any]) -> None:
        if self.fp16:
            raise ValueError(
                "ONNX models can be downloaded only for the following precisions: BF16, FP8. This pipeline is running in FP16."
            )

        hf_download_path = "-".join([load.get_path(self.version, self.pipeline_type.name), "tensorrt"])
        model_path = model_config["onnx_opt_path"]
        base_dir = os.path.dirname(os.path.dirname(model_config["onnx_opt_path"]))

        if not os.path.exists(model_path):
            if model_name == "transformer":
                if model_config["use_fp8"]:
                    dirname = os.path.join(model_name, "fp8")
                elif self.bf16:
                    dirname = os.path.join(model_name, "bf16")
            elif "controlnet" in model_name:
                hf_download_path_cnet = hf_download_path.replace("large", "controlnets")
                dirname = f"controlnet_{self.controlnet}"
                if "blur" in model_name:
                    pass
                elif model_config["use_fp8"]:
                    dirname = os.path.join(dirname, "fp8")
                elif self.bf16:
                    dirname = os.path.join(dirname, "bf16")
            elif model_name in self.stages:
                dirname = model_name
            else:
                raise ValueError(f"{model_name} not found in {self.stages}")

            dirname = os.path.join("ONNX", dirname)
            snapshot_download(
                repo_id=hf_download_path if "controlnet" not in model_name else hf_download_path_cnet,
                allow_patterns=os.path.join(dirname, "*"),
                local_dir=base_dir,
                token=self.hf_token,
            )
            # Rename directory from ONNX/<model_name> to <model_name>
            saved_dir = os.path.join(base_dir, dirname)
            model_dir = os.path.dirname(model_path)
            os.rename(saved_dir, model_dir)

    def load_resources(
        self,
        image_height: int,
        image_width: int,
        batch_size: int,
        seed: int,
    ):
        super().load_resources(image_height, image_width, batch_size, seed)

    def _initialize_models(self, framework_model_dir, int8, fp8, fp4):
        # Load text tokenizer(s)
        self.tokenizer = make_tokenizer(
            self.version,
            self.pipeline_type,
            self.hf_token,
            framework_model_dir,
        )
        self.tokenizer2 = make_tokenizer(
            self.version,
            self.pipeline_type,
            self.hf_token,
            framework_model_dir,
            subfolder="tokenizer_2",
        )
        self.tokenizer3 = make_tokenizer(
            self.version,
            self.pipeline_type,
            self.hf_token,
            framework_model_dir,
            subfolder="tokenizer_3",
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
        self.fp8 = fp8
        self.int8 = int8
        self.fp4 = fp4
        if "clip_l" in self.stages:
            self.models["clip_l"] = CLIPWithProjModel(
                **models_args,
                fp16=self.fp16,
                bf16=self.bf16,
                subfolder="text_encoder",
                output_hidden_states=self.config.get("clip_hidden_states", False),
            )

        if "clip_g" in self.stages:
            self.models["clip_g"] = CLIPWithProjModel(
                **models_args,
                fp16=self.fp16,
                bf16=self.bf16,
                subfolder="text_encoder_2",
                output_hidden_states=self.config.get("clip_hidden_states", False),
            )

        if "t5" in self.stages:
            # Known accuracy issues with FP16
            self.models["t5"] = T5Model(
                **models_args,
                fp16=self.fp16,
                bf16=self.bf16,
                tf32=self.tf32,
                subfolder="text_encoder_3",
                text_maxlen=self.max_sequence_length,
                build_strongly_typed=True,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.text_encoder_weight_streaming_budget_percentage,
            )

        if "transformer" in self.stages:
            self.models["transformer"] = SD3TransformerModel(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                fp8=self.fp8,
                int8=self.int8,
                fp4=self.fp4,
                text_maxlen=self.models["t5"].text_maxlen + self.models["clip_g"].text_maxlen,
                build_strongly_typed=False,
                weight_streaming=self.weight_streaming,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

        if f"controlnet_{self.controlnet}" in self.stages:
            self.models[f"controlnet_{self.controlnet}"] = SD3ControlNet(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                controlnet=self.controlnet,
            )

        if "vae" in self.stages:
            self.models["vae"] = VAEModel(**models_args, fp16=self.fp16, tf32=self.tf32, bf16=self.bf16)

        self.vae_scale_factor = (
            2 ** (len(self.models["vae"].config["block_out_channels"]) - 1) if "vae" in self.models else 8
        )
        self.patch_size = (
            self.models["transformer"].config["patch_size"]
            if "transformer" in self.stages and self.models["transformer"] is not None
            else 2
        )

        if "vae_encoder" in self.stages:
            self.models["vae_encoder"] = VAEEncoderModel(**models_args, fp16=False, tf32=self.tf32, bf16=self.bf16, do_classifier_free_guidance=self.do_classifier_free_guidance)
            self.vae_latent_channels = (
                self.models["vae"].config["latent_channels"]
                if "vae" in self.stages and self.models["vae"] is not None
                else 16
            )
            if self.controlnet and "canny" in self.controlnet:
                self.image_processor = SD3CannyImageProcessor()
            else:
                self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def print_summary(self, denoising_steps, walltime_ms):
        print("|-----------------|--------------|")
        print("| {:^15} | {:^12} |".format("Module", "Latency"))
        print("|-----------------|--------------|")
        for stage in self.stages:
            # controlnet is profiled in the denoising step
            if "controlnet" in stage:
                continue
            stage_name = stage
            if "transformer" in stage:
                if f"controlnet_{self.controlnet}" in self.stages:
                    stage_name += '+cnet'
                stage_name += ' x ' + str(denoising_steps)
            print(
                "| {:^15} | {:>9.2f} ms |".format(
                    stage_name, cudart.cudaEventElapsedTime(self.events[stage][0], self.events[stage][1])[1],
                )
            )
        print("|-----------------|--------------|")
        print("| {:^15} | {:>9.2f} ms |".format("Pipeline", walltime_ms))
        print("|-----------------|--------------|")
        print("Throughput: {:.5f} image/s".format(self.batch_size * 1000.0 / walltime_ms))

    @staticmethod
    def _tokenize(
        tokenizer: PreTrainedTokenizerBase,
        prompt: list[str],
        max_sequence_length: int,
        device: torch.device,
    ):
        text_input_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids = text_input_ids.type(torch.int32)

        untruncated_ids = tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
        ).input_ids.type(torch.int32)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            warnings.warn(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )
        text_input_ids = text_input_ids.to(device)
        return text_input_ids

    def _get_prompt_embed(
        self,
        prompt: list[str],
        encoder_name: str,
        domain="positive_prompt",
    ):
        if encoder_name == "clip_l":
            tokenizer = self.tokenizer
            max_sequence_length = tokenizer.model_max_length
            output_hidden_states = True
        elif encoder_name == "clip_g":
            tokenizer = self.tokenizer2
            max_sequence_length = tokenizer.model_max_length
            output_hidden_states = True
        elif encoder_name == "t5":
            tokenizer = self.tokenizer3
            max_sequence_length = self.max_sequence_length
            output_hidden_states = False
        else:
            raise NotImplementedError(f"encoder not found: {encoder_name}")

        self.profile_start(encoder_name, color="green", domain=domain)

        text_input_ids = self._tokenize(
            tokenizer=tokenizer,
            prompt=prompt,
            device=self.device,
            max_sequence_length=max_sequence_length,
        )

        text_hidden_states = None
        if self.torch_inference or self.torch_fallback[encoder_name]:
            outputs = self.torch_models[encoder_name](
                text_input_ids,
                output_hidden_states=output_hidden_states,
            )
            text_embeddings = outputs[0].clone()
            if output_hidden_states:
                text_hidden_states = outputs["hidden_states"][-2].clone()
        else:
            # NOTE: output tensor for the encoder must be cloned because it will be overwritten when called again for prompt2
            outputs = self.run_engine(encoder_name, {"input_ids": text_input_ids})
            text_embeddings = outputs["text_embeddings"].clone()
            if output_hidden_states:
                text_hidden_states = outputs["hidden_states"].clone()

        self.profile_stop(encoder_name)
        return text_hidden_states, text_embeddings

    @staticmethod
    def _duplicate_text_embed(
        prompt_embed: torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        pooled_prompt_embed: torch.Tensor | None = None,
    ):
        _, seq_len, _ = prompt_embed.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embed = prompt_embed.repeat(1, num_images_per_prompt, 1)
        prompt_embed = prompt_embed.view(batch_size * num_images_per_prompt, seq_len, -1)

        if pooled_prompt_embed is not None:
            pooled_prompt_embed = pooled_prompt_embed.repeat(1, num_images_per_prompt, 1)
            pooled_prompt_embed = pooled_prompt_embed.view(batch_size * num_images_per_prompt, -1)

        return prompt_embed, pooled_prompt_embed

    def encode_prompt(
        self,
        prompt: list[str],
        negative_prompt: list[str] | None = None,
        num_images_per_prompt: int = 1,
    ):
        clip_l_prompt_embed, clip_l_pooled_embed = self._get_prompt_embed(
            prompt=prompt,
            encoder_name="clip_l",
        )
        prompt_embed, pooled_prompt_embed = self._duplicate_text_embed(
            prompt_embed=clip_l_prompt_embed.clone(),
            pooled_prompt_embed=clip_l_pooled_embed.clone(),
            num_images_per_prompt=num_images_per_prompt,
            batch_size=self.batch_size,
        )

        clip_g_prompt_embed, clip_g_pooled_embed = self._get_prompt_embed(
            prompt=prompt,
            encoder_name="clip_g",
        )
        prompt_2_embed, pooled_prompt_2_embed = self._duplicate_text_embed(
            prompt_embed=clip_g_prompt_embed.clone(),
            pooled_prompt_embed=clip_g_pooled_embed.clone(),
            batch_size=self.batch_size,
            num_images_per_prompt=num_images_per_prompt,
        )

        _, t5_prompt_embed = self._get_prompt_embed(
            prompt=prompt,
            encoder_name="t5",
        )

        t5_prompt_embed, _ = self._duplicate_text_embed(
            prompt_embed=t5_prompt_embed.clone(),
            batch_size=self.batch_size,
            num_images_per_prompt=num_images_per_prompt,
        )

        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if negative_prompt is None:
            negative_prompt = ""

        clip_l_negative_prompt_embed, clip_l_negative_pooled_embed = self._get_prompt_embed(
            prompt=negative_prompt,
            encoder_name="clip_l",
        )
        negative_prompt_embed, negative_pooled_prompt_embed = self._duplicate_text_embed(
            prompt_embed=clip_l_negative_prompt_embed.clone(),
            pooled_prompt_embed=clip_l_negative_pooled_embed.clone(),
            batch_size=self.batch_size,
            num_images_per_prompt=num_images_per_prompt,
        )

        clip_g_negative_prompt_embed, clip_g_negative_pooled_embed = self._get_prompt_embed(
            prompt=negative_prompt,
            encoder_name="clip_g",
        )
        negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._duplicate_text_embed(
            prompt_embed=clip_g_negative_prompt_embed.clone(),
            pooled_prompt_embed=clip_g_negative_pooled_embed.clone(),
            batch_size=self.batch_size,
            num_images_per_prompt=num_images_per_prompt,
        )

        _, t5_negative_prompt_embed = self._get_prompt_embed(
            prompt=negative_prompt,
            encoder_name="t5",
        )

        t5_negative_prompt_embed, _ = self._duplicate_text_embed(
            prompt_embed=t5_negative_prompt_embed.clone(),
            batch_size=self.batch_size,
            num_images_per_prompt=num_images_per_prompt,
        )

        negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)
        negative_clip_prompt_embeds = torch.nn.functional.pad(
            negative_clip_prompt_embeds,
            (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
        )
        negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
        negative_pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @staticmethod
    def initialize_latents(
        batch_size: int,
        num_channels_latents: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        generator: torch.Generator,
        dtype=torch.float32,
        layout=torch.strided,
    ) -> torch.Tensor:
        latents_shape = (batch_size, num_channels_latents, latent_height, latent_width)
        latents = torch.randn(
            latents_shape,
            dtype=dtype,
            device="cuda",
            generator=generator,
            layout=layout,
        ).to(device)
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    @staticmethod
    def retrieve_timesteps(
        scheduler,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        **kwargs,
    ):
        r"""
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def get_control_block_samples(self, params_controlnet, controlnet_name="controlnet"):
        # Predict the controlnet block samples
        if self.torch_inference or self.torch_fallback[controlnet_name]:
            block_samples = self.torch_models[controlnet_name](**params_controlnet)
        else:
            block_samples = self.run_engine(controlnet_name, params_controlnet)["controlnet_block_samples"].clone()

        return block_samples

    def denoise_latents(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps: torch.FloatTensor,
        guidance_scale: float,
        denoiser="transformer",
        control_image=None,
        controlnet_scale=None,
        controlnet_keep=None,
    ) -> torch.Tensor:
        do_autocast = self.torch_inference != "" and self.models[denoiser].fp16
        with torch.autocast("cuda", enabled=do_autocast):
            self.profile_start(denoiser, color="blue")

            for step_index, timestep in enumerate(timesteps):
                # expand the latents as we are doing classifier free guidance
                latents_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep_inp = timestep.expand(latents_model_input.shape[0])

                controlnet_name = f"controlnet_{self.controlnet}"
                if control_image is not None:
                    cond_scale = controlnet_scale * controlnet_keep[step_index]

                    cast_to = (
                        torch.float16
                        if self.models[controlnet_name].fp16
                        else torch.bfloat16 if self.models[controlnet_name].bf16 else torch.float32
                    )
                    params_controlnet = {
                        "hidden_states": latents_model_input,
                        "timestep": timestep_inp,
                        "pooled_projections": pooled_prompt_embeds,
                        "controlnet_cond": control_image,
                        "conditioning_scale": cond_scale.to(self.device).to(cast_to),
                    }

                    control_block_samples = self.get_control_block_samples(params_controlnet, controlnet_name)
                else:
                    latent_shape = latents_model_input.shape
                    # Initialize control block samples with zeros. Hard-coding some dimensions that can only be queried if a controlnet is used.
                    control_block_samples = torch.zeros(
                        self.models["transformer"].num_controlnet_layers,
                        latent_shape[0],  # batch size
                        latent_shape[2] // 2 * latent_shape[3] // 2,
                        self.models["transformer"].config["num_attention_heads"]
                        * self.models["transformer"].config["attention_head_dim"],
                        dtype=latents.dtype,
                        device=latents.device,
                    )
                params = {
                    "hidden_states": latents_model_input,
                    "timestep": timestep_inp,
                    "encoder_hidden_states": prompt_embeds,
                    "pooled_projections": pooled_prompt_embeds,
                    "block_controlnet_hidden_states": control_block_samples,
                }

                # Predict the noise residual
                if self.torch_inference or self.torch_fallback[denoiser]:
                    noise_pred = self.torch_models[denoiser](**params)["sample"]
                else:
                    noise_pred = self.run_engine(denoiser, params)["latent"]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

            self.profile_stop(denoiser)
        return latents

    def prepare_image(
        self,
        image,
        width,
        height,
        device,
        dtype,
    ):
        image = self.image_processor.preprocess(image, height=height, width=width)

        image = image.to(device=device, dtype=dtype)

        if self.do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def encode_image(self, input_image, model_name="vae_encoder"):
        self.profile_start(model_name, color="red")
        cast_to = (
            torch.float16
            if self.models[model_name].fp16
            else torch.bfloat16 if self.models[model_name].bf16 else torch.float32
        )
        input_image = input_image.to(dtype=cast_to)
        if self.torch_inference or self.torch_fallback[model_name]:
            image_latents = self.torch_models[model_name](input_image)
        else:
            image_latents = self.run_engine(model_name, {"images": input_image})["latent"]
        image_latents = (image_latents - self.models["vae"].config["shift_factor"]) * self.models[
            "vae"
        ].config["scaling_factor"]
        self.profile_stop(model_name)
        return image_latents

    def decode_latents(self, latents: torch.Tensor, decoder="vae") -> torch.Tensor:
        cast_to = (
            torch.float16
            if self.models[decoder].fp16
            else torch.bfloat16
            if self.models[decoder].bf16
            else torch.float32
        )
        latents = latents.to(dtype=cast_to)
        self.profile_start(decoder, color="red")
        if self.torch_inference or self.torch_fallback[decoder]:
            images = self.torch_models[decoder](latents, return_dict=False)[0]
        else:
            images = self.run_engine(decoder, {"latent": latents})["images"]
        self.profile_stop(decoder)
        return images

    def infer(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        image_height: int,
        image_width: int,
        control_image=None,
        controlnet_scale=None,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        warmup=False,
        save_image=True,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (list[str]):
                The text prompt to guide image generation.
            negative_prompt (list[str]):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            control_image (PIL.Image.Image):
                The control image to guide the image generation.
            controlnet_scale (torch.Tensor):
                A tensor which contains ControlNet scale, essential for multi ControlNet.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            warmup (bool):
                Indicate if this is a warmup run.
            save_image (bool):
                Save the generated image (if applicable)
        """
        assert len(prompt) == len(negative_prompt)
        self.batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        assert image_height % (self.vae_scale_factor * self.patch_size) == 0, (
            f"image height not supported {image_height}"
        )
        assert image_width % (self.vae_scale_factor * self.patch_size) == 0, f"image width not supported {image_width}"
        latent_height = int(image_height) // self.vae_scale_factor
        latent_width = int(image_width) // self.vae_scale_factor

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # 1. encode inputs
            with self.model_memory_manager(["clip_g", "clip_l", "t5"], low_vram=self.low_vram):
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=1,
                )
            # do classifier free guidance
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            # 2. Prepare latent variables
            num_channels_latents = self.models["transformer"].config["in_channels"]
            latents = self.initialize_latents(
                batch_size=self.batch_size,
                num_channels_latents=num_channels_latents,
                latent_height=latent_height,
                latent_width=latent_width,
                device=prompt_embeds.device,
                generator=self.generator,
                dtype=torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32,
            )

            # 3. Prepare timesteps
            timesteps, num_inference_steps = self.retrieve_timesteps(
                scheduler=self.scheduler,
                num_inference_steps=self.denoising_steps,
                device=self.device,
                sigmas=None,
            )

            # 4. Prepare control image
            controlnet_keep = []
            if control_image is not None:
                # Process controlnet_scales
                for i in range(len(timesteps)):
                    keeps = [
                        1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                        for s, e in zip([control_guidance_start], [control_guidance_end])
                    ]
                    controlnet_keep.append(keeps[0])

                control_image = self.prepare_image(
                    image=control_image,
                    width=image_width,
                    height=image_height,
                    device=self.device,
                    dtype=torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32,
                )

                with self.model_memory_manager(["vae_encoder"], low_vram=self.low_vram):
                    control_image = self.encode_image(control_image)

            # 5. Denoise
            denoiser_list = ["transformer", f"controlnet_{self.controlnet}"] if self.controlnet else ["transformer"]
            with self.model_memory_manager(denoiser_list, low_vram=self.low_vram):
                latents = self.denoise_latents(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    timesteps=timesteps,
                    guidance_scale=self.guidance_scale,
                    control_image=control_image,
                    # TODO: support multiple controlnets
                    controlnet_scale=controlnet_scale,
                    controlnet_keep=controlnet_keep,
                )

            # 6. Decode Latents
            latents = (latents / self.models["vae"].config["scaling_factor"]) + self.models["vae"].config[
                "shift_factor"
            ]
            with self.model_memory_manager(["vae"], low_vram=self.low_vram):
                images = self.decode_latents(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.0
        if not warmup:
            self.print_summary(
                num_inference_steps,
                walltime_ms,
            )
            if save_image:
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
                self.save_image(images, self.pipeline_type.name.lower(), prompt, self.seed)

        return images, walltime_ms

    def run(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        height: int,
        width: int,
        batch_count: int,
        num_warmup_runs: int,
        use_cuda_graph: bool,
        **kwargs,
    ):
        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                self.infer(prompt, negative_prompt, height, width, warmup=True, **kwargs)

        for _ in range(batch_count):
            print("[I] Running StableDiffusion 3.5 pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            self.infer(prompt, negative_prompt, height, width, warmup=False, **kwargs)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()
