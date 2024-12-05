#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warnings
import numpy as np
from cuda import cudart
import inspect
from models import (
    get_clip_embedding_dim,
    make_tokenizer,
    CLIPModel,
    T5Model,
    FluxTransformerModel,
    VAEModel,
)
import tensorrt as trt
import time
import torch
from utilities import (
    PIPELINE_TYPE,
    TRT_LOGGER,
)
from diffusion_pipeline import DiffusionPipeline


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
        bf16=False,
        low_vram=False,
        torch_fallback=None,
        weight_streaming=False,
        t5_weight_streaming_budget_percentage=None,
        transformer_weight_streaming_budget_percentage=None,
        **kwargs
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
            bf16 (`bool`, defaults to False):
                Whether to run the pipeline in BFloat16 precision.
            weight_streaming (`bool`, defaults to False):
                Whether to enable weight streaming during TensorRT engine build.
            t5_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the T5 model.
            transformer_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the transformer model.
        """
        super().__init__(version=version, pipeline_type=pipeline_type, weight_streaming=weight_streaming, text_encoder_weight_streaming_budget_percentage=t5_weight_streaming_budget_percentage, denoiser_weight_streaming_budget_percentage=transformer_weight_streaming_budget_percentage, **kwargs)
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        self.bf16=bf16
        self.low_vram = low_vram

        # Pipeline type
        self.stages = ["clip", "t5", "transformer", "vae"]

        if torch_fallback:
            assert type(torch_fallback) is list
            for model_name in torch_fallback:
                if model_name not in self.stages:
                    raise ValueError(f'Model "{model_name}" set in --torch-fallback does not exist')
                self.config[model_name.replace('-','_')+'_torch_fallback'] = True
                print(f'[I] Setting torch_fallback for {model_name} model.')

    def _initialize_models(self, framework_model_dir, int8, fp8):
        # Load text tokenizer(s)
        self.tokenizer = make_tokenizer(
            self.version, self.pipeline_type, self.hf_token, framework_model_dir
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

        self.bf16 = True if int8 or fp8 else self.bf16
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
                build_strongly_typed=True if self.fp16 else False,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.text_encoder_weight_streaming_budget_percentage,
            )

        if "transformer" in self.stages:
            self.models["transformer"] = FluxTransformerModel(
                **models_args,
                bf16=True if int8 or fp8 else self.bf16,
                fp16=False if int8 or fp8 else self.fp16,
                int8=int8,
                fp8=fp8,
                tf32=self.tf32,
                text_maxlen=self.max_sequence_length,
                build_strongly_typed=True,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.denoiser_weight_streaming_budget_percentage,
            )

        if "vae" in self.stages:
            # Accuracy issues with FP16
            # WAR: VAE fallback to FP32 in BF16 pipeline. TRT support will be added in a future release
            self.models["vae"] = VAEModel(**models_args, fp16=False, tf32=self.tf32, bf16=False)

        self.vae_scale_factor = (
            2 ** (len(self.models["vae"].config["block_out_channels"]))
            if "vae" in self.stages and self.models["vae"] is not None
            else 16
        )

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

        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, latent_height, latent_width
        )

        latent_image_ids = self._prepare_latent_image_ids(
            latent_height, latent_width, latents_dtype, self.device
        )

        return latents, latent_image_ids

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
                # prepare inputs
                timestep_inp = timestep.expand(latents.shape[0]).to(latents.dtype)
                params = {
                    "hidden_states": latents,
                    "timestep": timestep_inp / 1000,
                    "pooled_projections": pooled_embeddings,
                    "encoder_hidden_states": text_embeddings,
                    "txt_ids": text_ids,
                    "img_ids": latent_image_ids,
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
        print("Throughput: {:.2f} image/s".format(batch_size * 1000.0 / walltime_ms))

    def infer(
        self,
        prompt,
        prompt2,
        image_height,
        image_width,
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

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Initialize latents
            latents, latent_image_ids = self.initialize_latents(
                batch_size=batch_size,
                num_channels_latents=self.models["transformer"].config["in_channels"]
                // 4,
                latent_height=latent_height,
                latent_width=latent_width,
                latents_dtype=torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32)

            class LoadModelContext:
                def __init__(ctx, model_names, low_vram=False):
                    ctx.model_names = model_names
                    ctx.low_vram = low_vram
                def __enter__(ctx):
                    if not ctx.low_vram:
                        return
                    for model_name in ctx.model_names:
                        if not self.torch_fallback[model_name]:
                            # creating engine object (load from plan file)
                            self.engine[model_name].load()
                            # creating context
                            self.engine[model_name].activate(device_memory=self.shared_device_memory)
                            # creating input and output buffer
                            self.engine[model_name].allocate_buffers(shape_dict=self.shape_dicts[model_name], device=self.device)
                        else:
                            print(f"[I] Reloading torch model {model_name} from cpu.")
                            self.torch_models[model_name] = self.torch_models[model_name].to('cuda')

                def __exit__(ctx, exc_type, exc_val, exc_tb):
                    if not ctx.low_vram:
                        return
                    for model_name in ctx.model_names:
                        if not self.torch_fallback[model_name]:
                            self.engine[model_name].deallocate_buffers()
                            self.engine[model_name].deactivate()
                            self.engine[model_name].unload()
                        else:
                            print(f"[I] Offloading torch model {model_name} to cpu.")
                            self.torch_models[model_name] = self.torch_models[model_name].to('cpu')
                            torch.cuda.empty_cache()

            # CLIP and T5 text encoder(s)

            with LoadModelContext(["clip","t5"], low_vram=self.low_vram):
                pooled_embeddings = self.encode_prompt(prompt, pooled_output=True)
                text_embeddings = self.encode_prompt(
                    prompt2, encoder="t5", max_sequence_length=self.max_sequence_length
                )
                text_ids = torch.zeros(text_embeddings.shape[1], 3).to(
                    device=self.device, dtype=text_embeddings.dtype
                )

            # Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = latents.shape[1]
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

            # DiT denoiser
            with LoadModelContext(["transformer"], low_vram=self.low_vram):
                latents = self.denoise_latent(
                    latents,
                    timesteps,
                    text_embeddings,
                    pooled_embeddings,
                    text_ids,
                    latent_image_ids,
                )

            # VAE decode latent
            with LoadModelContext(["vae"], low_vram=self.low_vram):
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
                self.save_image(
                    images, self.pipeline_type.name.lower(), prompt, self.seed
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

        for _ in range(batch_count):
            print("[I] Running Flux pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            self.infer(prompt, prompt2, height, width, warmup=False, **kwargs)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()
