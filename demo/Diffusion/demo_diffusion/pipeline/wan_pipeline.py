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

import argparse
import gc
import html
import os
import pathlib
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm.auto import tqdm

try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

import demo_diffusion.engine as engine_module
import demo_diffusion.image as image_module
import demo_diffusion.path as path_module
from demo_diffusion.model import (
    T5Model,
    WanTransformerModel,
    AutoencoderKLWanModel,
    make_tokenizer,
)
from demo_diffusion.pipeline.diffusion_pipeline import DiffusionPipeline
from demo_diffusion.pipeline.type import PIPELINE_TYPE

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L78
def basic_clean(text):
    if FTFY_AVAILABLE:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L84
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L90
def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class WanPipeline(DiffusionPipeline):
    """
    Application showcasing the acceleration of Wan 2.2 T2V pipeline using Nvidia TensorRT.
    """
    
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    
    def __init__(
        self,
        dd_path,
        version='wan2.2-t2v-a14b',
        pipeline_type=PIPELINE_TYPE.TXT2VID,
        boundary_ratio: float = 0.875,
        guidance_scale: float = 4.0,
        guidance_scale_2: float = 3.0,
        t5_weight_streaming_budget_percentage=None,
        transformer_weight_streaming_budget_percentage=None,
        **kwargs
    ):
        """
        Initializes the Wan T2V pipeline.

        Args:
            dd_path (load_module.DDPath): 
                DDPath object that contains all paths used in DemoDiffusion
            version (str):
                The version of the pipeline. Should be [wan2.2-t2v-a14b]
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline (TXT2VID)
            boundary_ratio (float, defaults to 0.875):
                Ratio of total timesteps to use as the boundary for switching between transformers
                in two-stage denoising. Transformer handles high-noise stages (timesteps >= boundary)
                and transformer_2 handles low-noise stages (timesteps < boundary). 
                Wan 2.2 T2V always uses two-stage denoising.
            guidance_scale (float):
                Guidance scale for high-noise stage (transformer). Wan default: 4.0
            guidance_scale_2 (float):
                Guidance scale for low-noise stage (transformer_2). Wan default: 3.0
            t5_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the T5 model.
            transformer_weight_streaming_budget_percentage (`int`, defaults to None):
                Weight streaming budget as a percentage of the size of total streamable weights for the transformer models.
        """
        super().__init__(
            dd_path=dd_path,
            version=version,
            pipeline_type=pipeline_type,
            scheduler="UniPC",
            bf16=True,
            text_encoder_weight_streaming_budget_percentage=t5_weight_streaming_budget_percentage,
            denoiser_weight_streaming_budget_percentage=transformer_weight_streaming_budget_percentage,
            **kwargs
        )

        # Validate boundary_ratio (required for Wan 2.2 two-stage denoising)
        if boundary_ratio is None or not (0.0 < boundary_ratio < 1.0):
            raise ValueError(
                f"`boundary_ratio` must be between 0.0 and 1.0, got {boundary_ratio}"
            )

        self.boundary_ratio = boundary_ratio
        self.guidance_scale = guidance_scale
        self.guidance_scale_2 = guidance_scale_2

        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        
        self.opt_image_height = 720
        self.opt_image_width = 1280
        self.opt_num_frames = 81
        self.max_sequence_length = 512

    @classmethod
    def FromArgs(cls, args: argparse.Namespace, pipeline_type: PIPELINE_TYPE) -> 'WanPipeline':
        """Factory method to construct a WanPipeline object from parsed arguments."""

        MAX_BATCH_SIZE = 1  # Wan always uses batch size 1
        DEVICE = "cuda"

        dd_path = path_module.resolve_path(
            cls.get_model_names(pipeline_type), args, pipeline_type, cls._get_pipeline_uid(args.version)
        )
        
        return cls(
            dd_path=dd_path,
            version=args.version,
            pipeline_type=pipeline_type,
            boundary_ratio=args.boundary_ratio,
            denoising_steps=args.denoising_steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            t5_weight_streaming_budget_percentage=args.t5_ws_percentage if hasattr(args, 't5_ws_percentage') else None,
            transformer_weight_streaming_budget_percentage=args.transformer_ws_percentage if hasattr(args, 'transformer_ws_percentage') else None,
            max_batch_size=MAX_BATCH_SIZE,
            device=DEVICE,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            verbose=args.verbose,
            nvtx_profile=args.nvtx_profile,
            use_cuda_graph=args.use_cuda_graph,
            framework_model_dir=args.framework_model_dir,
            low_vram=args.low_vram,
            torch_inference=args.torch_inference,
            torch_fallback=args.torch_fallback if hasattr(args, 'torch_fallback') else None,
            weight_streaming=args.ws if hasattr(args, 'ws') else False,
        )
    
    @classmethod
    def get_model_names(cls, pipeline_type: PIPELINE_TYPE, controlnet_type: str = None) -> List[str]:
        """Return a list of model names used by this pipeline.

        Overrides:
            DiffusionPipeline.get_model_names
        """
        return ["text_encoder", "transformer", "transformer_2", "vae_decoder"]
    
    def download_onnx_models(self, model_name: str, model_config: dict[str, Any]) -> None:
        raise NotImplementedError(
            "Pre-exported Wan ONNX models are not available for download. "
            "Export ONNX models locally using the provided export script."
        )
    
    def _initialize_models(self, framework_model_dir, int8=False, fp8=False, fp4=False):
        self.tokenizer = make_tokenizer(
            self.version,
            self.pipeline_type,
            self.hf_token,
            framework_model_dir,
            subfolder='tokenizer',
            tokenizer_type='t5'
        )

        models_args = {
            'version': self.version,
            'pipeline': self.pipeline_type,
            'device': self.device,
            'hf_token': self.hf_token,
            'verbose': self.verbose,
            'framework_model_dir': framework_model_dir,
            'max_batch_size': 1
        }

        if "text_encoder" in self.stages:
            self.models['text_encoder'] = T5Model(
                **models_args,
                fp16=False,
                bf16=True,
                text_maxlen=self.max_sequence_length,
                build_strongly_typed=True,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.text_encoder_weight_streaming_budget_percentage,
                use_attention_mask=True,
            )

        if "transformer" in self.stages:
            self.models['transformer'] = WanTransformerModel(
                **models_args,
                subfolder='transformer',
                fp16=False,
                bf16=True,
                text_maxlen=self.max_sequence_length,
                num_frames=self.opt_num_frames,
                height=self.opt_image_height,
                width=self.opt_image_width,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.denoiser_weight_streaming_budget_percentage,
            )

        if "transformer_2" in self.stages:
            self.models['transformer_2'] = WanTransformerModel(
                **models_args,
                subfolder='transformer_2',
                fp16=False,
                bf16=True,
                text_maxlen=self.max_sequence_length,
                num_frames=self.opt_num_frames,
                height=self.opt_image_height,
                width=self.opt_image_width,
                weight_streaming=self.weight_streaming,
                weight_streaming_budget_percentage=self.denoiser_weight_streaming_budget_percentage,
            )

        if "vae_decoder" in self.stages:
            self.models['vae_decoder'] = AutoencoderKLWanModel(
                **models_args,
            )

            self.config['vae_decoder_torch_fallback'] = True

    def load_resources(self, image_height, image_width, batch_size, seed):
        """Override to create additional 'denoise' event for combined transformer timing."""
        super().load_resources(image_height, image_width, batch_size, seed)
        # additional event for combined denoising timing (both transformers)
        self.events['denoise'] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]

    def print_summary(self, denoising_steps, walltime_ms, batch_size, num_frames):
        print("|----------------------|--------------|")
        print("| {:^20} | {:^12} |".format("Module", "Latency"))
        print("|----------------------|--------------|")
        
        # calculate transformer timings from combined denoise event
        total_denoise_time = cudart.cudaEventElapsedTime(self.events['denoise'][0], self.events['denoise'][1])[1]
        transformer_steps_map = {
            'transformer': self.transformer_steps if (self.transformer_steps > 0 and self.transformer_2_steps > 0) else denoising_steps,
            'transformer_2': self.transformer_2_steps if self.transformer_2_steps > 0 else 0
        }
        
        for stage in self.stages:
            if stage in transformer_steps_map and transformer_steps_map[stage] > 0:
                steps = transformer_steps_map[stage]
                time_ms = total_denoise_time * (steps / denoising_steps)
                stage_label = f"{stage} x {steps}"
            elif stage in transformer_steps_map:
                continue # skip transformer_2 if unused
            else:
                time_ms = cudart.cudaEventElapsedTime(self.events[stage][0], self.events[stage][1])[1]
                stage_label = stage
            
            print("| {:^20} | {:>9.2f} ms |".format(stage_label, time_ms))
        
        print("|----------------------|--------------|")
        print("| {:^20} | {:>9.2f} ms |".format("Pipeline", walltime_ms))
        print("|----------------------|--------------|")
        print("Throughput: {:.2f} videos/min ({} frames)".format(batch_size * 60000.0 / walltime_ms, num_frames))

    def save_video(self, frames, pipeline, prompt, seed):
        if isinstance(prompt, list):
            prompt_prefix = ''.join(set([p.replace(' ','_')[:10] for p in prompt]))
        else:
            prompt_prefix = prompt.replace(' ','_')[:10]
        
        seed_str = str(seed) if seed is not None else 'random'
        precision = 'bf16' if self.bf16 else 'fp16' if self.fp16 else 'fp32'
        video_name_prefix = '-'.join([pipeline, prompt_prefix, precision, seed_str, str(random.randint(1000,9999))])
        video_name_suffix = 'torch' if self.torch_inference else 'trt'
        video_path = video_name_prefix+'-'+video_name_suffix+'.gif'
        full_path = os.path.join(self.output_dir, video_path)
        print(f"Saving video to: {full_path}")
        frames[0].save(full_path, save_all=True, optimize=False, append_images=frames[1:], loop=0)

    # Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L198
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        
        Implementation modeled from diffusers Wan pipeline, adapted for TensorRT.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length for text encoder.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        self.profile_start('text_encoder', color='green')
        
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        self.profile_stop('text_encoder')
        return prompt_embeds, negative_prompt_embeds

    def denoise_latents(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        guidance_scale: float,
        guidance_scale_2: float,
        transformer_dtype: torch.dtype,
        num_warmup_steps: int,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        warmup: bool = False,
    ) -> torch.Tensor:
        boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps
        self.profile_start('denoise', color='blue')
        
        timestep_stages = []
        for i, t in enumerate(timesteps):
            if t >= boundary_timestep:
                timestep_stages.append((i, t, 'transformer', guidance_scale))
            else:
                timestep_stages.append((i, t, 'transformer_2', guidance_scale_2))
        
        stage_groups = []
        if timestep_stages:
            current_group = {
                'transformer': timestep_stages[0][2],
                'guidance_scale': timestep_stages[0][3],
                'timesteps': [(timestep_stages[0][0], timestep_stages[0][1])]
            }
            
            for i, t, transformer_name, gs in timestep_stages[1:]:
                if transformer_name == current_group['transformer']:
                    current_group['timesteps'].append((i, t))
                else:
                    stage_groups.append(current_group)
                    current_group = {
                        'transformer': transformer_name,
                        'guidance_scale': gs,
                        'timesteps': [(i, t)]
                    }
            stage_groups.append(current_group)
        
        self.transformer_steps = sum(len(g['timesteps']) for g in stage_groups if g['transformer'] == 'transformer')
        self.transformer_2_steps = sum(len(g['timesteps']) for g in stage_groups if g['transformer'] == 'transformer_2')
        
        with tqdm(total=len(timesteps)) as progress_bar:
            for stage_group in stage_groups:
                transformer_name = stage_group['transformer']
                current_guidance_scale = stage_group['guidance_scale']
                
                with self.model_memory_manager([transformer_name], low_vram=self.low_vram):
                    for step_index, t in stage_group['timesteps']:
                        latent_model_input = latents.to(transformer_dtype)
                        timestep = t.expand(latent_model_input.shape[0])

                        if self.torch_inference or self.torch_fallback[transformer_name]:
                            current_model = self.torch_models[transformer_name]
                            
                            noise_pred_cond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]

                            if self.do_classifier_free_guidance:
                                noise_pred_uncond = current_model(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]

                                noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                            else:
                                noise_pred = noise_pred_cond
                        else:
                            if self.do_classifier_free_guidance:
                                params_cond = {
                                    "hidden_states": latent_model_input,
                                    "timestep": timestep,
                                    "encoder_hidden_states": prompt_embeds,
                                }
                                
                                # conditional engine call
                                output_cond = self.run_engine(transformer_name, params_cond)['denoised_latents']
                                
                                noise_pred_cond = output_cond.clone()
                                
                                params_uncond = {
                                    "hidden_states": latent_model_input,
                                    "timestep": timestep,
                                    "encoder_hidden_states": negative_prompt_embeds,
                                }
                                
                                # unconditional engine call
                                output_uncond = self.run_engine(transformer_name, params_uncond)['denoised_latents']
                                
                                noise_pred_uncond = output_uncond.clone()
                                
                                # Apply classifier-free guidance
                                noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                            else:
                                # No CFG
                                params = {
                                    "hidden_states": latent_model_input,
                                    "timestep": timestep,
                                    "encoder_hidden_states": prompt_embeds,
                                }
                                noise_pred = self.run_engine(transformer_name, params)['denoised_latents']

                        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                        if callback_on_step_end is not None:
                            callback_kwargs = {}
                            for k in callback_on_step_end_tensor_inputs:
                                callback_kwargs[k] = locals()[k]
                            callback_outputs = callback_on_step_end(self, step_index, t, callback_kwargs)

                            latents = callback_outputs.pop("latents", latents)
                            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                        if step_index == len(timesteps) - 1 or ((step_index + 1) > num_warmup_steps and (step_index + 1) % self.scheduler.order == 0):
                            progress_bar.update()
            
        self.profile_stop('denoise')
        return latents

    # Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L324
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if generator is None and hasattr(self, 'generator'):
            generator = self.generator

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    # Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L279
    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None:
            pass

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    # Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L157
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        encoder: str = "text_encoder",
    ):
        device = device or self._execution_device
        dtype = dtype or (self.torch_models[encoder].dtype if self.torch_fallback.get(encoder) and encoder in self.torch_models else torch.float32)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        if self.torch_inference or self.torch_fallback[encoder]:
            outputs = self.torch_models[encoder](text_input_ids.to(device), mask.to(device))
            prompt_embeds = outputs.last_hidden_state.clone()
        else:
            outputs = self.run_engine(encoder, {
                "input_ids": text_input_ids.to(device),
                "attention_mask": mask.to(device)
            })
            prompt_embeds = outputs['text_embeddings'].clone()
        
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    @property
    def _execution_device(self):
        return self.device

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def decode_latents(self, latents, num_frames):
        self.profile_start('vae_decoder', color='red')
        
        vae_config = self.models['vae_decoder'].config
        z_dim = vae_config.get("z_dim", 16)

        vae_dtype = torch.float32
        latents = latents.to(vae_dtype)
        
        latents_mean = (
            torch.tensor(vae_config.get("latents_mean"))
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae_config.get("latents_std")).view(1, z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        
        latents = latents / latents_std + latents_mean

        frames = self.torch_models['vae_decoder'].decode(latents, return_dict=False)[0]
        
        self.profile_stop('vae_decoder')
        return frames

    def postprocess(self, video: torch.Tensor, output_type: str = "pil"):
        # Convert [F, C, H, W] -> [F, H, W, C]
        video = video.permute(0, 2, 3, 1)
        # Convert to list of PIL Images
        video = (video + 1.0) / 2.0
        video = torch.clamp(video, 0.0, 1.0)
        video = (video * 255.0).to(torch.uint8).cpu().numpy()
        pil_frames = [Image.fromarray(frame) for frame in video]
        return pil_frames

    def infer(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        warmup: bool = False,
        save_video: bool = True,
    ):
        """
        Run the Wan text-to-video diffusion pipeline.
        """

        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            self.guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            print(f"[W] `num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number.")
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self._execution_device
        batch_size = 1

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            with self.model_memory_manager(["text_encoder"], low_vram=self.low_vram):
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    num_videos_per_prompt=num_videos_per_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    max_sequence_length=max_sequence_length,
                    device=device,
                )

            transformer_dtype = torch.bfloat16
            prompt_embeds = prompt_embeds.to(transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            num_channels_latents = self.models["transformer"].config.get("in_channels", 16)
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )

            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            latents = self.denoise_latents(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                timesteps=timesteps,
                guidance_scale=self.guidance_scale,
                guidance_scale_2=self.guidance_scale_2,
                transformer_dtype=transformer_dtype,
                num_warmup_steps=num_warmup_steps,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                warmup=warmup,
            )

            with self.model_memory_manager(["vae_decoder"], low_vram=self.low_vram):
                video_raw = self.decode_latents(latents, num_frames)
            video = image_module.tensor2vid(video_raw, self, output_type="pil")

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.0
        if not warmup:
            self.print_summary(num_inference_steps, walltime_ms, batch_size, num_frames)
            if save_video:
                self.save_video(video[0], self.pipeline_type.name.lower(), prompt, self.seed)

        return video, walltime_ms

    def run(self, prompt, height, width, num_frames, batch_size, batch_count, num_warmup_runs, use_cuda_graph, **kwargs):
        if self.low_vram and self.use_cuda_graph:
            print("[W] Using low_vram, use_cuda_graph will be disabled")
            self.use_cuda_graph = False
        
        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                self.infer(prompt, height=height, width=width, num_frames=num_frames, warmup=True, **kwargs)

        for _ in range(batch_count):
            print("[I] Running Wan T2V pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            self.infer(prompt, height=height, width=width, num_frames=num_frames, warmup=False, **kwargs)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()
