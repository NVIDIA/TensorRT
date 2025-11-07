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
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from demo_diffusion.model import base_model, load, optimizer
from demo_diffusion.utils_sd3.other_impls import (
    SDClipModel,
    SDXLClipG,
    T5XXLModel,
    load_into,
)


def get_clipwithproj_embedding_dim(version: str, subfolder: str) -> int:
    """Return the embedding dimension of a CLIP with projection model."""
    if version in ("xl-1.0", "xl-turbo", "cascade"):
        return 1280
    elif version in {"3.5-medium", "3.5-large"} and subfolder == "text_encoder":
        return 768
    elif version in {"3.5-medium", "3.5-large"} and subfolder == "text_encoder_2":
        return 1280
    else:
        raise ValueError(f"Invalid version {version} + subfolder {subfolder}")


def get_clip_embedding_dim(version, pipeline):
    if version in (
        "1.4",
        "1.5",
        "dreamshaper-7",
        "flux.1-dev",
        "flux.1-schnell",
        "flux.1-dev-canny",
        "flux.1-dev-depth",
        "flux.1-kontext-dev",
    ):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_base():
        return 768
    elif version in ("sd3"):
        return 4096
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")


class CLIPModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        embedding_dim,
        fp16=False,
        tf32=False,
        bf16=False,
        output_hidden_states=False,
        keep_pooled_output=False,
        subfolder="text_encoder",
    ):
        super(CLIPModel, self).__init__(
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
            embedding_dim=embedding_dim,
        )
        self.subfolder = subfolder
        self.hidden_layer_offset = 0 if pipeline.is_cascade() else -1
        self.keep_pooled_output = keep_pooled_output

        # Output the final hidden state
        if output_hidden_states:
            self.extra_output_names = ["hidden_states"]

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        clip_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not load.is_model_cached(clip_model_dir, model_opts, self.hf_safetensor, model_name="model"):
            model = CLIPTextModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                attn_implementation="eager",
                **model_opts,
            ).to(self.device)
            model.save_pretrained(clip_model_dir, **model_opts)
        else:
            print(f"[I] Load CLIPTextModel model from: {clip_model_dir}")
            model = CLIPTextModel.from_pretrained(clip_model_dir, **model_opts).to(self.device)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        output_names = ["text_embeddings"]
        if self.keep_pooled_output:
            output_names += ["pooled_embeddings"]
        return output_names

    def get_dynamic_axes(self):
        dynamic_axes = {
            "input_ids": {0: "B"},
            "text_embeddings": {0: "B"},
        }
        if self.keep_pooled_output:
            dynamic_axes["pooled_embeddings"] = {0: "B"}
        return dynamic_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }
        if self.keep_pooled_output:
            output["pooled_embeddings"] = (batch_size, self.embedding_dim)
        if "hidden_states" in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = optimizer.Optimizer(onnx_graph, verbose=self.verbose, version=self.version)
        opt.info(self.name + ": original")
        keep_outputs = [0, 1] if self.keep_pooled_output else [0]
        opt.select_outputs(keep_outputs)
        opt.cleanup()
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs(keep_outputs, names=self.get_output_names())  # rename network outputs
        opt.info(self.name + ": rename network output(s)")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        if "hidden_states" in self.extra_output_names:
            opt_onnx_graph = opt.clip_add_hidden_states(self.hidden_layer_offset, return_onnx=True)
            opt.info(self.name + ": added hidden_states")
        opt.info(self.name + ": finished")
        return opt_onnx_graph

class CLIPWithProjModel(CLIPModel):

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
        output_hidden_states=False,
        subfolder="text_encoder_2",
    ):

        super(CLIPWithProjModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            bf16=bf16,
            max_batch_size=max_batch_size,
            embedding_dim=get_clipwithproj_embedding_dim(version, subfolder),
            output_hidden_states=output_hidden_states,
        )
        self.subfolder = subfolder

    def get_model(self, torch_inference=""):
        model_opts = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16}
        clip_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not load.is_model_cached(clip_model_dir, model_opts, self.hf_safetensor, model_name="model"):
            model = CLIPTextModelWithProjection.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                attn_implementation="eager",
                **model_opts,
            ).to(self.device)
            model.save_pretrained(clip_model_dir, **model_opts)
        else:
            print(f"[I] Load CLIPTextModelWithProjection model from: {clip_model_dir}")
            model = CLIPTextModelWithProjection.from_pretrained(clip_model_dir, **model_opts).to(self.device)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ["input_ids", "attention_mask"]

    def get_output_names(self):
        return ["text_embeddings"]

    def get_dynamic_axes(self):
        return {
            "input_ids": {0: "B"},
            "attention_mask": {0: "B"},
            "text_embeddings": {0: "B"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)],
            "attention_mask": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "attention_mask": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.embedding_dim),
        }
        if "hidden_states" in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        return (
            torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device),
            torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device),
        )

class SD3_CLIPGModel(CLIPModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        embedding_dim=None,
        fp16=False,
        pooled_output=False,
    ):
        self.CLIPG_CONFIG = {
            "hidden_act": "gelu",
            "hidden_size": 1280,
            "intermediate_size": 5120,
            "num_attention_heads": 20,
            "num_hidden_layers": 32,
        }
        super(SD3_CLIPGModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=self.CLIPG_CONFIG["hidden_size"] if embedding_dim is None else embedding_dim,
        )
        self.subfolder = "text_encoders"
        if pooled_output:
            self.extra_output_names = ["pooled_output"]

    def get_model(self, torch_inference=""):
        clip_g_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        clip_g_filename = "clip_g.safetensors"
        clip_g_model_path = f"{clip_g_model_dir}/{clip_g_filename}"
        if not os.path.exists(clip_g_model_path):
            hf_hub_download(
                repo_id=self.path,
                filename=clip_g_filename,
                local_dir=load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, ""),
                subfolder=self.subfolder,
            )
        with safe_open(clip_g_model_path, framework="pt", device=self.device) as f:
            dtype = torch.float16 if self.fp16 else torch.float32
            model = SDXLClipG(self.CLIPG_CONFIG, device=self.device, dtype=dtype)
            load_into(f, model.transformer, "", self.device, dtype)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }
        if "pooled_output" in self.extra_output_names:
            output["pooled_output"] = (batch_size, self.embedding_dim)

        return output

    def optimize(self, onnx_graph):
        opt = optimizer.Optimizer(onnx_graph, verbose=self.verbose, version=self.version)
        opt.info(self.name + ": original")
        opt.select_outputs([0, 1])
        opt.cleanup()
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0, 1], names=["text_embeddings", "pooled_output"])  # rename network output
        opt.info(self.name + ": rename output[0] and output[1]")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class SD3_CLIPLModel(SD3_CLIPGModel):
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
        pooled_output=False,
    ):
        self.CLIPL_CONFIG = {
            "hidden_act": "quick_gelu",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
        }
        super(SD3_CLIPLModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=self.CLIPL_CONFIG["hidden_size"],
        )
        self.subfolder = "text_encoders"
        if pooled_output:
            self.extra_output_names = ["pooled_output"]

    def get_model(self, torch_inference=""):
        clip_l_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        clip_l_filename = "clip_l.safetensors"
        clip_l_model_path = f"{clip_l_model_dir}/{clip_l_filename}"
        if not os.path.exists(clip_l_model_path):
            hf_hub_download(
                repo_id=self.path,
                filename=clip_l_filename,
                local_dir=load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, ""),
                subfolder=self.subfolder,
            )
        with safe_open(clip_l_model_path, framework="pt", device=self.device) as f:
            dtype = torch.float16 if self.fp16 else torch.float32
            model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device=self.device,
                dtype=dtype,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=self.CLIPL_CONFIG,
            )
            load_into(f, model.transformer, "", self.device, dtype)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model


# NOTE: For legacy reasons, even though this is a T5 model, it inherits from CLIPModel.
class SD3_T5XXLModel(CLIPModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        embedding_dim,
        fp16=False,
    ):
        super(SD3_T5XXLModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        )
        self.T5_CONFIG = {"d_ff": 10240, "d_model": 4096, "num_heads": 64, "num_layers": 24, "vocab_size": 32128}
        self.subfolder = "text_encoders"

    def get_model(self, torch_inference=""):
        t5xxl_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        t5xxl_filename = "t5xxl_fp16.safetensors"
        t5xxl_model_path = f"{t5xxl_model_dir}/{t5xxl_filename}"
        if not os.path.exists(t5xxl_model_path):
            hf_hub_download(
                repo_id=self.path,
                filename=t5xxl_filename,
                local_dir=load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, ""),
                subfolder=self.subfolder,
            )
        with safe_open(t5xxl_model_path, framework="pt", device=self.device) as f:
            dtype = torch.float16 if self.fp16 else torch.float32
            model = T5XXLModel(self.T5_CONFIG, device=self.device, dtype=dtype)
            load_into(f, model.transformer, "", self.device, dtype)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model


class CLIPVisionWithProjModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=1,
        subfolder="image_encoder",
    ):

        super(CLIPVisionWithProjModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            max_batch_size=max_batch_size,
        )
        self.subfolder = subfolder

    def get_model(self, torch_inference=""):
        clip_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not os.path.exists(clip_model_dir):
            model = CLIPVisionModelWithProjection.from_pretrained(
                self.path, subfolder=self.subfolder, use_safetensors=self.hf_safetensor, token=self.hf_token
            ).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIPVisionModelWithProjection model from: {clip_model_dir}")
            model = CLIPVisionModelWithProjection.from_pretrained(clip_model_dir).to(self.device)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model


class CLIPImageProcessorModel(base_model.BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=1,
        subfolder="feature_extractor",
    ):

        super(CLIPImageProcessorModel, self).__init__(
            version,
            pipeline,
            device=device,
            hf_token=hf_token,
            verbose=verbose,
            framework_model_dir=framework_model_dir,
            max_batch_size=max_batch_size,
        )
        self.subfolder = subfolder

    def get_model(self, torch_inference=""):
        clip_model_dir = load.get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        # NOTE to(device) not supported
        if not os.path.exists(clip_model_dir):
            model = CLIPImageProcessor.from_pretrained(
                self.path, subfolder=self.subfolder, use_safetensors=self.hf_safetensor, token=self.hf_token
            )
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIPImageProcessor model from: {clip_model_dir}")
            model = CLIPImageProcessor.from_pretrained(clip_model_dir)
        return model
