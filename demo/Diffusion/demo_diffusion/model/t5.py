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
from transformers import (
    AutoConfig,
    T5EncoderModel,
)

from demo_diffusion.model import base_model, load, optimizer


class T5Model(base_model.BaseModel):

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
        tf32=False,
        bf16=False,
        subfolder="text_encoder",
        text_maxlen=512,
        build_strongly_typed=False,
        weight_streaming=False,
        weight_streaming_budget_percentage=None,
        use_attention_mask=False,
    ):
        super(T5Model, self).__init__(
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
            text_maxlen=text_maxlen,
        )
        self.subfolder = subfolder
        self.t5_model_dir = load.get_checkpoint_dir(
            self.framework_model_dir, self.version, self.pipeline, self.subfolder
        )
        if not os.path.exists(self.t5_model_dir):
            self.config = AutoConfig.from_pretrained(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load T5Encoder Config from: {self.t5_model_dir}")
            self.config = AutoConfig.from_pretrained(self.t5_model_dir)
        self.build_strongly_typed = build_strongly_typed
        self.weight_streaming = weight_streaming
        self.weight_streaming_budget_percentage = weight_streaming_budget_percentage
        self.use_attention_mask = use_attention_mask

    def get_model(self, torch_inference=""):
        model_opts = (
            {"torch_dtype": torch.float16} if self.fp16 else {"torch_dtype": torch.bfloat16} if self.bf16 else {}
        )
        if not load.is_model_cached(self.t5_model_dir, model_opts, self.hf_safetensor, model_name="model"):
            model = T5EncoderModel.from_pretrained(
                self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts,
            ).to(self.device)
            model.save_pretrained(self.t5_model_dir, **model_opts)
        else:
            print(f"[I] Load T5EncoderModel model from: {self.t5_model_dir}")
            model = T5EncoderModel.from_pretrained(self.t5_model_dir, **model_opts).to(self.device)
        model = optimizer.optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        if self.use_attention_mask:
            return ["input_ids", "attention_mask"]
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings"]

    def get_dynamic_axes(self):
        if self.use_attention_mask:
            return {"input_ids": {0: "B"}, "attention_mask": {0: "B"}, "text_embeddings": {0: "B"}}
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        profile = {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }
        if self.use_attention_mask:
            profile["attention_mask"] = [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.config.d_model),
        }
        if self.use_attention_mask:
            output["attention_mask"] = (batch_size, self.text_maxlen)
        return output

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        inputs = {"input_ids": torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)}
        if self.use_attention_mask:
            inputs["attention_mask"] = torch.ones(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)
        return inputs
