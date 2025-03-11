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

from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)

from demo_diffusion.model import load


def make_tokenizer(version, pipeline, hf_token, framework_model_dir, subfolder="tokenizer", tokenizer_type="clip"):
    if tokenizer_type == "clip":
        tokenizer_class = CLIPTokenizer
    elif tokenizer_type == "t5":
        tokenizer_class = T5TokenizerFast
    else:
        raise ValueError(
            f"Unsupported tokenizer_type {tokenizer_type}. Only tokenizer_type clip and t5 are currently supported"
        )
    tokenizer_model_dir = load.get_checkpoint_dir(framework_model_dir, version, pipeline.name, subfolder)
    if not os.path.exists(tokenizer_model_dir):
        model = tokenizer_class.from_pretrained(
            load.get_path(version, pipeline), subfolder=subfolder, use_safetensors=pipeline.is_sd_xl(), token=hf_token
        )
        model.save_pretrained(tokenizer_model_dir)
    else:
        print(f"[I] Load {tokenizer_class.__name__} model from: {tokenizer_model_dir}")
        model = tokenizer_class.from_pretrained(tokenizer_model_dir)
    return model
