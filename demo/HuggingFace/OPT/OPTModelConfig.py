#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Base Class
from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig

class OPTModelTRTConfig(Seq2SeqModelTRTConfig):

    TARGET_MODELS = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        #"facebook/opt-30b", # Too big for single GPU
        #"facebook/opt-66b", # Too big for single GPU
    ]

    def __init__(self, **kwargs):

        super().__init__(
            network_name="OPT",
            **kwargs
        )

    def from_hf_config(self, hf_config):
        """
        OPT model's n_positions is too long (~2048). The model size for the models in the demo
        is not large enough to generate useful information and therefore will generate repetitive sentences.
        Truncate to 100 for useful informations.
        """
        super().from_hf_config(hf_config, model_max_len=100)
        # Additional parameter to disable HF warning
        self.pad_token_id = self.eos_token_id

    def set_generation_config(self, generation_config):
        super().set_generation_config(generation_config)
        self.generation_config.pad_token_id = self.eos_token_id

    def get_metadata_string(self, metadata) -> str:
        # Remove redundant OPT name
        metadata = metadata._replace(variant=metadata.variant.replace("facebook/",""))
        return super().get_metadata_string(metadata)
