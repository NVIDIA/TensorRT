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

from collections import OrderedDict
from NNDF.networks import Dims

# Base Class
from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig

# Adapted from GPT2/GPT2ModelConfig.py

class BLOOMModelTRTConfig(Seq2SeqModelTRTConfig):

    TARGET_MODELS = [
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "bigscience/bloom-7b1",
        "bigscience/bloomz-560m",
        "bigscience/bloomz-1b1",
        "bigscience/bloomz-1b7",
        "bigscience/bloomz-3b",
        "bigscience/bloomz-7b1",
    ]

    def __init__(self, **kwargs):

        super().__init__(
            network_name="BLOOM",
            **kwargs
        )

    def from_hf_config(self, hf_config):
        """
        BLOOM model's n_embed is too long (~2048). The model size for the models in the demo
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
        # Remove redundant name
        metadata = metadata._replace(variant=metadata.variant.replace("bigscience/",""))
        return super().get_metadata_string(metadata)

    # BLOOM's kv cache shape is unique.
    # Most GPT2-like models use past_key_values with shape
    # [batch_size, num_heads, sequence_length, embed_size_per_head]
    # for both keys and values.
    # BLOOM's kv cache uses the following format:
    # Keys: [batch_size * num_heads, embed_size_per_head, sequence_length]
    # Vals: [batch_size * num_heads, sequence_length, embed_size_per_head]
    # So the kv cache requires special treatment

    def _get_cache_inputs(self):
        decoder_cache_inputs = OrderedDict({})
        for i in range(self.num_decoder_layers):
            decoder_cache_inputs["past_key_values.{}.self.key".format(i)] = (Dims.BATCH + '_times_num_heads', self.d_kv, Dims.create_new_sequence_dim("past_decoder"))
            decoder_cache_inputs["past_key_values.{}.self.value".format(i)] = (Dims.BATCH + '_times_num_heads', Dims.create_new_sequence_dim("past_decoder"), self.d_kv)

        return decoder_cache_inputs

    def _get_self_attention_cache_outputs(self):
        self_attention_cache_outputs = OrderedDict({})
        for i in range(self.num_decoder_layers):
            self_attention_cache_outputs["present_key_values.{}.self.key".format(i)] = (Dims.BATCH + '_times_num_heads', self.d_kv, Dims.create_new_sequence_dim("present_decoder"))
            self_attention_cache_outputs["present_key_values.{}.self.value".format(i)] = (Dims.BATCH + '_times_num_heads', Dims.create_new_sequence_dim("present_decoder"), self.d_kv)
        return self_attention_cache_outputs
