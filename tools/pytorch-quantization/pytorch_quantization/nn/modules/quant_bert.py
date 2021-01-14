#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Quantized BERT Self Attention

This method has been adapted from the transformers repo:
https://github.com/huggingface/transformers/tree/v2.9.1
"""


import math

import torch
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

__all__ = ["QuantBertSelfAttention"]

class QuantBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Quantized implementations of torch.nn.Linear modules
        self.query = quant_nn.QuantLinear(config.hidden_size, self.all_head_size)
        self.key = quant_nn.QuantLinear(config.hidden_size, self.all_head_size)
        self.value = quant_nn.QuantLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Additional quantizers that will be needed to quantize the inputs to the torch.matmul() operation in the
        # forward method. Since it's a simple operation and no quantized version of it exists, the inputs to this
        # operation could be manually quantized to realize a quantized mat-mul operation.
        self.matmul_q_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_k_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_v_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_a_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # Quantized matrix multiplication. Achieved by quantizing the inputs to torch.matmul().
        attention_scores = torch.matmul(
                self.matmul_q_input_quantizer(query_layer),
                self.matmul_k_input_quantizer(key_layer.transpose(-1, -2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Quantized matrix multiplication. Achieved by quantizing the inputs to torch.matmul().
        context_layer = torch.matmul(
                self.matmul_a_input_quantizer(attention_probs),
                self.matmul_v_input_quantizer(value_layer))

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
