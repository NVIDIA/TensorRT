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

"""
Contains logic that captures HuggingFace models into ONNX models.
"""

from Seq2Seq.export import (
    DecoderTorchFile,
    DecoderONNXFile,
    DecoderTRTEngine,
    DecoderConverter,
    Seq2SeqModelClass
)

# Cumsum operation cannot be efficiently run in TensorRT's dependency, so WAR it.
import unittest.mock as mock
import torch
import math

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    '''
    A WAR to https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/bloom/modeling_bloom.py
    '''
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # WAR for cumsum for performance reason.
    num_paddings = torch.sum(attention_mask == 0, dim=1, keepdim=True).to(attention_mask.device)
    position_ids = torch.arange(0, seq_length).expand(attention_mask.shape).to(attention_mask.device) - num_paddings
    position_ids.masked_fill_(attention_mask == 0, 1)

    arange_tensor = position_ids[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


# Decoder File Encoding #
class BLOOMDecoderTorchFile(DecoderTorchFile):
    class TorchModule(DecoderTorchFile.TorchModule):
        """
        A simplied definition of BLOOM Decoder without support for loss.
        Decoder with lm-head attached.
        """

        @mock.patch("transformers.models.bloom.modeling_bloom.build_alibi_tensor", build_alibi_tensor)
        def forward(
            self,
            input_ids,
            attention_mask = None,
            encoder_outputs = None,
            past_key_values = None,
            use_cache = None,
            **kwargs,
        ):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLOOMDecoderConverter
        super().__init__(model, network_metadata, default_converter)

class BLOOMDecoderONNXFile(DecoderONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLOOMDecoderConverter

        super().__init__(model, network_metadata, default_converter)

class BLOOMDecoderTRTEngine(DecoderTRTEngine):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLOOMDecoderConverter

        super().__init__(model, network_metadata, default_converter)

class BLOOMDecoderConverter(DecoderConverter):
    def __init__(self,
        torch_class=BLOOMDecoderTorchFile,
        onnx_class=BLOOMDecoderONNXFile,
        trt_engine_class=BLOOMDecoderTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

class BLOOMModelClass(Seq2SeqModelClass):
    """
    A class to track which class to use for each model type.
    """

    decoder_classes = {
        "torch": BLOOMDecoderTorchFile,
        "onnx": BLOOMDecoderONNXFile,
        "engine": BLOOMDecoderTRTEngine
    }
