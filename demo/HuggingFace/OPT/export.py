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

class OPTLearnedPositionalEmbedding(torch.nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()
        seq_length = attention_mask.shape[1]

        # create positions depending on attention_mask
        num_paddings = torch.sum(attention_mask == 0, dim=1, keepdim=True).to(attention_mask.device)
        positions = torch.arange(0, seq_length).expand(attention_mask.shape).to(attention_mask.device) - num_paddings
        positions.masked_fill_(attention_mask == 0, 1)

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


# Decoder File Encoding #
class OPTDecoderTorchFile(DecoderTorchFile):
    class TorchModule(DecoderTorchFile.TorchModule):
        """
        A simplied definition of OPT Decoder without support for loss.
        Decoder with lm-head attached.
        """

        @mock.patch("transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding", OPTLearnedPositionalEmbedding)
        def forward(
            self,
            input_ids,
            attention_mask = None,
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
            default_converter = OPTDecoderConverter
        super().__init__(model, network_metadata, default_converter)

class OPTDecoderONNXFile(DecoderONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = OPTDecoderConverter

        super().__init__(model, network_metadata, default_converter)

class OPTDecoderTRTEngine(DecoderTRTEngine):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = OPTDecoderConverter

        super().__init__(model, network_metadata, default_converter)

class OPTDecoderConverter(DecoderConverter):
    def __init__(self,
        torch_class=OPTDecoderTorchFile,
        onnx_class=OPTDecoderONNXFile,
        trt_engine_class=OPTDecoderTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

class OPTModelClass(Seq2SeqModelClass):
    """
    A class to track which class to use for each model type.
    """

    decoder_classes = {
        "torch": OPTDecoderTorchFile,
        "onnx": OPTDecoderONNXFile,
        "engine": OPTDecoderTRTEngine
    }
