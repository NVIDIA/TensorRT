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

# Huggingface
from transformers.modeling_outputs import Seq2SeqLMOutput

from Seq2Seq.export import (
    DecoderTorchFile,
    DecoderONNXFile,
    DecoderTRTEngine,
    DecoderConverter,
    Seq2SeqModelClass
)

# In gpt-j, torch.repeat_interleave will export SplitToSequence with subgraph. Use transformers==4.27.4 implementation as a patch to avoid it
# https://github.com/pytorch/pytorch/pull/100575 is merged and is expected to fix this issue in the next PyTorch release.
from transformers.models.gptj.modeling_gptj import rotate_every_two
import unittest.mock as mock
import torch

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    bs = m.shape[0]
    seq_len = m.shape[1]
    m = m.reshape(-1,1)  # flatten the matrix
    m = m.repeat(1,2)  # repeat all elements into the last dimension
    m = m.view(bs, seq_len, 1, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(tensor, sin, cos):
    sin = duplicate_interleave(sin[:, :, None, :])
    cos = duplicate_interleave(cos[:, :, None, :])
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

# Decoder File Encoding #
class GPT2DecoderTorchFile(DecoderTorchFile):
    class TorchModule(DecoderTorchFile.TorchModule):
        """
        A simplied definition of GPT2 Decoder without support for loss.
        Decoder with lm-head attached.
        """

        @mock.patch("transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb", apply_rotary_pos_emb)
        def forward(
            self,
            input_ids,
            attention_mask = None,
            encoder_outputs = None,
            past_key_values = None,
            use_cache = None,
            **kwargs,
        ):
            # Because GPT2 does not use attention mask and use position_ids to encode positional information,
            # generate position_ids is required as in https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/gpt2/modeling_gpt2.py#L1021
            # wrap position_ids generation inside
            if attention_mask is not None:
                # TODO: cumsum operation is known to have issue with TensorRT performance. Use an WAR, which assumes only left padding for GPT models.
                # position_ids = attention_mask.long().cumsum(-1) - 1
                num_paddings = torch.sum(attention_mask == 0, dim=1, keepdim=True).to(attention_mask.device)
                position_ids = torch.arange(0, attention_mask.shape[-1]).expand(attention_mask.shape).to(attention_mask.device) - num_paddings
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.long()

                input_length = input_ids.shape[-1]
                if past_key_values:
                    position_ids = position_ids[:, -input_length:].unsqueeze(-1)
            else:
                position_ids = None

            decoder_outputs = self.decoder(
                input_ids=input_ids,
                use_cache=use_cache,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )

            sequence_output = decoder_outputs[0]
            logits = self.lm_head(sequence_output) if self.lm_head is not None else sequence_output
            past_key_values = decoder_outputs[1] if use_cache else None

            return Seq2SeqLMOutput(
                logits=logits,
                past_key_values=past_key_values
            )

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = GPT2DecoderConverter
        super().__init__(model, network_metadata, default_converter)

class GPT2DecoderONNXFile(DecoderONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = GPT2DecoderConverter

        super().__init__(model, network_metadata, default_converter)

class GPT2DecoderTRTEngine(DecoderTRTEngine):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = GPT2DecoderConverter

        super().__init__(model, network_metadata, default_converter)

class GPT2DecoderConverter(DecoderConverter):
    def __init__(self,
        torch_class=GPT2DecoderTorchFile,
        onnx_class=GPT2DecoderONNXFile,
        trt_engine_class=GPT2DecoderTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

class GPT2ModelClass(Seq2SeqModelClass):
    """
    A class to track which class to use for each model type.
    """

    decoder_classes = {
        "torch": GPT2DecoderTorchFile,
        "onnx": GPT2DecoderONNXFile,
        "engine": GPT2DecoderTRTEngine
    }
