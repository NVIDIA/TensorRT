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
Contains logic that captures BART HuggingFace models into ONNX models.
"""

# torch
import torch
from torch.nn import Module

from Seq2Seq.export import (
    CrossAttnCacheGeneratorTorchFile,
    CrossAttnCacheGeneratorONNXFile,
    CrossAttnCacheGeneratorTRTEngine,
    CrossAttnCacheGeneratorConverter,
    Seq2SeqModelClass
)

class BARTCrossAttnCacheGeneratorConverter(CrossAttnCacheGeneratorConverter):
    def __init__(self):
        super().__init__(
            torch_class=BARTCrossAttnCacheGeneratorTorchFile,
            onnx_class=BARTCrossAttnCacheGeneratorONNXFile,
            trt_engine_class=BARTCrossAttnCacheGeneratorTRTEngine,
        )

# Cross Attention Cache Generator File Encoding
class BARTCrossAttnCacheGeneratorTorchFile(CrossAttnCacheGeneratorTorchFile):
    class TorchModule(Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, encoder_hidden_states):
            '''
            Use same but simplified code as HF modeling_t5.py to generate cross attention kv cache from provided encoder_hidden_states. This needs to be implemented for different models.
            '''
            present_key_values = ()
            bs, enc_len, hidden_size = encoder_hidden_states.shape
            num_heads = self.model.config.decoder_attention_heads
            d_kv = hidden_size // num_heads
            output_shape = (bs, num_heads, enc_len, d_kv)

            for layer in self.model.get_decoder().layers:
                dummy_hidden_states = torch.zeros(bs,1,hidden_size).to(self.model.device)
                _, _, cross_attn_present_key_value = layer.encoder_attn(
                    hidden_states=dummy_hidden_states,
                    key_value_states=encoder_hidden_states,
                    past_key_value=None,
                )
                present_key_values = present_key_values + (cross_attn_present_key_value[0].view(*output_shape), cross_attn_present_key_value[1].view(*output_shape))

            return present_key_values

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = BARTCrossAttnCacheGeneratorConverter

        super().__init__(model, network_metadata, default_converter)

class BARTCrossAttnCacheGeneratorONNXFile(CrossAttnCacheGeneratorONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = CrossAttnCacheGeneratorConverter

        super().__init__(model, network_metadata, default_converter)

class BARTCrossAttnCacheGeneratorTRTEngine(CrossAttnCacheGeneratorTRTEngine):

    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = CrossAttnCacheGeneratorConverter

        super().__init__(model, network_metadata, default_converter)

class BARTCrossAttnCacheGeneratorConverter(CrossAttnCacheGeneratorConverter):
    def __init__(self,
        torch_class=BARTCrossAttnCacheGeneratorTorchFile,
        onnx_class=BARTCrossAttnCacheGeneratorONNXFile,
        trt_engine_class=BARTCrossAttnCacheGeneratorTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

class BARTModelClass(Seq2SeqModelClass):

    cross_attn_cache_generator_classes = {
        "torch": BARTCrossAttnCacheGeneratorTorchFile,
        "onnx": BARTCrossAttnCacheGeneratorONNXFile,
        "engine": BARTCrossAttnCacheGeneratorTRTEngine
    }
