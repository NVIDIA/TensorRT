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
Contains logic that captures T5 HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""


# tensorrt
import tensorrt as trt

# torch
import torch
from torch.nn import Module

# TRT-HuggingFace
from NNDF.tensorrt_utils import OnnxProcessOperation, process_onnx
from NNDF.networks import Precision

from Seq2Seq.export import (
    EncoderTorchFile,
    EncoderONNXFile,
    EncoderTRTEngine,
    EncoderConverter,
    DecoderTorchFile,
    DecoderONNXFile,
    DecoderTRTEngine,
    DecoderConverter,
    CrossAttnCacheGeneratorTorchFile,
    CrossAttnCacheGeneratorONNXFile,
    CrossAttnCacheGeneratorTRTEngine,
    CrossAttnCacheGeneratorConverter,
    Seq2SeqModelClass
)

class T5DecoderTorchFile(DecoderTorchFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = T5DecoderConverter

        super().__init__(model, network_metadata, default_converter)

class T5DecoderONNXFile(DecoderONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = T5DecoderConverter

        super().__init__(model, network_metadata, default_converter)

class T5DecoderTRTEngine(DecoderTRTEngine):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = T5DecoderConverter

        super().__init__(model, network_metadata, default_converter)

    def set_layer_precisions(self, network_definition):
        """
        Force operations involved in layer norm to run in FP32 precision.
        """
        if not self.network_metadata.precision.fp16:
            return network_definition

        pow_ops = {}
        for layer_index, layer in enumerate(network_definition[1]):
            if layer.type == trt.LayerType.IDENTITY:
                all_fp32 = all([layer.output_type_is_set(o) and layer.get_output_type(o) == trt.float32 for o in range(layer.num_outputs)])
                if all_fp32:
                    if layer.get_input(0).dtype == trt.float32:
                        layer.precision = trt.float32

            if layer.type == trt.LayerType.ELEMENTWISE:
                layer.__class__ = getattr(trt, "IElementWiseLayer")
                if layer.op == trt.ElementWiseOperation.POW:
                    pow_ops[layer] = layer_index
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)

        for _, index in pow_ops.items():
            # Iterate from few layers before pow to include residual add and cast op.
            # Iterate till 10 layers after pow op to include all operations included in layer norm.
            START_OFFSET = 4
            END_OFFSET = 12
            for i in range(index-START_OFFSET, index+END_OFFSET):
                l = network_definition[1].get_layer(i)
                if l.type == trt.LayerType.REDUCE:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

                if l.type == trt.LayerType.ELEMENTWISE:
                    l.__class__ = getattr(trt, "IElementWiseLayer")
                    if l.op == trt.ElementWiseOperation.SUM:
                        l.precision = trt.float32
                        l.set_output_type(0, trt.float32)

                if l.type == trt.LayerType.UNARY:
                    l.__class__ = getattr(trt, "IUnaryLayer")
                    if l.op == trt.UnaryOperation.SQRT:
                        l.precision = trt.float32
                        l.set_output_type(0, trt.float32)

                if l.type == trt.LayerType.ELEMENTWISE:
                    l.__class__ = getattr(trt, "IElementWiseLayer")
                    if l.op == trt.ElementWiseOperation.DIV:
                        l.precision = trt.float32
                        l.set_output_type(0, trt.float32)

                if l.type == trt.LayerType.ELEMENTWISE:
                    l.__class__ = getattr(trt, "IElementWiseLayer")
                    if l.op == trt.ElementWiseOperation.PROD:
                        l.precision = trt.float32
                        l.set_output_type(0, trt.float32)

        return network_definition

class T5DecoderConverter(DecoderConverter):
    def __init__(self,
        torch_class=T5DecoderTorchFile,
        onnx_class=T5DecoderONNXFile,
        trt_engine_class=T5DecoderTRTEngine
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

    def post_process_onnx(self, output_fpath):
        process_onnx([OnnxProcessOperation.MOVE_CAST_OP, OnnxProcessOperation.CLAMP_WEIGHTS], output_fpath, output_fpath)

class T5EncoderTorchFile(EncoderTorchFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = T5EncoderConverter

        super().__init__(model, network_metadata, default_converter)

class T5EncoderONNXFile(EncoderONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = T5EncoderConverter

        super().__init__(model, network_metadata, default_converter)

class T5EncoderTRTEngine(EncoderTRTEngine):

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = T5EncoderConverter

        super().__init__(model, network_metadata, default_converter)

    def get_network_definition(self, network_definition):
        # T5Encoder tend to overflow, so we still build fp32 engine for encoder
        return network_definition

    def use_obey_precision_constraints(self):
        return False

class T5EncoderConverter(EncoderConverter):
    def __init__(self,
        torch_class=T5EncoderTorchFile,
        onnx_class=T5EncoderONNXFile,
        trt_engine_class=T5EncoderTRTEngine
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)
    
    def onnx_to_trt(
        self,
        output_fpath,
        input_fpath,
        network_metadata,
        profiles,
        preview_features,
        nvtx_verbose,
        timing_cache,
    ):
        # Needs to overwrite precision for onnx_to_trt
        network_metadata = network_metadata._replace(precision=Precision(fp16=False))
        return super().onnx_to_trt(
            output_fpath, 
            input_fpath, 
            network_metadata,
            profiles,
            preview_features,
            nvtx_verbose,
            timing_cache
        )

class T5CrossAttnCacheGeneratorConverter(CrossAttnCacheGeneratorConverter):
    def __init__(self):
        super().__init__(
            torch_class=T5CrossAttnCacheGeneratorTorchFile,
            onnx_class=T5CrossAttnCacheGeneratorONNXFile,
            trt_engine_class=T5CrossAttnCacheGeneratorTRTEngine,
        )


# Cross Attention Cache Generator File Encoding
class T5CrossAttnCacheGeneratorTorchFile(CrossAttnCacheGeneratorTorchFile):
    class TorchModule(Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, encoder_hidden_states):
            '''
            Use same but simplified code as HF modeling_t5.py to generate cross attention kv cache from provided encoder_hidden_states. This needs to be implemented for different models.
            '''
            bs = encoder_hidden_states.shape[0]
            enc_len = encoder_hidden_states.shape[1]
            hidden_size = encoder_hidden_states.shape[2]
            present_key_values = ()
            output_shape = (bs, self.model.config.num_heads, enc_len, self.model.config.d_kv)

            for layer_module in self.model.get_decoder().block:
                # hidden_states and position_bias are required for the forward call, but irrelevant of cross attention kv cache calculation, so generate dummy variables
                dummy_hidden_states = torch.zeros(bs, 1, hidden_size).to(self.model.device)
                dummy_position_bias = torch.zeros(bs, self.model.config.num_heads, 1, enc_len).to(self.model.device)
                cross_attention_outputs = layer_module.layer[1].EncDecAttention(
                    hidden_states=dummy_hidden_states,
                    key_value_states=encoder_hidden_states,
                    use_cache=True,
                    past_key_value=None,
                    position_bias=dummy_position_bias
                )
                # Force kv generator to generate the specified shape
                present_key_values = present_key_values + (cross_attention_outputs[1][0].view(*output_shape),cross_attention_outputs[1][1].view(*output_shape))

            return present_key_values

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = T5CrossAttnCacheGeneratorConverter

        super().__init__(model, network_metadata, default_converter)

class T5CrossAttnCacheGeneratorONNXFile(CrossAttnCacheGeneratorONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = CrossAttnCacheGeneratorConverter

        super().__init__(model, network_metadata, default_converter)

class T5CrossAttnCacheGeneratorTRTEngine(CrossAttnCacheGeneratorTRTEngine):

    def __init__(self, model, network_metadata = None, default_converter = None):

        if default_converter is None:
            default_converter = CrossAttnCacheGeneratorConverter

        super().__init__(model, network_metadata, default_converter)

class T5CrossAttnCacheGeneratorConverter(CrossAttnCacheGeneratorConverter):
    def __init__(self,
        torch_class=T5CrossAttnCacheGeneratorTorchFile,
        onnx_class=T5CrossAttnCacheGeneratorONNXFile,
        trt_engine_class=T5CrossAttnCacheGeneratorTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

class T5ModelClass(Seq2SeqModelClass):
    decoder_classes = {
        "torch": T5DecoderTorchFile,
        "onnx": T5DecoderONNXFile,
        "engine": T5DecoderTRTEngine,
    }

    encoder_classes = {
        "torch": T5EncoderTorchFile,
        "onnx": T5EncoderONNXFile,
        "engine": T5EncoderTRTEngine,
    }

    cross_attn_cache_generator_classes = {
        "torch": T5CrossAttnCacheGeneratorTorchFile,
        "onnx": T5CrossAttnCacheGeneratorONNXFile,
        "engine": T5CrossAttnCacheGeneratorTRTEngine
    }
