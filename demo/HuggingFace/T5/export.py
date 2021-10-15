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

"""
Contains logic that captures T5 HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""

from itertools import islice

# tensorrt
import tensorrt as trt

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch
from torch.nn import Module

# huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

# TRT-HuggingFace
from T5.T5ModelConfig import T5ModelTRTConfig
from NNDF.tensorrt_utils import clamp_weights_onnx_to_fp16_bounds, move_t5_cast_op
from NNDF.networks import NetworkMetadata
from NNDF.logger import G_LOGGER
from NNDF.models import (
    TRTEngineFile,
    TorchModelFile,
    ONNXModelFile,
    ModelFileConverter,
)

def add_extra_fp32(network_definition):
    """
    Force operations involved in layer norm to run in FP32 precision.
    """
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
        END_OFFSET = 10
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

# Torch File Encoding #
class T5DecoderTorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of T5 Decoder without support for loss.
        Decoder with lm-head attached.
        """

        def __init__(self, decoder, lm_head, config):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {
                "input_ids": input_ids,
                "encoder_hidden_states": kwargs["encoder_hidden_states"],
            }

        def forward(self, input_ids, encoder_hidden_states, **kwargs):
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            )

            # self.config.d_model ** -0.5 for rescaling output on vocab.
            # as seen in https://huggingface.co/transformers/_modules/transformers/models/t5/modeling_t5.html#T5ForConditionalGeneration
            sequence_output = decoder_outputs[0] * self.config.d_model ** -0.5
            logits = self.lm_head(sequence_output)
            if not kwargs.get("return_dict", False):
                return (logits,) + decoder_outputs[1:]

            return Seq2SeqLMOutput(logits=logits)

    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)


class T5EncoderTorchFile(TorchModelFile):
    """Creation of a class to output only the last hidden state from the encoder."""

    class TorchModule(Module, GenerationMixin):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, *input, **kwargs):
            return self.encoder(*input, **kwargs)[0]

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)


# ONNX File Encoding #
class T5EncoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)


class T5DecoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)


# TRT Engine File Encoding #
class T5DecoderTRTEngine(TRTEngineFile):
    DEFAULT_TRT_WORKSPACE_MB = 3072

    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)

    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def get_dynamic_shape_profiles(self):
        max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[
            self.network_metadata.variant
        ]
        profile = Profile()
        profile.add(
            "input_ids",
            min=(1, 1),
            opt=(1, max_sequence_length // 2),
            max=(1, max_sequence_length),
        )
        profile.add(
            "encoder_hidden_states",
            min=(1, 1, max_sequence_length),
            opt=(1, max_sequence_length // 2, max_sequence_length),
            max=(1, max_sequence_length, max_sequence_length),
        )
        return [profile]

    def use_strict_types(self):
        return self.network_metadata.precision.fp16


class T5EncoderTRTEngine(TRTEngineFile):
    DEFAULT_TRT_WORKSPACE_MB = 2048

    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)

    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def get_dynamic_shape_profiles(self):
        max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[
            self.network_metadata.variant
        ]
        return [
            Profile().add(
                "input_ids",
                min=(1, 1),
                opt=(1, max_sequence_length // 2),
                max=(1, max_sequence_length),
            )
        ]

    def use_strict_types(self):
        return self.network_metadata.precision.fp16

# Converters #
class T5DecoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5DecoderTorchFile, T5DecoderONNXFile, T5DecoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Exports a given huggingface T5 to decoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            T5DecoderONNXFile: ONNX decoder object.
        """

        input_ids = torch.tensor([[42] * 10])
        # Exporting the decoder requires a basic instance of the encoder
        # Create one temporarily
        simplified_encoder = T5EncoderTorchFile.TorchModule(model.encoder)
        # Exports to ONNX
        decoder_with_lm_head = T5DecoderTorchFile.TorchModule(
            model.decoder, model.lm_head, model.config
        )

        # This code allows for huggingface compatible torch class to use onnx exporter
        old_forward = decoder_with_lm_head.forward
        def _export_forward(*args, **kwargs):
            result = old_forward(*args, **kwargs)
            return result[0]

        decoder_with_lm_head.forward = _export_forward

        inputs = T5ModelTRTConfig.get_input_dims(network_metadata)["decoder"]
        outputs = T5ModelTRTConfig.get_output_dims(network_metadata)["decoder"]

        torch.onnx.export(
            decoder_with_lm_head,
            (input_ids, simplified_encoder(input_ids)),
            output_fpath,
            export_params=True,
            opset_version=12,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
            use_external_data_format=True
        )

        if network_metadata.precision.fp16:
            G_LOGGER.debug("Clamping FP16 weights for T5")
            move_t5_cast_op(output_fpath, output_fpath)
            clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)

        return T5DecoderONNXFile(output_fpath, network_metadata)


class T5EncoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5EncoderTorchFile, T5EncoderONNXFile, T5EncoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Exports a given huggingface T5 to encoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            Tuple[str]: Names of generated models
        """
        input_ids = torch.tensor([[42] * 10])
        simplified_encoder = T5EncoderTorchFile.TorchModule(model.encoder)
        inputs = T5ModelTRTConfig.get_input_dims(network_metadata)["encoder"]
        outputs = T5ModelTRTConfig.get_output_dims(network_metadata)["encoder"]

        # Exports to ONNX
        torch.onnx._export(
            simplified_encoder,
            input_ids,
            output_fpath,
            export_params=True,
            opset_version=12,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
            use_external_data_format=True
        )

        if network_metadata.precision.fp16:
            G_LOGGER.debug("Clamping FP16 weights for T5")
            move_t5_cast_op(output_fpath, output_fpath)            
            clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)

        return T5EncoderONNXFile(output_fpath, network_metadata)
