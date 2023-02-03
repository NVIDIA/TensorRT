#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List

from json import encoder
import os
from collections import OrderedDict

# tensorrt
import tensorrt as trt
from tensorrt import PreviewFeature

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch
from torch.nn import Module

# huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import T5ForConditionalGeneration

# TRT-HuggingFace
from T5.T5ModelConfig import T5ModelTRTConfig
from NNDF.tensorrt_utils import clamp_weights_onnx_to_fp16_bounds, move_t5_cast_op
from NNDF.networks import NetworkMetadata, Precision, Dims
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

# Torch File Encoding #
class T5DecoderTorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of T5 Decoder without support for loss.
        Decoder with lm-head attached.
        """

        def __init__(self, decoder, lm_head, config, is_trt = False):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config
            self.device = "cuda" # HuggingFace's beam search requires to set self.device. Set it to avoid application crash
            # Use hardcoded value to extend compatibility with older HF versions.
            self.main_input_name = "input_ids"
            # trt uses cached and precomputed cross attention vs. framework uses the entire kv cache as output. Need to treat them differently.
            self.is_trt = is_trt

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            use_cache=None,
            **kwargs
        ):
            # cut decoder_input_ids if past is used
            if past is not None:
                input_ids = input_ids[:, -1:]

            return {
                "input_ids": input_ids,
                "encoder_hidden_states": kwargs["encoder_outputs"].last_hidden_state,
                "use_cache": use_cache,
                "past_key_values": past
            }

        def forward(
            self,
            input_ids,
            encoder_hidden_states,
            use_cache = None,
            past_key_values = None,
            return_dict = None,
            **kwargs,
        ):
            # self.decoder is the HuggingFace t5 decoder
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=return_dict,
                **kwargs
            )

            # self.config.d_model ** -0.5 for rescaling output on vocab.
            # as seen in https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration
            sequence_output = decoder_outputs[0] * self.config.d_model ** -0.5
            logits = self.lm_head(sequence_output)
            if use_cache:
                if self.is_trt:
                    past_key_values = ()
                    past_key_values_output = decoder_outputs[1]
                    for layer_past_states in past_key_values_output:
                        past_key_values = past_key_values + (layer_past_states[:2],)
                else:
                    past_key_values = decoder_outputs[1]

            if not return_dict:
                return (logits, past_key_values)

            return Seq2SeqLMOutput(
                logits=logits,
                past_key_values=past_key_values
            )

    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)

class T5DecoderCrossAttentionKVGenerator(Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, encoder_hidden_states):
        '''
        Use same but simplified code as HF modeling_t5.py to generate cross attention kv cache from provided encoder_hidden_states
        '''
        present_key_values = ()
        for layer_module in self.decoder.block:
            # hidden_states and position_bias are required for the forward call, but irrelevant of cross attention kv cache calculation, so generate dummy variables
            dummy_hidden_states = torch.zeros(1,1)
            dummy_position_bias = torch.zeros(1, layer_module.layer[1].EncDecAttention.n_heads, 1, encoder_hidden_states.shape[1])
            cross_attention_outputs = layer_module.layer[1](
                hidden_states=dummy_hidden_states,
                key_value_states=encoder_hidden_states,
                use_cache=True,
                past_key_value=None,
                position_bias=dummy_position_bias
            )
            present_key_values = present_key_values + cross_attention_outputs[1]

        return present_key_values

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class T5EncoderTorchFile(TorchModelFile):
    """Creation of a class to output only the last hidden state from the encoder."""

    class TorchModule(Module, GenerationMixin):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            # Use hardcoded value to extend compatibility with older HF versions.
            self.main_input_name = "input_ids"

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

    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)
        self.max_trt_workspace = T5ModelTRTConfig.MAX_DECODER_WORKSPACE_MB[network_metadata.variant]


    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16


class T5EncoderTRTEngine(TRTEngineFile):

    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)
        self.max_trt_workspace = T5ModelTRTConfig.MAX_ENCODER_WORKSPACE_MB[network_metadata.variant]

    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def use_obey_precision_constraints(self):
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
            model.decoder, model.lm_head, model.config, is_trt = True
        )

        inputs = T5ModelTRTConfig.get_input_dims(network_metadata)["decoder"]
        outputs = T5ModelTRTConfig.get_output_dims(network_metadata)["decoder"]

        # Exports to ONNX
        opt_args={}

        version_major = int((torch.__version__).split('.')[0])
        version_minor = int((torch.__version__).split('.')[1])
        if version_major < 1 or (version_major == 1 and version_minor < 11):
            opt_args['use_external_data_format'] = True

        if not network_metadata.other.kv_cache:
            # This code allows for huggingface compatible torch class to use onnx exporter
            old_forward = decoder_with_lm_head.forward
            def _export_forward(input_ids, encoder_hidden_states, **kwargs):
                result = old_forward(input_ids, encoder_hidden_states, use_cache=False, **kwargs)
                return result[0]
            decoder_with_lm_head.forward = _export_forward

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
                training=torch.onnx.TrainingMode.EVAL,
                **opt_args
            )
        else:
            encoder_hidden_states = simplified_encoder(input_ids)
            kv_decoder_input_ids = input_ids[:,-1:]
            decoder_output = decoder_with_lm_head.decoder(input_ids=kv_decoder_input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=True, past_key_values=None) # decoder output at t-1 step (logits, past_key_values from 0 to t-1)
            past_key_values = decoder_output[1]
            # This code allows for huggingface compatible torch class to use onnx exporter (change just before onnx.export)
            old_forward = decoder_with_lm_head.forward
            def _export_forward(input_ids, encoder_hidden_states, past_key_values):
                result = old_forward(input_ids, encoder_hidden_states, past_key_values=past_key_values, use_cache=True)
                return result
            decoder_with_lm_head.forward = _export_forward

            torch.onnx.export(
                decoder_with_lm_head,
                (kv_decoder_input_ids, encoder_hidden_states, past_key_values),
                output_fpath,
                export_params=True,
                opset_version=12,
                input_names=inputs[1].get_names(),
                output_names=outputs[1].get_names(),
                dynamic_axes={
                    **inputs[1].get_torch_dynamic_axis_encoding(),
                    **outputs[1].get_torch_dynamic_axis_encoding(),
                },
                training=torch.onnx.TrainingMode.EVAL,
                **opt_args
            )

            cross_attention_kv_generator = T5DecoderCrossAttentionKVGenerator(decoder_with_lm_head.decoder)
            decoder_folder, decoder_name = os.path.split(output_fpath)
            decoder_name, decoder_ext = os.path.splitext(decoder_name)
            output_fpath_kv_generator_folder = os.path.join(decoder_folder, "cross_attention_kv_generator")
            os.makedirs(output_fpath_kv_generator_folder, exist_ok = True)
            output_fpath_kv_generator = os.path.join(output_fpath_kv_generator_folder, decoder_name + "-cross_attention_kv_generator" + decoder_ext)
            torch.onnx.export(
                cross_attention_kv_generator,
                (encoder_hidden_states),
                output_fpath_kv_generator,
                export_params=True,
                opset_version=12,
                input_names=inputs[0].get_names(),
                output_names=outputs[0].get_names(),
                dynamic_axes={
                    **inputs[0].get_torch_dynamic_axis_encoding(),
                    **outputs[0].get_torch_dynamic_axis_encoding(),
                },
                training=torch.onnx.TrainingMode.EVAL,
                **opt_args
            )

        if network_metadata.precision.fp16:
            G_LOGGER.debug("Clamping FP16 weights for T5")
            move_t5_cast_op(output_fpath, output_fpath)
            clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)
            if network_metadata.other.kv_cache:
                move_t5_cast_op(output_fpath_kv_generator, output_fpath_kv_generator)
                clamp_weights_onnx_to_fp16_bounds(output_fpath_kv_generator, output_fpath_kv_generator)

        return T5DecoderONNXFile(output_fpath, network_metadata)


class T5EncoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5EncoderTorchFile, T5EncoderONNXFile, T5EncoderTRTEngine)

    def onnx_to_trt(
        self, output_fpath: str, input_fpath: str, network_metadata: NetworkMetadata, profiles: List[Profile], preview_features: List[PreviewFeature]
    ):
        """
        Override onnx_to_trt function from base.
        Workaround: T5-base and T5-large are too large and cause FP16 to overflow. Encoder should not use FP16 tactics even in FP16 mode.
        The perf decreases by less than 10% end-to-end. Usage with TRT is still substantial compared to frameworks.
        """
        # Force encoder to FP32 only if variants are anything larger than small
        # because of overflow and underflow issues
        if network_metadata.precision.fp16 and network_metadata.variant != "t5-small":
            network_metadata_cp_dct = network_metadata._asdict()
            del network_metadata_cp_dct["precision"]
            network_metadata = NetworkMetadata(**network_metadata_cp_dct, precision=Precision(fp16=False))

        return super().onnx_to_trt(output_fpath, input_fpath, network_metadata, profiles, preview_features)

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
        opt_args={}

        version_major = int((torch.__version__).split('.')[0])
        version_minor = int((torch.__version__).split('.')[1])
        if version_major < 1 or (version_major == 1 and version_minor < 11):
            opt_args['use_external_data_format'] = True
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
            training=torch.onnx.TrainingMode.EVAL,
            **opt_args
        )

        if network_metadata.precision.fp16:
            G_LOGGER.debug("Clamping FP16 weights for T5")
            move_t5_cast_op(output_fpath, output_fpath)
            clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)

        return T5EncoderONNXFile(output_fpath, network_metadata)
