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
Contains logic that captures BART HuggingFace models into ONNX models.
"""

from itertools import islice
from json import encoder
import os
from collections import OrderedDict

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
from transformers import BartForConditionalGeneration

# TRT-HuggingFace
from BART.BARTModelConfig import BARTModelTRTConfig
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
class BARTDecoderTorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of BART Decoder without support for loss.
        Decoder with lm-head attached.
        """

        def __init__(self, decoder, lm_head, final_logits_bias, config):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.bias = final_logits_bias
            self.config = config

        @staticmethod
        def _reorder_cache(past, beam_idx):
            return BartForConditionalGeneration._reorder_cache(past, beam_idx)

        def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=None, **kwargs):
            # cut decoder_input_ids if past is used
            if past is not None:
                input_ids = input_ids[:, -1:]

            ret = {
                "input_ids": input_ids,
                "encoder_hidden_states": kwargs["encoder_hidden_states"],
            }

            # To really enable KV cache in HuggingFace, these args must be passed. Just specifying use_cache = True in BartConfig is not enough. Also see the additional "past_key_values" fields in the forward() return below.
            if self.config.use_cache:
                ret["use_cache"] = use_cache
                ret["past_key_values"] = past

            return ret

        def forward(self, input_ids, encoder_hidden_states, **kwargs):
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            )

            sequence_output = decoder_outputs[0]
            self.bias = self.bias.to(sequence_output.device)
            logits = self.lm_head(sequence_output) + self.bias

            # temporary solution: force connection between encoder_hidden_states and outputs in KV cache mode, otherwise onnx.export elimiates it and cause inconsistency between non-KV cache & KV cache and also T5 & BART
            if self.config.use_cache:
                logits = logits.view(encoder_hidden_states.size(0),logits.size(1), logits.size(2)) # (batch_size, seq_len, vocab_size)

            if not kwargs.get("return_dict", False):
                return (logits,) + decoder_outputs[1:]

            return Seq2SeqLMOutput(logits=logits, past_key_values=decoder_outputs.past_key_values if self.config.use_cache else None,)

    def __init__(self, model, network_metadata):
        super().__init__(model, BARTDecoderConverter, network_metadata)


class BARTEncoderTorchFile(TorchModelFile):
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
        super().__init__(model, BARTEncoderConverter, network_metadata)


# ONNX File Encoding #
class BARTEncoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, BARTEncoderConverter, network_metadata)


class BARTDecoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, BARTDecoderConverter, network_metadata)


# TRT Engine File Encoding #
class BARTDecoderTRTEngine(TRTEngineFile):

    def __init__(self, model, network_metadata):
        super().__init__(model, BARTDecoderConverter, network_metadata)
        self.max_trt_workspace = BARTModelTRTConfig.MAX_DECODER_WORKSPACE_MB[network_metadata.variant]

    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16


class BARTEncoderTRTEngine(TRTEngineFile):

    def __init__(self, model, network_metadata):
        super().__init__(model, BARTEncoderConverter, network_metadata)
        self.max_trt_workspace = 2048

    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

# Converters #
class BARTDecoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(BARTDecoderTorchFile, BARTDecoderONNXFile, BARTDecoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Exports a given huggingface BART to decoder architecture only.

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            BARTDecoderONNXFile: ONNX decoder object.
        """

        input_ids = torch.tensor([[42] * 10])
        # Exporting the decoder requires a basic instance of the encoder
        # Create one temporarily
        simplified_encoder = BARTEncoderTorchFile.TorchModule(model.get_encoder())
        # Exports to ONNX
        decoder_with_lm_head_and_bias = BARTDecoderTorchFile.TorchModule(
            model.get_decoder(), model.lm_head, model.final_logits_bias, model.config
        )

        inputs = BARTModelTRTConfig.get_input_dims(network_metadata)["decoder"]
        outputs = BARTModelTRTConfig.get_output_dims(network_metadata)["decoder"]

        # Exports to ONNX
        opt_args={}

        version_major = int((torch.__version__).split('.')[0])
        version_minor = int((torch.__version__).split('.')[1])
        if version_major < 1 or (version_major == 1 and version_minor < 11):
            opt_args['use_external_data_format'] = True

        if not network_metadata.other.kv_cache:
            # This code allows for huggingface compatible torch class to use onnx exporter
            old_forward = decoder_with_lm_head_and_bias.forward
            def _export_forward(*args, **kwargs):
                result = old_forward(*args, **kwargs)
                return result[0]
            decoder_with_lm_head_and_bias.forward = _export_forward

            torch.onnx.export(
                decoder_with_lm_head_and_bias,
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
            decoder_output = decoder_with_lm_head_and_bias(input_ids[:,:-1], encoder_hidden_states) # decoder output at t-1 step (logits, past_key_values from 0 to t-1)
            past_key_values = decoder_output[1]

            decoder_root, decoder_fullname = os.path.split(output_fpath)
            # Split kv and non kv onnx into separate folders to avoid weight overlap
            non_kv_root = os.path.join(decoder_root, "non-kv")
            kv_root = os.path.join(decoder_root, "kv")
            decoder_name, decoder_ext = os.path.splitext(decoder_fullname)
            non_kv_fpath = os.path.join(non_kv_root, decoder_name + "-non-kv" + decoder_ext)
            kv_fpath = os.path.join(kv_root, decoder_fullname)

            # This code allows for huggingface compatible torch class to use onnx exporter (change just before onnx.export)
            old_forward = decoder_with_lm_head_and_bias.forward
            def _export_forward(input_ids, encoder_hidden_states, past_key_values):
                result = old_forward(input_ids, encoder_hidden_states, past_key_values=past_key_values)
                return (result[0], result[1])
            decoder_with_lm_head_and_bias.forward = _export_forward

            torch.onnx.export(
                decoder_with_lm_head_and_bias,
                (input_ids[:,-1:], encoder_hidden_states,past_key_values),
                # (1) input_ids should be the t token (last one) while past_key_values is 0 to t-1 caches
                # (2) since past_key_values is kwargs, ideally use "(input_ids[:,-1:], encoder_hidden_states, {"past_key_values": past_key_values})",
                # but onnx.export seems to unable to take kwargs properly (although PyTorch 1.11 claims it supports already).
                # Therefore, we need to wrap inside _export_forward() and make past_key_values indeed a kwargs
                kv_fpath,
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

            # dual-engine approach: also export non-kv onnx model. Note that this is different from the original "non-kv" model. This one traces the `use_cache` path and have present_key_values output
            def _export_forward(input_ids, encoder_hidden_states, use_cache):
                result = old_forward(input_ids, encoder_hidden_states, use_cache=use_cache)
                return (result[0], result[1])
            decoder_with_lm_head_and_bias.forward = _export_forward

            # inputs are same as non-kv model
            # outputs are same as kv model
            dict_inputs = inputs.get_dims()
            dict_inputs_non_kv = OrderedDict({k: dict_inputs[k] for k in ["input_ids", "encoder_hidden_states"]})
            inputs_non_kv = Dims(dict_inputs_non_kv)

            torch.onnx.export(
                decoder_with_lm_head_and_bias,
                (input_ids[:,-1:], encoder_hidden_states, True),
                non_kv_fpath,
                export_params=True,
                opset_version=12,
                input_names=inputs_non_kv.get_names(),
                output_names=outputs.get_names(),
                dynamic_axes={
                    **inputs_non_kv.get_torch_dynamic_axis_encoding(),
                    **outputs.get_torch_dynamic_axis_encoding(),
                },
                training=torch.onnx.TrainingMode.EVAL,
                **opt_args
            )

        if network_metadata.precision.fp16:
            G_LOGGER.debug("Clamping FP16 weights for BART")
            # move_t5_cast_op(output_fpath, output_fpath) # BART doesn't have T5's Add-Cast-Pow ordering issue
            if network_metadata.other.kv_cache:
                # both onnx files need clamp
                clamp_weights_onnx_to_fp16_bounds(non_kv_fpath, non_kv_fpath)
                clamp_weights_onnx_to_fp16_bounds(kv_fpath, kv_fpath)
            else:
                clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)

        return BARTDecoderONNXFile(output_fpath, network_metadata)


class BARTEncoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(BARTEncoderTorchFile, BARTEncoderONNXFile, BARTEncoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Exports a given huggingface BART to encoder architecture only.

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            Tuple[str]: Names of generated models
        """
        input_ids = torch.tensor([[42] * 10])
        simplified_encoder = BARTEncoderTorchFile.TorchModule(model.get_encoder())
        inputs = BARTModelTRTConfig.get_input_dims(network_metadata)["encoder"]
        outputs = BARTModelTRTConfig.get_output_dims(network_metadata)["encoder"]

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
            G_LOGGER.debug("Clamping FP16 weights for BART")
            # move_t5_cast_op(output_fpath, output_fpath) # BART doesn't have T5's Add-Cast-Pow ordering issue
            clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)

        return BARTEncoderONNXFile(output_fpath, network_metadata)
