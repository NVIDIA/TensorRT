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
Contains logic that captures GPT2 HuggingFace models into ONNX models and TRT engines.
"""

from itertools import tee
import os
from collections import OrderedDict

# tensorrt
import tensorrt as trt

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch
from torch.nn import Module

# # huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GPT2Tokenizer

# TRT-HuggingFace
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from NNDF.networks import NetworkMetadata, Dims
from NNDF.logger import G_LOGGER
from NNDF.models import (
    TRTEngineFile,
    TorchModelFile,
    ONNXModelFile,
    ModelFileConverter,
)

class GPT2TorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of GPT2 with LM head.
        """

        def __init__(self, transformer, lm_head, config):
            super().__init__()
            self.transformer = transformer
            self.lm_head = lm_head
            self.config = config
            self.device = torch.device('cuda') # WAR to avoid beam search in framework
            self.main_input_name = "input_ids" # For better HuggingFace version compatibility

        def prepare_inputs_for_generation(self, input_ids, past = None, use_cache=None, **kwargs):
            # Todo (@pchadha): add position_ids, token_type_ids support
            # cut decoder_input_ids if past is used
            if past is not None:
                input_ids = input_ids[:, -1:]

            return {
                "input_ids": input_ids,
                "use_cache": use_cache,
                "past_key_values": past
            }

        def forward(self, input_ids, **kwargs):
            transformer_outputs = self.transformer(input_ids, **kwargs)
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)

            return CausalLMOutputWithPast(
                logits=lm_logits, 
                past_key_values=transformer_outputs.past_key_values
            )

        def _reorder_cache(self, past, beam_idx):
            """
            This function is used to re-order the :obj:`past_key_values` cache if
            :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
            called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
            """
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                for layer_past in past
            )

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata):
        super().__init__(model, GPT2Converter, network_metadata)


class GPT2ONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, GPT2Converter, network_metadata)


# TRT Engine File Encoding #
class GPT2TRTEngine(TRTEngineFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, GPT2Converter, network_metadata)

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

    def get_network_definition(self, network_definition):

        def pairwise(iterable):
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        indices = list(range(0, network_definition[1].num_layers))
        for i, i_next in pairwise(indices):
            l = network_definition[1].get_layer(i)
            l_next = network_definition[1].get_layer(i_next)

            if not all([l.get_output(i).is_execution_tensor for i in range(l.num_outputs)]):
                continue

            if l.get_output_type(0) != trt.float32:
                continue

            if l.type == trt.LayerType.ELEMENTWISE and l_next.type == trt.LayerType.REDUCE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.POW:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

                l_next.precision = trt.float32
                l_next.set_output_type(0, trt.float32)

        if self.network_metadata.precision.fp16:
            for i in range(network_definition[1].num_inputs):
                t = network_definition[1].get_input(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

            for i in range(network_definition[1].num_outputs):
                t = network_definition[1].get_output(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

        return network_definition

# Converters
class GPT2Converter(ModelFileConverter):
    def __init__(self):
        super().__init__(GPT2TorchFile, GPT2ONNXFile, GPT2TRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Exports a GPT2LMHead model to ONNX.

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            GPT2ONNXFile: ONNX GPT2 decoder object.
        """
        # Currently does not support exporting GPU models to onnx.
        device = model.device
        tokenizer = GPT2Tokenizer.from_pretrained(network_metadata.variant)
        input_ids = torch.tensor(
            [
                tokenizer.encode(
                    "Here is some text to encode Hello World", add_special_tokens=True
                )
            ]
        ).to(device)

        gpt2_model = GPT2TorchFile.TorchModule(
            model.transformer, model.lm_head, model.config
        )

        inputs = GPT2ModelTRTConfig.get_input_dims(network_metadata)[
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
        ]
        outputs = GPT2ModelTRTConfig.get_output_dims(network_metadata)[
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
        ]

        # Exports to ONNX
        opt_args={}

        version_major = int((torch.__version__).split('.')[0])
        version_minor = int((torch.__version__).split('.')[1])
        if version_major < 1 or (version_major == 1 and version_minor < 11):
            opt_args['use_external_data_format'] = True
        if not network_metadata.other.kv_cache:
            # This code allows for huggingface compatible torch class to use onnx exporter
            # This code regulates the number of output = 1 if non kv-cache mode is used.
            # Otherwise it will automatically output key value pairs
            old_forward = gpt2_model.forward
            def _export_forward(input_ids, **kwargs):
                result = old_forward(input_ids, use_cache = False, **kwargs)
                return result[0]
            gpt2_model.forward = _export_forward

            torch.onnx.export(
                gpt2_model,
                input_ids,
                output_fpath,
                opset_version=13,
                do_constant_folding=True,
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
            decoder_output = gpt2_model(input_ids, use_cache = True)
            past_key_values = decoder_output[1]

            # Exporting the kv cache engine
            old_forward = gpt2_model.forward
            def _export_forward(input_ids, past_key_values, **kwargs):
                result = old_forward(input_ids, past_key_values=past_key_values, use_cache=True, **kwargs)
                return (result[0], result[1])
            gpt2_model.forward = _export_forward

            torch.onnx.export(
                gpt2_model,
                (input_ids, past_key_values),
                output_fpath,
                opset_version=13,
                do_constant_folding=True,
                input_names=inputs.get_names(),
                output_names=outputs.get_names(),
                dynamic_axes={
                    **inputs.get_torch_dynamic_axis_encoding(),
                    **outputs.get_torch_dynamic_axis_encoding(),
                },
                training=torch.onnx.TrainingMode.EVAL,
                **opt_args
            )

        return GPT2ONNXFile(output_fpath, network_metadata)
