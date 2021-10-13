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
Contains logic that captures GPT2 HuggingFace models into ONNX models and TRT engines.
"""

from itertools import tee

# tensorrt
import tensorrt as trt

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch
from torch.nn import Module

# # huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import GPT2Tokenizer

# TRT-HuggingFace
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from NNDF.networks import NetworkMetadata
from NNDF.models import TRTEngineFile, TorchModelFile, ONNXModelFile, ModelFileConverter

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

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            # Todo (@pchadha): add position_ids, token_type_ids support
            return {
                "input_ids": input_ids,
            }

        def forward(self, input_ids, **kwargs):
            transformer_outputs = self.transformer(input_ids)
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)
            return CausalLMOutputWithCrossAttentions(logits=lm_logits)

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

    def use_strict_types(self):
        return self.network_metadata.precision.fp16

    def get_dynamic_shape_profiles(self):
        max_sequence_length = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[
            self.network_metadata.variant
        ]
        profile = Profile()
        profile.add(
            "input_ids",
            min=(1, 1),
            opt=(1, max_sequence_length // 2),
            max=(1, max_sequence_length),
        )
        return [profile]

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
        tokenizer = GPT2Tokenizer.from_pretrained(network_metadata.variant)
        input_ids = torch.tensor(
            [
                tokenizer.encode(
                    "Here is some text to encode Hello World", add_special_tokens=True
                )
            ]
        )

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
        torch.onnx._export(
            gpt2_model,
            input_ids,
            output_fpath,
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
        return GPT2ONNXFile(output_fpath, network_metadata)
