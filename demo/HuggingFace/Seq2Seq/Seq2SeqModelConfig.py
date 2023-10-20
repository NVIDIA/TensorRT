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

from collections import OrderedDict
from itertools import product
from typing import Dict
import sys
import os

from NNDF.networks import Precision, NetworkMetadata, NNConfig, Dims

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

class Seq2SeqModelTRTConfig(NNConfig):

    TARGET_MODELS = []

    attribute_map = {
        "num_heads": ['n_head', 'num_heads', 'decoder_attention_heads', 'num_attention_heads'],
        "num_decoder_layers": ['n_layer', 'num_layers', 'decoder_layers', 'num_hidden_layers'],
        "hidden_size": ['hidden_size', 'n_embd', 'd_model', 'n_embed'],
        "n_positions": ['n_positions', "max_position_embeddings"],
    }

    NETWORK_FULL_NAME = "full"
    NETWORK_DECODER_SEGMENT_NAME = "decoder"
    NETWORK_ENCODER_SEGMENT_NAME = "encoder"
    NETWORK_CROSS_ATTENTION_CACHE_GENERATOR_NAME = "cross_attn_cache_generator"

    def __init__(
        self,
        network_name = "Seq2Seq",
        metadata = None,
        **kwargs
    ):
        if metadata is not None:
            self.metadata = metadata
            self.variant = metadata.variant
            self.use_cache = metadata.use_cache
            self.fp16 = metadata.precision.fp16
            self.batch_size = metadata.batch_size
            self.num_beams = metadata.num_beams
        else:
            # User defined config
            self.variant = kwargs.pop("variant", None)
            self.use_cache = kwargs.pop("use_cache", True)
            self.fp16 = kwargs.pop("fp16", False)
            self.num_beams = kwargs.pop("num_beams", 1)
            self.batch_size = kwargs.pop("batch_size", 1)
            self.metadata = self.get_metadata_from_config()

        self.use_fp32_encoder = False

        precision_fp16 = [False, True]
        use_caches = [False, True]
        variants = []
        for variant, fp16, use_cache in product(
            self.TARGET_MODELS, precision_fp16, use_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16),
                    use_cache=use_cache,
                    num_beams=1,
                    batch_size=1,
                )
            )

        self.use_mask = False
        # Use this flag to load torch model if and only if benchmarking seqlen > model n_positions
        self.ignore_mismatched_sizes = False

        # Use this flag to control whether there are vision encoder outputs (named as image_embeds) consumed as inputs
        self.consume_image_embeds = False

        super().__init__(network_name, variants=variants)

    def from_hf_config(self, hf_config, model_max_len = None):
        """
        Set up config from HuggingFace Config.
        Some model's n_positions is too long so TRT will build engines that is unrealistic for real-world cases.
        Therefore we need to constrain the max output length for some models like GPT, BLOOM, OPT, etc.
        """
        user_define_variables = ["num_beams", "use_cache", "torch_dtype", "max_length", "min_length"]
        self.hf_config = hf_config
        # Initialize all the fields from hf config.
        hf_config_dict = self.hf_config.to_dict()
        for k in hf_config_dict:
            if k not in user_define_variables:
                self.__setattr__(k, hf_config_dict[k])

        # Process required config to run TRT, but they may have different names in HuggingFace
        for variable in Seq2SeqModelTRTConfig.attribute_map:
            for name in Seq2SeqModelTRTConfig.attribute_map[variable]:
                if hasattr(self.hf_config, name):
                    self.__setattr__(variable, getattr(self.hf_config, name))

            if not hasattr(self, variable):
                if variable == "n_positions":
                    self.n_positions = self.hidden_size
                else:
                    raise ValueError("{} is not found in hf_config. Please ensure your model config has one of: {}".format(variable,", ".join(Seq2SeqModelTRTConfig.attribute_map[variable])))

        # Parse min and max length from task_specific_params
        self.min_length = 65536
        self.max_length = 0
        if hasattr(self.hf_config, "task_specific_params"):
            task_specific_params = self.hf_config.task_specific_params
            if task_specific_params is not None:
                for task in task_specific_params:
                    if "min_length" in task_specific_params[task]:
                        self.min_length = min(self.min_length, task_specific_params[task]["min_length"])
                    else:
                        # If min length is not set, set to 0
                        self.min_length = 0

                    if "max_length" in task_specific_params[task]:
                        self.max_length = max(self.max_length, task_specific_params[task]["max_length"])

        if self.max_length == 0:
            self.max_length = self.n_positions
        if self.min_length == 65536:
            self.min_length = 0

        if model_max_len:
            self.max_length = min(self.max_length, model_max_len)

        # These variables are used to control generation.
        self.min_output_length = self.min_length
        self.max_output_length = self.max_length

        # HuggingFace assume that max_length is for both input and output.
        # However, for benchmarking mode, they may have a difference.
        # These variables are used to control generated binding shapes
        self.max_input_length = self.max_length
        self.opt_input_length = self.max_input_length // 2
        self.opt_output_length = self.max_output_length // 2

        # These variables are only used to build TRT engines
        self.max_input_profile_length = self.max_input_length
        self.max_output_profile_length = self.max_output_length

        self.d_kv = hf_config_dict.pop("d_kv", self.hidden_size // self.num_heads)

        # For decoder-only models, we have context phase and generation phase, which will use different TRT profiles
        # So we need to maintain dynamic shape for onnx export, but we will fix seq len = 1 during TRT engine generation.
        self.decoder_dims = 1 if (self.use_cache and self.is_encoder_decoder) else Dims.SEQUENCE
        self.max_decoder_length = 1 if (self.use_cache and self.is_encoder_decoder) else self.max_output_length
        self.expand_size = self._compute_expand_size(self.batch_size, self.num_beams)

        # Assume that all inputs have size 1, can adjust later.
        self.input_case_size = 1

    def _compute_expand_size(self, batch_size, num_beams):
        """
        Computes expand size for beam search.
        """
        return batch_size * num_beams

    def set_model_classes(self, model_classes):
        """
        Different frameworks may use different Torch/ONNX/TRT Engine model classes.
        Currently it is hard to implement it into each classes. Therefore create a separate class in export.py for record
        """

        self.encoder_classes = model_classes.encoder_classes
        self.decoder_classes = model_classes.decoder_classes
        self.cross_attn_cache_generator_classes = model_classes.cross_attn_cache_generator_classes

    def set_generation_config(self, generation_config):
        """
        Set generation config for using HF `generate` to decode.
        """
        self.generation_config = generation_config

    def to_dict(self):
        return self.hf_config.to_dict()

    def get_metadata_from_config(self) -> NetworkMetadata:
        return NetworkMetadata(
            variant=self.variant,
            precision=Precision(self.fp16),
            use_cache=self.use_cache,
            num_beams=self.num_beams,
            batch_size=self.batch_size,
        )

    def get_python_requirements(self):
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.29.2")
        return base_requirements

    def get_network_segments(self):
        """
        Returns exportable segments for the given network.
        Used in the case where a single network needs to
        be exported into multiple parts.
        """

        # TODO: Currently just measure decoder performance and full inference only.
        # In current framework, it is hard to add encoder inference in a non-redundant way.
        self.network_segments = [
            self.NETWORK_DECODER_SEGMENT_NAME,
            self.NETWORK_FULL_NAME,
        ]

        return self.network_segments

    def _get_encoder_inputs(self):
        encoder_inputs = OrderedDict({
            "input_ids": (Dims.BATCH, Dims.SEQUENCE),
        })
        if self.use_mask:
            encoder_inputs["attention_mask"] = (Dims.BATCH, Dims.SEQUENCE)
        return encoder_inputs

    def _get_encoder_outputs(self):
        return {"encoder_hidden_states":(Dims.BATCH, Dims.SEQUENCE, "encoder_hidden_size")}

    def _get_cache_inputs(self):
        decoder_cache_inputs = OrderedDict({})
        for i in range(self.num_decoder_layers):
            decoder_cache_inputs["past_key_values.{}.self.key".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("past_decoder"), self.d_kv)
            decoder_cache_inputs["past_key_values.{}.self.value".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("past_decoder"), self.d_kv)
            if self.is_encoder_decoder:
                decoder_cache_inputs["past_key_values.{}.cross.key".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("encoder"), self.d_kv)
                decoder_cache_inputs["past_key_values.{}.cross.value".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("encoder"), self.d_kv)

        return decoder_cache_inputs

    def _get_self_attention_cache_outputs(self):
        self_attention_cache_outputs = OrderedDict({})
        for i in range(self.num_decoder_layers):
            self_attention_cache_outputs["present_key_values.{}.self.key".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("present_decoder"), self.d_kv)
            self_attention_cache_outputs["present_key_values.{}.self.value".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("present_decoder"), self.d_kv)
        return self_attention_cache_outputs

    def _get_decoder_inputs(self):
        decoder_inputs = OrderedDict({
            "input_ids": (Dims.BATCH, self.decoder_dims),
        })
        if self.use_mask:
            decoder_inputs["attention_mask"] = (Dims.BATCH, Dims.create_new_sequence_dim("past_attention_mask"))

        if self.is_encoder_decoder:
            decoder_inputs["encoder_hidden_states"] = (Dims.BATCH, Dims.create_new_sequence_dim("encoder"), "encoder_hidden_size")

        if self.use_cache:
            decoder_inputs.update(self._get_cache_inputs())

        return decoder_inputs

    def _get_decoder_outputs(self):
        decoder_outputs = OrderedDict({})
        decoder_outputs["logits"] = (Dims.BATCH, self.decoder_dims, self.vocab_size)

        if self.use_cache:
            decoder_outputs.update(self._get_self_attention_cache_outputs())

        return decoder_outputs

    # For encoder/decoder models, we need to generate cross attention separately
    def _get_cross_attention_cache_generator_inputs(self):
        return {"encoder_hidden_states":(Dims.BATCH, Dims.create_new_sequence_dim("encoder"), "encoder_hidden_size")}

    def _get_cross_attention_cache_generator_outputs(self):
        cross_attention_cache_outputs = OrderedDict({})
        for i in range(self.num_decoder_layers):
            cross_attention_cache_outputs["present_key_values.{}.cross.key".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("encoder"), self.d_kv)
            cross_attention_cache_outputs["present_key_values.{}.cross.value".format(i)] = (Dims.BATCH, self.num_heads, Dims.create_new_sequence_dim("encoder"), self.d_kv)
        return cross_attention_cache_outputs

    def get_input_dims(self) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        input_dims = {
            Seq2SeqModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: Dims(self._get_decoder_inputs())
        }

        if self.is_encoder_decoder:
            input_dims[Seq2SeqModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME] = Dims(self._get_encoder_inputs())
            if self.use_cache:
                input_dims[Seq2SeqModelTRTConfig.NETWORK_CROSS_ATTENTION_CACHE_GENERATOR_NAME] = Dims(self._get_cross_attention_cache_generator_inputs())

        return input_dims

    def get_output_dims(self) -> Dict:
        """
        Returns dictionary encoding of output dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """

        output_dims = {
            self.NETWORK_DECODER_SEGMENT_NAME: Dims(self._get_decoder_outputs())
        }

        if self.is_encoder_decoder:
            output_dims[self.NETWORK_ENCODER_SEGMENT_NAME] = Dims(self._get_encoder_outputs())
            if self.use_cache:
                output_dims[self.NETWORK_CROSS_ATTENTION_CACHE_GENERATOR_NAME] = Dims(self._get_cross_attention_cache_generator_outputs())

        return output_dims
