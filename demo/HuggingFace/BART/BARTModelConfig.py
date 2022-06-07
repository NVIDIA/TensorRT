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

import argparse

from collections import namedtuple, OrderedDict
from itertools import product
from typing import Dict

# TRT-HuggingFace
from NNDF.networks import Precision, NetworkMetadata, NNConfig, Dims
from NNDF.interface import MetadataArgparseInteropMixin

# Limitation of namedtuples. You must declare namedtuples in module scope and not in classes.
# Otherwise pickle doesn't work.
# See: https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
_BARTMetadata = namedtuple("BARTMetadata", ["kv_cache"])


class BARTMetadata(_BARTMetadata, MetadataArgparseInteropMixin):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add commandline interface parser."""
        network_group = parser.add_argument_group("BART network")
        network_group.add_argument(
            "--variant",
            help="BART variant to generate",
            choices=BARTModelTRTConfig.TARGET_MODELS,
            required=True,
        )
        network_group.add_argument(
            "--enable-kv-cache",
            help="BART enable KV cache",
            action="store_true",
            default=False,
        )

    @staticmethod
    def from_args(args: argparse.Namespace):
        return NetworkMetadata(
            variant=args.variant,
            precision=Precision(fp16=False),
            other=BARTMetadata(kv_cache=args.enable_kv_cache),
        )

    @staticmethod
    def add_inference_args(parser: argparse.ArgumentParser) -> None:
        BARTMetadata.add_args(parser)
        inference_group = parser.add_argument_group("inference group")
        inference_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )

    @staticmethod
    def from_inference_args(args: argparse.Namespace):
        base_metadata = BARTMetadata.from_args(args)
        return base_metadata._replace(precision=Precision(fp16=args.fp16))


    @staticmethod
    def add_benchmarking_args(parser: argparse.ArgumentParser) -> None:
        benchmarking_group = parser.add_argument_group("benchmarking group")
        benchmarking_group.add_argument(
            "--input-seq-len",
            type=int,
            help="Specify fixed input sequence length for perf benchmarking. (default: max supported sequence length)",
        )
        benchmarking_group.add_argument(
            "--output-seq-len",
            type=int,
            help="Specify fixed output sequence length for perf benchmarking. (default: max supported sequence length)",
        )


BARTBenchmarkingArgs = namedtuple("BARTBenchmarkingArgs", ["input_seq_len", "output_seq_len"])

class BARTModelTRTConfig(NNConfig):

    TARGET_MODELS = ["facebook/bart-base", "facebook/bart-large", "facebook/bart-large-cnn"]
    # bart-base: 12-layer, 768-hidden, 16-heads, 139M parameters
    # bart-large: 24-layer, 1024-hidden, 16-heads, 406M parameters
    NUMBER_OF_LAYERS = {TARGET_MODELS[0]: 12, TARGET_MODELS[1]: 24, TARGET_MODELS[2]: 24} 
    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0]: 768,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 1024,
    }

    # To achieve identical results with original HuggingFace implementation, the min_length in model config should be consistent with each model variant
    # see task-specific params in config.json of each variant model
    MIN_OUTPUT_LENGTH = {
        TARGET_MODELS[0]: 0,
        TARGET_MODELS[1]: 0,
        TARGET_MODELS[2]: 56,
    } 

    #TODO: this might better be an inference time input like the `max_length` arg in generate() and greedy_search(). The change needed is in NNDF/interface.py:__call__ so it's a fundamental change affecting GPT2 and T5 code. Here I just put this option in BART model config for now. But it's also reasonable to treat this as a model config, because the TRT engine building may need this to have fixed dimension (e.g., to enable KV-cache)
    # see task-specific params in config.json of each variant model
    MAX_OUTPUT_LENGTH = {
        TARGET_MODELS[0]: 768,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 142,
    } 

    NO_REPEAT_NGRAM_SIZE = 3

    VOCAB_SIZE = {
        TARGET_MODELS[0]: 50265,
        TARGET_MODELS[1]: 50265,
        TARGET_MODELS[2]: 50264, # for bart-large-cnn config it's 50264 somehow. If not change here, results are incorrect since the trt results dimension reshape depends on this
    }

    NETWORK_FULL_NAME = "full"
    NETWORK_DECODER_SEGMENT_NAME = "decoder"
    NETWORK_ENCODER_SEGMENT_NAME = "encoder"
    NETWORK_SEGMENTS = [NETWORK_DECODER_SEGMENT_NAME, NETWORK_ENCODER_SEGMENT_NAME]
    
    def __init__(self):
        precision_fp16 = [False, True]
        kv_caches = [False, True]

        variants = []
        for variant, fp16, kv_cache in product(
            BARTModelTRTConfig.TARGET_MODELS, precision_fp16, kv_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16),
                    other=BARTMetadata(kv_cache=kv_cache),
                )
            )

        super().__init__("BART", variants=variants)

    def get_python_requirements(self):
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.8.0")
        return base_requirements

    def get_network_segments(self):
        """
        Returns exportable segments for the given network.
        Used in the case where a single network needs to
        be exported into multiple parts.
        """
        return BARTModelTRTConfig.NETWORK_SEGMENTS

    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        # Remove redundant bart name prefix
        metadata = metadata._replace(variant=metadata.variant.replace("facebook/bart-",""))
        return super().get_metadata_string(metadata)

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_inputs_dict = OrderedDict(
            {
                "input_ids": (Dims.BATCH, Dims.SEQUENCE),
                "encoder_hidden_states": (
                    Dims.BATCH,
                    Dims.create_new_sequence_dim("encoder_hidden_length"),
                    BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
                ),
            }
        )
        if metadata.other.kv_cache:
            decoder_inputs_dict["use_cache"] = ("boolean")
            decoder_inputs_dict["past_key_values"] = (Dims.BATCH, "num_heads", Dims.SEQUENCE, "embedding_size_per_head") 

        decoder_inputs = Dims(decoder_inputs_dict)

        encoder_inputs = Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)}))

        return {
            BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_inputs,
            BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_inputs,
        }

    @staticmethod
    def get_output_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of output dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_outputs = Dims(
            OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)})
        )
        encoder_outputs = Dims(
            OrderedDict(
                {
                    "hidden_states": (
                        Dims.BATCH,
                        Dims.SEQUENCE,
                        BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
                    )
                }
            )
        )

        return {
            BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_outputs,
            BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_outputs,
        }
