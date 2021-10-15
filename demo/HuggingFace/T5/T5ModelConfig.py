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
_T5Metadata = namedtuple("T5Metadata", ["kv_cache"])


class T5Metadata(_T5Metadata, MetadataArgparseInteropMixin):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add commandline interface parser."""
        network_group = parser.add_argument_group("T5 network")
        network_group.add_argument(
            "--variant",
            help="T5 variant to generate",
            choices=T5ModelTRTConfig.TARGET_MODELS,
            required=True,
        )
        network_group.add_argument(
            "--enable-kv-cache",
            help="T5 enable KV cache",
            action="store_true",
            default=False,
        )

    @staticmethod
    def from_args(args: argparse.Namespace):
        return NetworkMetadata(
            variant=args.variant,
            precision=Precision(fp16=False),
            other=T5Metadata(kv_cache=args.enable_kv_cache),
        )

    @staticmethod
    def add_inference_args(parser: argparse.ArgumentParser) -> None:
        T5Metadata.add_args(parser)
        inference_group = parser.add_argument_group("inference group")
        inference_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )

    @staticmethod
    def from_inference_args(args: argparse.Namespace):
        base_metadata = T5Metadata.from_args(args)
        return base_metadata._replace(precision=Precision(fp16=args.fp16))


class T5ModelTRTConfig(NNConfig):

    TARGET_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b"]
    NUMBER_OF_LAYERS = {TARGET_MODELS[0]: 6, TARGET_MODELS[1]: 12, TARGET_MODELS[2]: 24, TARGET_MODELS[3]: 24}
    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0]: 512,
        TARGET_MODELS[1]: 768,
        TARGET_MODELS[2]: 1024,
        TARGET_MODELS[3]: 1024,
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
            T5ModelTRTConfig.TARGET_MODELS, precision_fp16, kv_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16),
                    other=T5Metadata(kv_cache=kv_cache),
                )
            )

        super().__init__("T5", variants=variants)

    def get_python_requirements(self):
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.6.1")
        return base_requirements

    def get_network_segments(self):
        """
        Returns exportable segments for the given network.
        Used in the case where a single network needs to
        be exported into multiple parts.
        """
        return T5ModelTRTConfig.NETWORK_SEGMENTS

    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        # Remove redundant t5 name
        metadata = metadata._replace(variant=metadata.variant.lstrip("t5-"))
        return super().get_metadata_string(metadata)

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_inputs = Dims(
            OrderedDict(
                {
                    "input_ids": (Dims.BATCH, Dims.SEQUENCE),
                    "encoder_hidden_states": (
                        Dims.BATCH,
                        Dims.create_new_sequence_dim("encoder_hidden_length"),
                        T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
                    ),
                }
            )
        )

        encoder_inputs = Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)}))

        return {
            T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_inputs,
            T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_inputs,
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
                        T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
                    )
                }
            )
        )

        return {
            T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_outputs,
            T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_outputs,
        }
