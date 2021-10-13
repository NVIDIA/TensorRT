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
_GPT2Metadata = namedtuple("GPT2Metadata", ["kv_cache"])


class GPT2Metadata(_GPT2Metadata, MetadataArgparseInteropMixin):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add commandline interface parser."""
        network_group = parser.add_argument_group("GPT2 network")
        network_group.add_argument(
            "--variant",
            help="GPT2 variant to generate",
            choices=GPT2ModelTRTConfig.TARGET_MODELS,
            required=True,
        )
        network_group.add_argument(
            "--enable-kv-cache",
            help="GPT2 enable KV cache",
            action="store_true",
            default=False,
        )

    @staticmethod
    def from_args(args: argparse.Namespace):
        return NetworkMetadata(
            variant=args.variant,
            precision=Precision(fp16=False),
            other=GPT2Metadata(kv_cache=args.enable_kv_cache),
        )

    @staticmethod
    def add_inference_args(parser: argparse.ArgumentParser) -> None:
        inference_group = parser.add_argument_group("inference group")
        inference_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )

    @staticmethod
    def from_inference_args(args: argparse.Namespace):
        base_metadata = GPT2Metadata.from_args(args)
        return base_metadata._replace(precision=Precision(fp16=args.fp16))


class GPT2ModelTRTConfig(NNConfig):
    VOCAB_SIZE = 50257  # Vocabulary size of the GPT-2 model
    TARGET_MODELS = ["gpt2", "gpt2-large"]
    NETWORK_DECODER_SEGMENT_NAME = "gpt2_decoder"
    NETWORK_SEGMENTS = [NETWORK_DECODER_SEGMENT_NAME]
    NETWORK_FULL_NAME = "full"

    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0]: 64,
        TARGET_MODELS[1]: 64,
    }

    def __init__(self):
        precision_fp16 = [False, True]
        kv_caches = [False]
        variants = []
        for variant, fp16, kv_cache in product(
            GPT2ModelTRTConfig.TARGET_MODELS, precision_fp16, kv_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16),
                    other=GPT2Metadata(kv_cache=kv_cache),
                )
            )

        super().__init__("GPT2", variants=variants)

    def get_python_requirements(self):
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.6.1")
        return base_requirements

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Returns:
            (Dict[str, Dims]): {"decoder": Dims}
        """
        return {
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: Dims(
                OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)})
            ),
        }

    @staticmethod
    def get_output_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of output dimensions.

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        return {
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: Dims(
                OrderedDict(
                    {
                        "logits": (
                            Dims.BATCH,
                            Dims.SEQUENCE,
                            GPT2ModelTRTConfig.VOCAB_SIZE,
                        )
                    }
                )
            ),
        }
