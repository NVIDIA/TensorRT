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
        network_group.add_argument(
            "--num-beams", type=int, default=1, help="Enables beam search during decoding."
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

    @staticmethod
    def add_benchmarking_args(parser: argparse.ArgumentParser) -> None:
        benchmarking_group = parser.add_argument_group("benchmarking group")
        benchmarking_group.add_argument(
            "--input-seq-len",
            type=int,
            help="Specify fixed input sequence length for perf benchmarking.",
        )
        benchmarking_group.add_argument(
            "--output-seq-len",
            type=int,
            help="Specify fixed output sequence length for perf benchmarking.",
        )


GPT2BenchmarkingArgs = namedtuple("GPT2BenchmarkingArgs", ["input_seq_len", "output_seq_len"])
GPT2TRTBenchmarkingArgs = namedtuple("GPT2BenchmarkingArgs", ["input_seq_len", "output_seq_len", "input_profile_max_len", "output_profile_max_len"])


class GPT2ModelTRTConfig(NNConfig):
    TARGET_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"]
    NETWORK_DECODER_SEGMENT_NAME = "gpt2_decoder"
    NETWORK_SEGMENTS = [NETWORK_DECODER_SEGMENT_NAME]
    NETWORK_FULL_NAME = "full"

    # Vocabulary size of the GPT-2 model
    VOCAB_SIZE = {
        TARGET_MODELS[0]: 50257,
        TARGET_MODELS[1]: 50257,
        TARGET_MODELS[2]: 50257,
        TARGET_MODELS[3]: 50257,
        TARGET_MODELS[4]: 50400,
    }

    NUMBER_OF_LAYERS = {
        TARGET_MODELS[0]: 12,
        TARGET_MODELS[1]: 24,
        TARGET_MODELS[2]: 36,
        TARGET_MODELS[3]: 48,
        TARGET_MODELS[4]: 28,
    }

    NUMBER_OF_HEADS = {
        TARGET_MODELS[0]: 12,
        TARGET_MODELS[1]: 16,
        TARGET_MODELS[2]: 20,
        TARGET_MODELS[3]: 25,
        TARGET_MODELS[4]: 16,
    }
    # This corresponds to max_length in task_specific_params for text-generation.
    # Both input and output length should not exceed 50.
    MAX_LENGTH = {
        TARGET_MODELS[0]: 50,
        TARGET_MODELS[1]: 50,
        TARGET_MODELS[2]: 50,
        TARGET_MODELS[3]: 50,
        TARGET_MODELS[4]: 50,
    }
    
    # The maximum sequence length that this model might ever be used with. 
    # Typically set this to something large. Use for benchmarking in this case
    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0]: 1024,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 1024,
        TARGET_MODELS[3]: 1024,
        TARGET_MODELS[4]: 2048,
    }

    # Dimensionality of the embeddings and hidden states.
    EMBEDDING_SIZE = {
        TARGET_MODELS[0]: 768,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 1280,
        TARGET_MODELS[3]: 1600,
        TARGET_MODELS[4]: 4096,
    }

    MIN_OUTPUT_LENGTH = {
        TARGET_MODELS[0]: 0,
        TARGET_MODELS[1]: 0,
        TARGET_MODELS[2]: 0,
        TARGET_MODELS[3]: 0,
        TARGET_MODELS[4]: 0,
    }

    BOS_TOKEN_ID = 50256
    EOS_TOKEN_ID = 50256

    def __init__(self):
        precision_fp16 = [False, True]
        kv_caches = [False, True]
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
        base_requirements.append('transformers==4.20.0; python_version>="3.7"')
        base_requirements.append('transformers==4.18.0; python_version<"3.7"')
        return base_requirements
    
    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        # Remove redundant GPT2 name
        metadata = metadata._replace(variant=metadata.variant.lstrip("GPT2-"))
        metadata = metadata._replace(variant=metadata.variant.lstrip("EleutherAI/"))
        return super().get_metadata_string(metadata)

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Returns:
            (Dict[str, Dims]): {"decoder": Dims}
        """
        decoder_inputs_dict = OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)})
        if metadata.other.kv_cache:
            # for KV cache version, we need add per-layer KV cache inputs. `past_key_values` at each layer is (self-attention K, self-attention V)
            for i in range(GPT2ModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant]):
                # decoder self-attention KV cache (dim[0] & dim[2] are dynamic, and dim[2] varies at each decoding timestep) 
                self_attention_past_kv_dims = (Dims.BATCH, "num_heads", Dims.create_new_sequence_dim("past_decoder_length"), "embedding_size_per_head")
                decoder_inputs_dict[f"past_key_values.{i}.decoder.key"] = self_attention_past_kv_dims
                decoder_inputs_dict[f"past_key_values.{i}.decoder.value"] = self_attention_past_kv_dims
        
        decoder_inputs = Dims(decoder_inputs_dict)

        return {
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_inputs
        }

    @staticmethod
    def get_output_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of output dimensions.

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_outputs_dict = OrderedDict(
            {
                "logits": (
                    Dims.BATCH,
                    Dims.SEQUENCE,
                    GPT2ModelTRTConfig.VOCAB_SIZE[metadata.variant],
                )
            }
        )
        if metadata.other.kv_cache:
            # for KV cache version, we need add per-layer KV cache inputs. `past_key_values` at each layer is (self-attention K, self-attention V)
            for i in range(GPT2ModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant]):
                # decoder self-attention KV cache (dim[0] & dim[2] are dynamic, and dim[2] varies at each decoding timestep) 
                self_attention_present_kv_dims = (Dims.BATCH, "num_heads", Dims.create_new_sequence_dim("decoder_length"), "embedding_size_per_head")
                decoder_outputs_dict[f"present_key_values.{i}.decoder.key"] = self_attention_present_kv_dims
                decoder_outputs_dict[f"present_key_values.{i}.decoder.value"] = self_attention_present_kv_dims
        
        decoder_outputs = Dims(decoder_outputs_dict)

        return {
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_outputs
        }
