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
        network_group.add_argument(
            "--num-beams", type=int, default=1, help="Enables beam search during decoding."
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
            help="Specify fixed input sequence length for perf benchmarking. Required for benchmark except when both input_profile_max and output_profile_max are provided for trt",
        )
        benchmarking_group.add_argument(
            "--output-seq-len",
            type=int,
            help="Specify fixed output sequence length for perf benchmarking. Required for benchmark except when both input_profile_max and output_profile_max are provided for trt",
        )

BARTBenchmarkingArgs = namedtuple("BARTBenchmarkingArgs", ["input_seq_len", "output_seq_len"])

# trt has more benchmarking arguments
BARTTRTBenchmarkingArgs = namedtuple("BARTTRTBenchmarkingArgs", ["input_seq_len", "output_seq_len", "input_profile_max_len", "output_profile_max_len"])

class BARTModelTRTConfig(NNConfig):

    TARGET_MODELS = ["facebook/bart-base", "facebook/bart-large", "facebook/bart-large-cnn", "facebook/mbart-large-50"]

    MAX_DECODER_WORKSPACE_MB = {
        TARGET_MODELS[0]: 3072,
        TARGET_MODELS[1]: 3072,
        TARGET_MODELS[2]: 3072,
        TARGET_MODELS[3]: 3072,
    }
    
    # bart-base: 12-layer, 768-hidden, 139M parameters
    # bart-large: 24-layer, 1024-hidden, 406M parameters
    # in all bart variants, # of encoder layers and # of decoder layers are the same
    NUMBER_OF_LAYERS = {
        TARGET_MODELS[0]: 12, 
        TARGET_MODELS[1]: 24, 
        TARGET_MODELS[2]: 24,
        TARGET_MODELS[3]: 24,
    } 
    
    NUMBER_OF_DECODER_LAYERS = {
        TARGET_MODELS[0]: 6, 
        TARGET_MODELS[1]: 12, 
        TARGET_MODELS[2]: 12,
        TARGET_MODELS[3]: 12,
    } 
    
    # in all bart variants, # of heads in encoder and decoder are the same
    NUMBER_OF_HEADS = {
        TARGET_MODELS[0]: 12, 
        TARGET_MODELS[1]: 16, 
        TARGET_MODELS[2]: 16,
        TARGET_MODELS[3]: 16,
    }

    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0]: 768,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 1024,
        TARGET_MODELS[3]: 1024,
    }

    # encoder hidden size is not necessarily same as max sequence length. Separate for clarification
    ENCODER_HIDDEN_SIZE = {
        TARGET_MODELS[0]: 768,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 1024,
        TARGET_MODELS[3]: 1024,
    } 

    # To achieve identical results with original HuggingFace implementation, the min_length in model config should be consistent with each model variant
    # see task-specific params in config.json of each variant model
    MIN_OUTPUT_LENGTH = {
        TARGET_MODELS[0]: 0,
        TARGET_MODELS[1]: 0,
        TARGET_MODELS[2]: 56,
        TARGET_MODELS[3]: 0,
    } 

    #TODO: this might better be an inference time input like the `max_length` arg in generate() and greedy_search(). The change needed is in NNDF/interface.py:__call__ so it's a fundamental change affecting GPT2 and T5 code. Here I just put this option in BART model config for now. But it's also reasonable to treat this as a model config, because the TRT engine building may need this to have fixed dimension (e.g., to enable KV-cache)
    # see task-specific params in config.json of each variant model
    MAX_OUTPUT_LENGTH = {
        TARGET_MODELS[0]: 768,
        TARGET_MODELS[1]: 1024,
        TARGET_MODELS[2]: 142,
        TARGET_MODELS[3]: 200,
    } 

    # BART specific configs: https://huggingface.co/facebook/bart-base/blob/main/config.json 
    NO_REPEAT_NGRAM_SIZE = 3
    BOS_TOKEN_ID = 0
    EOS_TOKEN_ID = 2

    VOCAB_SIZE = {
        TARGET_MODELS[0]: 50265,
        TARGET_MODELS[1]: 50265,
        TARGET_MODELS[2]: 50264, # for bart-large-cnn config it's 50264 somehow. If not change here, results are incorrect since the trt results dimension reshape depends on this
        TARGET_MODELS[3]: 250054 # for mbart multilingual models, vocab size is much larger
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
        if "mbart" in metadata.variant:
            metadata = metadata._replace(variant=metadata.variant.replace("facebook/mbart-","mbart-"))
        else:    
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
                    BARTModelTRTConfig.ENCODER_HIDDEN_SIZE[metadata.variant], # dim not containing string 'Dims.BATCH' or 'Dims.SEQUENCE' will be non-dynamic axis
                ),
            }
        )
        if metadata.other.kv_cache:
            # for KV cache version, we need add per-layer KV cache inputs. `past_key_values` at each layer is (self-attention K, self-attention V, cross-attention K, cross-attention V)
            for i in range(BARTModelTRTConfig.NUMBER_OF_DECODER_LAYERS[metadata.variant]):
                # decoder self-attention KV cache (dim[0] & dim[2] are dynamic, and dim[2] varies at each decoding timestep) 
                self_attention_past_kv_dims = (Dims.BATCH, "num_heads", Dims.create_new_sequence_dim("past_decoder_length"), "embedding_size_per_head")
                decoder_inputs_dict[f"past_key_values.{i}.decoder.key"] = self_attention_past_kv_dims
                decoder_inputs_dict[f"past_key_values.{i}.decoder.value"] = self_attention_past_kv_dims
                
                # encoder-decoder cross-attention KV cache (dim[0] & dim[2] are dynamic, but dim[2] is constant at each decoding timestep)
                cross_attention_past_kv_dims = (Dims.BATCH, "num_heads", Dims.create_new_sequence_dim("encoder_length"), "embedding_size_per_head") 
                decoder_inputs_dict[f"past_key_values.{i}.encoder.key"] = cross_attention_past_kv_dims
                decoder_inputs_dict[f"past_key_values.{i}.encoder.value"] = cross_attention_past_kv_dims

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
        decoder_outputs_dict = OrderedDict(
            {"hidden_states": (Dims.BATCH, Dims.SEQUENCE)})

        if metadata.other.kv_cache:
            # for KV cache version, we need add per-layer KV cache inputs. `past_key_values` at each layer is (self-attention K, self-attention V, cross-attention K, cross-attention V)
            
            # for all BART variants, # encoder layers = # decoder layers, so just divide total # layers by 2
            for i in range(BARTModelTRTConfig.NUMBER_OF_DECODER_LAYERS[metadata.variant]):
                # decoder self-attention KV cache (dim[0] & dim[2] are dynamic, and dim[2] varies at each decoding timestep) 
                self_attention_present_kv_dims = (Dims.BATCH, "num_heads", Dims.create_new_sequence_dim("decoder_length"), "embedding_size_per_head")
                decoder_outputs_dict[f"present_key_values.{i}.decoder.key"] = self_attention_present_kv_dims
                decoder_outputs_dict[f"present_key_values.{i}.decoder.value"] = self_attention_present_kv_dims
                
                # encoder-decoder cross-attention KV cache (dim[0] & dim[2] are dynamic, but dim[2] is constant at each decoding timestep)
                cross_attention_present_kv_dims = (Dims.BATCH, "num_heads", Dims.create_new_sequence_dim("encoder_length"), "embedding_size_per_head") 
                decoder_outputs_dict[f"present_key_values.{i}.encoder.key"] = cross_attention_present_kv_dims
                decoder_outputs_dict[f"present_key_values.{i}.encoder.value"] = cross_attention_present_kv_dims

        decoder_outputs = Dims(decoder_outputs_dict)

        encoder_outputs = Dims(
            OrderedDict(
                {
                    "hidden_states": (
                        Dims.BATCH,
                        Dims.SEQUENCE,
                        BARTModelTRTConfig.ENCODER_HIDDEN_SIZE[metadata.variant],
                    )
                }
            )
        )

        return {
            BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_outputs,
            BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_outputs,
        }
