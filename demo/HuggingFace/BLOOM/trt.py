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

import os
import sys

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# polygraphy
from polygraphy.backend.trt import Profile

from NNDF.models import TRTEngineFile
from NNDF.networks import NetworkMetadata
from Seq2Seq.trt import Seq2SeqTRTDecoder, Seq2SeqTRT
from BLOOM.BLOOMModelConfig import BLOOMModelTRTConfig
from BLOOM.export import BLOOMModelClass

import tensorrt as trt

class BLOOMTRTDecoder(Seq2SeqTRTDecoder):
    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        config: BLOOMModelTRTConfig,
        nvtx_verbose: bool,
    ):
        super().__init__(trt_engine_file, network_metadata, config, nvtx_verbose)

    def _get_self_attn_keys_cache_shape(self):
        return (self.expand_size * self.config.num_heads, self.embedding_size_per_head, self.past_decoder_length)

    def _get_self_attn_vals_cache_shape(self):
        return (self.expand_size * self.config.num_heads, self.past_decoder_length, self.embedding_size_per_head)

    def _get_next_self_attn_keys_cache_shape(self):
        return (self.expand_size * self.config.num_heads, self.embedding_size_per_head, self.past_decoder_length + 1)

    def _get_next_self_attn_vals_cache_shape(self):
        return (self.expand_size * self.config.num_heads, self.past_decoder_length + 1, self.embedding_size_per_head)

class BLOOMTRT(Seq2SeqTRT):
    def __init__(
        self,
        config_class=BLOOMModelTRTConfig,
        description="Runs trt results for BLOOM model.",
        **kwargs
    ):
        super().__init__(config_class, description=description, model_classes=BLOOMModelClass, **kwargs)
        self.decoder_class = BLOOMTRTDecoder

    # BLOOM's kv cache shape is unique.
    # Most GPT2-like models use past_key_values with shape
    # [batch_size, num_heads, sequence_length, embed_size_per_head]
    # for both keys and values.
    # BLOOM's kv cache uses the following format:
    # Keys: [batch_size * num_heads, embed_size_per_head, sequence_length]
    # Vals: [batch_size * num_heads, sequence_length, embed_size_per_head]
    # So the kv cache requires special treatment when setting up decoder profiles
    def _setup_decoder_profiles(self):
        if not self.config.use_cache:
            decoder_profile = Profile().add(
                "input_ids",
                min=(self.min_expand_size, 1),
                opt=(self.opt_expand_size, self.opt_output_seq_len),
                max=(self.max_expand_size, self.config.max_output_profile_length),
            )
            if self.config.use_mask:
                decoder_profile = decoder_profile.add(
                    "attention_mask",
                    min=(self.min_expand_size, 1),
                    opt=(self.opt_expand_size, self.opt_output_seq_len),
                    max=(self.max_expand_size, self.config.max_output_profile_length),
                )

            decoder_profiles = [decoder_profile]
        else:
            num_heads = self.config.num_heads
            embedding_size_per_head = self.config.d_kv
            num_decoder_layers = self.config.num_decoder_layers

            self_attn_keys_profile = {
                "min": (self.min_expand_size * num_heads, embedding_size_per_head, 0),
                "opt": (self.opt_expand_size * num_heads, embedding_size_per_head, self.opt_output_seq_len - 1),
                "max": (self.max_expand_size * num_heads, embedding_size_per_head, self.config.max_output_profile_length - 1),
            }

            self_attn_vals_profile = {
                "min": (self.min_expand_size * num_heads, 0, embedding_size_per_head),
                "opt": (self.opt_expand_size * num_heads, self.opt_output_seq_len - 1, embedding_size_per_head),
                "max": (self.max_expand_size * num_heads, self.config.max_output_profile_length - 1, embedding_size_per_head),
            }

            decoder_profile_generation = Profile().add(
                "input_ids",
                min=(self.min_expand_size, 1),
                opt=(self.opt_expand_size, 1),
                max=(self.max_expand_size, 1),
            )
            if self.config.use_mask:
                decoder_profile_generation = decoder_profile_generation.add(
                    "attention_mask",
                    min=(self.min_expand_size, 1),
                    opt=(self.opt_expand_size, self.opt_output_seq_len),
                    max=(self.max_expand_size, self.config.max_output_profile_length),
                )

            for i in range(num_decoder_layers):
                decoder_profile_generation = decoder_profile_generation.add(
                    f"past_key_values.{i}.self.key",
                    **self_attn_keys_profile
                ).add(
                    f"past_key_values.{i}.self.value",
                    **self_attn_vals_profile
                )

            decoder_profiles = [decoder_profile_generation]

            # Decoder only model has "context phase" that is only used for the 1st decoder phase.
            # Context phase takes various-length input_ids with no kv cache and generates initial cache for subsequent decoding steps.
            decoder_profile_context = Profile().add(
                "input_ids",
                min=(self.min_expand_size, 1),
                opt=(self.opt_expand_size, self.opt_input_seq_len),
                max=(self.max_expand_size, self.config.max_input_profile_length),
            )
            if self.config.use_mask:
                decoder_profile_context = decoder_profile_context.add(
                    "attention_mask",
                    min=(self.min_expand_size, 1),
                    opt=(self.opt_expand_size, self.opt_input_seq_len),
                    max=(self.max_expand_size, self.config.max_input_profile_length),
                )

            self_attn_keys_profile_context = {
                "min": (self.min_expand_size * num_heads, embedding_size_per_head, 0),
                "opt": (self.opt_expand_size * num_heads, embedding_size_per_head, 0),
                "max": (self.max_expand_size * num_heads, embedding_size_per_head, 0),
            }
            self_attn_vals_profile_context = {
                "min": (self.min_expand_size * num_heads, 0, embedding_size_per_head),
                "opt": (self.opt_expand_size * num_heads, 0, embedding_size_per_head),
                "max": (self.max_expand_size * num_heads, 0, embedding_size_per_head),
            }
            for i in range(num_decoder_layers):
                decoder_profile_context = decoder_profile_context.add(
                    f"past_key_values.{i}.self.key",
                    **self_attn_keys_profile_context
                ).add(
                    f"past_key_values.{i}.self.value",
                    **self_attn_vals_profile_context
                )

            decoder_profiles.append(decoder_profile_context)

        return decoder_profiles

    # Used to create decoder trtexec command string for debugging accuracy/performance.
    def _decoder_trtexec_command(self, decoder_engine_path):
        command = f"trtexec --onnx={self.onnx_decoder.fpath} --verbose --saveEngine={decoder_engine_path} "
        min_shape = "--minShapes="
        opt_shape = "--optShapes="
        max_shape = "--maxShapes="
        if not self.config.use_cache:
            min_shape += f"'input_ids':{self.min_expand_size}x1"
            opt_shape += f"'input_ids':{self.opt_expand_size}x{self.opt_output_seq_len}"
            max_shape += f"'input_ids':{self.max_expand_size}x{self.config.max_output_profile_length}"
            if self.config.use_mask:
                min_shape += f",'attention_mask':{self.min_expand_size}x1"
                opt_shape += f",'attention_mask':{self.opt_expand_size}x{self.opt_output_seq_len}"
                max_shape += f",'attention_mask':{self.max_expand_size}x{self.config.max_output_profile_length}"

            command += f"{min_shape} {opt_shape} {max_shape} "

        else:
            min_shape += f"'input_ids':{self.min_expand_size}x1"
            opt_shape += f"'input_ids':{self.opt_expand_size}x1"
            max_shape += f"'input_ids':{self.max_expand_size}x1"

            if self.config.use_mask:
                min_shape += f",'attention_mask':{self.min_expand_size}x1"
                opt_shape += f",'attention_mask':{self.opt_expand_size}x{self.opt_output_seq_len}"
                max_shape += f",'attention_mask':{self.max_expand_size}x{self.config.max_output_profile_length}"


            # build context phase profile
            cmin_shape = "--minShapes="
            cmax_shape = "--maxShapes="
            copt_shape = "--optShapes="
            cmin_shape += f"'input_ids':{self.min_expand_size}x1"
            copt_shape += f"'input_ids':{self.opt_expand_size}x{self.opt_input_seq_len}"
            cmax_shape += f"'input_ids':{self.max_expand_size}x{self.config.max_input_profile_length}"

            if self.config.use_mask:
                cmin_shape += f",'attention_mask':{self.min_expand_size}x1"
                copt_shape += f",'attention_mask':{self.opt_expand_size}x{self.opt_input_seq_len}"
                cmax_shape += f",'attention_mask':{self.max_expand_size}x{self.config.max_input_profile_length}"

            for i in range(self.config.num_decoder_layers):
                k = f",'past_key_values.{i}.self.key'"
                v = f",'past_key_values.{i}.self.value'"
                k_min_str = f":{self.min_expand_size * self.config.num_heads}x{self.embedding_size_per_head}x0"
                k_opt_str = f":{self.opt_expand_size * self.config.num_heads}x{self.embedding_size_per_head}x{self.opt_output_seq_len-1}"
                k_max_str = f":{self.max_expand_size * self.config.num_heads}x{self.embedding_size_per_head}x{self.config.max_output_profile_length-1}"
                v_min_str = f":{self.min_expand_size * self.config.num_heads}x0x{self.embedding_size_per_head}"
                v_opt_str = f":{self.opt_expand_size * self.config.num_heads}x{self.opt_output_seq_len-1}x{self.embedding_size_per_head}"
                v_max_str = f":{self.max_expand_size * self.config.num_heads}x{self.config.max_output_profile_length-1}x{self.embedding_size_per_head}"
                min_shape += k + k_min_str + v + v_min_str
                opt_shape += k + k_opt_str + v + v_opt_str
                max_shape += k + k_max_str + v + v_max_str

                k_zero_min_str = f":{self.min_expand_size * self.config.num_heads}x{self.embedding_size_per_head}x0"
                k_zero_opt_str = f":{self.opt_expand_size * self.config.num_heads}x{self.embedding_size_per_head}x0"
                k_zero_max_str = f":{self.max_expand_size * self.config.num_heads}x{self.embedding_size_per_head}x0"
                v_zero_min_str = f":{self.min_expand_size * self.config.num_heads}x0x{self.embedding_size_per_head}"
                v_zero_opt_str = f":{self.opt_expand_size * self.config.num_heads}x0x{self.embedding_size_per_head}"
                v_zero_max_str = f":{self.max_expand_size * self.config.num_heads}x0x{self.embedding_size_per_head}"
                cmin_shape += k + k_zero_min_str + v + v_zero_min_str
                copt_shape += k + k_zero_opt_str + v + v_zero_opt_str
                cmax_shape += k + k_zero_max_str + v + v_zero_max_str

            command += f"--profile=0 {min_shape} {opt_shape} {max_shape} --profile=1 {cmin_shape} {copt_shape} {cmax_shape} "

        if self.metadata.precision.fp16:
            int_type = "int64" if hasattr(trt,"int64") else "int32"
            inputIO = f"--inputIOFormats={int_type}:chw" # input_ids
            if self.config.use_mask:
                inputIO += f",{int_type}:chw" # attention_mask
            outputIO = "--outputIOFormats=fp16:chw" # logits
            if self.config.use_cache:
                for _ in range(self.config.num_decoder_layers):
                    kv_precision = ",fp16:chw,fp16:chw"
                    inputIO += kv_precision # self attention
                    outputIO += kv_precision # self attention

            command += f"--fp16 --precisionConstraints=obey {inputIO} {outputIO} "

        if self.timing_cache is not None:
            command += f"--timingCacheFile={self.timing_cache} "

        return command

    # Used to create encoder trtexec command string for debugging accuracy/performance.
    def _encoder_trtexec_command(self):
        raise NotImplementedError("BLOOM is decoder only: _encoder_trtexec_command() should not have been called")

    # Used to create generator trtexec command string for debugging accuracy/performance.
    def _generator_trtexec_command(self):
        raise NotImplementedError("BLOOM is decoder only: _generator_trtexec_command() should not have been called")

# Entry point
RUN_CMD = BLOOMTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
