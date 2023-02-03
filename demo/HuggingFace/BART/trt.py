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

import os
import sys
import copy
from typing import Dict, List, Tuple, Union
from functools import reduce

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# polygraphy
from polygraphy.backend.trt import Profile

# tensorrt
import tensorrt as trt

# torch
import torch

# huggingface
from transformers import BartTokenizer, BartConfig, MBart50Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin

# tensorrt
from tensorrt import PreviewFeature

# TRT-HuggingFace
from NNDF.interface import TRTInferenceCommand
from NNDF.networks import (
    BenchmarkingResult,
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
    NetworkResult,
    NetworkRuntime,
    Precision,
    TimingProfile,
)

from NNDF.tensorrt_utils import TRTNativeRunner, set_kv_data, allocate_binding_buffer, setup_benchmark_arg
from NNDF.torch_utils import expand_inputs_for_beam_search
from NNDF.general_utils import NNFolderWorkspace
from BART.frameworks import BARTHuggingFace
from BART.BARTModelConfig import BARTModelTRTConfig, BARTMetadata, BARTTRTBenchmarkingArgs
from BART.measurements import decoder_inference, encoder_inference, full_inference_greedy, full_inference_beam, calculate_perplexity
from BART.export import BARTDecoderONNXFile, BARTEncoderONNXFile
from NNDF.models import TRTEngineFile
from NNDF.logger import G_LOGGER

# from HuggingFace transformers
from transformers.generation_logits_process import (
    NoRepeatNGramLogitsProcessor,
    MinLengthLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.generation_beam_search import (
    BeamSearchScorer,
)

class TRTHFRunner(TRTNativeRunner, GenerationMixin):
    """Runner that adds interop support for HF and HF provided greedy_search functions."""

    # Stores the encoder input length received at runtime, which is used to slice decoder inputs.
    ENCODER_LENGTH = 0
    def _allocate_memory(self,
                         input_shapes: Dict[str, tuple],
                         input_types: Dict[str, torch.dtype],
                         output_shapes: Dict[str, tuple],
                         output_types: Dict[str, torch.dtype]):
        """Helper function for binding several inputs at once and pre-allocating the results."""
        # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
        self.inputs = allocate_binding_buffer(input_types, input_shapes)
        self.outputs = allocate_binding_buffer(output_types, output_shapes)

        bindings = [None] * self.trt_engine.num_bindings

        for input_name, input_array in self.inputs.items():
            # Allocate memory for inputs
            input_idx = self.trt_engine.get_binding_index(input_name)
            self.trt_context.set_binding_shape(input_idx, input_shapes[input_name])
            bindings[input_idx] = input_array.data_ptr()

        assert self.trt_context.all_binding_shapes_specified

        for output_name, output_array in self.outputs.items():
            # Output shape should be allocated from context size
            output_idx = self.trt_engine.get_binding_index(output_name)
            bindings[output_idx] = output_array.data_ptr()

        return bindings

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1
    ):
        super().__init__(trt_engine_file, network_metadata)
        self.config = hf_config
        self.batch_size = batch_size

class BARTTRTEncoder(TRTHFRunner):
    """TRT implemented network interface that can be used to measure inference time."""

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1,
        benchmarking_args: BARTTRTBenchmarkingArgs = None
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        # In benchmarking mode, the max_sequence_length should be the designated input_profile_max_len
        if benchmarking_args is not None and benchmarking_args.input_profile_max_len is not None:
            self.max_sequence_length = benchmarking_args.input_profile_max_len
        else:
            self.max_sequence_length = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[network_metadata.variant]
        self.encoder_hidden_size = BARTModelTRTConfig.ENCODER_HIDDEN_SIZE[network_metadata.variant]

        # We only have one profile to select so we can just grab the profile at the start of the class
        self.profile_idx = self.get_optimization_profile(batch_size=self.batch_size, sequence_length=1)

        self.input_shapes = {
            "input_ids": (self.batch_size, self.max_sequence_length)
        }
        self.input_types = {
            "input_ids": torch.int32
        }
        self.output_shapes = {
            "hidden_states": (self.batch_size, self.max_sequence_length, self.encoder_hidden_size)
        }
        self.output_types = {
            "hidden_states": torch.float32
        }
        self.bindings = self._allocate_memory(self.input_shapes, self.input_types, self.output_shapes, self.output_types)

    def forward(self, input_ids, *args, **kwargs):
        bs = self.batch_size
        max_length = self.max_sequence_length
        TRTHFRunner.ENCODER_LENGTH = input_ids.shape[1]
        input_length = input_ids.shape[1]
        encoder_hidden_size = self.encoder_hidden_size

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu"))

        # We allocate the buffers using max_length, but we only need to first portion of it, so copy the data into the
        # first portion of the input buffer.
        # TODO: Could we just reuse input_ids' data_ptr() as the first binding when input_ids is already contiguous to
        # avoid an additional D2D?
        if is_cpu_mode:
            self.inputs["input_ids"] = input_ids.int().flatten().contiguous().cuda()
            self.bindings[0] = self.inputs["input_ids"].data_ptr()
        else:
            self.inputs["input_ids"][:bs * input_length] = input_ids.flatten()

        # Set the binding shape of input_ids, which should be (bs, input_length).
        self.trt_context.set_binding_shape(0, input_ids.shape)

        # Launch TRT inference.
        # TODO: Could we use execute_v2_async() instead of execute_v2()?
        self.trt_context.execute_v2(bindings=self.bindings)

        # We allocate the buffers using max_length, but we only need to first portion of it, so get only the first
        # portion of the output buffer and return that.
        # TODO: Could we construct a Torch tensor using given data_ptr() to avoid this D2D copy?
        hidden_states_output = self.outputs["hidden_states"]
        if is_cpu_mode:
            hidden_states_output = hidden_states_output.cpu()
        
        folded = hidden_states_output[:bs * input_length * encoder_hidden_size].view(bs, input_length, encoder_hidden_size)

        return folded

class BARTTRTDecoder(TRTHFRunner):

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_args: BARTTRTBenchmarkingArgs = None
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        
        # In benchmarking mode, the max_sequence_length should be the user-provided input_profile_max_len
        if benchmarking_args is not None and benchmarking_args.input_profile_max_len is not None:
            self.max_sequence_length = benchmarking_args.input_profile_max_len
        else:
            self.max_sequence_length = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[network_metadata.variant]
        
        # Similarly, the max_output_length should be the user-provided output_profile_max_len
        if benchmarking_args is not None and benchmarking_args.output_profile_max_len is not None:
            self.max_output_length = benchmarking_args.output_profile_max_len
        else:
            self.max_output_length = BARTModelTRTConfig.MAX_OUTPUT_LENGTH[network_metadata.variant]

        self.encoder_hidden_size = BARTModelTRTConfig.ENCODER_HIDDEN_SIZE[network_metadata.variant]
        self.num_heads = BARTModelTRTConfig.NUMBER_OF_HEADS[network_metadata.variant]
        self.embedding_size_per_head = self.encoder_hidden_size // self.num_heads

        # We only have one profile to select so we can just grab the profile at the start of the class
        self.profile_idx = self.get_optimization_profile(batch_size=self.batch_size * num_beams, sequence_length=1)
        input_profile_length = self.max_output_length if (not self.config.use_cache) else 1
        self.input_types = {
            "input_ids": torch.int32,
            "encoder_hidden_states": torch.float32
        }
        self.input_shapes = {
            "input_ids": (self.batch_size * num_beams, input_profile_length),
            "encoder_hidden_states": (self.batch_size * num_beams, self.max_sequence_length, self.encoder_hidden_size)
        }

        self.output_shapes = {
            "hidden_states": (self.batch_size * num_beams, self.max_output_length, BARTModelTRTConfig.VOCAB_SIZE[network_metadata.variant])
        }
        self.output_types = {
            "hidden_states": torch.float32
        }

        if self.config.use_cache:

            self.num_decoder_layers = BARTModelTRTConfig.NUMBER_OF_DECODER_LAYERS[network_metadata.variant]
            # Set kv cache shape and type
            for i in range(self.num_decoder_layers):
                kv_type_dict = {"encoder": torch.float32, "decoder": torch.float32}
                set_kv_data(self.input_types, "past", i, kv_type_dict)
                set_kv_data(self.output_types,"present", i, kv_type_dict)

                self_attention_kv_shape = (self.batch_size * num_beams, self.num_heads, self.max_output_length - 1, self.embedding_size_per_head)
                cross_attention_kv_shape = (self.batch_size * num_beams, self.num_heads, self.max_sequence_length, self.embedding_size_per_head)
                kv_shape_dict = {"encoder": cross_attention_kv_shape, "decoder": self_attention_kv_shape}

                set_kv_data(self.input_shapes, "past", i, kv_shape_dict)
                set_kv_data(self.output_shapes, "present", i, kv_shape_dict)

            self.kv_cache_binding_offset = 2 # 0: input_ids, 1: encoder_hidden_states, kv cache input indices start from 2
        
        self.bindings = self._allocate_memory(self.input_shapes, self.input_types, self.output_shapes, self.output_types)

        # Optimization bit
        self.persist_encoder_hidden_states = False
        self.persist_cross_attention_kv_cache = False

        self.use_non_kv_engine = self.config.use_cache
        # trick: set flag based on kv cache mode. This maintains code simplicity in forward() where a common codeblock is shared between non kv-cache & kv-cache modes
        # non kv-cache mode: False. Then in forward(), trt_context and bindings are set to the default ones
        # kv-cache mode: True. By default 1st decoding step starts with non-kv engine's context and binding; then flag gets updated in prepare_inputs_for_generation()

        self.return_device = "cuda"

        self.variant = network_metadata.variant # record variant name to later index the vocab_size in forward()

    def set_non_kv_engine_for_kv_mode(self, trt_engine_file_non_kv: TRTEngineFile):
        # same steps in tensorrt_utils.py: TRTNativeRunner
        with open(trt_engine_file_non_kv.fpath, "rb") as f:
            self.trt_engine_non_kv = self.trt_runtime.deserialize_cuda_engine(f.read())
            self.trt_context_non_kv = self.trt_engine_non_kv.create_execution_context()
        
        # Input does not have kv cache, so only inpuy_ids and encoder_hidden_states
        self.input_types_non_kv = {k: self.input_types[k] for k in ["input_ids", "encoder_hidden_states"]}
        self.input_shapes_non_kv = {k: self.input_shapes[k] for k in ["input_ids", "encoder_hidden_states"]}
        
        # Output is the same as kv
        self.output_types_non_kv = copy.deepcopy(self.output_types)
        self.output_shapes_non_kv = copy.deepcopy(self.output_shapes)

        # follow same steps in _allocate_memory
        self.inputs_non_kv = allocate_binding_buffer(self.input_types_non_kv, self.input_shapes_non_kv)
        self.outputs_non_kv = allocate_binding_buffer(self.output_types_non_kv, self.output_shapes_non_kv)

        bindings = [None] * self.trt_engine_non_kv.num_bindings

        for input_name, input_array in self.inputs_non_kv.items():
            # Allocate memory for inputs
            input_idx = self.trt_engine_non_kv.get_binding_index(input_name)
            self.trt_context_non_kv.set_binding_shape(input_idx, self.input_shapes_non_kv[input_name])
            bindings[input_idx] = input_array.data_ptr()

        assert self.trt_context_non_kv.all_binding_shapes_specified

        for output_name, output_array in self.outputs_non_kv.items():
            # Output shape should be allocated from context size
            output_idx = self.trt_engine_non_kv.get_binding_index(output_name)
            bindings[output_idx] = output_array.data_ptr()

        self.bindings_non_kv = bindings

        G_LOGGER.info("Non-KV cache engine setup is successful in KV cache mode.")

    def set_encoder_hidden_states_for_inference_cycle(self, encoder_hidden_states):
        """Used to cache encoder hidden state runs across same encoder sessions"""
        self.persist_encoder_hidden_states = True

        bs = encoder_hidden_states.shape[0] # in beam search mode, bs is batch_size * num_beams
        encoder_hidden_size = self.encoder_hidden_size
        encoder_length = TRTHFRunner.ENCODER_LENGTH
        if encoder_hidden_states.device == torch.device("cpu"):
            self.inputs["encoder_hidden_states"] = encoder_hidden_states.flatten().contiguous().cuda()
            self.bindings[1] = self.inputs["encoder_hidden_states"].data_ptr()
        else:
            self.inputs["encoder_hidden_states"][:bs * encoder_length * encoder_hidden_size] = encoder_hidden_states.flatten()

        # for dual-engine approach in kv cache mode, set these for the non-kv engine as well
        if self.use_non_kv_engine:
            if encoder_hidden_states.device == torch.device("cpu"):
                self.inputs_non_kv["encoder_hidden_states"] = encoder_hidden_states.flatten().contiguous().cuda()
                self.bindings_non_kv[1] = self.inputs_non_kv["encoder_hidden_states"].data_ptr()
            else:
                self.inputs_non_kv["encoder_hidden_states"][:bs * encoder_length * encoder_hidden_size] = encoder_hidden_states.flatten()

    def set_cross_attention_kv_cache_for_inference_cycle(self, past_key_values):
        """
        Used to cache encoder-decoder cross attention kv caches across same encoder sessions.

        Unlike self-attention cache, cross attention is constant during the decoding process, so we only need to set its bindings once at the first decoding step, and skip in all later steps (by self.persist_cross_attention_kv_cache flag)
        """
        self.persist_cross_attention_kv_cache = True

        bs = past_key_values[0][0].shape[0] # In beam search, it should be batch_size * num_beams
        encoder_length = TRTHFRunner.ENCODER_LENGTH if past_key_values is not None else 0
        num_heads = self.num_heads
        embedding_size_per_head = self.embedding_size_per_head

        for i in range(self.num_decoder_layers):

            # Set the binding shape of cross-attention KV caches, which should be (bs, num_heads, encoder_length, embedding_size_per_head).
            cross_attention_kv_shape = (bs, num_heads, encoder_length, embedding_size_per_head)
            cross_attention_kv_flatten_length = bs * num_heads * encoder_length * embedding_size_per_head

            if past_key_values is not None:
                if past_key_values[0][0].device == torch.device("cpu"):
                    self.inputs[f"past_key_values.{i}.encoder.key"] = past_key_values[i][2].flatten().contiguous().cuda()
                    self.bindings[self.kv_cache_binding_offset+4*i+2] = self.inputs[f"past_key_values.{i}.encoder.key"].data_ptr()

                    self.inputs[f"past_key_values.{i}.encoder.value"] = past_key_values[i][3].flatten().contiguous().cuda()
                    self.bindings[self.kv_cache_binding_offset+4*i+3] = self.inputs[f"past_key_values.{i}.encoder.value"].data_ptr()
                else:
                    self.inputs[f"past_key_values.{i}.encoder.key"][:cross_attention_kv_flatten_length] = past_key_values[i][2].flatten()

                    self.inputs[f"past_key_values.{i}.encoder.value"][:cross_attention_kv_flatten_length] = past_key_values[i][3].flatten()

            self.trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i + 2, cross_attention_kv_shape)
            self.trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i + 3, cross_attention_kv_shape)

    def set_return_device(self, return_device):
        """
        Sets the return device of the return via to(). Device name should be the same as torch devices: cuda, cpu, etc.
        This is used in our measurement code.
        """
        self.return_device = return_device
    
    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def forward(self, input_ids, encoder_hidden_states, *args, **kwargs):
        # Get the batch size.
        bs = input_ids.shape[0] # in beam search mode, bs is batch_size * num_beams

        # Get the maximum sequence length.
        max_length = self.max_sequence_length

        # Get the vocab size.
        vocab_size = BARTModelTRTConfig.VOCAB_SIZE[self.variant]

        # Actual sequence length of the input_ids and the output hidden_states.
        input_length = input_ids.shape[1]

        # The sequence length of the encoder_hidden_states.
        encoder_length = TRTHFRunner.ENCODER_LENGTH

        # Encoder hidden size
        encoder_hidden_size = self.encoder_hidden_size

        # KV cache flag
        use_cache = kwargs.get("use_cache", False)

        # flag for switch between dual engines
        non_kv_flag = self.use_non_kv_engine or (self.config.use_cache and kwargs.get("past_key_values") is None)
        # condition 1: during e2e decoding test, based on flag
        # condition 2: during single-step decoder test, depending on whether past_key_values is empty
        # note: without --enable-kv-cache arg, this flag should remain False

        # denote as variable to allow switch between non-kv and kv engines in kv cache mode
        trt_context = self.trt_context_non_kv if non_kv_flag else self.trt_context
        bindings = self.bindings_non_kv if non_kv_flag else self.bindings
        inputs = self.inputs_non_kv if non_kv_flag else self.inputs
        outputs = self.outputs_non_kv if non_kv_flag else self.outputs

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu")) or (self.return_device == "cpu")

        # We allocate the buffers using max_length, but we only need to first portion of it, so copy the data into the
        # first portion of the input buffer.
        # TODO: Could we just reuse input_ids' data_ptr() as the first binding when input_ids is already contiguous to
        # avoid an additional D2D?
        if is_cpu_mode:
            inputs["input_ids"] = input_ids.int().flatten().contiguous().cuda()
            bindings[0] = inputs["input_ids"].data_ptr()
        else:
            inputs["input_ids"][:bs * input_length] = input_ids.flatten()

        # Set the binding shape of input_ids, which should be (bs, input_length).
        trt_context.set_binding_shape(0, input_ids.shape)

        # If encoder hidden states have not been copied yet, copy the hidden states to the input buffer.
        if not self.persist_encoder_hidden_states:
            if is_cpu_mode:
                inputs["encoder_hidden_states"] = encoder_hidden_states.flatten().contiguous().cuda()
                bindings[1] = inputs["encoder_hidden_states"].data_ptr()
            else:
                inputs["encoder_hidden_states"][:bs * encoder_length * encoder_hidden_size] = encoder_hidden_states.flatten()

        # Set the binding shape of encoder_hidden_states, which should be (bs, encoder_length, encoder_hidden_size).
        trt_context.set_binding_shape(1, (bs, encoder_length, encoder_hidden_size))

        if self.config.use_cache: # or use_cache
            if non_kv_flag:
                # use non-kv engine, no additional inputs
                past_decoder_length = 0
            else:
                # use kv engine
                past_key_values = kwargs.get("past_key_values") # set by prepare_inputs_for_generation() during HF e2e pipeline; if only test decoder, need to set this field
                past_decoder_length = past_key_values[0][0].size(2)
                num_heads = self.num_heads
                embedding_size_per_head = self.embedding_size_per_head

                # for all BART variants, # encoder layers = # decoder layers, so just divide total # layers by 2
                for i in range(self.num_decoder_layers):

                    # Set the binding shape of self-attention KV caches, which should be (bs, num_heads, past_decoder_length, embedding_size_per_head).
                    self_attention_kv_shape = (bs, num_heads, past_decoder_length, embedding_size_per_head)
                    self_attention_kv_flatten_length = bs * num_heads * past_decoder_length * embedding_size_per_head

                    if past_key_values is not None:
                        if past_key_values[0][0].device == torch.device("cpu"):
                            inputs[f"past_key_values.{i}.decoder.key"] = past_key_values[i][0].flatten().contiguous().cuda()
                            bindings[self.kv_cache_binding_offset+4*i] = inputs[f"past_key_values.{i}.decoder.key"].data_ptr()

                            inputs[f"past_key_values.{i}.decoder.value"] = past_key_values[i][1].flatten().contiguous().cuda()
                            bindings[self.kv_cache_binding_offset+4*i+1] = inputs[f"past_key_values.{i}.decoder.value"].data_ptr()

                        else:
                            inputs[f"past_key_values.{i}.decoder.key"][:self_attention_kv_flatten_length] = past_key_values[i][0].flatten()

                            inputs[f"past_key_values.{i}.decoder.value"][:self_attention_kv_flatten_length] = past_key_values[i][1].flatten()

                    trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i, self_attention_kv_shape)
                    trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i + 1, self_attention_kv_shape)

                # Set the binding shape of cross-attention KV caches, which should be (bs, num_heads, encoder_length, embedding_size_per_head).
                # since cross-attention KV cache dimension is fixed, we set once at the start and skip later
                if not self.persist_cross_attention_kv_cache:
                    self.set_cross_attention_kv_cache_for_inference_cycle(past_key_values)

        # Launch TRT inference.
        # TODO: Could we use execute_v2_async() instead of execute_v2()? Current profiling shows that there is a
        # synchronization inside TRT's inference body, so this change may not be needed.
        trt_context.execute_v2(bindings=bindings)

        # We allocate the buffers using max_length, but we only need to first portion of it, so get only the first
        # portion of the output buffer and return that.
        # TODO: Could we construct a Torch tensor using given data_ptr() to avoid this D2D copy?
        hidden_states_output = outputs["hidden_states"]
        if is_cpu_mode:
            hidden_states_output = hidden_states_output.cpu()
        
        folded = hidden_states_output[:bs * input_length * vocab_size].view(bs, input_length, vocab_size)
        present_key_values = None
        if self.config.use_cache:
            # 1st decoding step and steps after handle the outputs in the same way
            present_key_values = ()
            curr_decoder_length = past_decoder_length + input_length
            num_heads = self.num_heads
            embedding_size_per_head = self.embedding_size_per_head

            for i in range(self.num_decoder_layers):

                self_attention_kv_shape = (bs, num_heads, curr_decoder_length, embedding_size_per_head)
                self_attention_kv_flatten_length = bs * num_heads * curr_decoder_length * embedding_size_per_head

                cross_attention_kv_shape = (bs, num_heads, encoder_length, embedding_size_per_head)
                cross_attention_kv_flatten_length = bs * num_heads * encoder_length * embedding_size_per_head

                self_attn_k_output = outputs[f"present_key_values.{i}.decoder.key"]
                self_attn_v_output = outputs[f"present_key_values.{i}.decoder.value"]
                if is_cpu_mode:
                    self_attn_k_output = self_attn_k_output.cpu()
                    self_attn_v_output = self_attn_v_output.cpu()

                self_attn_k = self_attn_k_output[:self_attention_kv_flatten_length].view(*self_attention_kv_shape)
                self_attn_v = self_attn_v_output[:self_attention_kv_flatten_length].view(*self_attention_kv_shape)

                cross_attn_k = None
                cross_attn_v = None
                if is_cpu_mode or non_kv_flag:
                    cross_attn_k_output = outputs[f"present_key_values.{i}.encoder.key"]
                    cross_attn_v_output = outputs[f"present_key_values.{i}.encoder.value"]
                    if is_cpu_mode:
                        cross_attn_k_output = cross_attn_k_output.cpu()
                        cross_attn_v_output = cross_attn_v_output.cpu()
                    cross_attn_k = cross_attn_k_output[:cross_attention_kv_flatten_length].view(*cross_attention_kv_shape)
                    cross_attn_v = cross_attn_v_output[:cross_attention_kv_flatten_length].view(*cross_attention_kv_shape)

                present_key_values += ((self_attn_k, self_attn_v, cross_attn_k, cross_attn_v), ) # make multi-dim tuple

        # Transfer predictions back from GPU to do greedy search
        return Seq2SeqLMOutput(logits=folded.to(self.return_device), past_key_values=present_key_values,)

    def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=None, **kwargs):
        # in HuggingFace generation_utils.py, this function will be called at each decoding step, before running the decoder's forward(). 
        # So we can use it to set the flag indicating if this is the 1st decoding step (use non-kv engine) or steps after (use kv engine)
        # cut decoder_input_ids if past is used (with past cache, only need to process the current length 1 token)
        # also, if past exists, it means we're at > 1 decoding steps thus set non-kv engine flag to False
        if past is not None:
            input_ids = input_ids[:, -1:]
            self.use_non_kv_engine = False

        ret = {
            "input_ids": input_ids,
            "encoder_hidden_states": kwargs["encoder_hidden_states"],
        }

        if self.config.use_cache:
            ret["use_cache"] = use_cache
            ret["past_key_values"] = past

        return ret


class BARTTRT(TRTInferenceCommand):
    def __init__(self):
        super().__init__(
            BARTModelTRTConfig,
            "Runs trt results for BART model.",
            BARTHuggingFace,
        )
        self.BART_trt_decoder = None
        self.BART_trt_encoder = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_trt_engine: bool = False,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        if self.BART_trt_encoder:
            self.BART_trt_encoder.release()
        if self.BART_trt_decoder:
            self.BART_trt_decoder.release()

        if not keep_trt_engine:
            self.BART_trt_encoder_engine.cleanup()
            self.BART_trt_decoder_engine.cleanup()
            # TODO: Avoid using workspace.metadata to handle non_kv removals.
            if workspace.metadata.other.kv_cache:
                self.BART_trt_decoder_engine_non_kv.cleanup()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

    def setup(self, encoder, decoder):
        self.BART_trt_encoder = encoder
        self.BART_trt_decoder = decoder

    def generate(
        self,
        input_ids,
        min_length: int = None,
        max_length: int = None,
        num_beams: int = 1,
        use_cache: bool = False,
        early_stopping: bool = True, # Deprecated
    ):
        batch_size = input_ids.shape[0]

        if max_length is None:
            max_length = BARTModelTRTConfig.MAX_OUTPUT_LENGTH[self.metadata.variant]
        
        if min_length is None:
            min_length = BARTModelTRTConfig.MIN_OUTPUT_LENGTH[self.metadata.variant]

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])
        logits_processor = LogitsProcessorList([
            NoRepeatNGramLogitsProcessor(BARTModelTRTConfig.NO_REPEAT_NGRAM_SIZE), 
            MinLengthLogitsProcessor(min_length, BARTModelTRTConfig.EOS_TOKEN_ID),
            ForcedBOSTokenLogitsProcessor(BARTModelTRTConfig.BOS_TOKEN_ID),
            ForcedEOSTokenLogitsProcessor(max_length, BARTModelTRTConfig.EOS_TOKEN_ID)
        ]) 

        decoder_input_ids = torch.full(
            (batch_size, 1), BARTModelTRTConfig.EOS_TOKEN_ID, dtype=torch.int32
        ).to("cuda")
        
        if num_beams == 1:
            G_LOGGER.info("Running full inference with greedy decoding...")
            encoder_last_hidden_state = self.BART_trt_encoder(input_ids=input_ids)
            self.BART_trt_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
            decoder_output = self.BART_trt_decoder.greedy_search(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache
            )
        else:
            G_LOGGER.info(f"Running full inference with beam search (num_beams = {num_beams}) decoding...")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device="cuda",
                do_early_stopping=early_stopping,
            )

            decoder_input_ids = expand_inputs_for_beam_search(decoder_input_ids, expand_size=num_beams)

            encoder_last_hidden_state = self.BART_trt_encoder(input_ids=input_ids)
            
            encoder_last_hidden_state = expand_inputs_for_beam_search(encoder_last_hidden_state, expand_size=num_beams)
            
            self.BART_trt_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
            decoder_output = self.BART_trt_decoder.beam_search(
                input_ids=decoder_input_ids,
                beam_scorer=beam_scorer,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache
            )

        self.reset_decoder_state()

        return decoder_output

    def reset_decoder_state(self):
        # During execute_inference, set_encoder_hidden_states_for_inference_cycle will be called in full_inference_greedy anyway to overwrite the saved encoder_hidden_states
        # But explicit reset this flag is still beneficial
        self.BART_trt_decoder.persist_encoder_hidden_states = False
        # Because the same decoder is used for different inputs, need to reset the flags for different inputs.
        # TODO: In BARTTRTDecoder, maybe a reset function is needed to capture this issue after each task.
        if self.metadata.other.kv_cache:
            self.BART_trt_decoder.persist_cross_attention_kv_cache = False
            self.BART_trt_decoder.use_non_kv_engine = self.metadata.other.kv_cache

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_mode: bool = False,
        benchmarking_args: BARTTRTBenchmarkingArgs = None,
    ) -> Union[NetworkResult, BenchmarkingResult]:
        if "mbart" not in metadata.variant:
            tokenizer = BartTokenizer.from_pretrained(metadata.variant)
        else:
            tokenizer = MBart50Tokenizer.from_pretrained(metadata.variant, src_lang="en_XX")

        # Prepare the input tokens and find output sequence length.
        if not benchmarking_mode:
            output_seq_len = BARTModelTRTConfig.MAX_OUTPUT_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, padding=True, return_tensors="pt").input_ids
        else:
            input_seq_len = benchmarking_args.input_seq_len
            output_seq_len = benchmarking_args.output_seq_len
     
            input_ids = torch.randint(0, BARTModelTRTConfig.VOCAB_SIZE[metadata.variant], (batch_size, input_seq_len))
        
        encoder_last_hidden_state, encoder_e2e_time = encoder_inference(
            self.BART_trt_encoder, input_ids, timing_profile
        )        
        
        # Need to feed the decoder a new empty input_ids for text generation. 
        decoder_output_len = output_seq_len // 2 if (not metadata.other.kv_cache) else 1
        decoder_input_ids = torch.full(
            (batch_size, decoder_output_len), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
        )

        _, decoder_e2e_time = decoder_inference(
            self.BART_trt_decoder,
            expand_inputs_for_beam_search(decoder_input_ids, num_beams) if num_beams > 1 else decoder_input_ids,
            expand_inputs_for_beam_search(encoder_last_hidden_state, num_beams) if num_beams > 1 else encoder_last_hidden_state,
            timing_profile,
            use_cache=metadata.other.kv_cache,
        )

        if num_beams == 1:
            decoder_output, full_e2e_runtime = full_inference_greedy(
                self.BART_trt_encoder,
                self.BART_trt_decoder,
                input_ids,
                tokenizer,
                timing_profile,
                max_length=output_seq_len,
                min_length=BARTModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
                batch_size=batch_size,
                use_cache=metadata.other.kv_cache,
            )
        else:
            decoder_output, full_e2e_runtime = full_inference_beam(
                self.BART_trt_encoder,
                self.BART_trt_decoder,
                input_ids,
                tokenizer,
                timing_profile,
                num_beams=num_beams,
                max_length=output_seq_len,
                min_length=BARTModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
                batch_size=batch_size,
                use_cache=metadata.other.kv_cache,
            )

        # Prepare runtime results.
        runtime=[
            NetworkRuntime(
                name=BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                runtime=decoder_e2e_time,
            ),
            NetworkRuntime(
                name=BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                runtime=encoder_e2e_time,
            ),
            NetworkRuntime(
                name=BARTModelTRTConfig.NETWORK_FULL_NAME,
                runtime=full_e2e_runtime,
            ),
        ]
        models=NetworkModels(
            torch=None,
            onnx=list(onnx_fpaths.values()),
            trt=[
                NetworkModel(
                    name=BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    fpath=self.BART_trt_decoder_engine.fpath,
                ),
                NetworkModel(
                    name=BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                    fpath=self.BART_trt_encoder_engine.fpath,
                ),
            ],
        )

        # Skip result checking in benchmarking mode since the input data is random.
        if benchmarking_mode:
            return BenchmarkingResult(median_runtime=runtime, models=models)

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.decode(
            decoder_output[-1, :], skip_special_tokens=True
        )

        if isinstance(semantic_outputs, list):
            semantic_outputs = " ".join(semantic_outputs).strip()

        return NetworkResult(
            input=inference_input,
            output_tensor=decoder_output,
            semantic_output=semantic_outputs,
            median_runtime=runtime,
            models=models,
        )

    def execute_calculate_perplexity(
        self,
        metadata: NetworkMetadata,
        encoder_input: str,
        decoder_input: str,
    ):
        if "mbart" not in metadata.variant:
            tokenizer = BartTokenizer.from_pretrained(metadata.variant)
        else:
            tokenizer = MBart50Tokenizer.from_pretrained(metadata.variant, src_lang="en_XX")

        encoder_input_ids = tokenizer([encoder_input], padding=True, return_tensors="pt").input_ids
        decoder_input_ids = tokenizer([decoder_input], padding=True, return_tensors="pt").input_ids

        perplexity = calculate_perplexity(
            self.BART_trt_encoder, self.BART_trt_decoder, tokenizer, encoder_input_ids, decoder_input_ids,
            BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
        )
        return perplexity

    def _setup_workspace(self, metadata: NetworkMetadata, working_directory: str) -> NNFolderWorkspace:
        return NNFolderWorkspace(
            self.frameworks_cmd.config.network_name, metadata, working_directory
        )

    def _download_models(
        self,
        workspace: NNFolderWorkspace,
        metadata: NetworkMetadata,
    ) -> Tuple[NetworkModel]:
        # No fpath provided for onnx files, download them from HuggingFace repo.
        return self.frameworks_cmd.generate_and_download_framework(
            metadata, workspace
        ).onnx

    def _setup_engines(
        self,
        metadata: NetworkMetadata,
        hash_onnx_fpath: Dict[str, NetworkModel],
        batch_size: int,
        num_beams: int,
        preview_dynamic_shapes: bool,
        benchmarking_args: BARTTRTBenchmarkingArgs = None,
        seq_tag: bool = False, # whether the benchmark engine tag format should be seq or max
    ) -> None:

        # Output networks shall not exceed number of network segments explicitly defined by configuration file.
        assert len(hash_onnx_fpath) == len(
            BARTModelTRTConfig.NETWORK_SEGMENTS
        ), "There should only be {} exported ONNX segments in BART model.".format(
            len(BARTModelTRTConfig.NETWORK_SEGMENTS)
        )

        decoder_onnx_fpath = hash_onnx_fpath[
            BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
        ].fpath
        encoder_onnx_fpath = hash_onnx_fpath[
            BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME
        ].fpath

        # Generate optimization profiles.
        # non-benchmarking mode: opt profile length is by default half of the max profile
        # benchmarking mode: user can specify opt and max profile by flags. If no additional benchmarking flags are provided, it will just use the non-benchmarking mode defaults
        max_sequence_length = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
        max_output_length = BARTModelTRTConfig.MAX_OUTPUT_LENGTH[metadata.variant]
        opt_input_seq_len = max_sequence_length // 2
        opt_output_seq_len = max_output_length // 2
        
        # benchmarking flags
        if benchmarking_args is not None:
            max_sequence_length = benchmarking_args.input_profile_max_len
            max_output_length = benchmarking_args.output_profile_max_len
            opt_input_seq_len = benchmarking_args.input_seq_len
            opt_output_seq_len = benchmarking_args.output_seq_len

        encoder_hidden_size = BARTModelTRTConfig.ENCODER_HIDDEN_SIZE[metadata.variant]

        encoder_profiles = [
            Profile().add(
                "input_ids",
                min=(batch_size, 1),
                opt=(batch_size, opt_input_seq_len),
                max=(batch_size, max_sequence_length),
            )
        ]

        # Set up the non kv engine, used for non-kv mode and kv mode generation phase (1st decoder run uses the non-kv profile to generate kv cache)
        dec_profiles_non_kv = Profile()
        
        # for beam search, decoder engine's inputs are expanded `num_beams` times
        # optimization profiles should be changed accordingly, but onnx models can be shared across greedy/beam because the first dim (batch size) is already a dynamic value, so no change needed in export.py
        if not metadata.other.kv_cache:
            dec_profiles_non_kv = dec_profiles_non_kv.add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, opt_output_seq_len),
                max=(batch_size * num_beams, max_output_length),
            )
        else:
            dec_profiles_non_kv = dec_profiles_non_kv.add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, 1),
                max=(batch_size * num_beams, 1),
            )

        dec_profiles_non_kv = dec_profiles_non_kv.add(
            "encoder_hidden_states",
            min=(batch_size * num_beams, 1, encoder_hidden_size),
            opt=(batch_size * num_beams, opt_input_seq_len, encoder_hidden_size),
            max=(batch_size * num_beams, max_sequence_length, encoder_hidden_size),
        )

        decoder_profiles_non_kv = [dec_profiles_non_kv]
        dec_profiles_kv = copy.deepcopy(dec_profiles_non_kv)
        if metadata.other.kv_cache:

            num_heads = BARTModelTRTConfig.NUMBER_OF_HEADS[metadata.variant]
            embedding_size_per_head = encoder_hidden_size // num_heads
            num_decoder_layers = BARTModelTRTConfig.NUMBER_OF_DECODER_LAYERS[metadata.variant]

            self_attention_profile = {
                "min": (batch_size * num_beams, num_heads, 0, embedding_size_per_head),
                "opt": (batch_size * num_beams, num_heads, opt_output_seq_len - 1, embedding_size_per_head),
                "max": (batch_size * num_beams, num_heads, max_output_length - 1, embedding_size_per_head),
            }
            cross_attention_profile = {
                "min": (batch_size * num_beams, num_heads, 1, embedding_size_per_head),
                "opt": (batch_size * num_beams, num_heads, opt_input_seq_len, embedding_size_per_head),
                "max": (batch_size * num_beams, num_heads, max_sequence_length, embedding_size_per_head),
            }

            for i in range(num_decoder_layers):
                dec_profiles_kv = dec_profiles_kv.add(
                    f"past_key_values.{i}.decoder.key",
                    **self_attention_profile
                )
                dec_profiles_kv = dec_profiles_kv.add(
                    f"past_key_values.{i}.decoder.value",
                    **self_attention_profile
                )
                dec_profiles_kv = dec_profiles_kv.add(
                    f"past_key_values.{i}.encoder.key",
                    **cross_attention_profile
                )
                dec_profiles_kv = dec_profiles_kv.add(
                    f"past_key_values.{i}.encoder.value",
                    **cross_attention_profile
                )  
            decoder_profiles_kv = [dec_profiles_kv]
        
        decoder_profiles = decoder_profiles_kv if (metadata.other.kv_cache) else decoder_profiles_non_kv

        # Convert ONNX models to TRT engines.
        if benchmarking_args is None:
            engine_tag = "bs{}".format(batch_size)
        # When user does not input any profile_max_len, use seq as tag, both max are config max
        elif seq_tag:
            engine_tag = "bs{}-inseq{}-outseq{}".format(batch_size, benchmarking_args.input_seq_len, benchmarking_args.output_seq_len)
        # When user input profile_max_len, reuse the engine for future use with different seq_len
        else:
            engine_tag = "bs{}-inmax{}-outmax{}".format(batch_size, benchmarking_args.input_profile_max_len, benchmarking_args.output_profile_max_len)

        if num_beams > 1:
            engine_tag += "-beam{}".format(num_beams)

        preview_features = []
        if preview_dynamic_shapes:
            preview_features = [PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]
            engine_tag += "-previewFasterDynamicShapes"

        self.BART_trt_encoder_engine = BARTEncoderONNXFile(
            encoder_onnx_fpath, metadata
        ).as_trt_engine(
            encoder_onnx_fpath + "-{}.engine".format(engine_tag).replace(f"-beam{num_beams}", ""), # encoder engine name not affected by beam search
            profiles=encoder_profiles,
            preview_features=preview_features
        )

        if not metadata.other.kv_cache:
            self.BART_trt_decoder_engine = BARTDecoderONNXFile(
                decoder_onnx_fpath, metadata
            ).as_trt_engine(
                os.path.splitext(decoder_onnx_fpath)[0] + "-{}.engine".format(engine_tag),
                profiles=decoder_profiles,
                preview_features=preview_features
            )
        else:
            decoder_root, decoder_fullname = os.path.split(decoder_onnx_fpath)
            # Split kv and non kv engines into separate folders to avoid weight overlap
            non_kv_root = os.path.join(decoder_root, "non-kv")
            kv_root = os.path.join(decoder_root, "kv")
            decoder_name, decoder_ext = os.path.splitext(decoder_fullname)
            decoder_onnx_non_kv_fpath = os.path.join(non_kv_root, decoder_name + "-non-kv" + decoder_ext)
            decoder_onnx_kv_fpath = os.path.join(kv_root, decoder_fullname)
            self.BART_trt_decoder_engine = BARTDecoderONNXFile(
                decoder_onnx_kv_fpath, metadata
            ).as_trt_engine(
                os.path.splitext(decoder_onnx_kv_fpath)[0] + "-{}.engine".format(engine_tag),
                profiles=decoder_profiles,
                preview_features=preview_features
            )
            # dual-engine approach: still need to setup non-kv engine in kv mode
            # note: workspace cleanup is not handled for these extra non-kv files
            self.BART_trt_decoder_engine_non_kv = BARTDecoderONNXFile(
                decoder_onnx_non_kv_fpath, metadata
            ).as_trt_engine(
                os.path.splitext(decoder_onnx_non_kv_fpath)[0] + "-{}.engine".format(engine_tag),
                profiles=decoder_profiles_non_kv,
                preview_features=preview_features
            )

        # Create BARTTRTEncoder and BARTTRTDecoder instances.
        tfm_config = BartConfig(
            use_cache=metadata.other.kv_cache,
            num_layers=BARTModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant],
        )
        self.BART_trt_encoder = BARTTRTEncoder(
            self.BART_trt_encoder_engine, metadata, tfm_config, batch_size=batch_size, benchmarking_args = benchmarking_args
        )
        self.BART_trt_decoder = BARTTRTDecoder(
            self.BART_trt_decoder_engine, metadata, tfm_config, batch_size=batch_size, num_beams=num_beams, benchmarking_args = benchmarking_args
        )

        if metadata.other.kv_cache:
            # switch between BARTTRTDecoder is impossible (becase HF decoding step is bound to one decoder). Therefore, we need to add the non-kv engines inside the same decoder --> decoder contains two TRT engines
            self.BART_trt_decoder.set_non_kv_engine_for_kv_mode(self.BART_trt_decoder_engine_non_kv)
    
    def run_trt(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Tuple[NetworkModel],
        network_input: List[str],
        working_directory: str,
        keep_trt_engine: bool,
        keep_onnx_model: bool,
        keep_torch_model: bool,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        args: object = None,
        benchmarking_mode: bool = False,
        preview_dynamic_shapes: bool = False,
        perplexity_reference: List[str] = None,
    ) -> Union[List[NetworkResult], BenchmarkingResult] :

        self.working_directory = working_directory
        workspace = self._setup_workspace(metadata, working_directory)

        # Keep onnx and Torch models if they are provided by users.
        if len(onnx_fpaths) == 0:
            onnx_fpaths = self._download_models(workspace, metadata)
        else:
            keep_onnx_model = True
            keep_torch_model = True

        hash_onnx_fpath = {v.name: v for v in onnx_fpaths}

        inference_results = []
        ppl_results = []
        try:
            if not benchmarking_mode:
                self._setup_engines(metadata, hash_onnx_fpath, batch_size, args.num_beams, preview_dynamic_shapes)
                for ninput in network_input:
                    inference_results.append(
                        self.execute_inference(
                            metadata, hash_onnx_fpath, ninput, timing_profile, batch_size, args.num_beams
                        )
                    )
                    self.reset_decoder_state()
                    
                if perplexity_reference is not None:
                    assert len(network_input) == len(perplexity_reference), "Encoder and decoder inputs must pair up"
                    
                    if metadata.other.kv_cache or (args.num_beams > 1):
                        G_LOGGER.warning("Skipping perplexity calculation for TRT with KV cache or beam search because it is not supported yet.")
                    else:
                        for ei, di in zip(network_input, perplexity_reference):
                            ppl_results.append(
                                self.execute_calculate_perplexity(metadata, ei, di)
                            )

            else:
               # Check that input_seq_len and output_seq_len is valid and within required range
                max_input_seq_len = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
                max_output_seq_len = BARTModelTRTConfig.MAX_OUTPUT_LENGTH[metadata.variant]

                seq_tag = args.input_profile_max_len is None and args.output_profile_max_len is None
                # User must provide either a pair of profile_max_len or a profile of seq_len for input/output
                if args.input_profile_max_len is None or args.output_profile_max_len is None:
                    if args.input_seq_len is None or args.output_seq_len is None:
                        assert False, "Please provide at least one pair of inputs: [input/output]_seq_len or [input/output]_profile_max_len"
                
                input_profile_max_len = setup_benchmark_arg(args.input_profile_max_len, "input_profile_max_len", max_input_seq_len)
                output_profile_max_len = setup_benchmark_arg(args.output_profile_max_len, "output_profile_max_len", max_output_seq_len)
                input_seq_len = setup_benchmark_arg(args.input_seq_len, "input_seq_len", input_profile_max_len // 2)
                output_seq_len = setup_benchmark_arg(args.output_seq_len, "output_seq_len", output_profile_max_len // 2)
                
                benchmarking_args = BARTTRTBenchmarkingArgs(input_seq_len, output_seq_len, input_profile_max_len, output_profile_max_len)

                # Assert to ensure the validity of benchmarking arguments
                assert benchmarking_args.input_seq_len <= benchmarking_args.input_profile_max_len, "input_seq_len should <= input_profile_max_len = {} for benchmarking mode".format(benchmarking_args.input_profile_max_len)
                assert benchmarking_args.output_seq_len <= benchmarking_args.output_profile_max_len, "output_seq_len should <= output_profile_max_len = {} for benchmarking mode".format(benchmarking_args.output_profile_max_len)
                assert benchmarking_args.input_profile_max_len <= max_input_seq_len, "Model config restrict input_profile_max_len <= {} for benchmark mode".format(max_input_seq_len)
                assert benchmarking_args.output_profile_max_len <= max_output_seq_len, "Model config restrict output_profile_max_len <= {} for benchmark mode".format(max_output_seq_len)
                
                self._setup_engines(metadata, hash_onnx_fpath, batch_size, args.num_beams, preview_dynamic_shapes, benchmarking_args, seq_tag)
                inference_results = self.execute_inference(
                    metadata, hash_onnx_fpath, None, timing_profile, batch_size, args.num_beams, True, benchmarking_args
                )

        finally:
            self.cleanup(workspace, keep_trt_engine, keep_onnx_model, keep_torch_model)

        return inference_results, ppl_results

    def add_args(self, parser) -> None:
        super().add_args(parser)
        polygraphy_group = parser.add_argument_group("polygraphy models")
        polygraphy_group.add_argument(
            "--onnx-decoder-fpath",
            default=None,
            help="Path to ONNX decoder. If None is supplied, scripts will generate them from HuggingFace.",
        )
        polygraphy_group.add_argument(
            "--onnx-encoder-fpath",
            default=None,
            help="Path to ONNX encoder. If None is supplied, scripts will generate them from HuggingFace.",
        )

    def args_to_network_models(self, args) -> List[NetworkModel]:
        # Check if both flags are given otherwise error out
        decoder_fpath_check = args.onnx_decoder_fpath is None
        encoder_fpath_check = args.onnx_encoder_fpath is None

        network_models = None
        if decoder_fpath_check and encoder_fpath_check:
            network_models = tuple()
        elif decoder_fpath_check or encoder_fpath_check:
            raise self._parser.error(
                "Both --onnx-decoder-fpath and --onnx-encoder-fpath must be given. Otherwise neither should be provided for script to download them."
            )
        else:
            onnx_decoder = NetworkModel(
                name=BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_decoder_fpath,
            )
            onnx_encoder = NetworkModel(
                name=BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                fpath=args.onnx_encoder_fpath,
            )
            network_models = (onnx_decoder, onnx_encoder)

        return network_models

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        frameworks_parsed_metadata = self.frameworks_cmd.args_to_network_metadata(args)

        return NetworkMetadata(
            variant=frameworks_parsed_metadata.variant,
            precision=Precision(fp16=args.fp16),
            other=frameworks_parsed_metadata.other,
        )


RUN_CMD = BARTTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
