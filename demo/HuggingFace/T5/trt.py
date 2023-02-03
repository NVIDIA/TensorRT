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
from transformers import T5Tokenizer, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput

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
from T5.frameworks import T5FHuggingFace
from T5.T5ModelConfig import T5ModelTRTConfig, T5TRTBenchmarkingArgs
from T5.measurements import decoder_inference, encoder_inference, full_inference, calculate_perplexity
from T5.export import T5DecoderONNXFile, T5EncoderONNXFile, T5DecoderTRTEngine, T5EncoderTRTEngine
from NNDF.models import TRTEngineFile
from NNDF.logger import G_LOGGER


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

class T5TRTEncoder(TRTHFRunner):
    """TRT implemented network interface that can be used to measure inference time."""

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1,
        benchmarking_args: T5TRTBenchmarkingArgs = None
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        # In benchmarking mode, the max_sequence_length should be the designated input_profile_max_len
        if benchmarking_args is not None and benchmarking_args.input_profile_max_len is not None:
            self.max_sequence_length = benchmarking_args.input_profile_max_len
        else: 
            self.max_sequence_length = hf_config.d_model
        self.encoder_hidden_size = hf_config.d_model
        self.main_input_name = "input_ids"
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

class T5TRTDecoder(TRTHFRunner):

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_args: T5TRTBenchmarkingArgs = None,
        cache_type = torch.float32
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        
        # In benchmarking mode, the max_sequence_length should be the user-provided input_profile_max_len
        if benchmarking_args is not None and benchmarking_args.input_profile_max_len is not None:
            self.max_input_length = benchmarking_args.input_profile_max_len
        else: 
            self.max_input_length = hf_config.d_model
        
        # Similarly, the max_output_length should be the user-provided output_profile_max_len
        if benchmarking_args is not None and benchmarking_args.output_profile_max_len is not None:
            self.max_output_length = benchmarking_args.output_profile_max_len
        else: 
            self.max_output_length = hf_config.d_model
        
        self.main_input_name = "input_ids"
        self.encoder_hidden_size = hf_config.d_model
        self.num_heads = hf_config.num_heads
        self.embedding_size_per_head = hf_config.d_kv
        self.num_decoder_layers = hf_config.num_decoder_layers
        self.profile_idx = 0
        self.bindings = [0] * self.trt_engine.num_bindings

        hidden_states_profile_length = self.max_output_length if not self.config.use_cache else 1
        # Construct buffer for hidden states outputs
        self.hidden_states = torch.zeros((self.batch_size * num_beams, hidden_states_profile_length, hf_config.vocab_size), dtype = torch.float32).cuda()
        self.bindings[self.trt_engine.get_binding_index("hidden_states")] = self.hidden_states.data_ptr()

        if self.config.use_cache:

            self.self_attention_cache = {}
            self.cross_attention_cache = {}

            # We are using cached cross attention, and not outputing redundant cross attention information. We only output self attention cache increment
            self_attention_kv_shape = (self.batch_size * num_beams, self.num_heads, self.max_output_length - 1, self.embedding_size_per_head)
            cross_attention_kv_shape = (self.batch_size * num_beams, self.num_heads, self.max_input_length, self.embedding_size_per_head)

            # Set self attention kv cache shape and type
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:
                    # Allocate self attention buffer. The buffer is used both as inputs and outputs
                    self_attention_name = f"key_values.{i}.decoder.{code}"
                    input_buffer = torch.zeros(self_attention_kv_shape, dtype = cache_type).cuda()
                    input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                    self.self_attention_cache[self_attention_name] = input_buffer
                    self.bindings[input_idx] = input_buffer.data_ptr()

                    output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)
                    self.bindings[output_idx] = input_buffer.data_ptr()

                    # Allocate cross attention buffer
                    cross_attention_past_name = f"past_key_values.{i}.encoder.{code}"
                    cross_attention_buffer = torch.zeros(cross_attention_kv_shape, dtype = cache_type).cuda()
                    cross_attention_idx = self.trt_engine.get_binding_index(cross_attention_past_name)
                    self.cross_attention_cache[cross_attention_past_name] = cross_attention_buffer
                    self.bindings[cross_attention_idx] = cross_attention_buffer.data_ptr()

            self.kv_cache_binding_offset = 2 # 0: input_ids, 1: encoder_hidden_states, kv cache input indices start from 2
            self.past_decoder_length = 0

        # Optimization bit
        self.persist_encoder_hidden_states = False
        self.encoder_hidden_states = None
        self.persist_cross_attention_kv_cache = False

        self.return_device = "cuda"
        self.variant = network_metadata.variant # record variant name to later index the vocab_size in forward()
    
    def set_encoder_hidden_states_for_inference_cycle(self, encoder_hidden_states):
        """Used to cache encoder hidden state runs across same encoder sessions"""

        if encoder_hidden_states.device == torch.device("cpu"):
            self.encoder_hidden_states = encoder_hidden_states.cuda()
        else:
            self.encoder_hidden_states = encoder_hidden_states
        
        self.bindings[1] = self.encoder_hidden_states.data_ptr()
        self.persist_encoder_hidden_states = True

    def set_cross_attention_kv_cache_engine(self, cross_attention_kv_generator):
        self.cross_attention_kv_generator = cross_attention_kv_generator
        with open(self.cross_attention_kv_generator.fpath, "rb") as f:
            trt_runtime = trt.Runtime(self.trt_logger)
            self.cross_attention_kv_generator_trt_engine = trt_runtime.deserialize_cuda_engine(f.read())
            self.cross_attention_kv_generator_trt_context = self.cross_attention_kv_generator_trt_engine.create_execution_context()

    def set_cross_attention_kv_cache_for_inference_cycle(self, encoder_hidden_states):
        """
        Used to cache encoder-decoder cross attention kv caches across same encoder sessions.

        Unlike self-attention cache, cross attention is constant during the decoding process, so we only need to set its bindings once at the first decoding step, and skip in all later steps (by self.persist_cross_attention_kv_cache flag)
        """
        self.cross_attention_kv_generator_trt_context.set_binding_shape(0, encoder_hidden_states.shape)
        bindings = [None] * self.cross_attention_kv_generator_trt_engine.num_bindings
        bindings[0] = encoder_hidden_states.data_ptr()
        assert self.cross_attention_kv_generator_trt_context.all_binding_shapes_specified

        cross_attention_kv_shape_output = (encoder_hidden_states.shape[0], self.num_heads, self.max_input_length, self.embedding_size_per_head)
        # Cross attention cache as outputs
        for i in range(self.num_decoder_layers):
            bindings[2*i+1] = self.cross_attention_cache[f"past_key_values.{i}.encoder.key"].data_ptr()
            bindings[2*i+2] = self.cross_attention_cache[f"past_key_values.{i}.encoder.value"].data_ptr()

        self.cross_attention_kv_generator_trt_context.execute_v2(bindings=bindings)
        self.persist_cross_attention_kv_cache = True

    def set_return_device(self, return_device):
        """
        Sets the return device of the return via to(). Device name should be the same as torch devices: cuda, cpu, etc.
        This is used in our measurement code.
        """
        self.return_device = return_device
        self.device = return_device
    
    def _reorder_cache(self, past, beam_idx):
        # Reference: https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/t5/modeling_t5.html
        # Note that for BART, this function is static, but for T5, it is not
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            print("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                if layer_past_state is not None:
                    # need to set correct `past` for each of the four key / value states
                    reordered_layer_past_states = reordered_layer_past_states + (
                        layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                    )
                else:
                    reordered_layer_past_states = reordered_layer_past_states + (None,)

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def forward(self, input_ids, encoder_hidden_states, *args, **kwargs):
        # Get the batch size.
        bs = input_ids.shape[0] # in beam search mode, bs is batch_size * num_beams

        # Actual sequence length of the input_ids and the output hidden_states.
        input_length = input_ids.shape[1]

        # The sequence length of the encoder_hidden_states.
        encoder_length = TRTHFRunner.ENCODER_LENGTH

        is_cpu_mode = (input_ids.device == torch.device("cpu")) or (self.return_device == "cpu")

        if is_cpu_mode:
            input_ids = input_ids.int().cuda()

        # input_ids needs to be an in int type.
        self.bindings[0] = input_ids.int().data_ptr()
        self.trt_context.set_binding_shape(0, input_ids.shape)

        # If encoder hidden states have not been copied yet, copy the hidden states to the input buffer.
        if not self.persist_encoder_hidden_states:
            self.set_encoder_hidden_states_for_inference_cycle(encoder_hidden_states)

        self.trt_context.set_binding_shape(1, self.encoder_hidden_states.shape)

        if self.config.use_cache:
            if (kwargs.get("past_key_values") is None):
                self.past_decoder_length = 0
            if not self.persist_cross_attention_kv_cache:
                self.set_cross_attention_kv_cache_for_inference_cycle(encoder_hidden_states)
                cross_attention_kv_shape = (bs, self.num_heads, encoder_length, self.embedding_size_per_head)
                for i in range(self.num_decoder_layers):
                    self.trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i + 2, cross_attention_kv_shape)
                    self.trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i + 3, cross_attention_kv_shape)

            # When switching trt profiles, the binding shape needs to be reset, so we set binding shape at each forward pass
            self_attention_kv_shape = (bs, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)
            for i in range(self.num_decoder_layers):
                self.trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i, self_attention_kv_shape)
                self.trt_context.set_binding_shape(self.kv_cache_binding_offset+4*i + 1, self_attention_kv_shape)

        # Launch TRT inference.
        assert self.trt_context.all_binding_shapes_specified
        self.trt_context.execute_v2(bindings=self.bindings)
        
        logits = self.hidden_states[:,:input_length,:]
        if is_cpu_mode:
            logits = logits.cpu()

        present_key_values = None
        if self.config.use_cache:
            present_key_values = ()
            num_heads = self.num_heads
            embedding_size_per_head = self.embedding_size_per_head

            for i in range(self.num_decoder_layers):
                self_attention_k_output = self.self_attention_cache[f"key_values.{i}.decoder.key"]
                self_attention_v_output = self.self_attention_cache[f"key_values.{i}.decoder.value"]
                if is_cpu_mode:
                    self_attention_k_output = self_attention_k_output.cpu()
                    self_attention_v_output = self_attention_v_output.cpu()

                present_key_values += ((self_attention_k_output, self_attention_v_output),)

            self.past_decoder_length += 1

        # Transfer predictions back from GPU to do greedy search
        return Seq2SeqLMOutput(logits=logits.to(self.return_device), past_key_values=present_key_values,)

    def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=None, **kwargs):
        # In HuggingFace generation_utils.py, this function will be called at each decoding step, before running the decoder's forward().
        
        if past is not None:
            input_ids = input_ids[:, -1:]

        ret = {
            "input_ids": input_ids,
            "encoder_hidden_states": kwargs["encoder_outputs"].get("last_hidden_state"),
        }

        if self.config.use_cache:
            ret["use_cache"] = use_cache
            ret["past_key_values"] = past
        
        return ret

    def reset(self):
        '''
        You should always call this function after a use case because T5TRTDecoder does not clear the cached encoder_hidden_states or cross_attention itself.
        '''
        self.persist_encoder_hidden_states = False
        self.encoder_hidden_states = None
        if self.config.use_cache:
            self.persist_cross_attention_kv_cache = False

class T5TRT(TRTInferenceCommand):
    def __init__(self):
        super().__init__(
            T5ModelTRTConfig,
            "Runs trt results for T5 model.",
            T5FHuggingFace,
        )
        self.t5_trt_decoder = None
        self.t5_trt_encoder = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_trt_engine: bool = False,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        if self.t5_trt_encoder:
            self.t5_trt_encoder.release()
        if self.t5_trt_decoder:
            self.t5_trt_decoder.release()

        if not keep_trt_engine:
            self.t5_trt_encoder_engine.cleanup()
            self.t5_trt_decoder_engine.cleanup()
            # TODO: Avoid using workspace.metadata to handle additional removals.
            if workspace.metadata.other.kv_cache:
                self.t5_trt_cross_attention_kv_generator.cleanup()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

    def generate(
        self,
        input_ids,
        min_length: int = None,
        max_length: int = None,
        num_beams: int = 1,
        use_cache: bool = False,
        early_stopping: bool = True,
    ):
        batch_size = input_ids.shape[0]
        hf_config = self.t5_trt_decoder.config

        if max_length is None:
            max_length = T5ModelTRTConfig.MAX_OUTPUT_LENGTH[self.metadata.variant]
        
        if min_length is None:
            min_length = T5ModelTRTConfig.MIN_OUTPUT_LENGTH[self.metadata.variant]
        
        encoder_last_hidden_state = self.t5_trt_encoder(input_ids=input_ids).to("cuda")

        decoder_output = self.t5_trt_decoder.generate(
            input_ids,
            max_length = max_length,
            min_length = min_length,
            num_beams = num_beams,
            early_stopping = early_stopping,
            eos_token_id = self.t5_trt_decoder.config.eos_token_id,
            pad_token_id = self.t5_trt_decoder.config.pad_token_id,
            use_cache = use_cache,
            encoder_outputs = BaseModelOutput(last_hidden_state = encoder_last_hidden_state),
        )

        self.t5_trt_decoder.reset()
        return decoder_output

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_mode: bool = False,
        benchmarking_args: T5TRTBenchmarkingArgs = None,
    ) -> Union[NetworkResult, BenchmarkingResult]:

        tokenizer = T5Tokenizer.from_pretrained(metadata.variant)
        hf_config = self.t5_trt_decoder.config
        # Prepare the input tokens and find out output sequence length.
        if not benchmarking_mode:
            output_seq_len = T5ModelTRTConfig.MAX_OUTPUT_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, padding=True, return_tensors="pt").input_ids
        else:
            input_seq_len = benchmarking_args.input_seq_len
            output_seq_len = benchmarking_args.output_seq_len
     
            input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, input_seq_len))

        encoder_last_hidden_state, encoder_e2e_time = encoder_inference(
            self.t5_trt_encoder, input_ids, timing_profile
        )
        
        # Need to feed the decoder a new empty input_ids for text generation. 
        decoder_output_len = output_seq_len // 2 if (not metadata.other.kv_cache) else 1

        decoder_input_ids = torch.full(
            (batch_size, decoder_output_len), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
        )

        _, decoder_e2e_time = decoder_inference(
            self.t5_trt_decoder,
            expand_inputs_for_beam_search(decoder_input_ids, num_beams) if num_beams > 1 else decoder_input_ids,
            expand_inputs_for_beam_search(encoder_last_hidden_state, num_beams) if num_beams > 1 else encoder_last_hidden_state,
            timing_profile,
            use_cache=metadata.other.kv_cache,
        )

        self.t5_trt_decoder.reset()

        decoder_output, full_e2e_runtime = full_inference(
            self.t5_trt_encoder,
            self.t5_trt_decoder,
            input_ids,
            tokenizer,
            timing_profile,
            max_length=output_seq_len,
            min_length=T5ModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
            batch_size=batch_size,
            use_cache=metadata.other.kv_cache,
            num_beams = num_beams,
        )

        # Prepare runtime results.
        runtime = [
            NetworkRuntime(
                name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                runtime=decoder_e2e_time,
            ),
            NetworkRuntime(
                name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                runtime=encoder_e2e_time,
            ),
            NetworkRuntime(
                name=T5ModelTRTConfig.NETWORK_FULL_NAME,
                runtime=full_e2e_runtime,
            ),
        ]
        models=NetworkModels(
            torch=None,
            onnx=list(onnx_fpaths.values()),
            trt=[
                NetworkModel(
                    name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    fpath=self.t5_trt_decoder_engine.fpath,
                ),
                NetworkModel(
                    name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                    fpath=self.t5_trt_encoder_engine.fpath,
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
        tokenizer = T5Tokenizer.from_pretrained(metadata.variant)
        encoder_input_ids = tokenizer([encoder_input], padding=True, return_tensors="pt").input_ids
        decoder_input_ids = tokenizer([decoder_input], padding=True, return_tensors="pt").input_ids

        perplexity = calculate_perplexity(
            self.t5_trt_encoder, self.t5_trt_decoder, tokenizer, encoder_input_ids, decoder_input_ids,
            T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
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
        benchmarking_args: T5TRTBenchmarkingArgs = None,
        seq_tag: bool = False, # whether the benchmark engine tag format should be seq or max
    ) -> None:

        # Output networks shall not exceed number of network segments explicitly defined by configuration file.
        assert len(hash_onnx_fpath) == len(
            T5ModelTRTConfig.NETWORK_SEGMENTS
        ), "There should only be {} exported ONNX segments in T5 model.".format(
            len(T5ModelTRTConfig.NETWORK_SEGMENTS)
        )

        decoder_onnx_fpath = hash_onnx_fpath[
            T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
        ].fpath
        encoder_onnx_fpath = hash_onnx_fpath[
            T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME
        ].fpath

        # Use HuggingFace T5Config to set up parameter instead of harc-coded values.
        hf_config = T5Config.from_pretrained(
            metadata.variant,
            use_cache=metadata.other.kv_cache
        )

        # Generate optimization profiles.
        # non-benchmarking mode: opt profile length is by default half of the max profile
        # benchmarking mode: user can specify opt and max profile by flags. If no additional benchmarking flags are provided, it will just use the non-benchmarking mode defaults
        max_input_length = hf_config.d_model
        max_output_length = hf_config.d_model
        opt_input_seq_len = max_input_length // 2
        opt_output_seq_len = max_output_length // 2
        
        # benchmarking flags
        if benchmarking_args is not None:
            max_input_length = benchmarking_args.input_profile_max_len
            max_output_length = benchmarking_args.output_profile_max_len
            opt_input_seq_len = benchmarking_args.input_seq_len
            opt_output_seq_len = benchmarking_args.output_seq_len

        encoder_hidden_size = hf_config.d_model

        encoder_profiles = [
            Profile().add(
                "input_ids",
                min=(batch_size, 1),
                opt=(batch_size, opt_input_seq_len),
                max=(batch_size, max_input_length),
            )
        ]

        # Set up the non kv engine, used for non-kv mode and kv mode generation phase (1st decoder run uses the non-kv profile to generate kv cache)
        dec_profiles = Profile()
        
        # for beam search, decoder engine's inputs are expanded `num_beams` times
        # optimization profiles should be changed accordingly, but onnx models can be shared across greedy/beam because the first dim (batch size) is already a dynamic value, so no change needed in export.py
        if not hf_config.use_cache:
            dec_profiles = dec_profiles.add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, opt_output_seq_len),
                max=(batch_size * num_beams, max_output_length),
            )
        else:
            dec_profiles = dec_profiles.add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, 1),
                max=(batch_size * num_beams, 1),
            )

        dec_profiles = dec_profiles.add(
            "encoder_hidden_states",
            min=(batch_size * num_beams, 1, encoder_hidden_size),
            opt=(batch_size * num_beams, opt_input_seq_len, encoder_hidden_size),
            max=(batch_size * num_beams, max_input_length, encoder_hidden_size),
        )
        
        if hf_config.use_cache:

            num_heads = hf_config.num_heads
            embedding_size_per_head = hf_config.d_kv
            num_decoder_layers = hf_config.num_decoder_layers
            # Use TensorRT Zero-Tensor feature for the 1st decoder run, self attention is growing with increasing sequence.
            self_attention_profile = {
                "min": (batch_size * num_beams, num_heads, 0, embedding_size_per_head),
                "opt": (batch_size * num_beams, num_heads, opt_output_seq_len - 1, embedding_size_per_head),
                "max": (batch_size * num_beams, num_heads, max_output_length - 1, embedding_size_per_head),
            }

            # Cross attention kv cache does not change during single decoder iteration.
            cross_attention_profile = {
                "min": (batch_size * num_beams, num_heads, 1, embedding_size_per_head),
                "opt": (batch_size * num_beams, num_heads, opt_input_seq_len, embedding_size_per_head),
                "max": (batch_size * num_beams, num_heads, max_input_length, embedding_size_per_head),
            }

            for i in range(num_decoder_layers):
                dec_profiles = dec_profiles.add(
                    f"past_key_values.{i}.decoder.key",
                    **self_attention_profile
                ).add(
                    f"past_key_values.{i}.decoder.value",
                    **self_attention_profile
                ).add(
                    f"past_key_values.{i}.encoder.key",
                    **cross_attention_profile
                ).add(
                    f"past_key_values.{i}.encoder.value",
                    **cross_attention_profile
                )
        
        decoder_profiles = [dec_profiles]

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

        self.t5_trt_encoder_engine = T5EncoderONNXFile(
            encoder_onnx_fpath, metadata
        ).as_trt_engine(
            os.path.splitext(encoder_onnx_fpath)[0] + "-{}.engine".format(engine_tag).replace(f"-beam{num_beams}", ""), # encoder engine name not affected by beam search
            profiles=encoder_profiles,
            preview_features=preview_features
        )

        self.t5_trt_decoder_engine = T5DecoderONNXFile(
            decoder_onnx_fpath, metadata
        ).as_trt_engine(
            os.path.splitext(decoder_onnx_fpath)[0] + "-{}.engine".format(engine_tag),
            profiles=decoder_profiles,
            preview_features=preview_features
        )

        # Create T5TRTEncoder and T5TRTDecoder instances.
        self.t5_trt_encoder = T5TRTEncoder(
            self.t5_trt_encoder_engine, metadata, hf_config, batch_size=batch_size, benchmarking_args=benchmarking_args
        )
        self.t5_trt_decoder = T5TRTDecoder(
            self.t5_trt_decoder_engine, metadata, hf_config, batch_size=batch_size, num_beams=num_beams, benchmarking_args=benchmarking_args
        )

        if metadata.other.kv_cache:
            # Set up context phase profile. Context phase will use encoder_hidden_states to generate cross attention kv cache.
            cross_attention_kv_generation_profiles = [Profile().add(
                "encoder_hidden_states",
                min=(batch_size * num_beams, 1, encoder_hidden_size),
                opt=(batch_size * num_beams, opt_input_seq_len, encoder_hidden_size),
                max=(batch_size * num_beams, max_input_length, encoder_hidden_size),
            )]
            decoder_folder, decoder_name = os.path.split(decoder_onnx_fpath)
            decoder_name, decoder_ext = os.path.splitext(decoder_name)
            decoder_onnx_fpath_kv_generator = os.path.join(decoder_folder, "cross_attention_kv_generator", decoder_name + "-cross_attention_kv_generator" + decoder_ext)
            self.t5_trt_cross_attention_kv_generator = T5DecoderONNXFile(
                decoder_onnx_fpath_kv_generator, metadata
            ).as_trt_engine(
                os.path.splitext(decoder_onnx_fpath_kv_generator)[0] + "-{}.engine".format(engine_tag),
                profiles=cross_attention_kv_generation_profiles,
                preview_features=preview_features
            )

            self.t5_trt_decoder.set_cross_attention_kv_cache_engine(self.t5_trt_cross_attention_kv_generator)

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
                    self.t5_trt_decoder.reset()
                        
                if perplexity_reference is not None:
                    assert len(network_input) == len(perplexity_reference), "Encoder and decoder inputs must pair up"
                    if metadata.other.kv_cache or (args.num_beams > 1):
                        G_LOGGER.warning("Skipping perplexity calculation for TRT with KV cache or beam search because it is not supported yet.")
                    else:
                        for ei, di in zip(network_input, perplexity_reference):
                            ppl_results.append(
                                self.execute_calculate_perplexity(metadata, ei, di)
                            )
                            self.t5_trt_decoder.reset()

            else:
                # Check that input_seq_len and output_seq_len is valid and within required range
                max_input_seq_len = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
                max_output_seq_len = T5ModelTRTConfig.MAX_OUTPUT_LENGTH[metadata.variant]

                seq_tag = args.input_profile_max_len is None and args.output_profile_max_len is None
                # User must provide either a pair of profile_max_len or a profile of seq_len for input/output
                if args.input_profile_max_len is None or args.output_profile_max_len is None:
                    if args.input_seq_len is None or args.output_seq_len is None:
                        assert False, "Please provide at least one pair of inputs: [input/output]_seq_len or [input/output]_profile_max_len"
                
                input_profile_max_len = setup_benchmark_arg(args.input_profile_max_len, "input_profile_max_len", max_input_seq_len)
                output_profile_max_len = setup_benchmark_arg(args.output_profile_max_len, "output_profile_max_len", max_output_seq_len)
                input_seq_len = setup_benchmark_arg(args.input_seq_len, "input_seq_len", input_profile_max_len // 2)
                output_seq_len = setup_benchmark_arg(args.output_seq_len, "output_seq_len", output_profile_max_len // 2)
                
                benchmarking_args = T5TRTBenchmarkingArgs(input_seq_len, output_seq_len, input_profile_max_len, output_profile_max_len)

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
                name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_decoder_fpath,
            )
            onnx_encoder = NetworkModel(
                name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
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


RUN_CMD = T5TRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
