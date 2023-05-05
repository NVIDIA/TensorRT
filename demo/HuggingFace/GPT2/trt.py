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
import copy
from typing import Dict, List, Tuple, Union

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# numpy
import numpy as np

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch

# huggingface
from transformers import GPT2Tokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
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

from NNDF.tensorrt_utils import TRTNativeRunner, TRTPolygraphyRunner, set_kv_data, allocate_binding_buffer, setup_benchmark_arg
from NNDF.torch_utils import expand_inputs_for_beam_search
from GPT2.frameworks import GPT2HuggingFace
from NNDF.general_utils import NNFolderWorkspace
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig, GPT2BenchmarkingArgs, GPT2TRTBenchmarkingArgs
from GPT2.measurements import gpt2_inference, full_inference, calculate_perplexity
from GPT2.export import GPT2ONNXFile, GPT2TRTEngine
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

class GPT2TRTDecoder(TRTHFRunner):
    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_args: GPT2BenchmarkingArgs = None
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        self.network_metadata = network_metadata
        self.data_type = torch.float32 if not network_metadata.precision.fp16 else torch.float16
        # In benchmarking mode, if input_profile_max is provided, should use that as max_sequence_length
        if benchmarking_args is not None:
            if benchmarking_args.input_profile_max_len is not None:
                self.max_input_length = benchmarking_args.input_profile_max_len
            else:
                self.max_input_length = hf_config.n_positions
        # In non-benchmarking mode, we are provided a text generation task. We need to use the max_length as max sequence length
        else:
            self.max_sequence_length = GPT2ModelTRTConfig.MAX_LENGTH[network_metadata.variant]

        # Similarly, the max_output_length should be the user-provided output_profile_max_len if provided
        if benchmarking_args is not None and benchmarking_args.output_profile_max_len is not None:
            self.max_output_length = benchmarking_args.output_profile_max_len
        else:
            self.max_output_length = self.max_sequence_length

        self.main_input_name = "input_ids"
        self.num_heads = self.config.n_head
        self.embedding_size_per_head = self.config.n_embd // self.num_heads
        self.num_decoder_layers = self.config.n_layer

        self.profile_idx = 0
        self.bindings = [0] * self.trt_engine.num_bindings
        self.logits = torch.zeros((self.batch_size * num_beams, self.max_output_length, hf_config.vocab_size), dtype = self.data_type).cuda()
        self.bindings[self.trt_engine.get_binding_index("logits")] = self.logits.data_ptr()
        # This will be used to calculate the offset for each binding
        self.num_bindings = self.trt_engine.num_bindings // 2 if self.config.use_cache else self.trt_engine.num_bindings

        if self.config.use_cache:
            self.bindings[self.trt_engine.get_binding_index("logits") + self.num_bindings] = self.logits.data_ptr()
            
            # Setting input and output the same does not work for GPT2. Needs separate cache and copy the memory address after each iteration
            self.self_attention_cache_1 = {}
            self.self_attention_cache_2 = {}

            self_attention_kv_shape = (self.batch_size * num_beams, self.num_heads, self.max_output_length - 1, self.embedding_size_per_head)

            # Set kv cache shape and type
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:

                    self_attention_name = f"key_values.{i}.decoder.{code}"
                    kv_buffer_1 = torch.zeros(self_attention_kv_shape, dtype = self.data_type).cuda()
                    kv_buffer_2 = torch.zeros(self_attention_kv_shape, dtype = self.data_type).cuda()
                    self.self_attention_cache_1[self_attention_name] = kv_buffer_1
                    self.self_attention_cache_2[self_attention_name] = kv_buffer_2

                    input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                    output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)
                    
                    self.bindings[input_idx] = kv_buffer_1.data_ptr() # Generation phase
                    self.bindings[output_idx] = kv_buffer_2.data_ptr()  

                    # Context mode will always use buffer 1 as output
                    self.bindings[input_idx + self.num_bindings] = 0 # Context phase, should be 0
                    self.bindings[output_idx + self.num_bindings] = kv_buffer_1.data_ptr()

            self.kv_cache_binding_offset = 1 # 0: input_ids, kv cache input indices start from 1
            self.past_decoder_length = 0
            self.use_cache_1_as_input = True
            self._set_context_mode_trt_context()
        
        self.context_mode = self.config.use_cache
        self.return_device = torch.device('cuda')
        self.device = torch.device('cuda')

    def reset(self):
        '''
        Resets the input specific fields after finishing a task.
        '''
        self.context_mode = self.config.use_cache
    
    def _switch_input_output_binding(self):
        '''
        For kv cache mode, switch input and output pointers to avoid data concurrency issue and D2D copy
        '''
        # When context mode (output in cache 1) and cache 1 is used as inputs, no need to switch bindings
        if not (self.use_cache_1_as_input and self.context_mode):
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:
                    self_attention_name = f"key_values.{i}.decoder.{code}"
                    input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                    output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)

                    # Switch generation mode kv cache bindings
                    temp = self.bindings[output_idx]
                    self.bindings[output_idx] = self.bindings[input_idx]
                    self.bindings[input_idx] = temp
            self.use_cache_1_as_input = not self.use_cache_1_as_input
 
    def prepare_inputs_for_generation(self, input_ids, past = None, use_cache = None, **kwargs):
        # TODO: add position_ids, token_type_ids support
        if past is not None:
            input_ids = input_ids[:, -1:]
            self.context_mode = False
        else:
            self.context_mode = self.config.use_cache
        
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def set_return_device(self, return_device):
        """
        Sets the return device of the return via to(). Device name should be the same as torch devices: cuda, cpu, etc.
        This is used in our measurement code.
        """
        self.return_device = return_device

    def _reorder_cache(self, past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
    
    def _set_context_mode_trt_context(self):
        # Create TRT context for context mode (1st decoder run) with optimization profile = 1
        self.context_trt_context = self.trt_engine.create_execution_context()
        self.context_trt_context.active_optimization_profile = 1


    def forward(self, input_ids, *args, **kwargs):
        bs = input_ids.shape[0]
        input_length = input_ids.shape[1]

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu")) or (self.return_device == "cpu")

        if is_cpu_mode:
            input_ids = input_ids.int().cuda()
        
        # Set the binding shape of input_ids, which should be (bs, input_length).
        if not self.context_mode:
            self.bindings[0] = input_ids.int().data_ptr()
            self.trt_context.set_binding_shape(0, input_ids.shape)
        else:
            self.bindings[self.num_bindings] = input_ids.int().data_ptr()
            self.context_trt_context.set_binding_shape(self.num_bindings, input_ids.shape)

        if self.config.use_cache:            
            if self.context_mode:
                self.past_decoder_length = 0

            self_attention_kv_shape = (bs, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)

            for i in range(self.num_decoder_layers):
                if not self.context_mode:
                    # Optimization Profile 1 is generation phase with no kv inputs
                    self.trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i, self_attention_kv_shape)
                    self.trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + 1, self_attention_kv_shape)
                else:
                    # Optimization Profile 0 is context phase with kv inputs
                    self.context_trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + self.num_bindings, self_attention_kv_shape)
                    self.context_trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + 1 + self.num_bindings, self_attention_kv_shape)
                    
        # Launch TRT inference.
        if not self.context_mode:
            assert self.trt_context.all_binding_shapes_specified
            self.trt_context.execute_v2(bindings=self.bindings)
        else:
            assert self.context_trt_context.all_binding_shapes_specified
            self.context_trt_context.execute_v2(bindings=self.bindings)
        
        # For bs > 1, this is required, so cannnot avoid this D2D copy
        logits_length = bs * input_length * self.config.vocab_size
        logits = self.logits.flatten()[:logits_length].view(bs, input_length, self.config.vocab_size)

        if is_cpu_mode:
            logits = logits.cpu()

        present_key_values = None
        if self.config.use_cache:
            self.past_decoder_length += input_length

            present_key_values = ()
            self_attention_cache = self.self_attention_cache_1 if self.use_cache_1_as_input or (self.profile_idx == 0) else self.self_attention_cache_2
            
            for i in range(self.num_decoder_layers):

                self_attention_k_output = self_attention_cache[f"key_values.{i}.decoder.key"]
                self_attention_v_output = self_attention_cache[f"key_values.{i}.decoder.value"]

                if is_cpu_mode:
                    self_attention_k_output = self_attention_k_output.cpu()
                    self_attention_v_output = self_attention_v_output.cpu()

                present_key_values += ((self_attention_k_output, self_attention_v_output),) 

            self._switch_input_output_binding()
        return CausalLMOutputWithPast(logits=logits.to(self.return_device), past_key_values = present_key_values)

class GPT2TRT(TRTInferenceCommand):
    def __init__(self):
        super().__init__(
            GPT2ModelTRTConfig, "Runs polygraphy results for GPT2 model.", GPT2HuggingFace
        )
        self.gpt2_trt = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_trt_engine: bool = False,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        if self.gpt2_trt is not None:
            self.gpt2_trt.release()

        if not keep_trt_engine:
            self.gpt2_trt_engine.cleanup()

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
        if max_length is None:
            max_length = GPT2ModelTRTConfig.MAX_OUTPUT_LENGTH[self.metadata.variant]

        if min_length is None:
            min_length = GPT2ModelTRTConfig.MIN_OUTPUT_LENGTH[self.metadata.variant]

        output = self.gpt2_trt.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            use_cache=use_cache,
            early_stopping=early_stopping
        )

        self.gpt2_trt.reset()
        return output

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_mode: bool = False,
        benchmarking_args: GPT2TRTBenchmarkingArgs = None,
    ) -> Union[NetworkResult, BenchmarkingResult]:

        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)

        # GPT2 has no proper token set. Use custom token. Only "generate()" will auto
        # replace with EOS token when using generating mode
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        hf_config = self.gpt2_trt.config

        # Prepare the input tokens and find out output sequence length.
        if not benchmarking_mode:
            output_seq_len = GPT2ModelTRTConfig.MAX_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, return_tensors="pt").input_ids
        else:
            input_seq_len = benchmarking_args.input_seq_len
            output_seq_len = benchmarking_args.output_seq_len
            input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, input_seq_len))

        # get single decoder iteration inference timing profile
        _, decoder_e2e_time = gpt2_inference(
            self.gpt2_trt,
            expand_inputs_for_beam_search(input_ids, num_beams) if num_beams > 1 else input_ids,
            timing_profile,
            use_cache = metadata.other.kv_cache,
        )
        
        # get complete decoder inference result and its timing profile
        sample_output, full_e2e_runtime = full_inference(
            self.gpt2_trt,
            input_ids,
            tokenizer,
            timing_profile,
            max_length=output_seq_len,
            min_length=GPT2ModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
            batch_size=batch_size,
            use_cache=metadata.other.kv_cache,
            num_beams=num_beams,
        )

        # Prepare runtime results.
        runtime = [
            NetworkRuntime(
                name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                runtime=decoder_e2e_time,
            ),
            NetworkRuntime(
                name=GPT2ModelTRTConfig.NETWORK_FULL_NAME,
                runtime=full_e2e_runtime,
            ),
        ]
        models = NetworkModels(
            torch=None,
            onnx=list(onnx_fpaths.values()),
            trt=[
                NetworkModel(
                    name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    fpath=self.gpt2_trt_engine.fpath,
                ),
            ],
        )

        # Skip result checking in benchmarking mode since the input data is random.
        if benchmarking_mode:
            return BenchmarkingResult(median_runtime=runtime, models=models)

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.decode(
            sample_output[-1, :], skip_special_tokens=True
        )

        if isinstance(semantic_outputs, list):
            semantic_outputs = " ".join(semantic_outputs).strip()

        return NetworkResult(
            input=inference_input,
            output_tensor=sample_output,
            semantic_output=semantic_outputs,
            median_runtime=runtime,
            models=models,
        )

    def execute_calculate_perplexity(
        self,
        metadata: NetworkMetadata,
        reference: str,
        batch_size: int, 
    ):
        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)

        # GPT2 has no proper token set. Use custom token. Only "generate()" will auto
        # replace with EOS token when using generating mode
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        reference = reference.replace("\\n", "\n")
        ppl_input_ids = tokenizer([reference] * batch_size, padding=False, return_tensors="pt").input_ids

        perplexity = calculate_perplexity(
            self.gpt2_trt, ppl_input_ids, GPT2ModelTRTConfig.MAX_LENGTH[metadata.variant]
        )
        return perplexity

    def _setup_engines(
        self,
        metadata: NetworkMetadata,
        hash_onnx_fpath: Dict[str, NetworkModel],
        batch_size: int,
        num_beams: int,
        disable_preview_dynamic_shapes: bool,
        benchmarking_args: GPT2TRTBenchmarkingArgs = None,
        seq_tag: bool = False, # whether the benchmark engine tag format should be seq or max
    ) -> None:

        hf_config = AutoConfig.from_pretrained(
            metadata.variant,
            use_cache=metadata.other.kv_cache
        )

        # Output networks shall not exceed number of network segments explicitly defined by configuration file.
        assert len(hash_onnx_fpath) == len(
            GPT2ModelTRTConfig.NETWORK_SEGMENTS
        ), "There should only be {} exported ONNX segments in GPT2 model.".format(
            len(GPT2ModelTRTConfig.NETWORK_SEGMENTS)
        )

        decoder_onnx_fpath = hash_onnx_fpath[
            GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
        ].fpath

        # Generate optimization profiles.
        # non-benchmarking mode: opt profile length is by default half of the max profile
        # benchmarking mode: user can specify opt and max profile by flags. If no additional benchmarking flags are provided, it will just use the non-benchmarking mode defaults
        # Note that this should be set to GPT2's MAX_LENGTH for text generation.
        max_sequence_length = GPT2ModelTRTConfig.MAX_LENGTH[metadata.variant]
        max_output_length = GPT2ModelTRTConfig.MAX_LENGTH[metadata.variant]
        opt_input_seq_len = max_sequence_length // 2
        opt_output_seq_len = max_output_length // 2

        # benchmarking flags
        if benchmarking_args is not None:
            max_sequence_length = benchmarking_args.input_profile_max_len
            max_output_length = benchmarking_args.output_profile_max_len
            opt_input_seq_len = benchmarking_args.input_seq_len
            opt_output_seq_len = benchmarking_args.output_seq_len
        
        if not hf_config.use_cache:
            # If not using kv cache, only input_ids is passed
            decoder_profiles = [Profile().add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, opt_output_seq_len),
                max=(batch_size * num_beams, max_output_length),
            )]
        else:
            num_heads = hf_config.n_head
            embedding_size_per_head = hf_config.n_embd // num_heads
            num_layers = hf_config.n_layer

            # context phase uses the provided input_ids to generate hidden states and self attention kv cache
            # It is only used in the 1st decoder run.
            dec_profiles_context = Profile().add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, opt_output_seq_len),
                max=(batch_size * num_beams, max_output_length),
            )
            self_attention_profile_context = {
                "min": (batch_size * num_beams, num_heads, 0, embedding_size_per_head),
                "opt": (batch_size * num_beams, num_heads, 0, embedding_size_per_head),
                "max": (batch_size * num_beams, num_heads, 0, embedding_size_per_head),
            }

            # generation phase uses previous self attention kv cache with the last input_ids token to generate the next hidden states and self attention kv cache
            # This optimization profile is used after the 1st decoder run.
            dec_profiles_generation = Profile().add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, 1),
                max=(batch_size * num_beams, 1),
            )
            
            self_attention_profile_generation = {
                "min": (batch_size * num_beams, num_heads, 1, embedding_size_per_head),
                "opt": (batch_size * num_beams, num_heads, opt_output_seq_len - 1, embedding_size_per_head),
                "max": (batch_size * num_beams, num_heads, max_output_length - 1, embedding_size_per_head),
            }

            for i in range(num_layers):
                dec_profiles_context = dec_profiles_context.add(
                    f"past_key_values.{i}.decoder.key",
                    **self_attention_profile_context
                ).add(
                    f"past_key_values.{i}.decoder.value",
                    **self_attention_profile_context
                )

                dec_profiles_generation = dec_profiles_generation.add(
                    f"past_key_values.{i}.decoder.key",
                    **self_attention_profile_generation
                ).add(
                    f"past_key_values.{i}.decoder.value",
                    **self_attention_profile_generation
                )
            
            # TensorRT accepts multiple optimization engines for the same model.
            # Profile 1 is only used in the first decoder iterations.
            decoder_profiles = [dec_profiles_generation, dec_profiles_context]
        
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

        preview_features = [PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        if disable_preview_dynamic_shapes:
            engine_tag += "-noPreviewFasterDynamicShapes"
        else:
            preview_features.append(PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        
        self.gpt2_trt_engine = GPT2ONNXFile(
            decoder_onnx_fpath, metadata
        ).as_trt_engine(
            os.path.splitext(decoder_onnx_fpath)[0] + "-{}.engine".format(engine_tag),
            profiles=decoder_profiles,
            preview_features=preview_features
        )
        self.gpt2_trt = GPT2TRTDecoder(
            self.gpt2_trt_engine, metadata, hf_config, batch_size=batch_size, num_beams=num_beams, benchmarking_args = benchmarking_args
        )

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
        disable_preview_dynamic_shapes: bool = False,
        perplexity_reference: List[str] = None,
    ) -> Union[List[NetworkResult], BenchmarkingResult]:

        workspace = self._setup_workspace(metadata, working_directory)

        # no fpath provided for onnx files, download them
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
                self._setup_engines(metadata, hash_onnx_fpath, batch_size, args.num_beams, disable_preview_dynamic_shapes)
                for ninput in network_input:
                    inference_results.append(
                        self.execute_inference(
                            metadata, hash_onnx_fpath, ninput, timing_profile, batch_size, args.num_beams
                        )
                    )
                    # reset the decoder
                    self.gpt2_trt.reset()

                if perplexity_reference is not None:
                    assert len(network_input) == len(perplexity_reference), "Inputs must pair up"
                    if metadata.other.kv_cache or (args.num_beams > 1):
                        G_LOGGER.warning("Skipping perplexity calculation for TRT with KV cache or beam search because it is not supported yet.")
                    else:
                        for r in perplexity_reference:
                            ppl_results.append(
                                self.execute_calculate_perplexity(metadata, r, batch_size)
                            )
            else:
                hf_config = AutoConfig.from_pretrained(metadata.variant, use_cache = metadata.other.kv_cache)
                # Check that input_seq_len and output_seq_len is valid and within required range
                max_input_seq_len = hf_config.n_positions
                max_output_seq_len = hf_config.n_positions

                seq_tag = args.input_profile_max_len is None and args.output_profile_max_len is None
                # User must provide either a pair of profile_max_len or a profile of seq_len for input/output
                if args.input_profile_max_len is None or args.output_profile_max_len is None:
                    if args.input_seq_len is None or args.output_seq_len is None:
                        assert False, "Please provide at least one pair of inputs: [input/output]_seq_len or [input/output]_profile_max_len"

                input_profile_max_len = setup_benchmark_arg(args.input_profile_max_len, "input_profile_max_len", max_input_seq_len)
                output_profile_max_len = setup_benchmark_arg(args.output_profile_max_len, "output_profile_max_len", max_output_seq_len)
                input_seq_len = setup_benchmark_arg(args.input_seq_len, "input_seq_len", input_profile_max_len // 2)
                output_seq_len = setup_benchmark_arg(args.output_seq_len, "output_seq_len", output_profile_max_len // 2)

                benchmarking_args = GPT2TRTBenchmarkingArgs(input_seq_len, output_seq_len, input_profile_max_len, output_profile_max_len)

                # Assert to ensure the validity of benchmarking arguments
                assert benchmarking_args.input_seq_len <= benchmarking_args.input_profile_max_len, "input_seq_len should <= input_profile_max_len = {} for benchmarking mode".format(benchmarking_args.input_profile_max_len)
                assert benchmarking_args.output_seq_len <= benchmarking_args.output_profile_max_len, "output_seq_len should <= output_profile_max_len = {} for benchmarking mode".format(benchmarking_args.output_profile_max_len)
                assert benchmarking_args.input_profile_max_len <= max_input_seq_len, "Model config restrict input_profile_max_len <= {} for benchmark mode".format(max_input_seq_len)
                assert benchmarking_args.output_profile_max_len <= max_output_seq_len, "Model config restrict output_profile_max_len <= {} for benchmark mode".format(max_output_seq_len)
                # GPT2 model requires output_seq_len > input_seq_len since it is a text generation model.
                assert benchmarking_args.input_seq_len <= benchmarking_args.output_seq_len, "GPT2 model text generation requires output_seq_len > input_seq_len."
                assert benchmarking_args.input_profile_max_len <= benchmarking_args.output_profile_max_len, "GPT2 model text generation requires output_profile_max_len > input_profile_max_len"
                self._setup_engines(metadata, hash_onnx_fpath, batch_size, args.num_beams, disable_preview_dynamic_shapes, benchmarking_args, seq_tag)
                inference_results = self.execute_inference(
                    metadata, hash_onnx_fpath, None, timing_profile, batch_size, args.num_beams, True, benchmarking_args
                )

        finally:
            self.cleanup(workspace, keep_trt_engine, keep_onnx_model, keep_torch_model)

        return inference_results, ppl_results

    def add_args(self, parser) -> None:
        super().add_args(parser)

        # use the same args as frameworks.py
        self.frameworks_cmd.add_args(parser)
        polygraphy_group = parser.add_argument_group("polygraphy")
        polygraphy_group.add_argument(
            "--onnx-fpath",
            default=None,
            help="Path to GPT2 ONNX model. If None is supplied, scripts will generate them from HuggingFace.",
        )
        polygraphy_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )
        polygraphy_group.add_argument(
            "--save-trt-engine",
            action="store_true",
            help="Saves TensorRT runtime engine in working directory.",
        )

    def args_to_network_models(self, args) -> List[NetworkModel]:
        gpt2_fpath_check = args.onnx_fpath is None

        network_models = None
        if gpt2_fpath_check:
            network_models = tuple()
        else:
            onnx_decoder = NetworkModel(
                name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_fpath,
            )
            network_models = (onnx_decoder)

        return network_models

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        frameworks_parsed_metadata = self.frameworks_cmd.args_to_network_metadata(args)

        return NetworkMetadata(
            variant=frameworks_parsed_metadata.variant,
            precision=Precision(fp16=args.fp16),
            other=frameworks_parsed_metadata.other,
        )


RUN_CMD = GPT2TRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
