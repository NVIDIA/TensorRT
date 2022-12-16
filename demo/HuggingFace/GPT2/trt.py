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
from transformers import GPT2Tokenizer, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
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
        # In benchmarking mode, if input_profile_max is provided, should use that as max_sequence_length
        if benchmarking_args is not None:
            if benchmarking_args.input_profile_max_len is not None:
                self.max_sequence_length = benchmarking_args.input_profile_max_len
            else:
                self.max_sequence_length = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[network_metadata.variant]
        # In non-benchmarking mode, we are provided a text generation task. We need to use the max_length as max sequence length
        else:
            self.max_sequence_length = GPT2ModelTRTConfig.MAX_LENGTH[network_metadata.variant]
        
        # Similarly, the max_output_length should be the user-provided output_profile_max_len if provided
        if benchmarking_args is not None and benchmarking_args.output_profile_max_len is not None:
            self.max_output_length = benchmarking_args.output_profile_max_len
        else: 
            self.max_output_length = self.max_sequence_length

        # We only have one profile to select so we can just grab the profile at the start of the class
        self.profile_idx = self.get_optimization_profile(batch_size=self.batch_size * num_beams, sequence_length=1)
        input_profile_length = self.max_output_length if (not self.config.use_cache) else 1
        self.input_shapes = {
            "input_ids": (self.batch_size * num_beams, input_profile_length)
        }
        self.input_types = {
            "input_ids": torch.int32
        }

        self.output_shapes = {
            "logits": (self.batch_size * num_beams, self.max_output_length, GPT2ModelTRTConfig.VOCAB_SIZE[network_metadata.variant])
        }
        self.output_types = {
            "logits": torch.float32
        }
        self.main_input_name = "input_ids"

        self.num_heads = GPT2ModelTRTConfig.NUMBER_OF_HEADS[network_metadata.variant]
        self.embedding_size_per_head = GPT2ModelTRTConfig.EMBEDDING_SIZE[network_metadata.variant] // self.num_heads

        if self.config.use_cache:
            self.num_decoder_layers = GPT2ModelTRTConfig.NUMBER_OF_LAYERS[network_metadata.variant]
            # Set kv cache shape and type
            for i in range(self.num_decoder_layers):
                kv_type_dict = {"decoder": torch.float32}
                set_kv_data(self.input_types, "past", i, kv_type_dict)
                set_kv_data(self.output_types,"present", i, kv_type_dict)

                self_attention_kv_shape = (self.batch_size * num_beams, self.num_heads, self.max_output_length - 1, self.embedding_size_per_head)
                kv_shape_dict = {"decoder": self_attention_kv_shape}

                set_kv_data(self.input_shapes, "past", i, kv_shape_dict)
                set_kv_data(self.output_shapes, "present", i, kv_shape_dict)

            self.kv_cache_binding_offset = 1 # 0: input_ids, kv cache input indices start from 1
        
        self.bindings = self._allocate_memory(self.input_shapes, self.input_types, self.output_shapes, self.output_types)
        self.use_non_kv_engine = self.config.use_cache
        # This flag is true if and only if we are at the 1st step of decoding with kv cache. We need the special engine that
        # only has input_ids as input and output both logits and kv-cache. After that, we can use the kvcache engine for trt
        self.return_device = "cuda"
        self.device = "cuda"
        self.variant = network_metadata.variant
    
    def reset(self):
        '''
        Resets the input specific fields after finishing a task.
        '''
        self.use_non_kv_engine = self.config.use_cache
    
    def _set_non_kv_engine_for_kv_mode(self, trt_engine_file_non_kv: TRTEngineFile):
        # same steps in tensorrt_utils.py: TRTNativeRunner
        with open(trt_engine_file_non_kv.fpath, "rb") as f:
            self.trt_engine_non_kv = self.trt_runtime.deserialize_cuda_engine(f.read())
            self.trt_context_non_kv = self.trt_engine_non_kv.create_execution_context()
        
        # The only input for GPT2 is the inpud_ids. 
        input_name = "input_ids"
        self.input_types_non_kv = {input_name: self.input_types[input_name]}
        # non_kv engine profile is different from kv since it needs to accept input_ids.
        self.input_shapes_non_kv = {input_name: (self.input_shapes[input_name][0], self.max_sequence_length)}

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
    
    def prepare_inputs_for_generation(self, input_ids, past = None, use_cache = None, **kwargs):
        # TODO: add position_ids, token_type_ids support
        if past:
            input_ids = input_ids[:, -1:]
            self.use_non_kv_engine = False
        
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
        
    def forward(self, input_ids, *args, **kwargs):
        bs = input_ids.shape[0]
        input_length = input_ids.shape[1]
        vocab_size = GPT2ModelTRTConfig.VOCAB_SIZE[self.network_metadata.variant]
        max_length = self.max_sequence_length
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
            bindings[0] = self.inputs["input_ids"].data_ptr()
        else:
            inputs["input_ids"][:bs * input_length] = input_ids.flatten()

        # Set the binding shape of input_ids, which should be (bs, input_length).
        trt_context.set_binding_shape(0, input_ids.shape)
       
        if self.config.use_cache: # or use_cache
            if non_kv_flag:
                # use non-kv engine, no additional inputs
                past_decoder_length = 0
            else:
                # use kv engine
                # past_key_values set by prepare_inputs_for_generation() during HF e2e pipeline; if only test decoder, need to set this field
                past_key_values = kwargs.get("past_key_values") 
                past_decoder_length = past_key_values[0][0].size(2)
                num_heads = self.num_heads
                embedding_size_per_head = self.embedding_size_per_head
                # Set the binding shape of self-attention KV caches, which should be (bs, num_heads, past_decoder_length, embedding_size_per_head).
                self_attention_kv_shape = (bs, num_heads, past_decoder_length, embedding_size_per_head)
                self_attention_kv_flatten_length = bs * num_heads * past_decoder_length * embedding_size_per_head
                
                for i in range(self.num_decoder_layers):
                    if past_key_values is not None:
                        if past_key_values[0][0].device == torch.device("cpu"):
                            inputs[f"past_key_values.{i}.decoder.key"] = past_key_values[i][0].flatten().contiguous().cuda()
                            bindings[self.kv_cache_binding_offset+2*i] = inputs[f"past_key_values.{i}.decoder.key"].data_ptr()

                            inputs[f"past_key_values.{i}.decoder.value"] = past_key_values[i][1].flatten().contiguous().cuda()
                            bindings[self.kv_cache_binding_offset+2*i+1] = inputs[f"past_key_values.{i}.decoder.value"].data_ptr()

                        else:
                            inputs[f"past_key_values.{i}.decoder.key"][:self_attention_kv_flatten_length] = past_key_values[i][0].flatten()

                            inputs[f"past_key_values.{i}.decoder.value"][:self_attention_kv_flatten_length] = past_key_values[i][1].flatten()

                    trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i, self_attention_kv_shape)
                    trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + 1, self_attention_kv_shape)

        # Launch TRT inference.
        # TODO: Could we use execute_v2_async() instead of execute_v2()? Current profiling shows that there is a
        # synchronization inside TRT's inference body, so this change may not be needed.
        trt_context.execute_v2(bindings=bindings)

        # We allocate the buffers using max_length, but we only need to first portion of it, so get only the first
        # portion of the output buffer and return that.
        # TODO: Could we construct a Torch tensor using given data_ptr() to avoid this D2D copy?
        logits_output = outputs["logits"]
        if is_cpu_mode:
            logits_output = logits_output.cpu()
        
        folded = logits_output[:bs * input_length * vocab_size].view(bs, input_length, vocab_size)
        
        present_key_values = None
        if self.config.use_cache:
            # 1st decoding step and steps after handle the outputs in the same way
            present_key_values = ()
            curr_decoder_length = past_decoder_length + input_length
            num_heads = self.num_heads
            embedding_size_per_head = self.embedding_size_per_head
            self_attention_kv_shape = (bs, num_heads, curr_decoder_length, embedding_size_per_head)
            self_attention_kv_flatten_length = bs * num_heads * curr_decoder_length * embedding_size_per_head

            for i in range(self.num_decoder_layers):
                self_attn_k_output = outputs[f"present_key_values.{i}.decoder.key"]
                self_attn_v_output = outputs[f"present_key_values.{i}.decoder.value"]
                if is_cpu_mode:
                    self_attn_k_output = self_attn_k_output.cpu()
                    self_attn_v_output = self_attn_v_output.cpu()

                self_attn_k = self_attn_k_output[:self_attention_kv_flatten_length].view(*self_attention_kv_shape)
                self_attn_v = self_attn_v_output[:self_attention_kv_flatten_length].view(*self_attention_kv_shape)

                present_key_values += ((self_attn_k, self_attn_v), ) # make multi-dim tuple
        
        return CausalLMOutputWithCrossAttentions(logits=folded.to(self.return_device), past_key_values = present_key_values)

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
            # TODO: Avoid using workspace.metadata to handle non_kv removals.
            if workspace.metadata.other.kv_cache:
                self.gpt2_trt_engine_non_kv.cleanup()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

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

        # Prepare the input tokens and find out output sequence length.
        if not benchmarking_mode:
            output_seq_len = GPT2ModelTRTConfig.MAX_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, return_tensors="pt").input_ids
        else:
            input_seq_len = benchmarking_args.input_seq_len
            output_seq_len = benchmarking_args.output_seq_len
            input_ids = torch.randint(0, GPT2ModelTRTConfig.VOCAB_SIZE[metadata.variant], (batch_size, input_seq_len))

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
    ):
        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)

        # GPT2 has no proper token set. Use custom token. Only "generate()" will auto
        # replace with EOS token when using generating mode
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        reference = reference.replace("\\n", "\n")
        ppl_input_ids = tokenizer([reference], padding=False, return_tensors="pt").input_ids

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
        preview_dynamic_shapes: bool,
        benchmarking_args: GPT2TRTBenchmarkingArgs = None,
        seq_tag: bool = False, # whether the benchmark engine tag format should be seq or max
    ) -> None:

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

       # Set up the non kv engine, used for non-kv mode and kv mode generation phase (1st decoder run uses the non-kv profile to generate kv cache)
        dec_profiles_non_kv = Profile()
        
        # for beam search, decoder engine's inputs are expanded `num_beams` times
        # optimization profiles should be changed accordingly, but onnx models can be shared across greedy/beam because the first dim (batch size) is already a dynamic value, so no change needed in export.py
        dec_profiles_non_kv = dec_profiles_non_kv.add(
            "input_ids",
            min=(batch_size * num_beams, 1),
            opt=(batch_size * num_beams, opt_output_seq_len),
            max=(batch_size * num_beams, max_output_length),
        )

        decoder_profiles_non_kv = [dec_profiles_non_kv]
        
        dec_profiles_kv = Profile()
        if metadata.other.kv_cache:
            # Note that the kv profile only accept length 1
            dec_profiles_kv = dec_profiles_kv.add(
                "input_ids",
                min=(batch_size * num_beams, 1),
                opt=(batch_size * num_beams, 1),
                max=(batch_size * num_beams, 1),
            )

            num_heads = GPT2ModelTRTConfig.NUMBER_OF_HEADS[metadata.variant]
            embedding_size_per_head = GPT2ModelTRTConfig.EMBEDDING_SIZE[metadata.variant] // num_heads
            num_decoder_layers = GPT2ModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant]
            self_attention_profile = {
                "min": (batch_size * num_beams, num_heads, 0, embedding_size_per_head),
                "opt": (batch_size * num_beams, num_heads, opt_output_seq_len - 1, embedding_size_per_head),
                "max": (batch_size * num_beams, num_heads, max_output_length - 1, embedding_size_per_head),
            }
            
            # TODO: move this logic (and some other similar place) into utils.
            for i in range(num_decoder_layers):

                dec_profiles_kv = dec_profiles_kv.add(
                    f"past_key_values.{i}.decoder.key",
                    **self_attention_profile
                )
                dec_profiles_kv = dec_profiles_kv.add(
                    f"past_key_values.{i}.decoder.value",
                    **self_attention_profile
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
        
        if not metadata.other.kv_cache:
            self.gpt2_trt_engine = GPT2ONNXFile(
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
            self.gpt2_trt_engine = GPT2ONNXFile(
                decoder_onnx_kv_fpath, metadata
            ).as_trt_engine(
                os.path.splitext(decoder_onnx_kv_fpath)[0] + "-{}.engine".format(engine_tag),
                profiles=decoder_profiles,
                preview_features=preview_features
            )
            # dual-engine approach: still need to setup non-kv engine in kv mode
            # note: workspace cleanup is not handled for these extra non-kv files
            self.gpt2_trt_engine_non_kv = GPT2ONNXFile(
                decoder_onnx_non_kv_fpath, metadata
            ).as_trt_engine(
                os.path.splitext(decoder_onnx_non_kv_fpath)[0] + "-{}.engine".format(engine_tag),
                profiles=decoder_profiles_non_kv,
                preview_features=preview_features
            )


        # Create GPT2TRTDecoder instances.
        tfm_config = GPT2Config(
            use_cache=metadata.other.kv_cache,
            num_layers=GPT2ModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant],
        )

        self.gpt2_trt = GPT2TRTDecoder(
            self.gpt2_trt_engine, metadata, tfm_config, batch_size=batch_size, num_beams=num_beams, benchmarking_args = benchmarking_args
        )

        if metadata.other.kv_cache:
            # switch between GPT2TRT is impossible (becase HF decoding step is bound to one decoder). Therefore, we need to add the non-kv engines inside the same decoder --> decoder contains two TRT engines
            self.gpt2_trt._set_non_kv_engine_for_kv_mode(self.gpt2_trt_engine_non_kv)

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
    ) -> Union[List[NetworkResult], BenchmarkingResult]:

        workspace = NNFolderWorkspace(
            self.frameworks_cmd.config.network_name, metadata, working_directory
        )

        # no fpath provided for onnx files, download them
        if len(onnx_fpaths) == 0:
            onnx_fpaths = self.frameworks_cmd.generate_and_download_framework(
                metadata, workspace
            ).onnx
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
                    # reset the decoder 
                    self.gpt2_trt.reset()

                if perplexity_reference is not None:
                    assert len(network_input) == len(perplexity_reference), "Inputs must pair up"
                    if metadata.other.kv_cache or (args.num_beams > 1):
                        G_LOGGER.warning("Skipping perplexity calculation for TRT with KV cache or beam search because it is not supported yet.")
                    else:
                        for r in perplexity_reference:
                            ppl_results.append(
                                self.execute_calculate_perplexity(metadata, r)
                            )
            else:
                # Check that input_seq_len and output_seq_len is valid and within required range
                max_input_seq_len = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant] 
                max_output_seq_len = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]

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
                self._setup_engines(metadata, hash_onnx_fpath, batch_size, args.num_beams, preview_dynamic_shapes, benchmarking_args, seq_tag)
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
