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
import time

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
from transformers import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

# tensorrt
from tensorrt import PreviewFeature

# TRT-HuggingFace
from NNDF.interface import TRTInferenceCommand
from NNDF.networks import (
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
    Precision,
)

from NNDF.general_utils import confirm_folder_delete
from NNDF.tensorrt_utils import TRTNativeRunner, setup_benchmark_arg, CUASSERT
from NNDF.models import TRTEngineFile
from NNDF.logger import G_LOGGER

from Seq2Seq.Seq2SeqModelConfig import Seq2SeqModelTRTConfig
from Seq2Seq.export import Seq2SeqModelClass
from cuda import cudart

class Seq2SeqTRTEncoder(TRTNativeRunner):
    """TRT implemented network interface that can be used to measure inference time."""

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        config: Seq2SeqModelTRTConfig,
        nvtx_verbose: bool,
    ):
        super().__init__(trt_engine_file, network_metadata, config, nvtx_verbose)

        self.data_type = torch.float16 if network_metadata.precision.fp16 else torch.float32
        self.main_input_name = "input_ids"
        # Older TRT version does not have int64 support, and therefore requires int32 bindings
        self.int_type = torch.int32 if self.trt_engine.get_tensor_dtype(self.main_input_name) == trt.int32 else torch.int64

        self.device = torch.device("cuda")

        self.encoder_hidden_states = torch.zeros(
            config.batch_size*config.max_input_length*config.hidden_size,
            dtype=self.data_type,
            device=self.device
        )
        self.trt_context.set_tensor_address("encoder_hidden_states", self.encoder_hidden_states.data_ptr())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):

        input_length = input_ids.shape[1]

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu"))

        input_ids = input_ids.to(self.int_type).contiguous()

        if is_cpu_mode:
            input_ids = input_ids.cuda()

        self.trt_context.set_tensor_address(self.main_input_name, input_ids.data_ptr())
        self.trt_context.set_input_shape(self.main_input_name, input_ids.shape)

        if self.config.use_mask:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            if self.trt_engine.get_tensor_dtype("attention_mask") == trt.int32:
                attention_mask = attention_mask.int().contiguous()
            else:
                attention_mask = attention_mask.long().contiguous()

            if is_cpu_mode:
                attention_mask = attention_mask.cuda()
            self.trt_context.set_tensor_address("attention_mask", attention_mask.data_ptr())
            self.trt_context.set_input_shape("attention_mask", attention_mask.shape)

        assert self.trt_context.all_shape_inputs_specified

        # Launch TRT inference.
        self.trt_context.execute_async_v3(self.stream)
        CUASSERT(cudart.cudaStreamSynchronize(self.stream))

        last_hidden_state = self.encoder_hidden_states[:self.config.batch_size * input_length * self.config.hidden_size].view(self.config.batch_size, input_length, self.config.hidden_size)

        if is_cpu_mode:
            last_hidden_state = last_hidden_state.cpu()

        return BaseModelOutput(last_hidden_state = last_hidden_state)

class Seq2SeqTRTDecoder(TRTNativeRunner, GenerationMixin):

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        config: Seq2SeqModelTRTConfig,
        nvtx_verbose: bool,
        use_cuda_graph: bool = False,
    ):
        super().__init__(trt_engine_file, network_metadata, config, nvtx_verbose)
        self.main_input_name = "input_ids"

        # Older TRT version does not have int64 support, and therefore requires int32 bindings
        self.int_type = torch.int32 if self.trt_engine.get_tensor_dtype(self.main_input_name) == trt.int32 else torch.int64
        self.generation_config = config.generation_config
        self.device = torch.device("cuda")
        self.encoder_hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.embedding_size_per_head = config.d_kv
        self.num_decoder_layers = config.num_decoder_layers
        self.expand_size = config.expand_size
        self.profile_idx = 0
        # For accuracy mode only
        self._return_full_logits = False
        self.target_ids = None

        # Setting next input shape when execution
        # Use dictionary to record which inputs have changed.
        self.input_shape_change_record = dict()

        # CUDA Graph
        self.stream_for_cuda_graph = CUASSERT(cudart.cudaStreamCreate())[0]
        self.use_cuda_graph = use_cuda_graph
        # two cuda graph ping-pong instance
        self.cuda_graph_instances = [None, None]
        self.cur_cuda_graph_instance = 0

        # Construct buffer for logits outputs
        self.logits = torch.zeros(
            config.expand_size*config.max_decoder_length*config.vocab_size,
            dtype=config.precision,
            device=self.device,
        )
        self.trt_context.set_tensor_address("logits", self.logits.data_ptr())

        self.context_mode = (not config.is_encoder_decoder) and (config.use_cache)

        if config.consume_image_embeds:
            self.image_embeds = torch.zeros(
                config.expand_size*config.num_positions*config.hidden_size,
                dtype=config.precision,
                device=self.device
            )
            self.trt_context.set_tensor_address("image_embeds", self.image_embeds.data_ptr())

        if self.use_cuda_graph:
            self.cuda_graph_input_data = torch.zeros(
                (config.expand_size,1), dtype=self.int_type, device=self.device
            )

        if config.is_encoder_decoder:
            self.encoder_hidden_states = torch.zeros(
                config.expand_size*config.max_input_length*config.hidden_size,
                dtype=config.precision,
                device=self.device
            )
            self.trt_context.set_tensor_address("encoder_hidden_states", self.encoder_hidden_states.data_ptr())
            self.encoder_length = 0

        if config.use_cache:
            if not self.config.is_encoder_decoder:
                self.set_context_mode_trt_context()
                self.context_trt_context.set_tensor_address("logits", self.logits.data_ptr())
            if config.consume_image_embeds:
                self.context_trt_context.set_tensor_address("image_embeds", self.image_embeds.data_ptr())

            self.self_attn_cache = {}
            self.past_cross_attn_cache = {}
            self_attn_cache_size = config.expand_size * config.num_heads * config.max_output_length * config.d_kv
            cross_attn_cache_size = config.expand_size * config.num_heads * config.max_input_length * config.d_kv

            # Set self attention kv cache shape and type
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:
                    # Allocate self attention buffer. The buffer is used both as inputs and outputs
                    self_attn_name = f"key_values.{i}.self.{code}"
                    self_attn_buffer = [torch.zeros(
                        self_attn_cache_size,
                        dtype=config.precision,
                        device=self.device,
                    ), torch.zeros(
                        self_attn_cache_size,
                        dtype=config.precision,
                        device=self.device,
                    )]
                    # Set input = output for cache. Might break for GPT/BART. Let's give it a try first.
                    self.self_attn_cache[self_attn_name] = self_attn_buffer
                    past_kv_cache_tensor_name = "past_" + self_attn_name
                    present_kv_cache_tensor_name = "present_" + self_attn_name
                    past_kv_cache_address = self_attn_buffer[0].data_ptr()
                    present_kv_cache_address = self_attn_buffer[1].data_ptr()
                    self.trt_context.set_tensor_address(past_kv_cache_tensor_name, past_kv_cache_address)
                    self.trt_context.set_tensor_address(present_kv_cache_tensor_name, present_kv_cache_address)

                    if config.is_encoder_decoder:
                        # Allocate cross attention buffer
                        cross_attn_past_name = f"past_key_values.{i}.cross.{code}"
                        cross_attn_buffer = torch.zeros(
                            cross_attn_cache_size,
                            dtype=config.precision,
                            device=self.device,
                        )
                        self.past_cross_attn_cache[cross_attn_past_name] = cross_attn_buffer
                        self.trt_context.set_tensor_address(cross_attn_past_name, cross_attn_buffer.data_ptr())
                    else:
                        # Context phase, execution V3 can't set nullptr, set same buffer as output
                        self.context_trt_context.set_tensor_address(past_kv_cache_tensor_name, past_kv_cache_address)
                        self.context_trt_context.set_tensor_address(present_kv_cache_tensor_name, past_kv_cache_address)
                    self.cache_id = 0

            self.past_decoder_length = 0

    def accuracy_mode(self, target_ids):
        self._return_full_logits = True
        self.full_logits = None
        self.target_ids = target_ids

    def disable_accuracy_mode(self):
        self._return_full_logits = False
        self.target_ids = None
        self.full_logits = None

    def can_generate(self):
        return True

    def set_image_embeds(self, image_embeds: torch.Tensor):
        """Used to cache encoder hidden state runs across same encoder sessions"""
        self.image_embeds[:image_embeds.numel()] = image_embeds.flatten().contiguous()
        self.trt_context.set_input_shape("image_embeds", image_embeds.shape)

    def set_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor):
        """Used to cache encoder hidden state runs across same encoder sessions"""
        self.encoder_hidden_states[:encoder_hidden_states.numel()] = encoder_hidden_states.flatten().contiguous()
        self.trt_context.set_input_shape("encoder_hidden_states", encoder_hidden_states.shape)

    def set_cross_attn_cache_generator_engine(self, cross_attn_cache_generator):

        # Sharing memory between 2 TRT context may have memory corruption. Therefore create separate memory for cross attn generator
        self.cross_attn_cache_generator = cross_attn_cache_generator
        with open(self.cross_attn_cache_generator.fpath, "rb") as f:
            trt_runtime = trt.Runtime(self.trt_logger)
            self.cross_attn_cache_generator_engine = trt_runtime.deserialize_cuda_engine(f.read())
            self.cross_attn_cache_generator_context = self.cross_attn_cache_generator_engine.create_execution_context()
            self.cross_attn_cache_generator_context.nvtx_verbosity = self.trt_context.nvtx_verbosity

        self.cross_attn_cache_generator_context.set_tensor_address("encoder_hidden_states", self.encoder_hidden_states.data_ptr())
        # Cross attention cache as outputs
        for i in range(self.num_decoder_layers):
            past_key_name = f"past_key_values.{i}.cross.key"
            past_value_name = f"past_key_values.{i}.cross.value"
            present_key_name = f"present_key_values.{i}.cross.key"
            present_value_name = f"present_key_values.{i}.cross.value"
            self.cross_attn_cache_generator_context.set_tensor_address(present_key_name, self.past_cross_attn_cache[past_key_name].data_ptr())
            self.cross_attn_cache_generator_context.set_tensor_address(present_value_name, self.past_cross_attn_cache[past_value_name].data_ptr())

    def set_cross_attn_cache(self, encoder_hidden_states: torch.Tensor):
        """
        Used to cache encoder-decoder cross attention kv caches across same encoder sessions.

        Unlike self-attention cache, cross attention is constant during the decoding process, so we only need to set its bindings once at the first decoding step, and skip in all later steps (by self.persist_cross_attention_kv_cache flag)
        """

        self.cross_attn_cache_generator_context.set_input_shape("encoder_hidden_states", encoder_hidden_states.shape)

        assert self.cross_attn_cache_generator_context.all_shape_inputs_specified
        self.cross_attn_cache_generator_context.execute_async_v3(self.stream)
        CUASSERT(cudart.cudaStreamSynchronize(self.stream))

        encoder_length = encoder_hidden_states.shape[1]
        cross_attn_cache_shape = (self.expand_size, self.num_heads, encoder_length, self.embedding_size_per_head)

        for i in range(self.num_decoder_layers):
            past_key_name = f"past_key_values.{i}.cross.key"
            past_value_name = f"past_key_values.{i}.cross.value"
            self.trt_context.set_input_shape(past_key_name, cross_attn_cache_shape)
            self.trt_context.set_input_shape(past_value_name, cross_attn_cache_shape)

    def set_context_mode_trt_context(self):
        # Create TRT context for context mode (1st decoder run) with optimization profile = 1
        self.context_trt_context = self.trt_engine.create_execution_context()
        self.context_trt_context.active_optimization_profile = 1

    def forward(
        self,
        input_ids,
        encoder_outputs: BaseModelOutput = None,
        attention_mask = None,
        image_embeds = None,
        **kwargs
    ) -> Seq2SeqLMOutput:

        # Actual sequence length of the input_ids and the output hidden_states.
        input_length = input_ids.shape[1]

        # Compute past decoder length
        if not self.config.use_cache:
            mask_length = input_length
        else:
            if kwargs.get("past_key_values") is None:
                self.past_decoder_length = 0
            mask_length = input_length + self.past_decoder_length

        if self.config.consume_image_embeds and image_embeds is None:
            raise RuntimeError("You are using a Vision2Seq model but does not provide image_embeds.")

        if self.config.is_encoder_decoder:
            if encoder_outputs is None:
                raise RuntimeError("You are using a encoder_decoder model but does not provide encoder_outputs.")
            encoder_hidden_states = encoder_outputs.last_hidden_state
        else:
            # When seq len != 1 for decoder only model, meaning it is context mode.
            if self.config.use_cache and input_length > 1:
                self.context_mode = True

        # Choose context
        ctx = self.context_trt_context if self.context_mode else self.trt_context
        input_shape = input_ids.shape
        is_cpu_mode = input_ids.device == torch.device("cpu")

        input_ids = input_ids.to(self.int_type).contiguous()

        if is_cpu_mode:
            input_ids = input_ids.cuda()

        use_cuda_graph = self.use_cuda_graph

        # Only enable CUDA graph in generate phase with kv_cache
        if not self.config.use_cache or input_length != 1 and self.context_mode:
            use_cuda_graph = False
            for self.cur_cuda_graph_instance in range(len(self.cuda_graph_instances)):
                if self.cuda_graph_instances[self.cur_cuda_graph_instance] is not None:
                    CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instances[self.cur_cuda_graph_instance]))
                self.cuda_graph_instances[self.cur_cuda_graph_instance] = None

        if use_cuda_graph:
            self.cuda_graph_input_data[:,:] = input_ids[:,:]
            ctx.set_tensor_address(self.main_input_name, self.cuda_graph_input_data.data_ptr())
        else:
            ctx.set_tensor_address(self.main_input_name, input_ids.data_ptr())
        ctx.set_input_shape(self.main_input_name, input_shape)


        if self.config.use_mask:

            if attention_mask is None:
                attention_mask = torch.ones((input_ids.shape[0], mask_length), device=input_ids.device)

            assert attention_mask.shape[1] == mask_length, f"Mask length should equal to {mask_length}"

            attention_mask = attention_mask.to(self.int_type).contiguous()

            if is_cpu_mode:
                attention_mask = attention_mask.cuda()

            ctx.set_tensor_address("attention_mask", attention_mask.data_ptr())
            ctx.set_input_shape("attention_mask", attention_mask.shape)

        if self.config.use_cache:
            if kwargs.get("past_key_values") is None:
                # If no past_key_values are passed in from prepare_inputs_for_generation, need to reset encoder_hidden_states and cross_attn_cache
                if self.config.is_encoder_decoder:
                    self.set_encoder_hidden_states(encoder_hidden_states)
                    self.set_cross_attn_cache(encoder_hidden_states)
                if self.config.consume_image_embeds:
                    self.set_image_embeds(image_embeds)

            self_attn_keys_cache_shape = self._get_self_attn_keys_cache_shape()
            self_attn_values_cache_shape = self._get_self_attn_vals_cache_shape()
            for i in range(self.num_decoder_layers):
                # If inputs shape has record and the record of context and shape are correct, skip it
                key_record = self.input_shape_change_record.get("past_key_values.{i}.self.key", None)
                value_record = self.input_shape_change_record.get("past_key_values.{i}.self.value", None)
                if key_record is None or value_record is None or \
                    key_record[0] != ctx or value_record[0] != ctx or \
                    key_record[1] != self_attn_keys_cache_shape or value_record[1] != self_attn_values_cache_shape:
                    # record miss, set input shape
                    ctx.set_input_shape(f"past_key_values.{i}.self.key", self_attn_keys_cache_shape)
                    ctx.set_input_shape(f"past_key_values.{i}.self.value", self_attn_values_cache_shape)

        elif input_length == 1:
            if self.config.is_encoder_decoder:
                encoder_hidden_states = encoder_outputs.last_hidden_state
                self.set_encoder_hidden_states(encoder_hidden_states)
            if self.config.consume_image_embeds:
                self.set_image_embeds(image_embeds)

        # Launch TRT inference.
        assert ctx.all_shape_inputs_specified

        if self.use_cuda_graph:
            if self.cuda_graph_instances[self.cur_cuda_graph_instance] is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instances[self.cur_cuda_graph_instance], self.stream))
            else:
                ctx.execute_async_v3(self.stream)
        else:
            ctx.execute_async_v3(self.stream)

        early_switch = False
        if self.config.use_cache and self.past_decoder_length != 0:
            for i in range(self.num_decoder_layers):
                next_self_attn_keys_shape = self._get_next_self_attn_keys_cache_shape()
                next_self_attn_vals_shape = self._get_next_self_attn_vals_cache_shape()
                self.input_shape_change_record.clear()
                # set next iter input shape when cpu is waiting for the current iter to complete on gpu
                ctx.set_input_shape(f"past_key_values.{i}.self.key", next_self_attn_keys_shape)
                ctx.set_input_shape(f"past_key_values.{i}.self.value", next_self_attn_vals_shape)
                key_record = self.input_shape_change_record["past_key_values.{i}.self.key"] = [ctx, next_self_attn_keys_shape]
                value_record = self.input_shape_change_record["past_key_values.{i}.self.value"] = [ctx, next_self_attn_vals_shape]

            if self.use_cuda_graph:
                self._switch_cache()
                early_switch = True

                ctx.set_tensor_address(self.main_input_name, self.cuda_graph_input_data.data_ptr())
                ctx.set_input_shape(self.main_input_name, input_shape)
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(self.stream_for_cuda_graph, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
                ctx.execute_async_v3(self.stream_for_cuda_graph)
                graph = CUASSERT(cudart.cudaStreamEndCapture(self.stream_for_cuda_graph))[0]
                # instantiate cuda graph execution
                self.cur_cuda_graph_instance = ((self.cur_cuda_graph_instance + 1) % 2)
                if self.cuda_graph_instances[self.cur_cuda_graph_instance] is not None:
                    err = cudart.cudaGraphExecUpdate(self.cuda_graph_instances[self.cur_cuda_graph_instance], graph)
                    if err != cudart.cudaError_t.cudaSuccess:
                        CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instances[self.cur_cuda_graph_instance]))
                        self.cuda_graph_instances[self.cur_cuda_graph_instance] = CUASSERT(cudart.cudaGraphInstantiate(graph, 0))[0]
                    CUASSERT(cudart.cudaGraphUpload(self.cuda_graph_instances[self.cur_cuda_graph_instance], self.stream))
                else:
                    self.cuda_graph_instances[self.cur_cuda_graph_instance] = CUASSERT(cudart.cudaGraphInstantiate(graph, 0))[0]

                CUASSERT(cudart.cudaGraphDestroy(graph))

        CUASSERT(cudart.cudaStreamSynchronize(self.stream))

        logits_length = self.expand_size * input_length * self.config.vocab_size
        logits = self.logits[:logits_length].view(self.expand_size, input_length, self.config.vocab_size)
        if self._return_full_logits:
            if not self.config.use_cache or (self.config.use_cache and kwargs.get("past_key_values") is None):
                self.full_logits = logits
            else:
                # KV Cache mode, concat logits for seq > 1
                self.full_logits = torch.cat((self.full_logits, logits), dim=1)
        present_key_values = None
        if self.config.use_cache:
            if not early_switch:
                self._switch_cache()

            present_key_values = ()
            self.past_decoder_length += input_length

            if self.context_mode:
                result_id = 0
            else:
                result_id = self.cache_id

            for i in range(self.num_decoder_layers):
                self_attn_k_output = self.self_attn_cache[f"key_values.{i}.self.key"][result_id]
                self_attn_v_output = self.self_attn_cache[f"key_values.{i}.self.value"][result_id]

                present_key_values += ((self_attn_k_output, self_attn_v_output),)

            if self.context_mode:
                self.context_mode = False

        return Seq2SeqLMOutput(logits=logits, past_key_values=present_key_values,)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        encoder_outputs = None,
        image_embeds = None,
        **kwargs
    ):
        # In HuggingFace generation_utils.py, this function will be called at each decoding step, before running the decoder's forward().

        if self.target_ids is not None:
            input_ids = self.target_ids[:,:input_ids.shape[1]]

        input_args = {}

        if self.config.is_encoder_decoder and self.config.use_mask:
            input_args["attention_mask"] = torch.ones_like(input_ids)

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        input_args["input_ids"] = input_ids

        if self.config.consume_image_embeds:
            input_args["image_embeds"] = image_embeds

        if self.config.is_encoder_decoder:
            # In HuggingFace, once a sequence has finished, HuggingFace will only decode unfinished sequence, so decoder attention mask is always 1.
            # But let's move attention mask generation out and controllable.
            input_args["encoder_outputs"] = encoder_outputs
        elif self.config.use_mask:
            input_args["attention_mask"] = kwargs["attention_mask"]

        if self.config.use_cache:
            input_args["use_cache"] = True
            input_args["past_key_values"] = past_key_values

        return input_args

    # Models with unique kv-cache shapes (like BLOOM) can override this
    def _get_self_attn_keys_cache_shape(self):
        return (self.expand_size, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)

    # Models with unique kv-cache shapes (like BLOOM) can override this
    def _get_self_attn_vals_cache_shape(self):
        return (self.expand_size, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)

    # Models with unique kv-cache shapes (like BLOOM) can override this
    def _get_next_self_attn_keys_cache_shape(self):
        return (self.expand_size, self.num_heads, self.past_decoder_length + 1, self.embedding_size_per_head)

    # Models with unique kv-cache shapes (like BLOOM) can override this
    def _get_next_self_attn_vals_cache_shape(self):
        return (self.expand_size, self.num_heads, self.past_decoder_length + 1, self.embedding_size_per_head)

    def _switch_cache(self):

        if not (self.cache_id == 0 and self.context_mode):
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:
                    self_attn_name = f"key_values.{i}.self.{code}"
                    self_attn_buffer = self.self_attn_cache[self_attn_name]
                    past_name = "past_" + self_attn_name
                    present_name = "present_" + self_attn_name
                    past_address = self_attn_buffer[1-self.cache_id].data_ptr()
                    present_address = self_attn_buffer[self.cache_id].data_ptr()
                    self.trt_context.set_tensor_address(past_name, past_address)
                    self.trt_context.set_tensor_address(present_name, present_address)

            self.cache_id = (self.cache_id + 1) % 2

    def _reorder_cache(self, past, beam_idx):

        # `past` does not change, but we have reordered the cache within the class.
        if past is None:
            G_LOGGER.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        self_attn_output_shape = (self.expand_size, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)
        self_attn_output_size = self.expand_size * self.num_heads * (self.past_decoder_length) * self.embedding_size_per_head


        # TODO: This step has significant D2D memory copy.
        for i in range(self.num_decoder_layers):

            k = self.self_attn_cache[f"key_values.{i}.self.key"][self.cache_id][:self_attn_output_size].view(*self_attn_output_shape)
            v = self.self_attn_cache[f"key_values.{i}.self.value"][self.cache_id][:self_attn_output_size].view(*self_attn_output_shape)

            reordered_k = k.index_select(0, beam_idx.to(k.device))
            reordered_v = v.index_select(0, beam_idx.to(v.device))

            self.self_attn_cache[f"key_values.{i}.self.key"][self.cache_id][:self_attn_output_size] = reordered_k.flatten().contiguous()
            self.self_attn_cache[f"key_values.{i}.self.value"][self.cache_id][:self_attn_output_size] = reordered_v.flatten().contiguous()

        return past

class Seq2SeqTRT(TRTInferenceCommand):
    def __init__(
        self,
        config_class=Seq2SeqModelTRTConfig,
        description="Runs trt results for Seq2Seq model.",
        model_classes=Seq2SeqModelClass,
        **kwargs
    ):
        super().__init__(
            network_config=config_class,
            description=description,
            model_classes=model_classes,
            **kwargs
        )
        self.onnx_encoder = None
        self.onnx_decoder = None
        self.encoder_class = Seq2SeqTRTEncoder
        self.decoder_class = Seq2SeqTRTDecoder

    def process_framework_specific_arguments(
        self,
        disable_preview_dynamic_shapes: bool = False,
        dynamic_batch: bool = False,
        min_dynamic_batch: int = None,
        max_dynamic_batch: int = None,
        encoder_engine: str = None,
        decoder_engine: str = None,
        cache_generator_engine: str = None,
        use_timing_cache: bool = False,
        nvtx_verbose: bool = False,
        use_cuda_graph: bool = False,
        **kwargs
    ):
        self.encoder_hidden_size = self.config.hidden_size
        self.disable_preview_dynamic_shapes = disable_preview_dynamic_shapes
        self.dynamic_batch = dynamic_batch
        self.opt_input_seq_len = self.config.opt_input_length
        self.opt_output_seq_len = self.config.opt_output_length

        # Ensure validity of batch size being built
        if self.dynamic_batch:
            min_dynamic_batch = int(setup_benchmark_arg(min_dynamic_batch, "min_dynamic_batch", 1))
            assert min_dynamic_batch <= self.config.batch_size, \
                "min_dynamic_batch {} should be <= batch_size {}".format(min_dynamic_batch, self.config.batch_size)
            self.min_dynamic_batch = min_dynamic_batch

            max_dynamic_batch = int(setup_benchmark_arg(max_dynamic_batch, "max_dynamic_batch", self.config.batch_size))
            assert self.config.batch_size <= max_dynamic_batch, \
                "max_dynamic_batch {} should be >= batch_size {}".format(max_dynamic_batch, self.config.batch_size)
            self.max_dynamic_batch = max_dynamic_batch

        self.min_batch_size = self.min_dynamic_batch if self.dynamic_batch else self.config.batch_size
        self.max_batch_size = self.max_dynamic_batch if self.dynamic_batch else self.config.batch_size

        self.min_expand_size = self.config._compute_expand_size(self.min_batch_size, self.config.num_beams)
        self.opt_expand_size = self.config._compute_expand_size(self.config.batch_size, self.config.num_beams)
        self.max_expand_size = self.config._compute_expand_size(self.max_batch_size, self.config.num_beams)

        self.use_generator = self.config.is_encoder_decoder and self.config.use_cache

        self.workspace.set_encoder_engine_path(encoder_engine)
        self.workspace.set_decoder_engine_path(decoder_engine)
        self.workspace.set_cross_attn_generator_engine_path(cache_generator_engine)

        self.use_timing_cache = use_timing_cache
        self.timing_cache = self.workspace.get_timing_cache() if self.use_timing_cache else None

        self.embedding_size_per_head = self.config.d_kv

        # In building the engine, setting nvtx verbose level does not affect performance, so always set to True.
        self.nvtx_verbose_build = True

        self.nvtx_verbose_inference = nvtx_verbose # In inference, nvtx verbose level may affect performance.

        self.use_cuda_graph = use_cuda_graph
        if self.use_generator:
            self.cross_attn_cache_generator_engine = None
            self.onnx_cross_attn_cache_generator = None
        return kwargs


    def setup_tokenizer_and_model(self):
        """
        Set up tokenizer and TRT engines for TRT Runner.
        """
        self.tokenizer = self.download_tokenizer()
        t0 = time.time()
        # Check whether user passed engine
        if self.check_engine_inputs_valid() and self.setup_engines_from_path(
            encoder_engine_fpath=self.workspace.encoder_engine_path,
            decoder_engine_fpath=self.workspace.decoder_engine_path,
            cross_attn_cache_generator_engine_fpath=self.workspace.cross_attn_generator_engine_path
        ):
            G_LOGGER.info("TRT engine loaded successful from arguments. Engine loading time: {:.4f}s".format(time.time() - t0))
        else:
            G_LOGGER.info("Cannot load existing TRT engines from arguments. Attempt to obtain from onnx model.")
            self.workspace.create_onnx_folders()
            self.load_onnx_model()
            self.setup_engines_from_onnx()
            G_LOGGER.info("TRT engine successfully obtained from onnx models. Total engine loading/building time: {:.4f}s".format(time.time() - t0))

        trt_models = [
            NetworkModel(
                name=self.config.NETWORK_DECODER_SEGMENT_NAME,
                fpath=self.decoder_engine.fpath,
            )
        ]

        if self.config.is_encoder_decoder:
            trt_models.append(
                NetworkModel(
                    name=self.config.NETWORK_ENCODER_SEGMENT_NAME,
                    fpath=self.encoder_engine.fpath,
                )
            )

        if self.use_generator:
            trt_models.append(
                NetworkModel(
                    name=self.config.NETWORK_CROSS_ATTENTION_CACHE_GENERATOR_NAME,
                    fpath=self.cross_attn_cache_generator_engine.fpath,
                )
            )

        return NetworkModels(torch=None, onnx=None, trt=trt_models)

    def check_engine_inputs_valid(self):
        """
        Check whether all engines are valid.
        """
        encoder_engine_fpath = self.workspace.encoder_engine_path
        decoder_engine_fpath = self.workspace.decoder_engine_path
        cache_generator_engine_fpath = self.workspace.cross_attn_generator_engine_path
        is_encoder_valid = encoder_engine_fpath is not None and os.path.exists(encoder_engine_fpath)
        is_decoder_valid = decoder_engine_fpath is not None and os.path.exists(decoder_engine_fpath)
        is_generator_valid = cache_generator_engine_fpath is not None and os.path.exists(cache_generator_engine_fpath)
        if self.config.is_encoder_decoder:
            if self.config.use_cache:
                return is_encoder_valid and is_decoder_valid and is_generator_valid
            else:
                return is_encoder_valid and is_decoder_valid

        return is_decoder_valid

    def setup_engines_from_path(
        self,
        encoder_engine_fpath = None,
        decoder_engine_fpath = None,
        cross_attn_cache_generator_engine_fpath = None
    ):
        """
        Check whether user has passed in all required TRT engine name.
        If user passed valid TRT engines, will skip onnx export and use engine directly.
        """
        if decoder_engine_fpath is None:
            return False
        if self.config.is_encoder_decoder and encoder_engine_fpath is None:
            return False
        if self.use_generator and cross_attn_cache_generator_engine_fpath is None:
            return False

        try:
            self.decoder_engine = self.config.decoder_classes["engine"](decoder_engine_fpath, self.metadata)
            self.decoder = self.decoder_class(
                self.decoder_engine, self.metadata, self.config, self.nvtx_verbose_inference, self.use_cuda_graph
            )

            if self.config.is_encoder_decoder:
                self.encoder_engine = self.config.encoder_classes["engine"](encoder_engine_fpath, self.metadata)
                if self.config.use_fp32_encoder:
                    encoder_metadata = self.metadata._replace(precision=Precision(fp16=False))
                else:
                    encoder_metadata = self.metadata

                self.encoder = self.encoder_class(
                    self.encoder_engine, encoder_metadata, self.config, self.nvtx_verbose_inference
                )

            if self.use_generator:
                self.cross_attn_cache_generator_engine = self.config.cross_attn_cache_generator_classes["engine"](cross_attn_cache_generator_engine_fpath, self.metadata)
                self.decoder.set_cross_attn_cache_generator_engine(self.cross_attn_cache_generator_engine)

            return True
        except Exception as e:
            G_LOGGER.error("Cannot proceed with the provided engine. Attempt to generate from onnx. Reason is: {}".format(str(e)))

        return False

    def setup_engines_from_onnx(self) -> None:
        """
        Set up TRT engines from onnx files.
        """

        # Generate optimization profiles.
        # non-benchmarking mode: opt profile length is by default half of the max profile
        # benchmarking mode: user can specify opt and max profile by flags.
        #                    If no additional benchmarking flags are provided, it will just use n_positions for max coverage

        # Convert ONNX models to TRT engines.
        # bs tag first
        if self.dynamic_batch:
            engine_tag = f"minbs{self.min_dynamic_batch}-maxbs{self.max_dynamic_batch}"
        else:
            engine_tag = f"bs{self.config.batch_size}"

        if self.action == "benchmark":
            # When user does not input any profile_max_len, use seq as tag, both max are config max
            if self.seq_tag:
                engine_tag += f"-inseq{self.opt_input_seq_len}-outseq{self.opt_output_seq_len}"

            # When user input profile_max_len, reuse the engine for future use with different seq_len
            else:
                # When user inputs dynamic batch, enable engine reuse in future with different batch size.
                engine_tag += f"-inmax{self.config.max_input_profile_length}-outmax{self.config.max_output_profile_length}"

        if self.config.num_beams > 1:
            engine_tag += f"-beam{self.config.num_beams}"

        preview_features = [PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]

        if self.disable_preview_dynamic_shapes:
            engine_tag += "-noPreviewFasterDynamicShapes"
        else:
            preview_features.append(PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)

        # Set up decoder engine
        decoder_profiles = self._setup_decoder_profiles()

        decoder_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_decoder.fpath, engine_tag, self.engine_postfix)

        G_LOGGER.info("Setting up decoder engine in {}...".format(decoder_engine_path))

        # Print trtexec command for decoder
        G_LOGGER.debug(self._decoder_trtexec_command(decoder_engine_path))

        self.decoder_engine = self.onnx_decoder.as_trt_engine(
            decoder_engine_path,
            profiles=decoder_profiles,
            preview_features=preview_features,
            nvtx_verbose=self.nvtx_verbose_build,
            timing_cache=self.timing_cache,
        )

        self.decoder = self.decoder_class(
            self.decoder_engine, self.metadata, self.config, self.nvtx_verbose_inference, self.use_cuda_graph
        )
        self.workspace.set_decoder_engine_path(decoder_engine_path)

        # Set up encoder if needed
        if self.config.is_encoder_decoder:
            encoder_profiles = self._setup_encoder_profiles()
            encoder_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_encoder.fpath, engine_tag, self.engine_postfix).replace(f"-beam{self.config.num_beams}", "")

            G_LOGGER.info("Setting up encoder engine in {}...".format(encoder_engine_path))

            if self.config.use_fp32_encoder:
                encoder_metadata = self.metadata._replace(precision=Precision(fp16=False))
            else:
                encoder_metadata = self.metadata

            # Print trtexec command for encoder
            G_LOGGER.debug(self._encoder_trtexec_command(encoder_engine_path, encoder_metadata))

            self.encoder_engine = self.onnx_encoder.as_trt_engine(
                encoder_engine_path, # encoder engine name not affected by beam search
                profiles=encoder_profiles,
                preview_features=preview_features,
                nvtx_verbose=self.nvtx_verbose_build,
                timing_cache=self.timing_cache,
            )

            self.encoder = self.encoder_class(
                self.encoder_engine, encoder_metadata, self.config, self.nvtx_verbose_inference,
            )
            self.workspace.set_encoder_engine_path(encoder_engine_path)

        # Set up cross attn cache generator if needed
        if self.use_generator:
            generator_profiles = self._setup_generator_profiles()

            cross_attn_cache_generator_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_cross_attn_cache_generator.fpath, engine_tag, self.engine_postfix)

            G_LOGGER.info("use_cache=True. Setting up cross attention kv cache generator in {}...".format(cross_attn_cache_generator_engine_path))

            # Print trtexec command for generator
            G_LOGGER.debug(self._generator_trtexec_command(cross_attn_cache_generator_engine_path))

            self.cross_attn_cache_generator_engine = self.onnx_cross_attn_cache_generator.as_trt_engine(
                cross_attn_cache_generator_engine_path,
                profiles=generator_profiles,
                preview_features=preview_features,
                nvtx_verbose=self.nvtx_verbose_build,
                timing_cache=self.timing_cache,
            )
            self.decoder.set_cross_attn_cache_generator_engine(self.cross_attn_cache_generator_engine)
            self.workspace.set_cross_attn_generator_engine_path(cross_attn_cache_generator_engine_path)

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
            if self.config.is_encoder_decoder:
                decoder_profile.add(
                    "encoder_hidden_states",
                    min=(self.min_expand_size, 1, self.encoder_hidden_size),
                    opt=(self.opt_expand_size, self.opt_input_seq_len, self.encoder_hidden_size),
                    max=(self.max_expand_size, self.config.max_input_profile_length, self.encoder_hidden_size),
                )

            decoder_profiles = [decoder_profile]
        else:
            self_attn_profile = {
                "min": (self.min_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                "opt": (self.opt_expand_size, self.config.num_heads, self.opt_output_seq_len - 1, self.embedding_size_per_head),
                "max": (self.max_expand_size, self.config.num_heads, self.config.max_output_profile_length - 1, self.embedding_size_per_head),
            }

            cross_attn_profile = {
                "min": (self.min_expand_size, self.config.num_heads, 1, self.embedding_size_per_head),
                "opt": (self.opt_expand_size, self.config.num_heads, self.opt_input_seq_len, self.embedding_size_per_head),
                "max": (self.max_expand_size, self.config.num_heads, self.config.max_input_profile_length, self.embedding_size_per_head),
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

            if self.config.is_encoder_decoder:
                decoder_profile_generation.add(
                    "encoder_hidden_states",
                    min=(self.min_expand_size, 1, self.encoder_hidden_size),
                    opt=(self.opt_expand_size, self.opt_input_seq_len, self.encoder_hidden_size),
                    max=(self.max_expand_size, self.config.max_input_profile_length, self.encoder_hidden_size),
                )

            for i in range(self.config.num_decoder_layers):
                decoder_profile_generation = decoder_profile_generation.add(
                    f"past_key_values.{i}.self.key",
                    **self_attn_profile
                ).add(
                    f"past_key_values.{i}.self.value",
                    **self_attn_profile
                )
                if self.config.is_encoder_decoder:
                    decoder_profile_generation = decoder_profile_generation.add(
                        f"past_key_values.{i}.cross.key",
                        **cross_attn_profile
                    ).add(
                        f"past_key_values.{i}.cross.value",
                        **cross_attn_profile
                    )

            decoder_profiles = [decoder_profile_generation]

            # Decoder only model has "context phase" that is only used for the 1st decoder phase.
            if not self.config.is_encoder_decoder:
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

                self_attn_profile_context = {
                    "min": (self.min_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                    "opt": (self.opt_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                    "max": (self.max_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                }
                for i in range(self.config.num_decoder_layers):
                    decoder_profile_context = decoder_profile_context.add(
                        f"past_key_values.{i}.self.key",
                        **self_attn_profile_context
                    ).add(
                        f"past_key_values.{i}.self.value",
                        **self_attn_profile_context
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

            if self.config.is_encoder_decoder:
                min_shape += f",'encoder_hidden_states':{self.min_expand_size}x1x{self.encoder_hidden_size}"
                opt_shape += f",'encoder_hidden_states':{self.opt_expand_size}x{self.opt_input_seq_len}x{self.encoder_hidden_size}"
                max_shape += f",'encoder_hidden_states':{self.max_expand_size}x{self.config.max_input_profile_length}x{self.encoder_hidden_size}"

            command += f"{min_shape} {opt_shape} {max_shape} "

        else:
            min_shape += f"'input_ids':{self.min_expand_size}x1"
            opt_shape += f"'input_ids':{self.opt_expand_size}x1"
            max_shape += f"'input_ids':{self.max_expand_size}x1"
            if self.config.use_mask:
                min_shape += f",'attention_mask':{self.min_expand_size}x1"
                opt_shape += f",'attention_mask':{self.opt_expand_size}x{self.opt_output_seq_len}"
                max_shape += f",'attention_mask':{self.max_expand_size}x{self.config.max_output_profile_length}"

            if self.config.is_encoder_decoder:
                min_shape += f",'encoder_hidden_states':{self.min_expand_size}x1x{self.encoder_hidden_size}"
                opt_shape += f",'encoder_hidden_states':{self.opt_expand_size}x{self.opt_input_seq_len}x{self.encoder_hidden_size}"
                max_shape += f",'encoder_hidden_states':{self.max_expand_size}x{self.config.max_input_profile_length}x{self.encoder_hidden_size}"
            else:
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

            k = f",'past_key_values.*.self.key'"
            v = f",'past_key_values.*.self.value'"
            kv_min_str = f":{self.min_expand_size}x{self.config.num_heads}x0x{self.embedding_size_per_head}"
            kv_opt_str = f":{self.opt_expand_size}x{self.config.num_heads}x{self.opt_output_seq_len-1}x{self.embedding_size_per_head}"
            kv_max_str = f":{self.max_expand_size}x{self.config.num_heads}x{self.config.max_output_profile_length-1}x{self.embedding_size_per_head}"
            min_shape += k + kv_min_str + v + kv_min_str
            opt_shape += k + kv_opt_str + v + kv_opt_str
            max_shape += k + kv_max_str + v + kv_max_str

            if self.config.is_encoder_decoder:
                ck = f",'past_key_values.*.cross.key'"
                cv = f",'past_key_values.*.cross.value'"
                ckv_min_str = f":{self.min_expand_size}x{self.config.num_heads}x1x{self.embedding_size_per_head}"
                ckv_opt_str = f":{self.opt_expand_size}x{self.config.num_heads}x{self.opt_input_seq_len}x{self.embedding_size_per_head}"
                ckv_max_str = f":{self.max_expand_size}x{self.config.num_heads}x{self.config.max_input_profile_length}x{self.embedding_size_per_head}"
                min_shape += ck + ckv_min_str + cv + ckv_min_str
                opt_shape += ck + ckv_opt_str + cv + ckv_opt_str
                max_shape += ck + ckv_max_str + cv + ckv_max_str
            else:
                zero_min_str = f":{self.min_expand_size}x{self.config.num_heads}x0x{self.embedding_size_per_head}"
                zero_opt_str = f":{self.opt_expand_size}x{self.config.num_heads}x0x{self.embedding_size_per_head}"
                zero_max_str = f":{self.max_expand_size}x{self.config.num_heads}x0x{self.embedding_size_per_head}"
                cmin_shape += k + zero_min_str + v + zero_min_str
                copt_shape += k + zero_opt_str + v + zero_opt_str
                cmax_shape += k + zero_max_str + v + zero_max_str

            if self.config.is_encoder_decoder:
                command += f"{min_shape} {opt_shape} {max_shape} "
            else:
                command += f"--profile=0 {min_shape} {opt_shape} {max_shape} --profile=1 {cmin_shape} {copt_shape} {cmax_shape} "

        if self.metadata.precision.fp16:
            # For version compatibility. Older version of TRT does not support int64.
            int_type = "int64" if hasattr(trt,"int64") else "int32"
            inputIO = f"--inputIOFormats={int_type}:chw" # input_ids
            if self.config.use_mask:
                inputIO += f",{int_type}:chw" # attention_mask
            outputIO = "--outputIOFormats=fp16:chw" # logits
            if self.config.is_encoder_decoder:
                inputIO += ",fp16:chw" # encoder_hidden_states
            if self.config.use_cache:
                for _ in range(self.config.num_decoder_layers):
                    kv_precision = ",fp16:chw,fp16:chw"
                    inputIO += kv_precision # self attention
                    outputIO += kv_precision # self attention
                    if self.config.is_encoder_decoder:
                        inputIO += kv_precision # cross attention

            command += f"--fp16 --precisionConstraints=obey {inputIO} {outputIO} "

        if self.timing_cache is not None:
            command += f"--timingCacheFile={self.timing_cache} "

        return command

    def _setup_encoder_profiles(self):
        if not self.config.is_encoder_decoder:
            raise NotImplementedError("You are setting encoder profile for a non encoder_decoder model!")
        encoder_profile = Profile().add(
            "input_ids",
            min=(self.min_batch_size, 1),
            opt=(self.config.batch_size, self.opt_input_seq_len),
            max=(self.max_batch_size, self.config.max_input_profile_length),
        )

        if self.config.use_mask:
            encoder_profile = encoder_profile.add(
                "attention_mask",
                min=(self.min_batch_size, 1),
                opt=(self.config.batch_size, self.opt_input_seq_len),
                max=(self.max_batch_size, self.config.max_input_profile_length),
            )

        return [encoder_profile]

    # Used to create encoder trtexec command string for debugging accuracy/performance.
    def _encoder_trtexec_command(self, encoder_engine_path, encoder_metadata):
        command = f"trtexec --onnx={self.onnx_encoder.fpath} --verbose --saveEngine={encoder_engine_path} "
        min_shape = f"--minShapes='input_ids':{self.min_batch_size}x1"
        opt_shape = f"--optShapes='input_ids':{self.config.batch_size}x{self.opt_input_seq_len}"
        max_shape = f"--maxShapes='input_ids':{self.max_batch_size}x{self.config.max_input_profile_length}"
        command += f"{min_shape} {opt_shape} {max_shape} "

        if encoder_metadata.precision.fp16:
            inputIO = "--inputIOFormats=int32:chw" # input_ids
            outputIO = "--outputIOFormats=fp16:chw" # encoder_hidden_states
            command += f"--fp16 --precisionConstraints=obey {inputIO} {outputIO} "

        if self.timing_cache is not None:
            command += f"--timingCacheFile={self.timing_cache} "

        return command

    def _setup_generator_profiles(self):
        if not self.use_generator:
            raise NotImplementedError("You do not need cross attention generator but is setting it up!")
        return [Profile().add(
            "encoder_hidden_states",
            min=(self.min_expand_size, 1, self.encoder_hidden_size),
            opt=(self.opt_expand_size, self.opt_input_seq_len, self.encoder_hidden_size),
            max=(self.max_expand_size, self.config.max_input_profile_length, self.encoder_hidden_size),
        )]

    # Used to create generator trtexec command string for debugging accuracy/performance.
    def _generator_trtexec_command(self, cross_attn_cache_generator_engine_path):
        command = f"trtexec --onnx={self.onnx_cross_attn_cache_generator.fpath} --verbose --saveEngine={cross_attn_cache_generator_engine_path} "
        min_shape = f"--minShapes='encoder_hidden_states':{self.min_expand_size}x1x{self.encoder_hidden_size}"
        opt_shape = f"--optShapes='encoder_hidden_states':{self.opt_expand_size}x{self.opt_input_seq_len}x{self.encoder_hidden_size}"
        max_shape = f"--maxShapes='encoder_hidden_states':{self.max_expand_size}x{self.config.max_input_profile_length}x{self.encoder_hidden_size}"
        command += f"{min_shape} {opt_shape} {max_shape} "

        if self.metadata.precision.fp16:
            inputIO = "--inputIOFormats=fp16:chw" # input_ids
            outputIO = "--outputIOFormats="
            for _ in range(self.config.num_decoder_layers):
                kv_precision = "fp16:chw,fp16:chw,"
                outputIO += kv_precision # cross attention

            command += f"--fp16 --precisionConstraints=obey {inputIO} {outputIO.rstrip(',')} "

        if self.timing_cache is not None:
            command += f"--timingCacheFile={self.timing_cache} "

        return command

    def cleanup(self) -> None:

        # Deactivates context
        if self.encoder:
            self.encoder.release()
        if self.decoder:
            self.decoder.release()

        if not self.keep_trt_engine:
            self.decoder_engine.cleanup()
            if self.config.is_encoder_decoder:
                self.encoder_engine.cleanup()
            if self.use_generator:
                self.cross_attn_cache_generator_engine.cleanup()

        if not self.keep_onnx_model:
            if self.onnx_decoder:
                self.onnx_decoder.cleanup()
            if self.config.is_encoder_decoder and self.onnx_encoder:
                self.onnx_encoder.cleanup()
            if self.use_generator and self.onnx_cross_attn_cache_generator:
                self.onnx_cross_attn_cache_generator.cleanup()

        if not self.keep_torch_model and self.workspace.torch_path is not None:

            confirm_folder_delete(
                self.workspace.torch_path,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not self.keep_trt_engine:
            self.workspace.cleanup(force_remove=False)


RUN_CMD = Seq2SeqTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
