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
import sys

import numpy as np
import tensorrt as trt
import torch
from transformers.configuration_utils import PretrainedConfig

sys.path.append('../../HuggingFace') # Include HuggingFace directory
from NNDF.models import TRTEngineFile
from NNDF.networks import NetworkMetadata
from NNDF.tensorrt_utils import TRTNativeRunner
from NNDF.logger import G_LOGGER
from Seq2Seq.export import DecoderTRTEngine

from HuggingFace.NNDF.tensorrt_utils import TRTNativeRunner, CUASSERT
from cuda import cudart


class GPTTRTDecoder(TRTNativeRunner):

    INPUT_IDS_INDEX = 0
    POSITION_IDS_INDEX = 1
    ATTENTION_MASK_INDEX = 2

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        use_cache: bool,
        use_fp8_storage: bool,
        cfg,
        network_metadata: NetworkMetadata = None,
        hf_config: PretrainedConfig = None,
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config)
        self.use_cache = use_cache
        self.use_fp8_storage = use_fp8_storage
        if self.use_cache:
            self._set_context_mode_trt_context()
        self.io_names = set()
        self.input_tensor_names = set()
        for i in range(self.trt_engine.num_io_tensors):
            tensor_name = self.trt_engine.get_tensor_name(i)
            self.io_names.add(tensor_name)
            if self.trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_tensor_names.add(tensor_name)

        self.cfg = cfg
        logits_size = self.cfg.batch_size * self.cfg.model.max_seq_len * self.cfg.model.vocab_size

        self.batch_size = self.cfg.batch_size
        self.max_seq_len = self.cfg.model.max_seq_len
        self.num_layers = self.cfg.model.num_layers
        self.nb_heads = self.cfg.model.nb_heads
        self.head_size = self.cfg.model.head_size

        dtype = self.get_torch_type(self.get_output_name())
        self.logits = torch.zeros(logits_size, dtype=dtype).contiguous().cuda()


        self.init_kv_cache()
        self.past_decoder_length = 0

        # Setting next input shape when executing gpu kernel.
        # Use dict to record which inputs have changed.
        self.input_shape_change_record = dict()

    def init_kv_cache(self):
        # kv cache buffer
        self.attention_kv_cache_buffer = dict()
        cache_dtype = torch.float16
        if self.use_fp8_storage:
            cache_dtype = torch.uint8
        for i in range(self.num_layers):
            for code in ["key", "value"]:
                attention_kv_cache_name = self.make_kv_cache_name(i, code)
                self.attention_kv_cache_buffer[attention_kv_cache_name] = torch.empty(
                    self.max_seq_len,
                    self.batch_size,
                    self.nb_heads,
                    self.head_size,
                    dtype=cache_dtype,
                    device=torch.cuda.current_device(),
                ).contiguous().cuda()
                

    def make_kv_cache_name(self, layer, code):
        return f"key_values.{layer}.decoder.{code}"

    def _set_context_mode_trt_context(self):
        # Create TRT context for context mode (1st decoder run) with optimization profile index = 1
        self.context_trt_context = self.trt_engine.create_execution_context()
        self.context_trt_context.active_optimization_profile = 1
        self.kv_cache_binding_offset = self.trt_engine.num_bindings // self.trt_engine.num_optimization_profiles

    def get_torch_type(self, name):
        trt_type = self.trt_engine.get_binding_dtype(name)
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
            trt.uint8: torch.uint8,
            trt.bfloat16: torch.bfloat16,
        }
        if trt_type in mapping:
            return mapping[trt_type]
        raise ValueError(f"Got unexpected tensorrt dtype {trt_type} in get_torch_type().")

    def get_input_ids_name(self):
        return self.trt_engine.get_binding_name(self.INPUT_IDS_INDEX)

    def has_position_ids(self):
        # If the input at POSITION_IDS_INDEX has a dimension of 2, assume it is position_ids.
        return len(self.trt_engine.get_binding_shape(self.POSITION_IDS_INDEX)) == 2

    def get_position_ids_name(self):
        if self.has_position_ids():
            return self.trt_engine.get_binding_name(self.POSITION_IDS_INDEX)
        else:
            return None

    def get_output_name(self):
        if self.use_cache:
            return self.trt_engine.get_binding_name(self.kv_cache_binding_offset - 1)
        return self.trt_engine.get_binding_name(self.trt_engine.num_bindings - 1)

    def has_attention_mask(self):
        if self.ATTENTION_MASK_INDEX < self.trt_engine.num_bindings:
            return self.trt_engine.get_binding_name(self.ATTENTION_MASK_INDEX) == "attention_mask"
        return False

    def get_attention_mask_name(self):
        if self.has_attention_mask():
            return self.trt_engine.get_binding_name(self.ATTENTION_MASK_INDEX)
        return None

    def run(self, output_name, io_descs, seq_len, context_mode=False):
        torch.cuda.nvtx.range_push("TRT Setup")
        if self.use_cache:
            if context_mode:
                self.past_decoder_length = 0
            else:
                # When kv-cache is used, seq_len is always 1 in Generation phase.
                seq_len = 1
            cur_shape = (self.past_decoder_length, self.batch_size, self.nb_heads, self.head_size)
            new_shape = (seq_len, self.batch_size, self.nb_heads, self.head_size)
            assert self.past_decoder_length + seq_len < self.max_seq_len
            offset = self.batch_size*self.nb_heads*self.head_size*self.past_decoder_length
            for i in range(self.num_layers):
                for code in ["key", "value"]:
                    attention_kv_cache_name = self.make_kv_cache_name(i, code)
                    cur_address = self.attention_kv_cache_buffer[attention_kv_cache_name].data_ptr()
                    # new kv address start from the past kv-cache data end
                    io_descs[f"past_{attention_kv_cache_name}"] = (cur_address, cur_shape)
                    new_address = cur_address + offset*self.attention_kv_cache_buffer[attention_kv_cache_name].element_size()
                    modifier = ""
                    if self.use_fp8_storage:
                        modifier = "_qfp8"
                    new_kv_name = f"new_{attention_kv_cache_name}{modifier}"
                    io_descs[new_kv_name] = (new_address, new_shape)
            self.past_decoder_length += seq_len
        else:
            self.past_decoder_length = 0
        # Set active optimization profile and active execution context.
        self.trt_context.active_optimization_profile = self.profile_idx
        active_context = self.trt_context
        if context_mode and self.use_cache:
            active_context = self.context_trt_context

        # Set up input bindings.
        for name, tensor_shape in io_descs.items():
            active_context.set_tensor_address(name, tensor_shape[0])
            if name in self.input_tensor_names:
                if name in self.input_shape_change_record and \
                    self.input_shape_change_record[name][0] == active_context and \
                    self.input_shape_change_record[name][1] == tensor_shape[1]:
                    continue
                else:
                    active_context.set_input_shape(name, tensor_shape[1])
            elif self.use_cache:
                pass
            else:
                assert False, "All tensors must be inputs for non-KV mode"
        assert active_context.all_shape_inputs_specified

        # Set up output bindings.
        assert output_name == self.get_output_name()
        engine_out_torch_type = self.get_torch_type(output_name)
        if self.logits.dtype != engine_out_torch_type:
            raise ValueError(f"Output data type does not match, {self.logits.dtype} vs. {engine_out_torch_type}.")
        shape = active_context.get_tensor_shape(output_name)
        active_context.set_tensor_address(output_name, self.logits.data_ptr())


        # Execute inference.
        torch.cuda.nvtx.range_pop() # "TRT Setup"
        active_context.execute_async_v3(self.stream)
        if not context_mode and self.use_cache:
            self.input_shape_change_record.clear()
            for i in range(self.num_layers):
                for code in ["key", "value"]:
                    next_past_shape = (self.past_decoder_length, self.batch_size, self.nb_heads, self.head_size)
                    attention_kv_cache_name = self.make_kv_cache_name(i, code)
                    # set next iter input shape when cpu idle
                    active_context.set_input_shape(f"past_{attention_kv_cache_name}", next_past_shape)
                    self.input_shape_change_record[f"past_{attention_kv_cache_name}"] = [active_context, next_past_shape]
        CUASSERT(cudart.cudaStreamSynchronize(self.stream))
        if len(shape) != 3:
            raise ValueError("Output must have a dimension of 3.")
        output = self.logits[:shape[0] * shape[1] * shape[2]].view(tuple(shape))
        return output

def load_trt_model(cfg):
    G_LOGGER.info(f'Loading TensorRT engine from {cfg.trt_engine_file} with use_cache={cfg.use_cache}, use_fp8_storage={cfg.onnx_export_options.use_fp8_storage} ')
    trt_engine_file = DecoderTRTEngine(cfg.trt_engine_file)
    return GPTTRTDecoder(trt_engine_file, cfg.use_cache, cfg.onnx_export_options.use_fp8_storage, cfg)
