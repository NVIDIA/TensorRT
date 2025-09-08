#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import os
import subprocess
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import onnx
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from onnx import numpy_helper
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    engine_from_bytes,
)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


# Map of TensorRT dtype -> torch dtype
trt_to_torch_dtype_dict = {
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
}


def _CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


def get_refit_weights(state_dict, onnx_opt_path, weight_name_mapping, weight_shape_mapping):
    onnx_opt_dir = os.path.dirname(onnx_opt_path)
    onnx_opt_model = onnx.load(onnx_opt_path)
    # Create initializer data hashes
    initializer_hash_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
        initializer_hash = hash(initializer_data.data.tobytes())
        initializer_hash_mapping[initializer.name] = initializer_hash

    refit_weights = OrderedDict()
    updated_weight_names = set()  # save names of updated weights to refit only the required weights
    for wt_name, wt in state_dict.items():
        # query initializer to compare
        initializer_name = weight_name_mapping[wt_name]
        initializer_hash = initializer_hash_mapping[initializer_name]

        # get shape transform info
        initializer_shape, is_transpose = weight_shape_mapping[wt_name]
        if is_transpose:
            wt = torch.transpose(wt, 0, 1)
        else:
            wt = torch.reshape(wt, initializer_shape)

        # include weight if hashes differ
        wt_hash = hash(wt.cpu().detach().numpy().astype(np.float16).data.tobytes())
        if initializer_hash != wt_hash:
            updated_weight_names.add(initializer_name)
        # Store all weights as the refitter may require unchanged weights too
        # docs: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#refitting-engine-c
        refit_weights[initializer_name] = wt.contiguous()
    return refit_weights, updated_weight_names


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, refit_weights, updated_weight_names):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        refitted_weights = set()

        def refit_single_weight(trt_weight_name):
            # get weight from state dict
            trt_datatype = refitter.get_weights_prototype(trt_weight_name).dtype
            refit_weights[trt_weight_name] = refit_weights[trt_weight_name].to(trt_to_torch_dtype_dict[trt_datatype])

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(
                trt_datatype, refit_weights[trt_weight_name].data_ptr(), torch.numel(refit_weights[trt_weight_name])
            )
            trt_wt_location = (
                trt.TensorLocation.DEVICE if refit_weights[trt_weight_name].is_cuda else trt.TensorLocation.HOST
            )

            # apply refit
            refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            if trt_weight_name not in updated_weight_names:
                continue

            refit_single_weight(trt_weight_name)

        # iterate through missing weights required by tensorrt - addresses the case where lora_scale=0
        for trt_weight_name in refitter.get_missing_weights():
            refit_single_weight(trt_weight_name)

        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(
        self,
        onnx_path,
        strongly_typed=False,
        fp16=True,
        bf16=False,
        tf32=False,
        int8=False,
        fp8=False,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
        native_instancenorm=True,
        verbose=False,
        weight_streaming=False,
        builder_optimization_level=3,
        precision_constraints='none',
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")

        # Handle weight streaming case: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#streaming-weights.
        if weight_streaming:
            strongly_typed, fp16, bf16, int8, fp8 = True, False, False, False, False

        # Base command
        build_command = [f"polygraphy convert {onnx_path} --convert-to trt --output {self.engine_path}"]

        # Precision flags
        build_args = [
            "--fp16" if fp16 else "",
            "--bf16" if bf16 else "",
            "--tf32" if tf32 else "",
            "--fp8" if fp8 else "",
            "--int8" if int8 else "",
            "--strongly-typed" if strongly_typed else "",
        ]

        # Additional arguments
        build_args.extend([
            "--weight-streaming" if weight_streaming else "",
            "--refittable" if enable_refit else "",
            "--tactic-sources" if not enable_all_tactics else "",
            "--onnx-flags native_instancenorm" if native_instancenorm else "",
            f"--builder-optimization-level {builder_optimization_level}",
            f"--precision-constraints {precision_constraints}",
        ])

        # Timing cache
        if timing_cache:
            build_args.extend([
                f"--load-timing-cache {timing_cache}",
                f"--save-timing-cache {timing_cache}"
            ])

        # Verbosity setting
        verbosity = "extra_verbose" if verbose else "error"
        build_args.append(f"--verbosity {verbosity}")

        # Output names
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            build_args.append(f"--trt-outputs {' '.join(update_output_names)}")

        # Input profiles
        if input_profile:
            profile_args = defaultdict(str)
            for name, dims in input_profile.items():
                assert len(dims) == 3
                profile_args["--trt-min-shapes"] += f"{name}:{str(list(dims[0])).replace(' ', '')} "
                profile_args["--trt-opt-shapes"] += f"{name}:{str(list(dims[1])).replace(' ', '')} "
                profile_args["--trt-max-shapes"] += f"{name}:{str(list(dims[2])).replace(' ', '')} "

            build_args.extend(f"{k} {v}" for k, v in profile_args.items())

        # Filter out empty strings and join command
        build_args = [arg for arg in build_args if arg]
        final_command = ' '.join(build_command + build_args)

        # Execute command with improved error handling
        try:
            print(f"Engine build command: {final_command}")
            subprocess.run(final_command, check=True, shell=True)
        except subprocess.CalledProcessError as exc:
            error_msg = (
                f"Failed to build TensorRT engine. Error details:\n"
                f"Command: {exc.cmd}\n"
            )
            raise RuntimeError(error_msg) from exc

    def load(self, weight_streaming=False, weight_streaming_budget_percentage=None):
        if self.engine is not None:
            print(f"[W]: Engine {self.engine_path} already loaded, skip reloading")
            return
        if not hasattr(self, "engine_bytes_cpu") or self.engine_bytes_cpu is None:
            # keep a cpu copy of the engine to reduce reloading time.
            print(f"Loading TensorRT engine to cpu bytes: {self.engine_path}")
            self.engine_bytes_cpu = bytes_from_path(self.engine_path)
        print(f"Loading TensorRT engine from bytes: {self.engine_path}")
        self.engine = engine_from_bytes(self.engine_bytes_cpu)
        if weight_streaming:
            if weight_streaming_budget_percentage is None:
                warnings.warn(
                    f"Weight streaming budget is not set for {self.engine_path}. Weights will not be streamed."
                )
            else:
                self.engine.weight_streaming_budget_v2 = int(
                    weight_streaming_budget_percentage / 100 * self.engine.streamable_weights_size
                )

    def unload(self, verbose=True):
        if self.engine is not None:
            if verbose:
                print(f"Unloading TensorRT engine: {self.engine_path}")
            del self.engine
            self.engine = None
            gc.collect()
        else:
            if verbose:
                print(f"[W]: Unload an unloaded engine {self.engine_path}, skip unloading")

    def activate(self, device_memory=None):
        if device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = device_memory
        else:
            self.context = self.engine.create_execution_context()

    def reactivate(self, device_memory):
        assert self.context
        self.context.device_memory = device_memory

    def deactivate(self):
        del self.context
        self.context = None

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
                print(
                    f"[W]: {self.engine_path}: Could not find '{name}' in shape dict {shape_dict}.  Using shape {shape} inferred from the engine."
                )
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            dtype = trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            tensor = torch.empty(tuple(shape), dtype=dtype).to(device=device)
            self.tensors[name] = tensor

    def deallocate_buffers(self):
        if not self.engine:
            return
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            del self.tensors[binding]

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                _CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                _CUASSERT(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError(f"ERROR: inference of {self.engine_path} failed.")
                # capture cuda graph
                _CUASSERT(
                    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream)
                self.graph = _CUASSERT(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = _CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference of {self.engine_path} failed.")

        return self.tensors
