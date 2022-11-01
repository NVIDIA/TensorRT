#!/usr/bin/env python3
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

import numpy as np

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

import tensorrt as trt
from data_processing import get_inputs, preprocess

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        builder = trt.Builder(TRT_LOGGER)
        # Set max threads that can be used by builder.
        builder.max_threads = 10
        network = builder.create_network(common.EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        runtime = trt.Runtime(TRT_LOGGER)
        # Set max threads that can be used by runtime.
        runtime.max_threads = 10

        # Parse model file
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("Completed parsing of ONNX file")

        # Print input info
        print("Network inputs:")
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        network.get_input(0).shape = [10, 1]
        network.get_input(1).shape = [10, 1, 1, 16]
        network.get_input(2).shape = [6, 1]
        network.get_input(3).shape = [6, 1, 1, 16]

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.REFIT)
        config.max_workspace_size = 1 << 28  # 256MiB

        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            runtime.max_threads = 10
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    onnx_file_path = "bidaf-modified.onnx"
    engine_file_path = "bidaf.trt"

    # input
    context = "A quick brown fox jumps over the lazy dog."
    query = "What color is the fox?"
    cw_str, _ = preprocess(context)
    # get ravelled data
    cw, cc, qw, qc = get_inputs(context, query)

    # Do inference with TensorRT
    weights_names = ["Parameter576_B_0", "W_0"]
    refit_weights_dict = {name: np.load("{}.npy".format(name)) for name in weights_names}
    fake_weights_dict = {name: np.ones_like(weights) for name, weights in refit_weights_dict.items()}
    engine = get_engine(onnx_file_path, engine_file_path)
    refitter = trt.Refitter(engine, TRT_LOGGER)
    # Set max threads that can be used by refitter.
    refitter.max_threads = 10

    for weights_dict, answer_correct in [(fake_weights_dict, False), (refit_weights_dict, True)]:
        print("Refitting engine...")
        # To get a list of all refittable weights' names
        # in the network, use refitter.get_all_weights().

        # Refit named weights via set_named_weights
        for name in weights_names:
            refitter.set_named_weights(name, weights_dict[name])

        # Get missing weights names. This should return empty
        # lists in this case.
        missing_weights = refitter.get_missing_weights()
        assert (
            len(missing_weights) == 0
        ), "Refitter found missing weights. Call set_named_weights() or set_weights() for all missing weights"
        # Refit the engine with the new weights. This will return True if
        # the refit operation succeeded.
        assert refitter.refit_cuda_engine()

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        print("Doing inference...")
        # Do inference
        # Set host input. The common.do_inference_v2 function will copy the input to the GPU before executing.
        inputs[0].host = cw
        inputs[1].host = cc
        inputs[2].host = qw
        inputs[3].host = qc
        execution_context = engine.create_execution_context()
        trt_outputs = common.do_inference_v2(
            execution_context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        start = trt_outputs[0].item()
        end = trt_outputs[1].item()
        answer = [w.encode() for w in cw_str[start : end + 1].reshape(-1)]
        assert answer_correct == (answer == [b"brown"])
    print("Passed")


if __name__ == "__main__":
    main()
