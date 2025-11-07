#!/usr/bin/env python3
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

import os
import sys

import numpy as np
import argparse
import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from cuda.bindings import runtime as cudart

TRT_LOGGER = trt.Logger()


def get_plan(onnx_file_path, engine_file_path, version_compatible):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_plan():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        import tensorrt as trt

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, TRT_LOGGER)

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

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.REFIT)
        if version_compatible:
            config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)

        for opt in [6, 10]:
            profile = builder.create_optimization_profile()

            input0_min = (1, 1)
            input0_opt = (opt, 1)
            input0_max = (15, 1)
            profile.set_shape(
                network.get_input(0).name,
                min=input0_min,
                opt=input0_opt,
                max=input0_max,
            )

            input1_min = (1, 1, 1, 16)
            input1_opt = (opt, 1, 1, 16)
            input1_max = (15, 1, 1, 16)
            profile.set_shape(
                network.get_input(1).name,
                min=input1_min,
                opt=input1_opt,
                max=input1_max,
            )

            input2_min = (1, 1)
            input2_opt = (opt, 1)
            input2_max = (15, 1)
            profile.set_shape(
                network.get_input(2).name,
                min=input2_min,
                opt=input2_opt,
                max=input2_max,
            )

            input3_min = (1, 1, 1, 16)
            input3_opt = (opt, 1, 1, 16)
            input3_max = (15, 1, 1, 16)
            profile.set_shape(
                network.get_input(3).name,
                min=input3_min,
                opt=input3_opt,
                max=input3_max,
            )

            config.add_optimization_profile(profile)

        print(
            "Building an engine from file {}; this may take a while...".format(
                onnx_file_path
            )
        )
        plan = builder.build_serialized_network(network, config)
        print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return plan

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}...".format(engine_file_path))
        f = open(engine_file_path, "rb")
        return f.read()
    return build_plan()


def main():
    global trt
    global TRT_LOGGER

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--weights-location",
        dest="weights_location",
        default="GPU",
        choices=["GPU", "CPU"],
        help="The location for weights passed to refitter, either GPU/CPU, default: GPU",
    )
    parser.add_argument(
        "--version-compatible",
        dest="version_compatible",
        action="store_true",
        help="Build a version compatible engine for refitting",
    )
    args = parser.parse_args()

    onnx_file_path = "bidaf-modified.onnx"
    engine_file_path = "bidaf{}.trt".format("-vc" if args.version_compatible else "")

    plan = get_plan(onnx_file_path, engine_file_path, args.version_compatible)

    if args.version_compatible:
        # Try using dispatch runtime for refitting and inference. If failed, fallback to full runtime.
        try:
            del sys.modules["tensorrt"]
            sys.modules["tensorrt"] = __import__("tensorrt_dispatch")
            sys.modules["trt"] = sys.modules["tensorrt"]
            import tensorrt_dispatch as trt

            print(
                "Importing tensorrt_dispatch instead of full tensorrt for refitting and running vc engines."
            )
        except:
            print(
                "Failed to import tensorrt_dispatch for refitting and running vc engines. Please install the package first!"
            )
            sys.modules["tensorrt"] = __import__("tensorrt")
        TRT_LOGGER = trt.Logger()

    engine = None
    with open(engine_file_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        if args.version_compatible:
            runtime.engine_host_code_allowed = True
        engine = runtime.deserialize_cuda_engine(plan)

    # should be after get_engine
    from data_processing import get_inputs, preprocess
    import common_runtime as common

    # input
    context = "A quick brown fox jumps over the lazy dog."
    query = "What color is the fox?"
    cw_str, _ = preprocess(context)
    # get ravelled data
    cw, cc, qw, qc = get_inputs(context, query)

    # Do inference with TensorRT
    weights_names = ["Parameter576_B_0", "W_0"]
    refit_weights_dict = {
        name: np.load("{}.npy".format(name)) for name in weights_names
    }
    fake_weights_dict = {
        name: np.ones_like(weights) for name, weights in refit_weights_dict.items()
    }
    device_mem_dict = {}
    if args.weights_location == "GPU":
        for name, weights in refit_weights_dict.items():
            nbytes = weights.size * weights.itemsize
            device_mem_dict[name] = common.DeviceMem(nbytes)

    execution_context = engine.create_execution_context()
    refitter = trt.Refitter(engine, TRT_LOGGER)
    # Skip weights validation since we are confident that the new weights are similar to the weights used to build engine.
    refitter.weights_validation = False
    # To get a list of all refittable weights' names
    # in the network, use refitter.get_all_weights().

    if args.weights_location == "GPU":
        for name, device_mem in device_mem_dict.items():
            device_weights = trt.Weights(
                trt.DataType.FLOAT, device_mem.device_ptr, refit_weights_dict[name].size
            )
            weights_prototype = refitter.get_weights_prototype(name)
            assert device_weights.dtype == weights_prototype.dtype
            assert device_weights.size == weights_prototype.size
            refitter.set_named_weights(name, device_weights, trt.TensorLocation.DEVICE)

    for weights_dict, answer_correct in [
        (fake_weights_dict, False),
        (refit_weights_dict, True),
    ]:
        import time

        T1 = time.perf_counter()
        device_mem_list = []
        # Refit named weights via set_named_weights
        for name in weights_names:
            host_weights = weights_dict[name]
            if args.weights_location == "CPU":
                weights = host_weights
                location = trt.TensorLocation.HOST
                refitter.set_named_weights(name, weights, location)
            else:
                common.memcpy_host_to_device(device_mem_dict[name].device_ptr, host_weights)

        # Get missing weights names. This should return empty lists in this case.
        missing_weights = refitter.get_missing_weights()
        assert (
            len(missing_weights) == 0
        ), "Refitter found missing weights. Call set_named_weights() or set_weights() for all missing weights"

        print(f"Refitting engine from {args.weights_location} weights...")
        # Refit the engine with the new weights. This will return True if the refit operation succeeded.
        assert refitter.refit_cuda_engine()

        T2 = time.perf_counter()
        print("Engine refitted in {:.2f} ms.".format((T2 - T1) * 1000))

        for profile_idx in range(engine.num_optimization_profiles):
            print("Doing inference...")
            # Do inference
            inputs, outputs, bindings = common.allocate_buffers(
                engine, profile_idx
            )
            padding_bindings = [0] * (len(bindings) * profile_idx)
            new_bindings = padding_bindings + bindings

            # Use context manager for proper stream lifecycle management
            with common.CudaStreamContext() as stream:
                # Set host input. The common.do_inference function will copy the input to the GPU before executing.
                inputs[0].host = cw
                inputs[1].host = cc
                inputs[2].host = qw
                inputs[3].host = qc
                execution_context.set_optimization_profile_async(profile_idx, stream.stream)
                execution_context.set_input_shape("CategoryMapper_4", (10, 1))
                execution_context.set_input_shape("CategoryMapper_5", (10, 1, 1, 16))
                execution_context.set_input_shape("CategoryMapper_6", (6, 1))
                execution_context.set_input_shape("CategoryMapper_7", (6, 1, 1, 16))

                trt_outputs = common.do_inference(
                    execution_context,
                    engine=engine,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                )

            start = trt_outputs[0].item()
            end = trt_outputs[1].item()
            answer = [w.encode() for w in cw_str[start : end + 1].reshape(-1)]
            assert answer_correct == (answer == [b"brown"]), answer
            common.free_buffers(inputs, outputs)

    for _, device_mem in device_mem_dict.items():
        device_mem.free()

    print("Passed")


if __name__ == "__main__":
    main()
