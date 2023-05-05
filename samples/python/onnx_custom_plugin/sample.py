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

import tensorrt as trt

from model import TRT_MODEL_PATH
from load_plugin_lib import load_plugin_lib

# ../common.py
parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(1, parent_dir)
import common

# Reuse some BiDAF-specific methods
# ../engine_refit_onnx_bidaf/data_processing.py
sys.path.insert(1, os.path.join(parent_dir, 'engine_refit_onnx_bidaf'))
from engine_refit_onnx_bidaf.data_processing import preprocess, get_inputs

# Maxmimum number of words in context or query text.
# Used in optimization profile when building engine.
# Adjustable.
MAX_TEXT_LENGTH = 64

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

# Path to which trained model will be saved (check README.md)
ENGINE_FILE_PATH = os.path.join(WORKING_DIR, 'bidaf.trt')

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Builds TensorRT Engine
def build_engine(model_path):

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    runtime = trt.Runtime(TRT_LOGGER)

    # Parse model file
    print("Loading ONNX file from path {}...".format(model_path))
    with open(model_path, "rb") as model:
        print("Beginning ONNX file parsing")
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing of ONNX file")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))

    # The input text length is variable, so we need to specify an optimization profile.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input = network.get_input(i)
        assert input.shape[0] == -1
        min_shape = [1] + list(input.shape[1:])
        opt_shape = [8] + list(input.shape[1:])
        max_shape = [MAX_TEXT_LENGTH] + list(input.shape[1:])
        profile.set_shape(input.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print("Building TensorRT engine. This may take a few minutes.")
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    with open(ENGINE_FILE_PATH, "wb") as f:
        f.write(plan)
    return engine

def load_test_case(inputs, context_text, query_text, trt_context):
    # Part 1: Specify Input shapes
    cw, cc = preprocess(context_text)
    qw, qc = preprocess(query_text)
    for arr in (cw, cc, qw, qc):
        assert arr.shape[0] <= MAX_TEXT_LENGTH, "Input context or query is too long! " + \
                                                "Either decrease the input length or increase MAX_TEXT_LENGTH"
    trt_context.set_input_shape('CategoryMapper_4', cw.shape)
    trt_context.set_input_shape('CategoryMapper_5', cc.shape)
    trt_context.set_input_shape('CategoryMapper_6', qw.shape)
    trt_context.set_input_shape('CategoryMapper_7', qc.shape)

    # Part 2: load input data
    cw_flat, cc_flat, qw_flat, qc_flat = get_inputs(context_text, query_text)
    for i, arr in enumerate([cw_flat, cc_flat, qw_flat, qc_flat]):
        inputs[i].host = arr


def main():
    # Load the shared object file containing the Hardmax plugin implementation.
    # By doing this, you will also register the Hardmax plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/customHardmaxPlugin.cpp for more details.
    load_plugin_lib()

    # Load pretrained model
    if not os.path.isfile(TRT_MODEL_PATH):
        raise IOError(
            "\n{}\n{}\n{}\n".format(
                "Failed to load model file ({}).".format(TRT_MODEL_PATH),
                "Please use 'python3 model.py' to generate the ONNX model.",
                "For more information, see README.md",
            )
        )

    if os.path.exists(ENGINE_FILE_PATH):
        print(f"Loading saved TRT engine from {ENGINE_FILE_PATH}")
        with open(ENGINE_FILE_PATH, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            runtime.max_threads = 10
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        print("Engine plan not saved. Building new engine...")
        engine = build_engine(TRT_MODEL_PATH)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine, profile_idx=0)

    testcases = [
        ('Garry the lion is 5 years old. He lives in the savanna.', 'Where does the lion live?'),
        ('A quick brown fox jumps over the lazy dog.', 'What color is the fox?')
    ]

    print("\n=== Testing ===")

    interactive = '--interactive' in sys.argv
    if interactive:
        context_text = input("Enter context: ")
        query_text = input("Enter query: ")
        testcases = [(context_text, query_text)]

    trt_context = engine.create_execution_context()
    for (context_text, query_text) in testcases:

        context_words, _ = preprocess(context_text)

        load_test_case(inputs, context_text, query_text, trt_context)
        if not interactive:
            print(f"Input context: {context_text}")
            print(f"Input query: {query_text}")
        trt_outputs = common.do_inference_v2(
            trt_context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        start = trt_outputs[1].item()
        end = trt_outputs[0].item()
        answer = context_words[start : end + 1].flatten()
        print(f"Model prediction: ", " ".join(answer))
        print()
    common.free_buffers(inputs, outputs, stream)
    print("Passed")

if __name__ == "__main__":
    main()
