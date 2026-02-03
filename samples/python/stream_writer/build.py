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

import tensorrt as trt
import numpy as np
from polygraphy.logger import G_LOGGER
from polygraphy.backend.trt import (
    CreateNetwork,
    CreateConfig,
    engine_bytes_from_network,
    get_trt_logger
)
DEBUG_LOG = False  # Turn on to print TRT verbose logs

if DEBUG_LOG:
    verbose = G_LOGGER.verbosity(G_LOGGER.SUPER_VERBOSE)
    verbose.__enter__()

def build_network():
    builder, network = CreateNetwork()()
    # A simple network with internal tensors
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 3, 224, 224))
    conv1_w = np.random.randn(16, 3, 3, 3).astype(np.float32)
    conv1_b = np.random.randn(16).astype(np.float32)
    conv1 = network.add_convolution_nd(input=input_tensor, num_output_maps=16, kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
    relu1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)
    conv2_w = np.random.randn(32, 16, 3, 3).astype(np.float32)
    conv2_b = np.random.randn(32).astype(np.float32)
    conv2 = network.add_convolution_nd(input=relu1.get_output(0), num_output_maps=32, kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
    relu2 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.RELU)
    network.mark_output(tensor=relu2.get_output(0))
    return builder, network

class StreamWriter(trt.IStreamWriter):
    def __init__(self):
        trt.IStreamWriter.__init__(self)
        self.bytes = bytes()

    def write(self, data):
        self.bytes += data
        return len(data)

def build_engine():
    print("Constructing network...")
    builder, network = build_network()
    config = CreateConfig()(builder, network)
    stream_writer = StreamWriter()
    print("Building engine and serializing to stream...")
    engine_bytes = builder.build_serialized_network_to_stream(network, config, stream_writer)
    print("The total bytes written to stream is: ", len(stream_writer.bytes))
    runtime = trt.Runtime(get_trt_logger())
    print("Deserializing engine from stream...")
    engine = runtime.deserialize_cuda_engine(stream_writer.bytes)
    assert engine is not None, "Engine deserialization failed"
    print("Engine deserialized successfully")

if __name__ == "__main__":
    build_engine()
    if DEBUG_LOG:
        verbose.__exit__(None, None, None)

