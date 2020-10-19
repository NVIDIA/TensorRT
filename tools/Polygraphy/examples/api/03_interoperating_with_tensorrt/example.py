#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script demonstrates how to use Polygraphy in conjunction with APIs
provided by a backend. Specifically, in this case, we use TensorRT APIs
to print the network name and enable FP16 mode.
"""
from polygraphy.backend.trt import NetworkFromOnnxPath, CreateConfig, EngineFromNetwork, TrtRunner
from polygraphy.common.func import extend

import tensorrt as trt
import numpy as np
import os


MODEL = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "models", "identity.onnx")

# We can use the `extend` decorator to easily extend loaders provided by Polygraphy
# The parameters our decorated function takes should match the return values of the loader we are extending.

# For `NetworkFromOnnxPath`, we can see from the API documentation that it returns a TensorRT
# builder, network and parser. That is what our function will receive.
@extend(NetworkFromOnnxPath(MODEL))
def load_network(builder, network, parser):
    # Here we can modify the network. For this example, we'll just set the network name.
    network.name = "MyIdentity"
    print("Network name: {:}".format(network.name))


# In case a builder configuration option is missing from Polygraphy, we can easily set it using TensorRT APIs.
# Our function will receive a TensorRT builder config since that's what `CreateConfig` returns.
@extend(CreateConfig())
def load_config(config):
    # Polygraphy supports the fp16 flag, but in case it didn't, we could do this:
    config.set_flag(trt.BuilderFlag.FP16)


# Since we have no further need of TensorRT APIs, we can come back to regular Polygraphy.
build_engine = EngineFromNetwork(load_network, config=load_config)

with TrtRunner(build_engine) as runner:
    runner.infer({"x": np.ones(shape=(1, 1, 2, 2), dtype=np.float32)})
