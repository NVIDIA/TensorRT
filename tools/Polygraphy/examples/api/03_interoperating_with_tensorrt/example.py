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

"""
This script demonstrates how to use Polygraphy in conjunction with APIs
provided by a backend. Specifically, in this case, we use TensorRT APIs
to print the network name and enable FP16 mode.
"""
import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner


# TIP: The immediately evaluated functional API makes it very easy to interoperate
# with backends like TensorRT. For details, see example 06 (`examples/api/06_immediate_eval_api`).

# We can use the `extend` decorator to easily extend lazy loaders provided by Polygraphy
# The parameters our decorated function takes should match the return values of the loader we are extending.

# For `NetworkFromOnnxPath`, we can see from the API documentation that it returns a TensorRT
# builder, network and parser. That is what our function will receive.
@func.extend(NetworkFromOnnxPath("identity.onnx"))
def load_network(builder, network, parser):
    # Here we can modify the network. For this example, we'll just set the network name.
    network.name = "MyIdentity"
    print(f"Network name: {network.name}")

    # Notice that we don't need to return anything - `extend()` takes care of that for us!


# In case a builder configuration option is missing from Polygraphy, we can easily set it using TensorRT APIs.
# Our function will receive a TensorRT IBuilderConfig since that's what `CreateConfig` returns.
@func.extend(CreateConfig())
def load_config(config):
    # Polygraphy supports the fp16 flag, but in case it didn't, we could do this:
    config.set_flag(trt.BuilderFlag.FP16)


def main():
    # Since we have no further need of TensorRT APIs, we can come back to regular Polygraphy.
    #
    # NOTE: Since we're using lazy loaders, we provide the functions as arguments - we do *not* call them ourselves.
    build_engine = EngineFromNetwork(load_network, config=load_config)

    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()
