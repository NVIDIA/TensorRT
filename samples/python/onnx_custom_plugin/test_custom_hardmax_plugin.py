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

import numpy as np
import os
import sys
import tensorrt as trt

# ../common.py
parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(1, parent_dir)
import common

from load_plugin_lib import load_plugin_lib

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def hardmax_reference_impl(arr, axis):
    one_hot = np.zeros(arr.shape, dtype=arr.dtype)
    argmax = np.expand_dims(np.argmax(arr, axis), axis)
    np.put_along_axis(one_hot,argmax,1,axis=axis)
    return one_hot

def make_trt_network_and_engine(input_shape, axis):
    registry = trt.get_plugin_registry()
    plugin_creator = registry.get_plugin_creator("CustomHardmax", "1")
    axis_buffer = np.array([axis])
    axis_attr = trt.PluginField("axis", axis_buffer, type=trt.PluginFieldType.INT32)
    field_collection = trt.PluginFieldCollection([axis_attr])
    plugin = plugin_creator.create_plugin(name="CustomHardmax", field_collection=field_collection)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=input_shape)
    hardmax = network.add_plugin_v2(inputs=[input_layer], plugin=plugin)
    network.mark_output(hardmax.get_output(0))

    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)

    return engine

def custom_plugin_impl(input_arr, engine):
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()
    inputs[0].host = input_arr.astype(trt.nptype(trt.float32))
    trt_outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )
    output = trt_outputs[0].copy()
    common.free_buffers(inputs, outputs, stream)
    return output

def main():
    load_plugin_lib()
    for num_dims in range(1, 8):
        for axis in range(-num_dims, num_dims):
            shape = np.random.randint(1, 4, size=num_dims)
            arr = np.random.rand(*shape)
            arr = (arr - 0.5) * 200
            engine = make_trt_network_and_engine(shape, axis)
            res1 = hardmax_reference_impl(arr, axis)
            res2 = custom_plugin_impl(arr, engine).reshape(res1.shape)
            assert np.all(res1 == res2), f"Test failed for shape={shape}, axis={axis}"
    print("Passed")

if __name__ == '__main__':
    main()
