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

# For a single dimension this will return the min, opt, and max size when given
# input of either one or three (comma delimited) values
#   dim="1" or dim=1 returns (1, 1, 1)
#   dim="1,4,5" returns (1, 4, 5)
def parse_dynamic_size(dim):
    split = str(dim).split(',')
    assert len(split) in (1,3) , "Dynamic size input must be either 1 or 3 comma-separated integers"
    ints = [int(i) for i in split]

    if len(ints) == 1:
        ints *= 3

    assert ints[0] <= ints[1] <= ints[2]
    return tuple(ints)


def is_dimension_dynamic(dim):
    return dim is None or dim <= 0


def is_shape_dynamic(shape):
    return any([is_dimension_dynamic(dim) for dim in shape])


def run_trt_engine(context, engine, tensors):

    bindings = [0] * engine.num_io_tensors

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            tensor = tensors['inputs'][tensor_name]
            bindings[i] = tensor.data_ptr()
            if is_shape_dynamic(engine.get_tensor_shape(tensor_name)):
                context.set_input_shape(tensor_name, tensor.shape)
        elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            tensor = tensors['outputs'][tensor_name]
            bindings[i] = tensor.data_ptr()

    context.execute_v2(bindings=bindings)


def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def engine_info(engine_filepath):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_filepath, TRT_LOGGER)

    binding_template = r"""
{btype} {{
  name: "{bname}"
  data_type: {dtype}
  dims: {dims}
}}"""
    type_mapping = {"DataType.HALF": "TYPE_FP16",
                    "DataType.FLOAT": "TYPE_FP32",
                    "DataType.INT32": "TYPE_INT32",
                    "DataType.BOOL" : "TYPE_BOOL"}

    print("engine name", engine.name)
    start_dim = 1
    print("num_optimization_profiles", engine.num_optimization_profiles)
    print("device_memory_size:", engine.device_memory_size)
    print("max_workspace_size:", engine.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE))
    print("num_layers:", engine.num_layers)

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i) 
        btype = "input" if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT else "output"
        dtype = engine.get_tensor_dtype(tensor_name)
        bdims = engine.get_tensor_shape(tensor_name)
        config_values = {
            "btype": btype,
            "bname": tensor_name,
            "dtype": type_mapping[str(dtype)],
            "dims": list(bdims[start_dim:])
        }
        final_binding_str = binding_template.format_map(config_values)
        print(final_binding_str)


def build_engine(model_file, shapes, max_ws=512*1024*1024, fp16=False, timing_cache=None):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_ws)
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for s in shapes:
        profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
    config.add_optimization_profile(profile)

    timing_cache_available = int(trt.__version__[0]) >= 8 and timing_cache != None
    # load global timing cache
    if timing_cache_available:
        if os.path.exists(timing_cache):
            with open(timing_cache, "rb") as f:
                cache = config.create_timing_cache(f.read())
                config.set_timing_cache(cache, ignore_mismatch = False)
        else:
            cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, ignore_mismatch = False)

    network_creation_flag = 0
    if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys():
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_creation_flag)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            for i in range(parser.num_errors):
                print("TensorRT ONNX parser error:", parser.get_error(i))
            engine = builder.build_serialized_network(network, config=config)

            # save global timing cache
            if timing_cache_available:
                cache = config.get_timing_cache()
                with cache.serialize() as buffer:
                    with open(timing_cache, "wb") as f:
                        f.write(buffer)
                        f.flush()
                        os.fsync(f)

            return engine
