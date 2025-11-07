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

import onnx_graphsurgeon as gs
import numpy as np
import sys
import os

import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    TrtRunner,
    create_network,
    engine_from_network,
)

from polygraphy.json import to_json, from_json

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from plugin_utils import cuda_call, KernelHelper, parseArgs, CudaCtxManager
from cuda.bindings import driver as cuda

circ_pad_half_kernel = r"""
#include <cuda_fp16.h>
extern "C" __global__
void circ_pad_half(half const* X, int const* all_pads, int const* orig_dims, half* Y, int const* Y_shape, int Y_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < Y_len; i += stride)
    {
        int i3 = i % Y_shape[3];
        int i2 = (i / Y_shape[3]) % Y_shape[2];
        int i1 = (i / Y_shape[3] / Y_shape[2]) % Y_shape[1];
        int i0 = i / Y_shape[3] / Y_shape[2] / Y_shape[1];

        int j0 = (i0 - all_pads[0] + orig_dims[0]) % orig_dims[0];
        int j1 = (i1 - all_pads[2] + orig_dims[1]) % orig_dims[1];
        int j2 = (i2 - all_pads[4] + orig_dims[2]) % orig_dims[2];
        int j3 = (i3 - all_pads[6] + orig_dims[3]) % orig_dims[3];

        Y[i] = X[
            orig_dims[3] * orig_dims[2] * orig_dims[1] * j0
            + orig_dims[3] * orig_dims[2] * j1
            + orig_dims[3] * j2
            + j3
        ];
    }
}
"""

circ_pad_float_kernel = r"""
extern "C" __global__
void circ_pad_float(float const* X, int const* all_pads, int const* orig_dims, float* Y, int const* Y_shape, int Y_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < Y_len; i += stride)
    {
        int i3 = i % Y_shape[3];
        int i2 = (i / Y_shape[3]) % Y_shape[2];
        int i1 = (i / Y_shape[3] / Y_shape[2]) % Y_shape[1];
        int i0 = i / Y_shape[3] / Y_shape[2] / Y_shape[1];

        int j0 = (i0 - all_pads[0] + orig_dims[0]) % orig_dims[0];
        int j1 = (i1 - all_pads[2] + orig_dims[1]) % orig_dims[1];
        int j2 = (i2 - all_pads[4] + orig_dims[2]) % orig_dims[2];
        int j3 = (i3 - all_pads[6] + orig_dims[3]) % orig_dims[3];

        Y[i] = X[
            orig_dims[3] * orig_dims[2] * orig_dims[1] * j0
            + orig_dims[3] * orig_dims[2] * j1
            + orig_dims[3] * j2
            + j3
        ];
    }
}
"""


class CircPadPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, fc=None):
        trt.IPluginV2DynamicExt.__init__(self)
        self.pads = []
        self.X_shape = []
        self.N = 0

        self.all_pads_d = None
        self.orig_dims_d = None
        self.Y_shape_d = None

        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_type = "CircPadPlugin"
        self.plugin_version = "1"

        self.cuDevice = None

        if fc is not None:
            assert set([f.name for f in fc]) == set(
                ["pads", "N"]
            ), "Field collection invalid"
            for f in fc:
                if f.name == "pads":
                    self.pads = f.data
                elif f.name == "N":
                    self.N = int(f.data)

    def initialize(self):
        self.cuDevice = cuda_call(cuda.cuDeviceGet(0))
        trt.get_plugin_registry().acquire_plugin_resource(
            "cuda_ctx", CudaCtxManager(self.cuDevice)
        )
        self.all_pads_d = cuda_call(
            cuda.cuMemAlloc(np.int32().itemsize * self.N * 2)
        )
        self.orig_dims_d = cuda_call(
            cuda.cuMemAlloc(np.int32().itemsize * self.N)
        )
        self.Y_shape_d = cuda_call(cuda.cuMemAlloc(np.int32().itemsize * self.N))

    def get_output_datatype(self, index, input_types):
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder):

        output_dims = trt.DimsExprs(inputs[0])

        for i in range(np.size(self.pads) // 2):
            output_dims[len(output_dims) - i - 1] = exprBuilder.operation(
                trt.DimensionOperation.SUM,
                inputs[0][len(output_dims) - i - 1],
                exprBuilder.constant(self.pads[i * 2] + self.pads[i * 2 + 1]),
            )

        return output_dims

    def serialize(self):
        return to_json({"pads": self.pads, "N": self.N})

    def configure_plugin(self, inp, out):
        X_dims = inp[0].desc.dims
        self.X_shape = np.zeros((len(X_dims),))
        for i in range(len(X_dims)):
            self.X_shape[i] = X_dims[i]

        all_pads = np.zeros((self.N * 2,), dtype=np.int32)
        orig_dims = np.array(self.X_shape, dtype=np.int32)
        out_dims = np.array(self.X_shape, dtype=np.int32)

        for i in range(np.size(self.pads) // 2):
            out_dims[self.N - i - 1] += self.pads[i * 2] + self.pads[i * 2 + 1]
            all_pads[self.N * 2 - 2 * i - 2] = self.pads[i * 2]
            all_pads[self.N * 2 - 2 * i - 1] = self.pads[i * 2 + 1]

        # Copy vectors from host memory to device memory
        if self.all_pads_d:
            cuda_call(
                cuda.cuMemcpyHtoD(self.all_pads_d, all_pads, all_pads.nbytes)
            )
        if self.orig_dims_d:
            cuda_call(
                cuda.cuMemcpyHtoD(self.orig_dims_d, orig_dims, orig_dims.nbytes)
            )
        if self.Y_shape_d:
            cuda_call(
                cuda.cuMemcpyHtoD(self.Y_shape_d, out_dims, out_dims.nbytes)
            )

        self.Y_len_d = np.prod(out_dims)

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 1
        assert pos < len(in_out)

        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be float16 or float32
        if pos == 0:
            return desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF

        # output should have the same type as the input
        if pos == 1:
            return in_out[0].type == desc.type

        assert False

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        inp_dtype = trt.nptype(input_desc[0].type)

        blockSize = 256
        numBlocks = int((np.prod(np.array(self.X_shape)) + blockSize - 1) // blockSize)

        da = np.array([inputs[0]], dtype=np.uint64)
        dc = np.array([outputs[0]], dtype=np.uint64)

        d_all_pads = np.array([int(self.all_pads_d)], dtype=np.uint64)
        d_orig_dims = np.array([int(self.orig_dims_d)], dtype=np.uint64)
        d_Y_shape = np.array([int(self.Y_shape_d)], dtype=np.uint64)
        Y_len = np.array(self.Y_len_d, dtype=np.uint32)

        args = [da, d_all_pads, d_orig_dims, dc, d_Y_shape, Y_len]
        kernelArgs = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        stream_ptr = np.array([stream], dtype=np.uint64)

        if inp_dtype == np.float32:
            kernelHelper = KernelHelper(circ_pad_float_kernel, int(self.cuDevice))
            _circ_pad_float_kernel = kernelHelper.getFunction(b"circ_pad_float")
            cuda_call(
                cuda.cuLaunchKernel(
                    _circ_pad_float_kernel,
                    numBlocks,
                    1,
                    1,
                    blockSize,
                    1,
                    1,
                    0,
                    stream_ptr,
                    kernelArgs,
                    0,
                )
            )
        elif inp_dtype == np.float16:
            kernelHelper = KernelHelper(circ_pad_half_kernel, int(self.cuDevice))
            _circ_pad_half_kernel = kernelHelper.getFunction(b"circ_pad_half")
            cuda_call(
                cuda.cuLaunchKernel(
                    _circ_pad_half_kernel,
                    numBlocks,
                    1,
                    1,
                    blockSize,
                    1,
                    1,
                    0,
                    stream_ptr,
                    kernelArgs,
                    0,
                )
            )
        else:
            raise ValueError("inp_dtype not valid")

    def clone(self):
        cloned_plugin = CircPadPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def terminate(self):
        if self.all_pads_d:
            cuda_call(cuda.cuMemFree(self.all_pads_d))
        if self.orig_dims_d:
            cuda_call(cuda.cuMemFree(self.orig_dims_d))
        if self.Y_shape_d:
            cuda_call(cuda.cuMemFree(self.Y_shape_d))

        plg_registry.release_plugin_resource("cuda_ctx")

    #
    # The following defaults take effect since the respective methods are not overriden
    #

    # def get_serialization_size(self):
    #     return len(to_json({"pads": self.pads}))

    # def get_workspace_size(self, input_desc, output_desc):
    #     return 0

    # def destroy(self):
    #     pass


class CircPadPluginCreator(trt.IPluginCreator):
    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = "CircPadPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
                trt.PluginField("pads", np.array([]), trt.PluginFieldType.INT32),
                trt.PluginField("N", np.array([]), trt.PluginFieldType.INT32),
            ]
        )

    def create_plugin(self, name, fc):
        return CircPadPlugin(fc)

    def deserialize_plugin(self, name, data):
        deserialized = CircPadPlugin()
        j = dict(from_json(data))
        deserialized.__dict__.update(j)
        return deserialized


if __name__ == "__main__":

    args = parseArgs()
    precision = np.float32 if args.precision == "fp32" else np.float16

    # Initialize CUDA Driver API
    cuda_call(cuda.cuInit(0))

    # Retrieve handle for device 0
    cuDevice = cuda_call(cuda.cuDeviceGet(0))

    plg_registry = trt.get_plugin_registry()

    # Create context
    plg_registry.acquire_plugin_resource("cuda_ctx", CudaCtxManager(cuDevice))

    inp_shape = (100, 2, 32, 32)
    X = np.random.normal(size=inp_shape).astype(precision)

    pads = (1, 1, 1, 1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # Load standard plugins (if needed)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

    # Register plugin creator
    my_plugin_creator = CircPadPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    plg_creator = plg_registry.get_plugin_creator("CircPadPlugin", "1", "")
    plugin_fields_list = [
        trt.PluginField(
            "pads", np.array(pads, dtype=np.int32), trt.PluginFieldType.INT32
        ),
        trt.PluginField("N", np.array([4], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    pfc = trt.PluginFieldCollection(plugin_fields_list)
    plugin = plg_creator.create_plugin("CircPadPlugin", pfc)

    # Populate network
    input_X = network.add_input(
        name="X",
        dtype=trt.float32 if precision == np.float32 else trt.float16,
        shape=X.shape,
    )
    out = network.add_plugin_v2([input_X], plugin)
    out.get_output(0).name = "Y"
    network.mark_output(tensor=out.get_output(0))

    # Build engine
    config = builder.create_builder_config()
    engine = engine_from_network(
        (builder, network), CreateConfig(fp16=precision == trt.float16)
    )

    Y_ref = np.pad(X, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")
    # Run
    with TrtRunner(engine, "trt_runner") as runner:
        outputs = runner.infer({"X": X})
        Y = outputs["Y"]

        if np.allclose(Y, Y_ref):
            print("Inference result correct!")
        else:
            print("Inference result incorrect!")

    plg_registry.release_plugin_resource("cuda_ctx")
