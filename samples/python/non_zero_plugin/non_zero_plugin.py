#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import onnx
import os
import sys

import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
    create_network,
    engine_from_network,
)

import argparse

from polygraphy import mod

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from plugin_utils import cuda_call, KernelHelper, UnownedMemory, volume


cuda = mod.lazy_import("cuda.bindings.driver")
cudart = mod.lazy_import("cuda.bindings.runtime")
nvrtc = mod.lazy_import("cuda.bindings.nvrtc")

torch = mod.lazy_import("torch")
cp = mod.lazy_import("cupy")

non_zero_half_kernel = r'''
#include <cuda_fp16.h>
extern "C" __global__
void find_non_zero_indices_half(
    half const* X, int* indices, int* count, int R, int C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the row index is within bounds
    if (row < R)
    {

        for (int col = 0; col < C; ++col)
        {
            half const z = static_cast<half>(0.F);
            if (X[col + C * row] != z)
            {
                int index = atomicAdd(count, 1); // Increment count atomically and get the previous value
                indices[2 * index] = row;
                indices[2 * index + 1] = col;
            }
        }
    }
}
'''

non_zero_float_kernel = r'''
extern "C" __global__
void find_non_zero_indices_float(
    float const* X, int* indices, int* count, int R, int C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the row index is within bounds
    if (row < R)
    {

        for (int col = 0; col < C; ++col)
        {
            if (X[col + C * row] != 0.F)
            {
                int index = atomicAdd(count, 1); // Increment count atomically and get the previous value
                indices[2 * index] = row;
                indices[2 * index + 1] = col;
            }
        }
    }
}
'''

class NonZeroPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self, backend = None):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.num_outputs = 2
        self.plugin_namespace = ""
        self.plugin_name = "NonZeroPlugin"
        self.plugin_version = "1"

        if backend is not None:
            self.backend = backend.tobytes().decode("utf-8")
        else:
            self.backend = "cuda_python"

        self.cuDevice = None

    def get_capability_interface(self, type):
        return self

    def get_output_data_types(self, input_types):
        return [trt.DataType.INT32, trt.DataType.INT32]

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        # First output is 2-D
        # Second output is a size tensor, which must be declared a scalar (0-D)
        output_dims = [trt.DimsExprs(2), trt.DimsExprs(0)]

        upper_bound = exprBuilder.operation(trt.DimensionOperation.PROD, inputs[0][0], inputs[0][1])
        opt_value = exprBuilder.operation(trt.DimensionOperation.FLOOR_DIV, upper_bound, exprBuilder.constant(2))
        num_non_zero_size_tensor = exprBuilder.declare_size_tensor(1, opt_value, upper_bound)

        output_dims[0][0] = num_non_zero_size_tensor
        output_dims[0][1] = exprBuilder.constant(2)

        return output_dims

    def get_fields_to_serialize(self):
        return trt.PluginFieldCollection(
            [
                trt.PluginField(
                    "backend", self.backend.encode(), trt.PluginFieldType.CHAR
                )
            ]
        )

    def configure_plugin(self, inp, out):
        if self.backend == "cuda_python":
            self.cuDevice = cuda_call(cuda.cuDeviceGet(0))

    def on_shape_change(self, inp, out):
        if self.backend == "cuda_python":
            self.cuDevice = cuda_call(cuda.cuDeviceGet(0))

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 1
        assert pos < len(in_out)

        type_ok = False

        # first input should be float16 or float32
        if pos == 0:
            type_ok = in_out[0].desc.type == trt.DataType.FLOAT or in_out[0].desc.type == trt.DataType.HALF
        elif pos == 1:
            type_ok = in_out[1].desc.type == trt.DataType.INT32
        else: # pos == 2
            # size tensor outputs must be NCHW INT32
            type_ok = in_out[2].desc.type == trt.DataType.INT32

        return in_out[pos].desc.format == trt.TensorFormat.LINEAR and type_ok

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        inp_dtype = trt.nptype(input_desc[0].type)

        if self.backend == "cuda_python":
            R = input_desc[0].dims[0]
            C = input_desc[0].dims[1]

            blockSize = 256
            numBlocks = int((C + blockSize - 1) // blockSize)

            d_in = np.array([inputs[0]], dtype=np.uint64)
            d_out_0 = np.array([outputs[0]], dtype=np.uint64)
            d_out_1 = np.array([outputs[1]], dtype=np.uint64)

            args = [d_in, d_out_0, d_out_1, np.array(R, dtype=np.uint32), np.array(C, dtype=np.uint32)]
            kernelArgs = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

            stream_ptr = np.array([stream], dtype=np.uint64)

            if inp_dtype == np.float32:
                kernelHelper = KernelHelper(non_zero_float_kernel, int(self.cuDevice))
                _non_zero_float_kernel = kernelHelper.getFunction(b'find_non_zero_indices_float')
                cuda_call(cuda.cuLaunchKernel(_non_zero_float_kernel,
                                            numBlocks, 1, 1,
                                            blockSize, 1, 1,
                                            0,
                                            stream_ptr,
                                            kernelArgs, 0))
            elif inp_dtype == np.float16:
                kernelHelper = KernelHelper(non_zero_half_kernel, int(self.cuDevice))
                _non_zero_half_kernel = kernelHelper.getFunction(b'find_non_zero_indices_half')
                cuda_call(cuda.cuLaunchKernel(_non_zero_half_kernel,
                                            numBlocks, 1, 1,
                                            blockSize, 1, 1,
                                            0,
                                            stream_ptr,
                                            kernelArgs, 0))
            else:
                raise ValueError("inp_dtype not valid")

        elif self.backend == "torch":
            inp_mem = UnownedMemory(inputs[0], input_desc[0].dims, inp_dtype)

            out_mem = UnownedMemory(
                outputs[0], 2 * volume(input_desc[0].dims), np.int32
            )

            out_1_mem = UnownedMemory(outputs[1], 1, np.int32)

            a_t = torch.as_tensor(inp_mem.d, device="cuda")
            out = torch.nonzero(a_t)

            out_mem.d[: volume(out.shape)] = cp.reshape(cp.asarray(out), (-1,))
            cp.copyto(out_1_mem.d, cp.reshape(cp.asarray([out.shape[0]]), (-1,)))

        else:
            raise ValueError(f"backend not valid: {self.backend}")

    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        cloned_plugin = NonZeroPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    #
    # The following defaults take effect since the respective methods are not overriden
    #

    # def get_valid_tactics(self):
    #     return []

    # def get_workspace_size(self, input_desc, output_desc):
    #     return 0

    # def destroy(self):
    #     pass


class NonZeroPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "NonZeroPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [trt.PluginField("backend", np.array([]), trt.PluginFieldType.CHAR)]
        )

    def create_plugin(self, name, fc, phase):
        backend = None
        for f in fc:
            if f.name == "backend":
                backend = f.data[:-1] if f.data[-1] == 0 else f.data
        return NonZeroPlugin(backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--backend", type=str, default="torch", choices=["cuda_python", "torch"])
    parser.add_argument('--net_type', type=str, default="onnx", choices=["onnx", "inetdef"])

    args = parser.parse_args()

    if args.backend == "cuda_python":
        # Initialize CUDA and create default context
        cuda_call(cudart.cudaFree(0))

    elif args.backend == "torch":
        # Initialize CUDA and create default context
        torch.cuda.init()

    precision = np.float32 if args.precision == "fp32" else np.float16

    inp_shape = (128, 128)
    X = np.random.normal(size=inp_shape).astype(precision)
    # Zero out a random set of indices
    indices = np.random.choice(np.prod(inp_shape), replace=False, size=np.random.randint(0, np.prod(inp_shape) + 1))
    X[np.unravel_index(indices, inp_shape)] = 0

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = NonZeroPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    if args.net_type == "onnx":
        # create ONNX model
        onnx_path = "test_NonZeroPlugin.onnx"
        inputX = gs.Variable(name="X", shape=inp_shape, dtype=precision)
        Y = gs.Variable(name="Y", dtype=np.int32)
        Y_num = gs.Variable(name="Y_num", dtype=np.int32)
        nonZeroPluginNode = gs.Node(
            name="NonZeroPlugin",
            op="NonZeroPlugin",
            inputs=[inputX],
            outputs=[Y, Y_num],
            attrs={"backend": args.backend.encode()},
        )
        graph = gs.Graph(nodes=[nonZeroPluginNode], inputs=[inputX], outputs=[Y], opset=16)
        onnx.save(gs.export_onnx(graph), onnx_path)

        # build engine
        build_engine = EngineFromNetwork(
            NetworkFromOnnxPath(onnx_path), CreateConfig(fp16=precision==np.float16)
        )
    else:
        # Create plugin object
        builder, network = create_network()
        plg_creator = plg_registry.get_creator("NonZeroPlugin", "1", "")
        plugin_fields_list = [
            trt.PluginField("backend", args.backend.encode(), trt.PluginFieldType.CHAR)
        ]
        pfc = trt.PluginFieldCollection(plugin_fields_list)
        plugin = plg_creator.create_plugin("NonZeroPlugin", pfc, trt.TensorRTPhase.BUILD)

        # Populate network
        inputX = network.add_input(name="X", dtype=trt.float32 if precision==np.float32 else trt.float16, shape=inp_shape)
        out = network.add_plugin_v3([inputX], [], plugin)
        out.get_output(0).name = "Y"
        network.mark_output(tensor=out.get_output(0))
        build_engine = engine_from_network((builder, network), CreateConfig(fp16=precision==trt.float16))

    # Compare against Numpy's nonzero
    Y_ref = np.transpose(np.nonzero(X))

    # Run
    with TrtRunner(build_engine, "trt_runner")as runner:
        outputs = runner.infer({"X": X})
        Y = outputs["Y"]
        Y = Y[np.lexsort(np.fliplr(Y).T)]

        if np.allclose(Y, Y_ref):
            print("Inference result correct!")
        else:
            print("Inference result incorrect!")


