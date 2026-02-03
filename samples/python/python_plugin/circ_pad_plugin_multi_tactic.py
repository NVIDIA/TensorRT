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
import onnx
import cupy as cp
import logging
import sys
import os

import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)

import triton
import triton.language as tl

from enum import IntEnum

from polygraphy.json import to_json, from_json
import torch

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from plugin_utils import volume, parseArgs

import argparse

logger = logging.getLogger("CircPadMultiTactic")

class Tactic(IntEnum):
    TORCH = 1
    TRITON = 2

@triton.jit
def circ_pad(X,
            all_pads_0, all_pads_2, all_pads_4, all_pads_6,
            orig_dims_0, orig_dims_1, orig_dims_2, orig_dims_3,
            Y,
            Y_shape_1, Y_shape_2, Y_shape_3,
            X_len, Y_len, BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_y = i < Y_len

    i3 = i % Y_shape_3
    i2 = (i // Y_shape_3) % Y_shape_2
    i1 = (i // Y_shape_3 // Y_shape_2) % Y_shape_1
    i0 = i // Y_shape_3 // Y_shape_2 // Y_shape_1

    j0 = (i0 - all_pads_0 + orig_dims_0) % orig_dims_0
    j1 = (i1 - all_pads_2 + orig_dims_1) % orig_dims_1
    j2 = (i2 - all_pads_4 + orig_dims_2) % orig_dims_2
    j3 = (i3 - all_pads_6 + orig_dims_3) % orig_dims_3

    load_idx = orig_dims_3 * orig_dims_2 * orig_dims_1 * j0 + orig_dims_3 * orig_dims_2 * j1 + orig_dims_3 * j2 + j3
    mask_x = load_idx < X_len

    x = tl.load(X + load_idx, mask=mask_x)

    tl.store(Y + i, x, mask=mask_y)

class CircPadPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self, fc=None, phase=None):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.pads = []
        self.X_shape = []

        self.per_format_tactics = (
            False  # whether per-format tactics or global tactics should be used
        )
        self.curr_type = None  # format being timed currently by TRT auto-tuner

        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_name = "CircPadPlugin"
        self.plugin_version = "1"

        # Set the timing cache ID to prevent unnecessary timing of second plugin instance
        self.timing_cache_id = ""
        
        self.tactic = None

        if fc is not None:
            for f in fc:
                if f.name == "pads":
                    self.pads = f.data
                elif f.name == "per_format_tactics":
                    self.per_format_tactics = int(f.data)

        if phase is not None:
            self.phase = phase

    def get_capability_interface(self, type):
        return self

    def get_output_data_types(self, input_types):
        return [input_types[0]]

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        output_dims = trt.DimsExprs(inputs[0])

        for i in range(np.size(self.pads) // 2):
            output_dims[len(output_dims) - i - 1] = exprBuilder.operation(
                trt.DimensionOperation.SUM,
                inputs[0][len(output_dims) - i - 1],
                exprBuilder.constant(self.pads[i * 2] + self.pads[i * 2 + 1]),
            )

        return [output_dims]

    def get_fields_to_serialize(self):
        return trt.PluginFieldCollection([
            trt.PluginField("pads", self.pads, trt.PluginFieldType.INT32),
            trt.PluginField(
                "per_format_tactics",
                np.array([self.per_format_tactics], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
        ])

    def configure_plugin(self, inp, out):
        assert inp[0].desc.type == trt.float32 or inp[0].desc.type == trt.float16
        self.curr_type = inp[0].desc.type

    def on_shape_change(self, inp, out):
        if (
            self.phase == trt.TensorRTPhase.RUNTIME
            and self.per_format_tactics
            and inp[0].type == trt.float16
        ):
            assert self.tactic == Tactic.TRITON

        X_dims = inp[0].dims
        self.X_shape = np.zeros((len(X_dims),))
        for i in range(len(X_dims)):
            self.X_shape[i] = X_dims[i]

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 1
        assert pos < len(in_out)

        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be float16 or float32
        if pos == 0:
            return desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF

        # output should have the same type as the input
        if pos == 1:
            return in_out[0].desc.type == desc.type

        assert False

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        inp_dtype = trt.nptype(input_desc[0].type)

        a_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(inp_dtype).itemsize, self
        )
        c_mem = cp.cuda.UnownedMemory(
            outputs[0],
            volume(output_desc[0].dims) * cp.dtype(inp_dtype).itemsize,
            self,
        )

        a_ptr = cp.cuda.MemoryPointer(a_mem, 0)
        c_ptr = cp.cuda.MemoryPointer(c_mem, 0)

        c_d = cp.ndarray((volume(output_desc[0].dims)), dtype=inp_dtype, memptr=c_ptr)
        
        if self.phase == trt.TensorRTPhase.BUILD:
            logger.info(f"Timing tactic: {self.tactic}")

        if self.tactic == Tactic.TORCH:
            # Use PyTorch functional op - no need to write kernel
            a_d = cp.ndarray(tuple(input_desc[0].dims), dtype=inp_dtype, memptr=a_ptr)
            a_t = torch.as_tensor(a_d, device='cuda')
            out = torch.nn.functional.pad(a_t, self.pads.tolist(), mode='circular')
            cp.copyto(c_d, cp.reshape(cp.asarray(out), (-1,)))
        elif self.tactic == Tactic.TRITON:
            a_d = cp.ndarray((volume(input_desc[0].dims)), dtype=inp_dtype, memptr=a_ptr)
            a_t = torch.as_tensor(a_d, device='cuda')
            c_t = torch.as_tensor(c_d, device='cuda')

            N = len(self.X_shape)
            all_pads = np.zeros((N * 2,), dtype=np.int32)
            orig_dims = np.array(self.X_shape, dtype=np.int32)
            out_dims = np.array(self.X_shape, dtype=np.int32)

            for i in range(np.size(pads) // 2):
                out_dims[N - i - 1] += pads[i * 2] + pads[i * 2 + 1]
                all_pads[N * 2 - 2 * i - 2] = pads[i * 2]
                all_pads[N * 2 - 2 * i - 1] = pads[i * 2 + 1]

            all_pads = all_pads.tolist()
            orig_dims = orig_dims.tolist()
            out_dims = out_dims.tolist()

            blockSize = 256
            numBlocks = tuple([int((np.prod(out_dims) + blockSize - 1) // blockSize)])

            circ_pad[numBlocks](a_t,
                all_pads[0], all_pads[2], all_pads[4], all_pads[6],
                orig_dims[0], orig_dims[1], orig_dims[2], orig_dims[3],
                c_t,
                out_dims[1], out_dims[2], out_dims[3],
                int(np.prod(orig_dims)), int(np.prod(out_dims)), BLOCK_SIZE=256
            )
        else:
            raise RuntimeError("Invalid tactic")
    
    def attach_to_context(self, context):
        return self.clone()
    
    def get_valid_tactics(self):
        assert self.curr_type is not None
        if self.per_format_tactics and self.curr_type == trt.float16:
            return [int(Tactic.TRITON)]

        return [int(Tactic.TORCH), int(Tactic.TRITON)]

    def set_tactic(self, tactic):
        self.tactic = Tactic(tactic)

        if self.phase == trt.TensorRTPhase.RUNTIME:
            logger.info(f"Best tactic chosen: {self.tactic}")

    def clone(self):
        cloned_plugin = CircPadPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    # 
    # The following defaults take effect since the respective methods are not overriden
    #

    # def get_workspace_size(self, input_desc, output_desc):
    #     return 0
    
    # def destroy(self):
    #     pass


class CircPadPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "CircPadPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([
            trt.PluginField("pads", np.array([]), trt.PluginFieldType.INT32),
            trt.PluginField(
                "per_format_tactics", np.array([]), trt.PluginFieldType.INT32
            ),
        ])

    def create_plugin(self, name, fc, phase):
        return CircPadPlugin(fc, phase)


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Options for Circular Padding plugin multi-tactic sample"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision to use for plugin",
    )
    parser.add_argument(
        "--per-format-tactics",
        action="store_true",
        help="Whether per-format tactics or global tactics should be used",
    )

    args = parser.parse_args()

    precision = np.float32 if args.precision == "fp32" else np.float16
    is_tactics_per_format = 1 if args.per_format_tactics else 0

    inp_shape = (10, 3, 32, 32)
    X_A = np.random.normal(size=inp_shape).astype(precision)
    X_B = np.random.normal(size=inp_shape).astype(precision)

    pads = (1, 1, 1, 1)

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = CircPadPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # create ONNX model
    onnx_path = f"test_CircPadPlugin_multi_tactic_{args.precision}.onnx"
    inputA = gs.Variable(name="X_A", shape=inp_shape, dtype=precision)
    inputB = gs.Variable(name="X_B", shape=inp_shape, dtype=precision)
    Y_A = gs.Variable(name="Y_A", dtype=precision)
    Y_B = gs.Variable(name="Y_B", dtype=precision)
    myPluginNode_A = gs.Node(
        name="CircPadPlugin_A",
        op="CircPadPlugin",
        inputs=[inputA],
        outputs=[Y_A],
        attrs={
            "pads": pads,
            "per_format_tactics": np.array([is_tactics_per_format], dtype=np.int32),
        },
    )
    myPluginNode_B = gs.Node(
        name="CircPadPlugin_B",
        op="CircPadPlugin",
        inputs=[inputB],
        outputs=[Y_B],
        attrs={
            "pads": pads,
            "per_format_tactics": np.array([is_tactics_per_format], dtype=np.int32),
        },
    )

    graph = gs.Graph(nodes=[myPluginNode_A, myPluginNode_B], inputs=[inputA, inputB], outputs=[Y_A, Y_B], opset=16)
    onnx.save(gs.export_onnx(graph), onnx_path)

    # build engine
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(onnx_path, strongly_typed=True), CreateConfig()
    )

    Y_A_ref = np.pad(X_A, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")
    Y_B_ref = np.pad(X_B, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")

    # Run
    with TrtRunner(build_engine, "trt_runner")as runner:
        outputs = runner.infer({"X_A": X_A, "X_B": X_B})
        Y_A_out = outputs["Y_A"]
        Y_B_out = outputs["Y_B"]

        if np.allclose(Y_A_out, Y_A_ref):
            print("Inference result A correct!")
        else:
            print("Inference result A incorrect!")

        if np.allclose(Y_B_out, Y_B_ref):
            print("Inference result B correct!")
        else:
            print("Inference result B incorrect!")
