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

import argparse
import onnx_graphsurgeon as gs
import numpy as np
import onnx
import ctypes

import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Options for Circular Padding plugin C++ example"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision to use for plugin",
    )
    parser.add_argument(
        "--plugin-lib",
        type=str,
        help="Path to the Circular Padding plugin lib",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()

    handle = ctypes.CDLL(args.plugin_lib)
    if not handle:
        raise RuntimeError("Could not load Circular Padding plugin library")

    precision = np.float32 if args.precision == "fp32" else np.float16
    inp_shape = (10, 3, 32, 32)
    X = np.random.normal(size=inp_shape).astype(precision)

    pads = (1, 1, 1, 1)

    # create ONNX model
    onnx_path = f"test_CircPadPlugin_cpp_{args.precision}.onnx"
    inputA = gs.Variable(name="X", shape=inp_shape, dtype=precision)
    Y = gs.Variable(name="Y", dtype=precision)
    myPluginNode = gs.Node(
        name="CircPadPlugin",
        op="CircPadPlugin",
        inputs=[inputA],
        outputs=[Y],
        attrs={"pads": pads},
    )
    graph = gs.Graph(nodes=[myPluginNode], inputs=[inputA], outputs=[Y], opset=16)
    onnx.save(gs.export_onnx(graph), onnx_path)

    # build engine
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(onnx_path, strongly_typed=True), CreateConfig()
    )

    Y_ref = np.pad(X, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")
    # Run
    with TrtRunner(build_engine, "trt_runner") as runner:
        outputs = runner.infer({"X": X})
        Y = outputs["Y"]

        if np.allclose(Y, Y_ref):
            print("Inference result correct!")
        else:
            print("Inference result incorrect!")
