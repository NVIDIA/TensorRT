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
import argparse

import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

import numpy as np
import tensorflow as tf
from tf2onnx import tfonnx, optimizer, tf_loader


def main(args):
    # Load saved model
    saved_model_path = os.path.realpath(args.saved_model)
    assert os.path.isdir(saved_model_path)
    graph_def, inputs, outputs = tf_loader.from_saved_model(saved_model_path, None, None, "serve", ["serving_default"])
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name="")
    with tf_loader.tf_session(graph=tf_graph):
        onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=11)
    onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))
    graph = gs.import_onnx(onnx_model)
    assert graph
    print()
    print("ONNX graph created successfully")

    # Set the I/O tensor shapes
    graph.inputs[0].shape[0] = args.batch_size
    graph.outputs[0].shape[0] = args.batch_size
    if args.input_size and args.input_size > 0:
        if graph.inputs[0].shape[3] == 3:
            # Format NHWC
            graph.inputs[0].shape[1] = args.input_size
            graph.inputs[0].shape[2] = args.input_size
        elif graph.inputs[0].shape[1] == 3:
            # Format NCHW
            graph.inputs[0].shape[2] = args.input_size
            graph.inputs[0].shape[3] = args.input_size
    print("ONNX input named '{}' with shape {}".format(graph.inputs[0].name, graph.inputs[0].shape))
    print("ONNX output named '{}' with shape {}".format(graph.outputs[0].name, graph.outputs[0].shape))
    for i in range(4):
        if type(graph.inputs[0].shape[i]) != int or graph.inputs[0].shape[i] <= 0:
            print("The input shape of the graph is invalid, try overriding it by giving a fixed size with --input_size")
            sys.exit(1)

    # Fix Clip Nodes (ReLU6)
    for node in [n for n in graph.nodes if n.op == "Clip"]:
        for input in node.inputs[1:]:
            # In TensorRT, the min/max inputs on a Clip op *must* have fp32 datatype
            input.values = np.float32(input.values)

    # Run tensor shape inference
    graph.cleanup().toposort()
    model = shape_inference.infer_shapes(gs.export_onnx(graph))
    graph = gs.import_onnx(model)

    # Save updated model
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx_path = os.path.realpath(args.onnx)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    onnx.save(model, onnx_path)
    engine_path = os.path.join(os.path.dirname(onnx_path), "engine.trt")
    print("ONNX model saved to {}".format(onnx_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--saved_model", help="The TensorFlow saved model directory to load")
    parser.add_argument("-o", "--onnx", help="The output ONNX model file to write")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Set the batch size, default: 1")
    parser.add_argument(
        "-i",
        "--input_size",
        type=int,
        help="Override the input height and width, e.g. '380', default: keep original size",
    )
    args = parser.parse_args()
    if not all([args.saved_model, args.onnx]):
        parser.print_help()
        print("\nThese arguments are required: --saved_model and --onnx")
        sys.exit(1)
    main(args)
