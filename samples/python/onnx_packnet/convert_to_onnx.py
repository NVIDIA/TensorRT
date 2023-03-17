#!/usr/bin/env python3
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

import onnx
import torch
import numpy as np
import argparse
import onnx_graphsurgeon as gs
from post_processing import *
from packnet_sfm.networks.depth.PackNet01 import PackNet01


def post_process_packnet(model_file, opset=11):
    """
    Use ONNX graph surgeon to replace upsample and instance normalization nodes. Refer to post_processing.py for details.
    Args:
        model_file : Path to ONNX file
    """
    # Load the packnet graph
    graph = gs.import_onnx(onnx.load(model_file))

    if opset >= 11:
        graph = process_pad_nodes(graph)

    # Replace the subgraph of upsample with a single node with input and scale factor.
    if torch.__version__ < "1.5.0":
        graph = process_upsample_nodes(graph, opset)

    # Convert the group normalization subgraph into a single plugin node.
    graph = process_groupnorm_nodes(graph)

    # Remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort()

    # Export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), model_file)

    print("Saving the ONNX model to {}".format(model_file))


def build_packnet(model_file, args):
    """
    Construct the packnet network and export it to ONNX
    """
    input_pyt = torch.randn((1, 3, 192, 640), requires_grad=False)

    # Build the model
    model_pyt = PackNet01(version="1A")

    # Convert the model into ONNX
    torch.onnx.export(model_pyt, input_pyt, model_file, verbose=args.verbose, opset_version=args.opset)


def main():
    parser = argparse.ArgumentParser(
        description="Exports PackNet01 to ONNX, and post-processes it to insert TensorRT plugins"
    )
    parser.add_argument("-o", "--output", help="Path to save the generated ONNX model", default="model.onnx")
    parser.add_argument("-op", "--opset", type=int, help="ONNX opset to use", default=11)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Flag to enable verbose logging for torch.onnx.export"
    )
    args = parser.parse_args()

    # Construct the packnet graph and generate the onnx graph
    build_packnet(args.output, args)

    # Perform post processing on Instance Normalization and upsampling nodes and create a new ONNX graph
    post_process_packnet(args.output, args.opset)


if __name__ == "__main__":
    main()
