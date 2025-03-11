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
import onnx
import numpy as np
import argparse


def modify_maskrcnn_opset12(path_to_model, output_path):
    graph = gs.import_onnx(onnx.load(path_to_model))
    """
        Step 1: Remove unnecessary UINT8 cast
            - Pattern match Cast[BOOL->UINT8] -> Cast[UINT8 -> BOOL]
            - Fixes node 2838 - casts bool to uint8 for slice / gather. Can keep all operations in bool.
    """
    for node in graph.nodes:
        if node.op == "Cast" and node.attrs["to"] == onnx.TensorProto.UINT8:
            node.attrs["to"] = onnx.TensorProto.BOOL
            node.outputs[0].dtype = np.bool_
            # Need to modify output_node output to be bool as well.
            for output_node in node.outputs[0].outputs:
                output_node.outputs[0].dtype = np.bool_
            print(f"Removed UINT8 casts in node {node.name}")

    onnx.save(gs.export_onnx(graph.cleanup()), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="FasterRCNN-12.onnx",
        help="Path to the onnx model obtained from https://github.com/onnx/models/raw/refs/heads/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx",
    )
    parser.add_argument(
        "-o", "--output", default="fasterrcnn12_trt.onnx", help="Desired path for the output onnx model"
    )
    args = parser.parse_args()

    modify_maskrcnn_opset12(args.input, args.output)
