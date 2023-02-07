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

import onnx_graphsurgeon as gs
import argparse
import onnx
import numpy as np
import torch

# Pad layer subgraph structure in ONNX (specific to opset 11):
#               Constant
#                  |
#                Shape
#                  |
#         Mul   Gather
#          \     /
#            Sub
#             |
#       ConstantOfShape
#             |
#          Concat
#             |
#          Reshape
#             |
#           Slice
#             |
#          Transpose
#             |
#          Reshape
#             |
#    Input  Cast  Constant
#       \     |    /
#            Pad
def process_pad_nodes(graph):
    """
    Fold the pad subgraph into a single layer with pad values as input
      Input
       |
      Pad
       |
      Conv
    """
    pad_nodes = [node for node in graph.nodes if node.op == "Pad"]
    for node in pad_nodes:
        fold_pad_inputs(node, graph)

    return graph


def fold_pad_inputs(node, graph):
    # Gather the amount of padding in each dimension from pytorch graph.
    if torch.__version__ < "1.5.0":
        pad_values_pyt = node.i(1).i(0).i(0).i(0).i(0).i(0).i(0).i(0).attrs["value"].values
    else:
        pad_values_pyt = node.i(1).i(0).i(0).i(0).i(0).i(0).inputs[0].values

    # Assumption a 4d input tensor
    onnx_pad_values = [0] * 4 * 2  # 4d tensor and 2 sides padding for each dimension
    j = 3
    for i in range(0, len(pad_values_pyt), 2):
        onnx_pad_values[j] = pad_values_pyt[i]
        onnx_pad_values[j + 4] = pad_values_pyt[i + 1]
        j -= 1

    # Change the existing pad tensor to the new onnx_pad values tensor
    pads_folded_tensor = gs.Constant(name=node.inputs[1].name, values=np.array(onnx_pad_values))
    node.inputs[1] = pads_folded_tensor


# Pytorch-exported Upsample structure in ONNX:
#        Mul        Mul
#         |          |
#        Cast       Cast
#         |          |
#        Floor      Floor
#         |          |
#      Unsqueeze  Unsqueeze
#         \         /
#           Concat
#             |
#            Cast    Cast
#              \      /
#                Div
#                 |
#     Input     Concat
#       \         /
#         Upsample
def process_upsample_nodes(graph, opset=11):
    """
    Replace the upsample structure with structure below
      Conv   scale_factor
       |      /
      Upsample
       |
      ReLU
    """
    if opset >= 11:
        upsample_layer_name = "Resize"
    else:
        upsample_layer_name = "Upsample"

    upsample_nodes = [node for node in graph.nodes if node.op == upsample_layer_name]
    for node in upsample_nodes:
        fold_upsample_inputs(node, graph, opset)

    return graph


def fold_upsample_inputs(upsample, graph, opset=11):
    """
    Inplace transformation of the graph. The upsample subgraph is collapsed
    to single upsample node with input and scale factor (constant tensor).
    Args:
        upsample: upsample node in the original graph.
        graph: graph object.
    """

    if opset == 9:
        # Gather the scale factor from mul op in the upsample input subgraph
        scale_factor = upsample.i(1).i(1).i(0).i(0).i(0).i(0).i(0).i(0).i(1).attrs["value"].values

        # Create the new scales tensor
        scales = np.array([1.0, 1.0, scale_factor, scale_factor], dtype=np.float32)
        scale_tensor = gs.Constant(name=upsample.inputs[-1].name, values=scales)

        # Change the last input to the node to the new constant scales tensor.
        upsample.inputs[-1] = scale_tensor
    else:
        # In opset 11, upsample layer is exported as Resize. We will transform this Resize layer into an Upsample layer
        # and collapse the input
        sizes_tensor_name = upsample.inputs[3].name

        # Create the new scales tensor
        scale_factor = upsample.i(3).i(1).i().i().i().i().i(0).i(1).attrs["value"].values
        scales = np.array([1.0, 1.0, scale_factor, scale_factor], dtype=np.float32)
        scale_tensor = gs.Constant(name=sizes_tensor_name, values=scales)

        # Rename the Resize op to upsample and add the data and scales as inputs to the upsample layer.
        input_tensor = upsample.inputs[0]
        upsample.inputs = [input_tensor, scale_tensor]
        upsample.op = "Upsample"


# Pytorch-exported GroupNorm subgraph in ONNX:
# Conv
#   |
# Reshape    Scale    Bias
#     \       |       /
# InstanceNormalization
#         |
#      Reshape    Unsqueeze
#          \      /
#        Mul (scale)   Unsqueeze
#           \         /
#           Add (bias)
#              |
#            ReLU
def process_groupnorm_nodes(graph):
    """
    Gather the instance normalization nodes and the rest of the subgraph
    and convert into a single group normalization node.
    """
    instancenorms = [node for node in graph.nodes if node.op == "InstanceNormalization"]
    for node in instancenorms:
        convert_to_groupnorm(node, graph)

    return graph


def retrieve_attrs(instancenorm):
    """
    Gather the required attributes for the GroupNorm plugin from the subgraph.
    Args:
        instancenorm: Instance Normalization node in the graph.
    """
    attrs = {}
    # The 2nd dimension of the Reshape shape is the number of groups
    attrs["num_groups"] = instancenorm.i().i(1).attrs["value"].values[1]
    attrs["eps"] = instancenorm.attrs["epsilon"]

    # 1 is the default plugin version the parser will search for, and therefore can be omitted,
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    # "" is the default plugin namespace the parser will use, included here for illustrative purposes
    attrs["plugin_namespace"] = ""

    return attrs


def convert_to_groupnorm(instancenorm, graph):
    """
    Convert the Pytorch-exported GroupNorm subgraph to the subgraph below
    Conv
      |
    GroupNorm
      |
    ReLU
    Attributes:
        instancenorm: Instance Normalization node in the graph.
        graph: Input graph object
    """
    # Retrieve the instancenorm attributes and create the replacement node
    attrs = retrieve_attrs(instancenorm)
    groupnorm = gs.Node(op="GroupNormalizationPlugin", attrs=attrs)
    graph.nodes.append(groupnorm)

    # The plugin needs to receive an input from the Conv node, and output to the ReLU node
    conv_output_tensor = instancenorm.i().inputs[0]  # Output of Conv
    relu_input_tensor = instancenorm.o().o().o().outputs[0]  # Output of Add

    # Reconnect inputs/outputs to the groupnorm plugin
    conv_output_tensor.outputs[0] = groupnorm
    relu_input_tensor.inputs[0] = groupnorm

    # Add scale and bias constant tensors to group norm plugin
    if torch.__version__ < "1.5.0":
        groupnorm.inputs.append(instancenorm.o().o().i(1).inputs[0])
        groupnorm.inputs.append(instancenorm.o().o().o().i(1).inputs[0])
    else:
        groupnorm.inputs.append(instancenorm.o().o().inputs[1])
        groupnorm.inputs.append(instancenorm.o().o().o().inputs[1])
