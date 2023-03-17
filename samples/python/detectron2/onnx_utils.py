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

import logging
import numpy as np
import onnx_graphsurgeon as gs

logging.basicConfig(level=logging.INFO)
logging.getLogger("ModelHelper").setLevel(logging.INFO)
log = logging.getLogger("ModelHelper")

@gs.Graph.register()
def op_with_const(self, op, name, input, value):
    """
    Add an operation with constant to the graph which will operate on the input tensor with the value(s) given.
    :param op: The ONNX operation to perform, i.e. "Add" or "Mul".
    :param input: The tensor to operate on.
    :param value: The value array to operate with.
    :param name: The name to use for the node.
    """
    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created {} node '{}': {}".format(op, name, value.squeeze()))
    const = gs.Constant(name="{}_value:0".format(name), values=value)
    return self.layer(name=name, op=op, inputs=[input_tensor, const], outputs=[name + ":0"])

@gs.Graph.register()
def matmul(self, name, input, value):
    """
    Add MatMul operation to the graph which will operate on the input tensor with the value(s) given.
    :param input: The tensor to operate on.
    :param value: The linear transformation matrix to operate with.
    :param name: The name to use for the node.
    """
    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created {} node '{}': {}".format("MatMul", name, value.squeeze()))
    const = gs.Constant(name="{}_value:0".format(name), values=value)
    return self.layer(name=name, op="MatMul", inputs=[input_tensor, const], outputs=[name + ":0"])

@gs.Graph.register()
def clip(self, name, input, clip_min, clip_max):
    """
    Add Clip operation to the graph which will operate on the input tensor with the value(s) given.
    :param input: The tensor to operate on.
    :param name: The name to use for the node.
    :param clip_min: Minimum value to include, less is clipped.
    :param clip_max: Maximum value to include, more is clipped.
    """
    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created {} node '{}".format("Clip", name))
    const_min = gs.Constant(name="{}_value:0".format(name), values=np.asarray([clip_min], dtype=np.float32))
    const_max = gs.Constant(name="{}_value:1".format(name), values=np.asarray([clip_max], dtype=np.float32))
    return self.layer(name=name, op="Clip", inputs=[input_tensor, const_min, const_max], outputs=[name + ":0"])

@gs.Graph.register()
def slice(self, name, input, starts, ends, axes):
    """
    Add Slice operation to the graph which will operate on the input tensor with the value(s) given.
    :param op: The ONNX operation to perform, i.e. "Add" or "Mul".
    :param input: The tensor to operate on.
    :param name: The name to use for the node.
    :param starts: Value at which Slice starts.
    :param ends: Value at which Slice ends.
    :param axes: Axes on which Slice operation should be performed.
    """

    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created {} node '{}".format("Slice", name))
    const_start = gs.Constant(name="{}_value:0".format(name), values=np.asarray([starts], dtype=np.int64))
    const_end = gs.Constant(name="{}_value:1".format(name), values=np.asarray([ends], dtype=np.int64))
    const_axes = gs.Constant(name="{}_value:2".format(name), values=np.asarray([axes], dtype=np.int64))
    return self.layer(name=name, op="Slice", inputs=[input_tensor, const_start, const_end, const_axes], outputs=[name + ":0"])

@gs.Graph.register()
def unsqueeze(self, name, input, axes=[3]):
    """
    Adds to the graph an Unsqueeze node for the given axes and to the given input.
    :param self: The gs.Graph object being extended.
    :param name: The name to use for the node.
    :param input: The tensor to be "unsqueezed".
    :param axes: A list of axes on which to add the new dimension(s).
    :return: The first output tensor, to allow chained graph construction.
    """
    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created Unsqueeze node '{}': {}".format(name, axes))
    return self.layer(name=name, op="Unsqueeze", inputs=[input_tensor], outputs=[name + ":0"], attrs={'axes': axes})

@gs.Graph.register()
def squeeze(self, name, input, axes=[2]):
    """
    Adds to the graph an Squeeze node for the given axes and to the given input.
    :param self: The gs.Graph object being extended.
    :param name: The name to use for the node.
    :param input: The tensor to be "squeezed".
    :param axes: A list of axes on which to remove a dimension(s).
    :return: The first output tensor, to allow chained graph construction.
    """
    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created Squeeze node '{}': {}".format(name, axes))
    return self.layer(name=name, op="Squeeze", inputs=[input_tensor], outputs=[name + ":0"], attrs={'axes': axes})

@gs.Graph.register()
def gather(self, name, data, indices, axes=0):
    """
    Adds to the graph a Gather node for the given axes and to the given input.
    :param self: The gs.Graph object being extended.
    :param name: The name to use for the node.
    :param data: Data from which to gather specific tensors.
    :param indices: Indices by which to gather data tensors.
    :param axes: A list of axes on which to perform gather operation
    """
    data_tensor = data if type(data) is gs.Variable else data[0]
    indices_tensor = indices if type(indices) is gs.Variable else indices[0]
    log.debug("Created Gather node '{}': {}".format(name, axes))
    return self.layer(name=name, op="Gather", inputs=[data_tensor, indices_tensor], outputs=[name + ":0"], attrs={'axes': axes})

@gs.Graph.register()
def transpose(self, name, input, perm):
    """
    Adds to the graph a Transpose node for the given axes permutation and to the given input.
    :param self: The gs.Graph object being extended.
    :param name: The name to use for the node.
    :param input: The tensor to be transposed.
    :param perm: A list of axes defining their order after transposing occurs.
    :return: The first output tensor, to allow chained graph construction.
    """
    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created Transpose node '{}': {}".format(name, perm))
    return self.layer(name=name, op="Transpose", inputs=[input_tensor], outputs=[name + ":0"], attrs={'perm': perm})

@gs.Graph.register()
def sigmoid(self, name, input):
    """
    Adds to the graph a Sigmoid node for the given input.
    :param self: The gs.Graph object being extended.
    :param name: The name to use for the node.
    :param input: The tensor to be applied to.
    :return: The first output tensor, to allow chained graph construction.
    """
    input_tensor = input if type(input) is gs.Variable else input[0]
    log.debug("Created Sigmoid node '{}'".format(name))
    return self.layer(name=name, op="Sigmoid", inputs=[input_tensor], outputs=[name + ":0"])

@gs.Graph.register()
def plugin(self, op, name, inputs: list, outputs: list, attrs):
    """
    Adds to the graph a TensorRT plugin node with the given name, inputs and outputs. The attrs dictionary holds
    attributes to be added to the plugin node.
    :param self: The gs.Graph object being extended.
    :param op: The registered name for the TensorRT plugin.
    :param name: The name to use for the node.
    :param inputs: The list of tensors to use an inputs.
    :param outputs: The list of tensors to use as outputs.
    :param attrs: The dictionary to use as attributes.
    :return: The first output tensor, to allow chained graph construction.
    """
    log.debug("Created TRT Plugin node '{}': {}".format(name, attrs))
    return self.layer(op=op, name=name, inputs=inputs, outputs=outputs, attrs=attrs)

@gs.Graph.register()
def find_node_by_op(self, op):
    """
    Finds the first node in the graph with the given operation name.
    :param self: The gs.Graph object being extended.
    :param op: The operation name to search for.
    :return: The first node matching that performs that op.
    """
    for node in self.nodes:
        if node.op == op:
            return node
    return None

@gs.Graph.register()
def find_node_by_op_name(self, op, name):
    """
    Finds the first node in the graph with the given operation name.
    :param self: The gs.Graph object being extended.
    :param op: The operation name to search for.
    :param name: Selected node name.
    :return: The first node matching that performs that op.
    """
    for node in self.nodes:
        if node.op == op and node.name == name:
            return node
    return None

@gs.Graph.register()
def find_node_by_op_input_output_name(self, op, input_name, output_name, input_pos=0, output_pos=0):
    """
    Finds the first node in the graph with the given operation name.
    :param self: The gs.Graph object being extended.
    :param op: The operation name to search for.
    :param input_pos: Which input to consider, default is 0.
    :param output_pos: Which output to consider, default is 0.
    :param input_name: Selected input's name.
    :param output_name: Selected output's name.
    :return: The first node matching that performs that op.
    """
    for node in self.nodes:
        if node.op == op and node.inputs[input_pos].name == input_name and node.outputs[output_pos].name == output_name:
            return node
    return None

@gs.Graph.register()
def find_descendant_by_op(self, node, op, depth=10):
    """
    Starting from the given node, finds a node lower in the graph matching the given operation name.
    This is not an exhaustive graph search.
    In order to graph search bfs is used, so runtime complexity is O(V+E).
    :param self: The gs.Graph object being extended.
    :param node: The node to start searching from.
    :param op: The operation name to search for.
    :param depth: Stop searching after traversing these many nodes.
    :return: The first descendant node matching that performs that op.
    """
    queue = []
    for i in range(depth):
        queue.append(node.o())
        while queue:
            node = queue.pop(0)
            if node.op == op:
                return node
            for child in node.outputs[0].outputs:
                queue.append(child)
    return None

@gs.Graph.register()
def find_ancestor_by_op(self, node, op, depth=10):
    """
    Starting from the given node, finds a node higher in the graph matching the given operation name.
    This is not an exhaustive graph search.
    In order to graph search bfs is used, so runtime complexity is O(V+E).
    :param self: The gs.Graph object being extended.
    :param node: The node to start searching from.
    :param op: The operation name to search for.
    :param depth: Stop searching after traversing these many nodes.
    :return: The first ancestor node matching that performs that op.
    """
    queue = []
    for i in range(depth):
        queue.append(node.i())
        while queue:
            node = queue.pop(0)
            if node.op == op:
                return node
            for child in node.inputs[-1].inputs:
                queue.append(child)
    return None
