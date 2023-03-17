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

import onnx_graphsurgeon as gs

logging.basicConfig(level=logging.INFO)
logging.getLogger("EfficientDetHelper").setLevel(logging.INFO)
log = logging.getLogger("EfficientDetHelper")


@gs.Graph.register()
def elt_const(self, op, name, input, value):
    """
    Add an element-wise operation to the graph which will operate on the input tensor with the value(s) given.
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
def unsqueeze(self, name, input, axes=[-1]):
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
    return self.layer(name=name, op="Unsqueeze", inputs=[input_tensor], outputs=[name + ":0"], attrs={"axes": axes})


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
    return self.layer(name=name, op="Transpose", inputs=[input_tensor], outputs=[name + ":0"], attrs={"perm": perm})


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
def plugin(self, op, name, inputs, outputs, attrs):
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
    input_tensors = inputs if type(inputs) is list else [inputs]
    log.debug("Created TRT Plugin node '{}': {}".format(name, attrs))
    return self.layer(op=op, name=name, inputs=input_tensors, outputs=outputs, attrs=attrs)


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
def find_descendant_by_op(self, node, op, depth=10):
    """
    Starting from the given node, finds a node lower in the graph matching the given operation name. This is not an
    exhaustive graph search, it will take only the first output of each node traversed while searching depth-first.
    :param self: The gs.Graph object being extended.
    :param node: The node to start searching from.
    :param op: The operation name to search for.
    :param depth: Stop searching after traversing these many nodes.
    :return: The first descendant node matching that performs that op.
    """
    for i in range(depth):
        node = node.o()
        if node.op == op:
            return node
    return None


@gs.Graph.register()
def find_ancestor_by_op(self, node, op, depth=10):
    """
    Starting from the given node, finds a node higher in the graph matching the given operation name. This is not an
    exhaustive graph search, it will take only the first input of each node traversed while searching depth-first.
    :param self: The gs.Graph object being extended.
    :param node: The node to start searching from.
    :param op: The operation name to search for.
    :param depth: Stop searching after traversing these many nodes.
    :return: The first ancestor node matching that performs that op.
    """
    for i in range(depth):
        node = node.i()
        if node.op == op:
            return node
    return None
