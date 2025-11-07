#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Dict, List, Union
import copy

from onnx_graphsurgeon.ir.graph import Constant, Graph, Node
from onnx_graphsurgeon.logger import G_LOGGER


class PatternMapping(dict):
    """
    Represents a graph pattern mapping result.
    """

    def __init__(self, onnx_node=None) -> None:
        super().__init__()
        self.onnx_node = onnx_node

        self.inputs = []
        self.outputs = []
        if onnx_node is not None:
            self.inputs = onnx_node.inputs
            self.outputs = onnx_node.outputs

        self.constants = dict()  # constant name -> onnx tensor mapping

    def set_input_onnx_tensor(self, onnx_tensor, index):
        length = len(self.inputs)
        for _ in range(index - length + 1):
            self.inputs.append(None)
        if (
            self.inputs[index] is not None
            and self.inputs[index].name != onnx_tensor.name
        ):
            return False  # This input tensor has been set up by another onnx tensor
        self.inputs[index] = onnx_tensor
        return True

    def set_output_onnx_tensor(self, onnx_tensor, index):
        length = len(self.outputs)
        for _ in range(index - length + 1):
            self.outputs.append(None)
        if (
            self.outputs[index] is not None
            and self.outputs[index].name != onnx_tensor.name
        ):
            return False  # This output tensor has been set up by another onnx tensor
        self.outputs[index] = onnx_tensor
        return True

    def set_constant_onnx_tensor(self, onnx_tensor, name):
        if name in self.constants and self.constants[name].name != onnx_tensor.name:
            return False
        self.constants[name] = onnx_tensor
        return True

    def _get_node(self):
        return self.onnx_node

    def get(self, name: str):
        """
        Retrieve a pattern-to-graph mapping given the pattern node name.

        Args:
            name (str): The name of the pattern node. The pattern node can be a single op node or a subpattern.

        Returns:
            PatternMapping for a subpattern node or gs.Node for a single op node.
        """
        if self[name].onnx_node is not None:
            return self[name].onnx_node

        return self[name]

    def __str__(self) -> str:
        if self.onnx_node is None:
            return (
                "{"
                + str.join(
                    ", ", [f"{key}: {str(value)}" for key, value in self.items()]
                )
                + "}"
            )
        return self.onnx_node.name


class GraphPattern:
    """
    Represent a graph pattern.

    Example:
    ::

        pattern = GraphPattern()
        conv = pattern.add("Conv")
        leaky_relu = pattern.add("LeakyReLU", inputs=[conv], check_func=lambda node: node.attrs["alpha"] < 1.0)
    """

    def __init__(self) -> None:
        self.op = None  # op (str)
        self.check_func = None  # callback function for single node
        # pattern node name -> GraphPattern nodes(single or subpattern)
        self.nodes: Dict[str, "GraphPattern"] = dict()
        # pattern node name -> input tensors
        self.node_inputs: Dict[str, List[int]] = dict()
        # pattern node name -> output tensors
        self.node_outputs: Dict[str, List[int]] = dict()
        self.num_tensors = 0  # number of all tensors in the pattern
        self.tensor_inputs: Dict[int, List[str]] = dict()  # tensor id -> input node
        self.tensor_outputs: Dict[int, List[str]] = dict()  # tensor id -> output nodes
        self.input_tensors: List[int] = []  # a list of input tensor ids of this pattern
        self.output_tensors: List[int] = []
        # tensor id -> tensor name of constant tensors.
        self.constant_tensors: Dict[int, str] = dict()

    def _add_tensor(self, input_node=None) -> int:
        tensor_id = self.num_tensors
        self.tensor_inputs[tensor_id] = []
        if input_node is not None:
            self.tensor_inputs[tensor_id].append(input_node)
        self.tensor_outputs[tensor_id] = []

        self.num_tensors += 1
        return tensor_id

    def variable(self) -> int:
        """
        Add a variable tensor without a input node - This tensor will be an input tensor of this graph pattern.

        Return:
            int: the tensor id.
        """
        tensor_id = self._add_tensor()
        self.input_tensors.append(tensor_id)
        return tensor_id

    def constant(self, name=None) -> int:
        """
        Add a constant tensor. If name is not provided, a default name will be assigned.

        Args:
            name(str): the constant tensor name

        Return:
            int: the tensor id.
        """
        tensor_id = self._add_tensor()
        if name is None:
            name = f"unnamed_constant_tensor_{tensor_id}"
        self.constant_tensors[tensor_id] = name
        return tensor_id

    def set_output_tensors(self, output_tensors) -> None:
        for tensor_id in output_tensors:
            assert tensor_id in self.tensor_inputs
        self.output_tensors = output_tensors

    def _init_single_node(self, op, check_func=None) -> None:
        self.op = op
        self.check_func = check_func

    def add(
        self,
        name: str,
        op: Union["GraphPattern", str],
        check_func=None,
        inputs=None,
        num_output_tensors=1,
    ):
        """
        Add an op node or a subpattern node to the current pattern.

        Args:
            name (str): the node name.
            op (Union[GraphPattern, str]): the GraphPattern instance if adding a subpattern node or the op name if adding a single op node.
            check_func (function): the callback function for additional matching rules of an op node if adding a single op node.
            inputs (list): the list of input tensors. If this node is a sub-pattern, the sequence of this list should align with the sequence of the sub-pattern's input tensors.
            num_output_tensors (int): number of output tensors

        Return:
            tuple(int) or int or None: output tensors.
        """
        assert self.op is None
        assert name not in self.nodes

        if inputs is None:
            inputs = []

        if isinstance(op, str):
            op_name = op
            op = GraphPattern()
            op._init_single_node(op_name, check_func)

        self.nodes[name] = op

        self.node_inputs[name] = inputs

        self.node_outputs[name] = []
        for _ in range(num_output_tensors):
            self.node_outputs[name].append(self._add_tensor(input_node=name))

        for input in inputs:
            self.tensor_outputs[input].append(name)

        if len(self.node_outputs[name]) == 0:
            return None
        elif len(self.node_outputs[name]) == 1:
            return self.node_outputs[name][0]
        return tuple(self.node_outputs[name])

    def _get_inbound(self, tensor_index):
        if len(self.input_tensors) > tensor_index:
            tensor_id = self.input_tensors[tensor_index]
            if len(self.tensor_outputs[tensor_id]):
                inbound_node = self.tensor_outputs[tensor_id][0]
                return tensor_id, inbound_node
        return None, None

    def _get_outbound(self, tensor_index):
        if len(self.output_tensors) > tensor_index:
            tensor_id = self.output_tensors[tensor_index]
            if len(self.tensor_inputs[tensor_id]):
                outbound_node = self.tensor_inputs[tensor_id][0]
                return tensor_id, outbound_node
        return None, None

    def _single_node_match(self, onnx_node: Node) -> bool:
        assert self.op is not None
        with G_LOGGER.indent():
            if self.op != onnx_node.op:
                G_LOGGER.info(
                    "No match because: Op did not match. Node op was: {:} but pattern op was: {:}.".format(
                        onnx_node.op, self.op
                    )
                )
                return False
            if self.check_func is not None:
                if not self.check_func(onnx_node):
                    G_LOGGER.info("No match because: check_func returned false.")
                    return False
            G_LOGGER.info(
                "Single node is matched: {:}, {:}".format(self.op, onnx_node.name)
            )
        return True

    def _get_tensor_index_for_node(
        self, node: str, tensor_id: int, is_node_input: bool
    ):
        if is_node_input:
            return self.node_inputs[node].index(tensor_id)
        else:
            return self.node_outputs[node].index(tensor_id)

    def get_inbound_or_outbound_onnx_node(
        self, mapping: PatternMapping, is_inbound: bool, tensor_index: int
    ):
        if self.op is not None:
            onnx_node = mapping._get_node()
            return onnx_node
        if is_inbound:
            inbound_tensor, inbound_node = self._get_inbound(tensor_index)
            if inbound_node is not None:
                return self.nodes[inbound_node].get_inbound_or_outbound_onnx_node(
                    mapping[inbound_node],
                    is_inbound=True,
                    tensor_index=self._get_tensor_index_for_node(
                        inbound_node, inbound_tensor, is_node_input=True
                    ),
                )

        else:
            outbound_tensor, outbound_node = self._get_outbound(tensor_index)
            if outbound_node is not None:
                return self.nodes[outbound_node].get_inbound_or_outbound_onnx_node(
                    mapping[outbound_node],
                    is_inbound=False,
                    tensor_index=self._get_tensor_index_for_node(
                        outbound_node, outbound_tensor, is_node_input=False
                    ),
                )
        return None

    # Match an onnx node and its subgraph with the current pattern.
    def match(
        self,
        onnx_node: Node,
        from_inbound: bool,
        from_tensor_index: int,
        mapped_onnx_nodes: set,
        onnx_graph_output_tensors: set,
    ):
        if onnx_node.id in mapped_onnx_nodes:
            return None
        if self.op is not None:  # is single node
            if self._single_node_match(onnx_node):
                mapped_onnx_nodes.add(onnx_node.id)
                return PatternMapping(onnx_node=onnx_node)
            else:
                return None

        initial_node = None
        if from_inbound:
            from_tensor, initial_node = self._get_inbound(from_tensor_index)
        else:
            from_tensor, initial_node = self._get_outbound(from_tensor_index)
        assert initial_node is not None

        mapping = PatternMapping()
        match = self._match_node(
            initial_node,
            onnx_node,
            from_tensor,
            mapping,
            mapped_onnx_nodes,
            onnx_graph_output_tensors,
            from_inbound,
        )
        if match:
            # If we entered this subpattern via an inbound boundary tensor, ensure
            # that all internal consumers of the boundary input tensor are covered.
            # This supports cases where a single subpattern input fans out to multiple
            # internal nodes.
            if from_inbound:
                # Determine the boundary ONNX tensor corresponding to this inbound edge
                tensor_index_for_node = self._get_tensor_index_for_node(
                    initial_node, from_tensor, is_node_input=True
                )
                if tensor_index_for_node >= len(onnx_node.inputs):
                    return None
                boundary_onnx_tensor = onnx_node.inputs[tensor_index_for_node]

                # All internal consumer nodes of the boundary input tensor
                internal_consumers = list(self.tensor_outputs.get(from_tensor, []))

                # Build candidate ONNX consumer nodes for the boundary tensor
                candidate_onnx_consumers = list(getattr(boundary_onnx_tensor, "outputs", []))
                used_consumer_ids = set()

                # Mark already mapped consumers as used if applicable
                for consumer_node in internal_consumers:
                    if consumer_node in mapping:
                        inbound_tensor_index = self._get_tensor_index_for_node(
                            consumer_node, from_tensor, is_node_input=True
                        )
                        inbound_mapped = self.nodes[consumer_node].get_inbound_or_outbound_onnx_node(
                            mapping[consumer_node], is_inbound=True, tensor_index=inbound_tensor_index
                        )
                        if inbound_mapped is None:
                            return None
                        # Ensure the mapped node is indeed one of the boundary tensor's consumers
                        if all(c.id != inbound_mapped.id for c in candidate_onnx_consumers):
                            return None
                        used_consumer_ids.add(inbound_mapped.id)

                # For remaining internal consumers not yet mapped, try to match them to unused ONNX consumers
                for consumer_node in internal_consumers:
                    if consumer_node in mapping:
                        continue
                    matched_this_consumer = False
                    for c in candidate_onnx_consumers:
                        if c.id in used_consumer_ids or c.id in mapped_onnx_nodes:
                            continue
                        # Try to extend mapping with this branch. Use copies to allow backtracking on failure.
                        mapping_copy = copy.deepcopy(mapping)
                        mapped_onnx_nodes_copy = set(mapped_onnx_nodes)
                        ok = self._match_node(
                            consumer_node,
                            c,
                            from_tensor,
                            mapping_copy,
                            mapped_onnx_nodes_copy,
                            onnx_graph_output_tensors,
                            from_inbound=True,
                        )
                        if ok:
                            mapping.clear()
                            for _k, _v in mapping_copy.items():
                                mapping[_k] = _v
                            mapping.inputs = mapping_copy.inputs
                            mapping.outputs = mapping_copy.outputs
                            mapping.constants = mapping_copy.constants
                            mapped_onnx_nodes = mapped_onnx_nodes_copy
                            used_consumer_ids.add(c.id)
                            matched_this_consumer = True
                            break
                    if not matched_this_consumer:
                        return None

            return mapping
        else:
            return None

    # Match an onnx node and its subgraph with a starting pattern node(can be a subpattern node or a single node) and its subgraph. This is the actual dfs.
    def _match_node(
        self,
        node_name: str,
        onnx_node: Node,
        from_tensor: int,
        mapping: PatternMapping,
        mapped_onnx_nodes: set,
        onnx_graph_output_tensors: set,
        from_inbound: bool,
    ) -> bool:
        with G_LOGGER.indent():
            G_LOGGER.info(
                "Checking node: {:} against pattern node: {:}.".format(
                    onnx_node.name, node_name
                )
            )
        tensor_index_for_node = self._get_tensor_index_for_node(
            node_name, from_tensor, is_node_input=from_inbound
        )
        subgraph_mapping = self.nodes[node_name].match(
            onnx_node,
            from_inbound,
            tensor_index_for_node,
            mapped_onnx_nodes,
            onnx_graph_output_tensors,
        )
        if subgraph_mapping is not None:
            mapping[node_name] = subgraph_mapping
        else:
            return False

        input_onnx_tensors = subgraph_mapping.inputs
        if len(input_onnx_tensors) != len(self.node_inputs[node_name]):
            return False  # Number of node inputs should equal to number of input onnx tensors of the node.
        for node_input_tensor, onnx_tensor in zip(
            self.node_inputs[node_name], input_onnx_tensors
        ):
            if onnx_tensor is None:
                return False
            # tensor paired up.
            if node_input_tensor in self.input_tensors:
                if not mapping.set_input_onnx_tensor(
                    onnx_tensor, self.input_tensors.index(node_input_tensor)
                ):
                    return False  # this tensor is mapped to another onnx tensor
                continue
            if node_input_tensor in self.constant_tensors:
                if not isinstance(onnx_tensor, Constant):
                    return False  # constant tensor not match
                if not mapping.set_constant_onnx_tensor(
                    onnx_tensor, self.constant_tensors[node_input_tensor]
                ):
                    # this constant tensor is mapped to another onnx tensor
                    return False
                continue
            if len(self.tensor_inputs[node_input_tensor]) != len(onnx_tensor.inputs):
                return False
            for input_node, input_onnx_node in zip(
                self.tensor_inputs[node_input_tensor], onnx_tensor.inputs
            ):
                # dfs ends when revisiting a node. We need to check if the edges are matched.
                if input_node in mapping:
                    outbound_tensor_index = self._get_tensor_index_for_node(
                        input_node, node_input_tensor, is_node_input=False
                    )
                    outbound_onnx_node_of_input_node = self.nodes[
                        input_node
                    ].get_inbound_or_outbound_onnx_node(
                        mapping[input_node],
                        is_inbound=False,
                        tensor_index=outbound_tensor_index,
                    )
                    if (
                        outbound_onnx_node_of_input_node is None
                        or outbound_onnx_node_of_input_node.name != input_onnx_node.name
                    ):
                        return False
                    continue
                match = self._match_node(
                    input_node,
                    input_onnx_node,
                    node_input_tensor,
                    mapping,
                    mapped_onnx_nodes,
                    onnx_graph_output_tensors,
                    from_inbound=False,
                )
                if not match:
                    return False

        output_onnx_tensors = subgraph_mapping.outputs
        if len(output_onnx_tensors) != len(self.node_outputs[node_name]):
            return False  # Number of node outputs should be equal to number of output onnx tensors of the node.
        for node_output_tensor, onnx_tensor in zip(
            self.node_outputs[node_name], output_onnx_tensors
        ):
            if onnx_tensor is None:
                return False
            # tensor matched
            if node_output_tensor in self.output_tensors:
                if not mapping.set_output_onnx_tensor(
                    onnx_tensor, self.output_tensors.index(node_output_tensor)
                ):
                    return False  # this tensor is mapped to another onnx tensor
                continue
            if onnx_tensor.name in onnx_graph_output_tensors:
                return False  # The pattern tensor is not an output but the onnx tensor is an output tensor of the onnx graph.

            # Flexible consumer matching: each pattern consumer must map to some ONNX consumer,
            # but the ONNX tensor may have additional consumers we can ignore.
            pattern_consumers = list(self.tensor_outputs[node_output_tensor])
            candidate_consumers = list(getattr(onnx_tensor, "outputs", []))
            used_candidate_ids = set()

            # First validate already mapped consumers
            for output_node in pattern_consumers:
                if output_node in mapping:
                    inbound_tensor_index = self._get_tensor_index_for_node(
                        output_node, node_output_tensor, is_node_input=True
                    )
                    inbound_onnx_node_of_output_node = self.nodes[
                        output_node
                    ].get_inbound_or_outbound_onnx_node(
                        mapping[output_node],
                        is_inbound=True,
                        tensor_index=inbound_tensor_index,
                    )
                    if (
                        inbound_onnx_node_of_output_node is None
                        or all(c.id != inbound_onnx_node_of_output_node.id for c in candidate_consumers)
                    ):
                        return False
                    used_candidate_ids.add(inbound_onnx_node_of_output_node.id)

            # Then match remaining consumers greedily with backtracking via copies
            for output_node in pattern_consumers:
                if output_node in mapping:
                    continue
                matched_output_consumer = False
                for oc in candidate_consumers:
                    if oc.id in used_candidate_ids or oc.id in mapped_onnx_nodes:
                        continue
                    mapping_copy = copy.deepcopy(mapping)
                    mapped_onnx_nodes_copy = set(mapped_onnx_nodes)
                    ok = self._match_node(
                        output_node,
                        oc,
                        node_output_tensor,
                        mapping_copy,
                        mapped_onnx_nodes_copy,
                        onnx_graph_output_tensors,
                        from_inbound=True,
                    )
                    if ok:
                        # Adopt mapping changes in-place so caller's reference sees updates
                        mapping.clear()
                        for _k, _v in mapping_copy.items():
                            mapping[_k] = _v
                        mapping.inputs = mapping_copy.inputs
                        mapping.outputs = mapping_copy.outputs
                        mapping.constants = mapping_copy.constants
                        mapped_onnx_nodes = mapped_onnx_nodes_copy
                        used_candidate_ids.add(oc.id)
                        matched_output_consumer = True
                        break
                if not matched_output_consumer:
                    return False
            # For non-pattern-output tensors, disallow extra external consumers.
            # All consumers of this ONNX tensor must be accounted for by the pattern.
            if node_output_tensor not in self.output_tensors:
                for oc in candidate_consumers:
                    if oc.id not in used_candidate_ids:
                        return False
        return True

    def match_all(self, graph: Graph) -> List[PatternMapping]:
        """
        Find all the matched instances of subgraph with the current pattern in the given graph.

        Args:
            graph (Graph): the graph to match.

        Return:
            List[PatternMapping]: list of mappings.
        """
        mappings = []
        onnx_graph_output_tensors = set(tensor.name for tensor in graph.outputs)
        with graph.node_ids():
            for node in graph.nodes:
                G_LOGGER.info("Start a subgraph matching...")
                mapped_onnx_nodes = set()
                mapping = self.match(
                    node,
                    from_inbound=True,
                    from_tensor_index=0,
                    mapped_onnx_nodes=mapped_onnx_nodes,
                    onnx_graph_output_tensors=onnx_graph_output_tensors,
                )
                if mapping is not None:
                    G_LOGGER.info("Found a matched subgraph!")
                    mappings.append(mapping)
        return mappings
