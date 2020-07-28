#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor, Constant, Variable
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.util import misc

from collections import OrderedDict, defaultdict
from typing import Sequence, Set, Dict, Tuple
import numpy as np
import copy


# Functor that returns whether a Tensor has never been seen before
class UnseenTensor(object):
    def __init__(self, initial_tensors=None):
        tensors = misc.default_value(initial_tensors, [])
        self.seen_tensors = set([tensor.name for tensor in tensors])

    def __call__(self, tensor):
        # Empty tensors are never "seen"
        if tensor.is_empty():
            return True
        elif tensor.name not in self.seen_tensors:
            self.seen_tensors.add(tensor.name)
            return True
        return False


class NodeIDAdder(object):
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        # To get unique ids for each node, add an `id` attribute. This will be removed before the function returns.
        # Using the index in the node list allows the same object to count as different nodes.
        for index, node in enumerate(self.graph.nodes):
            node.id = index

    def __exit__(self, exc_type, exc_value, traceback):
        for node in self.graph.nodes:
            del node.id


class Graph(object):
    DEFAULT_OPSET = 11
    OPSET_FUNC_MAP = defaultdict(dict)

    @staticmethod
    def register(opsets=list(range(DEFAULT_OPSET + 1))):
        """
        Registers a function with the Graph class for the specified group of opsets.
        After registering the function, it can be accessed like a normal member function.

        For example:
        ::
            @Graph.register()
            def add(self, a, b):
                return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])

            graph.add(a, b)

        Optional Args:
            opsets (Sequence[int]): A group of opsets for which to register the function. By default, the function is registered for all opsets up to and including Graph.DEFAULT_OPSET. Multiple functions with the same name may be registered simultaneously if they are registered for different opsets. Registering a function with a duplicate name for the same opsets will overwrite any function previously registered for those opsets.
        """
        def register_func(func):
            if hasattr(Graph, func.__name__):
                G_LOGGER.warning("Registered function: {:} is hidden by a Graph attribute or function with the same name. This function will never be called!".format(func.__name__))

            for opset in opsets:
                Graph.OPSET_FUNC_MAP[opset][func.__name__] = func
            return func
        return register_func


    def __init__(self, nodes: Sequence[Node]=None, inputs: Sequence[Tensor]=None, outputs: Sequence[Tensor]=None, name=None, doc_string=None, opset=None):
        """
        Represents a graph containing nodes and tensors.

        Optional Args:
            nodes (Sequence[Node]): A list of the nodes in this graph.
            inputs (Sequence[Tensor]): A list of graph input Tensors.
            outputs (Sequence[Tensor]): A list of graph output Tensors.
            name (str): The name of the graph. Defaults to "onnx_graphsurgeon".
            doc_string (str): A doc_string for the graph. Defaults to "".
        """
        self.nodes = misc.default_value(nodes, [])
        self.inputs = misc.default_value(inputs, [])
        self.outputs = misc.default_value(outputs, [])

        self.name = misc.default_value(name, "onnx_graphsurgeon")
        self.__name__ = self.name

        self.doc_string = misc.default_value(doc_string, "")
        self.opset = misc.default_value(opset, Graph.DEFAULT_OPSET)
        # Printing graphs can be very expensive
        G_LOGGER.ultra_verbose(lambda: "Created Graph: {:}".format(self))
        # For layer() function
        self.name_idx = 0


    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as err:
            if self.opset not in Graph.OPSET_FUNC_MAP or name not in Graph.OPSET_FUNC_MAP[self.opset]:
                G_LOGGER.error("No function: {:} registered for opset: {:}".format(name, self.opset))
                raise err
            return lambda *args, **kwargs: Graph.OPSET_FUNC_MAP[self.opset][name](self, *args, **kwargs)


    def __eq__(self, other: "Graph"):
        nodes_match = len(self.nodes) == len(other.nodes) and all([node == other_node for node, other_node in zip(self.nodes, other.nodes)])
        inputs_match = len(self.inputs) == len(other.inputs) and all([inp == other_inp for inp, other_inp in zip(self.inputs, other.inputs)])
        outputs_match = len(self.outputs) == len(other.outputs) and all([out == other_out for out, other_out in zip(self.outputs, other.outputs)])
        return nodes_match and inputs_match and outputs_match


    def node_ids(self):
        """
        Returns a context manager that supplies unique integer IDs for Nodes in the Graph.

        Example:
            with graph.node_ids():
                assert graph.nodes[0].id != graph.nodes[1].id

        Returns:
            NodeIDAdder: A context manager that supplies unique integer IDs for Nodes.
        """
        return NodeIDAdder(self)


    def _get_node_id(self, node):
        try:
            return node.id
        except AttributeError:
            G_LOGGER.critical("Encountered a node not in the graph:\n{:}.\n\nTo fix this, please append the node to this graph's `nodes` attribute.".format(node))


    # Returns a list of node ids of used nodes, and a list of used tensors.
    def _get_used_node_ids(self):
        used_node_ids = set()
        # Traverse backwards from outputs to find all used nodes.
        ignore_seen = UnseenTensor()
        used_tensors = list(filter(ignore_seen, self.outputs))

        index = 0
        while index < len(used_tensors):
            used_tensor = used_tensors[index]
            index += 1
            for node in used_tensor.inputs:
                used_node_ids.add(self._get_node_id(node))
                used_tensors.extend(filter(ignore_seen, node.inputs))
        return used_node_ids, used_tensors


    def cleanup(self, remove_unused_node_outputs=True):
        """
        Removes unused nodes and tensors from the graph.
        A node or tensor is considered unused if it does not contribute to any of the graph outputs.

        Note: This function will never modify graph output tensors.

        Optional Args:
            remove_unused_node_outputs (bool): Whether to remove unused output tensors of nodes. This will never remove
                empty tensor outputs. If this is set to False, outputs of nodes kept in the graph will not be modified.

        Returns:
            self
        """
        with self.node_ids():
            used_node_ids, used_tensors = self._get_used_node_ids()

            inputs = []
            for inp in self.inputs:
                if inp in used_tensors:
                    inputs.append(inp)
                else:
                    G_LOGGER.debug("Removing unused input: {:}".format(inp))
            self.inputs = inputs

            nodes = []
            for node in self.nodes:
                if self._get_node_id(node) in used_node_ids:
                    nodes.append(node)
                else:
                    node.inputs.clear()
                    node.outputs.clear()
                    G_LOGGER.verbose("Removing unused node: {:}".format(node))

            # Last pass to remove any hanging tensors - tensors without outputs
            if remove_unused_node_outputs:
                graph_output_names = set([tensor.name for tensor in self.outputs])
                for node in nodes:
                    def is_hanging_tensor(tensor):
                        return not tensor.is_empty() and len(tensor.outputs) == 0 and tensor.name not in graph_output_names

                    [node.outputs.remove(out) for out in node.outputs if is_hanging_tensor(out)]

            self.nodes = nodes
            return self


    def toposort(self):
        """
        Topologically sort the graph in place.

        Returns:
            self
        """
        # Keeps track of a node and it's level in the graph hierarchy. 0 corresponds to an input node, N corresponds to a node with N layers of inputs.
        class HierarchyDescriptor(object):
            def __init__(self, node=None, level=None):
                self.node = node
                self.level = level


            def __lt__(self, other):
                return self.level < other.level

        hierarchy_levels = {} # Dict[int, HierarchyDescriptor]

        def get_hierarchy_level(node):
            # Return all nodes that contribute to this node.
            def get_input_nodes(node):
                inputs = {}
                for tensor in node.inputs:
                    for node in tensor.inputs:
                        inputs[self._get_node_id(node)] = node
                return inputs.values()

            if self._get_node_id(node) in hierarchy_levels:
                return hierarchy_levels[self._get_node_id(node)].level

            # The level of a node is the level of it's highest input + 1.
            try:
                max_input_level = max([get_hierarchy_level(input_node) for input_node in get_input_nodes(node)] + [-1])
            except RecursionError:
                G_LOGGER.critical("Cycle detected in graph! Are there tensors with duplicate names in the graph?")

            return max_input_level + 1

        with self.node_ids():
            for node in self.nodes:
                hierarchy_levels[self._get_node_id(node)] = HierarchyDescriptor(node, level=get_hierarchy_level(node))

        self.nodes = [hd.node for hd in sorted(hierarchy_levels.values())]
        return self


    def tensors(self, check_duplicates=False):
        """
        Creates a tensor map of all the tensors in this graph by walking over all nodes. Empty tensors are omitted from this map. The graph must not contain tensors with duplicate names.

        Tensors are guaranteed to be in order of the nodes in the graph. Hence, if the graph is topologically sorted, the tensor map will be too.

        Optional Args:
            check_duplicates (bool): Whether to fail if multiple tensors with the same name are encountered.

        Raises:
            OnnxGraphSurgeonException: If check_duplicates is True, and multiple distinct tensors in the graph share the same name.

        Returns:
            OrderedDict[str, Tensor]: A mapping of tensor names to tensors.
        """
        tensor_map = OrderedDict()

        def add_to_tensor_map(tensor):
            if not tensor.is_empty():
                if check_duplicates and tensor.name in tensor_map and not (tensor_map[tensor.name] is tensor):
                    G_LOGGER.critical("Found distinct tensors that share the same name:\n[id: {:}] {:}\n[id: {:}] {:}"
                        .format(id(tensor_map[tensor.name]), tensor_map[tensor.name], id(tensor), tensor))
                tensor_map[tensor.name] = tensor

        for node in self.nodes:
            for tensor in node.inputs + node.outputs:
                add_to_tensor_map(tensor)
        return tensor_map


    def fold_constants(self):
        """
        Folds constants in-place in the graph. The graph must be topologically sorted prior to calling this function (see `toposort()`).

        NOTE: This function will not remove constants after folding them. In order to get rid of these hanging nodes, you can run the `cleanup()` function.

        NOTE: Due to how this is implemented, the graph must be exportable to ONNX, and evaluable in ONNX Runtime.

        Returns:
            self
        """
        from onnx_graphsurgeon.api.api import export_onnx
        import onnxruntime
        import onnx

        temp_graph = copy.deepcopy(self)

        # Since the graph is topologically sorted, this should find all constant nodes in the graph.
        graph_constants = {tensor.name: tensor for tensor in temp_graph.tensors().values() if isinstance(tensor, Constant)}
        for node in temp_graph.nodes:
            if all([inp.name in graph_constants for inp in node.inputs]):
                graph_constants.update({out.name: out for out in node.outputs})

        # Next build a graph with just the constants, and evaluate - no need to evaluate constants
        outputs_to_evaluate = [tensor for tensor in graph_constants.values() if isinstance(tensor, Variable)]

        if not outputs_to_evaluate:
            G_LOGGER.warning("Could not find any operations in this graph that can be folded. This could mean that constant folding has already been run on this graph. Skipping.")
            return self

        output_names = [out.name for out in outputs_to_evaluate]

        temp_graph.outputs = outputs_to_evaluate
        temp_graph.cleanup()

        # Determining types is not trivial, and ONNX-RT does its own type inference.
        sess = onnxruntime.InferenceSession(export_onnx(temp_graph, do_type_check=False).SerializeToString())
        constant_values = sess.run(output_names, {})

        # Finally, replace the Variables in the original graph with constants.
        graph_tensors = self.tensors()
        for name, values in zip(output_names, constant_values):
            graph_tensors[name].to_constant(values)
            graph_tensors[name].inputs.clear() # Constants do not need inputs

        return self


    def _generate_name(self, prefix):
        name = "{}_{}".format(prefix, self.name_idx)
        self.name_idx += 1
        return name


    def layer(self, inputs=[], outputs=[], *args, **kwargs):
        """
        Creates a node, adds it to this graph, and optionally creates its input and output tensors.

        The input and output lists can include various different types:
            - Tensor: Any Tensors provided will be used as-is in the inputs/outputs of the node created
            - str: If a string is provided, this function will generate a new tensor, using the string to generate a name. It will append an index to the end of the provided string to attempt to avoid duplicates, but since this doesn't guarantee that the name is unique, you should try to ensure that the string provided is as specific as possible.
            - np.ndarray: If a NumPy array is provided, this function will generate a Constant tensor using the prefix: "onnx_graphsurgeon_constant"

        Optional Args:
            inputs (List[Union[Tensor, str, np.array]]): The list of inputs
            outputs (List[Union[Tensor, str, np.array]]): The list of outputs
            *args and **kwargs: These are passed directly to the constructor of Node

        Returns:
            List[Tensor]: The output tensors of the node
        """
        def process_io(io):
            new_io = []
            for elem in io:
                if isinstance(elem, Tensor):
                    new_io.append(elem)
                elif isinstance(elem, str):
                    tensor = Variable(name=self._generate_name(elem))
                    new_io.append(tensor)
                elif isinstance(elem, np.ndarray):
                    new_io.append(Constant(name=self._generate_name("onnx_graphsurgeon_constant"), values=elem))
                else:
                    G_LOGGER.critical("Unrecognized type passed to Graph.layer: {:}.\n\tHint: Did you forget to unpack a list with `*`?\n\tPlease use Tensors, strings, or NumPy arrays.".format(elem))
            return new_io

        inputs = process_io(inputs)
        outputs = process_io(outputs)

        if "name" not in kwargs:
            kwargs["name"] = self._generate_name("onnx_graphsurgeon_node")

        node = Node(*args, **kwargs, inputs=inputs, outputs=outputs)
        self.nodes.append(node)
        return node.outputs


    def __deepcopy__(self, memo):
        """
        Makes a deep copy of this graph.
        """
        # First, reconstruct each tensor in the graph, but with no inputs or outputs
        tensor_map = self.tensors()
        new_tensors = {name: tensor.copy() for name, tensor in tensor_map.items()}

        # Next, copy nodes, and update inputs/outputs
        new_nodes = []
        for node in self.nodes:
            new_node = node.copy(inputs=[new_tensors[inp.name] for inp in node.inputs], outputs=[new_tensors[out.name] for out in node.outputs])
            new_nodes.append(new_node)

        new_graph_inputs = [new_tensors[inp.name] for inp in self.inputs]
        new_graph_outputs = [new_tensors[out.name] for out in self.outputs]
        return Graph(nodes=new_nodes, inputs=new_graph_inputs, outputs=new_graph_outputs, name=copy.deepcopy(self.name, memo), doc_string=copy.deepcopy(self.doc_string, memo), opset=copy.deepcopy(self.opset, memo))


    def __str__(self):
        nodes_str = "\n".join([str(node) for node in self.nodes])
        return "Graph {:} (Opset: {:})\nInputs: {:}\nNodes: {:}\nOutputs: {:}".format(self.name, self.opset, self.inputs, nodes_str, self.outputs)


    def __repr__(self):
        return self.__str__()
