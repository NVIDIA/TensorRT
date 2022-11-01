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

import copy
import numbers
from collections import OrderedDict, defaultdict
from typing import Sequence

import numpy as np
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, Tensor, Variable
from onnx_graphsurgeon.logger import G_LOGGER, LogMode
from onnx_graphsurgeon.util import misc


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
    """
    Represents a graph containing nodes and tensors.
    """

    DEFAULT_OPSET = 11
    OPSET_FUNC_MAP = defaultdict(dict)  # Ops registered for specific opsets.
    GLOBAL_FUNC_MAP = dict()  # Ops registered for ALL opsets.

    @staticmethod
    def register(opsets=None):
        """
        Registers a function with the Graph class for the specified group of opsets.
        After registering the function, it can be accessed like a normal member function.

        For example:
        ::

            @Graph.register()
            def add(self, a, b):
                return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])

            graph.add(a, b)

        Args:
            opsets (Sequence[int]):
                    A group of opsets for which to register the function. Multiple functions with the same
                    name may be registered simultaneously if they are registered for different opsets.
                    Registering a function with a duplicate name for the same opsets will overwrite any
                    function previously registered for those opsets.  By default, the function is
                    registered for all opsets.
        """

        def register_func(func):
            if hasattr(Graph, func.__name__):
                G_LOGGER.warning(
                    "Registered function: {:} is hidden by a Graph attribute or function with the same name. "
                    "This function will never be called!".format(func.__name__)
                )

            # Default behavior is to register functions for all opsets.
            if opsets is None:
                Graph.GLOBAL_FUNC_MAP[func.__name__] = func
            else:
                for opset in opsets:
                    Graph.OPSET_FUNC_MAP[opset][func.__name__] = func
            return func

        return register_func

    def __init__(
        self,
        nodes: Sequence[Node] = None,
        inputs: Sequence[Tensor] = None,
        outputs: Sequence[Tensor] = None,
        name=None,
        doc_string=None,
        opset=None,
        import_domains=None,
        producer_name: str = None,
        producer_version: str = None,
    ):
        """
        Args:
            nodes (Sequence[Node]): A list of the nodes in this graph.
            inputs (Sequence[Tensor]): A list of graph input Tensors.
            outputs (Sequence[Tensor]): A list of graph output Tensors.
            name (str): The name of the graph. Defaults to "onnx_graphsurgeon_graph".
            doc_string (str): A doc_string for the graph. Defaults to "".
            opset (int): The ONNX opset to use when exporting this graph.
            producer_name (str): The name of the tool used to generate the model. Defaults to "".
            producer_version (str): The version of the generating tool. Defaults to "".
        """
        self.nodes = misc.default_value(nodes, [])
        self.inputs = list(misc.default_value(inputs, []))
        self.outputs = list(misc.default_value(outputs, []))

        self.name = misc.default_value(name, "onnx_graphsurgeon_graph")
        self.__name__ = self.name

        self.doc_string = misc.default_value(doc_string, "")
        self.opset = misc.default_value(opset, Graph.DEFAULT_OPSET)
        self.producer_name = misc.default_value(producer_name, "")
        self.producer_version = misc.default_value(producer_version, "")
        self.import_domains = import_domains
        # Printing graphs can be very expensive
        G_LOGGER.ultra_verbose(lambda: "Created Graph: {:}".format(self))
        # For layer() function
        self.name_idx = 0

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as err:
            # Opset specific ops always take priority over global ops.
            if self.opset in Graph.OPSET_FUNC_MAP and name in Graph.OPSET_FUNC_MAP[self.opset]:
                return lambda *args, **kwargs: Graph.OPSET_FUNC_MAP[self.opset][name](self, *args, **kwargs)

            if name in Graph.GLOBAL_FUNC_MAP:
                return lambda *args, **kwargs: Graph.GLOBAL_FUNC_MAP[name](self, *args, **kwargs)

            G_LOGGER.error("No function: {:} registered for opset: {:}".format(name, self.opset))
            raise err

    def __setattr__(self, name, value):
        # We don't want graph inputs/outputs to be SynchronizedLists
        if name in ["inputs", "outputs"]:
            value = list(value)
        return super().__setattr__(name, value)

    def __eq__(self, other: "Graph"):
        nodes_match = len(self.nodes) == len(other.nodes) and all(
            [node == other_node for node, other_node in zip(self.nodes, other.nodes)]
        )
        inputs_match = len(self.inputs) == len(other.inputs) and all(
            [inp == other_inp for inp, other_inp in zip(self.inputs, other.inputs)]
        )
        outputs_match = len(self.outputs) == len(other.outputs) and all(
            [out == other_out for out, other_out in zip(self.outputs, other.outputs)]
        )
        opset_matches = self.opset == other.opset and self.import_domains == other.import_domains
        return nodes_match and inputs_match and outputs_match and opset_matches

    def node_ids(self):
        """
        Returns a context manager that supplies unique integer IDs for Nodes in the Graph.

        For example:
        ::

            with graph.node_ids():
                assert graph.nodes[0].id != graph.nodes[1].id

        Returns:
            NodeIDAdder: A context manager that supplies unique integer IDs for Nodes.
        """
        return NodeIDAdder(self)

    # Gets the node ID for a node. All internal code should use this instead of accessing `node.id` directly.
    def _get_node_id(self, node):
        try:
            return node.id
        except AttributeError:
            G_LOGGER.critical(
                "Encountered a node not in the graph:\n{:}.\n\n"
                "To fix this, please append the node to this graph's `nodes` attribute.".format(node)
            )

    # A tensor is local if it is produced in this graph, or is explicitly a graph input.
    def _local_tensors(self):
        local_tensors = {t.name: t for node in self.nodes for t in node.outputs if not t.is_empty()}
        local_tensors.update({t.name: t for t in self.inputs})
        local_tensors.update({t.name: t for t in self.tensors().values() if isinstance(t, Constant)})
        return local_tensors

    # Returns tensors used by this graph which are not present in the graph.
    # These may come from an outer graph for example.
    def _foreign_tensors(self):
        local_tensors = self._local_tensors()
        foreign_tensors = {}

        def is_foreign_tensor(tensor):
            return tensor.name not in local_tensors

        for node in self.nodes:
            foreign_tensors.update({t.name: t for t in node.inputs if is_foreign_tensor(t)})

            for attr in node.attrs.values():
                if isinstance(attr, Graph):
                    subgraph_foreign_tensors = attr._foreign_tensors()
                    # Some of the foreign tensors from a subgraph may come from this graph.
                    subgraph_foreign_tensors = {
                        t.name: t for t in subgraph_foreign_tensors.values() if is_foreign_tensor(t)
                    }
                    foreign_tensors.update(subgraph_foreign_tensors)

        return foreign_tensors

    def _get_used_node_ids(self):
        local_tensors = self._local_tensors()

        # We only want to consider tensors that are local to this graph, because we can't
        # remove external tensors (e.g. from outer graphs) anyway.
        class IgnoreDupAndForeign(object):
            def __init__(self, initial_tensors=None):
                tensors = misc.default_value(initial_tensors, [])
                self.seen_tensors = set([tensor.name for tensor in tensors])

            def __call__(self, tensor):
                # Returns True if a tensor should included,
                # False if it should be filtered out.
                if tensor.is_empty():
                    return True
                elif tensor.name not in local_tensors:
                    return False
                elif tensor.name not in self.seen_tensors:
                    self.seen_tensors.add(tensor.name)
                    return True
                return False

        # Traverse backwards from outputs to find all used nodes.
        ignore_tensors = IgnoreDupAndForeign()
        used_tensors = list(filter(ignore_tensors, self.outputs))
        used_node_ids = set()

        index = 0
        while index < len(used_tensors):
            used_tensor = used_tensors[index]
            index += 1
            for node in used_tensor.inputs:
                # Must cast to list here, otherwise node_used_tensors will be SynchronizedList!
                node_used_tensors = list(node.inputs)

                # If a node includes a subgraph, get any tensors that it uses from the outer graph.
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        node_used_tensors += list(attr._foreign_tensors().values())

                used_node_ids.add(self._get_node_id(node))
                used_tensors.extend(filter(ignore_tensors, node_used_tensors))
        return used_node_ids, used_tensors

    def cleanup(self, remove_unused_node_outputs=False, recurse_subgraphs=True, remove_unused_graph_inputs=False):
        """
        Removes unused nodes and tensors from the graph.
        A node or tensor is considered unused if it does not contribute to any of the graph outputs.

        Additionally, any producer nodes of graph input tensors, as well as consumer nodes of graph output
        tensors that are not in the graph, are removed from the graph.

        *Note: This function will never modify graph output tensors.*

        Args:
            remove_unused_node_outputs (bool): Whether to remove unused output tensors of nodes. This will never remove
                empty-tensor (i.e. optional, but omitted) outputs. Defaults to False.
            recurse_subgraphs (bool):
                    Whether to recursively cleanup subgraphs.
                    Defaults to True.
            remove_unused_graph_inputs (bool):
                    Whether to remove unused graph inputs.
                    Defaults to False.

        Returns:
            self
        """

        def cleanup_subgraphs():
            for node in self.nodes:
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        attr.cleanup(
                            remove_unused_node_outputs=remove_unused_node_outputs,
                            remove_unused_graph_inputs=remove_unused_graph_inputs,
                        )

        if recurse_subgraphs:
            cleanup_subgraphs()

        G_LOGGER.verbose("Cleaning up {:}".format(self.name))

        with self.node_ids():
            # Graph input producers must be removed first so used_node_ids is correct.
            for inp in self.inputs:
                inp.inputs.clear()

            used_node_ids, used_tensors = self._get_used_node_ids()

            inputs = []
            for inp in self.inputs:
                if inp in used_tensors or not remove_unused_graph_inputs:
                    inputs.append(inp)
                else:
                    G_LOGGER.ultra_verbose("Removing unused input: {:}".format(inp))
            self.inputs = inputs

            nodes = []
            for node in self.nodes:
                if self._get_node_id(node) in used_node_ids:
                    nodes.append(node)
                else:
                    node.inputs.clear()
                    node.outputs.clear()
                    G_LOGGER.ultra_verbose("Removing unused node: {:}".format(node))

            # Remove any hanging tensors - tensors without outputs
            if remove_unused_node_outputs:
                graph_output_names = set([tensor.name for tensor in self.outputs])
                for node in nodes:

                    def is_hanging_tensor(tensor):
                        return (
                            not tensor.is_empty() and len(tensor.outputs) == 0 and tensor.name not in graph_output_names
                        )

                    to_remove = [out for out in node.outputs if is_hanging_tensor(out)]
                    for out in to_remove:
                        if out in node.outputs:
                            node.outputs.remove(out)

            self.nodes = nodes
            return self

    def toposort(self, recurse_subgraphs=True):
        """
        Topologically sort the graph in place.

        Args:
            recurse_subgraphs (bool):
                    Whether to recursively topologically sort subgraphs.
                    Defaults to True.

        Returns:
            self
        """
        if recurse_subgraphs:
            for node in self.nodes:
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        attr.toposort()

        G_LOGGER.debug("Topologically sorting {:}".format(self.name))

        # Keeps track of a node and it's level in the graph hierarchy.
        # 0 corresponds to an input node, N corresponds to a node with N layers of inputs.
        class HierarchyDescriptor(object):
            def __init__(self, node=None, level=None):
                self.node = node
                self.level = level

            def __lt__(self, other):
                return self.level < other.level

        hierarchy_levels = {}  # Dict[int, HierarchyDescriptor]

        local_tensors = self._local_tensors()

        def get_hierarchy_level(node):
            # Return all local nodes that contribute to this node.
            def get_input_nodes(node):
                inputs = {}

                def add_local_producers(tensor):
                    nonlocal inputs
                    if tensor.name in local_tensors:
                        for inp_node in tensor.inputs:
                            inputs[self._get_node_id(inp_node)] = inp_node

                for tensor in node.inputs:
                    add_local_producers(tensor)

                # If a node includes a subgraph, get any tensors that it uses from the outer graph.
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        for tensor in attr._foreign_tensors().values():
                            add_local_producers(tensor)

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
        Creates a tensor map of all the tensors used by this graph by walking over all nodes. Empty tensors are omitted from this map.

        Tensors are guaranteed to be in order of the nodes in the graph. Hence, if the graph is topologically sorted, the tensor map will be too.

        Args:
            check_duplicates (bool): Whether to fail if multiple tensors with the same name are encountered.

        Raises:
            OnnxGraphSurgeonException: If check_duplicates is True and multiple distinct tensors in the graph share the same name.

        Returns:
            OrderedDict[str, Tensor]: A mapping of tensor names to tensors.
        """
        tensor_map = OrderedDict()

        def add_to_tensor_map(tensor):
            if not tensor.is_empty():
                if tensor.name in tensor_map and not (tensor_map[tensor.name] is tensor):
                    msg = "Found distinct tensors that share the same name:\n[id: {:}] {:}\n[id: {:}] {:}\n".format(
                        id(tensor_map[tensor.name]),
                        tensor_map[tensor.name],
                        id(tensor),
                        tensor,
                    )
                    msg += (
                        "Note: Producer node(s) of first tensor:\n{:}\nProducer node(s) of second tensor:\n{:}".format(
                            tensor_map[tensor.name].inputs,
                            tensor.inputs,
                        )
                    )

                    if check_duplicates:
                        G_LOGGER.critical(msg)
                    G_LOGGER.warning(msg)

                tensor_map[tensor.name] = tensor

        # I/O tensors may not be attached to nodes.
        for io_tensor in self.inputs:
            add_to_tensor_map(io_tensor)

        for node in self.nodes:
            for tensor in node.inputs + node.outputs:
                add_to_tensor_map(tensor)

        for io_tensor in self.outputs:
            add_to_tensor_map(io_tensor)

        return tensor_map

    def fold_constants(
        self,
        fold_shapes=True,
        recurse_subgraphs=True,
        partitioning=None,
        error_ok=True,
        flatten_subgraphs=True,
        size_threshold=None,
        should_exclude_node=None,
    ):
        """
        Folds constants in-place in the graph. The graph must be topologically sorted prior to
        calling this function (see `toposort()`).

        This function will not remove constants after folding them. In order to get rid of
        these hanging nodes, you can run the `cleanup()` function.

        *Note: Due to how this function is implemented, the graph must be exportable to ONNX,
        and evaluable in ONNX-Runtime. Additionally, ONNX-Runtime must be installed.*

        Args:
            fold_shapes (bool):
                    Whether to fold `Shape` nodes in the graph.
                    This requires shapes to be inferred in the graph, and can only fold
                    static shapes.
                    Defaults to True.
            recurse_subgraphs (bool):
                    Whether to recursively fold constants in subgraphs.
                    Defaults to True.
            partitioning (Union[str, None]):
                    Whether/How to partition the graph so that errors in folding one
                    part of a model do not affect other parts. Available modes are:

                    - None: Do not partition the graph. If inference fails, no constants are folded.
                    - "basic": Partition the graph. If inference fails in one partition, other partitions will
                            remain unaffected.
                    - "recursive": Parition the graph recursively. If inference fails in a partition, the partition
                            will be further paritioned.

                    Defaults to None.
            error_ok (bool):
                    Whether inference errors should be suppressed.
                    When this is False, any errors encountered during inference will be re-raised.
                    Defaults to True.
            flatten_subgraphs (bool):
                    Whether to flatten subgraphs where possible. For example, `If` nodes with a constant condition
                    can be flattened into the parent graph.
            size_threshold (int):
                    The maximum size threshold, in bytes, for which to fold constants.
                    Any tensors larger than this value will not be folded.
                    Set to ``None`` to disable the size threshold and always fold constants.
                    For example, some models may apply ops like `Tile` or `Expand` to constants, which can
                    result in very large tensors. Rather than pre-computing those constants and bloating
                    the model size, it may be desirable to skip folding them and allow them to be computed
                    at runtime.
                    Defaults to None.
            should_exclude_node (Callable[[gs.Node], bool]):
                    A callable that accepts an onnx-graphsurgeon node from the graph and reports whether it should
                    be excluded from folding. This is only called for nodes which are otherwise foldable.
                    Note that preventing a node from being folded also prevents its consumers from being folded.
                    Defaults to a callable that always returns False.

        Returns:
            self
        """
        from onnx_graphsurgeon.exporters.onnx_exporter import dtype_to_onnx, export_onnx

        should_exclude_node = misc.default_value(should_exclude_node, lambda node: False)

        PARTITIONING_MODES = [None, "basic", "recursive"]
        if partitioning not in PARTITIONING_MODES:
            G_LOGGER.critical("Argument for parameter 'partitioning' must be one of: {:}".format(PARTITIONING_MODES))
        ORT_PROVIDERS = ["CPUExecutionProvider"]

        G_LOGGER.debug("Folding constants in {:}".format(self.name))

        # We apply constant folding in 5 passes:
        # Pass 1 lowers 'Constant' nodes into Constant tensors.
        # Pass 2 elides casts applied to shape tensors. This is done separately from other shape folding
        #   since it operates on the original graph rather than a clone.
        # Pass 3 finds all Constant tensors in the graph, then finds all descendants which are dependent
        #   only on constants.
        # Pass 4 searches for Shape nodes that have variable inputs (i.e. not marked const in pass 1)
        #    and turns them into Constants iff the input has a statically known shape.
        # Pass 5 computes the descendants determined in Pass 3 using ONNX-Runtime and replaces them in the graph.

        # Pass 1: Lower constant nodes
        for tensor in self.tensors().values():
            if len(tensor.inputs) == 1:
                node = tensor.inputs[0]
                if node.op == "Constant":
                    tensor.to_constant(node.attrs["value"]._values)  # Using ._values avoids copying
                    tensor.inputs.clear()

        # Pass 2: Run shape-tensor cast elision
        def run_cast_elision(node):
            import onnx

            # Search for Cast(s) (from int -> float) -> intermediate operator (with float constants) -> Cast(s) (back to int)
            # This pattern is problematic for TensorRT since these operations may be performed on Shape Tensors, which
            # are not allowed to be floating point type. Attempt to fold the pattern here
            VALID_CAST_ELISION_OPS = ["Add", "Sub", "Mul", "Div", "Max", "Min", "Equal", "Greater", "Less", "Concat"]

            if node.op not in VALID_CAST_ELISION_OPS:
                return

            # If the uncasted outputs of this node have any consumers other than "Cast" nodes,
            # then we cannot elide the cast.
            for out_tensor in node.outputs:
                if out_tensor in self.outputs:
                    return

                if any(out_node.op != "Cast" for out_node in out_tensor.outputs):
                    return

            # Get list of input nodes that cast to float32
            inp_casts = [
                inp_node
                for inp_tensor in node.inputs
                for inp_node in inp_tensor.inputs
                if inp_node.op == "Cast" and inp_node.attrs["to"] == onnx.TensorProto.DataType.FLOAT
            ]

            # No cast nodes found, return early
            if not inp_casts:
                return

            # Ensure that all input cast nodes are casting from the same type
            inp_dtypes = [dtype_to_onnx(inp_cast.inputs[0].dtype) for inp_cast in inp_casts]
            if len(set(inp_dtypes)) != 1:
                return

            final_type = inp_dtypes[0]

            # Get list of output nodes that cast to int32 or int64
            out_casts = [
                out_node
                for out_tensor in node.outputs
                for out_node in out_tensor.outputs
                if out_node.op == "Cast"
                and out_node.attrs["to"] in [onnx.TensorProto.DataType.INT32, onnx.TensorProto.DataType.INT64]
            ]

            # No cast node found on outputs, return early
            if not out_casts:
                return

            # Ensure that all output cast nodes are casting to the same type and that this
            # matches the original type before the inputs were casted.
            out_dtypes = [out_cast.attrs["to"] for out_cast in out_casts]
            if len(set(out_dtypes)) != 1 or out_dtypes[0] != final_type:
                return

            # If all checks passed, reconnect inputs/outputs to the consumers/producers
            # of the Cast nodes.
            # Note that we need to be careful in how we rebind tensors since they may
            # be used by multiple nodes. Thus, it is not necessarily safe to assume that
            # `cast_node.inputs[0].outputs[0] == cast_node`.
            for index, inp in enumerate(node.inputs):
                if isinstance(inp, Constant):
                    inp.values = inp.values.astype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[final_type])

                for cast in inp_casts:
                    if cast.outputs[0] == inp:
                        node.inputs[index] = cast.inputs[0]

            for index, out in enumerate(node.outputs):
                for cast in out_casts:
                    if cast.inputs[0] == out:
                        out_tensor = cast.outputs[0]
                        out_tensor.inputs.clear()  # Disconnect from Cast
                        node.outputs[index] = out_tensor

        if fold_shapes:
            # Perform shape tensor cast elision prior to most other folding
            G_LOGGER.debug("Performing shape tensor cast elision in {:}".format(self.name))
            try:
                with self.node_ids():
                    for node in self.nodes:
                        run_cast_elision(node)
            except Exception as err:
                if not error_ok:
                    raise err
                G_LOGGER.warning("'{:}' routine failed with: {:}".format("Shape tensor cast elision", err))

        # Note that most of the remaining passes operate on a clone of the original graph.
        # Pass 3: Find all descendants of constant tensors

        graph_clone = self.copy()
        clone_tensors = graph_clone.tensors()

        def update_foldable_outputs(graph_constants):
            def is_foldable(node):
                NO_FOLD_OPS = ["QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"]
                if node.op in NO_FOLD_OPS:
                    return False

                def all_tensors_const(tensors):
                    return all([t.name in graph_constants for t in tensors])

                if not all_tensors_const(node.inputs):
                    return False

                all_subgraph_foreign_tensors_const = True
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        foreign_tensors = attr._foreign_tensors().values()
                        all_subgraph_foreign_tensors_const &= all_tensors_const(foreign_tensors)

                return all_subgraph_foreign_tensors_const and not should_exclude_node(node)

            # Walks along the outputs of graph_constants to see if they can also be computed statically.
            # Since the graph is topologically sorted, this should find all constant nodes in the graph.
            for node in graph_clone.nodes:
                if is_foldable(node):
                    graph_constants.update({out.name: out for out in node.outputs})
            return graph_constants

        graph_constants = {name: tensor for name, tensor in clone_tensors.items() if isinstance(tensor, Constant)}
        graph_constants = update_foldable_outputs(graph_constants)

        # Pass 4: Shape Folding

        def get_producer(tensor, op):
            """
            Get the producer of the specified tensor iff it matches op
            """
            if len(tensor.inputs) != 1:
                return None

            node = tensor.inputs[0]
            if node.op != op:
                return None
            return node

        def get_input(node, index=0):
            """
            Get the input tensor of a node iff the input tensor is not already marked a graph constant.
            """
            if node is None:
                return None

            inp = node.inputs[index]

            # If the input was already found to be a constant, it will be folded anyway.
            if inp.name in graph_constants:
                return None

            return inp

        def get_scalar_value(tensor):
            """
            Gets the scalar value of a constant tensor with a single item
            """
            if not tensor.shape:
                return tensor.values
            else:
                return list(tensor.values)[0]

        def fold_shape(tensor):
            inp = get_input(get_producer(tensor, "Shape"))
            if inp is None:
                return None

            if inp.shape is None or misc.is_dynamic_shape(inp.shape):
                return None
            return np.array(inp.shape, dtype=np.int64)

        def fold_shape_gather(tensor):
            gather = get_producer(tensor, "Gather")
            if gather is None:
                return None

            data = gather.inputs[0]
            indices_tensor = gather.inputs[1]

            inp = get_input(get_producer(data, "Shape"))
            if inp is None or inp.shape is None:
                return None

            if not isinstance(indices_tensor, Constant):
                return None

            indices = indices_tensor.values
            if not indices.shape:  # Scalar-case
                shape = inp.shape[int(indices)]
                if misc.is_dynamic_dimension(shape):
                    return None
            else:
                shape = [inp.shape[index] for index in indices]
                if misc.is_dynamic_shape(shape):
                    return None

            return np.array(shape, dtype=np.int64)

        def fold_shape_slice(tensor):
            slice = get_producer(tensor, "Slice")
            if slice is None:
                return None

            data = slice.inputs[0]

            if len(slice.inputs) >= 3:
                starts, ends = slice.inputs[1:3]
                if any(not isinstance(t, Constant) for t in [starts, ends]):
                    return None
                starts, ends = get_scalar_value(starts), get_scalar_value(ends)
            elif "starts" in slice.attrs and "ends" in slice.attrs:
                starts, ends = slice.attrs["starts"][0], slice.attrs["ends"][0]
            else:
                return None

            inp = get_input(get_producer(data, "Shape"))
            if inp is None or inp.shape is None:
                return None

            # For shape tensors, we can only slice on the 0th dimension.
            if len(slice.inputs) > 3:
                axes = slice.inputs[3]
                if not isinstance(axes, Constant):
                    return None

                if get_scalar_value(axes) != 0:
                    return None
            elif "axes" in slice.attrs:
                if slice.attrs["axes"][0] != 0:
                    return None

            steps = 1
            if len(slice.inputs) > 4:
                steps = slice.inputs[4]
                if not isinstance(steps, Constant):
                    return None
                steps = get_scalar_value(steps)
            elif "steps" in slice.attrs:
                steps = slice.attrs["steps"][0]

            shape = inp.shape[starts:ends:steps]
            if misc.is_dynamic_shape(shape):
                return None

            return np.array(shape, dtype=np.int64)

        if fold_shapes:
            # NOTE: The order of shape folding passes is important to maximize how much we fold (phase-ordering problem).
            SHAPE_FOLD_FUNCS = [fold_shape_gather, fold_shape_slice, fold_shape]
            for shape_fold_func in SHAPE_FOLD_FUNCS:
                try:
                    for tensor in clone_tensors.values():
                        shape_of = shape_fold_func(tensor)

                        if shape_of is not None:
                            G_LOGGER.ultra_verbose("Folding shape tensor: {:} to: {:}".format(tensor.name, shape_of))
                            graph_constants[tensor.name] = tensor.to_constant(shape_of)
                            graph_constants[tensor.name].inputs.clear()
                except Exception as err:
                    if not error_ok:
                        raise err
                    G_LOGGER.warning("'{:}' routine failed with:\n{:}".format(shape_fold_func.__name__, err))
                else:
                    graph_constants = update_foldable_outputs(graph_constants)

        # Pass 5: Evaluate all tensors descended from constants with ONNX-Runtime and replace them with constant values.

        def partition_and_infer(subgraph):
            def get_out_node_ids():
                # Gets the final output nodes - producer nodes of graph output tensors without other outputs.
                with subgraph.node_ids():
                    out_node_ids = set()
                    for out in subgraph.outputs:
                        if not out.outputs and not isinstance(out, Constant):
                            for n_inp in out.inputs:
                                out_node_ids.add(subgraph._get_node_id(n_inp))
                return out_node_ids

            # Compute each output node in a separate subgraph.
            out_node_ids = get_out_node_ids()
            constant_values = {}

            for index in out_node_ids:  # Have to use index since 'node' is not in part
                part = subgraph.copy()
                out_node = part.nodes[index]
                part.outputs = out_node.outputs
                part.name = "Folding: {:}".format([out.name for out in part.outputs])
                part.cleanup(remove_unused_graph_inputs=True)
                names = [out.name for out in part.outputs]

                try:
                    # Determining types is not trivial, and ONNX-RT does its own type inference.
                    import onnxruntime as onnxrt

                    sess = onnxrt.InferenceSession(
                        export_onnx(part, do_type_check=False).SerializeToString(), providers=ORT_PROVIDERS
                    )
                    values = sess.run(names, {})
                except Exception as err:
                    G_LOGGER.warning("Inference failed for subgraph: {:}. Note: Error was:\n{:}".format(part.name, err))
                    if partitioning == "recursive":
                        G_LOGGER.verbose("Attempting to recursively partition subgraph")
                        # Partition failed, peel off last node.
                        # We only need to remove one node, so avoid doing an expensive call to cleanup()
                        part.outputs = out_node.inputs
                        del part.nodes[part.nodes.index(out_node)]
                        out_node.outputs.clear()
                        out_node.inputs.clear()
                    else:
                        G_LOGGER.info("You may see better results if you set partitioning='recursive'")
                        if not error_ok:
                            raise err

                    constant_values.update(partition_and_infer(part))
                else:
                    constant_values.update({name: val for name, val in zip(names, values)})

            return constant_values

        # Only evaluate foldable values that have non-foldable outputs or are graph outputs.
        # Otherwise, if all the outputs are foldable, then we can just evaluate the outputs directly.
        # Additionally, if we can determine tensor size, do not evaluate tensors whose sizes exceed the size threshold.
        def should_eval_foldable(tensor):
            non_const = not isinstance(tensor, Constant)
            is_graph_output = not tensor.outputs
            has_non_foldable_outputs = any(out.name not in graph_constants for out in tensor.outputs)
            exceeds_size_threshold = (
                tensor.shape is not None
                and not misc.is_dynamic_shape(tensor.shape)
                and tensor.dtype is not None
                and size_threshold is not None
            ) and (misc.volume(tensor.shape) * np.dtype(tensor.dtype).itemsize > size_threshold)

            return non_const and (is_graph_output or has_non_foldable_outputs) and not exceeds_size_threshold

        graph_clone.outputs = [t for t in graph_constants.values() if should_eval_foldable(t)]
        G_LOGGER.debug("Folding tensors: {:}".format(graph_clone.outputs))
        graph_clone.cleanup(remove_unused_graph_inputs=True)

        # Using ._values avoids a deep copy of the values.
        constant_values = {
            name: tensor._values for name, tensor in graph_constants.items() if isinstance(tensor, Constant)
        }
        if graph_clone.outputs:
            if partitioning:
                constant_values.update(partition_and_infer(graph_clone))
            else:
                names = [t.name for t in graph_clone.outputs]
                try:
                    import onnxruntime as onnxrt

                    sess = onnxrt.InferenceSession(
                        export_onnx(graph_clone, do_type_check=False).SerializeToString(), providers=ORT_PROVIDERS
                    )
                    values = sess.run(names, {})
                    constant_values.update({name: val for name, val in zip(names, values)})
                except Exception as err:
                    G_LOGGER.warning(
                        "Inference failed. You may want to try enabling partitioning to see better results. "
                        "Note: Error was:\n{:}".format(err)
                    )
                    G_LOGGER.verbose("Note: Graph was:\n{:}".format(graph_clone))
                    if not error_ok:
                        raise
        elif not constant_values:
            G_LOGGER.debug(
                "Could not find any nodes in this graph ({:}) that can be folded. "
                "This could mean that constant folding has already been run on this graph. "
                "Skipping.".format(self.name)
            )

        # Finally, replace the Variables in the original graph with constants.
        large_tensors = {}
        if constant_values:
            graph_tensors = self.tensors()
            for name, values in constant_values.items():
                tensor = graph_tensors[name]
                if isinstance(tensor, Constant):
                    # No need to fold tensors that are already constant.
                    continue

                if size_threshold is not None and values.nbytes > size_threshold:
                    G_LOGGER.debug(
                        "Will not fold: '{:}' since its size in bytes ({:}) exceeds the size threshold ({:})".format(
                            name, values.nbytes, size_threshold
                        )
                    )
                    continue
                elif size_threshold is None and values.nbytes > (1 << 20):
                    large_tensors[name] = values.nbytes

                tensor.to_constant(values)
                tensor.inputs.clear()  # Constants do not need inputs

            if large_tensors:
                large_tensors_mib = {
                    tensor_name: "{:} MiB".format(value // (1 << 20)) for tensor_name, value in large_tensors.items()
                }
                G_LOGGER.warning(
                    "It looks like this model contains foldable nodes that produce large outputs.\n"
                    "In order to avoid bloating the model, you may want to set a constant-folding size threshold.\n"
                    "Note: Large tensors and their corresponding sizes were: {:}".format(large_tensors_mib),
                    mode=LogMode.ONCE,
                )

        # Folding subgraphs after the outer graph can lead to better folding.
        def fold_subgraphs():
            for node in self.nodes:
                for attr in node.attrs.values():
                    if isinstance(attr, Graph):
                        attr.fold_constants(
                            fold_shapes=fold_shapes,
                            recurse_subgraphs=recurse_subgraphs,
                            partitioning=partitioning,
                            error_ok=error_ok,
                            flatten_subgraphs=flatten_subgraphs,
                            size_threshold=size_threshold,
                        )

        if recurse_subgraphs:
            fold_subgraphs()

        if flatten_subgraphs:
            # Flatten conditional subgraphs
            index = 0
            while index < len(self.nodes):
                node = self.nodes[index]
                if node.op == "If" and isinstance(node.inputs[0], Constant):
                    G_LOGGER.debug("Flattening conditional: {:}".format(node))
                    cond = get_scalar_value(node.inputs[0])
                    subgraph = node.attrs["then_branch"] if cond else node.attrs["else_branch"]
                    # Need to add a suffix to subgraph tensors so they don't collide with outer graph tensors
                    for tensor in subgraph._local_tensors().values():
                        tensor.name += "_subg_{:}_{:}".format(index, subgraph.name)

                    # The subgraph outputs correspond to the If node outputs. Only the latter are visible
                    # in the parent graph, so we rebind the producer nodes of the subgraph outputs to point
                    # to the output tensors of the If instead.
                    for node_out, subgraph_out in zip(node.outputs, subgraph.outputs):
                        node_out.inputs.clear()
                        for producer in subgraph_out.inputs:
                            for tensor_idx, out_tensor in enumerate(producer.outputs):
                                if out_tensor == subgraph_out:
                                    producer.outputs[tensor_idx] = node_out

                    # Copy subgraph nodes into parent graph at the index of the If.
                    del self.nodes[index]
                    self.nodes[index:index] = subgraph.nodes
                    index += len(subgraph.nodes) - 1

                index += 1

        return self

    def _generate_name(self, prefix):
        name = "{}_{}".format(prefix, self.name_idx)
        self.name_idx += 1
        return name

    def layer(self, inputs=[], outputs=[], *args, **kwargs):
        """
        Creates a node, adds it to this graph, and optionally creates its input and output tensors.

        The input and output lists can include various different types:

            - ``Tensor``: Any Tensors provided will be used as-is in the inputs/outputs of the node created.
            - ``str``:
                    If a string is provided, this function will generate a new tensor using
                    the string to generate a name. It will append an index to the end of the provided string
                    to attempt to avoid duplicate tensor names, but since this doesn't guarantee that the name will
                    be unique, you should try to ensure that the string provided is as unique as possible.
                    To avoid problems with duplicate names, you can generate names yourself and provide ``Tensor`` s.
            - ``numpy.ndarray``:
                    If a NumPy array is provided, this function will generate a Constant tensor
                    using the name prefix: "onnx_graphsurgeon_constant"
            - ``Union[List[Number], Tuple[Number]]``:
                    If a list or tuple of numbers (int or float) is provided, this function will
                    generate a Constant tensor using the name prefix: "onnx_graphsurgeon_lst_constant".
                    The values of the tensor will be a 1D array containing the specified values.
                    The datatype will be either `np.float32` or `np.int64`.

        Args:
            inputs (List[Union[Tensor, str, numpy.ndarray]]): The list of inputs
            outputs (List[Union[Tensor, str, numpy.ndarray]]): The list of outputs
            args/kwargs: These are passed directly to the constructor of Node

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
                elif isinstance(elem, list) or isinstance(elem, tuple) or isinstance(elem, numbers.Number):
                    if isinstance(elem, list) or isinstance(elem, tuple):
                        dtype = np.float32 if any([isinstance(x, float) for x in elem]) else np.int64
                    else:
                        dtype = np.float32 if isinstance(elem, float) else np.int64
                    arr = np.array(elem, dtype=dtype)
                    new_io.append(Constant(name=self._generate_name("onnx_graphsurgeon_lst_constant"), values=arr))
                else:
                    G_LOGGER.critical(
                        "Unrecognized type passed to Graph.layer: {:}.\n"
                        "\tHint: Did you forget to unpack a list with `*`?\n"
                        "\tPlease use Tensors, strings, or NumPy arrays.".format(elem)
                    )
            return new_io

        inputs = process_io(inputs)
        outputs = process_io(outputs)

        if "name" not in kwargs:
            kwargs["name"] = self._generate_name("onnx_graphsurgeon_node")

        node = Node(*args, **kwargs, inputs=inputs, outputs=outputs)
        self.nodes.append(node)
        return node.outputs

    def copy(self, tensor_map: "OrderedDict[str, Tensor]" = None):
        """
        Copy the graph.

        This makes copies of all nodes and tensors in the graph, but will not
        do a deep-copy of weights or attributes (with the exception of ``Graph``
        attributes, which will be copied using their ``copy`` method).

        Args:
            tensor_map (OrderedDict[str, Tensor]):
                A mapping of tensor names to tensors from the outer graph.
                This should be ``None`` if this is the outer-most graph.

        Returns:
            Graph: A copy of the graph.
        """
        # First, reconstruct each tensor in the graph, but with no inputs or outputs
        tensor_map = copy.copy(misc.default_value(tensor_map, {}))

        local_tensor_copies = {}
        # When we're cloning a subgraph by itself, we need to use `tensors()` to get all
        # required tensors - even those produced by outer graphs.
        local_tensor_copies.update({n: t.copy() for n, t in self.tensors().items()})
        # However, we should prioritize copies already made by the outer graph.
        local_tensor_copies.update(tensor_map)
        # And locally produced tensors should take precedence over everything else.
        local_tensor_copies.update({n: t.copy() for n, t in self._local_tensors().items()})

        def get_tensor(name):
            if not name:
                return Variable.empty()
            return local_tensor_copies[name]

        # Next, copy nodes, and update inputs/outputs
        new_nodes = []
        for node in self.nodes:
            new_node = node.copy(
                inputs=[get_tensor(inp.name) for inp in node.inputs],
                outputs=[get_tensor(out.name) for out in node.outputs],
                tensor_map=local_tensor_copies,
            )
            new_nodes.append(new_node)

        new_graph_inputs = [get_tensor(inp.name) for inp in self.inputs]
        new_graph_outputs = [get_tensor(out.name) for out in self.outputs]
        return Graph(
            nodes=new_nodes,
            inputs=new_graph_inputs,
            outputs=new_graph_outputs,
            name=copy.copy(self.name),
            doc_string=copy.copy(self.doc_string),
            opset=copy.copy(self.opset),
            import_domains=self.import_domains,
        )

    def __str__(self):
        nodes_str = "\n".join([str(node) for node in self.nodes])
        return "Graph {:} (Opset: {:})\nInputs: {:}\nNodes:\n{:}\nOutputs: {:}".format(
            self.name, self.opset, self.inputs, nodes_str, self.outputs
        )

    def __repr__(self):
        return self.__str__()
