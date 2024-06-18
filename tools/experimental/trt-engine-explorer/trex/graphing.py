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

"""
This file contains code to generate graph diagrams for an engine-plan.
"""

__all__ = [
    "to_dot",
    "PlanGraph",
    "DotGraph",
    "OnnxGraph",
    "render_dot",
    "latency_types",
    "layer_precision_formatter",
    "layer_type_formatter",
    "layer_node_formatters",
    "layer_node_renderers"]

import os
import warnings
from enum import Enum
from graphviz import Digraph
from typing import Callable, NamedTuple, List, Dict
from dataclasses import dataclass, field
from trex.engine_plan import EnginePlan
from trex.layer import Layer
from trex.activations import Activation
from trex.colors import precision_colormap, layer_colormap


class PortDesc(NamedTuple):
    """A port identifies a specific layer input or output."""
    layer_name: str
    port: int


class Edge(NamedTuple):
    """An edge in a PlanGraph graph.

    The src and dst are names of engine layers or regions.
    """
    src: PortDesc
    dst: PortDesc
    tensor: Activation
    region_gen: int


@dataclass
class RegionGeneration:
    """Represents one generation of a memory Region.

    Regions represent activation memory buffers or graph inputs and outputs.
    Each generation represents a consumable state of the memory region.
    """
    tensor: Activation
    id: int
    is_user: bool = False
    is_forked: bool = False
    writers: List[PortDesc] = field(default_factory=list)
    readers: List[PortDesc] = field(default_factory=list)


class Region:
    """Represents a memory region.

    Regions represent activation memory buffers or graph inputs and outputs.
    Each generation represents a consumable state of the memory region.
    """
    def __init__(self):
        self.__generations: List[RegionGeneration] = list()
        self.name: str = None

    @property
    def generations(self):
        return self.__generations

    def add_generation(self, tensor: Activation):
        self.name = self.name or tensor.name
        gen_id = len(self.generations)
        self.__generations.append(
            RegionGeneration(tensor, gen_id))

    def nb_generations(self):
        return len(self.__generations)

    def __update_shape(self, tensor: Activation, generation: int):
        if tensor.size_bytes > self.__generations[generation].tensor.size_bytes:
                self.__generations[generation].tensor = tensor

    def add_writer(self, gen_id: int, writer: PortDesc, tensor: Activation):
        self.__generations[gen_id].writers.append(writer)
        self.__update_shape(tensor, gen_id)

    def add_reader(self, gen_id: int, reader: PortDesc, tensor: Activation):
        self.__generations[gen_id].readers.append(reader)
        self.__update_shape(tensor, gen_id)
        if gen_id > 0:
            self.__update_shape(tensor, gen_id-1)

    def writers(self, gen_id: int=None):
        try:
            if gen_id is not None:
                return self.__generations[gen_id].writers
            # Return a list of writers from all generations
            writers = []
            for generation in self.__generations:
                writers.extend(generation.writers)
            return writers
        except KeyError:
            return []

    def readers(self, gen_id: int=None):
        try:
            if gen_id is not None:
                return self.__generations[gen_id].readers
            # Return a list of readers from all generations
            readers = []
            for generation in self.__generations:
                readers.extend(generation.readers)
            return readers
        except KeyError:
            return []

    def is_placeholder(self) -> bool:
        return len(self.generations) > 1


def regions_factory(plan: EnginePlan) -> List[Region]:
    """This method processes an EnginePlan instance to create a Region dictionary"""
    class RegionMemOp(Enum):
        """
        A WRITE event represents some layer writing to a memory region.
        A READ event represents some layer reading from a memory region.
        """
        WRITE = 1
        READ = 2
    class RegionEvent(NamedTuple):
        """A region read/write event performed by a layer"""
        owner_layer: str
        # input port for WRITE event; output port for READ event.
        port: int
        mem_op: RegionMemOp
        tensor: Activation

    def set_is_user(region: Region):
        n_generations = len(region.generations)
        for gen_id, generation in enumerate(region.generations):
            nb_writers = len(generation.writers)
            nb_readers = len(generation.readers)
            is_binding = region_name in plan.bindings
            if gen_id == n_generations - 1:
                # If marked as a network binding and this is last generation, we always display.
                generation.is_user = is_binding
            else:
                generation.is_user = is_binding and (nb_readers==0 or nb_writers==0)

    def set_is_forked(region: Region):
        for generation in region.generations:
            nb_writers = len(generation.writers)
            nb_readers = len(generation.readers)
            generation.is_forked = nb_writers > 1 or nb_readers>1

    # A region_story is a map from region name to the list of read/write
    # events on that region.
    story = {}
    for layer in plan.layers:
        for port, inp in enumerate(layer.inputs):
            story.setdefault(inp.name, []).append(
                RegionEvent(layer.name, port, RegionMemOp.READ, inp))
        for port, outp in enumerate(layer.outputs):
            story.setdefault(outp.name, []).append(
                RegionEvent(layer.name, port, RegionMemOp.WRITE, outp))

    # Create region generations: each generation is a list of read/write events.
    # Every time there's a switch between read events and write events we
    # create a new RegionGeneration.
    regions: List[Region] = []
    for region_name, region_evts in story.items():
        region = Region()
        current_gen = -1
        previous_mem_op = None
        for evt in region_evts:
            if evt.mem_op != previous_mem_op:
                if evt.mem_op == RegionMemOp.WRITE or not previous_mem_op:
                    # A new region generation.
                    current_gen += 1
                    region.add_generation(evt.tensor)
            evt_layer = PortDesc(evt.owner_layer, evt.port)
            if evt.mem_op == RegionMemOp.WRITE:
                region.add_writer(current_gen, evt_layer, evt.tensor)
            else:
                region.add_reader(current_gen, evt_layer, evt.tensor)
            previous_mem_op = evt.mem_op
        set_is_user(region)
        set_is_forked(region)
        regions.append(region)
    return regions


def _make_memory_node_name(region: Region, generation: RegionGeneration) -> str:
    if generation.is_user:
        return region.name
    return ".".join((region.name, str(generation.id)))


class LayerNode(object):
    """A graph node representing an engine layer."""
    def __init__(self, layer: Layer):
        self.layer: Layer = layer


class MemoryNode(NamedTuple):
    """MemoryNode instances represent activation memory buffers, or graph inputs/outputs.

    Most (memory) Regions have only a single generation and they can be considered
    as simple activation buffers. These are not added to the PlanGraph unless
    `include_regions` is true.
    When a Region has several generations, we unravel the Region and each generation
    is represented by one MemoryNode. This is done in order to make the
    graph easier to understand and to remove circular edges.
    """
    name: str
    tensor: Activation
    region_gen: int
    is_user: bool


class PlanGraph(object):
    """View of a TensorRT plan as a data-dependencies graph.

    A PlanGraph describes the plan layers' data-dependencies. It is represented
    using two types of nodes and an edge list.

    layer_nodes represent engine execution layers.
    memory_nodes represent memory buffers.
    edges_list is a list of edges that have either LayerNode or MemoryNode endpoints.
    """
    def __init__(self,
        plan: EnginePlan,
        include_regions: bool=False,
        include_constants: bool=False,
        include_forking_regions: bool=False,
    ):
        self.include_regions = include_regions
        self.include_constants = include_constants
        self.include_forking_regions = include_forking_regions
        self.regions = regions_factory(plan)

        # These lists will be populated by __create_graph
        self._edges_list: List[Edge] = []
        self._layer_nodes: List[Layer] = []
        self._memory_nodes: List[MemoryNode] = []
        self.__create_graph(plan)

    # PlanGraph interface
    @property
    def edges_list(self): return self._edges_list

    @property
    def layer_nodes(self): return self._layer_nodes

    @property
    def memory_nodes(self): return self._memory_nodes

    def __create_graph(self, plan: EnginePlan):
        self.__add_layer_nodes(plan)
        self.__add_memory_nodes(plan)
        self.__add_inter_region_edges()

    def __add_layer_nodes(self, plan):
        self._layer_nodes = [LayerNode(layer) for layer in plan.all_layers]

    def __add_constant_node(self, region: Region, generation: RegionGeneration, constants_producers):
        assert not generation.writers

        is_user = generation.is_user
        activation_name = _make_memory_node_name(region, generation)
        constant = constants_producers[activation_name]
        if self.include_regions:
            self._edges_list.append(Edge(
                PortDesc(constant.name, 0),
                PortDesc(activation_name, 0),
                generation.tensor,
                 generation.id))
            self.__add_egress_edges(region, generation)
            # Add a memory node (represents a region generation that we chose to display)
            self._memory_nodes.append(
                MemoryNode(activation_name, generation.tensor, generation.id, is_user))
        else:
            self.__connect_writer_to_all_readers(PortDesc(constant.name, 0), generation)

    def __add_memory_nodes(self, plan):
        constants_outputs = [const.outputs[0].name for const in plan.constants]
        constants_producers = {const.outputs[0].name + ".0": const for const in plan.constants}
        for region in self.regions:
            is_myelin_const = len(region.writers()) == 0
            is_constant = region.name in constants_outputs
            if (is_constant or is_myelin_const) and not self.include_constants:
                continue
            for generation in region.generations:
                include_region = self.should_include_region(region, generation, is_constant)
                if not include_region:
                    # Bypass a region and connect the region writers to the region readers.
                    self.__add_region_bypass_edges(generation)
                    continue
                if is_constant:
                    self.__add_constant_node(region, generation, constants_producers)
                else:
                    self.__add_memory_node(region, generation)

    def __add_memory_node(self, region, generation):
        # Add an activation node (represents a region generation that we chose to display)
        is_user = generation.is_user
        node_name = _make_memory_node_name(region, generation)
        self._memory_nodes.append(MemoryNode(node_name, generation.tensor, generation.id, is_user))
        self.__add_ingress_edges(region, generation)
        self.__add_egress_edges(region, generation)

    def __add_region_bypass_edges(self, generation: RegionGeneration):
        for writer in generation.writers:
            writer_port = None if generation.is_user else writer.port
            writer_desc = PortDesc(writer.layer_name, writer_port)
            self.__connect_writer_to_all_readers(writer_desc, generation)

    def __add_ingress_edges(self, region: Region, generation: RegionGeneration):
        node_name = _make_memory_node_name(region, generation)
        node_port = None if generation.is_user else 0
        for writer in generation.writers:
            self._edges_list.append(Edge(
                PortDesc(writer.layer_name, writer.port),
                PortDesc(node_name, node_port),
                generation.tensor,
                generation.id))

    def __add_egress_edges(self, region: Region, generation: RegionGeneration):
        activation_name = _make_memory_node_name(region, generation)
        activation_port = None if generation.is_user else 0
        self.__connect_writer_to_all_readers(
            PortDesc(activation_name, activation_port),
            generation)

    def __connect_writer_to_all_readers(
        self, writer_desc: PortDesc, generation: RegionGeneration):
        for reader in generation.readers:
            self._edges_list.append(Edge(
                writer_desc,
                PortDesc(reader.layer_name, reader.port),
                generation.tensor,
                generation.id))

    def __add_inter_region_edges(self):
        """Add edges connecting the different generations of a region."""
        for region in self.regions:
            if len(region.generations) == 1:
                continue
            prev_generation = region.generations[0]
            for gen_id in range(1, len(region.generations)):
                curr_generation = region.generations[gen_id]
                self._edges_list.append(Edge(
                        PortDesc(_make_memory_node_name(region, prev_generation), 0),
                        PortDesc(_make_memory_node_name(region, curr_generation), 0),
                        curr_generation.tensor,
                        gen_id))
                prev_generation = curr_generation

    def should_include_region(self,
        region:Region,
        generation:RegionGeneration,
        is_constant: bool
    ) -> bool:
        nb_gens = region.nb_generations()
        is_user = generation.is_user
        is_forked = self.include_forking_regions and generation.is_forked
        include = self.include_regions or is_user or nb_gens > 1 or is_forked
        return (self.include_constants and is_constant) or (not is_constant and include)


"""
Graphviz DOT file format
"""


latency_types = (
    # Place default type in first entry.
    'avg_time',
    'median_time',
    'time')


def render_dot(dot_graph: Digraph, engine_name: str, output_format: str):
    """Render dot graph to an external file using the specified format."""
    dot_graph.format = output_format
    output_fname = os.path.abspath(f'{engine_name}.' + output_format)
    dot_graph.render(outfile=output_fname, view=False, overwrite_source=False)
    print(f"Created file://{output_fname}")
    return output_fname


def name_or_metadata(layer: Layer, prefer_matadata: bool=True):
    def clean_layer_name(layer_name: str):
        layer_name = layer_name.replace("||", "\|\|")
        layer_name = layer_name.replace("{", "")
        layer_name = layer_name.replace("}", "")
        return layer_name

    try:
        metadata = layer.metadata
        if metadata is not None and len(metadata) == 0:
            metadata = None
    except AttributeError:
        metadata = None
    if prefer_matadata and metadata is not None:
        return metadata
    return clean_layer_name(layer.name)


def layer_node_renderer_simple(
    layer: Layer,
    latency: float,
    display_layer_names: bool=True,
    expand_layer_details: bool=False,
    stack_layer_names: bool=True,
    prefer_matadata: bool=True,
) -> str:
    return f"{layer.name}\\n{layer.type}" if display_layer_names else f"{layer.type}"


def layer_node_renderer_keras(
    layer: Layer,
    latency: float,
    display_layer_names: bool=True,
    expand_layer_details: bool=False,
    stack_layer_names: bool=True,
    prefer_matadata: bool=True,
) -> str:
    """Keras-style node label formatting."""

    def add_io(tensors: List):
        io_desc_str = ""
        for t in tensors:
            io_desc_str += str(t.shape)
        return io_desc_str

    name = name_or_metadata(layer, prefer_matadata)
    label =  f"{name}\\n" if display_layer_names else ""
    label += f"{layer.type}"
    label += "|{input:|output:}|{{"
    label += add_io(layer.inputs)
    label += "}|"
    label += add_io(layer.outputs)
    label += "}"
    return label


def layer_node_highlighter(node_id: str, highlighted_layers_ids: List[int]
) -> Dict:
    """Highlight a layer node.

    Create a yellow hailo around the node.
    """
    should_highlight = highlighted_layers_ids and node_id in highlighted_layers_ids
    formatting = {'penwidth': str(6), 'color': 'yellow'}
    return formatting if should_highlight else {}


def layer_node_configurable_renderer(
    layer: Layer,
    latency: float,
    display_layer_names: bool=True,
    expand_layer_details: bool=False,
    stack_layer_names: bool=True,
    prefer_matadata: bool=True,
) -> str:
    def html_tbl(rows: List[str]):
        def html_tbl_row(row_content, bold:bool, color: str=None):
            row_content = row_content if not bold else f"<b>{row_content}</b>"
            if color:
                row = f"<TR><TD BGCOLOR=\"{color}\">{row_content}</TD></TR>"
            else:
                row = f"<TR><TD>{row_content}</TD></TR>"
            return row

        header = '''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" color="transparent">"'''
        footer = "</TABLE>>"
        tbl = header
        for i, row in enumerate(rows):
            tbl += html_tbl_row(row[0], i==0, row[1] if len(row)>1 else None)
        tbl += footer
        return tbl

    def handle_pwgen(layer: Layer, rows: List[str]):
        if layer.type != "PointWise":
            return
        try:
            subtype = layer.raw_dict['ParameterSubType']
            if subtype == 'PointWiseExpression':
                ops = layer.raw_dict['Operations']
                for op in ops:
                    prefix = "const auto"
                    op_str = op[len(prefix):]
                    HoneyDew = "#F0FFF0"
                    rows.append((op_str, HoneyDew))
        except KeyError:
            pass

    def handle_pwgen_act(layer: Layer, rows: List[str]):
        handled = False
        if layer.type != "Convolution":
            return handled
        try:
            subtype = layer.raw_dict['ParameterSubType']
            if subtype == 'PointWiseExpression':
                handled = True
                ops = layer.raw_dict['PointWiseExpressionOperations']
                for op in ops:
                    prefix = "const auto"
                    op_str = op[len(prefix):]
                    HoneyDew = "#F0FFF0"
                    rows.append((op_str, HoneyDew))
        except KeyError:
            pass
        return handled

    def handle_conv_deconv(layer: Layer, rows: List[str]):
        if layer.type not in ("Convolution", "Deconvolution"):
            return
        try:
            if len(layer.inputs) == 2:
                rows.append(("Conv(input0) + input1", "lightblue"))
            act = layer.raw_dict['Activation']
            if act is not None and act != 'NONE':
                handled = False
                if act == 'GENERIC':
                    handled = handle_pwgen_act(layer, rows)
                if not handled:
                    rows.append((act, "lightblue"))
        except KeyError:
            pass

    def handle_reformat(layer: Layer, rows: List[str]):
        if layer.type != "Reformat":
            return
        rows.append((layer.raw_dict['Origin'], None))

    def add_node_name(layer: Layer, rows: List[str], stack_layer_names: bool, prefer_matadata: bool):
        layer_name = name_or_metadata(layer, prefer_matadata) if display_layer_names else ""
        if stack_layer_names:
            # This is layer name "stacking": splitting on '+' and stacking in several rows
            parts = layer_name.split('+')
            for p in parts:
                rows.append((p,))
        else:
            rows.append((layer_name,))

    rows = [(f"{layer.type}",)]
    if latency:
        rows.append((f"{latency} ms",))
    if display_layer_names:
        add_node_name(layer, rows, stack_layer_names, prefer_matadata)
    handle_reformat(layer, rows)
    if expand_layer_details:
        handle_pwgen(layer, rows)
        handle_conv_deconv(layer, rows)
    tbl = html_tbl(rows)
    return tbl


def layer_precision_formatter(layer: Layer):
    """Format Dot nodes by layer precision"""
    formatting = {'shape': 'Mrecord',
                  'style': 'filled',
                  'tooltip': layer.tooltip(),
                  'fillcolor': precision_colormap[layer.precision],
                  'color': 'lightgray',
                  'fontname': 'Helvetica',}
    return formatting


def layer_type_formatter(layer: Layer):
    """Format Dot nodes by layer type"""

    def handle_reformat(layer: Layer):
        if layer.type != 'Reformat':
            return None
        try:
            origin = layer.raw_dict['Origin']
            if origin == 'QDQ':
                return layer_colormap['Quantize']
        except KeyError:
            return None

    try:
        layer_color = layer_colormap[layer.type]
        layer_color = handle_reformat(layer) or layer_color
    except KeyError:
        layer_color = "#E5E7E9"

    formatting = {'shape': 'Mrecord',
                  'style': 'filled',
                  'tooltip': layer.tooltip(),
                  'fillcolor': layer_color,
                  'color': 'lightgray',
                  'fontname': 'Helvetica'}
    return formatting


def region_precision_formatter(tensor: Activation, is_user: bool):
    """Format Dot edges by region precision"""
    formatting = {
        'style': 'filled' if is_user else 'dashed',
        # Hover popup text
        'tooltip': tensor.name,
        'penwidth': '3',
        'color': precision_colormap[tensor.precision],
        'fontname': 'Helvetica'}
    return formatting


layer_node_formatters = {
    "layer_type": layer_type_formatter,
    "layer_precision_formatter": layer_precision_formatter,
}

layer_node_renderers = {
    "Minimal": layer_node_renderer_simple,
    "Keras": layer_node_renderer_keras,
    "Configurable": layer_node_configurable_renderer,
}


def _get_latency(plan: EnginePlan, layer: Layer, latency_type) -> float:
    try:
        latency = plan.df[plan.df['Name'] == layer.name][f"latency.{latency_type}"].iloc[0]
    except (KeyError, IndexError):
        # Constant layer or no latency data.
        latency = 0
    return latency

def get_dot_id(layer_name: str) -> str:
    return layer_name.replace(":", "###") # f"l_{dot_node_id}"

def _get_dot_id(layer_name: str) -> str:
    return layer_name.replace(":", "###")

def _is_trainstation(layer):
    return layer.type == 'TrainStation'

def get_dot_id(layer_name: str) -> str:
    return layer_name.replace(":", "###") # f"l_{dot_node_id}"


class DotGraph(object):
    """This class converts a TensorRT plan into Graphviz DOT graphs"""
    def __init__(self,
        plan: EnginePlan,
        layer_node_formatter: Callable,
        layer_node_highlighter: Callable=layer_node_highlighter,
        layer_node_renderer: Callable=layer_node_configurable_renderer,
        region_formatter: Callable=region_precision_formatter,
        display_layer_names: bool=True,
        stack_layer_names: bool=True,
        display_regions: bool=False,
        display_forking_regions: bool=False,
        expand_layer_details: bool=False,
        display_constants: bool=False,
        display_latency: bool=True,
        latency_type: str='avg_time',
        display_region_reuse: bool=False,
        display_region_names: bool=False,
        display_edge_name: bool=False,
        display_edge_details: bool=True,
        highlight_layers: list=None,
        remove_disconnected_layers: bool=False,
        display_matadata: bool=True,
    ):
        plan_graph = PlanGraph(
            plan, display_regions, display_constants, display_forking_regions)
        self.dot = Digraph()
        self.layer_node_formatter = layer_node_formatter
        self.layer_node_highlighter = layer_node_highlighter
        self.layer_node_renderer = layer_node_renderer
        self.region_formatter = region_formatter
        self.expand_layer_details = expand_layer_details
        self.display_layer_names = display_layer_names
        self.stack_layer_names = stack_layer_names
        self.display_latency = display_latency
        self.latency_type = latency_type if latency_type in latency_types else latency_types[0]
        self.display_region_reuse = display_region_reuse
        self.render_region_reuse_edges = False
        self.display_region_names = display_region_names
        self.display_edge_name = display_edge_name
        self.display_edge_details = display_edge_details
        # Get the node names of the layers to highlight
        self.highlighted_layers_ids = None
        if highlight_layers:
            try:
                highlight_layers_name = plan.df['Name'].iloc[highlight_layers].to_list()
                self.highlighted_layers_ids = [_get_dot_id(name) for name in highlight_layers_name]
            except IndexError:
                warnings.warn("The layers indices specified for highlighting are incorrect")

        node_name_2_node_id = {}
        if remove_disconnected_layers:
            self.__remove_disconnected_layers(plan_graph)
        self.display_matadata = display_matadata
        self.__add_dot_region_nodes(plan_graph, node_name_2_node_id)
        self.__add_dot_layer_nodes(plan, plan_graph, node_name_2_node_id)
        self.__add_edges(plan_graph, node_name_2_node_id)
        self.__connect_train_station(plan_graph, node_name_2_node_id)

    def __add_dot_region_nodes(self, plan_graph, node_name_2_node_id):
        dot_node_id = 0
        for mem_node in plan_graph.memory_nodes:
            node_name_2_node_id[mem_node.name] = dot_id = _get_dot_id(mem_node.name)
            self.__create_dot_region_node(dot_id, mem_node.tensor, mem_node.is_user, mem_node.region_gen)
            dot_node_id += 1

    def __add_dot_layer_nodes(self, plan, plan_graph, node_name_2_node_id):
        for layer_node in plan_graph.layer_nodes:
            layer = layer_node.layer
            latency = _get_latency(plan, layer, self.latency_type)
            if not layer.type == 'Constant' or plan_graph.include_constants:
                dot_id = _get_dot_id(layer.name)
                node_name_2_node_id[layer.name] = dot_id
                self.__create_dot_layer_node(
                    dot_id, layer, latency, layer_node_renderer=self.layer_node_renderer)

    def __remove_disconnected_layers(self, plan_graph):
        # Remove layer nodes that have no inputs and no outputs.
        # TrainStation layers are exempt.
        disconnected = lambda layer_node: len(layer_node.layer.inputs) == len(layer_node.layer.outputs) == 0
        nodes_to_remove = [layer_node for layer_node in plan_graph.layer_nodes if
            disconnected(layer_node) and not _is_trainstation(layer_node.layer)]
        for node in nodes_to_remove:
            plan_graph.layer_nodes.remove(node)

    def __connect_train_station(self, plan_graph, node_name_2_node_id):
        prev_layer = None
        n_layers = len(plan_graph.layer_nodes)
        for i, layer_node in enumerate(plan_graph.layer_nodes):
            layer = layer_node.layer
            if _is_trainstation(layer) and prev_layer and i < n_layers-1:
                src_id = node_name_2_node_id[prev_layer.name]
                dst_id = node_name_2_node_id[layer.name]
                self.__create_dot_dependency_edge(src_id, dst_id)
            if prev_layer and _is_trainstation(prev_layer) and i>1:
                src_id = node_name_2_node_id[prev_layer.name]
                dst_id = node_name_2_node_id[layer.name]
                self.__create_dot_dependency_edge(src_id, dst_id)
            prev_layer = layer

    def __add_edges(self, plan_graph, node_name_2_node_id):
        for edge in plan_graph.edges_list:
            src_id = node_name_2_node_id[edge.src.layer_name]
            dst_id = node_name_2_node_id[edge.dst.layer_name]
            self.__create_dot_edge(src_id, dst_id, edge.tensor, edge.region_gen)

    def __create_dot_region_node(self, node_id: int, tensor: Activation, is_user: bool, gen: int):
        formatter = self.region_formatter(tensor, is_user)
        is_minimal = False
        if is_minimal and not is_user:
            self.dot.node(
                str(node_id),
                label="",
                shape='diamond',
                height=".2", width=".2",
                **formatter)
        else:
            desc = tensor.name if (self.display_region_names or is_user) else ""
            if self.display_region_reuse and gen > 0:
                desc = f"{desc} (gen={gen})"
            desc = f"{desc}\n{tensor.tooltip()}" if desc else tensor.tooltip()
            self.dot.node(
                str(node_id),
                desc,
                shape='rectangle',
                fillcolor='gray' if is_user else None,
                **formatter)

    def __create_dot_layer_node(
        self, node_id: str, layer: Layer, latency: float, layer_node_renderer: Callable
    ):
        formatting = self.layer_node_formatter(layer)
        formatting.update(self.layer_node_highlighter(node_id, self.highlighted_layers_ids))
        self.dot.node(
            str(node_id),
            layer_node_renderer(
                layer,
                latency if (self.display_latency and latency) else None,
                expand_layer_details=self.expand_layer_details,
                display_layer_names=self.display_layer_names,
                stack_layer_names=self.stack_layer_names,
                prefer_matadata=self.display_matadata),
                **formatting)

    def __create_dot_dependency_edge(self, src, dst):
        self.dot.edge(src, dst, "", color='lightgray', style="dashed")

    def __create_dot_edge(self, src, dst, tensor, region_gen):
        def generation_color(gen: int, line_color: str) -> str:
            edge_color = ""
            if self.render_region_reuse_edges:
                for _ in range(gen):
                    edge_color += f"{line_color}:white:"
            edge_color += f"{line_color}"
            return edge_color

        edge_color = precision_colormap[tensor.precision]

        if region_gen:
            edge_color = generation_color(region_gen, edge_color)
        desc = []
        if self.display_edge_name:
            desc.append(tensor.name)
        if self.display_edge_details:
            desc.append(tensor.tooltip())
        desc_str = "\n".join(desc)
        self.dot.edge(src, dst, desc_str, color=edge_color)


def to_dot(*args, **kwargs) -> Digraph:
    """Convert plan-graph to dot format"""
    g = DotGraph(*args, **kwargs)
    return g.dot


"""
ONNX file format
"""

import onnx

def make_onnx_tensor(tensor):
    def get_type(desc):
        desc = desc.lower()
        if 'int8' in desc:
            return onnx.TensorProto.INT8
        elif 'fp32' in desc:
            return onnx.TensorProto.FLOAT
        elif 'fp16' in desc:
            return onnx.TensorProto.FLOAT16
        elif 'int32' in desc:
            return onnx.TensorProto.INT32
        else:
            raise ValueError(f"Uknown precision {desc}")

    t = onnx.helper.make_tensor_value_info(
            tensor.name,
            get_type(tensor.format),
            tensor.shape)
    return t

class OnnxGraph(object):
    def __init__(self, plan: EnginePlan, display_forking_regions: bool):
        def get_adjacency_lists():
            inputs_map, outputs_map = {}, {}
            layer_nodes_names = [node.layer.name for node in self.plan_graph.layer_nodes]
            for edge in self.plan_graph.edges_list:
                if edge.src.port != None and edge.dst.port != None:
                    if False and edge.src.layer_name in layer_nodes_names and edge.dst.layer_name in layer_nodes_names:
                        edge_name = edge.tensor.name
                    else:
                        edge_name = f"{edge.src.layer_name}:{edge.src.port}##{edge.dst.layer_name}:{edge.dst.port}"
                elif edge.src.port != None:
                    edge_name = edge.dst.layer_name
                else:
                    edge_name = edge.src.layer_name
                try:
                    outputs_map[edge.src.layer_name].append(edge_name)
                except KeyError:
                    outputs_map[edge.src.layer_name] = list((edge_name,))

                try:
                    inputs_map[edge.dst.layer_name].append(edge_name)
                except KeyError:
                    inputs_map[edge.dst.layer_name] = list((edge_name,))
            return inputs_map, outputs_map

        def add_layer_nodes():
            for layer_id, layer_node in enumerate(self.plan_graph.layer_nodes):
                layer = layer_node.layer
                try:
                    self.__add_layer_node(layer_id, layer, inputs_map[layer.name], outputs_map[layer.name])
                except KeyError:
                    # Graph inputs/outputs (bindings) will not have an entry in the inputs_map or outputs_map
                    pass

        def add_memory_nodes():
            for mem_node in self.plan_graph.memory_nodes:
                is_user = mem_node.is_user
                if not is_user:
                    self.__add_region_node(
                        mem_node.region_gen, mem_node.name, is_user,
                        inputs_map[mem_node.name], outputs_map[mem_node.name])


        def add_graph_inputs_outputs():
            # Graph inputs/outputs handling
            g_inputs, g_outputs = plan.get_bindings()
            for inp in g_inputs:
                graph_inputs.append(make_onnx_tensor(inp))
            for outp in g_outputs:
                graph_outputs.append(make_onnx_tensor(outp))

        def finalize_onnx_graph():
            graph_def = onnx.helper.make_graph(
                self.onnx_nodes,
                'test-model',
                graph_inputs,
                graph_outputs)
            self.onnx_model = onnx.helper.make_model(
                graph_def, producer_name='nvidia-trex')

        self.onnx_nodes = []
        graph_inputs, graph_outputs = [], []
        self.plan = plan
        self.plan_graph = PlanGraph(plan, include_forking_regions=display_forking_regions)

        inputs_map, outputs_map = get_adjacency_lists()
        add_memory_nodes()
        add_layer_nodes()
        add_graph_inputs_outputs()
        finalize_onnx_graph()

    def __add_region_node(self, gen: int, region_name: str, is_user: bool, inputs, outputs):
        assert not is_user
        node_def = onnx.helper.make_node("Region", inputs, outputs, region_name)
        self.onnx_nodes.append(node_def)

    def __add_layer_node(self, node_id: int, layer: Layer, inputs, outputs):
        def get_type(layer):
            op_type = layer.type
            # Convert Op Type to onnx.ai namepsace because
            # Netron assigns these nodes specific colors.
            if op_type == "Convolution":
                op_type = "Conv"
            if op_type == "Pooling":
                if layer.raw_dict["PoolingType"] == "AVERAGE":
                    op_type = "AveragePool"
                if layer.raw_dict["PoolingType"] == "Max":
                    op_type = "MaxPool"
            return op_type

        def add_attributes(layer, node_def):
            for key, value in sorted(layer.items()):
                if key not in [
                    'InputRegions', 'OutputRegions', 'Inputs',
                    'Outputs', 'Name', 'name', 'ParameterType', 'LayerName']:
                    node_def.attribute.extend([onnx.helper.make_attribute(key, value)])

        op_type = get_type(layer)
        if op_type == 'Constant':
            return

        # Find inputs to layer id, outputs
        node_def = onnx.helper.make_node(op_type, inputs, outputs, layer.name)
        add_attributes(layer.raw_dict, node_def)
        self.onnx_nodes.append(node_def)
