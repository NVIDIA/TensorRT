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

"""
This file contains code to generate graph diagrams for an engine-plan.
"""


from ast import Call
import warnings
import os
import re
from graphviz import Digraph
from typing import Callable, NamedTuple, List
from .engine_plan import EnginePlan
from .layer import Layer
from .activations import Activation
from .plotting import precision_colormap, layer_colormap


class Region(NamedTuple):
    id: int
    tensor: Activation
    is_user: bool
    should_display: bool


class Edge(NamedTuple):
    src: str
    dst: str
    tensor: Activation
    region_gen: int = None


class RegionGenerations:
    """A memory region may have several generations, where each generation
    represents a consumable state of the region.
    """
    def __init__(self, plan):
        """
        A P event represents some layer writing to (producing part of) a memory
        region.
        A C event represents some layer reading from (consuming part of) a
        memory region.

        A region_story is a map from region name to the list of C/P events on
        that region.
        """
        story = {}
        for layer in plan.layers:
            for inp in layer.inputs:
                story.setdefault(inp.name, []).append(('C', layer.name))
            for outp in layer.outputs:
                story.setdefault(outp.name, []).append(('P', layer.name))

        # Create region generations.

        # Each time we switch between C events and P events we create a new
        # Region generation.
        regions_gens = {}
        for region, region_evts in story.items():
            # A List of region generations.
            # Each generation is a list of C/P events.
            generations = [list()]
            current_gen = 0
            last = 'P'
            for evt in region_evts:
                if evt[0] != last:
                    last = evt[0]
                    if evt[0] == 'P':
                        current_gen += 1
                        generations.append(list())
                generations[current_gen].append(evt)
            regions_gens[region] = generations
        self.regions_gens = regions_gens

    def lookup_region_gen(self, layer_name: str, region_name: str):
        """Lookup the generation of a Region, given a layer name.

        If a layer produces or consumes the Region, we return the corresponding
        Region generation.
        """
        try:
            region_gens = self.regions_gens[region_name]
        except KeyError:
            # A KeyError can happen if we have a disconneted graph.
            return -1
        for gen_id, gen in enumerate(region_gens):
            for evt in gen:
                if evt[1] == layer_name:
                    return gen_id
        # Assume for now that this is OK - it's probably a Constant
        # layer that produced this region.
        return 0

    def nb_generations(self, region_name: str):
        region_gens = self.regions_gens[region_name]
        return len(region_gens)

    def debug_generation(self):
        for r, generations in self.regions_gens.items():
            for g in generations:
                print(r, g)

    def create_id(self, region_id:int , region_generation: int):
        return region_id + 100000 * region_generation


def render_dot(dot_graph: Digraph, engine_name: str, output_format: str):
    """Render dot graph to an external file using the specified format."""
    dot_graph.format = output_format
    output_fname = os.path.abspath(f'{engine_name}.' + output_format)
    dot_graph.render(outfile=output_fname, view=False, overwrite_source=False)
    print(f"Created file://{output_fname}")
    return output_fname


def node_label_simple(
    layer: Layer,
    latency: float,
    display_layer_name: bool=True,
    expand_layer_details: bool=False,
) -> str:
    return f"{layer.name}\\n{layer.type}"


def node_label_keras(
    layer: Layer,
    latency: float,
    display_layer_name: bool=True,
    expand_layer_details: bool=False,
) -> str:
    """Keras-style node label formatting."""

    def add_io(tensors: List):
        io_desc_str = ""
        for t in tensors:
            io_desc_str += str(t.shape)
        return io_desc_str

    label =  f"{layer.name}\\n" if display_layer_name else ""
    label += f"{layer.type}"
    label += "|{input:|output:}|{{"
    label += add_io(layer.inputs)
    label += "}|"
    label += add_io(layer.outputs)
    label += "}"
    return label


def node_label_tbl(
    layer: Layer,
    latency: float,
    display_layer_name: bool=True,
    expand_layer_details: bool=False,
) -> str:
    def clean_layer_name(layer_name: str):
        layer_name = layer_name.replace("||", "\|\|")
        layer_name = layer_name.replace("{", "")
        layer_name = layer_name.replace("}", "")
        return layer_name

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

    def handle_conv_deconv(layer: Layer, rows: List[str]):
        if layer.type not in ("Convolution", "Deconvolution"):
            return
        try:
            if len(layer.inputs) == 2:
                rows.append(("Incident Add", "lightblue"))
            act = layer.raw_dict['Activation']
            if act is not None and act != 'NONE':
                rows.append((act, "lightblue"))
        except KeyError:
            pass

    layer_name = clean_layer_name(layer.name)
    layer_type = f"{layer.type} ({latency} ms)"
    rows = [(layer_type,)]
    if display_layer_name:
        parts = layer_name.split('+')
        for p in parts:
            rows.append((p,))
    if expand_layer_details:
        handle_pwgen(layer, rows)
        handle_conv_deconv(layer, rows)
    tbl = html_tbl(rows)
    return tbl


def parse_operation(op):
    c = re.compile("\((.+\))")
    args = c.findall(op)
    args[0].split(",")
    args = args[0].split(",")

    c = re.compile("pwgen::.+\(")
    w = c.findall(op)
    opname = w[0][:-1]

    output = op.split(" ")[2]
    print(f"{opname}: {args} -> {output}")
    return opname, args, output


class PlanGraph(object):
    """This is a base-class for representing TensorRT plans as renderable graphs"""
    def __init__(self,
        plan: EnginePlan,
        display_regions: bool=False
    ):
        self.plan = plan
        self.display_regions = display_regions
        self.regions_dict = {}
        self.regions_generations = RegionGenerations(plan)
        self.edges_list = []
        self.__create_graph(plan)

    def __create_graph(self, plan: EnginePlan):
        region_id = len(plan.all_layers)
        for layer_id, layer in enumerate(plan.all_layers):
            for inp in layer.inputs:
                region_id = self.handle_region(layer_id, layer, inp, region_id, is_input=True)
            for outp in layer.outputs:
                region_id = self.handle_region(layer_id, layer, outp, region_id, is_input=False)
        for edge in self.edges_list:
            self.add_edge(edge.src, edge.dst, edge.tensor, edge.region_gen)

        for layer_id, layer in enumerate(plan.all_layers):
            try:
                latency = plan.df[plan.df['Name'] == layer.name]['latency.avg_time'].iloc[0]
            except (KeyError, IndexError):
                # Constants layer
                latency = 0
            self.add_layer_node(layer_id, layer, latency, node_labeler=node_label_tbl)

        for generations in self.regions_dict.values():
            for r in generations:
                if r.should_display:
                    self.add_region_node(r.id, r.tensor, r.is_user)
        self.check_consistency()

    def check_consistency(self):
        # Verify that every output is either an output binding, or an input
        # of another layer.
        for region_name, region in self.regions_dict.items():
            nb_prods = self._nb_producers(self.plan.all_layers, region_name)
            nb_cons = self._nb_consumers(self.plan.all_layers, region_name)
            is_user = region_name in self.plan.bindings# or nb_cons==0 or nb_prod==0
            if not is_user and nb_cons == 0:
                warnings.warn(f"Region {region_name} is neither a binding nor a layer input.")
            if not is_user and nb_prods == 0:
                warnings.warn(f"Region {region_name} is neither a binding nor a layer output.")

    def find_producers(self, layers, region_name):
        producers = []
        for i,l in enumerate(self.plan.all_layers):
            for o in l.outputs:
                if o.name == region_name:
                    producers.append((i, l.name))
        return producers

    def _nb_producers(self, layers, inp_name):
        return len(self.find_producers(layers, inp_name))

    def _nb_consumers(self, layers, region_name):
        consumers = []
        for id,l in enumerate(self.plan.all_layers):
            for i in l.inputs:
                if i.name == region_name:
                    consumers.append((i, region_name))
        return len(consumers)

    def should_display_region(self, region_name: str, display_regions: bool) -> bool:
        nb_gens = self.regions_generations.nb_generations(region_name)
        nb_prod = self._nb_producers(self.plan.all_layers, region_name)
        nb_cons = self._nb_consumers(self.plan.all_layers, region_name)
        is_user = region_name in self.plan.bindings or nb_cons==0 or nb_prod==0
        add = is_user or display_regions or nb_gens > 1 or nb_prod > 1
        return add

    def new_region(
        self,
        tensor:Activation,
        region_id:int,
        is_user: bool,
        should_display: bool
    ) -> int:
        self.regions_dict[tensor.name] = [Region(region_id, tensor, is_user, should_display)]
        region_id += 1
        return region_id

    def handle_region(
        self,
        layer_id:int,
        layer: Layer,
        tensor: Activation,
        region_id: int,
        is_input: bool
    ) -> int:
        region_gen = self.regions_generations.lookup_region_gen(layer.name, tensor.name)
        if region_gen == -1:
            should_display = True
            is_new_region = True
        else:
            should_display = self.should_display_region(tensor.name, self.display_regions)
            is_new_region = tensor.name not in self.regions_dict
        is_user = tensor.name in self.plan.bindings
        is_new_generation = (not is_new_region
                            and (region_gen+1) > len(self.regions_dict[tensor.name]))
        if is_new_region:
            region_id = self.new_region(tensor, region_id, is_user, should_display)
        elif is_new_generation:
            self.add_generation(tensor, region_gen, is_user, region_id)
        region = self.regions_dict[tensor.name][region_gen]
        if should_display:
            _from = str(region.id) if is_input else str(layer_id)
            _to = str(layer_id) if is_input else str(region.id)
            self.edges_list.append(Edge(
                _from, _to, tensor))
        elif is_input:
            producers = self.find_producers(self.plan.all_layers, tensor.name)
            for producer in producers:
                self.edges_list.append(Edge(
                    str(producer[0]),
                    str(layer_id), tensor))
        return region_id

    def add_region_node(self, id: int, tensor: Activation, is_user: bool):
        pass

    def add_layer_node(self, node_id: int, layer: Layer, latency: float, node_labeler: Callable):
        pass

    def add_edge(self, src, end, tensor, region_gen):
        pass

    def add_generation(self, tensor: Activation, region_gen: int, is_user: bool, region_id):
        new_region_id = self.regions_generations.create_id(region_id, region_gen+1)
        self.regions_dict[tensor.name].append(Region(new_region_id, tensor, is_user, True))

        # Add an edge between the previous and current region generations
        previous_gen_region = self.regions_dict[tensor.name][region_gen-1]
        self.edges_list.append(Edge(
            str(previous_gen_region.id),
            str(new_region_id),
            tensor, region_gen=region_gen))


def precision_formatter(layer: Layer):
    """Format Dot nodes by layer precision"""
    formatting = {'style': 'filled',
                  'tooltip': layer.tooltip(),
                  'fillcolor': precision_colormap[layer.precision]}
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

    formatting = {'style': 'filled',
                  'tooltip': layer.tooltip(),
                  'fillcolor': layer_color,
                  'color': 'white',}
    return formatting


def tensor_precision_formatter(tensor: Activation):
    """Format Dot edges by tensor precision"""
    formatting = {
        'color': precision_colormap[tensor.precision],
        'tooltip': str(tensor.shape)}
    return formatting


def region_precision_formatter(tensor: Activation):
    """Format Dot edges by region precision"""
    formatting = {
        'style': 'filled' if tensor.is_user else 'dashed',
        'tooltip': str(tensor.shape),
        'penwidth': '3',
        'color': precision_colormap[tensor.precision]}
    return formatting

class DotGraph(PlanGraph):
    """This class converts TensorRT plans to Graphviz DOT graphs"""
    def __init__(self,
        plan: EnginePlan,
        node_formatter: Callable,
        region_formatter: Callable=region_precision_formatter,
        display_layer_names: bool=True,
        display_regions: bool=False,
        expand_layer_details: bool=False,
    ):
        self.dot = Digraph()
        self.node_formatter = node_formatter
        self.region_formatter = region_formatter
        self.expand_layer_details = expand_layer_details
        super().__init__(plan, display_regions)

    def add_region_node(self, id: int, tensor: Activation, is_user: bool):
        tensor.is_user = is_user
        formatter = self.region_formatter(tensor)
        self.dot.node(
            str(id),
            f"{tensor.name}\n{tensor.tooltip()}",
            shape='rectangle',
            fillcolor='gray' if is_user else None,
            fontname="Helvetica",
            **formatter)

    def add_layer_node(
        self, node_id: int, layer: Layer, latency: float, node_labeler: Callable
    ):
        formatting = self.node_formatter(layer)

        self.dot.node(
            str(node_id),
            node_labeler(layer, latency, expand_layer_details=self.expand_layer_details),
            shape='Mrecord',
            fontname="Helvetica", **formatting)

    def add_edge(self, src, end, tensor, region_gen):
        def generation_color(gen: int, line_color: str) -> str:
            edge_color = ""
            for i in range(gen):
                edge_color += f"{line_color}:white:"
            edge_color += f"{line_color}"
            return edge_color

        edge_color = precision_colormap[tensor.precision]
        if region_gen:
            edge_color = generation_color(region_gen, edge_color)
            desc = tensor.tooltip()
        else:
            desc = tensor.tooltip()
        self.dot.edge(src, end, desc, color=edge_color)


def to_dot(plan: EnginePlan,
           node_formatter: Callable,
           region_formatter: Callable=region_precision_formatter,
           display_layer_names: bool=True,
           display_regions: bool=False,
           display_constants: bool=False,
           expand_layer_details: bool=False) -> Digraph:
    """Convert plan-graph to dot format"""
    g = DotGraph(
        plan, node_formatter, region_formatter,
        display_layer_names, display_regions, expand_layer_details)
    return g.dot


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

class OnnxGraph(PlanGraph):
    def __init__(self,
        plan: EnginePlan,
        display_layer_names: bool=True,
        display_regions: bool=False,
    ):
        self.onnx_nodes = []
        self.graph_inputs, self.graph_outputs = [], []
        self.regions_inputs = dict()
        self.regions_outputs = dict()
        self.plan = plan

        for layer in plan.layers:
            for inp in layer.inputs:
                if inp.name in self.plan.bindings:
                    self.graph_inputs.append(make_onnx_tensor(inp))
            for outp in layer.outputs:
                if outp.name in self.plan.bindings:
                    self.graph_outputs.append(make_onnx_tensor(outp))

        super().__init__(plan, display_regions)
        graph_def = onnx.helper.make_graph(
            self.onnx_nodes,
            'test-model',
            self.graph_inputs,
            self.graph_outputs)
        self.model_def = onnx.helper.make_model(
            graph_def, producer_name='engine2onnx')

    def add_region_node(self, id: int, tensor: Activation, is_user: bool):
        if is_user:
            # In the ONNX programming model we create
            # bindings when we create the graph.
            return
        try:
            inputs = self.regions_inputs[tensor.name]
        except:
            print(f"Did not find input {id}")
            inputs = None
        try:
            outputs = self.regions_outputs[tensor.name]
        except:
            print(f"Did not find output {id}")
            outputs = None
        name = str(id)
        node_def = onnx.helper.make_node("Region", inputs, outputs, name)
        self.onnx_nodes.append(node_def)

    def add_layer_node(self, node_id: int, layer: Layer, node_labeler: Callable):
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
        name = layer.name

        print(f"adding node {node_id} - {layer.name}")

        # Find layer id
        inputs, outputs = [], []
        for edge in self.edges_list:
            if edge.src == str(node_id):
                outputs.append(edge.tensor.name)
            if edge.dst == str(node_id):
                inputs.append(edge.tensor.name)

        # Find inputs to layer id, outputs
        node_def = onnx.helper.make_node(op_type, inputs, outputs, name)
        add_attributes(layer.raw_dict, node_def)
        self.onnx_nodes.append(node_def)

    def add_edge(self, src, end, tensor, region_gen):
        if True:
            edge = f"{src}_to_{end}"
            edge = tensor.name
            try:
                self.regions_inputs[tensor.name].append(edge)
            except KeyError:
                self.regions_inputs[tensor.name] = [edge,]

            try:
                self.regions_outputs[tensor.name].append(end)
            except KeyError:
                self.regions_outputs[tensor.name] = [end,]

