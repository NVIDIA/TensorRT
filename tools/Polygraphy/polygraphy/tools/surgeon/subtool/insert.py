#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import onnx_graphsurgeon as gs
from polygraphy.logger import G_LOGGER
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool


class Insert(BaseSurgeonSubtool):
    """
    [EXPERIMENTAL] Insert a single node into a graph with the specified inputs and outputs.
    Any existing subgraph between the inputs and outputs is replaced.
    """
    def __init__(self):
        super().__init__("insert")


    def add_parser_args(self, parser):
        parser.add_argument("--inputs", help="The names of input tensors for the new node. Order will be preserved. "
                            "Format: --inputs <name>. For example: --inputs name0 name1", nargs="+", required=True)
        parser.add_argument("--outputs", help="The names of output tensors for the new node. Order will be preserved. "
                            "If an output tensor is also specified as an input, a new tensor will be generated for the output"
                            "Format: --outputs <name>. For example: --outputs name0 name1", nargs="+", required=True)

        node_args = parser.add_argument_group("Inserted Node", "Options for the node to insert")
        node_args.add_argument("--op", help="The ONNX op to use for the new node", required=True)
        node_args.add_argument("--name", help="The name to use for the new node", default=None)

        super().add_parser_args(parser, gs=True, output=True)


    def run(self, args):
        _, graph = super().import_graph(args)

        TENSOR_MAP = graph.tensors()

        def get_tensor(name):
            if name not in TENSOR_MAP:
                G_LOGGER.critical("Tensor: {:} does not exist in the model.".format(name))
            return TENSOR_MAP[name]

        # We populate outputs first because we may need to update output nodes from the
        # input tensors if output == input.
        output_tensors = []
        for name in args.outputs:
            if name in args.inputs:
                tensor = gs.Variable(name="{:}_polygraphy_surgeon_insert_output".format(name))

                # Bind outputs to outputs of original inputs.
                # This construct is required to preserve ordering of the input tensors in the output nodes.
                for out in get_tensor(name).outputs:
                    for index, inp in enumerate(out.inputs):
                        if inp.name == name:
                            out.inputs[index] = tensor

                G_LOGGER.verbose("Generating new tensor for output: {:}".format(tensor))
            else:
                tensor = get_tensor(name)
            tensor.inputs.clear()
            output_tensors.append(tensor)

            if not tensor.outputs:
                for index, out in enumerate(graph.outputs):
                    if out.name == name:
                        graph.outputs[index] = tensor


        input_tensors = []
        for name in args.inputs:
            tensor = get_tensor(name)
            tensor.outputs.clear()
            input_tensors.append(tensor)

        new_node = gs.Node(op=args.op, name=args.name, inputs=input_tensors, outputs=output_tensors)
        G_LOGGER.verbose("Generated new node: {:}".format(new_node))

        graph.nodes.append(new_node)

        # Since new graph outputs may be added, and we don't know the types, we skip type checks in ONNX-GraphSurgeon.
        super().export_graph(graph, args, do_type_check=False)
