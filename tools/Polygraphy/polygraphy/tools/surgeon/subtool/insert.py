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
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import ModelArgs, OnnxLoaderArgs, OnnxSaveArgs, OnnxShapeInferenceArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool

gs = mod.lazy_import("onnx_graphsurgeon")
onnx_backend = mod.lazy_import("polygraphy.backend.onnx")


class OnnxNodeArgs(BaseArgs):
    def add_to_parser(self, parser):
        node_args = parser.add_argument_group("Inserted Node", "Options for the node to insert")
        node_args.add_argument(
            "--inputs",
            help="The names of input tensors for the new node. Order will be preserved. "
            "Format: --inputs <name>. For example: --inputs name0 name1",
            nargs="+",
            required=True,
        )
        node_args.add_argument(
            "--outputs",
            help="The names of output tensors for the new node. Order will be preserved. "
            "If an output tensor is also specified as an input, a new tensor will be generated for the output"
            "Format: --outputs <name>. For example: --outputs name0 name1",
            nargs="+",
            required=True,
        )
        node_args.add_argument("--op", help="The ONNX op to use for the new node", required=True)
        node_args.add_argument("--name", help="The name to use for the new node", default=None)
        node_args.add_argument(
            "--attrs",
            help="Attributes to set in the new node. "
            "Format: --attrs <name>=value. For example: --attrs axis=1 keepdims=1. "
            "Attributes of type: float, int, str, and lists of these types are supported. "
            "Numbers including a decimal point will always be parsed as floats, and quoted values "
            "(e.g. --attrs name='53') will always be parsed as strings. Values enclosed in brackets "
            "(e.g. --attrs axes=[0,1]) will be parsed as lists. ",
            nargs="+",
            default=[],
        )

    def parse(self, args):
        self.op = args_util.get(args, "op")
        self.name = args_util.get(args, "name")

        self.attrs = args_util.parse_dict_with_default(args_util.get(args, "attrs"), sep="=")
        self.inputs = args_util.get(args, "inputs")
        self.outputs = args_util.get(args, "outputs")


class Insert(BaseSurgeonSubtool):
    """
    [EXPERIMENTAL] Insert a single node into an ONNX model with the specified inputs and outputs.
    Any existing subgraph between the inputs and outputs is replaced.
    """

    def __init__(self):
        super().__init__("insert")
        self.subscribe_args(OnnxNodeArgs())
        self.subscribe_args(ModelArgs(model_required=True, inputs=None, model_type="onnx"))
        self.subscribe_args(OnnxShapeInferenceArgs())
        self.subscribe_args(OnnxLoaderArgs(output_prefix=None))
        self.subscribe_args(OnnxSaveArgs(infer_shapes=True, required=True))

    def run_impl(self, args):
        graph = onnx_backend.gs_from_onnx(super().load_model())

        TENSOR_MAP = graph.tensors()

        def get_tensor(name):
            if name not in TENSOR_MAP:
                G_LOGGER.critical("Tensor: {:} does not exist in the model.".format(name))
            return TENSOR_MAP[name]

        TENSOR_NAME_SUFFIX = "_polygraphy_surgeon_insert_output"

        output_tensors = []
        for name in self.arg_groups[OnnxNodeArgs].outputs:
            if name in self.arg_groups[OnnxNodeArgs].inputs:
                # When the new node's input == output, we need to generate a new tensor
                # If the tensor was a graph output, try to preserve the name.
                inp_tensor = get_tensor(name)
                if inp_tensor in graph.outputs:
                    inp_tensor.name += TENSOR_NAME_SUFFIX
                    tensor = gs.Variable(name=name)
                else:
                    tensor = gs.Variable(name=name + TENSOR_NAME_SUFFIX)

                def replace_tensor(tensors):
                    # This is needed to preserve ordering and handle cases where the tensor shows up more than once.
                    for index, t in enumerate(tensors):
                        if t.name == inp_tensor.name:
                            tensors[index] = tensor

                for out_node in inp_tensor.outputs:
                    replace_tensor(out_node.inputs)

                replace_tensor(graph.outputs)
                G_LOGGER.verbose("Generating new tensor for output: {:}".format(tensor))
            else:
                tensor = get_tensor(name)
            tensor.inputs.clear()
            output_tensors.append(tensor)

        input_tensors = [get_tensor(name) for name in self.arg_groups[OnnxNodeArgs].inputs]

        new_node = gs.Node(
            op=self.arg_groups[OnnxNodeArgs].op,
            name=self.arg_groups[OnnxNodeArgs].name,
            attrs=self.arg_groups[OnnxNodeArgs].attrs,
            inputs=input_tensors,
            outputs=output_tensors,
        )
        G_LOGGER.verbose("Generated new node: {:}".format(new_node))

        # Assuming the graph is topologically sorted, the node needs to be inserted
        # after its last input node to maintain the sorting.
        with graph.node_ids():
            # Nodes with no inputs can be inserted at index 0
            insert_index = max([node.id + 1 for inp in input_tensors for node in inp.inputs] + [0])

        graph.nodes.insert(insert_index, new_node)

        super().save_model(super().export_graph(graph.cleanup()))
