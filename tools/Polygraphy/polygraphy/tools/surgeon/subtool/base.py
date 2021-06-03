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
from collections import OrderedDict

import onnx
import onnx_graphsurgeon as gs
from polygraphy.common import func
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, OnnxLoaderArgs
from polygraphy.tools.base import Tool
from polygraphy.util import misc


class BaseSurgeonSubtool(Tool):
    def __init__(self, name, inputs=None, data=False, shape_inference_default=None):
        super().__init__(name)
        self.subscribe_args(ModelArgs(model_required=True, inputs=inputs, model_type="onnx"))
        self.subscribe_args(OnnxLoaderArgs(write=False, outputs=False, shape_inference_default=shape_inference_default))
        if data:
            self.subscribe_args(DataLoaderArgs())


    def add_parser_args(self, parser, gs=False, output=False):
        if gs:
            parser.add_argument("--no-cleanup", help="Skip cleanup and keep unused nodes in the graph", action="store_true")
            parser.add_argument("--no-toposort", help="Skip topologically sorting the graph", action="store_true")
        if output:
            parser.add_argument("-o", "--output", required=True, help="Path at which to write the final ONNX model")


    def import_graph(self, args):
        onnx_model = func.invoke(self.makers[OnnxLoaderArgs].get_onnx_loader())
        return onnx_model, gs.import_onnx(onnx_model)


    def export_graph(self, graph, args, do_type_check=True):
        if not args.no_cleanup:
            graph.cleanup()
        if not args.no_toposort:
            graph.toposort()

        G_LOGGER.info("Writing model to: {output}. To see more details about the model, use: polygraphy inspect model {output} --mode=basic".format(output=args.output))
        onnx.save(gs.export_onnx(graph, do_type_check=do_type_check), args.output)


# Weights should be stored separately, JSON can just have a reference to a key.
class Config(OrderedDict):
    @staticmethod
    def from_graph(graph):
        def names_from_tensors(tensors):
            return [tensor.name for tensor in tensors]


        def meta_from_tensors(tensors):
            meta = []
            for tensor in tensors:
                tensor_meta = {"name": tensor.name}
                if tensor.dtype:
                    tensor_meta["dtype"] = misc.STR_FROM_NP_TYPE[tensor.dtype]
                if tensor.shape:
                    tensor_meta["shape"] = tensor.shape
                meta.append(tensor_meta)
            return meta


        config = Config()
        config["graph_inputs"] = meta_from_tensors(graph.inputs)
        config["graph_outputs"] = meta_from_tensors(graph.outputs)

        config["nodes"] = []
        for node_id, node in enumerate(graph.nodes):
            node_info = {
                "id":  node_id,
                "name": node.name,
                "op": node.op,
                "inputs": names_from_tensors(node.inputs),
                "outputs": names_from_tensors(node.outputs),
            }
            config["nodes"].append(node_info)
        return config
