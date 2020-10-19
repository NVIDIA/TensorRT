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
import json
from collections import OrderedDict

import onnx
import onnx_graphsurgeon as gs
from polygraphy.common import TensorMetadata, constants
from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool
from polygraphy.tools.util import args as args_util
from polygraphy.tools.util import misc as tool_util
from polygraphy.util import misc


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


################################# SUBTOOLS #################################

class STSurgeonBase(Tool):
    def add_parser_args(self, parser, gs=False, inputs=False, shape_inference_default=None, data=False):
        if gs:
            parser.add_argument("--no-cleanup", help="Skip cleanup and keep unused nodes in the graph", action="store_true")
            parser.add_argument("--no-toposort", help="Skip topologically sorting the graph", action="store_true")
        args_util.add_model_args(parser, model_required=True, inputs=inputs)
        args_util.add_onnx_args(parser, write=False, outputs=False, shape_inference_default=shape_inference_default)
        args_util.add_tf_onnx_args(parser)
        if data:
            args_util.add_dataloader_args(parser)


    def setup(self, args):
        onnx_model = tool_util.get_onnx_model_loader(args)()
        return gs.import_onnx(onnx_model)


class STExtract(STSurgeonBase):
    """
    Extract a subgraph based on the specified inputs and outputs.
    """
    def __init__(self):
        self.name = "extract"


    def add_parser_args(self, parser):
        parser.add_argument("-o", "--output", required=True, help="Path at which to write the ONNX model including only the subgraph")

        parser.add_argument("--inputs", dest="input_meta", help="Input metadata for subgraph (names, shapes, and data types). "
                            "Use 'auto' to make `extract` determine these automatically. Format: "
                            "--inputs <name>,<shape>,<dtype>. "
                            "For example: --inputs input0,1x3x224x224,float32 input1,auto,auto. "
                            "If omitted, uses the current model inputs. Supported data types are: {:}".format(list(misc.NP_TYPE_FROM_STR.keys())),
                            nargs="+", default=None)

        parser.add_argument("--outputs", dest="output_meta", help="Output metadata for subgraph (names and data types). "
                            "Use 'auto' to make `extract` determine these automatically. Format: "
                            "--outputs <name>,<dtype>. "
                            "For example: --outputs output0:float32 output1:auto. "
                            "If omitted, uses the current model outputs. Supported data types are: {:}".format(list(misc.NP_TYPE_FROM_STR.keys())),
                            nargs="+", default=None)

        super().add_parser_args(parser, gs=True, inputs="--model-inputs", shape_inference_default=True, data=True)


    def __call__(self, args):
        def missing_meta_tensors(input_metadata, output_metadata):
            names = []
            for name, (dtype, shape) in input_metadata.items():
                if dtype is None or not shape:
                    names.append(name)
            for name, (dtype, shape) in output_metadata.items():
                if dtype is None:
                    names.append(name)
            return names


        def update_meta_from_tensor_map(meta, tensor_map):
            for name, (dtype, shape) in meta.items():
                tensor = tensor_map[name]
                meta[name] = (dtype or tensor.dtype, shape or tensor.shape)
            return meta


        def meta_from_tensors(tensors):
            meta = TensorMetadata()
            for tensor in tensors:
                meta.add(tensor.name, tensor.dtype, tensor.shape)
            return meta


        onnx_model = tool_util.get_onnx_model_loader(args)()
        graph = gs.import_onnx(onnx_model)
        tensor_map = graph.tensors()

        if args.input_meta:
            input_metadata = update_meta_from_tensor_map(args_util.parse_meta(args.input_meta), tensor_map)
        else:
            input_metadata = meta_from_tensors(graph.inputs)

        if args.output_meta:
            output_metadata = update_meta_from_tensor_map(args_util.parse_meta(args.output_meta, includes_shape=False), tensor_map)
        else:
            output_metadata = meta_from_tensors(graph.outputs)

        missing_tensors = missing_meta_tensors(input_metadata, output_metadata)
        if missing_tensors:
            # Use ONNX runtime with static shapes to infer shapes when all else fails
            # Returns a TensorMetadata for all tensors in the graph.
            def fallback_shape_inference(onnx_model):
                from polygraphy.backend.onnx import BytesFromOnnx, ModifyOnnx
                from polygraphy.backend.onnxrt import (OnnxrtRunner,
                                                       SessionFromOnnxBytes)

                load_model = ModifyOnnx(onnx_model, outputs=constants.MARK_ALL)
                with OnnxrtRunner(SessionFromOnnxBytes(BytesFromOnnx(load_model))) as runner:
                    data_loader = tool_util.get_data_loader(args)
                    data_loader.input_metadata = runner.get_input_metadata()
                    outputs = runner.infer(feed_dict=data_loader[0])

                    meta = TensorMetadata()
                    for name, output in outputs.items():
                        meta.add(name, output.dtype, output.shape)
                    return meta


            def update_meta_from_meta(meta, golden_meta):
                for name, (dtype, shape) in meta.items():
                    if name in golden_meta:
                        (golden_dtype, golden_shape) = golden_meta[name]
                        meta[name] = (dtype or golden_dtype, shape or golden_shape)
                        G_LOGGER.verbose("Updated tensor: {:} metadata to: {:}".format(name, meta[name]))
                return meta


            G_LOGGER.warning("Some tensor shapes or dtypes are missing in the model. Note: Missing Tensors: {:}. "
                             "\nWill run inference to determine shapes. This will cause dynamic "
                             "dimensions to become static.\nTo avoid this, please provide metadata on the command-line. "
                                .format(missing_tensors))
            golden_meta = fallback_shape_inference(onnx_model)
            input_metadata = update_meta_from_meta(input_metadata, golden_meta)
            output_metadata = update_meta_from_meta(output_metadata, golden_meta)


        # Set the graph inputs and outputs
        graph.inputs.clear()
        for name, (dtype, shape) in input_metadata.items():
            tensor = tensor_map[name]
            tensor.dtype, tensor.shape = dtype, shape
            tensor.inputs.clear()
            graph.inputs.append(tensor)

        graph.outputs.clear()
        for name, (dtype, shape) in output_metadata.items():
            tensor = tensor_map[name]
            tensor.dtype, tensor.shape = dtype, shape
            graph.outputs.append(tensor)

        G_LOGGER.info("Using Graph Inputs:\n{:}{:}".format(constants.TAB, graph.inputs))
        G_LOGGER.info("Using Graph Outputs:\n{:}{:}".format(constants.TAB, graph.outputs))

        if not args.no_cleanup:
            graph.cleanup()
        if not args.no_toposort:
            graph.toposort()

        onnx_model = gs.export_onnx(graph)
        G_LOGGER.info("Writing model to: {output}. To see more details about the model, use: polygraphy inspect model {output} --mode=basic".format(output=args.output))
        onnx.save(onnx_model, args.output)


class STPrepare(STSurgeonBase):
    """
    [EXPERIMENTAL] Prepare a JSON configuration file for a given model,
    which can be edited and provided to `operate`.
    """
    def __init__(self):
        self.name = "prepare"


    def add_parser_args(self, parser):
        parser.add_argument("-o", "--output", help="Path to save JSON configuration for the model. "
                            "If omitted, the JSON configuration is printed to standard output.")
        super().add_parser_args(parser)


    def __call__(self, args):
        graph = super().setup(args)
        config = Config.from_graph(graph)
        config_json = json.dumps(config, indent=constants.TAB)
        G_LOGGER.info("Please do NOT modify the node 'id' values in the configuration file, or things may not work!")

        if args.output:
            with open(args.output, "w") as f:
                f.write(config_json)
        else:
            print(config_json)


class STOperate(STSurgeonBase):
    """
    [EXPERIMENTAL] Modify a model according to the provided JSON configuration file.
    """
    def __init__(self):
        self.name = "operate"


    def add_parser_args(self, parser):
        parser.add_argument("-c", "--config", required=True, help="Path to JSON configuration that specifies how the model should be modified.")
        parser.add_argument("-o", "--output", required=True, help="Path to save the model")
        super().add_parser_args(parser, gs=True)


    def __call__(self, args):
        graph = super().setup(args)
        with open(args.config, "r") as f:
            config = json.loads(f.read())
        G_LOGGER.info("Please ensure you have not modified the node 'id' values in the configuration file, or things may not work!")

        tensor_map = graph.tensors()

        def get_tensor(name):
            if name not in tensor_map:
                G_LOGGER.verbose("Tensor: {:} does not exist in the model. Creating a new tensor".format(name))
                tensor_map[name] = gs.Variable(name)
            return tensor_map[name]


        def tensors_from_names(names):
            tensors = []
            for name in names:
                tensors.append(get_tensor(name))
            return tensors


        def tensors_from_meta(meta, shape_optional=False):
            tensors = []
            for tensor_meta in meta:
                tensor = get_tensor(tensor_meta["name"])
                if "shape" in tensor_meta:
                    tensor.shape = tensor_meta["shape"]
                elif not shape_optional:
                    G_LOGGER.critical("Could not find shape information for tensor: {:}".format(tensor.name))

                if "dtype" in tensor_meta:
                    tensor.dtype = misc.NP_TYPE_FROM_STR[tensor_meta["dtype"]]
                tensors.append(tensor)
            return tensors


        graph.inputs = tensors_from_meta(config["graph_inputs"])
        for inp in graph.inputs:
            # Need to disconnect inputs of graph inputs, or nodes prior to them will remain
            inp.inputs.clear()

        graph.outputs = tensors_from_meta(config["graph_outputs"], shape_optional=True)

        nodes = []
        for node_info in config["nodes"]:
            if node_info["id"] > len(graph.nodes):
                G_LOGGER.critical("Could not find node with ID: {:}. Were the node IDs modified in the config file?".format(node_info["id"]))
            node = graph.nodes[node_info["id"]]
            node.name = node_info["name"]
            node.op = node_info["op"]
            node.inputs = tensors_from_names(node_info["inputs"])
            node.outputs = tensors_from_names(node_info["outputs"])
            nodes.append(node)

        graph.nodes = nodes

        if not args.no_cleanup:
            graph.cleanup()
        if not args.no_toposort:
            graph.toposort()

        onnx.save(gs.export_onnx(graph), args.output)


################################# MAIN TOOL #################################

class Surgeon(Tool):
    """
    Modify models.
    """
    def __init__(self):
        self.name = "surgeon"


    def add_parser_args(self, parser):
        subparsers = parser.add_subparsers(title="Surgical Instruments", dest="instrument")
        subparsers.required = True

        SURGEON_SUBTOOLS = [
            STExtract(),
            STPrepare(),
            STOperate(),
        ]

        for subtool in SURGEON_SUBTOOLS:
            subtool.setup_parser(subparsers)


    def __call__(self, args):
        pass
