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
from polygraphy.common import TensorMetadata, constants
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import DataLoaderArgs
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool
from polygraphy.tools.util import misc as tools_util
from polygraphy.util import misc


class Extract(BaseSurgeonSubtool):
    """
    Extract a subgraph based on the specified inputs and outputs.
    """
    def __init__(self):
        super().__init__("extract", inputs="--model-inputs", data=True, shape_inference_default=True)


    def add_parser_args(self, parser):
        parser.add_argument("--inputs", dest="input_meta", help="Input metadata for subgraph (names, shapes, and data types). "
                            "Use 'auto' to make `extract` determine these automatically. Format: "
                            "--inputs <name>,<shape>,<dtype>. "
                            "For example: --inputs input0,1x3x224x224,float32 input1,auto,auto. "
                            "If omitted, uses the current model inputs. Supported data types are: {:}".format(list(misc.NP_TYPE_FROM_STR.keys())),
                            nargs="+", default=None)

        parser.add_argument("--outputs", dest="output_meta", help="Output metadata for subgraph (names and data types). "
                            "Use 'auto' to make `extract` determine these automatically. Format: "
                            "--outputs <name>,<dtype>. "
                            "For example: --outputs output0,float32 output1,auto. "
                            "If omitted, uses the current model outputs. Supported data types are: {:}".format(list(misc.NP_TYPE_FROM_STR.keys())),
                            nargs="+", default=None)

        super().add_parser_args(parser, gs=True, output=True)


    def run(self, args):
        onnx_model, graph = super().import_graph(args)
        TENSOR_MAP = graph.tensors()


        def get_tensor(name):
            if name not in TENSOR_MAP:
                G_LOGGER.critical("Tensor: {:} does not exist in the model.".format(name))
            return TENSOR_MAP[name]


        def missing_meta_tensors(input_metadata, output_metadata):
            names = []
            for name, (dtype, shape) in input_metadata.items():
                if dtype is None or not shape:
                    names.append(name)
            for name, (dtype, shape) in output_metadata.items():
                if dtype is None:
                    names.append(name)
            return names


        def update_meta_from_tensor_map(meta):
            for name, (dtype, shape) in meta.items():
                tensor = get_tensor(name)
                meta[name] = (dtype or tensor.dtype, shape or tensor.shape)
            return meta


        def meta_from_tensors(tensors):
            meta = TensorMetadata()
            for tensor in tensors:
                meta.add(tensor.name, tensor.dtype, tensor.shape)
            return meta


        if args.input_meta:
            input_metadata = update_meta_from_tensor_map(tools_util.parse_meta(args.input_meta))
        else:
            input_metadata = meta_from_tensors(graph.inputs)

        if args.output_meta:
            output_metadata = update_meta_from_tensor_map(tools_util.parse_meta(args.output_meta, includes_shape=False))
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
                    data_loader = self.makers[DataLoaderArgs].get_data_loader()
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
            tensor = get_tensor(name)
            tensor.dtype, tensor.shape = dtype, shape
            tensor.inputs.clear()
            graph.inputs.append(tensor)

        graph.outputs.clear()
        for name, (dtype, shape) in output_metadata.items():
            tensor = get_tensor(name)
            tensor.dtype, tensor.shape = dtype, shape
            graph.outputs.append(tensor)

        G_LOGGER.info("Using Graph Inputs:\n{:}{:}".format(constants.TAB, graph.inputs))
        G_LOGGER.info("Using Graph Outputs:\n{:}{:}".format(constants.TAB, graph.outputs))

        super().export_graph(graph, args)
