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
import contextlib

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    ModelArgs,
    OnnxLoaderArgs,
    OnnxShapeInferenceArgs,
    TfLoaderArgs,
    TrtEngineLoaderArgs,
    TrtNetworkLoaderArgs,
    TrtPluginLoaderArgs,
)
from polygraphy.tools.base import Tool

trt_util = mod.lazy_import("polygraphy.backend.trt.util")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")
tf_util = mod.lazy_import("polygraphy.backend.tf.util")


class Model(Tool):
    """
    Display information about a model, including inputs and outputs, as well as layers and their attributes.
    """

    def __init__(self):
        super().__init__("model")
        self.subscribe_args(ModelArgs(model_required=True, inputs=None))
        self.subscribe_args(TfLoaderArgs(artifacts=False, outputs=False))
        self.subscribe_args(OnnxShapeInferenceArgs())
        self.subscribe_args(OnnxLoaderArgs(output_prefix=None))
        self.subscribe_args(TrtPluginLoaderArgs())
        self.subscribe_args(TrtNetworkLoaderArgs(outputs=False))
        self.subscribe_args(TrtEngineLoaderArgs())

    def add_parser_args(self, parser):
        parser.add_argument(
            "--convert-to",
            "--display-as",
            help="Try to convert the model to the specified format before displaying",
            choices=["trt"],
            dest="display_as",
        )
        parser.add_argument(
            "--mode",
            "--layer-info",
            help="Display layers: {{"
            "'none': Display no layer information, "
            "'basic': Display layer inputs and outputs, "
            "'attrs': Display layer inputs, outputs and attributes, "
            "'full': Display layer inputs, outputs, attributes, and weights"
            "}}",
            choices=["none", "basic", "attrs", "full"],
            dest="mode",
            default="none",
        )

    def run(self, args):
        func = None

        if self.arg_groups[ModelArgs].model_type.is_tf():
            func = self.inspect_tf

        if self.arg_groups[ModelArgs].model_type.is_onnx():
            func = self.inspect_onnx

        if self.arg_groups[ModelArgs].model_type.is_trt() or args.display_as == "trt":
            func = self.inspect_trt

        if func is None:
            G_LOGGER.critical("Could not determine how to display this model. Maybe you need to specify --display-as?")

        func(args)

    def inspect_trt(self, args):
        if self.arg_groups[ModelArgs].model_type == "engine":
            if args.mode != "none":
                G_LOGGER.warning("Displaying layer information for TensorRT engines is not currently supported")

            with self.arg_groups[TrtEngineLoaderArgs].load_serialized_engine() as engine:
                engine_str = trt_util.str_from_engine(engine)
                G_LOGGER.info("==== TensorRT Engine ====\n{:}".format(engine_str))
        else:
            builder, network, parser = util.unpack_args(self.arg_groups[TrtNetworkLoaderArgs].load_network(), 3)
            with contextlib.ExitStack() as stack:
                stack.enter_context(builder)
                stack.enter_context(network)
                if parser:
                    stack.enter_context(parser)
                network_str = trt_util.str_from_network(network, mode=args.mode).strip()
                G_LOGGER.info("==== TensorRT Network ====\n{:}".format(network_str))

    def inspect_onnx(self, args):
        onnx_model = self.arg_groups[OnnxLoaderArgs].load_onnx()
        model_str = onnx_util.str_from_onnx(onnx_model, mode=args.mode).strip()
        G_LOGGER.info("==== ONNX Model ====\n{:}".format(model_str))

    def inspect_tf(self, args):
        tf_graph, _ = self.arg_groups[TfLoaderArgs].load_graph()
        graph_str = tf_util.str_from_graph(tf_graph, mode=args.mode).strip()
        G_LOGGER.info("==== TensorFlow Graph ====\n{:}".format(graph_str))
