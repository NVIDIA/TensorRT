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
from polygraphy.common import func
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (ModelArgs, OnnxLoaderArgs, TfLoaderArgs,
                                   TrtLoaderArgs)
from polygraphy.tools.base import Tool


class Model(Tool):
    """
    Display information about a model, including inputs and outputs, as well as layers and their attributes.
    """
    def __init__(self):
        super().__init__("model")
        self.subscribe_args(ModelArgs(model_required=True, inputs=None))
        self.subscribe_args(TfLoaderArgs(tftrt=False, artifacts=False, outputs=False))
        self.subscribe_args(OnnxLoaderArgs(outputs=False))
        self.subscribe_args(TrtLoaderArgs(config=False, outputs=False))


    def add_parser_args(self, parser):
        parser.add_argument("--convert-to", "--display-as", help="Convert the model to the specified format before displaying",
                            choices=["trt"], dest="display_as")
        parser.add_argument("--mode", "--layer-info", help="Display layers: {{"
                            "'none': Display no layer information, "
                            "'basic': Display layer inputs and outputs, "
                            "'attrs': Display layer inputs, outputs and attributes, "
                            "'full': Display layer inputs, outputs, attributes, and weights"
                            "}}",
                            choices=["none", "basic", "attrs", "full"], dest="mode", default="none")


    def run(self, args):
        func = None

        if self.makers[ModelArgs].model_type in ["frozen", "keras", "ckpt"]:
            func = self.inspect_tf

        if self.makers[ModelArgs].model_type == "onnx":
            func = self.inspect_onnx

        if self.makers[ModelArgs].model_type == "engine" or args.display_as == "trt":
            func = self.inspect_trt

        if func is None:
            G_LOGGER.critical("Could not determine how to display this model. Maybe you need to specify --display-as?")
        func(args)


    def inspect_trt(self, args):
        from polygraphy.backend.trt import util as trt_util

        if self.makers[ModelArgs].model_type == "engine":
            if args.mode != "none":
                G_LOGGER.warning("Displaying layer information for TensorRT engines is not currently supported")

            with func.invoke(self.makers[TrtLoaderArgs].get_trt_serialized_engine_loader()) as engine:
                engine_str = trt_util.str_from_engine(engine)
                G_LOGGER.info("==== TensorRT Engine ====\n{:}".format(engine_str))
        else:
            builder, network, parser = func.invoke(self.makers[TrtLoaderArgs].get_trt_network_loader())
            with builder, network, parser:
                network_str = trt_util.str_from_network(network, mode=args.mode).strip()
                G_LOGGER.info("==== TensorRT Network ====\n{:}".format(network_str))


    def inspect_onnx(self, args):
        from polygraphy.backend.onnx import util as onnx_util

        onnx_model = func.invoke(self.makers[OnnxLoaderArgs].get_onnx_loader())
        model_str = onnx_util.str_from_onnx(onnx_model, mode=args.mode).strip()
        G_LOGGER.info("==== ONNX Model ====\n{:}".format(model_str))


    def inspect_tf(self, args):
        from polygraphy.backend.tf import util as tf_util

        tf_graph, _ = func.invoke(self.makers[TfLoaderArgs].get_tf_loader())
        graph_str = tf_util.str_from_graph(tf_graph, mode=args.mode).strip()
        G_LOGGER.info("==== TensorFlow Graph ====\n{:}".format(graph_str))
