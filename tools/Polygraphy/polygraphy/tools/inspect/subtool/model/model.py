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
import contextlib

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    ModelArgs,
    OnnxInferShapesArgs,
    OnnxLoadArgs,
    TfLoadArgs,
    TrtLoadEngineArgs,
    TrtLoadEngineBytesArgs,
    TrtLoadNetworkArgs,
    TrtLoadPluginsArgs,
    TrtOnnxFlagArgs,
)
from polygraphy.tools.base import Tool
from polygraphy.tools.inspect.subtool.model.args import VisualArgs

trt_util = mod.lazy_import("polygraphy.backend.trt.util")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")
onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
tf_util = mod.lazy_import("polygraphy.backend.tf.util")


class Model(Tool):
    """
    Display information about a model, including inputs and outputs, as well as layers and their attributes.
    Pass ``--visual`` to launch an interactive GUI instead of printing text.
    """

    def __init__(self):
        super().__init__("model")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True, input_shapes_opt_name=False),
            TfLoadArgs(allow_artifacts=False, allow_custom_outputs=False),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(outputs_opt_prefix=False),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(allow_custom_outputs=False),
            TrtLoadEngineBytesArgs(),
            TrtLoadEngineArgs(),
            TrtOnnxFlagArgs(),
            VisualArgs(),
        ]

    def add_parser_args_impl(self, parser):
        parser.add_argument(
            "--convert-to",
            "--display-as",
            help="Try to convert the model to the specified format before displaying",
            choices=["trt"],
            dest="display_as",
        )

        parser.add_argument(
            "--show",
            help="Controls what is displayed: {{"
            "'layers': Display basic layer information like name, op, inputs, and outputs, "
            "'attrs': Display all available per-layer attributes; has no effect if 'layers' is not enabled, "
            "'weights': Display all weights in the model; if 'layers' is enabled, also shows per-layer constants"
            "}}. More than one option may be specified",
            choices=["layers", "attrs", "weights"],
            nargs="+",
            default=[],
        )

        parser.add_argument(
            "--list-unbounded-dds",
            help="""
            List all tensors with unbounded Data-Dependent Shapes (DDS).

            Note that listing unbounded DDS only works for models that have been constant folded and have shapes inferred.
            """,
            action="store_true",
            default=None,
            dest="show_unbounded_dds",
        )

        parser.add_argument(
            "--combine-tensor-info",
            help="""
            Set the path to the tensor JSON file to combine information from the file into layers' input and output information.
            This is only supported when --model-type is "engine" and --show includes "layers".
            To get the tensor JSON file, use '--mark-unfused-tensors-as-debug-tensors' and '--save-outputs' when running model with TensorRT.
            """,
            type=str,
            default=None,
            dest="combine_tensor_info",
        )

    def _launch_visual(self, graph_data):
        from polygraphy.tools.inspect.subtool.model.viewer import ModelViewer

        ModelViewer(
            graph_data,
            port=self.arg_groups[VisualArgs].port,
            save_path=self.arg_groups[VisualArgs].save_path,
            model_path=self.arg_groups[ModelArgs].path,
        ).run()

    def run_impl(self, args):
        def show(aspect):
            return aspect in args.show

        def inspect_trt():
            if self.arg_groups[ModelArgs].model_type == "engine":
                with self.arg_groups[TrtLoadEngineArgs].load_engine() as engine:
                    context = engine.create_execution_context()
                    if self.arg_groups[VisualArgs].visual:
                        from polygraphy.tools.inspect.subtool.model.extractors import (
                            graph_data_from_trt_engine,
                        )

                        graph_data = graph_data_from_trt_engine(
                            engine,
                            context,
                            combine_tensor_info=args.combine_tensor_info,
                        )
                        self._launch_visual(graph_data)
                    else:
                        engine_str = trt_util.str_from_engine(
                            engine,
                            context,
                            show_layers=show("layers"),
                            show_attrs=show("attrs"),
                            combine_tensor_info=args.combine_tensor_info,
                        )
                        G_LOGGER.info(f"==== TensorRT Engine ====\n{engine_str}")
            else:
                builder, network, parser = util.unpack_args(
                    self.arg_groups[TrtLoadNetworkArgs].load_network(), 3
                )
                with contextlib.ExitStack() as stack:
                    stack.enter_context(builder)
                    stack.enter_context(network)
                    if parser:
                        stack.enter_context(parser)
                    if self.arg_groups[VisualArgs].visual:
                        from polygraphy.tools.inspect.subtool.model.extractors import (
                            graph_data_from_trt_network,
                        )

                        graph_data = graph_data_from_trt_network(network)
                        self._launch_visual(graph_data)
                    else:
                        network_str = trt_util.str_from_network(
                            network,
                            show_layers=show("layers"),
                            show_attrs=show("attrs"),
                            show_weights=show("weights"),
                        ).strip()
                        G_LOGGER.info(f"==== TensorRT Network ====\n{network_str}")

        def inspect_onnx():
            onnx_model = self.arg_groups[OnnxLoadArgs].load_onnx()
            if self.arg_groups[VisualArgs].visual:
                from polygraphy.tools.inspect.subtool.model.extractors import (
                    graph_data_from_onnx,
                )

                graph_data = graph_data_from_onnx(onnx_model, show_weights=True)
                self._launch_visual(graph_data)
            else:
                model_str = onnx_util.str_from_onnx(
                    onnx_model,
                    show_layers=show("layers"),
                    show_attrs=show("attrs"),
                    show_weights=show("weights"),
                ).strip()
                G_LOGGER.info(f"==== ONNX Model ====\n{model_str}")
                if args.show_unbounded_dds:
                    graph = onnx_backend.gs_from_onnx(onnx_model)
                    unbounded_dds_tensors = onnx_util.get_unbounded_dds_tensors(graph)
                    G_LOGGER.info(
                        f"Found tensors with unbounded DDS: {unbounded_dds_tensors}"
                    )

        def inspect_tf():
            tf_graph, _ = self.arg_groups[TfLoadArgs].load_graph()
            graph_str = tf_util.str_from_graph(
                tf_graph,
                show_layers=show("layers"),
                show_attrs=show("attrs"),
                show_weights=show("weights"),
            ).strip()
            G_LOGGER.info(f"==== TensorFlow Graph ====\n{graph_str}")

        func = None
        if args.display_as == "trt" or self.arg_groups[ModelArgs].model_type.is_trt():
            func = inspect_trt
        elif self.arg_groups[ModelArgs].model_type.is_tf():
            func = inspect_tf
        elif self.arg_groups[ModelArgs].model_type.is_onnx():
            func = inspect_onnx
        if func is None:
            G_LOGGER.critical(
                "Could not determine how to display this model. Maybe you need to specify --display-as?"
            )
        func()
