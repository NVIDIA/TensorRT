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
from polygraphy import mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools import util as tools_util
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, OnnxInferShapesArgs, OnnxLoadArgs, OnnxSaveArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")
gs = mod.lazy_import("onnx_graphsurgeon")


class ConstFoldArgs(BaseArgs):
    """
    Constant Folding: folding constants

    Depends on:
        OnnxInferShapesArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--fold-constants",
            help="Fold constants in the graph by computing subgraphs whose values "
            "are not dependent on runtime inputs.",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--num-passes",
            "--num-const-fold-passes",
            help="The number of constant folding passes to run. "
            "Sometimes, subgraphs that compute tensor shapes may not be foldable in a single pass. "
            "If not specified, Polygraphy will automatically determine the number of passes required. ",
            type=int,
            default=None,
            dest="num_const_fold_passes",
        )
        self.group.add_argument(
            "--partitioning",
            help="Controls how to partition the graph during constant folding: {{"
            "'basic': Partition the graph so failures in one part do not affect other parts, "
            "'recursive': In addition to partitioning the graph, partition partitions where needed}} ",
            choices=["basic", "recursive"],
            default=None,
        )
        self.group.add_argument(
            "--no-fold-shapes",
            help="Disable folding Shape nodes and subgraphs that operate on shapes",
            dest="fold_shapes",
            default=None,
            action="store_false",
        )
        self.group.add_argument(
            "--no-per-pass-shape-inference",
            help="Disable shape inference between passes of constant folding",
            dest="per_pass_shape_inference",
            default=None,
            action="store_false",
        )
        self.group.add_argument(
            "--fold-size-threshold",
            help="""
            The maximum per-tensor size threshold, in bytes, for which to apply constant folding.
            Any nodes generating tensors larger than this size will not be folded away.
            For example, some models may apply ops like `Tile` or `Expand` to constants, which can
            result in very large tensors. Rather than pre-computing those constants and bloating
            the model size, it may be desirable to skip folding them and allow them to be computed
            at runtime.
            Optionally, use a `K`, `M`, or `G` suffix to indicate KiB, MiB, or GiB respectively.
            For example, `--fold-size-threshold 16M` is equivalent to `--fold-size-threshold 16777216`.
            """,
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            fold_constants (bool): Whether to apply constant folding.
            num_passes (int): The number of constant folding passes to apply.
            partitioning (str): The partitioning mode to use when constant folding.
            fold_shapes (bool): Whether to allow shape folding.
            per_pass_shape_inference (bool): Whether to apply shape inference between constant folding passes.
            size_threshold (int): The threshold in bytes over which to avoid folding constants.
        """
        self.fold_constants = args_util.get(args, "fold_constants")
        self.num_passes = args_util.get(args, "num_const_fold_passes")
        self.partitioning = args_util.get(args, "partitioning")
        self.fold_shapes = args_util.get(args, "fold_shapes")
        self.per_pass_shape_inference = args_util.get(args, "per_pass_shape_inference")
        self.size_threshold = args_util.parse_num_bytes(args_util.get(args, "fold_size_threshold"))

        if not self.fold_constants:
            for arg in [
                "num_const_fold_passes",
                "partitioning",
                "fold_shapes",
                "per_pass_shape_inference",
                "fold_size_threshold",
            ]:
                val = args_util.get(args, arg)
                if val is not None:
                    G_LOGGER.warning(
                        f"Argument: '--{arg.replace('_', '-')}' will be ignored since constant folding is not enabled.\n"
                        "This argument is only valid when the `--fold-constants` option is provided."
                    )

    def add_to_script_impl(self, script, loader_name):
        if not self.fold_constants:
            return loader_name

        script.add_import(imports=["FoldConstants"], frm="polygraphy.backend.onnx")
        loader_name = script.add_loader(
            make_invocable(
                "FoldConstants",
                loader_name,
                num_passes=self.num_passes,
                do_shape_inference=self.arg_groups[OnnxInferShapesArgs].do_shape_inference
                if self.per_pass_shape_inference is not False  # since `None` indicates default value
                else False,
                fold_shapes=self.fold_shapes,
                partitioning=self.partitioning,
                size_threshold=self.size_threshold,
                allow_onnxruntime_shape_inference=self.arg_groups[OnnxInferShapesArgs].allow_onnxruntime,
            ),
            "fold_constants",
        )
        return loader_name

    def fold(self, model):
        loader = args_util.run_script(self.add_to_script, model)
        return util.invoke_if_callable(loader)[0]


class Sanitize(BaseSurgeonSubtool):
    """
    Clean up, optimize, and/or change input shapes in an ONNX model.
    """

    def __init__(self):
        super().__init__("sanitize")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(
                model_opt_required=True,
                input_shapes_opt_name="override-inputs",
                required_model_type="onnx",
                input_shapes_opt_doc="Override input shapes in the model for the given inputs",
            ),
            DataLoaderArgs(),
            OnnxInferShapesArgs(default=True, allow_force_fallback=True),
            OnnxLoadArgs(outputs_opt_prefix=""),
            OnnxSaveArgs(allow_shape_inference=True, output_opt_required=True),
            ConstFoldArgs(),
        ]

    def add_parser_args_impl(self, parser):
        parser.add_argument(
            "--cleanup",
            help="Run dead layer removal on the graph. This is generally not required if other options are set. ",
            action="store_true",
            default=False,
        )

    def show_start_end_logging_impl(self, args):
        return True

    def run_impl(self, args):
        # First do all processing that requires an ONNX-GraphSurgeon graph, then do everything
        # that operates on the ONNX model. This lets us avoid ONNX-GraphSurgeon import if we don't
        # need it.
        def do_graph_processing(model):
            graph = None
            rerun_shape_inference = False

            def get_graph():
                nonlocal graph
                if graph is None:
                    graph = onnx_backend.gs_from_onnx(model)
                return graph

            user_input_metadata = self.arg_groups[ModelArgs].input_shapes
            if user_input_metadata:
                graph = get_graph()
                graph = tools_util.override_input_shapes(graph, user_input_metadata)
                rerun_shape_inference = True

            if self.arg_groups[OnnxInferShapesArgs].force_fallback:
                _, layerwise_meta = self.arg_groups[OnnxInferShapesArgs].fallback_inference(model)
                graph = get_graph()
                onnx_util.set_shapes_from_layerwise_meta(graph, layerwise_meta)

            if args.cleanup:
                graph = get_graph()
                graph.cleanup()

            if graph is not None:
                model = gs.export_onnx(graph)
            return model, rerun_shape_inference

        def do_model_processing(model):
            model = self.arg_groups[ConstFoldArgs].fold(model)
            return model

        model = super().load_model()
        model, rerun_shape_inference = do_graph_processing(model)

        if rerun_shape_inference:
            model = self.arg_groups[OnnxInferShapesArgs].infer_shapes(model)

        model = do_model_processing(model)
        super().save_model(model)
