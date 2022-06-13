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

import math

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.tools import util as tools_util
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, OnnxInferShapesArgs, OnnxLoadArgs, OnnxSaveArgs
from polygraphy.tools.base import Tool
from polygraphy.tools.debug.subtool.iterative_debug_args import ArtifactSortArgs, IterativeDebugArgs, CheckCmdArgs

gs = mod.lazy_import("onnx_graphsurgeon")
onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")


class MarkerBase:
    """
    Controls how layers are marked for reduction.
    """

    def __init__(self, num_nodes, node_index):
        self.num_nodes = num_nodes
        self.iteration = 0

        self.node_index = node_index

        # The node index value that leads to the fewest number of nodes but still fails.
        self.best_bad_node_index = None
        self._least_bad_nodes = self.num_nodes + 1

        # Maps num_nodes to node_index for every success. At the end, we can figure out which one is the
        # highest value that's still smaller than _least_bad_nodes.
        self._good_node_indices = {}
        self.best_good_node_index = None

    def step(self, success, num_nodes):
        self.iteration += 1
        if not success and num_nodes <= self._least_bad_nodes:
            self._least_bad_nodes = num_nodes
            self.best_bad_node_index = self.node_index

        if success:
            self._good_node_indices[num_nodes] = self.node_index

    def _clamp(self, x, min_val, max_val):
        return max(min(x, max_val), min_val)

    def finish(self):
        # Find the index of the node that has the highest number of nodes less than _least_bad_nodes, but still is successful.
        # Failing that, use the smallest possible subgraph (which will always be > _least_bad_nodes)
        def split_good(cond):
            return {num: idx for num, idx in self._good_node_indices.items() if cond(num)}

        max_smaller_graph = split_good(lambda num: num < self._least_bad_nodes)
        min_larger_graph = split_good(lambda num: num >= self._least_bad_nodes)

        if max_smaller_graph:
            self.best_good_node_index = max_smaller_graph[max(max_smaller_graph)]
        elif min_larger_graph:
            self.best_good_node_index = min_larger_graph[min(min_larger_graph)]


class LinearMarker(MarkerBase):
    def __init__(self, num_nodes, invert=False):
        super().__init__(num_nodes, node_index=num_nodes - 1 if not invert else 0)
        self.invert = invert

    def step(self, success, num_nodes):
        super().step(success, num_nodes)

        self.node_index += -1 if not self.invert else 1
        return self.node_index

    def stop(self):
        return (self.node_index < 0) or (self.node_index >= self.num_nodes)

    def remaining(self):
        return self.num_nodes - self.iteration


class BisectMarker(MarkerBase):
    def __init__(self, num_nodes, invert=False):
        # Assume the original model doesn't work, and start right in the middle.
        super().__init__(num_nodes, node_index=num_nodes // 2)

        self.good = 0
        self.bad = self.num_nodes

        if invert:
            self.good, self.bad = self.bad, self.good

    # Take a step in bisection.
    # This will return the index of the next node to try depending on the status of the previous run.
    def step(self, success, num_nodes):
        super().step(success, num_nodes)

        if success:
            self.good = self.node_index
            round_func = math.ceil
        else:
            self.bad = self.node_index
            round_func = math.floor

        self.node_index = round_func((self.good + self.bad) / 2.0)
        return self.node_index

    def stop(self):
        return abs(self.good - self.bad) <= 1

    def remaining(self):
        return int(math.log2(self.num_nodes) - self.iteration)


class Reduce(Tool):
    r"""
    [EXPERIMENTAL] Reduce a failing ONNX model to the minimum set of nodes that cause the failure.

    `debug reduce` follows the same general process as other `debug` subtools.
    Specifically, it does the following during each iteration:

    1. Generates a successively smaller subgraph of a given ONNX model and saves it in the
        current directory as `polygraphy_debug.onnx` by default.

    2. Evaluates it using one of two methods:
        a. In an automated fashion, if a `--check` command was provided.
        b. In an interactive fashion otherwise. In interactive mode, the tool will prompt you to report whether
            the iteration passed or failed.
       In either case, if the iteration fails, it further reduces the model during the subsequent iteration.
       Otherwise, it expands the model to include more nodes from the original.

    3. When the model cannot be reduced further, it saves it to the path specfied by `--output`.

    4. Optionally, as with other `debug` subtools, it can track and sort additional files specified by `--artifacts`.

    NOTE: When your model includes dynamic input shapes, it is generally a good idea to tell `debug reduce` what
        shapes to use with the `--model-input-shapes` argument. Further, if your model uses shape operations,
        you should freeze the input shapes and then fold the shape operations with:
            `polygraphy surgeon sanitize --fold-constants --override-input-shapes <static_input_shapes>`

    The typical usage of `debug reduce` is:

        polygraphy debug reduce <onnx_model> --output <reduced_model> \
            [--check <check_command>]

    `polygraphy run` is usually a good choice for the `--check` command.
    """

    def __init__(self):
        super().__init__("reduce")

    def get_subscriptions(self):
        return [
            CheckCmdArgs(),
            ArtifactSortArgs(allow_no_artifacts_warning=False),
            IterativeDebugArgs(iter_art_opt_default="polygraphy_debug.onnx"),
            ModelArgs(model_opt_required=True, input_shapes_opt_name="model-inputs", required_model_type="onnx"),
            OnnxSaveArgs(),
            OnnxInferShapesArgs(default=True, allow_force_fallback=True),
            OnnxLoadArgs(outputs_opt_prefix=False),
            DataLoaderArgs(),  # For fallback shape inference
        ]

    def add_parser_args(self, parser):
        parser.add_argument(
            "--min-good",
            "--minimal-good",
            dest="min_good",
            help="Path at which to save an ONNX model close in size to the reduced model "
            "that does not have the failure. This is not guaranteed to be generated.",
        )

        disable_passes = parser.add_mutually_exclusive_group()
        disable_passes.add_argument(
            "--no-reduce-inputs",
            help="Do not attempt to change the graph inputs to reduce the model further. "
            "'reduce' will then only attempt to find the earliest failing outputs. ",
            action="store_false",
            dest="reduce_inputs",
        )
        disable_passes.add_argument(
            "--no-reduce-outputs",
            help="Do not attempt to change the graph outputs to reduce the model further. "
            "'reduce' will then only attempt to find the latest failing inputs. ",
            action="store_false",
            dest="reduce_outputs",
        )

        parser.add_argument(
            "--mode",
            help="Strategy to use to iteratively remove nodes from the model. "
            "'bisect' will use binary search, and 'linear' will delete one node at a time. "
            "'linear' mode may be significantly slower, but can offer better results in models with branches. "
            "One strategy is to use 'bisect' first, and then further reduce the result with 'linear'. "
            "Defaults to 'bisect'.",
            choices=["bisect", "linear"],
            default="bisect",
        )

    def run(self, args):
        if not self.arg_groups[OnnxSaveArgs].path and not args.min_good:
            G_LOGGER.critical(
                "--output (where to write the reduced model) and/or "
                "--min-good (where to write a reduced model that passes) must be provided!"
            )

        model = self.arg_groups[OnnxLoadArgs].load_onnx()
        num_orig_nodes = len(model.graph.node)

        # When --model-input-shapes are set, we need to override the shapes in the model, and then run
        # shape inference to figure out the new shapes of intermediate tensors.
        user_input_metadata = self.arg_groups[ModelArgs].input_shapes
        if user_input_metadata:
            model = gs.export_onnx(
                tools_util.override_input_shapes(onnx_backend.gs_from_onnx(model), user_input_metadata)
            )
            if self.arg_groups[OnnxInferShapesArgs].do_shape_inference:
                model = onnx_backend.infer_shapes(model)

        # Lower Constant nodes into Constant tensors
        # If we don't do this, the outputs of Constant nodes may be incorrectly marked
        #   as variable inputs. Further, fallback shape inference does not apply to Constant nodes.
        GRAPH = onnx_util.lower_constant_nodes(onnx_backend.gs_from_onnx(model))

        _layerwise_outputs = None
        _layerwise_meta = None
        # Get metadata inferred by fallback shape inference. If fallback shape inference was
        # never run, then this function runs it.
        def layerwise(model, include_data=False):
            nonlocal _layerwise_outputs, _layerwise_meta
            if _layerwise_outputs is None or _layerwise_meta is None:
                G_LOGGER.info(
                    "Running inference with ONNX-Runtime to determine metadata for intermediate tensors.\n"
                    "This will cause intermediate models to have static shapes."
                )
                with G_LOGGER.indent():
                    _layerwise_outputs, _layerwise_meta = self.arg_groups[OnnxInferShapesArgs].fallback_inference(model)
            return _layerwise_outputs if include_data else _layerwise_meta

        if self.arg_groups[OnnxInferShapesArgs].force_fallback:
            G_LOGGER.info("Freezing shapes in the model according to values determined by fallback shape inference")
            onnx_util.set_shapes_from_layerwise_meta(GRAPH, layerwise(model))

        if any(util.is_shape_dynamic(inp.shape) for inp in GRAPH.inputs):
            G_LOGGER.warning(
                "This model uses dynamic input shapes.\n"
                "You may want to provide input shapes to `debug reduce` using the "
                "`--model-input-shapes` option to prevent unexpected behavior.\n"
            )
        elif any(tensor.shape is None or util.is_shape_dynamic(tensor.shape) for tensor in GRAPH.tensors().values()):
            msg = ""
            if self.arg_groups[OnnxInferShapesArgs].do_shape_inference:
                msg += "ONNX shape inference was unable to infer some shapes in this model.\n"
                msg += "You may want to use `--force-fallback-shape-inference` to freeze the shapes of intermediate tensors to prevent unexpected behavior."
            elif self.arg_groups[OnnxInferShapesArgs].force_fallback:
                msg += "Fallback shape inference was unable to infer some shapes in this model.\n"
                msg += "The shapes for those tensors will remain dynamic. Please ensure that your `--check` command can handle this."
            else:
                msg += "Shape inference was not run on this model.\n"
                msg += "You may want to enable shape inference to freeze the shapes of intermediate tensors to prevent unexpected behavior."
            G_LOGGER.warning(msg)

        if any(node.op == "Shape" for node in GRAPH.nodes):
            G_LOGGER.warning(
                "This model includes shape operations, which may cause issues while reducing.\n"
                "You may want to freeze the input shapes and fold the shape operations away with:\n"
                "\t`polygraphy surgeon sanitize --override-input-shapes <shapes> --fold-constants [--force-fallback-shape-inference]`\n"
                "You only need to use `--force-fallback-shape-inference` if ONNX shape inference is unable to infer shapes."
            )

        def fix_graph(graph, model):
            """
            Fix the graph so it is valid ONNX.
            """

            def fix_tensor_metadata(tensors, fix_shape=True):
                for tensor in tensors:
                    if not tensor.shape and fix_shape:
                        tensor.shape = layerwise(model)[tensor.name].shape
                    if not tensor.dtype:
                        tensor.dtype = layerwise(model)[tensor.name].dtype

            fix_tensor_metadata(graph.inputs)
            fix_tensor_metadata(graph.outputs, fix_shape=False)

            # If we're marking inputs, there may be cases where some other inputs are required - for
            # example, if the model is branchy. If, after cleanup(), there are any Variable tensors in
            # the graph without inputs, we'll replace them with constants and fold them away.
            tensor_map = graph.tensors()
            needs_const_fold = False
            for tensor in tensor_map.values():
                if isinstance(tensor, gs.Variable) and not tensor.inputs and tensor not in graph.inputs:
                    needs_const_fold = True
                    if self.arg_groups[DataLoaderArgs].is_using_random_data():
                        G_LOGGER.warning(
                            "This model includes multiple branches/paths. In order to continue reducing, one branch needs to be folded away.\n"
                            "Please ensure that you have provided a data loader argument directly to `debug reduce` (i.e. prior to `--check`) "
                            "if your `--check` command is using a non-default data loader.\n"
                            "Not doing so may result in false negatives!\n",
                            mode=LogMode.ONCE,
                        )
                    G_LOGGER.info(f"Freezing tensor: {tensor} to eliminate branches.")
                    tensor.to_constant(layerwise(model, include_data=True)[tensor.name])

            if needs_const_fold:
                G_LOGGER.verbose("Folding constants to remove extraneous subgraphs")
                graph.fold_constants().cleanup()

            return graph

        def mark_io(graph, attr, tensors, filter_const=True):
            if filter_const:
                tensors = [t for t in tensors if not isinstance(t, gs.Constant)]

            if not tensors:
                G_LOGGER.warning(
                    "No non-constant tensors are available to mark. "
                    "Try folding constants in the model with `polygraphy surgeon sanitize --fold-constants`"
                )

            setattr(graph, attr, tensors)
            G_LOGGER.info(f"Marking model {attr}: {getattr(graph, attr)}")
            return graph

        def names_from_tensors(tensors):
            return [t.name for t in tensors]

        def lookup_tensors(graph, names):
            tensor_map = graph.tensors()
            return [tensor_map[name] for name in names]

        # Bisect using the given marker, and modifying the given graph attribute.
        # attr should be one of ["inputs", "outputs"].
        # filter_const indicates whether to filter out constant tensors before updating graph I/O.
        # debug_replay is used to provide the debug_replay from previous iterations to subsequent iterations.
        #   Without this, the debug_replay would only contain entries for the final call to `bisect_io`.
        def bisect_io(graph, model, marker, attr, filter_const=True, debug_replay=None):
            G_LOGGER.start(f"Reducing model {attr}")

            def make_iter_art():
                iter_graph = graph.copy()  # This is a very light-weight copy of the entire graph.

                with G_LOGGER.indent():
                    io_list = list(getattr(iter_graph.nodes[marker.node_index], attr))
                    mark_io(iter_graph, attr, io_list, filter_const)
                    iter_graph.cleanup()
                    self.arg_groups[OnnxSaveArgs].save_onnx(
                        gs.export_onnx(fix_graph(iter_graph, model)),
                        self.arg_groups[IterativeDebugArgs].iter_artifact_path,
                    )

                return len(iter_graph.nodes)

            def advance(index, success, num_nodes):
                marker.step(success, num_nodes)
                if marker.stop():
                    self.arg_groups[IterativeDebugArgs].stop_iteration()

            debug_replay = self.arg_groups[IterativeDebugArgs].iterate(
                make_iter_art_func=make_iter_art,
                advance_func=advance,
                get_remaining_func=lambda: marker.remaining(),
                suffix=f"{attr}",
                initial_debug_replay=debug_replay,
            )

            marker.finish()
            G_LOGGER.finish(f"Finished reducing model {attr}")

            # Find minimal good/bad inputs/outputs, falling back to existing graph inputs/outputs.
            def get_io(index):
                if index is None:
                    return names_from_tensors(getattr(graph, attr))
                return names_from_tensors(list(getattr(graph.nodes[index], attr)))

            return get_io(marker.best_bad_node_index), get_io(marker.best_good_node_index), debug_replay

        # We reduce the model in 2 phases:
        #   1. Find the earliest output nodes that cause a failure.
        #   2. Find the latest input nodes cause a failure.

        MarkerType = BisectMarker if args.mode == "bisect" else LinearMarker

        bad_graph = GRAPH.copy()

        good_graph = None
        if args.min_good:
            good_graph = GRAPH.copy()

        # == Phase 1 ==

        debug_replay = None
        if args.reduce_outputs:
            out_marker = MarkerType(len(bad_graph.nodes))
            bad_outputs, good_outputs, debug_replay = bisect_io(
                bad_graph, model, out_marker, attr="outputs", filter_const=False, debug_replay=debug_replay
            )
            bad_graph = mark_io(bad_graph, "outputs", lookup_tensors(bad_graph, bad_outputs)).cleanup()
            if good_graph is not None:
                good_graph = mark_io(
                    good_graph, "outputs", lookup_tensors(good_graph, good_outputs)
                )  # Defer cleanup where possible.
            # Export the model with the reduced outputs so that reducing inputs is faster.
            model = gs.export_onnx(fix_graph(bad_graph, model))

        # == Phase 2 ==

        if args.reduce_inputs:
            in_marker = MarkerType(len(bad_graph.nodes), invert=True)
            bad_inputs, good_inputs, debug_replay = bisect_io(
                bad_graph, model, in_marker, attr="inputs", debug_replay=debug_replay
            )
            bad_graph = mark_io(bad_graph, "inputs", lookup_tensors(bad_graph, bad_inputs)).cleanup()
            if good_graph is not None:
                good_graph = mark_io(
                    good_graph, "inputs", lookup_tensors(good_graph, good_inputs)
                )  # Defer cleanup where possible.

        # == Write Bad Model ==

        reduced_model = gs.export_onnx(fix_graph(bad_graph, model))

        if self.arg_groups[OnnxSaveArgs].path:
            num_reduced_nodes = len(reduced_model.graph.node)

            if (
                float(num_reduced_nodes) / float(num_orig_nodes) >= 0.25
                and num_reduced_nodes > 1
                and args.mode == "bisect"
            ):
                G_LOGGER.warning(
                    f"It looks like this model could potentially be reduced further.\nYou may want to reduce {self.arg_groups[OnnxSaveArgs].path} again using --mode=linear. "
                )

            G_LOGGER.info(f"Minimum Bad Model:\n{onnx_util.str_from_onnx(reduced_model)}\n\n")
            self.arg_groups[OnnxSaveArgs].save_onnx(reduced_model)

        # == Write Good Model ==

        if good_graph is not None:
            min_good_model = gs.export_onnx(fix_graph(good_graph.cleanup(), model))
            if min_good_model == reduced_model:
                G_LOGGER.warning(
                    "Could not find a minimal model close in size to the reduced model that does not cause a failure."
                )
            else:
                G_LOGGER.info(f"Minimum Good Model:\n{onnx_util.str_from_onnx(min_good_model)}\n\n")
                self.arg_groups[OnnxSaveArgs].save_onnx(min_good_model, args.min_good)
