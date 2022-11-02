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
import copy
import os
import sys
import tempfile

from polygraphy import constants, mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.backend.onnx import util as onnx_util
from polygraphy.logger import G_LOGGER, LogMode

onnx = mod.lazy_import("onnx>=1.8.1")
onnxrt = mod.lazy_import("onnxruntime>=1.10.0")
onnxmltools = mod.lazy_import("onnxmltools==1.11.1", requires=["onnxconverter_common==1.12.2"])
tf = mod.lazy_import("tensorflow<2.0")
tf2onnx = mod.lazy_import("tf2onnx")
tf_util = mod.lazy_import("polygraphy.backend.tf.util", log=False)
gs = mod.lazy_import("onnx_graphsurgeon>=0.3.21")
shape_inference = mod.lazy_import("onnx.shape_inference")
external_data_helper = mod.lazy_import("onnx.external_data_helper")
# ONNX-RT's shape inference also requires "sympy", but it is not reported as a dependency,
# so we work around it by checking for it manually.
onnxrt_symbolic_shape_inference = mod.lazy_import("onnxruntime.tools.symbolic_shape_infer>=1.10.0", requires=["sympy"])

LARGE_MODEL_THRESHOLD = 512 << 20  # 512 MiB


class BaseLoadOnnxCopy(BaseLoader):
    """
    Abstract base class for loaders that require loading an ONNX model and potentially
    making a copy.
    """

    def __init__(self, model, copy=None):
        """
        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.

            copy (bool): Whether to create a copy of the model first. Defaults to False.
        """
        self._model = model
        self.copy = util.default(copy, False)

    def load(self):
        model, _ = util.invoke_if_callable(self._model)
        if self.copy:
            model = copy.copy(model)
        return model


class _GSGraphManager:
    """
    Imports an ONNX-GraphSurgeon graph.

    If the provided model is already a graph, the graph is not
    exported to ONNX.
    """

    def __init__(self, model):
        self._model = model

    def __enter__(self):
        model, _ = util.invoke_if_callable(self._model)
        self.USE_GS_GRAPH = isinstance(model, gs.Graph)
        if self.USE_GS_GRAPH:
            self.graph = model.copy()
        else:
            self.graph = gs_from_onnx(model)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.USE_GS_GRAPH:
            self.retval = self.graph
        else:
            self.retval = gs.export_onnx(self.graph, do_type_check=False)


@mod.export(funcify=True)
class GsFromOnnx(BaseLoader):
    """
    Functor that creates an ONNX-GraphSurgeon graph from an ONNX ModelProto.
    """

    def __init__(self, model):
        """
        Creates an ONNX-GraphSurgeon graph from an ONNX ModelProto.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.
        """
        self._model = model

    def call_impl(self):
        """
        Returns:
            onnx_graphsurgeon.Graph: The ONNX-GraphSurgeon representation of the ONNX model
        """
        model, _ = util.invoke_if_callable(self._model)
        return gs.import_onnx(model)


@mod.export(funcify=True)
class OnnxFromPath(BaseLoader):
    """
    Functor that loads an ONNX model from a file.
    """

    def __init__(self, path, external_data_dir=None, ignore_external_data=None):
        """
        Loads an ONNX model from a file.

        Args:
            path (str): The path from which to load the model.

            external_data_dir (str): The directory where external data for the model is stored.
            ignore_external_data (bool):
                    Whether to ignore any external data and just load the model structure without any weights.
                    The model will be usable only for purposes that don't require weights, such as extracting
                    subgraphs or inspecting model structure.
                    This can be useful in cases where external data is not available.
                    Defaults to False.
        """
        self.path = path
        self.external_data_dir = external_data_dir
        self.ignore_external_data = util.default(ignore_external_data, False)

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The ONNX model
        """
        G_LOGGER.info(f"Loading model: {self.path}")
        # If external_data_dir is not None, we'll load external data ourselves
        auto_load_ext_data = self.external_data_dir is None and not self.ignore_external_data
        try:
            model = onnx.load(self.path, load_external_data=auto_load_ext_data)
        except FileNotFoundError:
            if auto_load_ext_data:
                G_LOGGER.warning(
                    "Failed to load model. This could be because external data could not be loaded.\n"
                    "Hint: If you don't need the model weights, try ignoring external data by setting `ignore_external_data=True` "
                    "or using the `--ignore-external-data` command-line option."
                )
            raise

        if self.external_data_dir is not None:
            G_LOGGER.verbose(f"Loading external data from: {self.external_data_dir}")
            external_data_helper.load_external_data_for_model(model, self.external_data_dir)
        return model


@mod.export(funcify=True)
class OnnxFromTfGraph(BaseLoader):
    """
    Functor that loads a TensorFlow graph and converts it to ONNX using the tf2onnx converter.
    """

    def __init__(self, graph, opset=None, optimize=None):
        """
        Converts a TensorFlow model into ONNX.

        Args:
            graph (Union[Tuple[tf.Graph, Sequence[str]], Callable() -> Tuple[tf.Graph, Sequence[str]]]):
                    A tuple containing a TensorFlow graph and output names or a callable that returns one.


            opset (int): The ONNX opset to use during conversion.
            optimize (bool): Whether to use tf2onnx's graph optimization pass.
        """
        self._graph = graph
        self.opset = util.default(opset, 11)
        self.optimize = util.default(optimize, True)

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The ONNX model.
        """
        (graph, output_names), _ = util.invoke_if_callable(self._graph)
        input_names = list(tf_util.get_input_metadata(graph).keys())

        graphdef = graph.as_graph_def()
        if self.optimize:
            graphdef = tf2onnx.tfonnx.tf_optimize(input_names, output_names, graph.as_graph_def())

        with tf.Graph().as_default() as graph, tf.compat.v1.Session(graph=graph) as sess:
            tf.import_graph_def(graphdef, name="")

            onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                graph, input_names=input_names, output_names=output_names, opset=self.opset
            )
            if self.optimize:
                onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)
            return onnx_graph.make_model("model")


@mod.export(funcify=True)
class ModifyOutputs(BaseLoadOnnxCopy):
    """
    Functor that modifies the outputs of an ONNX model.
    """

    def __init__(self, model, outputs=None, exclude_outputs=None, copy=None):
        """
        Modifies outputs of an ONNX model.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.

            outputs (Sequence[str]):
                    Names of tensors to mark as outputs. If provided, this will override the
                    existing model outputs.
                    If a value of `constants.MARK_ALL` is used instead of a list, all tensors in the network are marked.
            exclude_outputs (Sequence[str]):
                    Names of tensors to exclude as outputs. This can be useful in conjunction with
                    ``outputs=constants.MARK_ALL`` to omit outputs.
            copy (bool): Whether to create a copy of the model first. Defaults to False.
        """
        super().__init__(model, copy)
        self.outputs = outputs
        self.exclude_outputs = exclude_outputs

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The ONNX model with modified outputs.
        """
        model = self.load()

        if self.outputs == constants.MARK_ALL:
            G_LOGGER.verbose("Marking all ONNX tensors as outputs")
            model = onnx_util.mark_layerwise(model)
        elif self.outputs is not None:
            model = onnx_util.mark_outputs(model, self.outputs)

        if self.exclude_outputs is not None:
            model = onnx_util.unmark_outputs(model, self.exclude_outputs)

        return model


@mod.export(funcify=True)
class ConvertToFp16(BaseLoadOnnxCopy):
    """
    Functor that converts all floating point tensors in the model to 16-bit precision.
    This is *not* needed in order to use TensorRT's fp16 precision, but may be useful for other backends.
    """

    def __init__(self, model, copy=None):
        """
        Converts all floating point tensors in the model to 16-bit precision.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.
            copy (bool): Whether to create a copy of the model first. Defaults to False.
        """
        super().__init__(model, copy)

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The modified ONNX model.
        """
        model = self.load()

        G_LOGGER.info("Converting float tensors to float16")
        model = onnxmltools.utils.float16_converter.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=True
        )
        return model


@mod.export(funcify=True)
class FoldConstants(BaseLoadOnnxCopy):
    """
    Functor that folds constants in an ONNX model.
    """

    def __init__(
        self,
        model,
        num_passes=None,
        do_shape_inference=None,
        partitioning=None,
        fold_shapes=None,
        copy=None,
        error_ok=None,
        size_threshold=None,
        allow_onnxruntime_shape_inference=None,
    ):
        """
        Fold constants in an ONNX model.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.

            num_passes (int):
                    The number of constant folding passes to run.
                    Sometimes, subgraphs that compute tensor shapes may not be foldable in a single pass.
                    By default, Polygraphy will automatically determine the number of passes required.
            do_shape_inference (bool):
                    Whether to run shape inference in the model between passes.
                    This enables the loader to fold `Shape` nodes.
                    Only effective if `fold_shapes` is True.
                    Defaults to True.
            partitioning (Union[str, None]):
                    Whether/How to partition the graph so that errors in folding one
                    part of a model do not affect other parts. Available modes are:

                    - None: Do not partition the graph. If inference fails, no constants are folded.
                    - 'basic': Partition the graph. If inference fails in one partition, other partitions will remain unaffected.
                    - 'recursive': Parition the graph recursively. If inference fails in a partition, the partition will be further partitioned.

                    Defaults to None.
            fold_shapes (bool):
                    Whether to fold `Shape` nodes in the graph.
                    This requires shapes to be inferred in the graph, and can only fold
                    static shapes.
                    Defaults to True.
            copy (bool):
                    Whether to create a copy of the model first.
                    Defaults to False.
            error_ok (bool):
                    Whether to suppress errors during constant folding.
                    If this is set to ``False``, errors will be re-raised.
                    Defaults to True.
            size_threshold (int):
                    The maximum size threshold, in bytes, for which to fold constants.
                    Any tensors larger than this value will not be folded.
                    Set to ``None`` to disable the size threshold and always fold constants.
                    For example, some models may apply ops like `Tile` or `Expand` to constants, which can
                    result in very large tensors. Rather than pre-computing those constants and bloating
                    the model size, it may be desirable to skip folding them and allow them to be computed
                    at runtime.
                    Defaults to None.
            allow_onnxruntime_shape_inference (bool):
                    Allow ONNX-Runtime's shape inference to be used if available instead of ONNX's
                    shape inference utilities. The former may provide performance or memory usage benefits.
                    Has no effect if ``do_shape_inference`` is False.
                    Defaults to True.
        """
        super().__init__(model, copy)
        self.num_passes = num_passes
        self.do_shape_inference = util.default(do_shape_inference, True)
        self.partitioning = partitioning
        self.fold_shapes = util.default(fold_shapes, True)
        self.error_ok = util.default(error_ok, True)
        self.size_threshold = size_threshold
        self.allow_onnxruntime_shape_inference = allow_onnxruntime_shape_inference

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The new ONNX model with constants folded.
        """

        def run_const_fold_pass(model):
            graph = gs_from_onnx(model)
            del model

            graph.fold_constants(
                fold_shapes=self.fold_shapes, partitioning=self.partitioning, size_threshold=self.size_threshold
            )

            model = gs.export_onnx(graph.cleanup(), do_type_check=False)
            del graph

            if self.fold_shapes and self.do_shape_inference:
                model = infer_shapes(model, allow_onnxruntime=self.allow_onnxruntime_shape_inference)
            return model

        mod.autoinstall(onnxrt)
        if not mod.has_mod("onnxruntime"):
            G_LOGGER.error(
                f"ONNX-Runtime is not installed, so constant folding may be suboptimal or not work at all.\n"
                f"Consider installing ONNX-Runtime: {sys.executable} -m pip install onnxruntime"
            )

        model = self.load()

        prefold_num_nodes = len(model.graph.node)
        postfold_num_nodes = -1
        index = 0

        while (prefold_num_nodes != postfold_num_nodes) and (self.num_passes is None or index < self.num_passes):
            prefold_num_nodes = onnx_util.get_num_nodes(model)

            G_LOGGER.start(f"Folding Constants | Pass {index + 1}")
            try:
                model = run_const_fold_pass(model)
            except Exception as err:
                if not self.error_ok:
                    raise
                G_LOGGER.warning(f"Constant folding pass failed. Skipping subsequent passes.\nNote: Error was:\n{err}")
                break
            else:
                postfold_num_nodes = onnx_util.get_num_nodes(model)
                index += 1

                G_LOGGER.finish(
                    f"{constants.TAB}Total Nodes | Original: {prefold_num_nodes:5}, "
                    f"After Folding: {postfold_num_nodes:5} | {prefold_num_nodes - postfold_num_nodes:5} Nodes Folded"
                )

        return model


@mod.export(funcify=True)
class InferShapes(BaseLoader):
    """
    Functor that runs shape inference on an ONNX model.
    """

    def __init__(
        self,
        model,
        error_ok=None,
        external_data_dir=None,
        save_to_disk_threshold_bytes=None,
        allow_onnxruntime=None,
    ):
        """
        Run shape inference on an ONNX model.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto, str, Callable() -> str]):
                    An ONNX model or a callable that returns one, or a path to a model.
                    Supports models larger than the 2 GiB protobuf limit.

            error_ok (bool):
                    Whether errors during shape inference should be suppressed.
                    Defaults to True.
            external_data_dir (str):
                    The directory where external data for the model is stored.
                    Only used if the model is provided via a path rather than a loader.
            save_to_disk_threshold_bytes (int):
                    The size in bytes above which a ModelProto will be serialized to the disk
                    before running shape inference.
                    This can be used to work around the 2 GiB protobuf limitation.
                    Defaults to ~2 GiB.
            allow_onnxruntime (bool):
                    Allow ONNX-Runtime's shape inference to be used if available instead of ONNX's
                    shape inference utilities. The former may provide performance or memory usage benefits.
                    Defaults to True.
        """
        self._model = model
        self.error_ok = util.default(error_ok, True)
        self.external_data_dir = external_data_dir
        # Subtract a little so we're below the real threshold
        self.save_to_disk_threshold_bytes = util.default(save_to_disk_threshold_bytes, (2 << 30) - 8192)
        self.allow_onnxruntime = util.default(allow_onnxruntime, True)

    def _run_onnx_shape_inference(self, model, external_data_dir):
        if isinstance(model, onnx.ModelProto):
            MODEL_SIZE = model.ByteSize()
            if MODEL_SIZE > LARGE_MODEL_THRESHOLD:
                G_LOGGER.warning(
                    f"Attempting to run shape inference on a large model ({MODEL_SIZE // 1024.0 ** 2} MiB). "
                    "This may require a large amount of memory.\nIf memory consumption becomes too high, "
                    "the process may be killed. You may want to try disabling shape inference in that case. ",
                    mode=LogMode.ONCE,
                )

            if MODEL_SIZE > self.save_to_disk_threshold_bytes:
                G_LOGGER.warning(
                    f"Model size ({MODEL_SIZE / 1024.0 ** 2} MiB) exceeds the in-memory size threshold: "
                    f"{self.save_to_disk_threshold_bytes / 1024.0 ** 2} MiB.\n"
                    f"The model will be saved to a temporary file before shape inference is run.",
                    mode=LogMode.ONCE,
                )
                outdir = tempfile.TemporaryDirectory()
                outpath = os.path.join(outdir.name, "tmp_model.onnx")
                save_onnx(model, outpath, external_data_path="ext.data")
                model = outpath
                external_data_dir = outdir.name

        if isinstance(model, onnx.ModelProto):
            model = shape_inference.infer_shapes(model)
        else:
            tmp_path = util.NamedTemporaryFile(prefix="tmp_polygraphy_", suffix=".onnx").name
            G_LOGGER.verbose(f"Writing shape-inferred model to: {tmp_path}")
            shape_inference.infer_shapes_path(model, tmp_path)
            # In cases where the original model had external data stored in the same directory,
            # the external data directory may not be explicitly specified.
            # In such cases, we need to use the model's directory as the external data path
            # for the newly generated model.
            model = onnx_from_path(
                tmp_path, external_data_dir=util.default(external_data_dir, os.path.dirname(model) or None)
            )
        return model

    def _run_onnxruntime_shape_inference(self, model, external_data_dir):
        if not isinstance(model, onnx.ModelProto):
            model = onnx_from_path(model, external_data_dir=external_data_dir)
        return onnxrt_symbolic_shape_inference.SymbolicShapeInference.infer_shapes(model, auto_merge=True)

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The new ONNX model with shapes inferred.
        """
        model, _ = util.invoke_if_callable(self._model)
        external_data_dir = self.external_data_dir

        G_LOGGER.verbose("Starting shape inference")

        mod.autoinstall(onnxrt_symbolic_shape_inference)
        try:
            if self.allow_onnxruntime and mod.has_mod("onnxruntime.tools.symbolic_shape_infer"):
                G_LOGGER.info(
                    "Inferring shapes in the model with `onnxruntime.tools.symbolic_shape_infer`.\n"
                    "Note: To force Polygraphy to use `onnx.shape_inference` instead, set `allow_onnxruntime=False` or "
                    "use the `--no-onnxruntime-shape-inference` command-line option.",
                    mode=LogMode.ONCE,
                )

                model = self._run_onnxruntime_shape_inference(model, external_data_dir)
            else:
                if self.allow_onnxruntime:
                    G_LOGGER.warning(
                        "Falling back to `onnx.shape_inference` because `onnxruntime.tools.symbolic_shape_infer` could not be loaded.\n"
                        "Note that using ONNX-Runtime for shape inference may be faster and require less memory.\n"
                        "Consider installing ONNX-Runtime or settting POLYGRAPHY_AUTOINSTALL_DEPS=1 in your environment "
                        "variables to allow Polygraphy to do so automatically.",
                        mode=LogMode.ONCE,
                    )
                model = self._run_onnx_shape_inference(model, external_data_dir)
        except Exception as err:
            if not self.error_ok:
                raise
            G_LOGGER.warning(f"ONNX shape inference exited with an error:\n{err}")
            G_LOGGER.internal_error(f"ONNX shape inference exited with an error:\n{err}")

            if not isinstance(model, onnx.ModelProto):
                model = onnx_from_path(model, external_data_dir=external_data_dir)
        else:
            G_LOGGER.verbose("Shape inference completed successfully")

        return model


@mod.export(funcify=True)
class ExtractSubgraph(BaseLoader):
    """
    Functor that extracts a subgraph from an ONNX model.
    """

    def __init__(self, model, input_metadata=None, output_metadata=None, check_meta=None):
        """
        Extracts a subgraph from an ONNX model.

        Args:
            model (Union[Union[onnx.ModelProto, onnx_graphsurgeon.Graph], Callable() -> Union[onnx.ModelProto, onnx_graphsurgeon.Graph]]):
                    An ONNX model or ONNX-GraphSurgeon Graph or a callable that returns one.

            input_metadata (TensorMetadata):
                    Metadata for the inputs of the subgraph.
                    Name, shape, and data type are required.
                    If not provided, the graph outputs are not modified.
            output_metadata (TensorMetadata):
                    Metadata for the outputs of the subgraph.
                    Name and data type are required.
                    If not provided, the graph outputs are not modified.
            check_meta (bool):
                    Whether to check that the provided input and output metadata include
                    all the expected fields.
                    Defaults to True.
        """
        self._model = model
        self.input_metadata = input_metadata
        self.output_metadata = output_metadata
        self.check_meta = util.default(check_meta, True)

    def call_impl(self):
        """
        Returns:
            Union[onnx.ModelProto, onnx_graphsurgeon.Graph]:
                    The new ONNX model or ONNX-GraphSurgeon Graph.
        """
        with _GSGraphManager(self._model) as manager:
            graph = manager.graph
            TENSOR_MAP = graph.tensors()

            def get_tensor(name):
                if name not in TENSOR_MAP:
                    G_LOGGER.critical(f"Tensor: {name} does not exist in the model.")
                return TENSOR_MAP[name]

            def update_tensor(name, dtype, shape):
                tensor = get_tensor(name)
                # No need to update constants
                if isinstance(tensor, gs.Variable):
                    tensor.dtype, tensor.shape = dtype or tensor.dtype, shape or tensor.shape
                return tensor

            def check_meta(name, dtype, shape, meta_type, needs_shape=True):
                if not self.check_meta:
                    return
                if needs_shape and shape is None:
                    G_LOGGER.warning(
                        f"{meta_type} metadata should include shape, but no shape was provided for tensor: {name}"
                    )
                if dtype is None:
                    G_LOGGER.warning(
                        f"{meta_type} metadata should include data type, but no data type was provided for tensor: {name}"
                    )

            if self.input_metadata is not None:
                graph.inputs.clear()
                for name, (dtype, shape) in self.input_metadata.items():
                    tensor = update_tensor(name, dtype, shape)
                    check_meta(name, tensor.dtype, tensor.shape, "Input")
                    tensor.inputs.clear()
                    graph.inputs.append(tensor)

            if self.output_metadata is not None:
                graph.outputs.clear()
                for name, (dtype, shape) in self.output_metadata.items():
                    tensor = update_tensor(name, dtype, shape)
                    check_meta(name, tensor.dtype, tensor.shape, "Output", needs_shape=False)
                    graph.outputs.append(tensor)

            graph.cleanup()

            tensor_map = graph.tensors()
            for tensor in tensor_map.values():
                if isinstance(tensor, gs.Variable) and not tensor.inputs and tensor not in graph.inputs:
                    consumer_nodes = [f"Node: '{node.name}' (Op: {node.op})" for node in tensor.outputs]
                    G_LOGGER.error(
                        f"Tensor: '{tensor.name}' is a variable tensor consumed by: {consumer_nodes}, "
                        "but is not produced by a node or marked as a graph input."
                        f"\nDid you forget to mark a tensor as a graph input? Hint: Try inspecting the resulting model. "
                        f"\nNote: The resulting model will not be valid!"
                    )

        return manager.retval


@mod.export(funcify=True)
class SaveOnnx(BaseLoader):
    """
    Functor that saves an ONNX model to the specified path.
    """

    def __init__(self, model, path, external_data_path=None, size_threshold=None, all_tensors_to_one_file=None):
        """
        Saves an ONNX model to the specified path.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.
            path (str): Path at which to write the ONNX model.
            external_data_path (str):
                    Path to save external data.
                    This is always a relative path; external data is always written to the same
                    directory as the model.
                    Set to an empty string to use the default path.
                    Set to None to disable.
                    Defaults to None.
            size_threshold (int):
                    Tensor size threshold, in bytes, above which tensor data will be
                    stored in the external file.
                    Tensors smaller that this threshold will remain in the ONNX file.
                    Has no effect if external_data_path is not set.
                    Defaults to 1024.
            all_tensors_to_one_file (bool):
                    Whether to write all tensors to one file when saving external data.
                    Has no effect if external_data_path is not set.
                    Defaults to True.
        """
        self._model = model
        self.path = path
        self.external_data_path = external_data_path
        self.size_threshold = size_threshold
        self.all_tensors_to_one_file = all_tensors_to_one_file

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The model, after saving it.
        """
        model, _ = util.invoke_if_callable(self._model)
        G_LOGGER.info(f"Saving ONNX model to: {self.path}")
        if self.external_data_path is not None:
            G_LOGGER.verbose(f"Saving external data for ONNX model to: {self.external_data_path}")
            try:
                external_data_helper.convert_model_to_external_data(
                    model,
                    location=self.external_data_path,
                    all_tensors_to_one_file=util.default(self.all_tensors_to_one_file, True),
                    size_threshold=util.default(self.size_threshold, 1024),
                )
            except TypeError:
                if self.size_threshold is not None:
                    G_LOGGER.warning(
                        "This version of onnx does not support size_threshold in convert_model_to_external_data"
                    )
                external_data_helper.convert_model_to_external_data(
                    model,
                    location=self.external_data_path,
                    all_tensors_to_one_file=util.default(self.all_tensors_to_one_file, True),
                )
        else:
            if self.size_threshold is not None:
                G_LOGGER.warning(
                    "size_threshold is set, but external data path has not been set. "
                    "No external data will be written."
                )
            if self.all_tensors_to_one_file is not None:
                G_LOGGER.warning(
                    "all_tensors_to_one_file is set, but external data path has not been set. "
                    "No external data will be written."
                )

        util.makedirs(self.path)
        onnx.save(model, self.path)
        return model


@mod.export(funcify=True)
class BytesFromOnnx(BaseLoader):
    """
    Functor that serializes an ONNX model.
    """

    def __init__(self, model):
        """
        Serializes an ONNX model.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.
        """
        self._model = model

    def call_impl(self):
        """
        Returns:
            bytes: The serialized model.
        """
        model, _ = util.invoke_if_callable(self._model)
        return model.SerializeToString()


@mod.export(funcify=True)
class OnnxFromBytes(BaseLoader):
    """
    Functor that deserializes an ONNX model.
    """

    def __init__(self, serialized_onnx):
        """
        Deserializes an ONNX model.

        Args:
            serialized_onnx (Union[bytes, Callable() -> bytes]):
                    A serialized ONNX model or a callable that returns one.
        """
        self._serialized_onnx = serialized_onnx

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The ONNX model.
        """
        serialized_onnx, _ = util.invoke_if_callable(self._serialized_onnx)
        model = onnx.ModelProto()
        model.ParseFromString(serialized_onnx)
        return model
