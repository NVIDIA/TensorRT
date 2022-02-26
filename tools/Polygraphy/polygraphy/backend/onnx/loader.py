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
import copy
import os
import sys
import tempfile

from polygraphy import constants, mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.backend.onnx import util as onnx_util
from polygraphy.logger import G_LOGGER, LogMode

onnx = mod.lazy_import("onnx", version=">=1.8.1")
onnxrt = mod.lazy_import("onnxruntime")
onnxmltools = mod.lazy_import("onnxmltools")
tf = mod.lazy_import("tensorflow", version="<2.0")
tf2onnx = mod.lazy_import("tf2onnx")
tf_util = mod.lazy_import("polygraphy.backend.tf.util", log=False)
gs = mod.lazy_import("onnx_graphsurgeon", version=mod.LATEST_VERSION)
shape_inference = mod.lazy_import("onnx.shape_inference")
external_data_helper = mod.lazy_import("onnx.external_data_helper")


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


class _GSGraphManager(object):
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

    def __init__(self, path, external_data_dir=None):
        """
        Loads an ONNX model from a file.

        Args:
            path (str): The path from which to load the model.

            external_data_dir (str): The directory where external data for the model is stored.
        """
        self.path = path
        self.external_data_dir = external_data_dir

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The ONNX model
        """
        G_LOGGER.info("Loading model: {:}".format(self.path))
        # If external_data_dir is not None, we'll load external data ourselves
        model = onnx.load(self.path, load_external_data=self.external_data_dir is None)
        if self.external_data_dir is not None:
            G_LOGGER.verbose("Loading external data from: {:}".format(self.external_data_dir))
            external_data_helper.load_external_data_for_model(model, self.external_data_dir)
        return model


@mod.export(funcify=True)
class OnnxFromTfGraph(BaseLoader):
    """
    Functor that loads a TensorFlow graph and converts it to ONNX using the tf2onnx converter.
    """

    def __init__(self, graph, opset=None, optimize=None, fold_constant=None):
        """
        Converts a TensorFlow model into ONNX.

        Args:
            graph (Union[Tuple[tf.Graph, Sequence[str]], Callable() -> Tuple[tf.Graph, Sequence[str]]]):
                    A tuple containing a TensorFlow graph and output names or a callable that returns one.


            opset (int): The ONNX opset to use during conversion.
            optimize (bool): Whether to use tf2onnx's graph optimization pass.
            fold_constant (bool):
                    Whether to fold constants in the TensorFlow Graph.
                    Requires that ``optimize`` is also enabled.
                    Defaults to True.
        """
        self._graph = graph
        self.opset = util.default(opset, 11)
        self.fold_constant = util.default(fold_constant, True)
        self.optimize = util.default(optimize, True)

        if self.fold_constant and not self.optimize:
            G_LOGGER.warning(
                "`fold_constant` is enabled, but `optimize` is disabled. Constant folding will not be performed"
            )

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The ONNX model.
        """
        (graph, output_names), _ = util.invoke_if_callable(self._graph)
        input_names = list(tf_util.get_input_metadata(graph).keys())

        if self.fold_constant:
            G_LOGGER.info("Folding constants in graph using tf2onnx.tfonnx.tf_optimize")
        graphdef = graph.as_graph_def()
        if self.optimize:
            graphdef = tf2onnx.tfonnx.tf_optimize(
                input_names, output_names, graph.as_graph_def(), fold_constant=self.fold_constant
            )

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
        try:
            model = onnxmltools.utils.float16_converter.convert_float_to_float16(
                model, keep_io_types=True, disable_shape_inference=True
            )
        except TypeError:  # Using an old version of onnxmltools
            model = onnxmltools.utils.float16_converter.convert_float_to_float16(model)

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
                    If this is set to `False`, errors will be re-raised.
                    Defaults to True.
        """
        super().__init__(model, copy)
        self.num_passes = num_passes
        self.do_shape_inference = util.default(do_shape_inference, True)
        self.partitioning = partitioning
        self.fold_shapes = util.default(fold_shapes, True)
        self.error_ok = util.default(error_ok, True)

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The new ONNX model with constants folded.
        """

        def run_const_fold_pass(model):
            graph = gs_from_onnx(model)
            del model

            try:
                graph.fold_constants(fold_shapes=self.fold_shapes, partitioning=self.partitioning)
            except TypeError as err:  # Using an old version of ONNX-GS
                if self.partitioning:
                    G_LOGGER.critical(
                        "This version of ONNX-GraphSurgeon may not support partitioning the graph.\n"
                        "Please upgrade to a newer version of ONNX-GraphSurgeon or disable partitioning.\n"
                        "Note: Error was:\n{:}".format(err)
                    )
                if self.fold_shapes:
                    G_LOGGER.critical(
                        "This version of ONNX-GraphSurgeon may not support folding shapes.\n"
                        "Please upgrade to a newer version of ONNX-GraphSurgeon or disable shape folding.\n"
                        "Note: Error was:\n{:}".format(err)
                    )

                graph.fold_constants()

            model = gs.export_onnx(graph.cleanup(), do_type_check=False)
            del graph

            if self.fold_shapes and self.do_shape_inference:
                model = infer_shapes(model)
            return model

        if not mod.has_mod(onnxrt):
            G_LOGGER.error(
                "ONNX-Runtime is not installed, so constant folding may be suboptimal or not work at all.\n"
                "Consider installing ONNX-Runtime: {:} -m pip install onnxruntime".format(sys.executable)
            )

        model = self.load()

        prefold_num_nodes = len(model.graph.node)
        postfold_num_nodes = -1
        index = 0

        while (prefold_num_nodes != postfold_num_nodes) and (self.num_passes is None or index < self.num_passes):
            prefold_num_nodes = onnx_util.get_num_nodes(model)

            G_LOGGER.start("Folding Constants | Pass {:}".format(index + 1))
            try:
                model = run_const_fold_pass(model)
            except Exception as err:
                if not self.error_ok:
                    raise
                G_LOGGER.warning(
                    "Constant folding pass failed. Skipping subsequent passes.\nNote: Error was:\n{:}".format(err)
                )
                break
            else:
                postfold_num_nodes = onnx_util.get_num_nodes(model)
                index += 1

                G_LOGGER.finish(
                    "\tTotal Nodes | Original: {:5}, After Folding: {:5} | {:5} Nodes Folded".format(
                        prefold_num_nodes, postfold_num_nodes, prefold_num_nodes - postfold_num_nodes
                    )
                )

        return model


@mod.export(funcify=True)
class InferShapes(BaseLoader):
    """
    Functor that runs shape inference on an ONNX model.
    """

    def __init__(self, model, error_ok=None, external_data_dir=None, save_to_disk_threshold_bytes=None):
        """
        Run shape inference on an ONNX model.

        Args:
            model (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one, or a path to a model.
                    Supports models larger than the 2 GiB protobuf limit.

            error_ok (bool):
                    Whether errors during shape inference should be suppressed. Defaults to True.
            external_data_dir (str):
                    The directory where external data for the model is stored.
                    Only used if the model is provided via a path rather than a loader.
            save_to_disk_threshold_bytes (int):
                    The size in bytes above which a ModelProto will be serialized to the disk
                    before running shape inference.
                    This can be used to work around the 2 GiB protobuf limitation.
                    Defaults to ~2 GiB.
        """
        self._model = model
        self.error_ok = util.default(error_ok, True)
        self.external_data_dir = external_data_dir
        # Subtract a little so we're below the real threshold
        self.save_to_disk_threshold_bytes = util.default(save_to_disk_threshold_bytes, (2 << 30) - 8192)

    def call_impl(self):
        """
        Returns:
            onnx.ModelProto: The new ONNX model with shapes inferred.
        """
        model, _ = util.invoke_if_callable(self._model)
        external_data_dir = self.external_data_dir

        try:
            if isinstance(model, onnx.ModelProto):
                MODEL_SIZE = model.ByteSize()
                if MODEL_SIZE > LARGE_MODEL_THRESHOLD:
                    G_LOGGER.warning(
                        "Attempting to run shape inference on a large model. "
                        "This may require a large amount of memory.\nIf memory consumption becomes too high, "
                        "the process may be killed. You may want to try disabling shape inference in that case. ",
                        mode=LogMode.ONCE,
                    )

                if MODEL_SIZE > self.save_to_disk_threshold_bytes:
                    G_LOGGER.warning(
                        "Model size ({:.3} MiB) exceeds the in-memory size threshold: {:.3} MiB.\n"
                        "The model will be saved to a temporary file before shape inference is run.".format(
                            MODEL_SIZE / (1024.0 ** 2), self.save_to_disk_threshold_bytes / (1024.0 ** 2)
                        ),
                        mode=LogMode.ONCE,
                    )
                    outdir = tempfile.TemporaryDirectory()
                    outpath = os.path.join(outdir.name, "tmp_model.onnx")
                    save_onnx(model, outpath, external_data_path="ext.data")
                    model = outpath
                    external_data_dir = outdir.name

            G_LOGGER.verbose("Starting ONNX shape inference")
            if isinstance(model, onnx.ModelProto):
                model = shape_inference.infer_shapes(model)
            else:
                tmp_path = util.NamedTemporaryFile(prefix="tmp_polygraphy_", suffix=".onnx").name
                G_LOGGER.verbose("Writing shape-inferred model to: {:}".format(tmp_path))
                shape_inference.infer_shapes_path(model, tmp_path)
                # When external_data_dir is unset, use the model's current directory
                model = onnx_from_path(
                    tmp_path, external_data_dir=util.default(external_data_dir, os.path.dirname(model) or None)
                )
            G_LOGGER.verbose("ONNX Shape Inference completed successfully")
        except Exception as err:
            if not self.error_ok:
                raise
            G_LOGGER.warning("ONNX shape inference exited with an error:\n{:}".format(err))
            G_LOGGER.internal_error("ONNX shape inference exited with an error:\n{:}".format(err))

            if not isinstance(model, onnx.ModelProto):
                model = onnx_from_path(model, external_data_dir=self.external_data_dir)
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
                    G_LOGGER.critical("Tensor: {:} does not exist in the model.".format(name))
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
                        "{:} metadata should include shape, but no shape was "
                        "provided for tensor: {:}".format(meta_type, name)
                    )
                if dtype is None:
                    G_LOGGER.warning(
                        "{:} metadata should include data type, but no data type was "
                        "provided for tensor: {:}".format(meta_type, name)
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

            graph.cleanup().toposort()

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
        G_LOGGER.info("Saving ONNX model to: {:}".format(self.path))
        if self.external_data_path is not None:
            G_LOGGER.verbose("Saving external data for ONNX model to: {:}".format(self.external_data_path))
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
