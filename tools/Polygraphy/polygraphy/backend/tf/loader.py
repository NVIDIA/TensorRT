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
# Sets up everything needed to perform inference in TensorFlow.
import os

from polygraphy import constants, mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.backend.tf import util as tf_util
from polygraphy.logger import G_LOGGER

tf = mod.lazy_import("tensorflow<2.0")


@mod.export(funcify=True)
class OptimizeGraph(BaseLoader):
    """
    Functor that freezes a TensorFlow graph, and folds constants.
    """

    def __init__(self, graph):
        """
        Freezes a TensorFlow graph and folds constants.

        Args:
            graph (Union[Tuple[tf.Graph, Sequence[str]], Callable() -> Tuple[tf.Graph, Sequence[str]]]):
                    A tuple containing a TensorFlow graph and output names or a callable that returns one.
        """
        self._graph = graph

    def constfold(self, graphdef, output_names):
        from tensorflow.core.protobuf import config_pb2, meta_graph_pb2, rewriter_config_pb2
        from tensorflow.python.framework import importer, ops
        from tensorflow.python.grappler import tf_optimizer
        from tensorflow.python.training import saver

        graph = ops.Graph()
        with graph.as_default():
            output_collection = meta_graph_pb2.CollectionDef()
            output_list = output_collection.node_list.value
            for output in output_names:
                output_list.append(output.encode("utf-8"))

            importer.import_graph_def(graphdef, name="")
            metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
            metagraph.collection_def["train_op"].CopyFrom(output_collection)

        rewriter_config = rewriter_config_pb2.RewriterConfig()
        rewriter_config.optimizers.extend(["constfold"])
        rewriter_config.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE

        session_config = config_pb2.ConfigProto()
        session_config.graph_options.resave_options.CopyFrom(rewriter_config)
        return tf_optimizer.OptimizeGraph(session_config, metagraph, graph_id=b"graph")

    def call_impl(self):
        """
        Returns:
            Tuple[tf.Graph, Sequence[str]]: The TensorFlow graph, and the names of its outputs.
        """
        (graph, output_names), _ = util.invoke_if_callable(self._graph)
        with tf.Session(graph=graph) as sess:
            sess.run(tf.initializers.global_variables())
            sess.run(tf.initializers.local_variables())

            graphdef = sess.graph.as_graph_def()
            removed = tf.graph_util.remove_training_nodes(graphdef)
            G_LOGGER.ultra_verbose(f"Removed nodes: {removed}")

            for node in graphdef.node:
                if node.op == "RefSwitch":
                    node.op = "Switch"
                    for index in range(len(node.input)):
                        if "moving_" in node.input[index]:
                            node.input[index] = node.input[index] + "/read"
                elif node.op == "AssignSub":
                    node.op = "Sub"
                    if "use_locking" in node.attr:
                        del node.attr["use_locking"]
                elif node.op == "AssignAdd":
                    node.op = "Add"
                    if "use_locking" in node.attr:
                        del node.attr["use_locking"]
                elif node.op == "Assign":
                    node.op = "Identity"
                    if "use_locking" in node.attr:
                        del node.attr["use_locking"]
                    if "validate_shape" in node.attr:
                        del node.attr["validate_shape"]
                    if len(node.input) == 2:
                        # input0: ref: Should be from a Variable node. May be uninitialized.
                        # input1: value: The value to be assigned to the variable.
                        node.input[0] = node.input[1]
                        del node.input[1]

            # Strip port information from outputs
            output_names = [name.split(":")[0] for name in output_names]
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_names)
            output_graph_def = self.constfold(output_graph_def, output_names)
            return graph_from_frozen(output_graph_def)


@mod.export(funcify=True)
class GraphFromKeras(BaseLoader):
    """
    Functor that loads a TensorFlow model from Keras.
    """

    def __init__(self, path):
        """
        Loads a TensorFlow model from Keras.

        Args:
            path (Union[str, h5py.File]): A path to the saved model, or the file object.
        """
        self.path = path

    def call_impl(self):
        """
        Returns:
            Tuple[tf.Graph, Sequence[str]]: The TensorFlow graph, and the names of its outputs.
        """

        from tensorflow.python import keras
        from tensorflow.python.keras import backend

        model = keras.models.load_model(self.path)
        graph = backend.get_session().graph
        return graph, tf_util.get_graph_output_names(graph)


@mod.export(funcify=True)
class GraphFromFrozen(BaseLoader):
    """
    Functor that loads a TensorFlow frozen model.
    """

    def __init__(self, path):
        """
        Loads a TensorFlow frozen model.

        Args:
            path (Union[str, tf.Graph, tf.GraphDef]):
                    A path to the frozen model, or a frozen TensorFlow graph or graphdef.
        """
        self.path = path

    def call_impl(self):
        """
        Returns:
            Tuple[tf.Graph, Sequence[str]]: The TensorFlow graph, and the names of its outputs.
        """
        graph = tf_util.load_graph(self.path)
        return graph, tf_util.get_graph_output_names(graph)


@mod.export(funcify=True)
class GraphFromCkpt(BaseLoader):
    """
    Functor that loads a TensorFlow model from a checkpoint. Note that in order to use checkpoints,
    you must NOT use subprocesses in the Comparator.
    """

    def __init__(self, dir, name=None):
        """
        Loads a TensorFlow model from a checkpoint.

        Args:
            dir (str): Path to a directory containing checkpoints.


            name (str):
                    The name of the checkpoint to load, not including the file extension.
                    For example, to load `model.meta`, the argument would be `model`.
        """
        self.dir = dir
        self.name = name

    def call_impl(self):
        """
        Returns:
            Tuple[tf.Graph, Sequence[str]]: The TensorFlow graph, and the names of its outputs.
        """
        # If `name` is not provided, this expects that the directory contains a `checkpoint` file with the contents:
        #
        # model_checkpoint_path: "model"
        # all_model_checkpoint_paths: "model"
        #
        # where "model" is the checkpoint name
        if not os.path.isdir(self.dir):
            G_LOGGER.warning(f"Specified checkpoint directory: {self.dir} does not look like a directory.")

        if self.name is None:
            G_LOGGER.verbose("Checkpoint name was not explicitly provided, searching for `checkpoint` file")
            checkpoint = tf.train.get_checkpoint_state(self.dir)
            if checkpoint is None:
                ckpt_file_contents = '\nmodel_checkpoint_path: "model"\nall_model_checkpoint_paths: "model"\n'
                G_LOGGER.critical(
                    f"Checkpoint directory: {self.dir} does not contain a `checkpoint` file, and the checkpoint name was not provided. Please either create a checkpoint file with the contents:\n{ckpt_file_contents} \nWhere `model` is the name of the checkpoint, or explicitly provide the name with --ckpt, not including file extensions"
                )
            input_checkpoint = checkpoint.model_checkpoint_path
        else:
            input_checkpoint = os.path.join(self.dir, self.name)

        meta_file = input_checkpoint + ".meta"
        with tf.Graph().as_default() as graph, tf.compat.v1.Session(graph=graph).as_default() as sess:
            saver = tf.compat.v1.train.import_meta_graph(meta_file, clear_devices=True)
            saver.restore(sess, input_checkpoint)
            return graph, tf_util.get_graph_output_names(graph)


@mod.export(funcify=True)
class UseTfTrt(BaseLoader):
    """
    [UNTESTED] Functor that optimizes a TensorFlow model using TF-TRT.
    """

    def __init__(
        self,
        graph,
        max_workspace_size=None,
        fp16=None,
        int8=None,
        max_batch_size=None,
        is_dynamic_op=False,
        minimum_segment_size=None,
    ):
        """
        Optimizes a TensorFlow model using TF-TRT.

        Args:
            graph (Union[Tuple[tf.Graph, Sequence[str]], Callable() -> Tuple[tf.Graph, Sequence[str]]]):
                    A tuple containing a TensorFlow graph and output names or a callable that returns one.
            max_workspace_size (int): The maximum workspace size.
            fp16 (bool): Whether to run in FP16 mode.
            max_batch_size (int): The maximum batch size.
        """
        self._graph = graph
        self.max_workspace_size = util.default(max_workspace_size, 1 << 24)
        self.fp16 = util.default(fp16, False)
        self.fp8 = util.default(fp8, False)
        self.int8 = util.default(int8, False)
        self.max_batch_size = util.default(max_batch_size, 1)
        self.is_dynamic_op = is_dynamic_op
        self.minimum_segment_size = util.default(minimum_segment_size, 3)

    def call_impl(self):
        """
        Returns:
            Tuple[tf.Graph, Sequence[str]]: The TensorFlow graph, and the names of its outputs.
        """
        from tensorflow.contrib import tensorrt as tf_trt

        (graph, output_names), _ = util.invoke_if_callable(self._graph)

        precision_mode = "FP16" if self.fp16 else "FP32"
        precision_mode = "INT8" if self.int8 else precision_mode
        precision_mode = "FP8" if self.fp8 else precision_mode

        G_LOGGER.info(
            f"For TF-TRT, using outputs={output_names}, max_workspace_size_bytes={self.max_workspace_size}, max_batch_size={self.max_batch_size}, minimum_segment_size={self.minimum_segment_size}, is_dynamic_op={self.is_dynamic_op}, precision_mode={precision_mode}"
        )

        graphdef = tf_trt.create_inference_graph(
            graph.as_graph_def(),
            outputs=output_names,
            max_workspace_size_bytes=self.max_workspace_size,
            max_batch_size=self.max_batch_size,
            minimum_segment_size=self.minimum_segment_size,
            is_dynamic_op=self.is_dynamic_op,
            precision_mode=precision_mode,
        )

        segment_number = 0
        for node in graphdef.node:
            if node.op == "TRTEngineOp":
                engine = node.attr["serialized_segment"].s
                segment_number += 1
        G_LOGGER.info(f"Found {segment_number} engines in TFTRT graph")

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graphdef, name="")
            return graph, tf_util.get_graph_output_names(graph)


@mod.export(funcify=True)
class ModifyGraphOutputs(BaseLoader):
    """
    Functor that modifies outputs of a TensorFlow graph.
    """

    def __init__(self, graph, outputs=None):
        """
        Modifies outputs of a TensorFlow graph.

        Args:
            graph (Union[Tuple[tf.Graph, Sequence[str]], Callable() -> Tuple[tf.Graph, Sequence[str]]]):
                    A tuple containing a TensorFlow graph and output names or a callable that returns one.


            outputs (List[str]):
                    Names of output tensors. If provided, this will override the outputs
                    determined by the loader.
                    If a value of `constants.MARK_ALL` is used instead of a list, all tensors in the network are marked.
        """
        self._graph = graph
        self.outputs = outputs

    def call_impl(self):
        """
        Returns:
            Tuple[tf.Graph, Sequence[str]]: The TensorFlow graph, and the names of its outputs.
        """
        (graph, outputs), _ = util.invoke_if_callable(self._graph)

        if self.outputs == constants.MARK_ALL:
            outputs = list(tf_util.get_output_metadata(graph, layerwise=True).keys())
        elif self.outputs is not None:
            outputs = self.outputs

        return graph, outputs


@mod.export(funcify=True)
class SaveGraph(BaseLoader):
    """
    Functor that writes out artifacts from a TensorFlow graph.
    """

    def __init__(self, graph, path=None, tensorboard_dir=None, engine_dir=None):
        """
        Writes out artifacts from a TensorFlow Graph.

        Args:
            graph (Union[Tuple[tf.Graph, Sequence[str]], Callable() -> Tuple[tf.Graph, Sequence[str]]]):
                    A tuple containing a TensorFlow graph and output names or a callable that returns one.


            path (str): Path at which to save the frozen graphdef.
            tensorboard_dir (str): The directory in which to write TensorBoard visualizations.
            engine_dir (str): The directory in which to save TF-TRT engines,
        """
        self._graph = graph
        self.path = path
        self.tensorboard_dir = tensorboard_dir
        self.engine_dir = engine_dir

    def call_impl(self):
        """
        Returns:
            Tuple[tf.Graph, Sequence[str]]: The TensorFlow graph, and the names of its outputs.
        """
        (graph, outputs), _ = util.invoke_if_callable(self._graph)

        if self.path:
            util.save_file(graph.as_graph_def().SerializeToString(), dest=self.path)
        if self.tensorboard_dir:
            G_LOGGER.info(f"Writing tensorboard events to {self.tensorboard_dir}")
            train_writer = tf.compat.v1.summary.FileWriter(self.tensorboard_dir)
            train_writer.add_graph(graph)

        if self.engine_dir is not None:
            graphdef = graph.as_graph_def()
            segment_number = 0
            for node in graphdef.node:
                if node.op == "TRTEngineOp":
                    engine = node.attr["serialized_segment"].s
                    if self.engine_dir is not None:
                        util.save_file(contents=engine, dest=os.path.join(self.engine_dir, f"segment-{segment_number}"))
                    segment_number += 1

        return graph, outputs


@mod.export(funcify=True)
class CreateConfig(BaseLoader):
    """
    Functor that creates a TensorFlow config.
    """

    def __init__(self, gpu_memory_fraction=None, allow_growth=None, use_xla=None):
        """
        Creates a TensorFlow config.

        Args:
            gpu_memory_fraction (float):
                The fraction of GPU memory that will be made available to TensorFlow.
                This should be a value between 0.0 and 1.0.
            allow_growth (bool): Whether to allow GPU memory allocated by TensorFlow to grow.
            use_xla (bool): Whether to attempt to enable XLA.
        """
        self.gpu_memory_fraction = util.default(gpu_memory_fraction, 0.9)
        self.allow_growth = util.default(allow_growth, False)
        self.use_xla = util.default(use_xla, False)

    def call_impl(self):
        """
        Returns:
            tf.ConfigProto: The TensorFlow config.
        """

        # Session configuration
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction, allow_growth=self.allow_growth
        )
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        if self.use_xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        G_LOGGER.verbose(f"Using gpu memory fraction: {self.gpu_memory_fraction}, XLA: {self.use_xla}")
        return config


@mod.export(funcify=True)
class SessionFromGraph(BaseLoader):
    """
    Functor that creates a TensorFlow session that can be used for inference.
    """

    def __init__(self, graph, config=None):
        """
        Creates a TensorFlow session.

        Args:
            graph (Union[Tuple[tf.Graph, Sequence[str]], Callable() -> Tuple[tf.Graph, Sequence[str]]]):
                    A tuple containing a TensorFlow graph and output names or a callable that returns one.


            config (Union[tf.ConfigProto, Callable() -> tf.ConfigProto]):
                    A TensorFlow ConfigProto or a callable that returns one.
        """
        self.graph = graph
        self.config = util.default(config, CreateConfig())

    def call_impl(self):
        """
        Returns:
            tf.Session: The TensorFlow session.
        """
        config, _ = util.invoke_if_callable(self.config)
        (graph, output_names), _ = util.invoke_if_callable(self.graph)

        with graph.as_default() as graph, tf.compat.v1.Session(graph=graph, config=config).as_default() as sess:
            G_LOGGER.verbose(f"Using TensorFlow outputs: {output_names}")
            G_LOGGER.extra_verbose("Initializing variables in TensorFlow Graph")
            sess.run(tf.compat.v1.initializers.global_variables())
            return sess, output_names
