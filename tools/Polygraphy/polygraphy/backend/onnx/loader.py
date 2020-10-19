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
from polygraphy.backend.base import BaseLoadModel
from polygraphy.backend.onnx import util as onnx_util
from polygraphy.common import constants
from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import misc


# ONNX loaders return ONNX models in memory.
class OnnxFromPath(BaseLoadModel):
    def __init__(self, path):
        """
        Functor that loads an ONNX model from a file.

        Args:
            path (str): The path from which to load the model.
        """
        self.path = path


    def __call__(self):
        """
        Loads an ONNX model from a file.

        Returns:
            onnx.ModelProto: The ONNX model
        """
        import onnx
        misc.log_module_info(onnx)

        G_LOGGER.verbose("Loading ONNX model: {:}".format(self.path))
        return onnx_util.check_model(onnx.load(self.path))


class OnnxFromTfGraph(BaseLoadModel):
    def __init__(self, graph, opset=None, optimize=None, fold_constant=None):
        """
        Functor that loads a TensorFlow graph and converts it to ONNX using the tf2onnx converter.

        Args:
            graph (Callable() -> Tuple[tf.Graph, Sequence[str]]):
                    A callable that can supply a tuple containing a TensorFlow
                    graph and output names.


            opset (int): The ONNX opset to use during conversion.
            optimize (bool): Whether to use tf2onnx's graph optimization pass.
            fold_constant (bool): Whether to fold constants in the TensorFlow Graph. Requires that ``optimize`` is also enabled. Defaults to True.
        """
        self._graph = graph
        self.opset = misc.default_value(opset, 11)
        self.fold_constant = misc.default_value(fold_constant, True)
        self.optimize = misc.default_value(optimize, True)

        if self.fold_constant and not self.optimize:
            G_LOGGER.warning("`fold_constant` is enabled, but `optimize` is disabled. Constant folding will not be performed")


    def __call__(self):
        """
        Converts a TensorFlow model into ONNX.

        Returns:
            onnx.ModelProto: The ONNX model.
        """
        import tensorflow as tf
        import tf2onnx
        from polygraphy.backend.tf import util as tf_util

        misc.log_module_info(tf2onnx)

        (graph, output_names), _ = misc.try_call(self._graph)
        input_names = list(tf_util.get_input_metadata(graph).keys())

        if self.fold_constant:
            G_LOGGER.info("Folding constants in graph using tf2onnx.tfonnx.tf_optimize")
        graphdef = graph.as_graph_def()
        if self.optimize:
            graphdef = tf2onnx.tfonnx.tf_optimize(input_names, output_names, graph.as_graph_def(), fold_constant=self.fold_constant)

        with tf.Graph().as_default() as graph, tf.compat.v1.Session(graph=graph) as sess:
            tf.import_graph_def(graphdef, name="")

            onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph, input_names=input_names, output_names=output_names, opset=self.opset)
            if self.optimize:
                onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)
            return onnx_util.check_model(onnx_graph.make_model("model"))


class ModifyOnnx(BaseLoadModel):
    def __init__(self, model, do_shape_inference=None, outputs=None, exclude_outputs=None):
        """
        Functor that modifies an ONNX model.

        Args:
            model (Callable() -> onnx.ModelProto): A loader that can supply an ONNX model.

            outputs (Sequence[str]):
                Names of tensors to mark as outputs. If provided, this will override the
                existing model outputs.
                If a value of `constants.MARK_ALL` is used instead of a list, all tensors in the network are marked.
            exclude_outputs (Sequence[str]):
                Names of tensors to exclude as outputs. This can be useful in conjunction with
                ``outputs=constants.MARK_ALL`` to omit outputs.
        """
        self._model = model
        self.do_shape_inference = misc.default_value(do_shape_inference, False)
        self.outputs = outputs
        self.exclude_outputs = exclude_outputs


    def __call__(self):
        """
        Modifies an ONNX model.

        Returns:
            onnx.ModelProto: The modified ONNX model.
        """
        model, _ = misc.try_call(self._model)

        if self.do_shape_inference:
            model = onnx_util.infer_shapes(model)

        if self.outputs == constants.MARK_ALL:
            G_LOGGER.verbose("Marking all ONNX tensors as outputs")
            model = onnx_util.mark_layerwise(model)
        elif self.outputs is not None:
            model = onnx_util.mark_outputs(model, self.outputs)

        if self.exclude_outputs is not None:
            model = onnx_util.unmark_outputs(model, self.exclude_outputs)

        return onnx_util.check_model(model)


class SaveOnnx(BaseLoadModel):
    def __init__(self, model, path=None):
        """
        Functor that saves an ONNX model to the specified path.

        Args:
            model (Callable() -> onnx.ModelProto): A loader that can supply an ONNX model.
            path (str): Path at which to write the ONNX model.
         """
        self._model = model
        self.path = path


    def __call__(self):
        """
        Saves an ONNX model to the specified path.

        Returns:
            onnx.ModelProto: The model, after saving it.
        """
        model, _ = misc.try_call(self._model)
        misc.lazy_write(contents=lambda: model.SerializeToString(), path=self.path)
        return model


class BytesFromOnnx(BaseLoadModel):
    def __init__(self, model):
        """
        Functor that serializes an ONNX model.

        Args:
            model (Callable() -> onnx.ModelProto): A loader that can supply an ONNX model.
         """
        self._model = model


    def __call__(self):
        """
        Serializes an ONNX model.

        Returns:
            bytes: The serialized model.
        """
        model, _ = misc.try_call(self._model)
        return model.SerializeToString()
