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


import tensorflow as tf
from collections import deque
from typing import List
import os
import shutil
from tf2onnx import tf_loader, utils, convert
import copy


def ensure_and_clean_dir(dir_path, do_clean_dir=True) -> None:
    """Create a directory to save test logs

    Args:
        dir_path (str): directory to create / clean.
        do_clean_dir (bool): boolean indicating whether to clean the directory if it already exists (remove+create).
    Returns:
        None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif do_clean_dir:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


class Folder:
    """
    Folder class that tracks all files for a single experiment.
    """

    def __init__(self, folder_name) -> None:
        self.base = folder_name
        ensure_and_clean_dir(self.base)
        self.fp32 = os.path.join(self.base, "fp32")
        ensure_and_clean_dir(self.fp32)
        self.fp32_saved_model = os.path.join(
            self.fp32, "saved_model"
        )  # location of fp32 saved keras model
        self.fp32_onnx_model = os.path.join(
            self.fp32, "original.onnx"
        )  # location of fp32 onnx model

        self.int8 = os.path.join(self.base, "int8")
        ensure_and_clean_dir(self.int8)
        self.int8_saved_model = os.path.join(
            self.int8, "saved_model"
        )  # location of int8 saved keras model
        self.int8_onnx_model = os.path.join(
            self.int8, "quantized.onnx"
        )  # location of int8 onnx model


class CreateAssetsFolders:
    """Create empty folders to save the original and quantized TensorFlow models and their respective ONNX
    models for each experiment.

    The following directory structure is created: base_directory -> experiment_directory (created by `add_folder` method) -> (fp32 [saved_model, .onnx model]),
    (int8 [saved_model, .onnx model]).
    """

    def __init__(self, base_experiment_directory) -> None:
        self.base = base_experiment_directory
        if not os.path.exists(self.base):
            os.mkdir(self.base)

    def add_folder(self, folder_name: str) -> None:
        """
        Create the experiment directory (sub-folder in the base directory passed to this class).

        Args:
            folder_name (str): name of folder

        Returns:
            None
        """
        setattr(self, folder_name, Folder(os.path.join(self.base, folder_name)))


def convert_saved_model_to_onnx(
    saved_model_dir: str, onnx_model_path: str, opset=13
) -> None:
    """Convert Keras saved model into ONNX format.
    Works directly with CreateAssetsFolder object path.

    Args:
        saved_model_dir (str): Path to keras saved model.
        onnx_model_path (str): Full path to ONNX model file.

    Returns:
        None
    """
    # 1. Let TensorRT optimize QDQ nodes instead of TF
    from tf2onnx.optimizer import _optimizers

    updated_optimizers = copy.deepcopy(_optimizers)
    del updated_optimizers["q_dq_optimizer"]
    del updated_optimizers["const_dequantize_optimizer"]

    # 2. Extract graph definition from SavedModel
    graph_def, inputs, outputs = tf_loader.from_saved_model(
        model_path=saved_model_dir,
        input_names=None,
        output_names=None,
        tag="serve",
        signatures=["serving_default"],
    )

    # 3. Convert tf2onnx and save onnx file
    model_proto, _ = convert._convert_common(
        graph_def,
        opset=opset,
        input_names=inputs,
        output_names=outputs,
        output_path=onnx_model_path,
        optimizers=updated_optimizers,
    )

    utils.save_protobuf(onnx_model_path, model_proto)
    print("ONNX conversion Done!")


def convert_keras_model_to_onnx(
    keras_model: tf.keras.Model, onnx_model_path: str, opset=13
) -> None:
    """Convert in-memory Keras model into ONNX format.
    Works directly with CreateAssetsFolder object path.

    Args:
        keras_model (tf.keras.Model): Keras model.
        onnx_model_path (str): Full path to ONNX model file.

    Returns:
        None
    """
    # 1. Let TensorRT optimize QDQ nodes instead of TF
    from tf2onnx.optimizer import _optimizers

    updated_optimizers = copy.deepcopy(_optimizers)
    del updated_optimizers["q_dq_optimizer"]
    del updated_optimizers["const_dequantize_optimizer"]

    # 2. Convert keras model directly and save onnx file.
    onnx_model_proto, _ = convert.from_keras(keras_model, opset=opset, optimizers=updated_optimizers)
    utils.save_protobuf(onnx_model_path, onnx_model_proto)


class KerasModelTraveller:
    """
    Utility class to travel Keras model and print out detailed layer information.
    """

    def __init__(self, print_layer_config=False) -> None:
        self._pc = print_layer_config
        self.model_list = deque([])
        # Used to filter which classes you want printed, by layer.__class__
        self._filter_by_class = None
        self._layer_names = []
        self._print_basic_info = None

    def _print_layer_info(self, layer):
        assert isinstance(layer, tf.keras.layers.Layer)
        if self._filter_by_class is None or layer.__class__ in self._filter_by_class:
            self._layer_names.append(layer.name)
            if self._print_basic_info:
                print(
                    "layer name:{layer_name}, layer class:{layer_class}".format(
                        layer_name=layer.name, layer_class=layer.__class__
                    )
                )
            if self._pc:
                print(layer.get_config())
            if self._print_basic_info:
                print("-----------------")

    def _dissect(self):
        if not self.model_list:
            return
        number_of_models = len(self.model_list)
        for _ in range(number_of_models):
            # Get a subclassed model
            current_model = self.model_list.pop()
            print("Keras Subclassed Model: {}".format(current_model.__class__.__name__))
            assert isinstance(current_model, tf.keras.Model)
            for l in current_model.layers:
                if isinstance(l, tf.keras.Model):
                    # This is another subclassed model inside
                    # Add this model to model queue for further analysis
                    self.model_list.appendleft(l)
                    self._dissect()
                else:
                    # This is a layer
                    self._print_layer_info(l)

    def _travel(
        self, keras_model: tf.keras.Model, filter_by_class=None, print_basic_info=False
    ):
        """Gets layer info by dissecting the model (need for multi-layered models)

        Args:
            keras_model (tf.keras.Model): Keras model
            filter_by_class (str): None or array of layer.__class__ to print

        Returns:
            None
        """
        self.filter_by_class = filter_by_class
        self._print_basic_info = print_basic_info
        assert isinstance(
            keras_model, tf.keras.Model
        ), "Model passed is not Keras model"
        self.model_list.appendleft(keras_model)
        self._dissect()
        self.filter_by_class = None

    def get_layer_names(self, keras_model: tf.keras.Model, filter_by_class=None):
        """Get name of all layers in the model.

        Args:
            keras_model (tf.keras.Model): Keras model
            filter_by_class (str): None or array of layer.__class__ to print

        Returns:
            None
        """
        self._travel(keras_model=keras_model, filter_by_class=filter_by_class)
        return self._layer_names

    def get_layer_information(self, keras_model: tf.keras.Model, filter_by_class=None):
        """Print information about all layers.

        Args:
            keras_model (tf.keras.Model): Keras model
            filter_by_class (str): None or array of layer.__class__ to print

        Returns:
            None
        """
        self._travel(
            keras_model=keras_model,
            filter_by_class=filter_by_class,
            print_basic_info=True,
        )


def _get_layer_info(layer: tf.keras.layers.Layer) -> dict:
    """
    Returns the layer's class, module, and name
    """
    return {
        "class": layer.__class__.__name__,
        "module": layer.__class__.__module__,
        "name": layer.name,
        "layer": layer,
    }


def _get_previous_layers_class_and_module_and_name(
    layer: tf.keras.layers.Layer,
) -> List[dict]:
    """
    For a given layer return a dictionary with name, module and class information of all previous layers.
    """
    r = []
    if isinstance(layer.input, list):
        for layer_input_tensor in layer.input:
            ip_tensor_parent_layer = layer_input_tensor._keras_history.layer
            r.append(_get_layer_info(ip_tensor_parent_layer))
    else:
        ip_tensor_parent_layer = layer.input._keras_history.layer
        r.append(_get_layer_info(ip_tensor_parent_layer))
    return r


def find_my_predecessors(model: tf.keras.Model, current_layer_name: str) -> List[dict]:
    """
    Given a layer name, find all predecessors of that layer.

    Args:
        model (tf.keras.Model): Keras functional model
        current_layer_name (str): name of a model layer for which predecessors has to be found.

    Returns:
        List[dict]: List of predecessors. Each dictionary has three keys as follows,
        ::
            {'class':<pred_layer_class>, 'module':<pred_layer_module>, 'name':<pred_layer_name>}

    Raises:
        AssertionError: If model is subclassed or current_layer_name is not string.
    """
    supported_model_classes = {"Functional", "Sequential"}
    assert isinstance(current_layer_name, str), "current layer name should be passed."
    assert (
        model.__class__.__name__ in supported_model_classes
    ), "model should be Functional or Sequential."

    for layer in model.layers:
        if layer.name == current_layer_name:
            return _get_previous_layers_class_and_module_and_name(layer)


def find_my_successors(model: tf.keras.Model, current_layer_name: str) -> List[dict]:
    """
    Given a layer name, find all successors of that layer.

    Args:
        model (tf.keras.Model): Keras functional model
        current_layer_name (str): name of a model layer for which successors has to be found.

    Returns:
        List[dict]: List of predecessors. Each dictionary has three keys as follows,
        ::
            {'class':<pred_layer_class>, 'module':<pred_layer_module>, 'name':<pred_layer_name>}

    Raises:
        AssertionError: If model is subclassed or current_layer_name is not string.
    """
    supported_model_classes = {"Functional", "Sequential"}
    assert isinstance(current_layer_name, str), "current layer name should be passed."
    assert (
        model.__class__.__name__ in supported_model_classes
    ), "model should be Functional or Sequential."

    def _check_all_next_layers_with_connection_to_current(
        next_layers: List[tf.keras.layers.Layer],
        current_layer_name: str,
        current_layer_class: str,
    ):
        successors = []
        for layer in next_layers:
            p_layers = _get_previous_layers_class_and_module_and_name(layer)
            for p_layer in p_layers:
                if (
                    p_layer["class"] == current_layer_class
                    and p_layer["name"] == current_layer_name
                ):
                    successors.append(_get_layer_info(layer))
        return successors

    all_layers = model.layers
    for i, layer in enumerate(all_layers):
        if layer.name == current_layer_name:
            next_layers = all_layers[i + 1 :]
            layer_info = _get_layer_info(layer)
            return _check_all_next_layers_with_connection_to_current(
                next_layers, layer_info["name"], layer_info["class"]
            )
