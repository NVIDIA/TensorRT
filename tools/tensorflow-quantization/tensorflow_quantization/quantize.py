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
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#    
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union
import tensorflow as tf
import tensorflow_quantization.quantize_wrappers as quantize_wrappers
import tensorflow_quantization.global_config as cfg
import tensorflow_quantization.quantize_config as quantize_config
from dataclasses import dataclass
from tensorflow_quantization.quantize_wrappers import DISABLED_LAYER_QUANTIZATION_DEFAULT


@dataclass
class LayerConfig:
    """
    Internal dataclass for a single layer config.
    Args:
        name (str): Name of the layer. As seen from utilities such as `model.summary()`
        is_keras_class (bool) : Set this to True if layer_name passed represents a layer class from Keras.
            Default is False.
        quantize_input (bool): Set this to True if input to the layers should be quantized. Default is True
            since default behavior is following Nvidia quantization recipe.
        quantize_weight (bool): Set this to True if weights to the layers should be quantized. Default is True
            since default behavior is following Nvidia quantization recipe. For weightless layers, value is
            ignored.
        quantization_index (List): Indices on inputs to which quantization is applied for the layers with
            multiple inputs. E.g Add, Concatenate
    Returns:
        None
    """

    name: str = None
    is_keras_class: bool = False
    quantize_input: bool = True
    quantize_weight: bool = True
    quantization_index: list = None


class QuantizationSpec:
    """
    Helper class holding config objects for all layers to quantize.
    """

    def __init__(self) -> None:
        self.layers = []

    def __str__(self) -> str:
        for l in self.layers:
            print(l)
        return ""

    def add(
        self,
        name: Union[str, List],
        is_keras_class: Union[bool, List] = False,
        quantize_input: Union[bool, List] = True,
        quantize_weight: Union[bool, List] = True,
        quantization_index: Union[List, List[List]] = None,
    ) -> None:
        """
        Takes user parameters and adds LayerConfig object to a list for each add call.

        Args:
            name (Union[str, List]): Name of the layer. As seen from utilities such as `model.summary()`
            is_keras_class (Union[bool, List]): List or a single value. Set this to True if layer_name passed represents a layer class from Keras.
                Default is False.
            quantize_input (Union[bool, List]): List or a single value. Set this to True if input to the layers should be quantized. Default is True
                since default behavior is following Nvidia quantization recipe.
            quantize_weight (Union[bool, List]): List or a single value. Set this to True if weights to the layers should be quantized. Default is True
                since default behavior is following Nvidia quantization recipe. For weightless layers, value is
                ignored.
            quantization_index (Union[List, List[List]]): List or List of List. List with indices on inputs to which quantization is applied for the layers with
                multiple inputs. E.g Add, Concatenate
        Returns:
            None
        """
        if not isinstance(name, list):
            self.layers.append(
                LayerConfig(
                    name=name,
                    is_keras_class=is_keras_class,
                    quantize_input=quantize_input,
                    quantize_weight=quantize_weight,
                    quantization_index=quantization_index,
                )
            )
        else:
            # layer names is passed as a list
            if isinstance(is_keras_class, list):
                assert len(name) == len(
                    is_keras_class
                ), "[E] `is_keras_class` is a list but length is not same as layer `name` list"
            if isinstance(quantize_input, list):
                assert len(name) == len(
                    quantize_input
                ), "[E] `quantize_input` is a list but length is not same as layer `name` list"
            if isinstance(quantize_weight, list):
                assert len(name) == len(
                    quantize_weight
                ), "[E] `quantize_weight` is a list but length is not same as layer `name` list"
            if isinstance(quantization_index, list):
                assert len(name) == len(
                    quantization_index
                ), "[E] `quantization_index` is list but length is not same as layer `name` list"
            for i, e in enumerate(name):
                cl_name = e
                cl_is_keras_class = (
                    is_keras_class[i]
                    if isinstance(is_keras_class, list)
                    else is_keras_class
                )
                cl_quantize_input = (
                    quantize_input[i]
                    if isinstance(quantize_input, list)
                    else quantize_input
                )
                cl_quantize_weight = (
                    quantize_weight[i]
                    if isinstance(quantize_weight, list)
                    else quantize_weight
                )
                cl_quantization_index = (
                    quantization_index[i]
                    if isinstance(quantization_index, list)
                    else quantization_index
                )
                self.layers.append(
                    LayerConfig(
                        name=cl_name,
                        is_keras_class=cl_is_keras_class,
                        quantize_input=cl_quantize_input,
                        quantize_weight=cl_quantize_weight,
                        quantization_index=cl_quantization_index,
                    )
                )


def _skip_layer(layer: tf.keras.layers.Layer) -> bool:
    """
    Decide whether quantization wrapping should be skipped for the given layer.
    The decision is made based on an internal quantize config object parameters.

    Args:
        layer (tf.keras.layers.Layer): Keras model layer
    Returns:
        bool: True if given layer should not be quantized else False
    """
    config_object = cfg.get_config_object()

    # Check if any layer with Disabled Quantization by default are in the 'config_object.layer_classes_to_quantize'.
    #   If so, that layer will be enabled for quantization. Otherwise, skip (return True).
    layer_class_name = layer.__class__.__name__
    if layer_class_name in DISABLED_LAYER_QUANTIZATION_DEFAULT:
        if layer_class_name not in config_object.layer_classes_to_quantize:
            if layer.name in config_object.get_layer_config():
                # User can enable a single layer even if the default behavior of a Class is to not quantize.
                # The decision of whether to quantize this layer or not will be left for later checks, such as when
                #   quantize_input and quantize_weight = False.
                pass
            else:
                # Default behavior: skip layer
                return True

    # 1. When quantize_input = False, quantize_weight = False and quantization_index=None, don't even wrap the layer.
    if layer.name in config_object.get_layer_config():
        current_layer_config = config_object.get_layer_config()[layer.name]
        if (
            current_layer_config["qbool_list"][0] == False  # quantize_input
            and current_layer_config["qbool_list"][1] == False  # quantize_weight
            and "qindex_list" not in current_layer_config
        ):
            print(
                "[I] Layer `{layer_name}` is not quantized. There is nothing to quantize since "
                "quantize_input = False, quantize_weight = False and quantization_index=None".format(
                    layer_name=layer.name
                )
            )

            return True

    # 2. Called when quantization_mode is `partial`
    if config_object.config_class_id == 2:
        # A. Skip current `layer class` if current layer class is not in user provided QuantizationSpec class
        #      object. However, when current layer name is passed by user to quantize, don't skip the layer.
        if (
                len(config_object.layer_classes_to_quantize) != 0
                and layer.__class__.__name__ not in config_object.layer_classes_to_quantize
        ):
            if layer.name in config_object.get_layer_config():
                return False
            else:
                print(
                    "[I] Layer class `{layer_class_name}` is not quantized. Partial quantization is enabled "
                    "and layer class is not in user provided QuantizationSpec class object".format(
                        layer_class_name=layer.__class__.__name__
                    )
                )
                return True
        # B. Skip current layer if `layer.name` is not in user provided QuantizationSpec class object.
        #      However, if current layer class is passed by user to quantize, don't skip the layer.
        elif layer.name not in config_object.get_layer_config():
            if layer.__class__.__name__ in config_object.layer_classes_to_quantize:
                return False
            else:
                print(
                    "[I] Layer `{layer_name}` is not quantized. Partial quantization is enabled and layer name is not "
                    "in user provided QuantizationSpec class object".format(
                        layer_name=layer.name
                    )
                )
                return True

    return False


def _quantize_model_layer_clone_function(
    layer: tf.keras.layers.Layer,
) -> "BaseQuantizeWrapper":
    """
    Wrap or leave given layer based on quantize config object parameters.
    Args:
        layer (tf.keras.layers.Layer): Keras model layer
    Returns:
        BaseQuantizeWrapper: layer wrapped in BaseQuantizeWrapper class.
    """
    layer_wrapper = layer
    if _skip_layer(layer):
        # Skip the layers not specified by the user.
        pass
    else:
        child_wrappers_dict = quantize_wrappers.BaseQuantizeWrapper.CHILD_WRAPPERS
        possible_wrapper_name_for_this_layer = (
            layer.__class__.__name__ + "QuantizeWrapper"
        )
        if possible_wrapper_name_for_this_layer in child_wrappers_dict:
            wrapper_function = child_wrappers_dict[possible_wrapper_name_for_this_layer]
            layer_wrapper = wrapper_function(layer)
    return layer_wrapper


def _execute_quantize_model(
    model: tf.keras.Model, class_id: int, qspec: QuantizationSpec = None
) -> tf.keras.Model:
    """
    clone the model and apply quantization to specific layers based on quantize config object parameters.
    Args:
        model (tf.keras.Model): Keras functional or sequential model.
            * Currently Subclassed models are not supported
        class_id (int): internal quantization class ID
        qspec (QuantizationSpec): object of QuantizationSpec class. If few layers or layer classes are to be treated
            differently, LayerConfig class objects for that layer/layer class are created internally and
            added to QuantizationSpec class.
    Returns:
        tf.keras.Model: Quantized model with QDQ nodes added.
    """
    config_id_class_name_map = {
        0: "FullNetworkQuantization",
        1: "FullNetworkSpecialQuantization",
        2: "PartialNetworkQuantization",
    }

    # 1. Create quantize config object
    q_config_object = getattr(quantize_config, config_id_class_name_map[class_id])()

    # 2. Update object attributes
    if qspec:
        q_config_object.add_quantization_spec_object(qspec, model.layers)

    assert (
        cfg.is_config_object_created()
    ), "[E] Have you created the quantization config object before calling `quantize_model`?"

    # Wrap quantizable layers
    model = tf.keras.models.clone_model(
        model, input_tensors=None, clone_function=_quantize_model_layer_clone_function
    )

    # Clean global space afterwards
    q_config_object.clean()

    return model


def _recognize_config_class_id(
    quantization_mode: str = "full", qspec: QuantizationSpec = None
) -> int:
    """
    Interpret internal quantize config class based on parameters passed by user to
    `quantize_model` function.
    Args:
        quantization_mode (str): Either 'full' or 'partial' quantization mode
        qspec (QuantizationSpec): object of QuantizationSpec class. If few layers or layer classes are to be treated
            differently, LayerConfig class objects for that layer/layer class are created internally and
            added to QuantizationSpec class.
    Returns:
        int: ID for quantization category class used internally.
    Raises:
        Exception: if no class can be interpreted for given parameter combination
    """
    if quantization_mode == "full" and qspec is None:
        return 0
    elif quantization_mode == "full" and qspec is not None:
        return 1
    elif quantization_mode == "partial" and qspec is not None:
        return 2
    else:
        raise Exception(
            "Could not recognize config class ID."
            " Are parameters passed to `quantize_model` function correct?"
        )


def _validate_config(
    quantization_mode: str = "full", qspec: QuantizationSpec = None
) -> None:
    """
    Validate if parameters passed to `quantize_model` makes sense.
    Args:
        quantization_mode (str): quantization mode can be either 'full' or 'partial'
        qspec (QuantizationSpec): object of QuantizationSpec class. If few layers or layer classes are to be treated
            differently, LayerConfig class objects for that layer/layer class are created internally and
            added to QuantizationSpec class.
    Returns:
        None
    Raises:
        AssertionError: when configuration is not valid.
    """

    def _verify_support_for_all_layer_classes(qspec: QuantizationSpec):
        for layer in qspec.layers:
            if layer.is_keras_class:
                # Layer class name is provided.
                child_wrappers_dict = (
                    quantize_wrappers.BaseQuantizeWrapper.CHILD_WRAPPERS
                )
                possible_wrapper_name_for_this_layer = layer.name + "QuantizeWrapper"
                assert possible_wrapper_name_for_this_layer in child_wrappers_dict, (
                    "[E] layer class `{layer_name}` is not supported yet! Either there is no native wrapper or user "
                    "provided wrapper registration failed.".format(
                        layer_name=layer.name
                    )
                )

    if qspec:
        _verify_support_for_all_layer_classes(qspec)

    if quantization_mode == "partial":
        assert (
            qspec is not None
        ), "[E] `QuantizationSpec` class object must be passed when `quantization_mode=partial`."


def quantize_model(
    model,
    quantization_mode: str = "full",
    quantization_spec: QuantizationSpec = None,
    custom_qdq_cases: List["CustomQDQInsertionCase"] = None,
) -> tf.keras.Model:
    """
    Insert Q/DQ nodes in Keras model and return a copy. Weights are preserved unlike native keras clone.

    Args:
        model(tf.keras.Model): Keras Functional or Sequential model.subclassed models are not yet supported.
        quantization_mode(str): quantization mode can be either 'full' or 'partial'
        quantization_spec(QuantizationSpec) : object of QuantizationSpec class. If few layers or layer classes are to
            be treated differently, LayerConfig class objects for that layer/layer class are created internally and
            added to QuantizationSpec class.
        custom_qdq_cases(List[CustomQDQInsertionCase]) : `Case` method on every object in this list is called by passing
            model and user passed quantization_spec as arguments. Each member of this list is an object of a class
            inherited from CustomQDQInsertionCase class.

    Raises:
        AssertionError: When passed model is subclassed.
        AssertionError: When CustomQDQInsertionCase does not return QuantizationSpec object.
        AssertionError: When quantization mode is `partial` but QuantizationSpec object is not passed.
        AssertionError: When quantization wrapper is not found for desired layer class.
        ExceptionError: When internal quantization class ID can't be detected. This happens when passed parameters
            do not make sense.
    Returns:
        tf.keras.Model: Quantized model with QDQ nodes inserted according to NVIDIA quantization recipe.
    """
    supported_model_classes = {"Functional", "Sequential"}
    assert (
        model.__class__.__name__ in supported_model_classes
    ), "[E] Currently only `Functional` or `Sequential` model quantization is supported."

    # Update quantization_spec object based on output of special QDQ cases.
    custom_quantization_spec = QuantizationSpec()
    if custom_qdq_cases:
        for custom_qdq_case in custom_qdq_cases:
            qspec_case_object = custom_qdq_case.case(model, quantization_spec)
            if qspec_case_object:
                assert isinstance(
                    qspec_case_object, QuantizationSpec
                ), "[E] {} \
                does not return an object of QuantizationSpec.".format(
                    qspec_case_object.__class__.__name__
                )
                custom_quantization_spec.layers.extend(qspec_case_object.layers)

    # if user has passed quantization_spec then extend it with custom_quantization_spec
    # else use just custom_quantization_spec
    if quantization_spec:
        quantization_spec.layers.extend(custom_quantization_spec.layers)
    else:
        if len(custom_quantization_spec.layers) != 0:
            quantization_spec = custom_quantization_spec

    # Check if config is valid and quantize model
    _validate_config(quantization_mode, quantization_spec)
    cid = _recognize_config_class_id(quantization_mode, quantization_spec)
    return _execute_quantize_model(model, cid, quantization_spec)
