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


"""
This module implements classes to configure three supported quantization modes:
1. Full: quantize all layers with standard protocol based NVIDIA quantization scheme.
2. Full special: quantize few layers in a specific way and remaining with standard protocol based on NVIDIA
                 quantization scheme.
3. Partial: quantize ONLY few layers.

Each quantization mode can quantize all supported Keras layer classes or only subset of it.
"""

from abc import ABC
import tensorflow_quantization.global_config as global_config
import warnings
from typing import List, Dict


class BaseConfig(ABC):
    """
    Base class from which four quantize config classes are derived.
    Default quantization recipe is Nvidia's recommendation.
    """

    def __new__(cls):
        instance = super().__new__(cls)
        # Add instance to global list
        global_config.add_config_object(instance)
        return instance

    def __init__(self) -> None:
        self.quantization_mode: str = "full"
        self.layerwise_config: dict = {}  # holds special layers information.
        self.layer_classes_to_quantize: set = set()
        self.config_class_id: int = 0

    def __str__(self) -> str:
        return (
            " quantization_mode: {quant_mode} \n "
            "layerwise_config: {layerwise_config} \n "
            "specific_layer_class: {specific_layer_class} \n "
            "config_class_id: {config_class_id} \n".format(
                quant_mode=self.quantization_mode,
                layerwise_config=self.layerwise_config,
                specific_layer_class=self.specific_layer_class,
                config_class_id=self.config_class_id,
            )
        )

    @staticmethod
    def _validate_layer_names(
        user_passed_layer_names: List, model_layers: List
    ) -> None:
        """
        Check whether user passed layer names exists in  Keras model being quantized.
        Args:
            user_passed_layer_names (List): Layer names passed by user to treat specially.
            model_layers (List): Keras model layers passed as a list.
        Returns:
            None
        Raise:
            Warning : when specific layer name is not found. Such layers are simply ignored.
        """
        model_layer_name_set = set()
        for l in model_layers:
            model_layer_name_set.add(l.name)

        for ul in user_passed_layer_names:
            if ul not in model_layer_name_set:
                warnings.warn(
                    "layer name {} is passed by user but could not find layer with this name in model.".format(
                        ul
                    )
                )

    def add_quantization_spec_object(
        self, qspec: "QuantizationSpec", original_model_layers: List
    ) -> None:
        """
        This method parses object of QuantizationSpec class and fill in `layerwise_config` dictionary
        holding information about layers that need to be treated specially.
        Specific layer classes that need to be treated specially are also here.
        Args:
            qspec (QuantizationSpec): object of QuantizationSpec class. If few layers or layer classes are to be treated
                differently, LayerConfig class objects for that layer/layer class are created internally and
                added to QuantizationSpec class.
            original_model_layers (List): Keras model layers passed as a list.
        Returns:
            None
        """
        for layer in qspec.layers:
            if layer.is_keras_class:
                self.add_special_layer_class(layer.name)
            else:
                layer_config_dict = {"qbool_list": [False, False]}
                layer_config_dict["qbool_list"][0] = layer.quantize_input
                layer_config_dict["qbool_list"][1] = layer.quantize_weight
                if layer.quantization_index:
                    layer_config_dict["qindex_list"] = layer.quantization_index
                self.add_special_layer(
                    layer_name=layer.name, config_dict=layer_config_dict
                )
        # Validate whether added layers exist in the model
        self._validate_layer_names(
            list(self.layerwise_config.keys()), original_model_layers
        )

    def add_special_layer(self, layer_name: str, config_dict: Dict) -> None:
        """
        Add layer specific quantization information to quantize config object.
        Args:
            layer_name (str): layer name
            config_dict (Dict): Layer specific quantization parameter dictionary in the
                following format.
                There are only two accepted keys `qbool_list` and `qindex_list`.
                `qbool_list` is list of length two where each value is
                [<True/False quantize inputs>, <True/False quantize weights>]
                e.g.
                To quantize inputs and weights, `qbool_list`=[True, True]

                `qindex_list` is a list of specific indices to quaintize for layers such as Add, Concatenate
                where more than two inputs are present.

                Based on above information,
                1. config_dict for weighted layer with name `dense_2`, to quantize inputs and weights will be
                {'qbool_list':[True, True]} with laye_name=`dense_2`
                2. config_dict for non weighted layer with name `add_3` to quantize input at index 1 will be
                {'qbool_list':[True, False], 'qindex_list':[1]} with layer_name=`add_3`
        Returns:
            None
        Raises:
            Exception: When invalid keys are detected.
        """
        self.layerwise_config[layer_name] = config_dict

    def remove_layer(self, layer_name: str) -> None:
        """
        Remove specific layer based on name from quantize config object.
        Args:
            layer_name (str): layer name
        Returns:
            None
        """
        if layer_name in self.layerwise_config:
            del self.layerwise_config[layer_name]

    def remove_layers(self, layers_name: List) -> None:
        """
        Bulk remove specific layers based on names from quantize config object.
        Args:
            layers_name (List): layers names, list of strings
        Returns:
            None
        """
        for layer_name in layers_name:
            self.remove_layer(layer_name=layer_name)

    def get_layer_config(self) -> Dict:
        """
        Return dictionary with information about layers to quantize for quantize
        config object.
        Args:
            None
        Returns:
            Dict: a dictionary with layerwise configuration parameters.
        """
        return self.layerwise_config

    def is_empty(self) -> bool:
        """
        Return True if no layer specific quantization information is available in quantize
        config object.
        Args:
            None
        Returns:
            bool: True if no special layers are passed else return False
        """
        return not self.layerwise_config

    def clear_layer_config(self) -> None:
        """
        Clear layer config information from quanize config object
        Args:
            None
        Returns:
            None
        """
        self.layerwise_config.clear()

    def add_special_layer_class(self, layer_class_name: str) -> None:
        """
        Add class name to quantize config object so that only layers with specific class are quantized.
        Args:
            layer_class_name : String that represents keras class
        Returns:
            None
        """
        self.layer_classes_to_quantize.add(layer_class_name)

    def clean(self):
        """
        Clean quantize config object from global space. Calling this is important to use `quantize_model` multiple times
        within a single module.
        Args:
            None
        Returns:
            None
        """
        global_config.remove_config_object()


class FullNetworkQuantization(BaseConfig):
    """
    Quantize all layers based on NV scheme.

    Nvidia recommended recipe for quantization is using Q/DQ only wth inputs/weights.
    Q/DQ output support is just to compare engine performance/accuracy when other quantization
    scheme is used.

    NV: Add Q/DQ at input and weights
    TF: Add Q/DQ at output and weights

    This is config class with index `0` which is default.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config_class_id = 0


class FullNetworkSpecialQuantization(BaseConfig):
    """
    Quantize few layers in specific way and remaining network in standard way based on NV scheme.
    Layers are selected based on 'names' which can be via 'model.summary()' for functional
    and sequential models.
    Subclassed model layer information can be found using `KerasModelTraveller` class from utils.

    This is config class with index 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config_class_id = 1


class PartialNetworkQuantization(BaseConfig):
    """
    Quantize only specific layers and not the entire network.
    Layers are selected based on name.

    This is config class with index 2.
    """

    def __init__(self) -> None:
        super().__init__()
        self.quantization_mode = "partial"
        self.config_class_id = 2
