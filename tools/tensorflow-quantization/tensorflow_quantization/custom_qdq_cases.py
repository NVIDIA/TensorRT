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

from tensorflow_quantization import CustomQDQInsertionCase
from tensorflow_quantization import QuantizationSpec
from tensorflow_quantization import utils
import tensorflow as tf
from typing import List


def is_parent_type(parent_class: str, class_type="Conv") -> bool:
    """
    Checks if 'parent_class' is of type 'type'.
    Examples of types: Conv, BatchNorm, Dropout, Activation.
    """
    return class_type in parent_class


def is_parent_pattern(parent_info: dict, pattern: List = ["BatchNorm", "Conv"]) -> bool:
    """ Checks if parent heritage follows a specific 'pattern'.
    Args:
        parent_info (dict): dictionary with parent's information.
        pattern (List): list containing a layer's parental heritage ([parent, grandparent, great-grandparent, ...]).
    Returns:
        bool: indicating whether a layer's parent heritage follows the given pattern.
    """
    grandparent_info = parent_info
    for i, p in enumerate(pattern):
        if i > 0:
            grandparent_info = utils._get_previous_layers_class_and_module_and_name(
                grandparent_info["layer"]
            )[0]
        if not is_parent_type(grandparent_info["class"], class_type=p):
            return False
    return True


def check_is_quantizable_by_layer_name(
    qspec: QuantizationSpec, current_layer_name: str
) -> bool:
    """
    Checks if 'current_layer_name' is in 'qspec'. It returns True if 'current_layer_name' is NOT in 'qspec' and
      False if it is. This means that the user's request will get prioritized over our automatic methods.

    Args:
         qspec (QuantizationSpec): quantization specification.

    Returns:
        is_quantizable_by_layer_name (bool): boolean indicating whether 'current_layer_name' is quantizable by our
          method (is NOT in 'qspec'), or not (is in 'qspec', so that configuration should be followed).
    """
    def _is_layer_in_user_passed_qspec(layer_name):
        for l in qspec.layers:
            if l.name == layer_name:
                return True
        return False

    is_quantizable_by_layer_name = qspec is None or (
        qspec is not None and not _is_layer_in_user_passed_qspec(current_layer_name)
    )
    return is_quantizable_by_layer_name


###################################################################
################# General Custom QDQ Cases ########################
###################################################################


class BNQDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        return "Avoids inserting QDQ before BatchNorm in cases where BN is connected to a Conv layer (since that BN " \
               "will be fused with previous Conv layer). This case happens in ResNet-v2, where the following pattern " \
               "exists: BN-ReLU-Conv blocks (pre-activation function). In that scenario, BN is sometimes connected to " \
               "`Add` layer, which doesn't fuse with BN."

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:
        def _check_if_quantizable_bn(layer):
            layer_parent = layer.input._keras_history.layer
            parent_class_name = layer_parent.__class__.__name__
            if not is_parent_type(parent_class_name, class_type="Conv"):
                if check_is_quantizable_by_layer_name(qspec, layer.name):
                    return True
            return False

        bn_qspec = QuantizationSpec()
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                """
                Returns quantizable BatchNorm layers: All BN layers that are not connected to a Conv layer.
                In other words, don't add QDQ to BN layers in a Conv-BN sequence (and of course, if it shouldn't be
                ignored due to the user's preference)."
                """
                if _check_if_quantizable_bn(layer):
                    bn_qspec.add(
                        name=layer.name, quantize_input=True, quantize_weight=False
                    )
        return bn_qspec


class ResidualConnectionQDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        info_str = "Goal: To return all quantizable residual inputs. " \
                   "Rules: Residual connection is represented by the Add layer. The recommendation from the TRT team " \
                   "        is to add QDQ to all of its inputs except when: " \
                   "     - the input is Bias. Note that TF sees MatMul+BiasAdd as a Dense layer, so no need to check " \
                   "       if the input is Bias. " \
                   "     - in the case of one of the inputs being a simple residual branch and the other Conv or " \
                   "       Conv+BN, add QDQ nodes to just the residual branch. This is needed to trigger an INT8 " \
                   "       kernel fusion with Add. " \
                   "     - in the case of more than one input being Conv or Conv+BN, add QDQ to all inputs except 1. " \
                   " The last 2 cases are needed to trigger an INT8 kernel fusion with Add. " \
                   " [ResNet-v1]:  Note that the connection between Conv2D and Add layer is not direct: " \
                   "                   Conv2D -> BatchNormalization -> Add " \
                   "               To get to the layer, we need to access `input._keras_history.layer` " \
                   "               This is the same for EfficientNet-B0. " \
                   " [ResNet-v2]: Connection is direct ReLU -> Conv2D -> Add " \
                   " [EfficientNet-B0]: Contains two special patterns: " \
                   "                      1. Conv -> BatchNorm -> Activation -> Add " \
                   "                      2. Conv -> BatchNorm -> Activation -> Dropout -> Add"
        return info_str

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:

        res_qspec = QuantizationSpec()
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Add) and check_is_quantizable_by_layer_name(qspec, layer.name):
                """
                Returns quantizable inputs to Add layers: all inputs except 1 with 'pattern'.
                Patterns checked for: Conv, Conv-BN, Conv-BN-Activation, Conv-BN-Activation-Dropout.
                """
                layer_parents = utils.find_my_predecessors(keras_model, layer.name)

                # Collect the non-quantizable input (1 branch with Conv pattern)
                input_indices_convs = []
                for i, l_parent_info in enumerate(layer_parents):
                    l_parent_class = l_parent_info["class"]
                    l_parent_layer = l_parent_info["layer"]
                    # Check that the input is a Conv pattern
                    if (
                            is_parent_type(l_parent_class, class_type="Conv")
                            or is_parent_pattern(l_parent_info, pattern=["BatchNorm", "Conv"])
                            or is_parent_pattern(l_parent_info, pattern=["Activation", "BatchNorm", "Conv"])
                            or is_parent_pattern(l_parent_info, pattern=["Dropout", "Activation", "BatchNorm", "Conv"])
                    ):
                        # Check that it's not a residual branch (input does not have more than 1 outbound node)
                        if hasattr(l_parent_layer, 'outbound_nodes'):
                            num_outbound_nodes = len(getattr(l_parent_layer, 'outbound_nodes'))
                            if num_outbound_nodes == 1:
                                # Branch without QDQ branch is chosen
                                input_indices_convs.append(i)
                                break

                # Default behavior: add QDQ in all inputs except 1 with Conv/BN
                input_indices = list(range(0, len(layer_parents)))
                if len(input_indices_convs) > 0:
                    # Don't quantize one of the Conv pattern branches.
                    index_to_delete = input_indices_convs[-1]
                    del input_indices[index_to_delete]
                if len(input_indices) > 0:
                    res_qspec.add(
                        layer.name,
                        quantize_input=True,
                        quantize_weight=False,
                        quantization_index=input_indices,
                    )
        return res_qspec


class MaxPoolQDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        return "Enables quantization of MaxPool layers. This is needed in cases where MaxPool is added to a residual " \
               "connection and where the other branches are already quantized (needed to trigger a horizontal fusion " \
               "in the residual connection. This case happens in ResNet-v2."

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:
        mp_qspec = QuantizationSpec()
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                """
                Returns quantizable MaxPooling2D layers.
                """
                if check_is_quantizable_by_layer_name(qspec, layer.name):
                    mp_qspec.add(
                        name=layer.name,
                        quantize_input=True,
                        quantize_weight=False
                    )
        return mp_qspec


###################################################################
############ Network Specific QDQ Cases ###########################
###################################################################


class ResNetV1QDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        return (
            "Returns all quantizable nodes in ResNet-v1: "
            "  1. Residual connections."
        )

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:
        special_qspec = QuantizationSpec()

        # Use Residual connection QDQ
        residual_cqdq = ResidualConnectionQDQCase()
        residual_cqdq_qspec = residual_cqdq.case(keras_model, qspec)
        special_qspec.layers.extend(residual_cqdq_qspec.layers)

        return special_qspec


class ResNetV2QDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        return (
            "Returns all quantizable nodes in ResNet-v2: "
            "  1. Residual connections, "
            "  2. BatchNorm not connected to Conv, "
            "  3. MaxPool layers."
        )

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:
        special_qspec = QuantizationSpec()

        # Use Residual connection QDQ
        residual_cqdq = ResidualConnectionQDQCase()
        residual_cqdq_qspec = residual_cqdq.case(keras_model, qspec)
        special_qspec.layers.extend(residual_cqdq_qspec.layers)

        # Use BN QDQ Case
        bn_cqdq = BNQDQCase()
        bn_cqdq_qspec = bn_cqdq.case(keras_model, qspec)
        special_qspec.layers.extend(bn_cqdq_qspec.layers)

        # Use MaxPool QDQ Case (necessary for ResNet-v2)
        mp_cqdq = MaxPoolQDQCase()
        mp_cqdq_qspec = mp_cqdq.case(keras_model, qspec)
        special_qspec.layers.extend(mp_cqdq_qspec.layers)

        return special_qspec


class EfficientNetQDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        return (
            "Returns all quantizable nodes in EfficientNet:"
            "  1. Residual connections,"
            "  2. Quantize inputs (0, 1) of Multiply layers in SE (Squeeze-Excite) block."
        )

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:
        special_qspec = QuantizationSpec()

        # Use Residual connection QDQ
        residual_cqdq = ResidualConnectionQDQCase()
        residual_cqdq_qspec = residual_cqdq.case(keras_model, qspec)
        special_qspec.layers.extend(residual_cqdq_qspec.layers)

        # Implement EfficientNet specific case to trigger horizontal fusion in Mul residual branch.
        #   Gives preference to the user-specified `qspec`.
        for layer in keras_model.layers:
            if (
                    isinstance(layer, tf.keras.layers.Multiply)
                    and check_is_quantizable_by_layer_name(qspec, layer.name)
            ):
                special_qspec.add(
                    layer.name,
                    quantize_input=True,
                    quantize_weight=False,
                    quantization_index=[0, 1],
                )
        return special_qspec


class MobileNetQDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        return (
            "Returns all quantizable nodes in MobileNet: "
            "  1. Residual connections."
        )

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:
        special_qspec = QuantizationSpec()

        # Use Residual connection QDQ
        residual_cqdq = ResidualConnectionQDQCase()
        residual_cqdq_qspec = residual_cqdq.case(keras_model, qspec)
        special_qspec.layers.extend(residual_cqdq_qspec.layers)

        return special_qspec


class InceptionQDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self) -> str:
        return (
            "Returns all quantizable nodes in Inception-v3: "
            "  1. MaxPool layers to trigger horizontal fusion in the output of Concat."
        )

    def case(
        self, keras_model: tf.keras.Model, qspec: QuantizationSpec
    ) -> QuantizationSpec:
        special_qspec = QuantizationSpec()

        # Use MaxPool QDQ Case
        mp_cqdq = MaxPoolQDQCase()
        mp_cqdq_qspec = mp_cqdq.case(keras_model, qspec)
        special_qspec.layers.extend(mp_cqdq_qspec.layers)

        return special_qspec