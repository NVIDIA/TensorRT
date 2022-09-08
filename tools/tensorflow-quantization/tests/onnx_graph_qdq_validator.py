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


import onnx
import onnx_graphsurgeon as gs
import tensorflow as tf
from tensorflow_quantization.quantize import LayerConfig, quantize_model
from typing import List, Tuple
from tensorflow_quantization.utils import convert_saved_model_to_onnx
import copy

EXPECTED_QDQ_INSERTION = [
    LayerConfig(name="Conv2D", is_keras_class=True),
    LayerConfig(name="Dense", is_keras_class=True),
    LayerConfig(name="DepthwiseConv2D", is_keras_class=True),
    LayerConfig(
        name="Concatenate",
        is_keras_class=True,
        quantize_weight=False,
        quantization_index=["all"],
    ),
    LayerConfig(
        name="AveragePooling2D", is_keras_class=True, quantize_weight=False
    ),
    LayerConfig(
        name="GlobalAveragePooling2D", is_keras_class=True, quantize_weight=False
    )
]


class ONNXQDQValidator:
    """
    Validate ONNX file for correct QDQ insertion.
    All onnx-graphsurgeon terminologies are used in the explanations.
    """

    def __init__(self) -> None:
        self.expected_qdq_layer_behavior = {}
        self.graph = None
        self.data_format = tf.keras.backend.image_data_format()

    @staticmethod
    def _extract_layer_names_from_class_type(
            expected_qdq_behavior, original_keras_model
    ):
        """Checks if expected_qdq_behavior has items where is_keras_class=True and extract all layers relevant to it.
        Also checks if the user didn't specifically name that layer in expected_qdq_behavior.
        """

        def layer_is_class_type(class_specs, origin_layer):
            for c in class_specs:
                if origin_layer.__class__.__name__ == c.name:
                    return c
            return None

        def skip_layer_name(layer_name):
            for layer in expected_qdq_behavior:
                if layer_name == layer.name:
                    return True
            return False

        expected_qdq_behavior_class = [
            layer for layer in expected_qdq_behavior if layer.is_keras_class
        ]
        expected_qdq_behavior_layers = [
            layer for layer in expected_qdq_behavior
            # Skip if quantize_input and quantize_weight=False
            if not layer.is_keras_class and (layer.quantize_input or layer.quantize_weight)
        ]

        if original_keras_model is not None:
            for original_layer in original_keras_model.layers:
                class_type = layer_is_class_type(
                    expected_qdq_behavior_class, original_layer
                )
                if class_type is not None and not skip_layer_name(original_layer.name):
                    # Skip if quantize_input and quantize_weight=False
                    if class_type.quantize_input or class_type.quantize_weight:
                        expected_qdq_behavior_layers.append(
                            LayerConfig(
                                name=original_layer.name,
                                quantize_input=class_type.quantize_input,
                                quantize_weight=class_type.quantize_weight,
                                quantization_index=class_type.quantization_index,
                            )
                        )

        return expected_qdq_behavior_layers

    def _collect_layer_names(self, expected_qdq_behavior):
        """
        Populates the global variable 'self.expected_qdq_layer_behavior', a dictionary in the format:
            key (string) : Layer name after quantization wrapper is applied.
            value (list) : List with layer specific parameters.
                value[0] (bool) = True if this layer is a keras class.
                value[1] (bool) = True if input to this layer should be quantized.
                value[2] (bool) = True if layer weight should be quantized.
                value[3] (list) = List of quantization index, if any.
                value[4] (bool) = Set to False initially but when quantization of this layer is verified, set to True.
        """
        for layer in expected_qdq_behavior:
            self.expected_qdq_layer_behavior["quant_" + layer.name] = []
            self.expected_qdq_layer_behavior["quant_" + layer.name].append(
                layer.is_keras_class
            )
            self.expected_qdq_layer_behavior["quant_" + layer.name].append(
                layer.quantize_input
            )
            self.expected_qdq_layer_behavior["quant_" + layer.name].append(
                layer.quantize_weight
            )
            self.expected_qdq_layer_behavior["quant_" + layer.name].append(
                layer.quantization_index if layer.quantization_index is not None else []
            )
            self.expected_qdq_layer_behavior["quant_" + layer.name].append(False)

    def _load_onnx_graph(self, onnx_model_path):
        self.graph = gs.import_onnx(onnx.load(onnx_model_path))

    def _get_tf_name_of_node(self, onnx_node):
        splitted_node_name = onnx_node.name.split("/")
        if len(splitted_node_name) > 1:
            # This is other node than QuantizeLinear or DequantizeLinear
            node_op = onnx_node.op

            # Most layers have their name in position -2
            #   List: Conv, BatchNormalization, Relu, Add, MatMul, Softmax, Pad, MaxPool, GlobalAveragePool
            # Exceptions: Squeeze, Transpose, Reshape
            if node_op == "Squeeze" or node_op == "Transpose" or node_op == "Reshape":
                return splitted_node_name[-1]

            # Quantized layers
            for exp_qdq_layer_name in self.expected_qdq_layer_behavior.keys():
                if exp_qdq_layer_name + "/" in onnx_node.name:
                    return exp_qdq_layer_name

            # Other layers
            return splitted_node_name[-2]
        else:
            return None

    def _get_input_tensor_parent(self, onnx_node, input_idx):
        """
        Get input Tensors parent recursively.
        Here we want to know id DequantizeLinear is tensors parent.
        Recursively we go up the graph since reshape layers are added while onnx conversion between QDQ and node.
        """
        current_node_ip_tensor = onnx_node.inputs[input_idx]
        try:
            current_node_ip_tensor_parent = current_node_ip_tensor.inputs[0]
        except IndexError:
            # Example for weight Tensor, parent is None
            return None
        while (
                current_node_ip_tensor_parent.op == "Transpose"
                or current_node_ip_tensor_parent.op == "Reshape"
        ):  # and self.data_format == "channels_last":
            # When image data format is 'channels_last' or Conv is of type 'Depthwise', Transpose and/or Reshape
            #   layers are added between QDQ and target layer. Always select input at index 0 since it's the
            #   variable coming from the previous node. Other indices, if present, are constant inputs to the node.
            current_node_ip_tensor = current_node_ip_tensor_parent.inputs[0]
            try:
                current_node_ip_tensor_parent = current_node_ip_tensor.inputs[0]
            except IndexError:
                # We can't move upwards anymore in the graph
                break

        return current_node_ip_tensor_parent.op

    def _weighted_qdq_behavior(self, node, tf_node_name):
        """
        For weighted layers such as MatMul/Conv2D and DepthwiseConv2D, only quantize_input and quantize_weight options
          are valid.
        Input has length of 2 usually.
        Index 0 is Variable i.e. output of previous op
        Index 1 is Constant i.e. weight
        NOTE: In general, for node with more than one inputs, index 0 is variable coming out of previous node.
              Other indices are constant inputs to the node.
        """
        # case 1. Only one of quantize_input or quantize_weight is True
        if (
                not self.expected_qdq_layer_behavior[tf_node_name][1]
                or not self.expected_qdq_layer_behavior[tf_node_name][2]
        ):
            # subcase 1. When quantize_input=False
            if not self.expected_qdq_layer_behavior[tf_node_name][1]:
                if self._get_input_tensor_parent(node, 0) == "DequantizeLinear":
                    print(
                        "[E] quantize_input=False but still input is quantized for weighted layer `{}`".format(
                            tf_node_name
                        )
                    )
                    return False
                # send update that correct quantization is found for intended layer.
                self.expected_qdq_layer_behavior[tf_node_name][-1] = True
            # subcase 2. When quantize_weight=False
            if not self.expected_qdq_layer_behavior[tf_node_name][2]:
                if self._get_input_tensor_parent(node, 1) == "DequantizeLinear":
                    print(
                        "[E] quantize_weight=False but still weight is quantized for weighted layer `{}`".format(
                            tf_node_name
                        )
                    )
                    return False
                # send update that correct quantization is found for intended layer.
                self.expected_qdq_layer_behavior[tf_node_name][-1] = True
        else:
            # case 2. Both quantize_input=True, quantize_weight=True
            # Every input should be output of DequantizeLinear op
            parent_check = []
            for idx in range(len(node.inputs)):
                input_tensor_parent = self._get_input_tensor_parent(node, idx)
                if input_tensor_parent != "DequantizeLinear":
                    parent_check.append(0)
                else:
                    parent_check.append(1)
            # Check if both input and weight are quantized (2 inputs == 'DequantizeLinear')
            #   This takes into consideration that Conv sometimes has a BiasAdd input, which is not quantized.
            if sum(parent_check) < 2:
                print(
                    "[E] quantize_weight=True and quantize_input=True but still not all inputs are quantized for "
                    "weighted layer `{}`".format(tf_node_name)
                )
                return False
            # send update that correct quantization is found for intended layer.
            self.expected_qdq_layer_behavior[tf_node_name][-1] = True
        return True

    def _pool_qdq_behavior(self, node, tf_node_name):
        """
        Pool layer has just one input which is variable coming from the previous op.
        """
        if self._get_input_tensor_parent(node, 0) != "DequantizeLinear":
            print(
                "[E] Variable input for MaxPool layer `{}` is not quantized.".format(
                    tf_node_name
                )
            )
            return False
        # send update that correct quantization is found for intended layer.
        self.expected_qdq_layer_behavior[tf_node_name][-1] = True
        return True

    def _bn_qdq_behavior(self, node, tf_node_name):
        """
        BN has one variable input and four (scale, beta, mean, var) constant inputs.
        Remember variable input is always at index 0
        For quantization, just check QDQ nodes insertion in variable input.
        """
        # Check if the parent node is not DequantizeLinear and input should be quantized.
        # Reason: in the ResNet CustomQDQCase, BN is only quantized when preceded by Conv. Otherwise, quantize_input
        #  (and quantize_weight) is set to False.
        quantize_input = self.expected_qdq_layer_behavior[tf_node_name][1]
        if (
                self._get_input_tensor_parent(node, 0) != "DequantizeLinear"
                and quantize_input
        ):
            print(
                "[E] Variable input for BatchNormalization layer `{}` is not quantized.".format(
                    tf_node_name
                )
            )
            return False
        # send update that correct quantization is found for intended layer.
        self.expected_qdq_layer_behavior[tf_node_name][-1] = True
        return True

    def _multi_input_qdq_behavior(self, node, tf_node_name):
        """
        For layers with multiple inputs, we need to check whether each intended layer is quantized.
        """
        # There is quantization index list, check if provided indices are output of DequantizeLinear
        for _, e in enumerate(self.expected_qdq_layer_behavior[tf_node_name][3]):
            if e in ["any", "all"]:
                all_inputs = len(node.inputs)
                q_inputs = 0
                for inp_idx in range(all_inputs):
                    if node.i(inp_idx).op == "DequantizeLinear":
                        q_inputs += 1
                if e == "any" and q_inputs != 1:
                    print(
                        "[E] quantization_index=['{}'] thus only one input should be quantized, but {} out of {} "
                        "inputs are quantized for layer `{}`".format(e, q_inputs, all_inputs, tf_node_name)
                    )
                    return False
                elif e == "all" and q_inputs != all_inputs:
                    print(
                        "[E] quantization_index=['{}'] thus all inputs should be quantized, but {} out of {} "
                        "inputs are quantized for layer `{}`".format(e, q_inputs, all_inputs, tf_node_name)
                    )
                    return False
                # send update that correct quantization is found for intended layer.
                self.expected_qdq_layer_behavior[tf_node_name][-1] = True
            else:
                if node.i(e).op != "DequantizeLinear":
                    print(
                        "[E] Input at index {e} in layer `{tf_node_name}` should be quantized but it is not.".format(
                            e=e, tf_node_name=tf_node_name
                        )
                    )
                    return False
                # send update that correct quantization is found for intended layer.
                self.expected_qdq_layer_behavior[tf_node_name][-1] = True
        return True

    def _non_quantized_layer_qdq_behavior(self, node, tf_node_name):
        """
        Squeeze layer should not be quantized.
        """
        if self._get_input_tensor_parent(node, 0) == "DequantizeLinear":
            print(
                "[E] Variable input for {node_op} layer `{tf_node_name}` is quantized.".format(
                    node_op=node.op, tf_node_name=tf_node_name
                )
            )
            return False
        # send update that correct quantization is found for intended layer.
        self.expected_qdq_layer_behavior[tf_node_name][-1] = True
        return True

    def _qdq_monitor(self, node, tf_node_name):
        m = {
            "MatMul": self._weighted_qdq_behavior,
            "Conv": self._weighted_qdq_behavior,
            "MaxPool": self._pool_qdq_behavior,
            "AveragePool": self._pool_qdq_behavior,
            "GlobalAveragePool": self._pool_qdq_behavior,
            "BatchNormalization": self._bn_qdq_behavior,
            "Concat": self._multi_input_qdq_behavior,
            "Add": self._multi_input_qdq_behavior,
            "Mul": self._multi_input_qdq_behavior,
        }
        if node.op not in m:  # Squeeze, Softmax, ...
            m[node.op] = self._non_quantized_layer_qdq_behavior
        return m[node.op](node, tf_node_name)

    def check_onnx_node(self, node_name):
        for node in self.graph.nodes:
            if node.name == node_name:
                print(node)

    def _unintended_layer_quantize_check_pass(self):
        """
        Check whether un-intended layer is quantized.
        If any un-intended layer is quantized, checking fails immediately.
        """
        for node in self.graph.nodes:
            tf_node_name = self._get_tf_name_of_node(node)
            if tf_node_name and "quant" in tf_node_name:
                if tf_node_name not in self.expected_qdq_layer_behavior:
                    print(
                        "[E] layer `{}` should not be quantized.".format(tf_node_name)
                    )
                    return False
        return True

    def _intended_layer_quantize_check_pass(self):
        """
        Checks if the layers exists and whether all expected layers are quantized.
        If any intended layer is not quantized, checking fails immediately.
        """
        for k, v in self.expected_qdq_layer_behavior.items():
            check_quant_layer_exists = any([k + "/" in node.name for node in self.graph.nodes])
            check_original_layer_exists = any([k.replace("quant_", "") + "/" in node.name for node in self.graph.nodes])
            if not check_quant_layer_exists:
                if check_original_layer_exists:
                    print("[E] layer `{}` should have been quantized but wasn't.".format(k.replace("quant_", "")))
                    return False
                else:
                    print("[W] layer `{}` does not exist.".format(k))
                    continue
            elif not v[-1]:
                print("[E] layer `{}` should be quantized but it did not.".format(k))
                return False
        return True

    def _qdq_insertion_check_pass(self):
        """
        Validate QDQ insertion.
        """
        check_status = True
        for node in self.graph.nodes:
            tf_node_name = self._get_tf_name_of_node(node)
            if tf_node_name and "quant" in tf_node_name:
                check_status = check_status and self._qdq_monitor(node, tf_node_name)
                if not check_status:
                    return check_status
        return check_status

    def validate(
            self, onnx_model_path, expected_qdq_behavior, original_keras_model=None
    ):
        self._load_onnx_graph(onnx_model_path)
        expected_qdq_behavior = self._extract_layer_names_from_class_type(
            expected_qdq_behavior, original_keras_model
        )
        # Populate 'self.expected_qdq_layer_behavior'
        self._collect_layer_names(expected_qdq_behavior)
        ulcp = self._unintended_layer_quantize_check_pass()
        if not ulcp:
            print("[I] Unintended layer quantization check failed.")
            return False
        qicp = self._qdq_insertion_check_pass()
        if not qicp:
            print("[I] Quantize insertion check failed.")
            return False
        ilcp = self._intended_layer_quantize_check_pass()
        if not ilcp:
            print("[I] Intended layer quantization check failed.")
            return False
        return True


def get_expected_qdq_insertion(
        nn_model_original: tf.keras.Model,
        qspec_test: "QuantizationSpec" = None,
        custom_qdq_cases: List["CustomQDQInsertionCase"] = None,
        quantization_mode: str = "full",
        expected_qdq_insertion_user: List[LayerConfig] = None
) -> List[LayerConfig]:
    """
    Gets expected QDQ insertion.

    Args:
        nn_model_original (tf.keras.Model): baseline model (non-quantized), needed to obtain all layers quantized with
            Custom QDQ Case.
        qspec_test (QuantizationSpec): Quantization specification to test the quantized model with.
        custom_qdq_cases (List[CustomQDQInsertionCase]): indicates layers with custom QDQ placements
            (i.e., ResidualConnectionQDQCase).
        quantization_mode (str): quantization mode, can be "full" or "partial".
        expected_qdq_insertion_user (List[LayerConfig]): List of layer configs specified by the user. If 'None', use
            the default quantization behavior.
    Returns:
        expected_qdq_insertion (List[LayerConfig]): list with expected QDQ node placements.
    """
    # 1. Establish QDQ node placement behavior for all relevant classes
    if expected_qdq_insertion_user is not None:
        # User-specified QDQ behavior
        expected_qdq_insertion = expected_qdq_insertion_user
    else:
        if quantization_mode == "partial":
            # No classes are quantized by default
            expected_qdq_insertion = []
        else:
            # Default quantization behavior
            expected_qdq_insertion = copy.deepcopy(EXPECTED_QDQ_INSERTION)

    # 2. Extend quantization behavior with the user's specifications
    if qspec_test is not None:
        # Only add layers that are being quantized (don't add when `quantize_input` or 'quantize_weight`=False)
        expected_qdq_insertion.extend(qspec_test.layers)

    # 3. Extend quantization behavior with the Custom QDQ Cases
    if custom_qdq_cases is not None:
        for custom_qdq_case in custom_qdq_cases:
            qspec_case_object = custom_qdq_case.case(nn_model_original, qspec=qspec_test)
            expected_qdq_insertion.extend(qspec_case_object.layers)

    # 4. Check if Multiple Input classes have empty or None 'quantization_index'. If so, update it to
    #   'quantization_index=["all"]'.
    for exp_insertion in expected_qdq_insertion:
        if exp_insertion.is_keras_class and exp_insertion.name in ['Add', 'Multiply', 'Concatenate']:
            if not exp_insertion.quantization_index:  # None or []
                exp_insertion.quantization_index = ["all"]

    return expected_qdq_insertion


# ###############################################
# ######### Full QAT workflow test ##############
# ###############################################


def validate_quantized_model(
        test_assets: "CreateAssetsFolders",
        nn_model_original: tf.keras.Model,
        quantization_mode: str = "full",
        qspec: "QuantizationSpec" = None,
        custom_qdq_cases: List["CustomQDQInsertionCase"] = None,
        test_name: str = "test",
        expected_qdq_insertion: List["LayerConfig"] = None
) -> Tuple[tf.keras.Model, bool]:
    """
    Full test workflow: quantization, obtain expected QDQ node placements, check node placements against expected.

    Args:
        test_assets (CreateAssetsFolders): Folder organizer.
        nn_model_original (tf.keras.Model): Keras model.
        quantization_mode (str): quantization mode, can be "full" or "partial".
        qspec (QuantizationSpec): QuantizationSpec for model quantization.
        custom_qdq_cases (List[CustomQDQInsertionCase]): list of custom QDQ cases for model quantization.
        test_name (str): name for this test workflow.
        expected_qdq_insertion (List[LayerConfig]): expected QDQ insertion classes and/or layers.
    Returns:
        q_model (tf.keras.Model): quantized model.
        validated (bool): indicates whether the quantized ONNX file is correct or not (according to QDQ node placements).
    """
    # Create test folders
    test_assets.add_folder(test_name)
    test_assets_attr = getattr(test_assets, test_name)

    # Save baseline model
    tf.keras.models.save_model(nn_model_original, test_assets_attr.fp32_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets_attr.fp32_saved_model,
        onnx_model_path=test_assets_attr.fp32_onnx_model,
    )

    # Quantize model
    q_model = quantize_model(
        model=nn_model_original,
        quantization_mode=quantization_mode,
        quantization_spec=copy.deepcopy(qspec),
        custom_qdq_cases=custom_qdq_cases
    )
    # Save quantized model
    tf.keras.models.save_model(q_model, test_assets_attr.int8_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets_attr.int8_saved_model,
        onnx_model_path=test_assets_attr.int8_onnx_model,
    )

    # Validate QDQ node placements in ONNX file
    expected_qdq_insertion = get_expected_qdq_insertion(
        tf.keras.models.clone_model(nn_model_original),
        qspec_test=copy.deepcopy(qspec),
        quantization_mode=quantization_mode,
        custom_qdq_cases=custom_qdq_cases,
        expected_qdq_insertion_user=expected_qdq_insertion
    )

    v = ONNXQDQValidator()
    validated = v.validate(
        test_assets_attr.int8_onnx_model, expected_qdq_insertion, original_keras_model=nn_model_original
    )
    return q_model, validated
