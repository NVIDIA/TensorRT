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


import sys
import tensorflow as tf
from tensorflow_quantization import QuantizationSpec
from tensorflow_quantization.custom_qdq_cases import ResNetV1QDQCase
from network_pool import frodo_32_32
from onnx_graph_qdq_validator import validate_quantized_model
from tensorflow_quantization.utils import CreateAssetsFolders
import pytest

test_assets = CreateAssetsFolders("test_quantize_qdq_insertion")

# ############################################
# ######### Full Quantize Test ###############
# ############################################


def test_quantize_full():
    this_function_name = sys._getframe().f_code.co_name

    nn_model_original = frodo_32_32()
    q_model, vr = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name
    )

    assert vr, "ONNX QDQ Validation for full network quantization failed!"
    # Necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()


def test_quantize_full_residual():
    this_function_name = sys._getframe().f_code.co_name

    nn_model_original = frodo_32_32()
    q_model, vr = validate_quantized_model(
        test_assets, nn_model_original, custom_qdq_cases=[ResNetV1QDQCase()], test_name=this_function_name
    )

    assert vr, "ONNX QDQ Validation for quantizing full network with special residual failed!"
    tf.keras.backend.clear_session()


# ############################################
# ######### Full Special Quantize Test #######
# ############################################


def test_quantize_full_special_layer():
    this_function_name = sys._getframe().f_code.co_name

    nn_model_original = frodo_32_32()

    # Create a Quantization Spec (dictionary telling how `add` layer should be treated differently).
    qspec = QuantizationSpec()
    qspec.add(name="add", quantization_index=[0])
    q_model, vr = validate_quantized_model(
        test_assets, nn_model_original, qspec=qspec, test_name=this_function_name
    )

    assert vr, "QDQ Validation for full network but one special layer quantization failed!"
    tf.keras.backend.clear_session()


# ##########################################
# ######### Partial Quantize Test ##########
# ##########################################


def test_quantize_partial():
    this_function_name = sys._getframe().f_code.co_name

    nn_model_original = frodo_32_32()

    # Create a qspec dictionary to quantize only two layers named 'conv2d_2' and 'dense'
    qspec = QuantizationSpec()
    qspec.add(name="conv2d_2")
    qspec.add(name="dense")
    q_model, vr = validate_quantized_model(
        test_assets, nn_model_original, quantization_mode="partial", qspec=qspec, test_name=this_function_name
    )

    assert vr, "ONNX QDQ Validation for partial network quantization failed!"
    tf.keras.backend.clear_session()


# ####################################################
# ######### Subset layers Test - Full quantize #######
# ####################################################


def test_quantize_specific_class_maxpool():
    this_function_name = sys._getframe().f_code.co_name

    nn_model_original = frodo_32_32()

    # Create a list with keras layer classes to quantize
    qspec = QuantizationSpec()
    qspec.add(name="MaxPooling2D", is_keras_class=True)
    q_model, vr = validate_quantized_model(
        test_assets, nn_model_original, qspec=qspec, test_name=this_function_name
    )

    assert vr, "ONNX QDQ Validation for specific class `Dense` quantization failed!"
    tf.keras.backend.clear_session()


def test_quantize_specific_class_add():
    this_function_name = sys._getframe().f_code.co_name

    nn_model_original = frodo_32_32()

    # Create a list with keras layer classes to quantize
    qspec = QuantizationSpec()
    qspec.add(name="Add", is_keras_class=True)
    q_model, vr = validate_quantized_model(
        test_assets, nn_model_original, qspec=qspec, test_name=this_function_name
    )

    assert vr, "ONNX QDQ Validation for quantizing specific class `Add` failed!"
    tf.keras.backend.clear_session()


# ####################################################
# ####### Subset layers Test - Partial quantize ######
# ####################################################


def test_quantize_specific_class_conv2d_partial():
    this_function_name = sys._getframe().f_code.co_name

    nn_model_original = frodo_32_32()

    # Create a list with keras layer classes to quantize
    qspec = QuantizationSpec()
    qspec.add(name="Conv2D", is_keras_class=True)
    q_model, vr = validate_quantized_model(
        test_assets, nn_model_original, quantization_mode="partial", qspec=qspec, test_name=this_function_name
    )

    assert vr, "ONNX QDQ Validation for quantizing specific class `Conv2D` and `conv2d_1` layer failed!"
    tf.keras.backend.clear_session()
