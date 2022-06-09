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
from tensorflow_quantization.quantize import LayerConfig
from onnx_graph_qdq_validator import validate_quantized_model
from tensorflow_quantization.utils import CreateAssetsFolders
from network_pool import (
    otho_28_28,
    lotho_28_28,
    lobelia_28_28,
    merry_28_28,
    pippin_28_28,
)
import pytest

# Create a directory to save wrapper test data
test_assets = CreateAssetsFolders("test_quantize_wrappers")


# ###################################################
# ####### Conv2D layer wrapper tests ################
# ###################################################


def test_conv2d_wrapper_quant_full_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = otho_28_28()

    # Quantization
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv_0"),
        LayerConfig(name="conv_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_conv2d_wrapper_quant_partial_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = otho_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="conv_1")
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_conv2d_wrapper_quant_partial_only_input_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = otho_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="conv_0", quantize_weight=False)
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv_0", quantize_weight=False),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_conv2d_wrapper_quant_partial_only_weight_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = otho_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="conv_0", quantize_input=False)
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv_0", quantize_input=False),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


# ###################################################
# ####### DepthwiseConv2D layer wrapper tests #######
# ###################################################


def test_depthwise_conv2d_wrapper_quant_full_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lotho_28_28()

    # Quantization
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dconv_0"),
        LayerConfig(name="dconv_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_depthwise_conv2d_wrapper_quant_partial_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lotho_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="dconv_1")
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dconv_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_depthwise_conv2d_wrapper_quant_partial_only_input_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lotho_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="dconv_1", quantize_weight=False)
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dconv_1", quantize_weight=False),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_depthwise_conv2d_wrapper_quant_partial_only_weight_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lotho_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="dconv_1", quantize_input=False)
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dconv_1", quantize_input=False),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


# ###################################################
# ####### Dense layer wrapper tests #################
# ###################################################


def test_dense_wrapper_quant_full_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lobelia_28_28()

    # Quantization
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dense_0"),
        LayerConfig(name="dense_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_dense_wrapper_quant_partial_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lobelia_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="dense_0")
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dense_0"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_dense_wrapper_quant_partial_only_input_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lobelia_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="dense_0", quantize_weight=False)
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dense_0", quantize_weight=False),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_dense_wrapper_quant_partial_only_weight_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = lobelia_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="dense_1", quantize_input=False)
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="dense_1", quantize_input=False),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"

    tf.keras.backend.clear_session()


# ###################################################
# ####### Concatenation layer wrapper tests #########
# ###################################################


def test_concat_wrapper_quant_full_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = merry_28_28()

    # Quantization
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv2d"),
        LayerConfig(name="conv2d_1"),
        LayerConfig(name="conv2d_3"),
        LayerConfig(name="conv2d_4"),
        LayerConfig(name="conv2d_2"),
        LayerConfig(name="dense"),
        LayerConfig(name="dense_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_concat_wrapper_quant_full_quant_bn_concat_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = merry_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="batch_normalization_3")
    qspec.add(name="concatenate", quantization_index=[0, 1])
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv2d"),
        LayerConfig(name="conv2d_1"),
        LayerConfig(name="conv2d_3"),
        LayerConfig(name="conv2d_4"),
        LayerConfig(name="batch_normalization_3"),
        LayerConfig(name="conv2d_2"),
        LayerConfig(name="concatenate", quantization_index=[0, 1]),
        LayerConfig(name="dense"),
        LayerConfig(name="dense_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_concat_wrapper_quant_specific_index_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = merry_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(
        name="concatenate",
        quantize_input=True,
        quantize_weight=False,
        quantization_index=[0],
    )
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="concatenate", quantization_index=[0]),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


# ###################################################
# ####### Add layer wrapper tests ###################
# ###################################################


# Use KerasModelLayersSurgeon() from utils to find layer names.
def test_add_wrapper_quant_full_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = pippin_28_28()

    # Quantization
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv2d"),
        LayerConfig(name="conv2d_1"),
        LayerConfig(name="conv2d_2"),
        LayerConfig(name="conv2d_3"),
        LayerConfig(name="conv2d_4"),
        LayerConfig(name="conv2d_6"),
        LayerConfig(name="conv2d_5"),
        LayerConfig(name="dense"),
        LayerConfig(name="dense_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_add_wrapper_quant_partial_specific_index_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = pippin_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(
        name="add", quantize_input=True, quantize_weight=False, quantization_index=[1]
    )
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="add", quantization_index=[1])
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode="partial", qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


def test_add_wrapper_quant_full_specific_index_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = pippin_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(
        name="add", quantize_input=True, quantize_weight=False, quantization_index=[1]
    )
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv2d"),
        LayerConfig(name="conv2d_1"),
        LayerConfig(name="conv2d_2"),
        LayerConfig(name="conv2d_3"),
        LayerConfig(name="add", quantization_index=[1]),
        LayerConfig(name="conv2d_4"),
        LayerConfig(name="conv2d_6"),
        LayerConfig(name="conv2d_5"),
        LayerConfig(name="dense"),
        LayerConfig(name="dense_1"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


# ########################################################
# ############ Test subset layer class selection #########
# ########################################################


def test_subset_layer_class_selection_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = pippin_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(name="Conv2D", is_keras_class=True)
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="conv2d"),
        LayerConfig(name="conv2d_1"),
        LayerConfig(name="conv2d_2"),
        LayerConfig(name="conv2d_3"),
        LayerConfig(name="conv2d_4"),
        LayerConfig(name="conv2d_6"),
        LayerConfig(name="conv2d_5"),
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode='partial', qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


# ########################################################
# ############ Test missing layer name warning ###########
# ########################################################


def test_missing_layer_name_warning_nv():
    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = pippin_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(
        name="add", quantize_input=True, quantize_weight=False, quantization_index=[1]
    )
    qspec.add(name="wrong_layer")
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="add", quantization_index=[1])
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode='partial', qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()


# ########################################################
# ##### Test Add,Concat out of range index warning #######
# ########################################################


@pytest.mark.skip(
    reason="When quantization index out of range does not give error but still wraps \
add layer without quantizing any input"
)
def test_out_of_range_index():

    # Create experiment specific directory
    this_function_name = sys._getframe().f_code.co_name

    # Baseline model
    nn_model = pippin_28_28()

    # Quantization
    qspec = QuantizationSpec()
    qspec.add(
        name="add", quantize_input=True, quantize_weight=False, quantization_index=[3]
    )
    qspec.add(name="dense")
    # (Optional) QDQ node placement check
    #   Here just to explicitly show the user which layers are quantized.
    expected_qdq_insertion = [
        LayerConfig(name="add"), LayerConfig(name="dense")
    ]
    q_model, vr = validate_quantized_model(
        test_assets, nn_model, quantization_mode='partial', qspec=qspec, test_name=this_function_name,
        expected_qdq_insertion=expected_qdq_insertion
    )

    assert vr, "ONNX QDQ Validation failed!"
    tf.keras.backend.clear_session()
