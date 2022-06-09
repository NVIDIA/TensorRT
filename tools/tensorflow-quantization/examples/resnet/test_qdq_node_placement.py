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
from tensorflow_quantization.quantize import QuantizationSpec
from tensorflow_quantization.custom_qdq_cases import ResNetV1QDQCase, ResNetV2QDQCase
from examples.resnet.utils import get_resnet_model
from tests.onnx_graph_qdq_validator import validate_quantized_model
from tensorflow_quantization.utils import CreateAssetsFolders
import pytest

# Create a directory to save test models
test_assets = CreateAssetsFolders("test_qdq_node_placement")


def test_resnet50v1_quantize_full():
    """
    ResNet-v1: Full model quantization
    """
    this_function_name = sys._getframe().f_code.co_name

    # Instantiate Baseline model
    nn_model_original = get_resnet_model(resnet_depth="50", resnet_version="v1")

    custom_qdq_cases = [ResNetV1QDQCase()]
    q_model, validated = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name,
        custom_qdq_cases=custom_qdq_cases
    )
    assert validated, "ONNX QDQ validation for full network quantization failed!"
    # necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()


def test_resnet50v1_quantize_full_special():
    """
    ResNet-v1: Full model quantization with the first Conv layer (conv2_block1_1_conv) not being quantized.
    """
    this_function_name = sys._getframe().f_code.co_name

    # Instantiate Baseline model
    nn_model_original = get_resnet_model(resnet_depth="50", resnet_version="v1")

    # Full quantization
    custom_qdq_cases = [ResNetV1QDQCase()]
    # Special case
    qspec = QuantizationSpec()
    qspec.add(name="conv2_block1_1_conv", quantize_input=False, quantize_weight=False)

    q_model, validated = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name,
        qspec=qspec, custom_qdq_cases=custom_qdq_cases
    )

    assert validated, "ONNX QDQ validation for full network quantization failed!"
    # necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()


def test_resnet50v2_quantize_full():
    """
    ResNet-v2: Full model quantization
    """
    this_function_name = sys._getframe().f_code.co_name

    # Instantiate Baseline model
    nn_model_original = get_resnet_model(resnet_depth="50", resnet_version="v2")

    custom_qdq_cases = [ResNetV2QDQCase()]
    q_model, validated = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name,
        custom_qdq_cases=custom_qdq_cases
    )
    assert validated, "ONNX QDQ validation for full network quantization failed!"
    # necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()


def test_resnet50v2_quantize_full_special():
    """
    ResNet-v2: Full model quantization with the first MaxPool layer (pool1_pool) not being quantized.
    """
    this_function_name = sys._getframe().f_code.co_name

    # Instantiate Baseline model
    nn_model_original = get_resnet_model(resnet_depth="50", resnet_version="v2")

    custom_qdq_cases = [ResNetV2QDQCase()]
    qspec = QuantizationSpec()
    qspec.add(name="pool1_pool", quantize_input=False, quantize_weight=False)
    q_model, validated = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name,
        qspec=qspec, custom_qdq_cases=custom_qdq_cases
    )
    assert validated, "ONNX QDQ validation for full network quantization failed!"
    # necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()
