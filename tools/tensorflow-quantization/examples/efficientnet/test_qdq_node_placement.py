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
from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase
from utils import create_efficientnet_model
from tests.onnx_graph_qdq_validator import validate_quantized_model
from tensorflow_quantization.utils import CreateAssetsFolders
import pytest

# Create a directory to save test models
test_assets = CreateAssetsFolders("test_qdq_node_placement")


def test_efficientnet_b0_quantize_full():
    """
    EfficientNet-B0: Full model quantization.

    Contains special patterns connected to Add layer:
        1. (Conv->BatchNorm->Activation)->Add
        2. (Conv->BatchNorm->Activation->Dropout)->Add
    Previously, only (Conv)->Add and (Conv->BatchNorm)->Add checks were done when checking for quantizable Residual
        connections (branches to `Add` layer). The new EfficientNet patterns have now also been added to the
        ResidualCustomQDQCases check.
    """
    this_function_name = sys._getframe().f_code.co_name

    # Instantiate Baseline model
    nn_model_original = create_efficientnet_model()

    custom_qdq_cases = [EfficientNetQDQCase()]
    q_model, validated = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name,
        custom_qdq_cases=custom_qdq_cases
    )
    assert validated, "ONNX QDQ validation for full network quantization failed!"
    # necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()


def test_efficientnet_b3_quantize_full():
    """
    EfficientNet-B3: Full model quantization.
    Contains the same special patterns as EfficientNet-B0.
    """
    this_function_name = sys._getframe().f_code.co_name

    # Instantiate Baseline model
    nn_model_original = create_efficientnet_model(model_version="b3")

    custom_qdq_cases = [EfficientNetQDQCase()]
    q_model, validated = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name,
        custom_qdq_cases=custom_qdq_cases
    )
    assert validated, "ONNX QDQ validation for full network quantization failed!"
    # necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()
