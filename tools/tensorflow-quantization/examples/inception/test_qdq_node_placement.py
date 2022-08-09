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
from tensorflow_quantization.custom_qdq_cases import InceptionQDQCase
from examples.utils import get_tfkeras_model
from tests.onnx_graph_qdq_validator import validate_quantized_model
from tensorflow_quantization.utils import CreateAssetsFolders
from tensorflow_quantization.quantize import LayerConfig
import pytest

# Create a directory to save test models
test_assets = CreateAssetsFolders("test_qdq_node_placement")

EXPECTED_QDQ_INSERTION = [
    LayerConfig(name="Conv2D", is_keras_class=True),
    LayerConfig(name="Dense", is_keras_class=True),
    LayerConfig(name="DepthwiseConv2D", is_keras_class=True),
    LayerConfig(
        name="AveragePooling2D", is_keras_class=True, quantize_weight=False
    ),
    LayerConfig(
        name="GlobalAveragePooling2D", is_keras_class=True, quantize_weight=False
    )
]


def test_inceptionv3_quantize_full():
    """
    Inception-v3: Full model quantization
    """
    this_function_name = sys._getframe().f_code.co_name

    # Instantiate Baseline model
    nn_model_original = get_tfkeras_model(model_name="inception_v3")

    custom_qdq_cases = [InceptionQDQCase()]
    q_model, validated = validate_quantized_model(
        test_assets, nn_model_original, test_name=this_function_name,
        custom_qdq_cases=custom_qdq_cases,
        expected_qdq_insertion=EXPECTED_QDQ_INSERTION
    )
    assert validated, "ONNX QDQ validation for full network quantization failed!"
    # necessary to clear model layer names from the memory
    tf.keras.backend.clear_session()
