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


import os
import sys
import tensorflow_quantization.utils as utils
import tensorflow as tf
from tensorflow_quantization import quantize_model
from tensorflow_quantization.utils import (
    CreateAssetsFolders,
    convert_saved_model_to_onnx,
)
from network_pool import sam_32_32
import pytest

test_assets = CreateAssetsFolders("test_utils")


def test_keras_traveller():
    kmt = utils.KerasModelTraveller()
    model = sam_32_32()
    layer_names = kmt.get_layer_names(keras_model=model)
    expected_layer_names = [
        "input_1",
        "conv2d",
        "re_lu",
        "conv2d_1",
        "re_lu_1",
        "conv2d_2",
        "re_lu_2",
        "conv2d_3",
        "add",
        "re_lu_3",
        "conv2d_4",
        "re_lu_4",
        "conv2d_5",
        "add_1",
        "re_lu_5",
        "conv2d_6",
        "re_lu_6",
        "conv2d_7",
        "conv2d_8",
        "add_2",
        "re_lu_7",
        "max_pooling2d",
        "flatten",
        "dense",
        "re_lu_8",
        "dense_1",
    ]
    assert layer_names == expected_layer_names, "Keras model traveller failed."
    tf.keras.backend.clear_session()


def test_convert_to_onnx():
    test_assets.add_folder("test_convert_to_onnx")

    model = sam_32_32()
    q_model = quantize_model(model)
    # Create experiment specific directory
    tf.keras.models.save_model(
        q_model, test_assets.test_convert_to_onnx.int8_saved_model
    )
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets.test_convert_to_onnx.int8_saved_model,
        onnx_model_path=test_assets.test_convert_to_onnx.int8_onnx_model,
    )
    tf.keras.backend.clear_session()


def test_find_my_predecessors():
    resnet50 = tf.keras.applications.resnet.ResNet50(weights=None)
    r = utils.find_my_predecessors(resnet50, "conv2_block1_add")
    assert r[0]["class"] == "BatchNormalization"
    assert r[0]["name"] == "conv2_block1_0_bn"
    assert r[1]["class"] == "BatchNormalization"
    assert r[1]["name"] == "conv2_block1_3_bn"


def test_find_my_successors():
    resnet50 = tf.keras.applications.resnet.ResNet50(weights=None)
    r = utils.find_my_successors(resnet50, "pool1_pool")
    assert r[0]["class"] == "Conv2D"
    assert r[0]["name"] == "conv2_block1_1_conv"
    assert r[1]["class"] == "Conv2D"
    assert r[1]["name"] == "conv2_block1_0_conv"
