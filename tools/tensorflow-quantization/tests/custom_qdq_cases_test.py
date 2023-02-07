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


from network_pool import pippin_28_28
from tensorflow_quantization import custom_qdq_cases
import pytest
from tensorflow_quantization import quantize_model
from tensorflow_quantization.utils import convert_saved_model_to_onnx
from tensorflow_quantization.utils import CreateAssetsFolders

import tensorflow as tf

test_assets = CreateAssetsFolders("test_custom_qdq_cases")


def test_resnet_residual_qdq_case():
    model = pippin_28_28()
    test_assets.add_folder("pipin_28_28")
    tf.keras.models.save_model(model, test_assets.pipin_28_28.fp32_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets.pipin_28_28.fp32_saved_model,
        onnx_model_path=test_assets.pipin_28_28.fp32_onnx_model,
    )

    resnet_residual_qdq = custom_qdq_cases.ResNetV1QDQCase()
    r = resnet_residual_qdq.case(model, None)
    expected_qdq_insertion = {
        "add": 1,
        "add_1": "any",
    }
    assert (
        len(r.layers) == 2
    ), "There should be 2 custom layers, but found {}".format(len(r.layers))
    for l in r.layers:
        if l.name not in expected_qdq_insertion:
            raise Exception(
                "Layer {} is not expected to be treated as custom layer".format(l.name)
            )
        else:
            if l.quantization_index != None:
                if expected_qdq_insertion[l.name] == "any":
                    continue
                assert (
                    l.quantization_index[0] == expected_qdq_insertion[l.name]
                ), "For layer {l_name}, only {expected_qdq} indices should be quantized".format(
                    l_name=l.name, expected_qdq=expected_qdq_insertion[l.name]
                )


def assert_add_bn_expected_layers(
        r, expected_add_layer_behavior, expected_bn_layer_behavior, expected_mp_layer_behavior
):
    assert len(r.layers) == (
            len(expected_add_layer_behavior) + len(expected_bn_layer_behavior) + len(expected_mp_layer_behavior)
    ), "Not all expected layers are captured for ResNet custom QDQ case."
    for l in r.layers:
        assert (
                l.name in expected_add_layer_behavior
                or l.name in expected_bn_layer_behavior
                or l.name in expected_mp_layer_behavior
        ), "layer {} is not expected to be captured for ResNet custom QDQ case".format(
            l.name
        )
        if "add" in l.name:
            if expected_add_layer_behavior[l.name] == "any":
                continue
            assert l.quantization_index[0] == expected_add_layer_behavior[l.name], (
                "For layer {l_name}, expected quantization index is {expected_add_behavior} but index {l_quant_id} "
                "is captured in ResNet custom QDQ case.".format(
                    l_name=l.name,
                    expected_add_behavior=expected_add_layer_behavior[l.name],
                    l_quant_idx=l.quantization_index[0],
                )
            )


def test_resnet50_residual_qdq_case():
    resnet50 = tf.keras.applications.resnet50.ResNet50(weights=None)
    test_assets.add_folder("resnet50_v1")
    tf.keras.models.save_model(resnet50, test_assets.resnet50_v1.fp32_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets.resnet50_v1.fp32_saved_model,
        onnx_model_path=test_assets.resnet50_v1.fp32_onnx_model,
    )

    resnet_custom_qdq_case = custom_qdq_cases.ResNetV1QDQCase()
    r = resnet_custom_qdq_case.case(resnet50, None)
    for r_1 in r.layers:
        print("\"{}\",".format(r_1.name))

    expected_add_layer_behavior = {
        "conv2_block1_add": "any",
        "conv2_block2_add": 0,
        "conv2_block3_add": 0,
        "conv3_block1_add": "any",
        "conv3_block2_add": 0,
        "conv3_block3_add": 0,
        "conv3_block4_add": 0,
        "conv4_block1_add": "any",
        "conv4_block2_add": 0,
        "conv4_block3_add": 0,
        "conv4_block4_add": 0,
        "conv4_block5_add": 0,
        "conv4_block6_add": 0,
        "conv5_block1_add": "any",
        "conv5_block2_add": 0,
        "conv5_block3_add": 0,
    }

    # Empty, no BatchNorm layers should be quantized in ResNet-v1
    expected_bn_layer_behavior = {}

    # MaxPool quantization is actually not needed in ResNet-v1
    expected_mp_layer_behavior = {}

    assert_add_bn_expected_layers(
      r, expected_add_layer_behavior, expected_bn_layer_behavior, expected_mp_layer_behavior
    )

    q_resnet50 = quantize_model(resnet50, custom_qdq_cases=[resnet_custom_qdq_case])
    tf.keras.models.save_model(q_resnet50, test_assets.resnet50_v1.int8_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets.resnet50_v1.int8_saved_model,
        onnx_model_path=test_assets.resnet50_v1.int8_onnx_model,
    )


def test_resnet50v2_bn_qdq_case():
    resnet50_v2 = tf.keras.applications.resnet_v2.ResNet50V2(weights=None)
    test_assets.add_folder("resnet50_v2")
    tf.keras.models.save_model(resnet50_v2, test_assets.resnet50_v2.fp32_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets.resnet50_v2.fp32_saved_model,
        onnx_model_path=test_assets.resnet50_v2.fp32_onnx_model,
    )

    resnet_custom_qdq_case = custom_qdq_cases.ResNetV2QDQCase()
    r = resnet_custom_qdq_case.case(resnet50_v2, None)
    for r_1 in r.layers:
        print("\"{}\",".format(r_1.name))

    expected_add_layer_behavior = {
        "conv2_block1_out": 0,
        "conv2_block2_out": 0,
        "conv2_block3_out": 0,
        "conv3_block1_out": 0,
        "conv3_block2_out": 0,
        "conv3_block3_out": 0,
        "conv3_block4_out": 0,
        "conv4_block1_out": 0,
        "conv4_block2_out": 0,
        "conv4_block3_out": 0,
        "conv4_block4_out": 0,
        "conv4_block5_out": 0,
        "conv4_block6_out": 0,
        "conv5_block1_out": 0,
        "conv5_block2_out": 0,
        "conv5_block3_out": 0,
    }

    # ResNet-v2 quantizes BatchNorms that are not connected to Conv layers
    expected_bn_layer_behavior = {
        "conv2_block1_preact_bn",
        "conv2_block2_preact_bn",
        "conv2_block3_preact_bn",
        "conv3_block1_preact_bn",
        "conv3_block2_preact_bn",
        "conv3_block3_preact_bn",
        "conv3_block4_preact_bn",
        "conv4_block1_preact_bn",
        "conv4_block2_preact_bn",
        "conv4_block3_preact_bn",
        "conv4_block4_preact_bn",
        "conv4_block5_preact_bn",
        "conv4_block6_preact_bn",
        "conv5_block1_preact_bn",
        "conv5_block2_preact_bn",
        "conv5_block3_preact_bn",
        "post_bn",
    }

    # ResNet-v2 quantizes all MaxPool layers
    expected_mp_layer_behavior = {
        "pool1_pool",
        "max_pooling2d",
        "max_pooling2d_1",
        "max_pooling2d_2",
    }

    assert_add_bn_expected_layers(
        r, expected_add_layer_behavior, expected_bn_layer_behavior, expected_mp_layer_behavior
    )

    q_resnet50_v2 = quantize_model(
        resnet50_v2, custom_qdq_cases=[resnet_custom_qdq_case]
    )
    tf.keras.models.save_model(q_resnet50_v2, test_assets.resnet50_v2.int8_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=test_assets.resnet50_v2.int8_saved_model,
        onnx_model_path=test_assets.resnet50_v2.int8_onnx_model,
    )
