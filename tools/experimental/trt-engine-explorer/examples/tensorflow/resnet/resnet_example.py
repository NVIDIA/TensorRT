#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# This script generates 2 different ONNX ResNet models:
# 1. FP32 Resnet.
# 2. QAT Resnet.


import tensorflow as tf
from tensorflow_quantization import quantize_model
from tensorflow_quantization.custom_qdq_cases import ResNetV1QDQCase
from tensorflow_quantization.utils import convert_keras_model_to_onnx


# Baseline model
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    classes=1000,
    classifier_activation="softmax",
)
convert_keras_model_to_onnx(model, onnx_model_path="generated/resnet.onnx")

# QAT model
q_model = quantize_model(model, custom_qdq_cases=[ResNetV1QDQCase()])
convert_keras_model_to_onnx(q_model, onnx_model_path="generated/resnet-qat.onnx")
