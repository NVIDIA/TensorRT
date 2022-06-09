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
import tensorflow as tf
tf_models_path = os.path.realpath("./models")
sys.path.insert(1, tf_models_path)
try:
    from official.legacy.image_classification.efficientnet import efficientnet_model
except Exception:
    print("Error importing TF official models codebase.")


def create_efficientnet_model(model_version="b0"):
    model_name = "efficientnet-" + model_version
    model_configs = dict(efficientnet_model.MODEL_CONFIGS)
    assert model_name in model_configs, "Model name is not valid!"
    config = model_configs[model_name]

    # Set the dataformat of the model to NCHW for training and inference
    tf.keras.backend.set_image_data_format("channels_first")
    # B0=(224, 224, 3); B3=(300, 300, 3)
    image_input = tf.keras.layers.Input(
        shape=(config.resolution, config.resolution, config.input_channels), name="image_input", dtype=tf.float32
    )
    outputs = efficientnet_model.efficientnet(image_input, config)
    model = tf.keras.Model(inputs=image_input, outputs=outputs)

    return model
