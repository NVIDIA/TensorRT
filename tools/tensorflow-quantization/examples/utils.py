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
import tensorflow as tf
from examples.data.data_loader import _NUM_CLASSES, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS
from typing import Tuple

MODELS_CLASSES_DICT = {
    "resnet_50v1": tf.keras.applications.ResNet50,
    "resnet_101v1": tf.keras.applications.ResNet101,
    "resnet_152v1": tf.keras.applications.ResNet152,
    "resnet_50v2": tf.keras.applications.ResNet50V2,
    "resnet_101v2": tf.keras.applications.ResNet101V2,
    "resnet_152v2": tf.keras.applications.ResNet152V2,
    "mobilenet_v1": tf.keras.applications.MobileNet,
    "mobilenet_v2": tf.keras.applications.MobileNetV2,
    "inception_v3": tf.keras.applications.InceptionV3,
}


def get_tfkeras_model(model_name: str = "mobilenet_v1", shape: Tuple = None) -> tf.keras.Model:
    """
    Creates a native tf.keras.applications model.

    Args:
        model_name (str): Options={model_name_options}.

    Returns:
        model (tf.keras.Model): model corresponding to 'model_name'.

    Raises:
        ValueError: raised when 'model_name' is not supported.
    """.format(
        model_name_options=list(MODELS_CLASSES_DICT.keys())
    )
    try:
        model_class = MODELS_CLASSES_DICT[model_name]
    except ValueError:
        raise ValueError("Model {} was not found!".format(model_name))
    print("Loading model as {}".format(model_class))

    if shape is None:
        shape = (
            _DEFAULT_IMAGE_SIZE[model_name],
            _DEFAULT_IMAGE_SIZE[model_name],
            _NUM_CHANNELS,
        )

    input_img = tf.keras.layers.Input(shape=shape, name="input_1")
    model = model_class(
        include_top=True,
        weights="imagenet",
        input_tensor=input_img,
        input_shape=None,
        pooling=None,
        classes=_NUM_CLASSES,
        classifier_activation="softmax",
    )

    return model


def print_model_weights_shapes(model):
    """
    Print shape of each layer weight.
    Args:
        model: Keras model
    """
    print([model.get_weights()[i].shape for i in range(len(model.get_weights()))])


def ensure_dir(dirname):
    """
    Create directory is doesn't exist already.
    Args:
        dirname: Name of the directory to create.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
