#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A small resnet-like network for quick testing.
"""

import tensorflow as tf


def identity_block(input_tensor):
    """
    Identity block with no shortcut convolution
    """
    y = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same")(
        input_tensor
    )
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same")(y)
    out = tf.keras.layers.Add()([y, input_tensor])
    out = tf.keras.layers.ReLU()(out)
    return out


def identity_block_short_conv(input_tensor):
    """
    Identity block with shortcut convolution
    """
    y = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same")(
        input_tensor
    )
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(
        filters=24, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(y)
    ds_input = tf.keras.layers.Conv2D(
        filters=24, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(input_tensor)
    out = tf.keras.layers.Add()([y, ds_input])
    out = tf.keras.layers.ReLU()(out)
    return out


def model():
    """
    Dummy network with resnet-like architecture.
    """
    input_img = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3))(input_img)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3))(x)
    x = tf.keras.layers.ReLU()(x)
    x = identity_block(x)
    x = identity_block_short_conv(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(input_img, x, name="Dummy_Model")


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)
