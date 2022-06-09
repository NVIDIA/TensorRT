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


"""
This module contains tiny networks used for testing across different modules.
They are named after famous Hobbits for obvious reasons.
"""

import tensorflow as tf

##################################################
###### Tiny, VGG like network ####################
##################################################
def bilbo_28_28():
    """
    Network with VGG like architecture.
    """
    input_img = tf.keras.layers.Input(shape=(28, 28), name="nn_input")
    x = tf.keras.layers.Reshape(target_shape=(28, 28, 1), name="reshape_0")(input_img)
    x = tf.keras.layers.Conv2D(filters=516, kernel_size=(3, 3), name="conv_0")(x)
    x = tf.keras.layers.ReLU(name="relu_0")(x)
    x = tf.keras.layers.Conv2D(filters=252, kernel_size=(3, 3), name="conv_1")(x)
    x = tf.keras.layers.ReLU(name="relu_1")(x)
    x = tf.keras.layers.Conv2D(filters=126, kernel_size=(3, 3), name="conv_2")(x)
    x = tf.keras.layers.ReLU(name="relu_2")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), name="conv_3")(x)
    x = tf.keras.layers.ReLU(name="relu_3")(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), name="conv_4")(x)
    x = tf.keras.layers.ReLU(name="relu_4")(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), name="conv_5")(x)
    x = tf.keras.layers.ReLU(name="relu_5")(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), name="conv_6")(x)
    x = tf.keras.layers.ReLU(name="relu_6")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_0")(x)
    x = tf.keras.layers.Flatten(name="flatten_0")(x)
    x = tf.keras.layers.Dense(100, name="dense_0")(x)
    x = tf.keras.layers.ReLU(name="relu_7")(x)
    x = tf.keras.layers.Dense(10, name="dense_1")(x)
    return tf.keras.Model(input_img, x, name="Bilbo")


#####################################################
###### Tiny, ResNet like network ####################
#####################################################
def identity_block_plain(input_tensor):
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


def identity_block_short_conv_plain(input_tensor):
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


def frodo_32_32():
    """
    Dummy network with resnet like architecture.
    """
    input_img = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3))(input_img)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = identity_block_plain(x)
    x = identity_block_short_conv_plain(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(input_img, x, name="Frodo")


def sam_32_32():
    """
    Dummy network with resnet like architecture.
    """
    input_img = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3))(input_img)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3))(x)
    x = tf.keras.layers.ReLU()(x)
    x = identity_block_plain(x)
    x = identity_block_plain(x)
    x = identity_block_short_conv_plain(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(input_img, x, name="Sam")


##############################################
###### Popular network blocks ################
##############################################
def relu_bn(input):
    """
    Block with BN+ReLU
    """
    bn = tf.keras.layers.BatchNormalization()(input)
    relu = tf.keras.layers.ReLU()(bn)
    return relu


def bn(input):
    return tf.keras.layers.BatchNormalization()(input)


def relu(input):
    return tf.keras.layers.ReLU()(input)


def inception_block(input_tensor):
    """
    Inception block from GoogleNet
    """
    b1x1 = tf.keras.layers.Conv2D(filters=12, kernel_size=(1, 1), padding="same")(
        input_tensor
    )
    b1x1 = relu_bn(b1x1)

    b5x5 = tf.keras.layers.Conv2D(filters=12, kernel_size=(1, 1), padding="same")(
        input_tensor
    )
    b5x5 = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), padding="same")(b5x5)
    b5x5 = relu_bn(b5x5)

    b3x3 = tf.keras.layers.Conv2D(filters=12, kernel_size=(1, 1), padding="same")(
        input_tensor
    )
    b3x3 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), padding="same")(b3x3)
    b3x3 = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same")(b3x3)
    b3x3 = relu_bn(b3x3)

    out = tf.keras.layers.Concatenate()([b1x1, b5x5])
    return out


def identity_block_bn(input_tensor):
    """
    Identity block with no shortcut convolution
    """
    y = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same")(
        input_tensor
    )
    y = relu_bn(y)
    y = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same")(y)
    y = bn(y)
    out = tf.keras.layers.Add()([y, input_tensor])
    out = relu(out)
    return out


def identity_block_short_conv_bn(input_tensor):
    """
    Identity block with shortcut convolution
    """
    y = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same")(
        input_tensor
    )
    y = relu_bn(y)
    y = tf.keras.layers.Conv2D(
        filters=24, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(y)
    y = bn(y)
    ds_input = tf.keras.layers.Conv2D(
        filters=24, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(input_tensor)
    ds_input = bn(ds_input)
    out = tf.keras.layers.Add()([y, ds_input])
    out = relu(out)
    return out


def otho_28_28():
    input_img = tf.keras.layers.Input(shape=(28, 28), name="input_0")
    r = tf.keras.layers.Reshape(target_shape=(28, 28, 1), name="reshape_0")(input_img)
    x = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), name="conv_0")(r)
    x = tf.keras.layers.ReLU(name="relu_0")(x)
    x = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), name="conv_1")(x)
    x = tf.keras.layers.ReLU(name="relu_1")(x)
    x = tf.keras.layers.Flatten(name="flatten_0")(x)
    return tf.keras.Model(input_img, x, name="Otho")


def lotho_28_28():
    input_img = tf.keras.layers.Input(shape=(28, 28), name="input_0")
    r = tf.keras.layers.Reshape(target_shape=(28, 28, 1), name="reshape_0")(input_img)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), name="dconv_0")(r)
    x = tf.keras.layers.ReLU(name="relu_0")(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), name="dconv_1")(x)
    x = tf.keras.layers.ReLU(name="relu_1")(x)
    x = tf.keras.layers.Flatten(name="flatten_0")(x)
    return tf.keras.Model(input_img, x, name="Lotho")


def lobelia_28_28():
    input_img = tf.keras.layers.Input(shape=(28, 28), name="input_0")
    r = tf.keras.layers.Reshape(target_shape=(28, 28, 1), name="reshape_0")(input_img)
    x = tf.keras.layers.Flatten(name="flatten_0")(r)
    x = tf.keras.layers.Dense(100, name="dense_0")(x)
    x = tf.keras.layers.ReLU(name="relu_0")(x)
    x = tf.keras.layers.Dense(10, name="dense_1")(x)
    return tf.keras.Model(input_img, x, name="Lobelia")


def merry_28_28():
    input_img = tf.keras.layers.Input(shape=(28, 28))
    x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_img)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3))(x)
    x = relu_bn(x)
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3))(x)
    x = relu_bn(x)
    x = inception_block(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(input_img, x, name="Merry")


def pippin_28_28():
    input_img = tf.keras.layers.Input(shape=(28, 28))
    x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_img)
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3))(x)
    x = relu_bn(x)
    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3))(x)
    x = relu_bn(x)
    x = identity_block_bn(x)
    x = identity_block_short_conv_bn(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(input_img, x, name="Pippin")
