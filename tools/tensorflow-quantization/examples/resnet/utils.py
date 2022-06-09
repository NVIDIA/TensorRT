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


import tensorflow as tf
from examples.data.data_loader import _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS
from examples.utils import get_tfkeras_model


def get_resnet_model(resnet_depth: str = "50", resnet_version: str = "v1") -> tf.keras.Model:
    """
    Creates a native tf.keras ResNet model.

    Args:
        resnet_depth (str): ResNet depth. Options=[50 (default), 101, 152].
        resnet_version (str): ResNet version. Options=[v1 (default), v2].

    Returns:
        model (tf.keras.Model): model corresponding to 'resnet_depth' and 'resnet_version'.
    """

    shape = (
        _DEFAULT_IMAGE_SIZE["resnet_{}".format(resnet_version)],
        _DEFAULT_IMAGE_SIZE["resnet_{}".format(resnet_version)],
        _NUM_CHANNELS,
    )

    model_name = "resnet_" + resnet_depth + resnet_version
    model = get_tfkeras_model(model_name=model_name, shape=shape)

    return model
