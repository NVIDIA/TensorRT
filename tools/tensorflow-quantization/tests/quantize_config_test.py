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
from tensorflow_quantization import quantize_config
import tensorflow_quantization.global_config as global_config
from tensorflow_quantization import QuantizationSpec
from network_pool import bilbo_28_28


def test_global_object_creation():
    fnq = quantize_config.FullNetworkQuantization()
    assert (
            len(global_config.G_CONFIG_OBJECT) == 1
    ), "quantization config class object is not added to the global list"
    assert isinstance(
        global_config.G_CONFIG_OBJECT[0], quantize_config.FullNetworkQuantization
    )
    fnq.clean()
    tf.keras.backend.clear_session()


def test_quantization_config_layer_names_add():
    model = bilbo_28_28()
    fnq = quantize_config.FullNetworkQuantization()
    qspec = QuantizationSpec()
    qspec.add(name="conv_0")
    qspec.add(name="conv_2")
    qspec.add(name="conv_4")
    fnq.add_quantization_spec_object(qspec, model.layers)
    assert (
        "conv_0" in fnq.layerwise_config
    ), "There seems to be an issue with layer name addition in `add_special_layers` function"
    assert (
        "conv_2" in fnq.layerwise_config
    ), "There seems to be an issue with layer name addition in `add_special_layers` function"
    assert (
        "conv_4" in fnq.layerwise_config
    ), "There seems to be an issue with layer name addition in `add_special_layers` function"
    fnq.clean()
    tf.keras.backend.clear_session()


def test_quantization_config_layer_class_add():
    model = bilbo_28_28()
    fnq = quantize_config.FullNetworkQuantization()
    qspec = QuantizationSpec()
    qspec.add(name="Dense", is_keras_class=True)
    fnq.add_quantization_spec_object(qspec, model.layers)
    assert (
        "Dense" in fnq.layer_classes_to_quantize
    ), "There seems to be an issue with layer class name addition in `add_special_layers` function"
    fnq.clean()
    tf.keras.backend.clear_session()
