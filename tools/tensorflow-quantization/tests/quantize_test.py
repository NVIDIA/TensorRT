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
This module contains test cases for `quantize_model` feature.
`quantize_model` feature quantizes all supported layers in the given Keras model with `NVIDIA` quantization scheme.

Tests if weights were copied correctly after quantization and end-to-end training accuracy.
"""

import tensorflow as tf
from tensorflow_quantization import quantize
from tensorflow_quantization import quantize_model
from network_pool import lobelia_28_28
from network_pool import bilbo_28_28
import pytest
import tensorflow_quantization
from tensorflow_quantization.utils import (
    CreateAssetsFolders,
    convert_saved_model_to_onnx,
)


def _print_model_weights_shapes(model):
    """
    Print shapes of all weights
    Args:
        model: Keras model
    """
    print([model.get_weights()[i].shape for i in range(len(model.get_weights()))])


def test_clone_numerics_quantize_whole_model(debug=False):
    """
    Checks whether weights are copied correctly when a dummy model is quantized.
    """
    model = lobelia_28_28()
    if debug:
        _print_model_weights_shapes(model)
    om_l0_test_weights = model.get_weights()[0][10, :5]
    om_l1_test_weights = model.get_weights()[2][10, :5]

    # Quantize model
    q_model = quantize_model(model)

    if debug:
        _print_model_weights_shapes(q_model)
    qm_l0_test_weights = q_model.get_weights()[1][10, :5]
    qm_l1_test_weights = q_model.get_weights()[8][10, :5]
    assert all([a == b for a, b in zip(om_l0_test_weights, qm_l0_test_weights)])
    assert all([a == b for a, b in zip(om_l1_test_weights, qm_l1_test_weights)])
    tf.keras.backend.clear_session()


def test_adding_one_layer_at_a_time():
    qspec = quantize.QuantizationSpec()
    qspec.add(name="conv2d_1")
    qspec.add(name="Dense", is_keras_class=True)

    assert isinstance(
        qspec.layers[0], quantize.LayerConfig
    ), "LayerConfig object is not created for newly added layer."
    assert (
        len(qspec.layers) == 2
    ), "New layers are not added to layer list of QuantizationSpec."


def test_adding_layer_name_list():
    qspec = quantize.QuantizationSpec()
    layer_name = ["conv2d", "conv2d_1", "conv2d_7", "dense"]
    layer_qip = [True, False, True, False]
    layer_idx = [None, [0], None, None]
    qspec.add(name=layer_name, quantize_input=layer_qip, quantization_index=layer_idx)
    assert (
        len(qspec.layers) == 4
    ), "Four layers are not added to qspec object as expected."


def train_quantize_fine_tune(exp_folder: "Folder", perform_four_bit_quantization: bool = False) -> None:
    """
    Train, quantize and fine-tune Keras model using NVIDIA's QAT wrapper library.

    Args:
        exp_folder (Folder): Base experiment folder object.
        perform_four_bit_quantization (bool): If True, 4 bit quantization is performed. 8 bit quantization is default.

    Returns:
        None
    """

    # Load MNIST dataset
    mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    nn_model_original = bilbo_28_28()
    # Train original classification model
    nn_model_original.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    nn_model_original.fit(
        train_images, train_labels, batch_size=128, epochs=5, validation_split=0.1
    )

    # get baseline model accuracy
    _, baseline_model_accuracy = nn_model_original.evaluate(
        test_images, test_labels, verbose=0
    )
    print("Baseline test accuracy:", baseline_model_accuracy)

    tf.keras.models.save_model(nn_model_original, exp_folder.fp32_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=exp_folder.fp32_saved_model,
        onnx_model_path=exp_folder.fp32_onnx_model,
    )

    if perform_four_bit_quantization:
        tensorflow_quantization.G_NUM_BITS = 4
    
    # quantize entire model using `quantize_model` feature
    q_model = quantize_model(nn_model_original)

    # fine tune annotated model
    q_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    q_model.fit(
        train_images, train_labels, batch_size=32, epochs=5, validation_split=0.1
    )

    # Get quantized accuracy
    _, q_aware_model_accuracy = q_model.evaluate(test_images, test_labels, verbose=0)
    print("Quant test accuracy:", q_aware_model_accuracy)

    assert (
        q_aware_model_accuracy >= baseline_model_accuracy or
        abs(baseline_model_accuracy - q_aware_model_accuracy) * 100 <= 2.0
    ), "QAT accuracy is not acceptable: {:.2f} vs {:.2f} for baseline".format(
        q_aware_model_accuracy * 100, baseline_model_accuracy * 100
    )

    # save quantized model and convert to ONNX
    tf.keras.models.save_model(q_model, exp_folder.int8_saved_model)
    convert_saved_model_to_onnx(
        saved_model_dir=exp_folder.int8_saved_model,
        onnx_model_path=exp_folder.int8_onnx_model,
    )


def test_end_to_end_workflow():
    """
    Test end-to-end QAT workflow using the `quantize_model` function.
    The following steps are included:
        1. Create a dummy model (baseline)
        2. Train model on Fashion MNIST dataset
        3. Calculate baseline FP32 model accuracy
        4. Perform 4 bit (default) quantization and fine-tuning
        5. Convert QAT model to ONNX
    """

    test_assets = CreateAssetsFolders("test_quantize_end_to_end")
    test_assets.add_folder("test_end_to_end_workflow")

    train_quantize_fine_tune(test_assets.test_end_to_end_workflow)
    tf.keras.backend.clear_session()

@pytest.mark.skip(reason="Just used to test 4 bit quantization feature.")
def test_end_to_end_workflow_4bit():
    """
    Test end-to-end QAT workflow using the `quantize_model` function for 4 bit quantization.
    The following steps are included:
        1. Create a dummy model (baseline)
        2. Train model on Fashion MNIST dataset
        3. Calculate baseline FP32 model accuracy
        4. Perform 4 bit quantization and fine-tuning
        5. Convert QAT model to ONNX
    """

    test_assets = CreateAssetsFolders("test_quantize_end_to_end")
    test_assets.add_folder("test_end_to_end_workflow_4bit")

    train_quantize_fine_tune(test_assets.test_end_to_end_workflow_4bit, perform_four_bit_quantization=True)
    tf.keras.backend.clear_session()