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
This module contains test cases for our data loader, which contains data loading and pre-processing functions
    for the ImageNet2012 dataset in 'tfrecord' format.
NOTE: the user needs to manually download the full ImageNet2012 dataset first.
"""

import tensorflow as tf
from examples.data.data_loader import load_data
import numpy as np
from collections import defaultdict
from typing import Dict
import pytest

DATA_HYPERPARAMS = {
    "tfrecord_data_dir": "/media/Data/ImageNet/train-val-tfrecord",
    "batch_size": 64,
    "train_data_size": 100,  # If 'None', consider all data, otherwise, consider subset.
    "val_data_size": 100,  # If 'None', consider all data, otherwise, consider subset.
}


def load_tfrecord_mean_min_max(model_name: str) -> [Dict, Dict, Dict]:
    """
    Loads `tfrecord` dataset and calculates the data's mean, min, and max values.

    Args:
        model_name (str): model name for data pre-processing.

    Returns:
        total_mean_dict: dictionary with MEAN values in 'R', 'G', 'B'.
        total_min_dict: dictionary with MIN values in 'R', 'G', 'B'.
        total_max_dict: dictionary with MAX values in 'R', 'G', 'B'.
    """
    # 1. Data loading
    train_batches, val_batches = load_data(
        hyperparams=DATA_HYPERPARAMS, model_name=model_name
    )
    assert isinstance(train_batches, tf.data.Dataset) and isinstance(
        val_batches, tf.data.Dataset
    )

    # 2. Test input preprocessing
    mean = defaultdict(list)
    min = defaultdict(list)
    max = defaultdict(list)
    for batch in [train_batches]:
        for examples in batch:
            image, label = examples
            image_dict = defaultdict()
            for i, c in zip([0, 1, 2], ["R", "G", "B"]):
                image_dict[c] = image[:, :, :, i]
                mean[c].append(tf.math.reduce_mean(image_dict[c]))
                min[c].append(tf.math.reduce_min(image_dict[c]))
                max[c].append(tf.math.reduce_max(image_dict[c]))

    total_mean_dict = defaultdict(list)
    total_min_dict = defaultdict(list)
    total_max_dict = defaultdict(list)
    for c in ["R", "G", "B"]:
        total_mean_dict[c] = tf.math.reduce_mean(mean[c])
        total_min_dict[c] = tf.math.reduce_min(min[c])
        total_max_dict[c] = tf.math.reduce_max(max[c])
    return total_mean_dict, total_min_dict, total_max_dict


def test_imagenet_tfrecord_efficientnetb0():
    """
    Tests data loading and pre-processing for EfficientNet-B0.
    Note that EfficientNet doesn't have any input pre-processing methods besides image resizing and cropping.
      See `data_loader.preprocess_image_record()`.
    """
    print("------------ EfficientNet-B0 pre-processing test -------------")

    # 1. Data loading and get mean, max, min of data without input preprocessing
    total_mean_dict, total_min_dict, total_max_dict = load_tfrecord_mean_min_max(
        model_name="efficientnet_b0"
    )

    # 2. Check if mean is as expected and max/min values
    mean_RGB = [123.68, 116.779, 103.939]
    mean_RGB_obtained = list(
        total_mean_dict.values()
    )  # [total_mean_dict['R'], total_mean_dict['G'], total_mean_dict['B']]
    mean_diff = abs(np.array(mean_RGB_obtained) - np.array(mean_RGB))
    print("  Expected mean (RGB): {}".format(mean_RGB))
    print("  Calculated mean (RGB): {}".format(np.array(mean_RGB_obtained)))
    print("  Difference: {}".format(np.array(mean_diff)))
    assert (mean_diff <= 3.0).all()

    # 3. Values expected to be between 0 and 255
    total_min = min(total_min_dict["R"], min(total_min_dict["G"], total_min_dict["B"]))
    total_max = max(total_max_dict["R"], max(total_max_dict["G"], total_max_dict["B"]))
    assert total_min >= 0.0 and total_max <= 255.0


def test_imagenet_tfrecord_resnetv1():
    """Tests data loading and pre-processing for ResNetv1.

    ResNetv1 input pre-processing:
        - Resizing + cropping
        - "The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the
        ImageNet dataset, without scaling."
        - Zero-center: (data - mean(data) / std(data)) -> In this case, std(data) = None.
    """
    print("------------ ResNetv1 pre-processing test -------------")

    # 1. Data loading and get mean, max, min of data without input preprocessing
    total_mean_dict, total_min_dict, total_max_dict = load_tfrecord_mean_min_max(
        model_name="resnet_v1"
    )

    # 2.1 Data should be zero-centered (mean=0)
    mean_RGB_obtained = list(total_mean_dict.values())
    print("  Expected mean (RGB): [0, 0, 0]")
    print("  Calculated mean (RGB): {}".format(np.array(mean_RGB_obtained)))
    print("  Min values: {}".format(np.array(list(total_min_dict.values()))))
    print("  Max values: {}".format(np.array(list(total_max_dict.values()))))
    assert (abs(np.array(mean_RGB_obtained)) <= 3.0).all()

    # 2.2 No scaling, meaning values are between -255 and 255 (after zero-centering)
    assert (np.array(list(total_min_dict.values())) >= -255.0).all()
    assert (np.array(list(total_max_dict.values())) <= 255.0).all()


def test_imagenet_tfrecord_resnetv2():
    """Tests data loading and pre-processing for ResNetv2.

    ResNetv2 input pre-processing:
        - Resizing + cropping
        - "The inputs pixel values are scaled between -1 and 1, sample-wise."
        - Sample-wise normalization: https://stackoverflow.com/questions/37625272/keras-batchnormalization-what-exactly-is-sample-wise-normalization

    MobileNet-v1/v2 input pre-processing:
        - "The inputs pixel values are scaled between -1 and 1, sample-wise."
        - This is the same as ResNet-v2, with the difference that the MobileNet model takes input shape 224x224x3
          (same as ResNet-v1).
    """
    print("------------ ResNetv2 pre-processing test -------------")

    # 1. Data loading and get mean, max, min of data without input preprocessing
    total_mean_dict, total_min_dict, total_max_dict = load_tfrecord_mean_min_max(
        model_name="resnet_v2"
    )

    # 2. Check that values are between -1 and 1
    print("  Min values: {}".format(np.array(list(total_min_dict.values()))))
    print("  Max values: {}".format(np.array(list(total_max_dict.values()))))
    print("  Mean values: {}".format(np.array(list(total_mean_dict.values()))))
    assert (np.array(list(total_min_dict.values())) >= -1.0).all()
    assert (np.array(list(total_max_dict.values())) <= 1.0).all()
