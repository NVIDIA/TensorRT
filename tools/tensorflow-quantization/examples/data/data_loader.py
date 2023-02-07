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
# Copyright 2018 & 2016 The TensorFlow Authors. All Rights Reserved.
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

"""
Changes made by NVIDIA (2022):
    - Added: load_data_tfrecord_tf() and load_data() functions
    - Modified preprocess_image_record(): preprocess_image() + tfrecord data deserialization + decode jpeg
    - Updated global constants with supported models: _DEFAULT_IMAGE_SIZE and _RESIZE_MIN

About this file: Standalone script for ImageNet TFRecord data loading and input image pre-processing for supported
    models. Follows TensorFlow's codebase data_loading + pre-processing workflow.

Important links:
    - TF's codebase:
    https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/imagenet_preprocessing.py
    - Deserialize tfrecord:
    https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/tf_example_decoder.py
"""

import os
import tensorflow as tf
import PIL.Image
import numpy as np
from typing import Dict, Union

_SUPPORTED_MODEL_NAMES = [
    "resnet_v1",
    "resnet_v2",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenet_v1",
    "mobilenet_v2",
    "inception_v3",
]

_NUM_CLASSES = 1000
_NUM_IMAGES = {
    "train": 1281167,
    "validation": 50000,
}

_DEFAULT_IMAGE_SIZE = {
    "resnet_v1": 224,
    "resnet_v2": 299,
    "efficientnet_b0": 224,
    "efficientnet_b3": 300,
    "mobilenet_v1": 224,
    "mobilenet_v2": 224,
    "inception_v3": 299,
}
_NUM_CHANNELS = 3
_RESIZE_MIN = {
    "resnet_v1": 256,
    "resnet_v2": 342,
    "efficientnet_b0": 256,
    "efficientnet_b3": 342,
    "mobilenet_v1": 256,
    "mobilenet_v2": 256,
    "inception_v3": 342,
}


def load_image_np(test_image, model_name: str = "resnet_v1"):
    # Image is loaded in NHWC format
    image_np = np.asarray(PIL.Image.open(test_image).convert('RGB'))
    image = tf.constant(image_np)
    image = _aspect_preserving_resize(image, _RESIZE_MIN[model_name])
    image = _central_crop(image, _DEFAULT_IMAGE_SIZE[model_name], _DEFAULT_IMAGE_SIZE[model_name])
    image = preprocess_model_func(image, model_name)
    return image


def get_filenames(
    data_dir: str,
    is_training: bool = False,
    num_train_files: int = 1024,
    num_val_files: int = 128,
):
    """
    Returns filenames for dataset.

    Args:
        data_dir (str): directory where data is stored.
        is_training (bool): indicates whether to return the 'train' (True) or 'validation' (False) data filenames.
        num_train_files (int): number of tfrecord shards available for training.
        num_val_files (int): number of tfrecord shards available for validation.

    Returns:
        List: list of shards filenames to compose the dataset.
    """
    if is_training:
        return [
            # Example: train-00000-of-01024
            os.path.join(data_dir, "train-{:05d}-of-{:05d}".format(i, num_train_files))
            for i in range(num_train_files)
        ]
    else:
        return [
            os.path.join(
                data_dir, "validation-{:05d}-of-{:05d}".format(i, num_val_files)
            )
            for i in range(num_val_files)
        ]


def _deserialize_image_record(record):
    feature_map = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string, ""),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64, -1),
        "image/class/text": tf.io.FixedLenFeature([], tf.string, ""),
        "image/object/bbox/xmin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(dtype=tf.float32),
    }
    with tf.name_scope("deserialize_image_record"):
        obj = tf.io.parse_single_example(record, feature_map)
        imgdata = obj["image/encoded"]
        label = tf.cast(obj["image/class/label"], tf.int32)
        bbox = tf.stack(
            [
                obj["image/object/bbox/%s" % x].values
                for x in ["ymin", "xmin", "ymax", "xmax"]
            ]
        )
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
        text = obj["image/class/text"]

        return imgdata, label, bbox, text


def _aspect_preserving_resize(image: tf.Tensor, resize_min: Union[int, tf.Tensor]):
    """Resize images preserving the original aspect ratio.

    Args:
        image (tf.Tensor): A 3-D image `Tensor`.
        resize_min (int): A python integer or scalar `Tensor` indicating the size of the smallest side after resize.

    Returns:
        resized_image (tf.Tensor): A 3-D `Tensor` containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    resized_image = tf.image.resize(
        image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR
    )

    return resized_image


def _smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = height - crop_height
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = width - crop_width
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def preprocess_image_record(record, min_size=256, image_height=224, image_width=224):
    """
    This function performs image cropping so all images in the dataset have the same height and width dimensions.
        No value pre-processing is done here.
    """

    imgdata, label, _, _ = _deserialize_image_record(record)
    # Subtract one so that ImageNet labels are in [0, 1000). This assumes your dataset contains 'background' as 0.
    label -= 1

    try:
        image = tf.image.decode_jpeg(
            imgdata,
            channels=_NUM_CHANNELS,
            fancy_upscaling=False,
            dct_method="INTEGER_FAST",
        )
    except:
        image = tf.image.decode_image(imgdata, channels=_NUM_CHANNELS)

    image = tf.cast(image, tf.float32)
    image = _aspect_preserving_resize(image, min_size)
    image = _central_crop(image, image_height, image_width)

    return image, label


def preprocess_model_func(image: tf.Tensor, model_name: str = "resnet_v1"):

    if model_name == "resnet_v1":
        return tf.keras.applications.resnet.preprocess_input(image)
    elif model_name == "resnet_v2":
        return tf.keras.applications.resnet_v2.preprocess_input(image)
    elif model_name == "mobilenet_v1":
        return tf.keras.applications.mobilenet.preprocess_input(image)
    elif model_name == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)
    elif model_name == "inception_v3":
        return tf.keras.applications.inception_v3.preprocess_input(image)
    else:
        # efficientnet doesn't need specific pre-processing (included in the model itself).
        print("No further pre-processing found for {}".format(model_name))

    return image


def load_data_tfrecord_tf(
    data_dir: str = "./data/imagenet",
    batch_size: int = 8,
    num_train_files: int = 1024,
    num_val_files: int = 128,
    model_name: str = "resnet_v1",
) -> Dict[str, tf.data.Dataset]:
    """
    Load ImageNet with TensorFlow Datasets (TFDS).

    Args:
        data_dir (str): directory where data is stored.
        batch_size (int): batch_size for dataloader.
        num_train_files (int): number of tfrecord shards available for training.
        num_val_files (int): number of tfrecord shards available for validation.
        model_name (str): Model name, used to decide which input pre-processing is needed.
            Options={supported_model_names}.

    Returns:
        dataset_dict (Dict[str, tf.data.Dataset]): dictionary with 'train' and 'validation' datasets.

    Raises:
        ValueError: raised if 'model_name' is not supported.
    """.format(
        supported_model_names=_SUPPORTED_MODEL_NAMES
    )

    # 1. Load ImageNet2012 train dataset - needs to manually download the full ImageNet2012 dataset first.
    assert os.path.exists(data_dir)
    if model_name not in _SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "Invalid model name ",
            model_name,
            " provided. Please select among {}".format(_SUPPORTED_MODEL_NAMES),
        )

    # 2. Make train/validation datasets
    dataset_dict = {}
    for key, is_training in zip(["train", "validation"], [True, False]):
        filenames = get_filenames(
            data_dir,
            is_training=is_training,
            num_train_files=num_train_files,
            num_val_files=num_val_files,
        )
        dataset = tf.data.TFRecordDataset(filenames)

        # Image cropping and resizing
        if model_name in _DEFAULT_IMAGE_SIZE and model_name in _RESIZE_MIN:
            dataset = dataset.map(
                lambda record: preprocess_image_record(
                    record,
                    min_size=_RESIZE_MIN[model_name],
                    image_height=_DEFAULT_IMAGE_SIZE[model_name],
                    image_width=_DEFAULT_IMAGE_SIZE[model_name],
                )
            )
        else:
            dataset = dataset.map(preprocess_image_record)

        dataset = dataset.map(lambda image, label: (preprocess_model_func(image, model_name), label))
        # Divide dataset into batches
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset_dict[key] = dataset

    return dataset_dict


def load_data(
    hyperparams: Dict, model_name: str = "resnet_v1"
) -> [tf.data.Dataset, tf.data.Dataset]:
    """ Loads ImageNet data in `tfrecord` format (requires manual data download).

    Args:
        hyperparams (Dict): dictionary with necessary hyper-parameters for data loading.
        model_name (str): Model name, used to decide which input pre-processing is needed.
            Options={supported_model_names}.

    Returns:
        train_batches (tf.data.Dataset): 'train' dataset.
        val_batches (tf.data.Dataset): 'validation' dataset.
    """.format(
        supported_model_names=_SUPPORTED_MODEL_NAMES
    )

    data_batches = load_data_tfrecord_tf(
        data_dir=hyperparams["tfrecord_data_dir"],
        batch_size=hyperparams["batch_size"],
        model_name=model_name,
    )
    train_batches, val_batches = (data_batches["train"], data_batches["validation"])
    if hyperparams["train_data_size"] is not None:
        train_batches = train_batches.take(hyperparams["train_data_size"])
    if hyperparams["val_data_size"] is not None:
        val_batches = val_batches.take(hyperparams["val_data_size"])

    return train_batches, val_batches
