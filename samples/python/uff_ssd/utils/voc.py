#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# VOC dataset utility functions
import numpy as np


VOC_CLASSES_LIST = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

VOC_CLASSES_SET = set(VOC_CLASSES_LIST)

VOC_CLASS_ID = {
    cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES_LIST)
}

# Random RGB colors for each class (useful for drawing bounding boxes)
VOC_COLORS = \
    np.random.uniform(0, 255, size=(len(VOC_CLASSES_LIST), 3)).astype(np.uint8)


def convert_coco_to_voc(label):
    """Converts COCO class name to VOC class name, if possible.

    COCO classes are a superset of VOC classes, but
    some classes have different names (e.g. airplane
    in COCO is aeroplane in VOC). This function gets
    COCO label and converts it to VOC label,
    if conversion is needed.

    Args:
        label (str): COCO label
    Returns:
        str: VOC label corresponding to given label if such exists,
            otherwise returns original label
    """
    COCO_VOC_DICT = {
        'airplane': 'aeroplane',
        'motorcycle': 'motorbike',
        'dining table': 'diningtable',
        'potted plant': 'pottedplant',
        'couch': 'sofa',
        'tv': 'tvmonitor'
    }
    if label in COCO_VOC_DICT:
        return COCO_VOC_DICT[label]
    else:
        return label

def coco_label_to_voc_label(label):
    """Returns VOC label corresponding to given COCO label.

    COCO classes are superset of VOC classes, this function
    returns label corresponding to given COCO class label
    or None if such label doesn't exist.

    Args:
        label (str): COCO class label
    Returns:
        str: VOC label corresponding to given label or None
    """
    label = convert_coco_to_voc(label)
    if label in VOC_CLASSES_SET:
        return label
    else:
        return None

def is_voc_label(label):
    """Returns boolean which tells if given label is VOC label.

    Args:
        label (str): object label
    Returns:
        bool: is given label a VOC class label
    """
    return label in VOC_CLASSES_SET

def get_voc_label_color(label):
    """Returns color corresponding to given VOC label, or None.

    Args:
        label (str): object label
    Returns:
        np.array: RGB color described in 3-element np.array
    """
    if not is_voc_label(label):
        return None
    else:
        return VOC_COLORS[VOC_CLASS_ID[label]]
