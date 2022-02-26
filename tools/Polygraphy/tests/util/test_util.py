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
import os
import tempfile
import random

import numpy as np
import pytest
from polygraphy import util


VOLUME_CASES = [
    ((1, 1, 1), 1),
    ((2, 3, 4), 24),
    (tuple(), 1),
]


@pytest.mark.parametrize("case", VOLUME_CASES)
def test_volume(case):
    it, vol = case
    assert util.volume(it) == vol


class FindInDictCase(object):
    def __init__(self, name, map, index, expected):
        self.name = name
        self.map = map
        self.index = index
        self.expected = expected


FIND_IN_DICT_CASES = [
    FindInDictCase(
        "resnet50_v1.5/output/Softmax:0",
        map={"resnet50_v1.5/output/Softmax:0": "x"},
        index=None,
        expected="resnet50_v1.5/output/Softmax:0",
    ),
    FindInDictCase(
        "resnet50_v1.5/output/Softmax:0",
        map={"resnet50_v1.5/output/softmax:0": "x"},
        index=None,
        expected="resnet50_v1.5/output/softmax:0",
    ),
]


@pytest.mark.parametrize("case", FIND_IN_DICT_CASES)
def test_find_in_dict(case):
    actual = util.find_in_dict(case.name, case.map, case.index)
    assert actual == case.expected


SHAPE_OVERRIDE_CASES = [
    ((1, 3, 224, 224), (None, 3, 224, 224), True),
]


@pytest.mark.parametrize("case", SHAPE_OVERRIDE_CASES)
def test_is_valid_shape_override(case):
    override, shape, expected = case
    assert util.is_valid_shape_override(new_shape=override, original_shape=shape) == expected


def arange(shape):
    return np.arange(util.volume(shape)).reshape(shape)


SHAPE_MATCHING_CASES = [
    (arange((1, 1, 3, 3)), (3, 3), arange((3, 3))),  # Squeeze array shape
    (
        arange((1, 3, 3, 1)),
        (1, 1, 3, 3),
        arange((1, 1, 3, 3)),
    ),  # Permutation should make no difference as other dimensions are 1s
    (arange((3, 3)), (1, 1, 3, 3), arange((1, 1, 3, 3))),  # Unsqueeze where needed
    (arange((3, 3)), (-1, 3), arange((3, 3))),  # Infer dynamic
    (arange((3 * 2 * 2,)), (None, 3, 2, 2), arange((1, 3, 2, 2))),  # Reshape with inferred dimension
    (arange((1, 3, 2, 2)), (None, 2, 2, 3), np.transpose(arange((1, 3, 2, 2)), [0, 2, 3, 1])),  # Permute
]


@pytest.mark.parametrize("arr, shape, expected", SHAPE_MATCHING_CASES)
def test_shape_matching(arr, shape, expected):
    arr = util.try_match_shape(arr, shape)
    assert np.array_equal(arr, expected)


UNPACK_ARGS_CASES = [
    ((0, 1, 2), 3, (0, 1, 2)),  # no extras
    ((0, 1, 2), 4, (0, 1, 2, None)),  # 1 extra
    ((0, 1, 2), 2, (0, 1)),  # 1 fewer
]


@pytest.mark.parametrize("case", UNPACK_ARGS_CASES)
def test_unpack_args(case):
    args, num, expected = case
    assert util.unpack_args(args, num) == expected


UNIQUE_LIST_CASES = [
    ([], []),
    ([3, 1, 2], [3, 1, 2]),
    ([1, 2, 3, 2, 1], [1, 2, 3]),
    ([0, 0, 0, 0, 1, 0, 0], [0, 1]),
    ([5, 5, 5, 5, 5], [5]),
]


@pytest.mark.parametrize("case", UNIQUE_LIST_CASES)
def test_unique_list(case):
    lst, expected = case
    assert util.unique_list(lst) == expected


def test_find_in_dirs():
    with tempfile.TemporaryDirectory() as topdir:
        dirs = list(map(lambda x: os.path.join(topdir, x), ["test0", "test1", "test2", "test3", "test4"]))
        for subdir in dirs:
            os.makedirs(subdir)

        path_dir = random.choice(dirs)
        path = os.path.join(path_dir, "cudart64_11.dll")

        with open(path, "w") as f:
            f.write("This file should be found by find_in_dirs")

        assert util.find_in_dirs("cudart64_*.dll", dirs) == [path]


@pytest.mark.parametrize(
    "val,key,default,expected",
    [
        (1.0, None, None, 1.0),  # Basic
        ({"inp": "hi"}, "inp", "", "hi"),  # Per-key
        ({"inp": "hi"}, "out", "default", "default"),  # Per-key missing
        ({"inp": 1.0, "": 2.0}, "out", 1.5, 2.0),  # Per-key with default
    ],
)
def test_value_or_from_dict(val, key, default, expected):
    actual = util.value_or_from_dict(val, key, default)
    assert actual == expected
