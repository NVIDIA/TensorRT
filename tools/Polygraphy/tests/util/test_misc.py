#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from polygraphy.util import misc

import numpy as np
import pytest


VOLUME_CASES = [
    ((1, 1, 1), 1),
    ((2, 3, 4), 24),
    (tuple(), 1),
]

@pytest.mark.parametrize("case", VOLUME_CASES)
def test_volume(case):
    it, vol = case
    assert misc.volume(it) == vol


class FindInDictCase(object):
    def __init__(self, name, map, index, expected):
        self.name = name
        self.map = map
        self.index = index
        self.expected = expected

FIND_IN_DICT_CASES = [
    FindInDictCase("resnet50_v1.5/output/Softmax:0", map={"resnet50_v1.5/output/Softmax:0": "x"}, index=None, expected="resnet50_v1.5/output/Softmax:0"),
    FindInDictCase("resnet50_v1.5/output/Softmax:0", map={"resnet50_v1.5/output/softmax:0": "x"}, index=None, expected="resnet50_v1.5/output/softmax:0"),
]

@pytest.mark.parametrize("case", FIND_IN_DICT_CASES)
def test_find_in_dict(case):
    actual = misc.find_in_dict(case.name, case.map, case.index)
    assert actual == case.expected


SHAPE_OVERRIDE_CASES = [
    ((1, 3, 224, 224), (None, 3, 224, 224), True),
]

@pytest.mark.parametrize("case", SHAPE_OVERRIDE_CASES)
def test_is_valid_shape_override(case):
    override, shape, expected = case
    assert misc.is_valid_shape_override(new_shape=override, original_shape=shape) == expected


SHAPE_MATCHING_CASES = [
    (np.zeros((1, 1, 3, 3)), (3, 3), (3, 3)), # Squeeze array shape
    (np.zeros((1, 3, 3, 1)), (1, 1, 3, 3), (1, 1, 3, 3)), # Permute
    (np.zeros((3, 3)), (1, 1, 3, 3), (3, 3)), # Squeeze specified shape
    (np.zeros((3, 3)), (-1, 3), (3, 3)), # Infer dynamic
    (np.zeros((3 * 224 * 224)), (None, 3, 224, 224), (1, 3, 224, 224)), # Reshape and Permute
    (np.zeros((1, 3, 224, 224)), (None, 224, 224, 3), (1, 224, 224, 3)), # Permute
]

@pytest.mark.parametrize("case", SHAPE_MATCHING_CASES)
def test_shape_matching(case):
    out, shape, expected_shape = case
    out = misc.try_match_shape(out, shape)
    assert out.shape == expected_shape



UNPACK_ARGS_CASES = [
    ((0, 1, 2), 3, (0, 1, 2)), # no extras
    ((0, 1, 2), 4, (0, 1, 2, None)), # 1 extra
    ((0, 1, 2), 2, (0, 1)), # 1 fewer
]

@pytest.mark.parametrize("case", UNPACK_ARGS_CASES)
def test_unpack_args(case):
    args, num, expected = case
    assert misc.unpack_args(args, num) == expected


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
    assert misc.unique_list(lst) == expected
