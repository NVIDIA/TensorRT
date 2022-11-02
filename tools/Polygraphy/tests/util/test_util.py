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
import os
import random
import tempfile
from multiprocessing import Process

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


class FindStrInIterableCase:
    def __init__(self, name, seq, index, expected):
        self.name = name
        self.seq = seq
        self.index = index
        self.expected = expected


FIND_STR_IN_ITERABLE_CASES = [
    # Case insensitve, plus function should return element from sequence, not name.
    FindStrInIterableCase("Softmax:0", seq=["Softmax:0"], index=None, expected="Softmax:0"),
    FindStrInIterableCase("Softmax:0", seq=["softmax:0"], index=None, expected="softmax:0"),
    # Exact matches should take priority
    FindStrInIterableCase("exact_name", seq=["exact_name_plus", "exact_name"], index=0, expected="exact_name"),
    # Index should come into play when no matches are found
    FindStrInIterableCase("non-existent", seq=["test", "test2"], index=1, expected="test2"),
]


@pytest.mark.parametrize("case", FIND_STR_IN_ITERABLE_CASES)
def test_find_str_in_iterable(case):
    actual = util.find_str_in_iterable(case.name, case.seq, case.index)
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


def test_atomic_open():
    def write_to_file(path, content):
        with util.LockFile(path):
            old_contents = util.load_file(path, mode="r")
            util.save_file(old_contents + content, path, mode="w")

    NUM_LINES = 10
    NUM_PROCESSES = 5

    outfile = util.NamedTemporaryFile()

    processes = [
        Process(target=write_to_file, args=(outfile.name, f"{proc} - writing line\n" * NUM_LINES))
        for proc in range(NUM_PROCESSES)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for process in processes:
        assert not process.is_alive()
        assert process.exitcode == 0

    # Since we write atomically, all processes should be able to write their
    # contents. Furthermore, the contents should be grouped by process.
    with open(outfile.name) as f:
        lines = list(f.readlines())
        assert len(lines) == NUM_LINES * NUM_PROCESSES

        for idx in range(NUM_PROCESSES):
            offset = idx * NUM_LINES
            expected_prefix = lines[offset].partition("-")[0].strip()
            assert all(line.startswith(expected_prefix) for line in lines[offset : offset + NUM_LINES])

    # Make sure the lock file is written to the correct path and not removed automatically.
    assert os.path.exists(outfile.name + ".lock")


def test_make_contiguous():
    arr = np.transpose(np.ones(shape=(5, 10), dtype=np.float32))
    assert not util.is_contiguous(arr)

    arr = util.make_contiguous(arr)
    assert util.is_contiguous(arr)


class TestMakeRepr:
    def test_basic(self):
        assert util.make_repr("Example", 1, x=2) == ("Example(1, x=2)", False, False)

    def test_default_args(self):
        assert util.make_repr("Example", None, None, x=2) == ("Example(None, None, x=2)", True, False)

    def test_empty_args_are_default(self):
        assert util.make_repr("Example", x=2) == ("Example(x=2)", True, False)

    def test_default_kwargs(self):
        assert util.make_repr("Example", 1, 2, x=None, y=None) == ("Example(1, 2)", False, True)

    def test_empty_kwargs_are_default(self):
        assert util.make_repr("Example", 1, 2) == ("Example(1, 2)", False, True)

    def test_does_not_modify(self):
        obj = {"x": float("inf")}
        assert util.make_repr("Example", obj) == ("Example({'x': float('inf')})", False, True)
        assert obj == {"x": float("inf")}

    @pytest.mark.parametrize("obj", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("recursion_depth", [0, 1, 2])
    def test_nan_inf(self, obj, recursion_depth):
        if obj == float("inf"):
            expected = "float('inf')"
        elif obj == float("-inf"):
            expected = "float('-inf')"
        else:
            expected = "float('nan')"

        for _ in range(recursion_depth):
            obj = {"x": obj}
            expected = f"{{'x': {expected}}}"

        assert util.make_repr("Example", obj) == (f"Example({expected})", False, True)
