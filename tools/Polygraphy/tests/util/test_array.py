#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy

import numpy as np
import pytest
import torch

from polygraphy import cuda, util
from polygraphy.datatype import DataType


@pytest.mark.parametrize(
    "obj",
    [
        np.transpose(np.ones((2, 3), dtype=np.float32)),
        torch.transpose(torch.ones((2, 3), dtype=torch.float32), 1, 0),
        cuda.DeviceArray(shape=(2, 3), dtype=DataType.FLOAT32),
    ],
    ids=[
        "numpy",
        "torch",
        "DeviceView",
    ],
)
class TestArrayFuncs:
    def test_nbytes(self, obj):
        nbytes = util.array.nbytes(obj)
        assert isinstance(nbytes, int)
        assert nbytes == 24

    def test_data_ptr(self, obj):
        data_ptr = util.array.data_ptr(obj)
        assert isinstance(data_ptr, int)

    def test_make_contiguous(self, obj):
        if isinstance(obj, cuda.DeviceView):
            pytest.skip("DeviceViews are always contiguous")

        obj = copy.copy(obj)
        assert not util.array.is_contiguous(obj)

        obj = util.array.make_contiguous(obj)
        assert util.array.is_contiguous(obj)

    def test_dtype(self, obj):
        assert util.array.dtype(obj) == DataType.FLOAT32

    def test_view(self, obj):
        obj = util.array.make_contiguous(obj)
        view = util.array.view(obj, dtype=DataType.UINT8, shape=(24, 1))
        assert util.array.dtype(view) == DataType.UINT8
        assert util.array.shape(view) == (24, 1)

    def test_resize(self, obj):
        # Need to make a copy since we're modifying the array.
        obj = copy.copy(util.array.make_contiguous(obj))
        obj = util.array.resize_or_reallocate(obj, (1, 1))
        assert util.array.shape(obj) == (1, 1)


@pytest.mark.parametrize(
    "obj, is_on_cpu",
    [
        (np.ones((2, 3)), True),
        (torch.ones((2, 3)), True),
        (torch.ones((2, 3), device="cuda"), False),
        (cuda.DeviceArray(shape=(2, 3), dtype=DataType.FLOAT32), False),
    ],
)
def test_is_on_cpu(obj, is_on_cpu):
    assert util.array.is_on_cpu(obj) == is_on_cpu


@pytest.mark.parametrize(
    "obj, is_on_gpu",
    [
        (np.ones((2, 3)), False),
        (torch.ones((2, 3)), False),
        (torch.ones((2, 3), device="cuda"), True),
        (cuda.DeviceArray(shape=(2, 3), dtype=DataType.FLOAT32), True),
    ],
)
def test_is_on_cpu(obj, is_on_gpu):
    assert util.array.is_on_gpu(obj) == is_on_gpu


@pytest.mark.parametrize(
    "lhs,rhs,expected",
    [
        (np.ones((2, 3)), np.ones((2, 3)), True),
        (np.zeros((2, 3)), np.ones((2, 3)), False),
        (torch.ones((2, 3)), torch.ones((2, 3)), True),
        (torch.zeros((2, 3)), torch.ones((2, 3)), False),
    ],
)
def test_equal(lhs, rhs, expected):
    assert util.array.equal(lhs, rhs) == expected


@pytest.mark.parametrize(
    "index,shape",
    [
        (7, (4, 4)),
        (12, (4, 4, 3, 2)),
    ],
)
def test_unravel_index(index, shape):
    assert util.array.unravel_index(index, shape) == np.unravel_index(index, shape)


@pytest.mark.parametrize(
    "lhs, rhs, expected",
    [
        (np.array([5.00001]), np.array([5.00]), True),
        (np.array([5.5]), np.array([5.00]), False),
        (torch.tensor([5.00001]), torch.tensor([5.00]), True),
        (torch.tensor([5.5]), torch.tensor([5.00]), False),
    ],
)
def test_allclose(lhs, rhs, expected):
    assert util.array.allclose(lhs, rhs) == expected


ARRAYS = [
    # Generate ints so FP rounding error is less of an issue
    np.random.randint(1, 25, size=(5, 2)).astype(np.float32),
    # Make sure functions work with an even or odd number of elements
    np.random.randint(1, 25, size=(1, 3)).astype(np.float32),
    # Generate binary values
    np.random.randint(0, 2, size=(5, 2)).astype(np.float32),
    # Test with scalars
    np.ones(shape=tuple(), dtype=np.float32),
]

TEST_CASES = []
IDS = []
for arr in ARRAYS:
    TEST_CASES.extend([(arr, arr), (torch.from_numpy(arr), arr)])
    IDS.extend(["numpy", "torch"])


@pytest.mark.parametrize("obj, np_arr", TEST_CASES, ids=IDS)
class TestArrayMathFuncs:
    # Test that the util.array implementations match NumPy
    @pytest.mark.parametrize(
        "func, np_func",
        [
            (util.array.max, np.amax),
            (util.array.argmax, np.argmax),
            (util.array.min, np.amin),
            (util.array.argmin, np.argmin),
            (util.array.mean, np.mean),
            (util.array.std, np.std),
            (util.array.var, np.var),
            (util.array.median, np.median),
            (util.array.any, np.any),
            (util.array.all, np.all),
        ],
    )
    def test_reduction_funcs(self, obj, np_arr, func, np_func):
        assert np.isclose(func(obj), np_func(np_arr))

    @pytest.mark.parametrize(
        "func, np_func",
        [
            (util.array.abs, np.abs),
            (util.array.isinf, np.isinf),
            (util.array.isnan, np.isnan),
            (util.array.argwhere, np.argwhere),
        ],
    )
    def test_array_funcs(self, obj, np_arr, func, np_func):
        obj = func(obj)
        assert util.array.equal(obj, np.array(np_func(np_arr)))

    def test_cast(self, obj, np_arr):
        dtype = DataType.INT32
        casted = util.array.cast(obj, dtype)
        assert util.array.dtype(casted) == dtype
        assert type(casted) == type(obj)

    def test_to_torch(self, obj, np_arr):
        assert isinstance(util.array.to_torch(obj), torch.Tensor)

    def test_to_numpy(self, obj, np_arr):
        assert isinstance(util.array.to_numpy(obj), np.ndarray)

    def test_histogram(self, obj, np_arr):
        hist, bins = util.array.histogram(obj)
        np_hist, np_bins = np.histogram(np_arr)
        np_hist = np_hist.astype(np_arr.dtype)

        assert util.array.allclose(hist, np_hist)
        assert util.array.allclose(bins, np_bins)

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_topk(self, obj, np_arr, k, axis):
        if axis >= len(util.array.shape(obj)):
            pytest.skip()
        topk_vals = util.array.topk(obj, k, axis)

        k_clamped = min(util.array.shape(obj)[axis], k)
        tensor = util.array.to_torch(np_arr)
        ref_topk_vals = torch.topk(tensor, k_clamped, axis)

        assert util.array.allclose(topk_vals[0], ref_topk_vals[0])

    @pytest.mark.parametrize(
        "func, np_func",
        [
            (util.array.subtract, np.subtract),
            (util.array.divide, np.divide),
            (util.array.logical_xor, np.logical_xor),
            (util.array.logical_and, np.logical_and),
            (util.array.greater, np.greater),
        ],
    )
    def test_binary_funcs(self, obj, np_arr, func, np_func):
        obj = func(obj, obj + 1)
        assert util.array.equal(obj, np.array(np_func(np_arr, np_arr + 1)))

    @pytest.mark.parametrize(
        "func, np_func, types",
        [
            (
                util.array.where,
                np.where,
                tuple(map(DataType.from_dtype, (np.bool8, np.float32, np.float32))),
            ),
        ],
    )
    def test_ternary_funcs(self, obj, np_arr, func, np_func, types):
        build_inputs = lambda input: map(
            lambda pair: util.array.cast(input + pair[0], pair[1]), enumerate(types)
        )
        obj = func(*build_inputs(obj))
        assert util.array.equal(obj, np.array(np_func(*build_inputs(np_arr))))
