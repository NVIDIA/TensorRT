# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from polygraphy import util


@pytest.mark.parametrize(
    "values, dtype, expected_values, expected_indices",
    [
        ([0, 2, 2, 1], np.uint8, [2, 2, 1], [1, 2, 3]),
        ([0, 2**64 - 1, 1], np.uint64, [2**64 - 1, 1, 0], [1, 2, 0]),
        ([-(2**31), 1, 0], np.int32, [1, 0, -(2**31)], [1, 2, 0]),
    ],
)
def test_numpy_topk_integer_order(values, dtype, expected_values, expected_indices):
    result, indices = util.array.topk(np.array(values, dtype=dtype), 3, 0)
    np.testing.assert_array_equal(result, np.array(expected_values, dtype=dtype))
    np.testing.assert_array_equal(indices, expected_indices)


def test_numpy_topk_integer_axis():
    values = np.array([[0, 5, 1], [9, 0, 9]], dtype=np.uint8)
    result, indices = util.array.topk(values, 2, -1)
    np.testing.assert_array_equal(result, [[5, 1], [9, 9]])
    np.testing.assert_array_equal(indices, [[1, 2], [0, 2]])
