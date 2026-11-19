# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from polygraphy.comparator import IterationResult, SimpleCompareFunc


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("error_stat", ["elemwise", "max"])
@pytest.mark.parametrize(
    "actual, reference, expected_difference",
    [
        (-(2**31), 0, 2**31),
        (2**31 - 1, -(2**31), 2**32 - 1),
        (2**24 + 1, 0, 2**24 + 1),
    ],
)
def test_int32_difference_does_not_overflow(
    backend, error_stat, actual, reference, expected_difference
):
    actual_array = np.array([actual], dtype=np.int32)
    reference_array = np.array([reference], dtype=np.int32)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        actual_array = torch.from_numpy(actual_array)
        reference_array = torch.from_numpy(reference_array)

    results = SimpleCompareFunc(rtol=0, atol=0, check_error_stat=error_stat)(
        IterationResult(outputs={"out": actual_array}),
        IterationResult(outputs={"out": reference_array}),
    )
    assert not results["out"]
    assert results["out"].max_absdiff == expected_difference
