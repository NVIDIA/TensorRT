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
import numpy as np
import pytest
from polygraphy import util
from polygraphy.comparator import CompareFunc, IterationResult
from polygraphy.datatype import DataType
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER

build_torch = lambda a, **kwargs: util.array.to_torch(np.array(a, **kwargs))


@pytest.mark.parametrize("array_type", [np.array, build_torch], ids=["numpy", "torch"])
class TestSimpleCompareFunc:
    @pytest.mark.parametrize(
        "values0, values1, dtype, expected_max_absdiff, expected_max_reldiff",
        [
            # Low precision arrays should be casted to higher precisions to avoid overflows/underflows.
            ([0], [1], DataType.UINT8, 1, 1.0),
            ([1], [0], DataType.UINT8, 1, np.inf),
            ([0], [1], DataType.UINT16, 1, 1.0),
            ([1], [0], DataType.UINT16, 1, np.inf),
            ([0], [1], DataType.UINT32, 1, 1.0),
            ([1], [0], DataType.UINT32, 1, np.inf),
            ([25], [30], DataType.INT8, 5, 5.0 / 30.0),
            (
                [25],
                [30],
                DataType.FLOAT16,
                5,
                np.array([5.0], dtype=np.float32) / np.array([30.0], dtype=np.float32),
            ),
            ([1], [0], DataType.FLOAT16, 1, 1 / np.finfo(float).eps),
        ],
    )
    def test_comparison(
        self,
        values0,
        values1,
        dtype,
        expected_max_absdiff,
        expected_max_reldiff,
        array_type,
    ):
        if array_type != np.array:
            try:
                DataType.to_dtype(dtype, "torch")
            except:
                pytest.skip(f"Cannot convert {dtype} to torch")
        iter_result0 = IterationResult(
            outputs={"output": array_type(values0, dtype=dtype.numpy())}
        )
        iter_result1 = IterationResult(
            outputs={"output": array_type(values1, dtype=dtype.numpy())}
        )

        compare_func = CompareFunc.simple()
        acc = compare_func(iter_result0, iter_result1)

        comp_result = acc["output"]
        assert np.isclose(comp_result.max_absdiff, expected_max_absdiff)
        assert np.isclose(comp_result.max_absdiff, comp_result.mean_absdiff)
        assert np.isclose(comp_result.max_absdiff, comp_result.median_absdiff)

        assert np.isclose(comp_result.max_reldiff, expected_max_reldiff)
        assert np.isclose(comp_result.max_reldiff, comp_result.mean_reldiff)
        assert np.isclose(comp_result.max_reldiff, comp_result.median_reldiff)

    @pytest.mark.parametrize(
        "values0, values1, dtype, quantile, expected_abs_quantile, expected_rel_quantile",
        [
            ([0, 0.1, 0.5, 0.75, 1], [2, 2, 2, 2, 2], DataType.FLOAT16, 0.5, 1.5, 0.75),
            (
                [0, 0.1, 0.5, 0.75, 1],
                [2, 2, 2, 2, 2],
                DataType.FLOAT16,
                0.75,
                1.9,
                0.95,
            ),
            (
                [0.2, 0.2, 0.125, 0.11],
                [0.1, 0.1, 0.1, 0.1],
                DataType.FLOAT16,
                0.5,
                0.0625,
                0.625,
            ),
            ([0, 1, 2, 3, 4], [2, 2, 2, 2, 2], DataType.UINT8, 0.5, 1, 0.5),
            ([0, 1, 2, 3, 4], [2, 2, 2, 2, 2], DataType.UINT8, 0.75, 2, 1),
        ],
    )
    def test_quantile(
        self,
        values0,
        values1,
        dtype,
        quantile,
        expected_abs_quantile,
        expected_rel_quantile,
        array_type,
    ):
        if array_type != np.array:
            try:
                DataType.to_dtype(dtype, "torch")
            except:
                pytest.skip(f"Cannot convert {dtype} to torch")
        iter_result0 = IterationResult(
            outputs={"output": array_type(values0, dtype=dtype.numpy())}
        )
        iter_result1 = IterationResult(
            outputs={"output": array_type(values1, dtype=dtype.numpy())}
        )

        compare_func = CompareFunc.simple(
            check_error_stat="quantile", error_quantile=quantile
        )
        acc = compare_func(iter_result0, iter_result1)

        comp_result = acc["output"]
        assert np.isclose(
            comp_result.quantile_absdiff, expected_abs_quantile, atol=1e-4, rtol=1e-4
        )
        assert comp_result.quantile_absdiff <= comp_result.max_absdiff
        assert (quantile >= 0.5) == (
            comp_result.quantile_absdiff >= comp_result.median_absdiff
        )

        assert np.isclose(
            comp_result.quantile_reldiff, expected_rel_quantile, atol=1e-4, rtol=1e-4
        )
        assert comp_result.quantile_reldiff <= comp_result.max_reldiff
        assert (quantile >= 0.5) == (
            comp_result.quantile_reldiff >= comp_result.median_reldiff
        )

    def test_can_compare_bool(self, array_type):
        iter_result0 = IterationResult(
            outputs={"output": array_type(np.zeros((4, 4), dtype=bool))}
        )
        iter_result1 = IterationResult(
            outputs={"output": array_type(np.ones((4, 4), dtype=bool))}
        )

        compare_func = CompareFunc.simple()
        acc = compare_func(iter_result0, iter_result1)

        assert not acc["output"]

    @pytest.mark.parametrize("mode", ["abs", "rel"])
    def test_per_output_tol(self, mode, array_type):
        OUT0_NAME = "output0"
        OUT1_NAME = "output1"
        OUT_VALS = array_type(np.ones((4, 4)))

        iter_result0 = IterationResult(
            outputs={OUT0_NAME: OUT_VALS, OUT1_NAME: OUT_VALS}
        )
        iter_result1 = IterationResult(
            outputs={OUT0_NAME: OUT_VALS, OUT1_NAME: OUT_VALS + 1}
        )

        # With default tolerances, out1 is wrong for the second result.
        compare_func = CompareFunc.simple()
        acc = compare_func(iter_result0, iter_result1)
        assert bool(acc[OUT0_NAME])
        assert not bool(acc[OUT1_NAME])

        # But with custom tolerances, it should pass.
        tols = {
            OUT0_NAME: 0.0,
            OUT1_NAME: 1.0,
        }

        if mode == "abs":
            compare_func = CompareFunc.simple(atol=tols)
        else:
            compare_func = CompareFunc.simple(rtol=tols)

        acc = compare_func(iter_result0, iter_result1)
        assert bool(acc[OUT0_NAME])
        assert bool(acc[OUT1_NAME])

    @pytest.mark.parametrize("mode", ["abs", "rel"])
    def test_per_output_tol_fallback(self, mode, array_type):
        OUT0_NAME = "output0"
        OUT1_NAME = "output1"
        OUT_VALS = array_type(np.ones((4, 4)))

        iter_result0 = IterationResult(
            outputs={OUT0_NAME: OUT_VALS + 1, OUT1_NAME: OUT_VALS}
        )
        iter_result1 = IterationResult(
            outputs={OUT0_NAME: OUT_VALS, OUT1_NAME: OUT_VALS + 1}
        )

        acc = CompareFunc.simple()(iter_result0, iter_result1)
        assert not bool(acc[OUT0_NAME])
        assert not bool(acc[OUT1_NAME])

        # Do not specify tolerance for OUT0_NAME - it should fail with fallback tolerance
        tols = {
            OUT1_NAME: 1.0,
        }

        if mode == "abs":
            compare_func = CompareFunc.simple(atol=tols)
        else:
            compare_func = CompareFunc.simple(rtol=tols)

        acc = compare_func(iter_result0, iter_result1)
        assert not bool(acc[OUT0_NAME])
        assert bool(acc[OUT1_NAME])

    @pytest.mark.parametrize("mode", ["abs", "rel"])
    def test_default_tol_in_map(self, mode, array_type):
        # "" can be used to indicate a global tolerance
        OUT0_NAME = "output0"
        OUT_VALS = array_type(np.ones((4, 4)))

        iter_result0 = IterationResult(outputs={OUT0_NAME: OUT_VALS})
        iter_result1 = IterationResult(outputs={OUT0_NAME: OUT_VALS + 1})

        tols = {
            "": 1.0,
        }

        if mode == "abs":
            compare_func = CompareFunc.simple(atol=tols)
        else:
            compare_func = CompareFunc.simple(rtol=tols)

        acc = compare_func(iter_result0, iter_result1)
        assert bool(acc[OUT0_NAME])

    @pytest.mark.parametrize(
        "shape",
        [
            tuple(),
            (0, 2, 1, 2),
            (1,),
            (2, 2, 2, 2),
        ],
    )
    def test_non_matching_outputs(self, shape, array_type):
        iter_result0 = IterationResult(
            outputs={"output": array_type(np.zeros(shape, dtype=np.float32))}
        )
        iter_result1 = IterationResult(
            outputs={"output": array_type(np.ones(shape, dtype=np.float32))}
        )

        compare_func = CompareFunc.simple()

        with G_LOGGER.verbosity(G_LOGGER.ULTRA_VERBOSE):
            acc = compare_func(iter_result0, iter_result1)

        assert util.is_empty_shape(shape) or not acc["output"]

    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "elemwise"])
    @pytest.mark.parametrize(
        "func",
        [
            np.zeros,
            np.ones,
        ],
    )
    def test_check_error_stat(self, func, check_error_stat, array_type):
        iter_result0 = IterationResult(
            outputs={"output": array_type(func((100,), dtype=np.float32))}
        )
        iter_result1 = IterationResult(
            outputs={"output": array_type(func((100,), dtype=np.float32))}
        )

        iter_result0["output"][0] += 100

        # Even though the max diff is 100, atol=1 should cause this to pass since we're checking
        # against the mean error.
        compare_func = CompareFunc.simple(check_error_stat=check_error_stat, atol=1)

        if check_error_stat in ["max", "elemwise"]:
            assert not compare_func(iter_result0, iter_result1)["output"]
        else:
            assert compare_func(iter_result0, iter_result1)["output"]

    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "elemwise"])
    def test_atol_rtol_either_pass(self, check_error_stat, array_type):
        # If either rtol/atol is sufficient, the compare_func should pass
        res0 = IterationResult(outputs={"output": array_type([1, 2], dtype=np.float32)})
        res1 = IterationResult(
            outputs={"output": array_type((1.25, 2.5), dtype=np.float32)}
        )

        assert not CompareFunc.simple(check_error_stat=check_error_stat)(res0, res1)[
            "output"
        ]

        assert CompareFunc.simple(check_error_stat=check_error_stat, rtol=0.25)(
            res0, res1
        )["output"]
        assert CompareFunc.simple(check_error_stat=check_error_stat, atol=0.5)(
            res0, res1
        )["output"]

    def test_atol_rtol_combined_pass(self, array_type):
        # We should also be able to mix them - i.e. rtol might enough for some, atol for others.
        # If they cover the entire output range, it should pass.
        res0 = IterationResult(
            outputs={"output": array_type([0, 1, 2, 3], dtype=np.float32)}
        )
        res1 = IterationResult(
            outputs={"output": array_type((0.15, 1.25, 2.5, 3.75), dtype=np.float32)}
        )

        assert not CompareFunc.simple()(res0, res1)["output"]

        assert not CompareFunc.simple(atol=0.3)(res0, res1)["output"]
        assert not CompareFunc.simple(rtol=0.25)(res0, res1)["output"]

        assert CompareFunc.simple(atol=0.3, rtol=0.25)(res0, res1)["output"]

    @pytest.mark.parametrize(
        "check_error_stat",
        [
            {"output0": "mean", "output1": "max"},
            {"": "mean", "output1": "elemwise"},
            {"output0": "mean"},
            {"": "mean"},
        ],
    )
    def test_per_output_error_stat(self, check_error_stat, array_type):
        # output0 will only pass when using check_error_stat=mean
        res0 = IterationResult(
            outputs={
                "output0": array_type([0, 1, 2, 3], dtype=np.float32),
                "output1": array_type([0, 1, 2, 3], dtype=np.float32),
            }
        )
        res1 = IterationResult(
            outputs={
                "output0": array_type((0.15, 1.25, 2.5, 3.75), dtype=np.float32),
                "output1": array_type((0, 1, 2, 3), dtype=np.float32),
            }
        )

        atol = 0.4125
        assert not CompareFunc.simple(atol=atol)(res0, res1)["output0"]

        assert CompareFunc.simple(check_error_stat=check_error_stat, atol=atol)(
            res0, res1
        )["output0"]
        assert CompareFunc.simple(check_error_stat=check_error_stat, atol=atol)(
            res0, res1
        )["output1"]

    def test_invalid_error_stat(self, array_type):
        res0 = IterationResult(
            outputs={"output": array_type([0, 1, 2, 3], dtype=np.float32)}
        )
        res1 = IterationResult(
            outputs={"output": array_type([0.15, 1.25, 2.5, 3.75], dtype=np.float32)}
        )

        with pytest.raises(PolygraphyException, match="Invalid choice"):
            CompareFunc.simple(check_error_stat="invalid-stat")(res0, res1)

    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "elemwise"])
    @pytest.mark.parametrize("val0, val1", [(np.nan, 0.15), (0.15, np.nan)])
    def test_nans_always_fail(self, check_error_stat, val0, val1, array_type):
        res0 = IterationResult(outputs={"output": array_type([val0], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": array_type([val1], dtype=np.float32)})

        assert not CompareFunc.simple(check_error_stat=check_error_stat)(res0, res1)[
            "output"
        ]

    @pytest.mark.parametrize("infinities_compare_equal", (False, True))
    @pytest.mark.parametrize("val", (np.inf, -np.inf))
    def test_infinities_compare_equal(self, infinities_compare_equal, val, array_type):
        res0 = IterationResult(outputs={"output": array_type([val], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": array_type([val], dtype=np.float32)})

        cf = CompareFunc.simple(infinities_compare_equal=infinities_compare_equal)
        assert bool(cf(res0, res1)["output"]) == infinities_compare_equal


@pytest.mark.parametrize("array_type", [np.array, build_torch])
class TestIndicesCompareFunc:
    @pytest.mark.parametrize(
        "out0,out1,index_tolerance,expected",
        [
            ([0, 1, 2, 3], [0, 1, 2, 3], 0, True),
            # Check that dictionaries work for index tolerance
            ([0, 1, 2, 3], [0, 1, 2, 3], {"": 0}, True),
            ([0, 1, 2, 3], [0, 1, 2, 3], {"output": 0}, True),
            ([[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], 0, True),
            ([1, 0, 2, 3], [0, 1, 2, 3], 0, False),
            ([1, 0, 2, 3], [0, 1, 2, 3], 1, True),
            ([0, 1, 2, 3], [0, 1, 3, 2], 1, True),
            # Last 'index_tolerance' indices should be ignored.
            ([1, 0, 2, 7], [0, 1, 2, 3], 0, False),
            ([1, 0, 2, 7], [0, 1, 2, 3], 1, True),
            ([[2, 3, 4], [5, 6, 9]], [[3, 2, 4], [5, 9, 6]], 0, False),
            ([[2, 3, 4], [5, 6, 9]], [[3, 2, 4], [5, 9, 6]], 1, True),
            ([0, 1, 2, 3, 4, 5, 6], [0, 3, 2, 1, 4, 5, 6], 0, False),
            ([0, 1, 2, 3, 4, 5, 6], [0, 3, 2, 1, 4, 5, 6], 2, True),
        ],
    )
    def test_index_tolerance(self, out0, out1, index_tolerance, expected, array_type):
        res0 = IterationResult(outputs={"output": array_type(out0, dtype=np.int32)})
        res1 = IterationResult(outputs={"output": array_type(out1, dtype=np.int32)})

        assert (
            CompareFunc.indices(index_tolerance=index_tolerance)(res0, res1)["output"]
            == expected
        )
