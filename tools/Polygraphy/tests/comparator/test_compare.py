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
import polygraphy.comparator as comparator_module
import pytest
from polygraphy import util
from polygraphy.comparator import (
    BaseCompareFunc,
    CosineSimilarityCompareFunc,
    IndicesCompareFunc,
    IterationResult,
    L2CompareFunc,
    OutputCompareResult,
    PerceptualMetricsCompareFunc,
    PsnrCompareFunc,
    SimpleCompareFunc,
    SnrCompareFunc,
)
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

        compare_func = SimpleCompareFunc()
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

        compare_func = SimpleCompareFunc(
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

        compare_func = SimpleCompareFunc()
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
        compare_func = SimpleCompareFunc()
        acc = compare_func(iter_result0, iter_result1)
        assert bool(acc[OUT0_NAME])
        assert not bool(acc[OUT1_NAME])

        # But with custom tolerances, it should pass.
        tols = {
            OUT0_NAME: 0.0,
            OUT1_NAME: 1.0,
        }

        if mode == "abs":
            compare_func = SimpleCompareFunc(atol=tols)
        else:
            compare_func = SimpleCompareFunc(rtol=tols)

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

        acc = SimpleCompareFunc()(iter_result0, iter_result1)
        assert not bool(acc[OUT0_NAME])
        assert not bool(acc[OUT1_NAME])

        # Do not specify tolerance for OUT0_NAME - it should fail with fallback tolerance
        tols = {
            OUT1_NAME: 1.0,
        }

        if mode == "abs":
            compare_func = SimpleCompareFunc(atol=tols)
        else:
            compare_func = SimpleCompareFunc(rtol=tols)

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
            compare_func = SimpleCompareFunc(atol=tols)
        else:
            compare_func = SimpleCompareFunc(rtol=tols)

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

        compare_func = SimpleCompareFunc()

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
        compare_func = SimpleCompareFunc(check_error_stat=check_error_stat, atol=1)

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

        assert not SimpleCompareFunc(check_error_stat=check_error_stat)(res0, res1)[
            "output"
        ]

        assert SimpleCompareFunc(check_error_stat=check_error_stat, rtol=0.25)(
            res0, res1
        )["output"]
        assert SimpleCompareFunc(check_error_stat=check_error_stat, atol=0.5)(
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

        assert not SimpleCompareFunc()(res0, res1)["output"]

        assert not SimpleCompareFunc(atol=0.3)(res0, res1)["output"]
        assert not SimpleCompareFunc(rtol=0.25)(res0, res1)["output"]

        assert SimpleCompareFunc(atol=0.3, rtol=0.25)(res0, res1)["output"]

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
        assert not SimpleCompareFunc(atol=atol)(res0, res1)["output0"]

        assert SimpleCompareFunc(check_error_stat=check_error_stat, atol=atol)(
            res0, res1
        )["output0"]
        assert SimpleCompareFunc(check_error_stat=check_error_stat, atol=atol)(
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
            SimpleCompareFunc(check_error_stat="invalid-stat")(res0, res1)

    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "elemwise"])
    @pytest.mark.parametrize("val0, val1", [(np.nan, 0.15), (0.15, np.nan)])
    def test_nans_always_fail(self, check_error_stat, val0, val1, array_type):
        res0 = IterationResult(outputs={"output": array_type([val0], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": array_type([val1], dtype=np.float32)})

        assert not SimpleCompareFunc(check_error_stat=check_error_stat)(res0, res1)[
            "output"
        ]

    @pytest.mark.parametrize("infinities_compare_equal", (False, True))
    @pytest.mark.parametrize("val", (np.inf, -np.inf))
    def test_infinities_compare_equal(self, infinities_compare_equal, val, array_type):
        res0 = IterationResult(outputs={"output": array_type([val], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": array_type([val], dtype=np.float32)})

        cf = SimpleCompareFunc(infinities_compare_equal=infinities_compare_equal)
        assert bool(cf(res0, res1)["output"]) == infinities_compare_equal

    def test_is_base_compare_func(self, array_type):
        # The factory returns a functor so that compare_accuracy can access the averaging API.
        assert isinstance(SimpleCompareFunc(), BaseCompareFunc)

    def test_average_fields(self, array_type):
        # The metric fields are now a property of the result type the functor produces.
        fields = SimpleCompareFunc()._RESULT_CLASS.metric_fields()
        for field in [
            "max_absdiff",
            "max_reldiff",
            "mean_absdiff",
            "mean_reldiff",
            "median_absdiff",
            "median_reldiff",
            "quantile_absdiff",
            "quantile_reldiff",
        ]:
            assert field in fields

    def test_averaged_result_pass(self, array_type):
        # mean_absdiff (0.5) is within atol (1.0), so the abs check passes and the
        # combined (abs AND rel) failure is False -> the averaged result passes.
        # Constructor order: max/mean/median/quantile abs/rel diffs.
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.0, rtol=0.0)
        result = OutputCompareResult(
            None,
            None,
            0.5,
            100.0,
            None,
            None,
            None,
            None,
            thresholds=cf.thresholds_for("output"),
        )
        assert bool(result)
        assert np.isclose(result.mean_absdiff, 0.5)

    def test_averaged_result_fail(self, array_type):
        # Both abs (2.0 > 1.0) and rel (100.0 > 0.0) exceed tolerance -> fails.
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.0, rtol=0.0)
        result = OutputCompareResult(
            None,
            None,
            2.0,
            100.0,
            None,
            None,
            None,
            None,
            thresholds=cf.thresholds_for("output"),
        )
        assert not bool(result)

    def test_averaged_result_elemwise_raises(self, array_type):
        # elemwise has no scalar statistic to average -> the verdict cannot be derived, so accessing
        # it is a hard error.
        cf = SimpleCompareFunc(check_error_stat="elemwise")
        result = OutputCompareResult(
            0.5,
            0.5,
            None,
            None,
            None,
            None,
            None,
            None,
            thresholds=cf.thresholds_for("output"),
        )
        with pytest.raises(
            PolygraphyException,
            match=r"Cannot re-threshold an 'elemwise' comparison",
        ):
            bool(result)


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
            IndicesCompareFunc(index_tolerance=index_tolerance)(res0, res1)["output"]
            == expected
        )

    def test_is_base_compare_func(self, array_type):
        assert isinstance(IndicesCompareFunc(), BaseCompareFunc)

    def test_no_averaging_support(self, array_type):
        # indices comparison produces bare booleans, so it has no result type with averageable
        # metrics and cannot be used with average comparison / re-thresholding.
        assert IndicesCompareFunc()._RESULT_CLASS is None


def _make_results(vals0, vals1):
    return (
        IterationResult(outputs={"output": np.array(vals0, dtype=np.float32)}),
        IterationResult(outputs={"output": np.array(vals1, dtype=np.float32)}),
    )


@pytest.mark.parametrize(
    "func_type",
    [L2CompareFunc, CosineSimilarityCompareFunc, PsnrCompareFunc, SnrCompareFunc],
)
def test_output_names_recorded(func_type):
    res0, res1 = _make_results([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    result = func_type()(res0, res1)["output"]
    assert result.output0_name == "output" and result.output1_name == "output"


class TestL2CompareFunc:
    def test_identical_outputs(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        result = L2CompareFunc()(res0, res1)["output"]
        assert result
        assert np.isclose(result.l2_norm, 0.0)

    def test_known_l2_norm(self):
        # [1,0] vs [0,1]: diff=[1,-1], L2=sqrt(2)
        res0, res1 = _make_results([1.0, 0.0], [0.0, 1.0])
        result = L2CompareFunc(l2_threshold=10.0)(res0, res1)["output"]
        assert np.isclose(result.l2_norm, np.sqrt(2), rtol=1e-5)

    def test_tolerance_fail(self):
        res0, res1 = _make_results([1.0, 0.0], [0.0, 1.0])
        assert not L2CompareFunc(l2_threshold=1.0)(res0, res1)["output"]

    def test_per_output_tolerance(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        res0 = IterationResult(outputs={"out0": a.copy(), "out1": a.copy()})
        res1 = IterationResult(outputs={"out0": b.copy(), "out1": b.copy()})
        result = L2CompareFunc(l2_threshold={"out0": 1.0, "out1": 2.0})(res0, res1)
        assert not result["out0"]
        assert result["out1"]


class TestCosineSimilarityCompareFunc:
    def test_identical_outputs(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        result = CosineSimilarityCompareFunc()(res0, res1)["output"]
        assert result
        assert np.isclose(result.cosine_similarity, 1.0)

    def test_known_cosine_similarity(self):
        # [3,4] vs [4,3]: dot=24, ||a||=||b||=5, cosine=24/25
        res0, res1 = _make_results([3.0, 4.0], [4.0, 3.0])
        result = CosineSimilarityCompareFunc(cosine_similarity_threshold=-1.0)(
            res0, res1
        )["output"]
        assert np.isclose(result.cosine_similarity, 24.0 / 25.0, rtol=1e-5)

    def test_threshold_fail(self):
        res0, res1 = _make_results([3.0, 4.0], [4.0, 3.0])
        assert not CosineSimilarityCompareFunc(cosine_similarity_threshold=0.99)(
            res0, res1
        )["output"]

    def test_orthogonal_vectors_cosine_zero(self):
        # [1,0] vs [0,1]: cosine similarity = 0
        res0, res1 = _make_results([1.0, 0.0], [0.0, 1.0])
        result = CosineSimilarityCompareFunc(cosine_similarity_threshold=-1.0)(
            res0, res1
        )["output"]
        assert np.isclose(result.cosine_similarity, 0.0, atol=1e-6)

    def test_both_zero_vectors_cosine_one(self):
        # Both zero vectors: similarity defined as 1.0
        res0, res1 = _make_results([0.0, 0.0], [0.0, 0.0])
        result = CosineSimilarityCompareFunc(cosine_similarity_threshold=1.0)(
            res0, res1
        )["output"]
        assert np.isclose(result.cosine_similarity, 1.0)

    def test_one_zero_vector_cosine_zero(self):
        # One zero vector: similarity defined as 0.0
        res0, res1 = _make_results([0.0, 0.0], [1.0, 2.0])
        result = CosineSimilarityCompareFunc(cosine_similarity_threshold=-1.0)(
            res0, res1
        )["output"]
        assert np.isclose(result.cosine_similarity, 0.0)

    def test_nan_fails_rather_than_clamping_to_one(self):
        # An Inf input drives cosine to NaN (inf/inf), which must FAIL rather than clamp to 1.0 (false PASS).
        res0, res1 = _make_results([np.inf, 1.0, 2.0], [1.0, 2.0, 3.0])
        result = CosineSimilarityCompareFunc()(res0, res1)["output"]
        assert not result
        assert np.isnan(result.cosine_similarity)


class TestPsnrCompareFunc:
    def test_identical_outputs(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        result = PsnrCompareFunc()(res0, res1)["output"]
        assert result
        assert result.psnr == float("inf")

    def test_threshold_pass(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
        assert PsnrCompareFunc(psnr_threshold=0.0)(res0, res1)["output"]

    def test_threshold_fail(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
        assert not PsnrCompareFunc(psnr_threshold=1000.0)(res0, res1)["output"]


class TestSnrCompareFunc:
    def test_identical_outputs(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        result = SnrCompareFunc()(res0, res1)["output"]
        assert result
        assert result.snr == float("inf")

    def test_threshold_pass(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
        assert SnrCompareFunc(snr_threshold=0.0)(res0, res1)["output"]

    def test_threshold_fail(self):
        res0, res1 = _make_results([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
        assert not SnrCompareFunc(snr_threshold=1000.0)(res0, res1)["output"]

    def test_zero_signal_snr(self):
        # Signal is all zeros: SNR = -inf, which is below any threshold -> fails.
        res0, res1 = _make_results([0.0, 0.0], [1.0, 1.0])
        result = SnrCompareFunc(snr_threshold=0.0)(res0, res1)["output"]
        assert result.snr == -float("inf")
        assert not result


_SINGLE_METRIC_PARAMS = [
    (L2CompareFunc, "l2_norm", "l2_threshold", 2.0, 1.0, 3.0),
    (
        CosineSimilarityCompareFunc,
        "cosine_similarity",
        "cosine_similarity_threshold",
        0.9,
        0.99,
        0.5,
    ),
    (PsnrCompareFunc, "psnr", "psnr_threshold", 30.0, 40.0, 20.0),
    (SnrCompareFunc, "snr", "snr_threshold", 20.0, 30.0, 10.0),
]


@pytest.mark.parametrize(
    "func_type, field, threshold_kwarg, thr, pass_val, fail_val", _SINGLE_METRIC_PARAMS
)
def test_single_metric_has_one_averaging_field(
    func_type, field, threshold_kwarg, thr, pass_val, fail_val
):
    cf = func_type(**{threshold_kwarg: thr})
    assert cf._RESULT_CLASS.metric_fields() == [field]


@pytest.mark.parametrize(
    "func_type, field, threshold_kwarg, thr, pass_val, fail_val", _SINGLE_METRIC_PARAMS
)
def test_single_metric_averaged_result_pass_fail(
    func_type, field, threshold_kwarg, thr, pass_val, fail_val
):
    cf = func_type(**{threshold_kwarg: thr})
    assert cf._RESULT_CLASS(pass_val, thresholds=cf.thresholds_for("output"))
    assert not cf._RESULT_CLASS(fail_val, thresholds=cf.thresholds_for("output"))


def test_metric_describe_increases_precision_near_boundary():
    # When the value and threshold would render identically at the default precision but actually
    # differ (a near-boundary PASS/FAIL), the summary line shows enough digits to tell them apart.
    from polygraphy.comparator import CosineSimilarityResult, CosineSimilarityThreshold

    result = CosineSimilarityResult(
        0.998436, thresholds=CosineSimilarityThreshold(0.998437)
    )
    line, passed = result.describe()
    assert not passed  # 0.998436 < 0.998437 (higher is better)
    assert "0.998436" in line and "0.998437" in line


def test_metric_describe_uses_default_precision_when_distinct():
    # Well-separated values keep the usual 5-significant-figure formatting.
    from polygraphy.comparator import L2Result, L2Threshold

    line, passed = L2Result(4.0, thresholds=L2Threshold(0.5)).describe()
    assert line == "L2 Norm: 4 (tolerance: 0.5) | FAILED"


@pytest.mark.parametrize(
    "func_type",
    [L2CompareFunc, CosineSimilarityCompareFunc, PsnrCompareFunc, SnrCompareFunc],
)
class TestSingleMetricShapeHandling:
    # Shape handling is shared via BaseCompareFunc, so it is tested once across all functors.
    def test_shape_mismatch_fails_by_default(self, func_type):
        res0 = IterationResult(outputs={"output": np.zeros((2, 2), dtype=np.float32)})
        res1 = IterationResult(outputs={"output": np.zeros((4,), dtype=np.float32)})
        # check_shapes defaults to True, so differing shapes is an immediate (bare-False) failure.
        assert not func_type()(res0, res1)["output"]

    def test_no_shape_check_reshapes_before_comparing(self, func_type):
        # With check_shapes=False, a (4,) buffer is reshaped to the reference (2, 2) and compared.
        res0 = IterationResult(
            outputs={"output": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
        )
        res1 = IterationResult(
            outputs={"output": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
        )
        result = func_type(check_shapes=False)(res0, res1)["output"]
        assert bool(result)

    def test_summary_label_set(self, func_type):
        # Each functor labels its pass/fail summary so multiple compare funcs can be told apart.
        # The label now lives on the threshold type the functor produces.
        assert func_type()._summary_label() == func_type._THRESHOLD_CLASS._METRIC_LABEL


_PER_OUTPUT_THRESHOLD_PARAMS = [
    # (func_type, threshold_kwarg, loose_thr, tight_thr, vals_a, vals_b)
    # For L2: sqrt(2) ≈ 1.414; loose=2.0 passes, tight=1.0 fails.
    (L2CompareFunc, "l2_threshold", 2.0, 1.0, [1.0, 0.0], [0.0, 1.0]),
    # For cosine: [3,4]·[4,3] = 24/25 = 0.96; loose=-1 passes, tight=0.99 fails.
    (
        CosineSimilarityCompareFunc,
        "cosine_similarity_threshold",
        -1.0,
        0.99,
        [3.0, 4.0],
        [4.0, 3.0],
    ),
    # For PSNR: [1,2,3] vs [1.1,2.1,3.1] -> finite PSNR; loose=0 passes, tight=1000 fails.
    (PsnrCompareFunc, "psnr_threshold", 0.0, 1000.0, [1.0, 2.0, 3.0], [1.1, 2.1, 3.1]),
    # For SNR: same data; loose=0 passes, tight=1000 fails.
    (SnrCompareFunc, "snr_threshold", 0.0, 1000.0, [1.0, 2.0, 3.0], [1.1, 2.1, 3.1]),
]


@pytest.mark.parametrize(
    "func_type, threshold_kwarg, loose_thr, tight_thr, vals_a, vals_b",
    _PER_OUTPUT_THRESHOLD_PARAMS,
)
def test_per_output_threshold(
    func_type, threshold_kwarg, loose_thr, tight_thr, vals_a, vals_b
):
    # out0 uses the loose threshold (passes), out1 uses the tight threshold (fails).
    res0 = IterationResult(
        outputs={
            "out0": np.array(vals_a, dtype=np.float32),
            "out1": np.array(vals_a, dtype=np.float32),
        }
    )
    res1 = IterationResult(
        outputs={
            "out0": np.array(vals_b, dtype=np.float32),
            "out1": np.array(vals_b, dtype=np.float32),
        }
    )
    result = func_type(**{threshold_kwarg: {"out0": loose_thr, "out1": tight_thr}})(
        res0, res1
    )
    assert result["out0"]
    assert not result["out1"]


class TestPerceptualMetricsCompareFunc:
    # NOTE: The functor loads the (torch-based) LPIPS model lazily, so the averaging API can be
    # tested without torch/lpips installed.
    def test_average_fields(self):
        assert PerceptualMetricsCompareFunc(
            lpips_threshold=0.1
        )._RESULT_CLASS.metric_fields() == ["lpips"]

    def test_averaged_result_pass(self):
        cf = PerceptualMetricsCompareFunc(lpips_threshold=0.1)
        assert cf._RESULT_CLASS(0.05, thresholds=cf.thresholds_for("output"))

    def test_averaged_result_fail(self):
        cf = PerceptualMetricsCompareFunc(lpips_threshold=0.1)
        assert not cf._RESULT_CLASS(0.2, thresholds=cf.thresholds_for("output"))


def test_removed_distance_and_quality_compare_funcs():
    # The combined DistanceMetrics/QualityMetrics functions were replaced by the single-metric ones.
    for name in (
        "DistanceMetricsCompareFunc",
        "QualityMetricsCompareFunc",
        "DistanceMetricsResult",
        "QualityMetricsResult",
    ):
        assert not hasattr(comparator_module, name), f"{name} should have been removed"
