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
import numpy as np
import pytest
from polygraphy import util
from polygraphy.comparator import CompareFunc, IterationResult
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER


class TestBasicCompareFunc(object):
    def test_can_compare_bool(self):
        iter_result0 = IterationResult(outputs={"output": np.zeros((4, 4), dtype=np.bool)})
        iter_result1 = IterationResult(outputs={"output": np.ones((4, 4), dtype=np.bool)})

        compare_func = CompareFunc.simple()
        acc = compare_func(iter_result0, iter_result1)

        assert not acc["output"]

    @pytest.mark.parametrize("mode", ["abs", "rel"])
    def test_per_output_tol(self, mode):
        OUT0_NAME = "output0"
        OUT1_NAME = "output1"
        OUT_VALS = np.ones((4, 4))

        iter_result0 = IterationResult(outputs={OUT0_NAME: OUT_VALS, OUT1_NAME: OUT_VALS})
        iter_result1 = IterationResult(outputs={OUT0_NAME: OUT_VALS, OUT1_NAME: OUT_VALS + 1})

        # With default tolerances, out1 is wrong for the second result.
        compare_func = CompareFunc.simple()
        acc = compare_func(iter_result0, iter_result1)
        assert acc[OUT0_NAME]
        assert not acc[OUT1_NAME]

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
        assert acc[OUT0_NAME]
        assert acc[OUT1_NAME]

    @pytest.mark.parametrize("mode", ["abs", "rel"])
    def test_per_output_tol_fallback(self, mode):
        OUT0_NAME = "output0"
        OUT1_NAME = "output1"
        OUT_VALS = np.ones((4, 4))

        iter_result0 = IterationResult(outputs={OUT0_NAME: OUT_VALS + 1, OUT1_NAME: OUT_VALS})
        iter_result1 = IterationResult(outputs={OUT0_NAME: OUT_VALS, OUT1_NAME: OUT_VALS + 1})

        acc = CompareFunc.simple()(iter_result0, iter_result1)
        assert not acc[OUT0_NAME]
        assert not acc[OUT1_NAME]

        # Do not specify tolerance for OUT0_NAME - it should fail with fallback tolerance
        tols = {
            OUT1_NAME: 1.0,
        }

        if mode == "abs":
            compare_func = CompareFunc.simple(atol=tols)
        else:
            compare_func = CompareFunc.simple(rtol=tols)

        acc = compare_func(iter_result0, iter_result1)
        assert not acc[OUT0_NAME]
        assert acc[OUT1_NAME]

    @pytest.mark.parametrize("mode", ["abs", "rel"])
    def test_default_tol_in_map(self, mode):
        # "" can be used to indicate a global tolerance
        OUT0_NAME = "output0"
        OUT_VALS = np.ones((4, 4))

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
        assert acc[OUT0_NAME]

    @pytest.mark.parametrize(
        "shape",
        [
            tuple(),
            (0, 2, 1, 2),
            (1,),
            (2, 2, 2, 2),
        ],
    )
    def test_non_matching_outputs(self, shape):
        iter_result0 = IterationResult(outputs={"output": np.zeros(shape, dtype=np.float32)})
        iter_result1 = IterationResult(outputs={"output": np.ones(shape, dtype=np.float32)})

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
    def test_check_error_stat(self, func, check_error_stat):
        iter_result0 = IterationResult(outputs={"output": func((100,), dtype=np.float32)})
        iter_result1 = IterationResult(outputs={"output": func((100,), dtype=np.float32)})

        iter_result0["output"][0] += 100

        # Even though the max diff is 100, atol=1 should cause this to pass since we're checking
        # against the mean error.
        compare_func = CompareFunc.simple(check_error_stat=check_error_stat, atol=1)

        if check_error_stat in ["max", "elemwise"]:
            assert not compare_func(iter_result0, iter_result1)["output"]
        else:
            assert compare_func(iter_result0, iter_result1)["output"]

    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "elemwise"])
    def test_atol_rtol_either_pass(self, check_error_stat):
        # If either rtol/atol is sufficient, the compare_func should pass
        res0 = IterationResult(outputs={"output": np.array([1, 2], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": np.array((1.25, 2.5), dtype=np.float32)})

        assert not CompareFunc.simple(check_error_stat=check_error_stat)(res0, res1)["output"]

        assert CompareFunc.simple(check_error_stat=check_error_stat, rtol=0.25)(res0, res1)["output"]
        assert CompareFunc.simple(check_error_stat=check_error_stat, atol=0.5)(res0, res1)["output"]

    def test_atol_rtol_combined_pass(self):
        # We should also be able to mix them - i.e. rtol might enough for some, atol for others.
        # If they cover the entire output range, it should pass.
        res0 = IterationResult(outputs={"output": np.array([0, 1, 2, 3], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": np.array((0.15, 1.25, 2.5, 3.75), dtype=np.float32)})

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
    def test_per_output_error_stat(self, check_error_stat):
        # output0 will only pass when using check_error_stat=mean
        res0 = IterationResult(
            outputs={
                "output0": np.array([0, 1, 2, 3], dtype=np.float32),
                "output1": np.array([0, 1, 2, 3], dtype=np.float32),
            }
        )
        res1 = IterationResult(
            outputs={
                "output0": np.array((0.15, 1.25, 2.5, 3.75), dtype=np.float32),
                "output1": np.array((0, 1, 2, 3), dtype=np.float32),
            }
        )

        atol = 0.4125
        assert not CompareFunc.simple(atol=atol)(res0, res1)["output0"]

        assert CompareFunc.simple(check_error_stat=check_error_stat, atol=atol)(res0, res1)["output0"]
        assert CompareFunc.simple(check_error_stat=check_error_stat, atol=atol)(res0, res1)["output1"]

    def test_invalid_error_stat(self):
        res0 = IterationResult(outputs={"output": np.array([0, 1, 2, 3], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": np.array((0.15, 1.25, 2.5, 3.75), dtype=np.float32)})

        with pytest.raises(PolygraphyException, match="Invalid choice"):
            CompareFunc.simple(check_error_stat="invalid-stat")(res0, res1)
