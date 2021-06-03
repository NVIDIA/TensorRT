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
from polygraphy.comparator import CompareFunc, IterationResult
from polygraphy.logger.logger import G_LOGGER


class TestBasicCompareFunc(object):
    def test_can_compare_bool(self):
        iter_result0 = IterationResult(outputs={"output": np.zeros((4, 4), dtype=np.bool)})
        iter_result1 = IterationResult(outputs={"output": np.ones((4, 4), dtype=np.bool)})

        compare_func = CompareFunc.basic_compare_func()
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
        compare_func = CompareFunc.basic_compare_func()
        acc = compare_func(iter_result0, iter_result1)
        assert acc[OUT0_NAME]
        assert not acc[OUT1_NAME]

        # But with custom tolerances, it should pass.
        tols = {
            OUT0_NAME: 0.0,
            OUT1_NAME: 1.0,
        }

        if mode == "abs":
            compare_func = CompareFunc.basic_compare_func(atol=tols)
        else:
            compare_func = CompareFunc.basic_compare_func(rtol=tols)

        acc = compare_func(iter_result0, iter_result1)
        assert acc[OUT0_NAME]
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
            compare_func = CompareFunc.basic_compare_func(atol=tols)
        else:
            compare_func = CompareFunc.basic_compare_func(rtol=tols)

        acc = compare_func(iter_result0, iter_result1)
        assert acc[OUT0_NAME]


    def test_non_matching_outputs(self):
        iter_result0 = IterationResult(outputs={"output": np.zeros((2, 2, 2, 2), dtype=np.float32)})
        iter_result1 = IterationResult(outputs={"output": np.ones((2, 2, 2, 2), dtype=np.float32)})

        compare_func = CompareFunc.basic_compare_func()

        with G_LOGGER.verbosity(G_LOGGER.ULTRA_VERBOSE):
            acc = compare_func(iter_result0, iter_result1)

        assert not acc["output"]
