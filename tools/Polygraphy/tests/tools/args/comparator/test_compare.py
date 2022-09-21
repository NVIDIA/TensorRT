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


import numpy as np
import pytest
from polygraphy.comparator import IterationResult
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import ComparatorCompareArgs, CompareFuncIndicesArgs, CompareFuncSimpleArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.script import Script
from tests.tools.args.helper import ArgGroupTestHelper


class TestCompareFuncSimple:
    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "elemwise"])
    def test_error_stat(self, check_error_stat):
        arg_group = ArgGroupTestHelper(CompareFuncSimpleArgs(), deps=[ComparatorCompareArgs()])
        arg_group.parse_args([f"--check-error-stat={check_error_stat}"])

        assert arg_group.check_error_stat == {"": check_error_stat}

    @pytest.mark.parametrize(
        "args, expected",
        [
            (["mean", "output0:median", "output1:max"], {"": "mean", "output0": "median", "output1": "max"}),
            (["output0:median", "output1:elemwise"], {"output0": "median", "output1": "elemwise"}),
        ],
    )
    def test_error_stat_per_output(self, args, expected):
        arg_group = ArgGroupTestHelper(CompareFuncSimpleArgs(), deps=[ComparatorCompareArgs()])
        arg_group.parse_args(["--check-error-stat"] + args)

        assert arg_group.check_error_stat == expected

    @pytest.mark.parametrize(
        "args",
        [
            ["not-a-stat"],
            ["output0:fake"],
        ],
    )
    def test_invalid_error_stat(self, args):
        with pytest.raises(PolygraphyException, match="Invalid choice"):
            arg_group = ArgGroupTestHelper(CompareFuncSimpleArgs(), deps=[ComparatorCompareArgs()])
            arg_group.parse_args(["--check-error-stat"] + args)

    @pytest.mark.parametrize("val", (np.inf, -np.inf))
    def test_infinities_compare_equal(self, val):
        arg_group = ArgGroupTestHelper(CompareFuncSimpleArgs(), deps=[ComparatorCompareArgs()])
        arg_group.parse_args([f"--infinities-compare-equal"])

        assert arg_group.infinities_compare_equal

        res0 = IterationResult(outputs={"output": np.array([val], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": np.array([val], dtype=np.float32)})

        cf = args_util.run_script(arg_group.add_to_script)
        assert bool(cf(res0, res1)["output"])


class TestCompareFuncIndices:
    def test_always_adds_to_script(self):
        # Indices is not the default comparison func, so it should always add itself to the script.
        arg_group = ArgGroupTestHelper(CompareFuncIndicesArgs(), deps=[ComparatorCompareArgs()])
        arg_group.parse_args([])

        script = Script()
        assert str(arg_group.add_to_script(script)) == "compare_func"
        assert script.suffix
