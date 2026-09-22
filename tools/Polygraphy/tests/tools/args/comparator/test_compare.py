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
from polygraphy.comparator import IterationResult
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import (
    ComparatorCompareArgs,
    CompareFuncCosineSimilarityArgs,
    CompareFuncIndicesArgs,
    CompareFuncL2Args,
    CompareFuncPerceptualMetricsArgs,
    CompareFuncPsnrArgs,
    CompareFuncSimpleArgs,
    CompareFuncSnrArgs,
)
from polygraphy.tools.args import util as args_util
from polygraphy.tools.script import Script
from tests.tools.args.helper import ArgGroupTestHelper


def _simple_arg_group():
    return ArgGroupTestHelper(
        CompareFuncSimpleArgs(),
        deps=[
            ComparatorCompareArgs(),
            CompareFuncIndicesArgs(),
            CompareFuncL2Args(),
            CompareFuncCosineSimilarityArgs(),
            CompareFuncPsnrArgs(),
            CompareFuncSnrArgs(),
            CompareFuncPerceptualMetricsArgs(),
        ],
    )


class TestCompareFuncSimple:
    @pytest.mark.parametrize(
        "check_error_stat", ["max", "median", "mean", "elemwise", "quantile"]
    )
    def test_error_stat(self, check_error_stat):
        arg_group = _simple_arg_group()
        arg_group.parse_args([f"--check-error-stat={check_error_stat}"])

        assert arg_group.check_error_stat == {"": check_error_stat}

    @pytest.mark.parametrize(
        "args, expected",
        [
            (
                ["mean", "output0:median", "output1:max"],
                {"": "mean", "output0": "median", "output1": "max"},
            ),
            (
                ["output0:median", "output1:elemwise"],
                {"output0": "median", "output1": "elemwise"},
            ),
        ],
    )
    def test_error_stat_per_output(self, args, expected):
        arg_group = _simple_arg_group()
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
            arg_group = _simple_arg_group()
            arg_group.parse_args(["--check-error-stat"] + args)

    @pytest.mark.parametrize("val", (np.inf, -np.inf))
    def test_infinities_compare_equal(self, val):
        arg_group = _simple_arg_group()
        arg_group.parse_args(["--infinities-compare-equal"])

        assert arg_group.infinities_compare_equal

        res0 = IterationResult(outputs={"output": np.array([val], dtype=np.float32)})
        res1 = IterationResult(outputs={"output": np.array([val], dtype=np.float32)})

        cf = args_util.run_script(arg_group.add_to_script)
        assert bool(cf(res0, res1)["output"])


class TestCompareFuncIndices:
    def test_always_adds_to_script(self):
        # Indices is not the default comparison func, so it should always add itself to the script.
        arg_group = ArgGroupTestHelper(
            CompareFuncIndicesArgs(),
            deps=[
                ComparatorCompareArgs(),
                CompareFuncSimpleArgs(),
                CompareFuncL2Args(),
                CompareFuncCosineSimilarityArgs(),
                CompareFuncPsnrArgs(),
                CompareFuncSnrArgs(),
                CompareFuncPerceptualMetricsArgs(),
            ],
        )
        arg_group.parse_args([])

        script = Script()
        assert str(arg_group.add_to_script(script)) == "indices_compare_func"
        assert script.vars


class TestDefaultNone:
    # Ensures that default values for all arguments are `None`.
    # See the comment in `polygraphy/tools/args/comparator/compare.py` for details.
    COMPARE_FUNC_GROUP_TYPES = [
        CompareFuncSimpleArgs,
        CompareFuncIndicesArgs,
        CompareFuncL2Args,
        CompareFuncCosineSimilarityArgs,
        CompareFuncPsnrArgs,
        CompareFuncSnrArgs,
        CompareFuncPerceptualMetricsArgs,
    ]

    @pytest.mark.parametrize("arg_group_type", COMPARE_FUNC_GROUP_TYPES)
    def test_default_args_are_none(self, arg_group_type):
        deps = [ComparatorCompareArgs()] + [
            g()
            for g in TestDefaultNone.COMPARE_FUNC_GROUP_TYPES
            if g is not arg_group_type
        ]
        arg_group = ArgGroupTestHelper(arg_group_type(), deps=deps)
        assert len(arg_group.arg_group.group._group_actions) > 0
        for action in arg_group.arg_group.group._group_actions:
            assert action.default is None


class TestSingleMetricThresholds:
    # The single-metric comparison functions parse their threshold into a per-output dict and
    # forward it to the constructed functor. (L2/cosine are also exercised end-to-end elsewhere;
    # this fills in value-level coverage for psnr/snr and the kept --l2-tolerance alias.)

    def _arg_group(self, arg_group_type):
        deps = [ComparatorCompareArgs()] + [
            g()
            for g in TestDefaultNone.COMPARE_FUNC_GROUP_TYPES
            if g is not arg_group_type
        ]
        return ArgGroupTestHelper(arg_group_type(), deps=deps)

    _THRESHOLD_PARAMS = [
        (CompareFuncL2Args, "--l2-threshold", "l2_threshold"),
        (
            CompareFuncCosineSimilarityArgs,
            "--cosine-similarity-threshold",
            "cosine_similarity_threshold",
        ),
        (CompareFuncPsnrArgs, "--psnr-threshold", "psnr_threshold"),
        (CompareFuncSnrArgs, "--snr-threshold", "snr_threshold"),
    ]

    @pytest.mark.parametrize("arg_group_type, flag, attr", _THRESHOLD_PARAMS)
    def test_threshold_parses_global_key(self, arg_group_type, flag, attr):
        arg_group = self._arg_group(arg_group_type)
        arg_group.parse_args([flag, "0.5"])
        assert getattr(arg_group, attr) == {"": 0.5}

    @pytest.mark.parametrize("arg_group_type, flag, attr", _THRESHOLD_PARAMS)
    def test_threshold_parses_per_output_key(self, arg_group_type, flag, attr):
        arg_group = self._arg_group(arg_group_type)
        arg_group.parse_args([flag, "out0:0.25"])
        assert getattr(arg_group, attr) == {"out0": 0.25}

    @pytest.mark.parametrize(
        "arg_group_type, alias, attr",
        [
            (CompareFuncL2Args, "--l2-tolerance", "l2_threshold"),
            (CompareFuncPsnrArgs, "--psnr-tolerance", "psnr_threshold"),
            (CompareFuncSnrArgs, "--snr-tolerance", "snr_threshold"),
        ],
    )
    def test_tolerance_alias_parses_to_threshold(self, arg_group_type, alias, attr):
        # The renamed single-metric flags keep their legacy --<metric>-tolerance spelling as an
        # alias for --<metric>-threshold, so previously-valid command lines keep working.
        arg_group = self._arg_group(arg_group_type)
        arg_group.parse_args([alias, "0.5"])
        assert getattr(arg_group, attr) == {"": 0.5}
