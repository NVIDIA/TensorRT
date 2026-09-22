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

import contextlib
import io

import pytest

from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import (
    ComparatorCompareArgs,
    ComparatorPostprocessArgs,
    CompareFuncCosineSimilarityArgs,
    CompareFuncIndicesArgs,
    CompareFuncL2Args,
    CompareFuncPerceptualMetricsArgs,
    CompareFuncPsnrArgs,
    CompareFuncSimpleArgs,
    CompareFuncSnrArgs,
    DataLoaderArgs,
    LoggerArgs,
)
from polygraphy.tools.script import Script
from tests.tools.args.helper import ArgGroupTestHelper


def _compare_arg_group():
    return ArgGroupTestHelper(
        ComparatorCompareArgs(),
        deps=[
            CompareFuncIndicesArgs(),
            CompareFuncSimpleArgs(),
            CompareFuncL2Args(),
            CompareFuncCosineSimilarityArgs(),
            CompareFuncPsnrArgs(),
            CompareFuncSnrArgs(),
            CompareFuncPerceptualMetricsArgs(),
            DataLoaderArgs(),
        ],
    )


def _trigger_check_average_validation(arg_group):
    # check_average validation is deferred to add_to_script (it reads attributes populated by
    # other argument groups during parsing, whose parse order is not guaranteed). Invoke
    # add_to_script the same way the run tool does so the validation actually runs. The
    # validation raises before the script/runner-dependent code is reached, so a bare results
    # variable name is sufficient.
    arg_group.add_to_script(Script(), "results")


class TestComparatorCompareArgs:
    def test_compare_func_allows_duplicates(self):
        arg_group = _compare_arg_group()

        arg_group.parse_args(["--compare-func", "simple", "simple", "indices"])

        assert arg_group.compare_funcs == ["simple", "simple", "indices"]

    def test_check_average_parsed(self):
        arg_group = _compare_arg_group()
        arg_group.parse_args(["--check-average", "--check-error-stat", "mean"])
        assert arg_group.check_average is True

    def test_check_average_default_none(self):
        arg_group = _compare_arg_group()
        arg_group.parse_args([])
        assert arg_group.check_average is None

    def test_check_average_rejects_fail_fast(self):
        arg_group = _compare_arg_group()
        arg_group.parse_args(
            ["--check-average", "--fail-fast", "--check-error-stat", "mean"]
        )
        with pytest.raises(
            PolygraphyException,
            match="--check-average cannot be combined with --fail-fast",
        ):
            _trigger_check_average_validation(arg_group)

    def test_check_average_rejects_elemwise_default(self):
        # 'simple' defaults to elemwise, which cannot be averaged.
        arg_group = _compare_arg_group()
        arg_group.parse_args(["--check-average"])
        with pytest.raises(
            PolygraphyException,
            match="--check-average is not supported with check_error_stat='elemwise'",
        ):
            _trigger_check_average_validation(arg_group)

    def test_check_average_rejects_explicit_elemwise(self):
        arg_group = _compare_arg_group()
        arg_group.parse_args(["--check-average", "--check-error-stat", "elemwise"])
        with pytest.raises(
            PolygraphyException,
            match="--check-average is not supported with check_error_stat='elemwise'",
        ):
            _trigger_check_average_validation(arg_group)

    def test_check_average_allows_compare_func_script(self):
        # A custom comparison function from a script may be a BaseCompareFunc that supports
        # averaging, so this combination is allowed at parse time (and validated at runtime).
        # The built-in 'simple'/elemwise check must not fire when a script overrides it.
        arg_group = _compare_arg_group()
        arg_group.parse_args(
            ["--check-average", "--compare-func-script", "my_compare.py"]
        )
        assert arg_group.check_average is True
        assert arg_group.compare_func_script == "my_compare.py"

    @pytest.mark.serial
    @pytest.mark.parametrize(
        "compare_func, options, option_names, valid_for",
        [
            ("simple", ["--index-tolerance=1"], ["--index-tolerance"], "['indices']"),
            ("indices", ["--rtol=1"], ["--rtol", "--rel-tol"], "['simple']"),
            ("indices", ["--atol=1"], ["--atol", "--abs-tol"], "['simple']"),
            (
                # The l2 arg group backs both 'l2' and the 'distance_metrics' alias, so the warning
                # names both; --l2-tolerance is a kept legacy alias of --l2-threshold.
                "simple",
                ["--l2-threshold=1"],
                ["--l2-threshold", "--l2-tolerance"],
                "['l2', 'distance_metrics']",
            ),
            (
                # --psnr-tolerance is a kept legacy alias of --psnr-threshold; the psnr arg group
                # backs both 'psnr' and the 'quality_metrics' alias, so the warning names both.
                "simple",
                ["--psnr-threshold=30"],
                ["--psnr-threshold", "--psnr-tolerance"],
                "['psnr', 'quality_metrics']",
            ),
            (
                "simple",
                ["--lpips-threshold=0.1"],
                ["--lpips-threshold", "--lpips-tolerance"],
                "['perceptual_metrics']",
            ),
        ],
    )
    def test_compare_func_warnings_for_unused_options(
        self, compare_func, options, option_names, valid_for
    ):
        outfile = io.StringIO()
        with contextlib.redirect_stdout(outfile), contextlib.redirect_stderr(outfile):
            # Keep logger arguments first they're parsed first so we actually write to the log file.
            arg_group = ArgGroupTestHelper(
                ComparatorCompareArgs(),
                deps=[
                    LoggerArgs(),
                    CompareFuncIndicesArgs(),
                    CompareFuncSimpleArgs(),
                    CompareFuncL2Args(),
                    CompareFuncCosineSimilarityArgs(),
                    CompareFuncPsnrArgs(),
                    CompareFuncSnrArgs(),
                    CompareFuncPerceptualMetricsArgs(),
                ],
            )
            arg_group.parse_args([f"--compare-func={compare_func}"] + options)

            outfile.seek(0)
            logging_out = outfile.read()
            assert (
                f"[W] Option: {'/'.join(option_names)} is only valid for comparison function(s): {valid_for}. "
                f"The selected comparison functions are: ['{compare_func}'], so this option will be ignored."
                in logging_out
            )


class TestComparatorPostprocess:
    @pytest.mark.parametrize(
        "args, expected",
        [
            (["top-6", "out0:top-1", "out1:top-3"], {"": 6, "out0": 1, "out1": 3}),
            (["top-6,axis=-1", "out0:top-1,axis=2"], {"": (6, -1), "out0": (1, 2)}),
        ],
    )
    def test_postprocess(self, args, expected):
        arg_group = ArgGroupTestHelper(
            ComparatorPostprocessArgs(),
        )
        arg_group.parse_args(["--postprocess"] + args)

        assert arg_group.postprocess == expected

    def test_postprocess_func_script_default_func_name(self):
        arg_group = ArgGroupTestHelper(ComparatorPostprocessArgs())
        arg_group.parse_args(["--postprocess-func-script", "postprocess.py"])

        assert arg_group.postprocess_func_script == "postprocess.py"
        assert arg_group.postprocess_func_name == "postprocess_outputs"

    def test_postprocess_func_script_custom_func_name(self):
        arg_group = ArgGroupTestHelper(ComparatorPostprocessArgs())
        arg_group.parse_args(
            ["--postprocess-func-script", "postprocess.py:my_postprocess"]
        )

        assert arg_group.postprocess_func_script == "postprocess.py"
        assert arg_group.postprocess_func_name == "my_postprocess"

    @pytest.mark.serial
    def test_postprocess_func_script_overrides_postprocess(self):
        outfile = io.StringIO()
        with contextlib.redirect_stdout(outfile), contextlib.redirect_stderr(outfile):
            arg_group = ArgGroupTestHelper(
                ComparatorPostprocessArgs(),
                deps=[LoggerArgs()],
            )
            arg_group.parse_args(
                [
                    "--postprocess",
                    "top-1",
                    "--postprocess-func-script",
                    "my_postprocess.py",
                ]
            )

            outfile.seek(0)
            logging_out = outfile.read()
            assert (
                "[W] Argument: '--postprocess/--postprocess-func' will be ignored since '--postprocess-func-script' was provided."
                in logging_out
            )
