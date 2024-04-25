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

from polygraphy import util
from polygraphy.tools.args import (
    ComparatorCompareArgs,
    ComparatorPostprocessArgs,
    CompareFuncIndicesArgs,
    CompareFuncSimpleArgs,
    LoggerArgs,
)
from tests.tools.args.helper import ArgGroupTestHelper


class TestComparatorCompareArgs:
    @pytest.mark.serial
    @pytest.mark.parametrize(
        "compare_func, options, option_names, valid_for",
        [
            ("simple", ["--index-tolerance=1"], ["--index-tolerance"], "indices"),
            ("indices", ["--rtol=1"], ["--rtol", "--rel-tol"], "simple"),
            ("indices", ["--atol=1"], ["--atol", "--abs-tol"], "simple"),
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
                deps=[LoggerArgs(), CompareFuncIndicesArgs(), CompareFuncSimpleArgs()],
            )
            arg_group.parse_args([f"--compare-func={compare_func}"] + options)

            outfile.seek(0)
            logging_out = outfile.read()
            assert (
                f"[W] Option: {'/'.join(option_names)} is only valid for comparison function: '{valid_for}'. "
                f"The selected comparison function is: '{compare_func}', so this option will be ignored."
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

        assert list(arg_group.postprocess.values())[0] == expected
