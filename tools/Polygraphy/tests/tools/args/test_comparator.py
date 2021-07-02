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

import pytest
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import ComparatorCompareArgs
from tests.tools.args.helper import ArgGroupTestHelper


class TestComparatorCompare(object):
    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "elemwise"])
    def test_error_stat(self, check_error_stat):
        arg_group = ArgGroupTestHelper(ComparatorCompareArgs())
        arg_group.parse_args(["--check-error-stat={:}".format(check_error_stat)])

        assert arg_group.check_error_stat == {"": check_error_stat}

    @pytest.mark.parametrize(
        "args, expected",
        [
            (["mean", "output0:median", "output1:max"], {"": "mean", "output0": "median", "output1": "max"}),
            (["output0:median", "output1:elemwise"], {"output0": "median", "output1": "elemwise"}),
        ],
    )
    def test_error_stat_per_output(self, args, expected):
        arg_group = ArgGroupTestHelper(ComparatorCompareArgs())
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
            arg_group = ArgGroupTestHelper(ComparatorCompareArgs())
            arg_group.parse_args(["--check-error-stat"] + args)
