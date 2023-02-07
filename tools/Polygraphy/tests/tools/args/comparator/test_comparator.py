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

import pytest
from polygraphy.tools.args import ComparatorPostprocessArgs
from tests.tools.args.helper import ArgGroupTestHelper


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
