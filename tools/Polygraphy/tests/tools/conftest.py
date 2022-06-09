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

from typing import List
import pytest


def make_poly_fixture(subtool: List[str]):
    @pytest.fixture()
    def poly_fixture(script_runner):
        def poly_fixture_impl(additional_opts: List[str] = [], expect_error: bool = False, *args, **kwargs):
            cmd = ["polygraphy"] + subtool + ["-v"] + additional_opts
            # NOTE: script_runner does not work very well in `in-process`` mode if you need to inspect stdout/stderr.
            # Occasionally, the output comes out empty - not clear why. Cave emptor!
            # Decorate your tests with `@pytest.mark.script_launch_mode("subprocess")` to use `subprocess` to avoid this issue.
            status = script_runner.run(*cmd, *args, **kwargs)
            assert status.success == (not expect_error)
            return status

        return poly_fixture_impl

    return poly_fixture


poly = make_poly_fixture([])
poly_run = make_poly_fixture(["run"])
poly_convert = make_poly_fixture(["convert"])
poly_inspect = make_poly_fixture(["inspect"])
poly_surgeon = make_poly_fixture(["surgeon"])
poly_surgeon_extract = make_poly_fixture(["surgeon", "extract"])
poly_template = make_poly_fixture(["template"])
poly_debug = make_poly_fixture(["debug"])
poly_data = make_poly_fixture(["data"])
