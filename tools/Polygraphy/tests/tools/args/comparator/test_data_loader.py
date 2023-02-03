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

import os
from textwrap import dedent

import numpy as np
import pytest
from polygraphy import util
from polygraphy.common import TensorMetadata
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import DataLoaderArgs, ModelArgs
from tests.tools.args.helper import ArgGroupTestHelper


@pytest.fixture()
def data_loader_args():
    return ArgGroupTestHelper(DataLoaderArgs(), deps=[ModelArgs()])


class TestDataLoaderArgs:
    @pytest.mark.parametrize(
        "case",
        [
            (["--seed=123"], ["seed"], [123]),
            (["--int-min=23", "--int-max=94"], ["_int_range"], [(23, 94)]),
            (["--float-min=2.3", "--float-max=9.4"], ["_float_range"], [(2.3, 9.4)]),
            ([], ["val_range"], [None], [(0.0, 1.0)]),  # When not specified, this should default to None.
            (["--val-range", "[0.0,2.3]"], ["val_range"], [{"": (0.0, 2.3)}]),
            (["--val-range", "[1,5]"], ["val_range"], [{"": (1, 5)}]),  # Should work for integral quantities
            (
                ["--val-range", "inp0:[0.0,2.3]", "inp1:[4.5,9.6]"],
                ["val_range"],
                [{"inp0": (0.0, 2.3), "inp1": (4.5, 9.6)}],
            ),
            (
                ["--val-range", "[-1,0]", "inp0:[0.0,2.3]", "inp1:[4.5,9.6]"],
                ["val_range"],
                [{"": (-1, 0), "inp0": (0.0, 2.3), "inp1": (4.5, 9.6)}],
            ),
            (["--val-range", "))):[0.0,2.3]"], ["val_range"], [{")))": (0.0, 2.3)}]),
            (["--val-range", "'\"':[0.0,2.3]"], ["val_range"], [{"'\"'": (0.0, 2.3)}]),
            (["--iterations=12"], ["iterations"], [12]),
            (["--val-range", "[0.0,inf]"], ["val_range"], [{"": (0.0, float("inf"))}]),
            (["--val-range", "[-inf,0.0]"], ["val_range"], [{"": (float("-inf"), 0.0)}]),
        ],
        ids=lambda c: c[1][0],
    )
    def test_parsing(self, data_loader_args, case):
        cli_args, attrs, expected, expected_dl = util.unpack_args(case, 4)
        expected_dl = expected_dl or expected

        data_loader_args.parse_args(cli_args)
        data_loader = data_loader_args.get_data_loader()

        for attr, exp, exp_dl in zip(attrs, expected, expected_dl):
            assert getattr(data_loader_args, attr) == exp
            assert getattr(data_loader, attr) == exp_dl

    def test_val_range_nan(self, data_loader_args):
        data_loader_args.parse_args(["--val-range", "[nan,0.0]"])
        data_loader = data_loader_args.get_data_loader()

        val_range = data_loader.val_range[""]
        assert util.is_nan(val_range[0])

    def test_input_metadata(self, data_loader_args):
        data_loader_args.parse_args(["--input-shapes", "test0:[1,1,1]", "test1:[2,32,2]"])
        data_loader = data_loader_args.get_data_loader()

        for feed_dict in data_loader:
            assert feed_dict["test0"].shape == (1, 1, 1)
            assert feed_dict["test1"].shape == (2, 32, 2)

    def test_override_input_metadata(self, data_loader_args):
        data_loader_args.parse_args([])
        data_loader = data_loader_args.get_data_loader(
            user_input_metadata=TensorMetadata().add("test0", dtype=np.float32, shape=(4, 4))
        )

        for feed_dict in data_loader:
            assert feed_dict["test0"].shape == (4, 4)

    def test_data_loader_script_default_func(self, data_loader_args):
        data_loader_args.parse_args(["--data-loader-script", "example.py"])
        assert data_loader_args.data_loader_func_name == "load_data"

    def test_data_loader_script(self, data_loader_args):
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(
                dedent(
                    """
                    import numpy as np

                    def my_load_data():
                        for _ in range(5):
                            yield {"inp": np.ones((3, 5), dtype=np.float32) * 6.4341}
                    """
                )
            )
            f.flush()
            os.fsync(f.fileno())

            data_loader_args.parse_args(["--data-loader-script", f"{f.name}:my_load_data"])

            assert data_loader_args.data_loader_script == f.name
            assert data_loader_args.data_loader_func_name == "my_load_data"

            data_loader = data_loader_args.get_data_loader()
            data = list(data_loader)
            assert len(data) == 5
            assert all(np.all(d["inp"] == np.ones((3, 5), dtype=np.float32) * 6.4341) for d in data)

    @pytest.mark.parametrize(
        "opts,expected_err",
        [
            (["--val-range", "x:[y,2]"], "could not be parsed as a number"),
            (["--val-range", "x:[1,2,3]"], "expected to receive exactly 2 values, but received 3"),
        ],
    )
    def test_val_range_errors(self, data_loader_args, opts, expected_err):

        with pytest.raises(PolygraphyException, match=expected_err):
            data_loader_args.parse_args(opts)

    def test_cannot_provide_two_custom_data_loader_methods(self, data_loader_args):

        with pytest.raises(SystemExit):
            data_loader_args.parse_args(["--data-loader-script", "my_script.py", "--load-inputs", "inputs.json"])
