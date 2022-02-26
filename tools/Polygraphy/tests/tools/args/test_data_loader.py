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

from textwrap import dedent

import numpy as np
import pytest
from polygraphy import util
from polygraphy.common import TensorMetadata
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import DataLoaderArgs, ModelArgs
from tests.tools.args.helper import ArgGroupTestHelper

ARG_CASES = [
    (["--seed=123"], ["seed"], [123]),
    (["--int-min=23", "--int-max=94"], ["int_range"], [(23, 94)]),
    (["--float-min=2.3", "--float-max=9.4"], ["float_range"], [(2.3, 9.4)]),
    ([], ["val_range"], [None], [(0.0, 1.0)]),  # When not specified, this should default to None.
    (["--val-range", "[0.0,2.3]"], ["val_range"], [{"": (0.0, 2.3)}]),
    (["--val-range", "[1,5]"], ["val_range"], [{"": (1, 5)}]),  # Should work for integral quantities
    (["--val-range", "inp0:[0.0,2.3]", "inp1:[4.5,9.6]"], ["val_range"], [{"inp0": (0.0, 2.3), "inp1": (4.5, 9.6)}]),
    (
        ["--val-range", "[-1,0]", "inp0:[0.0,2.3]", "inp1:[4.5,9.6]"],
        ["val_range"],
        [{"": (-1, 0), "inp0": (0.0, 2.3), "inp1": (4.5, 9.6)}],
    ),
    (["--val-range", "))):[0.0,2.3]"], ["val_range"], [{")))": (0.0, 2.3)}]),
    (["--val-range", "'\"':[0.0,2.3]"], ["val_range"], [{"'\"'": (0.0, 2.3)}]),
    (["--iterations=12"], ["iterations"], [12]),
]


class TestDataLoaderArgs(object):
    @pytest.mark.parametrize("case", ARG_CASES, ids=lambda c: c[1][0])
    def test_parsing(self, case):
        arg_group = ArgGroupTestHelper(DataLoaderArgs())
        cli_args, attrs, expected, expected_dl = util.unpack_args(case, 4)
        expected_dl = expected_dl or expected

        arg_group.parse_args(cli_args)
        data_loader = arg_group.get_data_loader()

        for attr, exp, exp_dl in zip(attrs, expected, expected_dl):
            assert getattr(arg_group, attr) == exp
            assert getattr(data_loader, attr) == exp_dl

    def test_input_metadata(self):
        arg_group = ArgGroupTestHelper(DataLoaderArgs(), deps=[ModelArgs()])
        arg_group.parse_args(["--input-shapes", "test0:[1,1,1]", "test1:[2,32,2]"])
        data_loader = arg_group.get_data_loader()

        for feed_dict in data_loader:
            assert feed_dict["test0"].shape == (1, 1, 1)
            assert feed_dict["test1"].shape == (2, 32, 2)

    def test_override_input_metadata(self):
        arg_group = ArgGroupTestHelper(DataLoaderArgs(), deps=[ModelArgs()])
        arg_group.parse_args([])
        data_loader = arg_group.get_data_loader(
            user_input_metadata=TensorMetadata().add("test0", dtype=np.float32, shape=(4, 4))
        )

        for feed_dict in data_loader:
            assert feed_dict["test0"].shape == (4, 4)

    def test_data_loader_script(self):
        arg_group = ArgGroupTestHelper(DataLoaderArgs())

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

            arg_group.parse_args(["--data-loader-script", f.name, "--data-loader-func-name=my_load_data"])

            assert arg_group.data_loader_script == f.name
            assert arg_group.data_loader_func_name == "my_load_data"

            data_loader = arg_group.get_data_loader()
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
    def test_val_range_errors(self, opts, expected_err):
        arg_group = ArgGroupTestHelper(DataLoaderArgs())

        with pytest.raises(PolygraphyException, match=expected_err):
            arg_group.parse_args(opts)
