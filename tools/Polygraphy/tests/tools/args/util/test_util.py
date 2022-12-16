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
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import util as args_util
from polygraphy.tools.script import inline, safe


@pytest.mark.parametrize("name", ["input", "input:0"])
class TestParseMeta:
    def test_parse_legacy(self, name):  # Legacy argument format used comma.
        meta_args = [f"{name},1x3x224x224"]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == [1, 3, 224, 224]
        assert meta[name].dtype is None

    def test_parse_shape_only(self, name):
        meta_args = [f"{name}:[1,3,224,224]"]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == [1, 3, 224, 224]
        assert meta[name].dtype is None

    def test_parse_empty_shape(self, name):
        meta_args = [f"{name}:[0,3,0,224]"]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == [0, 3, 0, 224]
        assert meta[name].dtype is None

    def test_parse_shape_scalar(self, name):
        meta_args = [f"{name}:[]"]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == []

    def test_parse_shape_single_dim(self, name):
        meta_args = [f"{name}:[1]"]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == [1]

    def test_parse_dtype_only(self, name):
        meta_args = [f"{name}:float32"]
        meta = args_util.parse_meta(meta_args, includes_shape=False)
        assert meta[name].shape is None
        assert meta[name].dtype == np.float32

    def test_parse_shape_dtype(self, name):
        meta_args = [f"{name}:[1,3,224,224]:float32"]
        meta = args_util.parse_meta(meta_args)
        assert meta[name].shape == [1, 3, 224, 224]
        assert meta[name].dtype == np.float32

    def test_parse_shape_dtype_auto(self, name):
        meta_args = [f"{name}:auto:auto"]
        meta = args_util.parse_meta(meta_args)
        assert meta[name].shape is None
        assert meta[name].dtype is None

    @pytest.mark.parametrize("quote", ['"', "'", ""])
    def test_parse_shape_with_dim_param_quoted(self, name, quote):
        meta_args = [f"{name}:[{quote}batch{quote},3,224,224]"]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == ["batch", 3, 224, 224]


class TestRunScript:
    def test_default_args(self):
        def script_add(script, arg0=0, arg1=0):
            result_name = safe("result")
            script.append_suffix(safe("{:} = {:} + {:}", inline(result_name), arg0, arg1))
            return result_name

        assert args_util.run_script(script_add) == 0
        assert args_util.run_script(script_add, 1) == 1
        assert args_util.run_script(script_add, 1, 2) == 3


class TestParseNumBytes:
    def test_none(self):
        assert args_util.parse_num_bytes(None) is None

    @pytest.mark.parametrize(
        "arg, expected",
        [
            ("16", 16),
            ("1e9", 1e9),
            ("2M", 2 << 20),
            ("2.3m", int(2.3 * (1 << 20))),
            ("4.3K", int(4.3 * (1 << 10))),
            ("7k", 7 << 10),
            ("1G", 1 << 30),
            ("2.5g", int(2.5 * (1 << 30))),
        ],
    )
    def test_num_bytes(self, arg, expected):
        assert args_util.parse_num_bytes(arg) == expected

    @pytest.mark.parametrize("arg", ["hi", "4.5x", "2.3.4"])
    def test_negative(self, arg):
        with pytest.raises(
            PolygraphyException,
            match=f"Could not convert {arg} to a number of bytes",
        ):
            args_util.parse_num_bytes(arg)
