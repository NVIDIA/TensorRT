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

import numpy as np
import pytest
from polygraphy import mod
from polygraphy.tools.args import util as args_util
from polygraphy.tools.script import inline, safe


@pytest.mark.parametrize("name", ["input", "input:0"])
class TestParseMeta(object):
    def test_parse_legacy(self, name):  # Legacy argument format used comma.
        meta_args = ["{:},1x3x224x224".format(name)]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == (1, 3, 224, 224)
        assert meta[name].dtype is None

    def test_parse_shape_only(self, name):
        meta_args = ["{name}:[1,3,224,224]".format(name=name)]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == (1, 3, 224, 224)
        assert meta[name].dtype is None

    def test_parse_empty_shape(self, name):
        meta_args = ["{name}:[0,3,0,224]".format(name=name)]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == (0, 3, 0, 224)
        assert meta[name].dtype is None

    def test_parse_shape_scalar(self, name):
        meta_args = ["{name}:[]".format(name=name)]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == tuple()

    def test_parse_shape_single_dim(self, name):
        meta_args = ["{name}:[1]".format(name=name)]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == (1,)

    def test_parse_dtype_only(self, name):
        meta_args = ["{name}:float32".format(name=name)]
        meta = args_util.parse_meta(meta_args, includes_shape=False)
        assert meta[name].shape is None
        assert meta[name].dtype == np.float32

    def test_parse_shape_dtype(self, name):
        meta_args = ["{name}:[1,3,224,224]:float32".format(name=name)]
        meta = args_util.parse_meta(meta_args)
        assert meta[name].shape == (1, 3, 224, 224)
        assert meta[name].dtype == np.float32

    def test_parse_shape_dtype_auto(self, name):
        meta_args = ["{name}:auto:auto".format(name=name)]
        meta = args_util.parse_meta(meta_args)
        assert meta[name].shape is None
        assert meta[name].dtype is None

    @pytest.mark.parametrize("quote", ['"', "'", ""])
    def test_parse_shape_with_dim_param_quoted(self, name, quote):
        meta_args = ["{name}:[{quote}batch{quote},3,224,224]".format(name=name, quote=quote)]
        meta = args_util.parse_meta(meta_args, includes_dtype=False)
        assert meta[name].shape == ("batch", 3, 224, 224)


class TestRunScript(object):
    def test_default_args(self):
        def script_add(script, arg0=0, arg1=0):
            result_name = safe("result")
            script.append_suffix(safe("{:} = {:} + {:}", inline(result_name), arg0, arg1))
            return result_name

        assert args_util.run_script(script_add) == 0
        assert args_util.run_script(script_add, 1) == 1
        assert args_util.run_script(script_add, 1, 2) == 3
