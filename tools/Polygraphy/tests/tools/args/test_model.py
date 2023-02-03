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
import sys
import pytest
from polygraphy.tools.args import ModelArgs
from tests.tools.args.helper import ArgGroupTestHelper


@pytest.fixture()
def group():
    return ArgGroupTestHelper(ModelArgs())


class TestModelArgs:
    def test_path(self, group):
        group.parse_args([])

        assert group.path is None
        assert group.model_type is None

        group.parse_args(["model.onnx"])

        assert group.path == os.path.abspath("model.onnx")
        assert group.model_type.is_onnx()

    def test_input_shapes(self, group):
        group.parse_args(["--input-shapes", "test0:[1,1]", "test1:[10]", "test:2:[25,301]", "test3:[]"])

        assert group.input_shapes["test0"].shape == [1, 1]
        assert group.input_shapes["test1"].shape == [10]
        assert group.input_shapes["test:2"].shape == [25, 301]
        assert group.input_shapes["test3"].shape == []

    def test_fixed_model_type(self):
        group = ArgGroupTestHelper(ModelArgs(required_model_type="onnx"))
        group.parse_args(["model.pb"])

        assert group.model_type.is_onnx()

    @pytest.mark.parametrize(
        "arg, expected_model, expected_extra_info",
        [
            ("model.onnx", "model.onnx", None),
            ("model.onnx:func", "model.onnx", "func"),
        ]
        if not "win" in sys.platform
        else [
            ("C:\\Users\\model.onnx", "C:\\Users\\model.onnx", None),
            ("C:\\Users\\model.onnx:func", "C:\\Users\\model.onnx", "func"),
        ],
    )
    def test_model_with_extra_info(self, group, arg, expected_model, expected_extra_info):
        group.parse_args([arg])

        assert group.path == os.path.abspath(expected_model)
        assert group.extra_model_info == expected_extra_info
