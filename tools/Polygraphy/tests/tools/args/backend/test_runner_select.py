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
from polygraphy.tools.args import (
    OnnxrtRunnerArgs,
    PluginRefRunnerArgs,
    RunnerSelectArgs,
    TfRunnerArgs,
    TrtLegacyRunnerArgs,
    TrtRunnerArgs,
)
from tests.tools.args.helper import ArgGroupTestHelper


@pytest.fixture()
def runner_select_args():
    return ArgGroupTestHelper(
        RunnerSelectArgs(),
        deps=[
            TfRunnerArgs(),
            OnnxrtRunnerArgs(),
            PluginRefRunnerArgs(),
            TrtRunnerArgs(),
            TrtLegacyRunnerArgs(),
        ],
    )


class TestRunnerSelectArgs:
    @pytest.mark.parametrize(
        "opts,expected_runners",
        [
            (
                ["--trt"],
                [("trt", "TensorRT")],
            ),
            (
                ["--trt", "--tf"],
                [("trt", "TensorRT"), ("tf", "TensorFlow")],
            ),
            (
                ["--trt", "--onnxrt"],
                [("trt", "TensorRT"), ("onnxrt", "ONNX-Runtime")],
            ),
            (
                ["--onnxrt", "--trt"],
                [("onnxrt", "ONNX-Runtime"), ("trt", "TensorRT")],
            ),
            # We should be able to specify the same runner multiple times.
            (
                ["--onnxrt", "--onnxrt", "--onnxrt"],
                [("onnxrt", "ONNX-Runtime"), ("onnxrt", "ONNX-Runtime"), ("onnxrt", "ONNX-Runtime")],
            ),
        ],
    )
    def test_order_preserved(self, runner_select_args, opts, expected_runners):
        runner_select_args.parse_args(opts)

        assert list(runner_select_args.runners.items()) == expected_runners
