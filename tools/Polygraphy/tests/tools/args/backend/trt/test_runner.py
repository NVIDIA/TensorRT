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
    ModelArgs,
    OnnxLoadArgs,
    TrtConfigArgs,
    TrtLoadEngineArgs,
    TrtLoadNetworkArgs,
    TrtLoadPluginsArgs,
    TrtRunnerArgs,
)
from polygraphy.tools.script import Script
from tests.models.meta import ONNX_MODELS
from tests.tools.args.helper import ArgGroupTestHelper
from polygraphy.tools.args import util as args_util


@pytest.fixture()
def trt_runner_args():
    return ArgGroupTestHelper(
        TrtRunnerArgs(),
        deps=[
            ModelArgs(),
            TrtLoadEngineArgs(),
            OnnxLoadArgs(allow_shape_inference=False),
            TrtConfigArgs(),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(),
        ],
    )


class TestTrtRunnerArgs:
    @pytest.mark.parametrize("index", range(0, 3))
    def test_optimization_profile(self, trt_runner_args, index):
        trt_runner_args.parse_args([ONNX_MODELS["identity"].path, f"--optimization-profile={index}"])

        assert trt_runner_args.optimization_profile == index

        script = Script()
        runners = args_util.run_script(trt_runner_args.add_to_script)

        assert runners[0].optimization_profile == index
