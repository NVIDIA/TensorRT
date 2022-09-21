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
    TrtConfigArgs,
    TrtLoadPluginsArgs,
    TrtLoadNetworkArgs,
    TrtSaveEngineArgs,
)
from polygraphy.tools.script import Script
from tests.models.meta import ONNX_MODELS_PATH
from tests.helper import *
from polygraphy.tools.args import util as args_util
from polygraphy_trtexec.args import TrtexecRunnerArgs

@pytest.fixture()
def trtexec_runner_args():
    return ArgGroupTestHelper(
        TrtexecRunnerArgs(),
        deps=[
            ModelArgs(),
            TrtConfigArgs(),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(),
            TrtSaveEngineArgs(),
        ],
    )


class TestTrtexecRunnerArgs:
    @pytest.mark.parametrize("trtexec_path_params", TRTEXEC_PATH_PARAMS)
    def test_trtexec_path(self, trtexec_runner_args, trtexec_path_params):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--trtexec-path={trtexec_path_params}"])
        assert trtexec_runner_args.trtexec_path == trtexec_path_params

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].trtexec_path == trtexec_path_params

    def test_use_cuda_graph(self, trtexec_runner_args):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--use-cuda-graph"])
        assert trtexec_runner_args.use_cuda_graph

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].use_cuda_graph

    @pytest.mark.parametrize("num_avg_runs", range(1, 16, 5))
    def test_avg_runs(self, trtexec_runner_args, num_avg_runs):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--avg-runs={num_avg_runs}"])
        assert trtexec_runner_args.avg_runs == num_avg_runs

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].avg_runs == num_avg_runs

    def test_best(self, trtexec_runner_args):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--best"])
        assert trtexec_runner_args.best

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].best

    @pytest.mark.parametrize("num_duration", range(1, 16, 5))
    def test_duration(self, trtexec_runner_args, num_duration):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--duration={num_duration}"])
        assert trtexec_runner_args.duration == num_duration

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].duration == num_duration

    @pytest.mark.parametrize("num_device", range(0, 3))
    def test_device(self, trtexec_runner_args, num_device):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--device={num_device}"])
        assert trtexec_runner_args.device == num_device

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].device == num_device

    @pytest.mark.parametrize("num_streams", range(1, 4))
    def test_streams(self, trtexec_runner_args, num_streams):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--streams={num_streams}"])
        assert trtexec_runner_args.streams == num_streams

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].streams == num_streams

    @pytest.mark.parametrize("num_min_timing", range(1, 4))
    def test_min_timing(self, trtexec_runner_args, num_min_timing):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--min-timing={num_min_timing}"])
        assert trtexec_runner_args.min_timing == num_min_timing

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].min_timing == num_min_timing

    @pytest.mark.parametrize("num_avg_timing", range(1, 4))
    def test_avg_timing(self, trtexec_runner_args, num_avg_timing):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--avg-timing={num_avg_timing}"])
        assert trtexec_runner_args.avg_timing == num_avg_timing

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].avg_timing == num_avg_timing

    def test_expose_dma(self, trtexec_runner_args):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--expose-dma"])
        assert trtexec_runner_args.expose_dma

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].expose_dma

    def test_no_data_transfers(self, trtexec_runner_args):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--no-data-transfers"])
        assert trtexec_runner_args.no_data_transfers

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].no_data_transfers

    @pytest.mark.parametrize("num_warmup", range(100, 400, 100))
    def test_trtexec_warmup(self, trtexec_runner_args, num_warmup):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--trtexec-warmup={num_warmup}"])
        assert trtexec_runner_args.trtexec_warmup == num_warmup

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].trtexec_warmup == num_warmup

    @pytest.mark.parametrize("num_iterations", range(100, 400, 100))
    def test_trtexec_iterations(self, trtexec_runner_args, num_iterations):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--trtexec-iterations={num_iterations}"])
        assert trtexec_runner_args.trtexec_iterations == num_iterations

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].trtexec_iterations == num_iterations

    @pytest.mark.parametrize("trtexec_export_times_params", TRTEXEC_EXPORT_TIMES_PARAMS)
    def test_trtexec_export_times(self, trtexec_runner_args, trtexec_export_times_params):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--trtexec-export-times={trtexec_export_times_params}"])
        assert trtexec_runner_args.trtexec_export_times == trtexec_export_times_params

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].trtexec_export_times == trtexec_export_times_params

    @pytest.mark.parametrize("trtexec_export_output_params", TRTEXEC_EXPORT_OUTPUT_PARAMS)
    def test_trtexec_export_output_params(self, trtexec_runner_args, trtexec_export_output_params):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--trtexec-export-output={trtexec_export_output_params}"])
        assert trtexec_runner_args.trtexec_export_output == trtexec_export_output_params

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].trtexec_export_output == trtexec_export_output_params

    @pytest.mark.parametrize("trtexec_export_profile_params", TRTEXEC_EXPORT_PROFILE_PARAMS)
    def test_trtexec_export_profile_params(self, trtexec_runner_args, trtexec_export_profile_params):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--trtexec-export-profile={trtexec_export_profile_params}"])
        assert trtexec_runner_args.trtexec_export_profile == trtexec_export_profile_params

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].trtexec_export_profile == trtexec_export_profile_params

    @pytest.mark.parametrize("trtexec_export_layer_info_params", TRTEXEC_EXPORT_LAYER_INFO_PARAMS)
    def test_trtexec_export_layer_info_params(self, trtexec_runner_args, trtexec_export_layer_info_params):
        trtexec_runner_args.parse_args([ONNX_MODELS_PATH["identity"], f"--trtexec-export-layer-info={trtexec_export_layer_info_params}"])
        assert trtexec_runner_args.trtexec_export_layer_info == trtexec_export_layer_info_params

        script = Script()
        runners = args_util.run_script(trtexec_runner_args.add_to_script)
        assert runners[0].trtexec_export_layer_info == trtexec_export_layer_info_params
