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

from polygraphy_trtexec.backend.runner import TrtexecRunner, TRTEXEC_DEFAULT_PATH
from tests.models.meta import ONNX_MODELS_PATH

class TestTrtRunner:

    def test_basic(self):
        with TrtexecRunner(ONNX_MODELS_PATH["identity"], model_type="onnx") as runner:
            assert runner.trtexec_path is TRTEXEC_DEFAULT_PATH
            assert runner.use_cuda_graph is None
            assert runner.avg_runs is None
            assert runner.best is None
            assert runner.duration is None
            assert runner.device is None
            assert runner.streams is None
            assert runner.min_timing is None
            assert runner.avg_timing is None
            assert runner.expose_dma is None
            assert runner.no_data_transfers is None
            assert runner.trtexec_warmup is None
            assert runner.trtexec_iterations is None
            assert runner.trtexec_export_times is None
            assert runner.trtexec_export_output is None
            assert runner.trtexec_export_profile is None
            assert runner.trtexec_export_layer_info is None
            assert runner.is_active
        assert not runner.is_active
