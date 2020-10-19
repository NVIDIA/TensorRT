#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from polygraphy.backend.onnx import OnnxTfRunner, OnnxFromPath, BytesFromOnnx

from tests.models.meta import ONNX_MODELS


class TestOnnxTfRunner(object):
    def test_can_name_runner(self):
        NAME = "runner"
        runner = OnnxTfRunner(None, name=NAME)
        assert runner.name == NAME


    def test_basic(self):
        model = ONNX_MODELS["identity"]
        with OnnxTfRunner(OnnxFromPath(model.path)) as runner:
            assert runner.is_active
            model.check_runner(runner)
        assert not runner.is_active
