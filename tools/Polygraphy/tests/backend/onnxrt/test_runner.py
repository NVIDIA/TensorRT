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
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnxBytes
from polygraphy.logger import G_LOGGER

from tests.models.meta import ONNX_MODELS

import pytest


class TestLoggerCallbacks(object):
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.severity = sev


class TestOnnxrtRunner(object):
    def test_can_name_runner(self):
        NAME = "runner"
        runner = OnnxrtRunner(None, name=NAME)
        assert runner.name == NAME


    def test_basic(self):
        model = ONNX_MODELS["identity"]
        with OnnxrtRunner(SessionFromOnnxBytes(model.loader)) as runner:
            assert runner.is_active
            model.check_runner(runner)
        assert not runner.is_active


    def test_dim_param_converted_to_int_shape(self):
        model = ONNX_MODELS["dim_param"]
        with OnnxrtRunner(SessionFromOnnxBytes(model.loader)) as runner:
            input_meta = runner.get_input_metadata()
            # In Polygraphy, we only use None to indicate a dynamic input dimension - not strings.
            for name, (dtype, shape) in input_meta.items():
                for dim in shape:
                    assert dim is None or isinstance(dim, int)
