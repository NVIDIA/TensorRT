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
from polygraphy.backend.onnx import GsFromOnnx, OnnxFromPath
from polygraphy.backend.pluginref import PluginRefRunner
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER
from tests.models.meta import ONNX_MODELS


class TestLoggerCallbacks:
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.module_severity = sev


class TestPluginRefRunner:
    def test_can_name_runner(self):
        NAME = "runner"
        runner = PluginRefRunner(None, name=NAME)
        assert runner.name == NAME

    def test_basic(self):
        model = ONNX_MODELS["identity"]
        with PluginRefRunner(GsFromOnnx(OnnxFromPath(model.path))) as runner:
            assert runner.is_active
            model.check_runner(runner)
        assert not runner.is_active

    def test_works_on_multiple_nodes(self):
        model = ONNX_MODELS["identity_identity"]
        with PluginRefRunner(GsFromOnnx(OnnxFromPath(model.path))) as runner:
            model.check_runner(runner)

    def test_fail_on_unsupported_node(self):
        model = ONNX_MODELS["and"]
        with PluginRefRunner(GsFromOnnx(OnnxFromPath(model.path))) as runner:
            with pytest.raises(PolygraphyException, match="does not have a reference implementation registered!"):
                runner.infer({"x": np.ones(shape=(3, 4), dtype=bool), "y": np.ones(shape=(3, 4), dtype=bool)})

    @pytest.mark.parametrize(
        "names, err",
        [
            (["fake-input", "x"], "Extra inputs in"),
            (["fake-input"], "The following inputs were not found"),
            ([], "The following inputs were not found"),
        ],
    )
    def test_error_on_wrong_name_feed_dict(self, names, err):
        model = ONNX_MODELS["identity"]
        with PluginRefRunner(GsFromOnnx(OnnxFromPath(model.path))) as runner:
            with pytest.raises(PolygraphyException, match=err):
                runner.infer({name: np.ones(shape=(1, 1, 2, 2), dtype=np.float32) for name in names})

    def test_error_on_wrong_dtype_feed_dict(self):
        model = ONNX_MODELS["identity"]
        with PluginRefRunner(GsFromOnnx(OnnxFromPath(model.path))) as runner:
            with pytest.raises(PolygraphyException, match="unexpected dtype."):
                runner.infer({"x": np.ones(shape=(1, 1, 2, 2), dtype=np.int32)})

    def test_error_on_wrong_shape_feed_dict(self):
        model = ONNX_MODELS["identity"]
        with PluginRefRunner(GsFromOnnx(OnnxFromPath(model.path))) as runner:
            with pytest.raises(PolygraphyException, match="incompatible shape."):
                runner.infer({"x": np.ones(shape=(1, 1, 3, 2), dtype=np.float32)})
