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
from polygraphy.backend.onnx import OnnxFromTfGraph, OnnxFromPath, ModifyOnnx, SaveOnnx
from polygraphy.backend.tf import GraphFromFrozen
from polygraphy.common import constants
from polygraphy.logger import G_LOGGER

from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.common import check_file_non_empty

import tempfile
import pytest
import os


class TestLoggerCallbacks(object):
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.severity = sev


class TestOnnxFileLoader(object):
    def test_basic(self):
        loader = OnnxFromPath(ONNX_MODELS["identity"].path)
        loader()


class TestExportOnnxFromTf(object):
    def test_no_optimize(self):
        loader = OnnxFromTfGraph(TF_MODELS["identity"].loader, optimize=False, fold_constant=False)
        model = loader()


    def test_opset(self):
        loader = OnnxFromTfGraph(TF_MODELS["identity"].loader, opset=9)
        model = loader()
        assert model.opset_import[0].version == 9


class TestModifyOnnx(object):
    def test_layerwise(self):
        loader = ModifyOnnx(OnnxFromPath(ONNX_MODELS["identity_identity"].path), outputs=constants.MARK_ALL)
        model = loader()
        assert len(model.graph.output) == 2


    def test_custom_outputs(self):
        loader = ModifyOnnx(OnnxFromPath(ONNX_MODELS["identity_identity"].path), outputs=["identity_out_0"])
        model = loader()
        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == "identity_out_0"


    def test_exclude_outputs_with_layerwise(self):
        loader = ModifyOnnx(OnnxFromPath(ONNX_MODELS["identity_identity"].path), outputs=constants.MARK_ALL, exclude_outputs=["identity_out_2"])
        model = loader()
        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == "identity_out_0"


class TestSaveOnnx(object):
    def test_save_onnx(self):
        with tempfile.NamedTemporaryFile() as outpath:
            loader = SaveOnnx(OnnxFromPath(ONNX_MODELS["identity"].path), path=outpath.name)
            loader()
            check_file_non_empty(outpath.name)
