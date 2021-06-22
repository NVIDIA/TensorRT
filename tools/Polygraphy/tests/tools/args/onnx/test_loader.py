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

import tempfile

from polygraphy.backend.onnx import onnx_from_path
from polygraphy.tools.args import (DataLoaderArgs, ModelArgs, OnnxLoaderArgs,
                                   OnnxSaveArgs, OnnxShapeInferenceArgs)
from tests.helper import check_file_non_empty
from tests.models.meta import ONNX_MODELS
from tests.tools.args.helper import ArgGroupTestHelper


class TestOnnxLoaderArgs(object):
    def test_basic(self):
        arg_group = ArgGroupTestHelper(OnnxLoaderArgs(), deps=[ModelArgs()])
        arg_group.parse_args([ONNX_MODELS["identity_identity"].path, "--onnx-outputs=identity_out_0"])
        model = arg_group.load_onnx()

        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == "identity_out_0"


    def test_external_data(self):
        arg_group = ArgGroupTestHelper(OnnxLoaderArgs(), deps=[ModelArgs()])
        model = ONNX_MODELS["ext_weights"]
        arg_group.parse_args([model.path, "--load-external-data", model.ext_data])
        model = arg_group.load_onnx()

        assert len(model.graph.node) == 3


class TestOnnxSaveArgs(object):
    def test_external_data(self):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoaderArgs()])
        with tempfile.NamedTemporaryFile() as path, tempfile.NamedTemporaryFile() as data:
            arg_group.parse_args(["-o", path.name, "--save-external-data", data.name])
            arg_group.save_onnx(model)

            check_file_non_empty(path.name)
            check_file_non_empty(data.name)


class TestOnnxShapeInferenceArgs(object):
    def test_shape_inference_disabled_on_fallback(self):
        arg_group = ArgGroupTestHelper(OnnxShapeInferenceArgs(default=True, enable_force_fallback=True), deps=[DataLoaderArgs()])
        arg_group.parse_args([])
        assert arg_group.do_shape_inference

        arg_group.parse_args(["--force-fallback-shape-inference"])
        assert not arg_group.do_shape_inference
