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

import glob
import os
import tempfile

import pytest
from polygraphy import util
from polygraphy.backend.onnx import onnx_from_path
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, OnnxLoaderArgs, OnnxSaveArgs, OnnxShapeInferenceArgs
from polygraphy.tools.script import Script
from tests.helper import is_file_empty, is_file_non_empty
from tests.models.meta import ONNX_MODELS
from tests.tools.args.helper import ArgGroupTestHelper


def _check_ext_weights_model(model):
    assert len(model.graph.node) == 3
    for init in model.graph.initializer:
        assert init


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
        arg_group.parse_args([model.path, "--external-data-dir", model.ext_data])
        model = arg_group.load_onnx()
        _check_ext_weights_model(model)

    def test_shape_inference(self):
        # When using shape inference, we should load directly from the path
        arg_group = ArgGroupTestHelper(OnnxLoaderArgs(), deps=[ModelArgs(), OnnxShapeInferenceArgs()])
        model = ONNX_MODELS["identity"]
        arg_group.parse_args([model.path, "--shape-inference"])

        assert arg_group.should_use_onnx_loader()

        script = Script()
        arg_group.add_onnx_loader(script)

        expected_loader = "InferShapes({:})".format(repr(model.path))
        assert expected_loader in str(script)

    def test_shape_inference_ext_data(self):
        arg_group = ArgGroupTestHelper(OnnxLoaderArgs(), deps=[ModelArgs(), OnnxShapeInferenceArgs()])
        model = ONNX_MODELS["ext_weights"]
        arg_group.parse_args([model.path, "--external-data-dir", model.ext_data, "--shape-inference"])

        assert arg_group.should_use_onnx_loader()

        script = Script()
        arg_group.add_onnx_loader(script)

        expected_loader = "InferShapes({:}, external_data_dir={:})".format(repr(model.path), repr(model.ext_data))
        assert expected_loader in str(script)

        model = arg_group.load_onnx()
        _check_ext_weights_model(model)


class TestOnnxSaveArgs(object):
    def test_defaults(self):
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoaderArgs()])
        arg_group.parse_args([])
        assert arg_group.size_threshold is None

    def test_external_data(self):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoaderArgs()])
        with util.NamedTemporaryFile() as path, util.NamedTemporaryFile() as data:
            arg_group.parse_args(
                ["-o", path.name, "--save-external-data", data.name, "--external-data-size-threshold=0"]
            )
            arg_group.save_onnx(model)

            assert is_file_non_empty(path.name)
            assert is_file_non_empty(data.name)

    def test_size_threshold(self):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoaderArgs()])
        with util.NamedTemporaryFile() as path, util.NamedTemporaryFile() as data:
            arg_group.parse_args(
                ["-o", path.name, "--save-external-data", data.name, "--external-data-size-threshold=1024"]
            )
            arg_group.save_onnx(model)

            assert is_file_non_empty(path.name)
            assert is_file_empty(data.name)

    def test_no_all_tensors_to_one_file(self):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoaderArgs()])
        with tempfile.TemporaryDirectory() as outdir:
            path = os.path.join(outdir, "model.onnx")
            arg_group.parse_args(
                [
                    "-o",
                    path,
                    "--save-external-data",
                    "--external-data-size-threshold=0",
                    "--no-save-all-tensors-to-one-file",
                ]
            )
            arg_group.save_onnx(model)

            assert is_file_non_empty(path)
            outfiles = glob.glob(os.path.join(outdir, "*"))
            assert len(outfiles) == 4

    @pytest.mark.parametrize(
        "arg, expected",
        [
            ("16", 16),
            ("1e9", 1e9),
            ("2M", 2 << 20),
        ],
    )
    def test_size_threshold_parsing(self, arg, expected):
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoaderArgs()])
        arg_group.parse_args(["--external-data-size-threshold", arg])
        assert arg_group.size_threshold == expected


class TestOnnxShapeInferenceArgs(object):
    def test_shape_inference_disabled_on_fallback(self):
        arg_group = ArgGroupTestHelper(
            OnnxShapeInferenceArgs(default=True, enable_force_fallback=True), deps=[DataLoaderArgs()]
        )
        arg_group.parse_args([])
        assert arg_group.do_shape_inference

        arg_group.parse_args(["--force-fallback-shape-inference"])
        assert not arg_group.do_shape_inference
