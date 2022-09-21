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

import glob
import os
import tempfile

import pytest
from polygraphy import util
from polygraphy.backend.onnx import onnx_from_path
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, OnnxLoadArgs, OnnxSaveArgs, OnnxInferShapesArgs
from polygraphy.tools.script import Script
from tests.helper import is_file_empty, is_file_non_empty
from tests.models.meta import ONNX_MODELS
from tests.tools.args.helper import ArgGroupTestHelper


def _check_ext_weights_model(model):
    assert len(model.graph.node) == 3
    for init in model.graph.initializer:
        assert init.data_location != 1


class TestOnnxLoaderArgs:
    def test_basic(self):
        arg_group = ArgGroupTestHelper(OnnxLoadArgs(), deps=[ModelArgs(), OnnxInferShapesArgs()])
        arg_group.parse_args([ONNX_MODELS["identity_identity"].path, "--onnx-outputs=identity_out_0"])
        model = arg_group.load_onnx()

        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == "identity_out_0"

    def test_external_data(self):
        arg_group = ArgGroupTestHelper(OnnxLoadArgs(), deps=[ModelArgs(), OnnxInferShapesArgs()])
        model = ONNX_MODELS["ext_weights"]
        arg_group.parse_args([model.path, "--external-data-dir", model.ext_data])
        model = arg_group.load_onnx()
        _check_ext_weights_model(model)

    def test_ignore_external_data(self):
        arg_group = ArgGroupTestHelper(OnnxLoadArgs(), deps=[ModelArgs(), OnnxInferShapesArgs()])
        model = ONNX_MODELS["ext_weights"]
        arg_group.parse_args([model.path, "--ignore-external-data"])
        model = arg_group.load_onnx()
        assert all(init.data_location == 1 for init in model.graph.initializer)

    @pytest.mark.parametrize("allow_onnxruntime", [True, False])
    def test_shape_inference(self, allow_onnxruntime):
        # When using shape inference, we should load directly from the path
        arg_group = ArgGroupTestHelper(OnnxLoadArgs(), deps=[ModelArgs(), OnnxInferShapesArgs()])
        model = ONNX_MODELS["identity"]
        arg_group.parse_args(
            [model.path, "--shape-inference"] + (["--no-onnxruntime-shape-inference"] if not allow_onnxruntime else [])
        )

        assert arg_group.must_use_onnx_loader()

        script = Script()
        arg_group.add_to_script(script)

        expected_loader = (
            f"InferShapes({repr(model.path)})"
            if allow_onnxruntime
            else f"InferShapes({repr(model.path)}, allow_onnxruntime=False)"
        )
        assert expected_loader in str(script)

    @pytest.mark.parametrize("allow_onnxruntime", [True, False])
    def test_shape_inference_ext_data(self, allow_onnxruntime):
        arg_group = ArgGroupTestHelper(OnnxLoadArgs(), deps=[ModelArgs(), OnnxInferShapesArgs()])
        model = ONNX_MODELS["ext_weights"]
        arg_group.parse_args(
            [model.path, "--external-data-dir", model.ext_data, "--shape-inference"]
            + (["--no-onnxruntime-shape-inference"] if not allow_onnxruntime else [])
        )

        assert arg_group.must_use_onnx_loader()

        script = Script()
        arg_group.add_to_script(script)

        expected_loader = (
            f"InferShapes({repr(model.path)}, external_data_dir={repr(model.ext_data)})"
            if allow_onnxruntime
            else f"InferShapes({repr(model.path)}, external_data_dir={repr(model.ext_data)}, allow_onnxruntime=False)"
        )
        assert expected_loader in str(script)

        model = arg_group.load_onnx()
        _check_ext_weights_model(model)


class TestOnnxSaveArgs:
    def test_defaults(self):
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoadArgs(allow_shape_inference=False)])
        arg_group.parse_args([])
        assert arg_group.size_threshold is None

    def test_external_data(self):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoadArgs(allow_shape_inference=False)])
        with util.NamedTemporaryFile() as path, util.NamedTemporaryFile() as data:
            arg_group.parse_args(
                ["-o", path.name, "--save-external-data", data.name, "--external-data-size-threshold=0"]
            )
            arg_group.save_onnx(model)

            assert is_file_non_empty(path.name)
            assert is_file_non_empty(data.name)

    def test_size_threshold(self):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoadArgs(allow_shape_inference=False)])
        with util.NamedTemporaryFile() as path, util.NamedTemporaryFile() as data:
            arg_group.parse_args(
                ["-o", path.name, "--save-external-data", data.name, "--external-data-size-threshold=1024"]
            )
            arg_group.save_onnx(model)

            assert is_file_non_empty(path.name)
            assert is_file_empty(data.name)

    def test_no_all_tensors_to_one_file(self):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoadArgs(allow_shape_inference=False)])
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
        arg_group = ArgGroupTestHelper(OnnxSaveArgs(), deps=[ModelArgs(), OnnxLoadArgs(allow_shape_inference=False)])
        arg_group.parse_args(["--external-data-size-threshold", arg])
        assert arg_group.size_threshold == expected


class TestOnnxShapeInferenceArgs:
    def test_shape_inference_disabled_on_fallback(self):
        arg_group = ArgGroupTestHelper(
            OnnxInferShapesArgs(default=True, allow_force_fallback=True), deps=[DataLoaderArgs()]
        )
        arg_group.parse_args([])
        assert arg_group.do_shape_inference

        arg_group.parse_args(["--force-fallback-shape-inference"])
        assert not arg_group.do_shape_inference

    @pytest.mark.parametrize("allow_onnxruntime", [True, False])
    def test_no_onnxruntime_shape_inference(self, allow_onnxruntime):
        arg_group = ArgGroupTestHelper(
            OnnxInferShapesArgs(default=True, allow_force_fallback=True), deps=[DataLoaderArgs()]
        )
        arg_group.parse_args([] if allow_onnxruntime else ["--no-onnxruntime-shape-inference"])
        assert arg_group.allow_onnxruntime == (None if allow_onnxruntime else False)
