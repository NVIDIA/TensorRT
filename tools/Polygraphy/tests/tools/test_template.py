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

import os
import shutil
import sys
import tempfile

import tensorrt as trt
from polygraphy import util
from polygraphy.backend.common import InvokeFromScript
from polygraphy.backend.trt import create_network
from tests.models.meta import ONNX_MODELS


class TestTrtNetwork:
    def test_no_model_file(self, poly_template):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            poly_template(["trt-network", "-o", template.name])

            load_network = InvokeFromScript(template.name, "load_network")
            builder, network = load_network()
            with builder, network:
                assert isinstance(builder, trt.Builder)
                assert isinstance(network, trt.INetworkDefinition)

    def test_with_model_file(self, poly_template):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            poly_template(["trt-network", ONNX_MODELS["identity"].path, "-o", template.name])

            load_network = InvokeFromScript(template.name, "load_network")
            builder, network, parser = load_network()
            with builder, network, parser:
                assert isinstance(builder, trt.Builder)
                assert isinstance(network, trt.INetworkDefinition)
                assert isinstance(parser, trt.OnnxParser)


class TestTrtConfig:
    def test_no_opts(self, poly_template):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            poly_template(["trt-config", "-o", template.name])

            builder, network = create_network()
            create_config = InvokeFromScript(template.name, "load_config")
            with builder, network, create_config(builder, network) as config:
                assert isinstance(config, trt.IBuilderConfig)

    def test_opts_basic(self, poly_template):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            poly_template(["trt-config", "--fp16", "--int8", "-o", template.name])

            builder, network = create_network()
            create_config = InvokeFromScript(template.name, "load_config")
            with builder, network, create_config(builder, network) as config:
                assert isinstance(config, trt.IBuilderConfig)
                assert config.get_flag(trt.BuilderFlag.FP16)
                assert config.get_flag(trt.BuilderFlag.INT8)


class TestOnnxGs:
    def test_basic(self, poly_template, sandboxed_install_run):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy a model into the temporary directory since the default output path
            # will write the new model into the same directory as the original.
            model_path = os.path.join(tmp_dir, "model.onnx")
            shutil.copyfile(ONNX_MODELS["identity"].path, model_path)

            template_path = os.path.join(tmp_dir, "process_model.py")
            poly_template(["onnx-gs", model_path, "-o", template_path])

            status = sandboxed_install_run([sys.executable, template_path], cwd=tmp_dir)
            assert status.success

            outpath = os.path.join(tmp_dir, "model_updated.onnx")
            assert os.path.exists(outpath)
