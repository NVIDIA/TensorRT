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

import tensorrt as trt
from polygraphy import util
from polygraphy.backend.common import InvokeFromScript
from polygraphy.backend.trt import create_network
from tests.models.meta import ONNX_MODELS
from tests.tools.common import run_polygraphy_template


class TestTrtNetwork(object):
    def test_no_model_file(self):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            run_polygraphy_template(["trt-network", "-o", template.name])

            load_network = InvokeFromScript(template.name, "load_network")
            builder, network = load_network()
            with builder, network:
                assert isinstance(builder, trt.Builder)
                assert isinstance(network, trt.INetworkDefinition)

    def test_with_model_file(self):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            run_polygraphy_template(["trt-network", ONNX_MODELS["identity"].path, "-o", template.name])

            load_network = InvokeFromScript(template.name, "load_network")
            builder, network, parser = load_network()
            with builder, network, parser:
                assert isinstance(builder, trt.Builder)
                assert isinstance(network, trt.INetworkDefinition)
                assert isinstance(parser, trt.OnnxParser)


class TestTrtConfig(object):
    def test_no_opts(self):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            run_polygraphy_template(["trt-config", "-o", template.name])

            builder, network = create_network()
            create_config = InvokeFromScript(template.name, "load_config")
            with builder, network, create_config(builder, network) as config:
                assert isinstance(config, trt.IBuilderConfig)

    def test_opts_basic(self):
        with util.NamedTemporaryFile("w+", suffix=".py") as template:
            run_polygraphy_template(["trt-config", "--fp16", "--int8", "-o", template.name])

            builder, network = create_network()
            create_config = InvokeFromScript(template.name, "load_config")
            with builder, network, create_config(builder, network) as config:
                assert isinstance(config, trt.IBuilderConfig)
                assert config.get_flag(trt.BuilderFlag.FP16)
                assert config.get_flag(trt.BuilderFlag.INT8)
