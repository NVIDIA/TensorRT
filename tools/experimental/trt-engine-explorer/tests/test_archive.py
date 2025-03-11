#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tensorrt as trt
from trex.archiving import EngineArchive


def test_archiving():
    cdir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(cdir, "resources", "single_relu.onnx")
    tea = EngineArchive("/tmp/x1y2z3-test.tea")
    TRT_LOGGER = trt.Logger()
    with tea.Builder(TRT_LOGGER) as builder, builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_name, 'rb') as model:
            assert parser.parse(model.read())
        config = builder.create_builder_config()
        engine = builder.build_serialized_network(network, config)
        assert engine
        del engine
