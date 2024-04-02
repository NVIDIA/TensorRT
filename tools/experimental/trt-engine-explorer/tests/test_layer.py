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


from .util import test_engine1_prefix_path
from trex import import_graph_file, Layer, Activation


class TestLayer:
    def test_initialization(self):
        graph_file = f'{test_engine1_prefix_path}.graph.json'
        raw_layers, _ = import_graph_file(graph_file)
        layer = Layer(raw_layers[0])

        assert layer.name == 'QuantizeLinear_2'
        assert layer.type == 'Reformat'
        assert layer.subtype == 'Reformat'
        assert isinstance(layer.inputs[0], (Activation))
        assert isinstance(layer.outputs[0], (Activation))
        assert layer.outputs_size_bytes == 150528
        assert layer.precision == 'FP32'
        assert layer.inputs_size_bytes == 602112
        assert layer.total_io_size_bytes == 752640
        assert layer.total_footprint_bytes == 752640
