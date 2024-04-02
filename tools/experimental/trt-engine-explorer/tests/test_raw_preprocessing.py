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
from trex import import_graph_file
import pytest

def test_import_graph_file():
    graph_file = f'{test_engine1_prefix_path}.graph.json'
    raw_layers, bindings = import_graph_file(graph_file)

    assert isinstance(raw_layers, list)
    # Check if no raw_layer['Name'] is repeated. If layer name has appeared
    # twice, it must be disambiguated
    names = [raw_layer['Name'] for raw_layer in raw_layers]
    assert len(set(names)) == len(names)

    #  Check if applicable layers have been converted to 'Deconvolution'
    assert sum([1 for raw_layer in raw_layers
        if raw_layer.get('LayerType') == 'CaskDeconvolutionV2' and
        raw_layer.get('ParameterType') == 'Convolution']) == 0

    raw_layer = raw_layers[0]
    assert raw_layer['Name'] == 'QuantizeLinear_2'
    assert raw_layer['LayerType'] == 'Reformat'
    assert raw_layer['ParameterType'] == 'Reformat'
    assert raw_layer['Origin'] == 'QDQ'
    assert raw_layer['TacticValue'] == '0x0000000000000000'

    assert isinstance(bindings, (list))
    assert bindings == ['input1', '1179']
