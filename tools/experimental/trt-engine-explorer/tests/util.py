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


from trex import EnginePlan
import pytest
import os
import numpy as np
import json
import trex
from graphviz.graphs import Digraph

test_dir_path = os.path.dirname(os.path.realpath(__file__))
test_engine1 = "mobilenet.qat.onnx.engine"
test_engine2 = "mobilenet_v2_residuals.qat.onnx.engine"
test_engine1_prefix_path = os.path.join(test_dir_path, 'inputs', test_engine1)
test_engine2_prefix_path = os.path.join(test_dir_path, 'inputs', test_engine2)

@pytest.fixture
def plan():
    plan = EnginePlan(f'{test_engine1_prefix_path}.graph.json', f'{test_engine1_prefix_path}.profile.json')
    return plan

@pytest.fixture
def plan2():
    plan = EnginePlan(f'{test_engine2_prefix_path}.graph.json', f'{test_engine2_prefix_path}.profile.json')
    return plan

class NpEncoder(json.JSONEncoder):
    """
    Custom Encoder to dump numpy objects
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(NpEncoder, self).default(obj)

def dump_recurse(item):
    """
    Recursively open objects to basic builtin data-types
    """
    if item is None:
        return json.dumps(item)

    if type(item).__module__ == "numpy":
        return json.dumps(item, cls=NpEncoder)

    elif isinstance(item, (bool, str, int, float)):
        return json.dumps(item)

    elif isinstance(item, (tuple)):
        return [dump_recurse(i) for i in tuple(item)]

    elif isinstance(item, (list)):
        return [dump_recurse(i) for i in item]

    elif isinstance(item, (dict)):
        return {k: dump_recurse(v) for k, v in item.items()}

    elif isinstance(item, (trex.PlanGraph, trex.DotGraph, trex.Region, trex.RegionGeneration, trex.Activation, trex.LayerNode, trex.Layer, Digraph)):
        return dump_recurse(item.__dict__)

    return json.dumps(dump_recurse(item.__dict__))
