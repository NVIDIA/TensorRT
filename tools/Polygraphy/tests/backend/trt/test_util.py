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

import pytest
import tensorrt as trt
from polygraphy.backend.trt import util as trt_util


@pytest.fixture(scope="session")
def layer_class_mapping():
    return trt_util.get_layer_class_mapping()


@pytest.mark.parametrize("layer_type", trt.LayerType.__members__.values())
def test_all_layer_types_mapped(layer_class_mapping, layer_type):
    if layer_type == trt.LayerType.PLUGIN:
        pytest.skip("PLUGIN has no corresponding ILayer")
    assert layer_type in layer_class_mapping
