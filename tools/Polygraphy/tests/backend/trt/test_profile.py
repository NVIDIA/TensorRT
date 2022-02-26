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
from polygraphy import mod
from polygraphy.backend.trt import Profile, network_from_onnx_bytes
from tests.models.meta import ONNX_MODELS


@pytest.fixture(scope="session")
def dynamic_identity_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["dynamic_identity"].loader)
    with builder, network, parser:
        yield builder, network, parser


class TestProfile(object):
    def test_can_add(self):
        profile = Profile()
        min, opt, max = (1, 1), (2, 2), (4, 4)
        assert profile.add("input", min=min, opt=opt, max=max) is profile
        shape_tuple = profile["input"]
        assert shape_tuple.min == min
        assert shape_tuple.opt == opt
        assert shape_tuple.max == max

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_fill_defaults_does_not_overwrite(self, dynamic_identity_network):
        _, network, _ = dynamic_identity_network
        profile = Profile().add("X", (1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 3, 3))

        profile.fill_defaults(network) is profile
        assert profile["X"].min == (1, 1, 1, 1)
        assert profile["X"].opt == (1, 1, 2, 2)
        assert profile["X"].max == (1, 1, 3, 3)

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_to_trt(self, dynamic_identity_network):
        builder, network, _ = dynamic_identity_network
        profile = Profile().add("X", (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4))

        trt_profile = profile.to_trt(builder, network)
        trt_profile.get_shape("X") == ((1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4))
