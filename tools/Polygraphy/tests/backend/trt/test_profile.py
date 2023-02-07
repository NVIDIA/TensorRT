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

import numpy as np
import pytest
import tensorrt as trt
from polygraphy.backend.trt import Profile, create_network, network_from_onnx_bytes
from tests.models.meta import ONNX_MODELS


@pytest.fixture(scope="session")
def dynamic_identity_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["dynamic_identity"].loader)
    with builder, network, parser:
        yield builder, network, parser


class TestProfile:
    def test_can_add(self):
        profile = Profile()
        min, opt, max = (1, 1), (2, 2), (4, 4)
        assert profile.add("input", min=min, opt=opt, max=max) is profile
        shape_tuple = profile["input"]
        assert shape_tuple.min == min
        assert shape_tuple.opt == opt
        assert shape_tuple.max == max

    def test_fill_defaults_does_not_overwrite(self, dynamic_identity_network):
        _, network, _ = dynamic_identity_network
        profile = Profile().add("X", (1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 3, 3))

        assert profile.fill_defaults(network) is profile
        assert profile["X"].min == (1, 1, 1, 1)
        assert profile["X"].opt == (1, 1, 2, 2)
        assert profile["X"].max == (1, 1, 3, 3)

    def test_fill_defaults_scalar_shape_tensor(self):
        _, network = create_network()
        fill_shape = network.add_input("fill_shape", shape=tuple(), dtype=trt.int32)

        # Need to add some other operations so TensorRT treats `fill_shape` as a shape tensor.
        fill = network.add_fill(tuple(), trt.FillOperation.LINSPACE)
        fill.set_input(0, fill_shape)
        fill.set_input(1, network.add_constant(shape=tuple(), weights=np.array(0).astype(np.int32)).get_output(0))
        fill.set_input(2, network.add_constant(shape=tuple(), weights=np.array(1).astype(np.int32)).get_output(0))

        network.mark_output(fill.get_output(0))

        assert fill_shape.is_shape_tensor

        profile = Profile()
        profile.fill_defaults(network)

        assert profile[fill_shape.name].min == (1,)
        assert profile[fill_shape.name].opt == (1,)
        assert profile[fill_shape.name].max == (1,)

    def test_to_trt(self, dynamic_identity_network):
        builder, network, _ = dynamic_identity_network
        profile = Profile().add("X", (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4))

        trt_profile = profile.to_trt(builder, network)
        trt_profile.get_shape("X") == ((1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4))
