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
from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import util
from polygraphy.backend.common import InvokeFromScript, invoke_from_script
from polygraphy.exception import PolygraphyException


class TestImporter:
    @pytest.mark.parametrize("loader", [InvokeFromScript, invoke_from_script])
    def test_import_from_script(self, loader):
        script = dedent(
            """
            from polygraphy.backend.trt import CreateNetwork
            from polygraphy import func
            import tensorrt as trt

            @func.extend(CreateNetwork())
            def load_network(builder, network):
                inp = network.add_input("input", dtype=trt.float32, shape=(1, 1))
                out = network.add_identity(inp).get_output(0)
                network.mark_output(out)
            """
        )

        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())

            if loader == InvokeFromScript:
                load_network = loader(f.name, "load_network")
                builder, network = load_network()
            else:
                builder, network = loader(f.name, "load_network")
            with builder, network:
                assert isinstance(builder, trt.Builder)
                assert isinstance(network, trt.INetworkDefinition)
                assert network.num_layers == 1
                assert network.get_layer(0).type == trt.LayerType.IDENTITY

    def test_import_non_existent(self):
        script = dedent(
            """
            def example():
                pass
            """
        )

        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())

            with pytest.raises(PolygraphyException, match="Could not import symbol: non_existent from"):
                invoke_from_script(f.name, "non_existent")
