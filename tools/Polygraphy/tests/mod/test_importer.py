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

import copy
import os
import sys
from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.exception import PolygraphyException
from polygraphy.mod.importer import _version_ok


class TestImporter:
    def test_import_from_script(self):
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

            orig_sys_path = copy.deepcopy(sys.path)
            load_network = mod.import_from_script(f.name, "load_network")
            assert sys.path == orig_sys_path
            builder, network = load_network()
            with builder, network:
                assert isinstance(builder, trt.Builder)
                assert isinstance(network, trt.INetworkDefinition)
                assert network.num_layers == 1
                assert network.get_layer(0).type == trt.LayerType.IDENTITY
            assert sys.path == orig_sys_path

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

            orig_sys_path = copy.deepcopy(sys.path)
            example = mod.import_from_script(f.name, "example")
            assert sys.path == orig_sys_path

            assert example is not None
            example()

            with pytest.raises(PolygraphyException, match="Could not import symbol: non_existent from"):
                mod.import_from_script(f.name, "non_existent")
            assert sys.path == orig_sys_path

    @pytest.mark.parametrize(
        "ver, pref, expected",
        [
            ("0.0.0", "==0.0.0", True),
            ("0.0.0", "== 0.0.1", False),
            ("0.0.0", ">= 0.0.0", True),
            ("0.0.0", ">=0.0.1", False),
            ("0.0.0", "<= 0.0.0", True),
            ("0.0.2", "<=0.0.1", False),
            ("0.0.1", "> 0.0.0", True),
            ("0.0.1", ">0.0.1", False),
            ("0.0.0", "< 0.0.1", True),
            ("0.0.0", "< 0.0.0", False),
            ("0.2.0", mod.LATEST_VERSION, False),
        ],
    )
    def test_version_ok(self, ver, pref, expected):
        assert _version_ok(ver, pref) == expected
