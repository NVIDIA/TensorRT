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

            with pytest.raises(
                PolygraphyException, match="Could not import symbol: non_existent from"
            ):
                invoke_from_script(f.name, "non_existent")

    def test_invoke_class_is_instantiated(self):
        # When the imported symbol is a class, it is instantiated and the instance is invoked.
        script = dedent(
            """
            from collections import OrderedDict
            from polygraphy.comparator import BaseCompareFunc

            class MyCompareFunc(BaseCompareFunc):
                def __call__(self, iter_result0, iter_result1):
                    return OrderedDict(out=True)
            """
        )
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())

            result = InvokeFromScript(f.name, "MyCompareFunc")(None, None)
            assert dict(result) == {"out": True}

    def test_invoke_forwards_attributes(self):
        # Attribute access is forwarded to the loaded object, so a BaseCompareFunc loaded from a
        # script exposes its threshold API (this is what lets --compare-func-script work with
        # --check-average).
        script = dedent(
            """
            from collections import OrderedDict
            from polygraphy.comparator import BaseCompareFunc, L2Threshold

            class MyCompareFunc(BaseCompareFunc):
                def __call__(self, iter_result0, iter_result1):
                    return OrderedDict(out=True)
                def thresholds_for(self, output_name):
                    return L2Threshold(1.0)
            """
        )
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())

            compare_func = InvokeFromScript(f.name, "MyCompareFunc")
            assert compare_func.thresholds_for("out").metric_fields() == ["l2_norm"]

    def test_invoke_does_not_forward_missing_attributes(self):
        # A plain function does not expose a threshold API, so the forwarding raises (which lets
        # compare_accuracy reject it cleanly with check_average=True).
        script = dedent(
            """
            from collections import OrderedDict
            def compare_outputs(iter_result0, iter_result1):
                return OrderedDict(out=True)
            """
        )
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())

            compare_func = InvokeFromScript(f.name, "compare_outputs")
            assert not hasattr(compare_func, "thresholds_for")
