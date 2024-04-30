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

import copy
import os
import sys
from textwrap import dedent

import pytest
import tempfile
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.exception import PolygraphyException
from polygraphy.mod.importer import _version_ok

common_backend = mod.lazy_import("polygraphy.backend.common")

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

    def test_import_from_script_same_method_different_modules(self):
        module1_script = dedent(
            """
            def print_message():
                print(f"msg1::print_message")
                return "msg1"
            """
        )

        module2_script = dedent(
            """
            def print_message():
                print(f"msg2::print_message")
                return "msg2"
            """
        )

        with tempfile.TemporaryDirectory() as tempdir:
            os.mkdir(os.path.join(tempdir, "msg1"))
            with open(os.path.join(tempdir, "msg1", "msg.py"), "w+") as msg1_msg:
                msg1_msg.write(module1_script)
                msg1_msg.flush()
                os.fsync(msg1_msg.fileno())

                os.mkdir(os.path.join(tempdir, "msg2"))
                with open(os.path.join(tempdir, "msg2", "msg.py"), "w+") as msg2_msg:
                    msg2_msg.write(module2_script)
                    msg2_msg.flush()
                    os.fsync(msg2_msg.fileno())
                    
                    for msg_module in ['msg1', 'msg2']:
                        msg_loc = os.path.join(tempdir,msg_module,'msg.py')
                        msg = common_backend.invoke_from_script(msg_loc, "print_message")
                        assert msg==msg_module

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

            with pytest.raises(
                PolygraphyException, match="Could not import symbol: non_existent from"
            ):
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

    def test_is_installed_works_when_package_name_differs_from_module_name(
        self, poly_venv
    ):
        assert "onnxruntime" not in poly_venv.installed_packages()
        assert "onnxruntime-gpu" not in poly_venv.installed_packages()

        poly_venv.run(
            [poly_venv.python, "-m", "pip", "install", "onnxruntime-gpu", "--no-deps"]
        )

        # The `onnxruntime-gpu` package provides the `onnxruntime` module.
        # `is_installed()` should be able to understand that.
        poly_venv.run(
            [
                poly_venv.python,
                "-c",
                "from polygraphy import mod; onnxrt = mod.lazy_import('onnxruntime<0'); assert onnxrt.is_installed()",
            ]
        )

    @pytest.mark.parametrize(
        "mod_check", ["mod.has_mod('colored')", "colored.is_installed()"]
    )
    def test_has_mod(self, poly_venv, mod_check):
        assert "colored" not in poly_venv.installed_packages()
        poly_venv.run(
            [
                poly_venv.python,
                "-c",
                f"from polygraphy import mod; colored = mod.lazy_import('colored'); assert not {mod_check}",
            ]
        )

        poly_venv.run([poly_venv.python, "-m", "pip", "install", "colored==1.4.0"])
        # Make sure `has_mod` doesn't actually import the package.
        poly_venv.run(
            [
                poly_venv.python,
                "-c",
                "from polygraphy import mod; import sys; assert mod.has_mod('colored'); assert 'colored' not in sys.modules; import colored; assert 'colored' in sys.modules",
            ]
        )
