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

import glob
import os
import subprocess as sp

import sys
import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.mod.importer import _version_ok

from tests.helper import ALL_TOOLS, POLYGRAPHY_CMD, ROOT_DIR
from tests.models.meta import ONNX_MODELS

"""
The tests here ensure that no additional dependencies are introduced into
the various modules under Polygraphy.
"""


@pytest.fixture()
def poly_venv(virtualenv):
    virtualenv.env["PYTHONPATH"] = ROOT_DIR
    virtualenv.env["LD_LIBRARY_PATH"] = ""
    return virtualenv


def is_submodule(path):
    file_mod = os.path.isfile(path) and path.endswith(".py") and os.path.basename(path) != "__init__.py"
    dir_mod = os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py"))
    return file_mod or dir_mod


MODULE_PATH = os.path.join(ROOT_DIR, "polygraphy")
SUBMODULE_PATHS = [
    os.path.relpath(os.path.splitext(path)[0], ROOT_DIR)
    for path in glob.iglob(os.path.join(MODULE_PATH, "**"), recursive=True)
    if is_submodule(path)
]


class TestPublicImports:
    def test_no_extra_submodule_dependencies_required(self, poly_venv):
        # Submodules should not require any extra dependencies to import.
        for submodule_path in SUBMODULE_PATHS:
            submodule_name = ".".join(submodule_path.split(os.path.sep))
            cmd = [poly_venv.python, "-c", f"from {submodule_name} import *"]
            print(" ".join(cmd))
            output = poly_venv.run(cmd, capture=True)
            print(output)

    def test_can_json_without_numpy(self, poly_venv):
        cmd = [
            poly_venv.python,
            "-c",
            "from polygraphy.json import to_json, from_json; x = to_json(1); x = from_json(x)",
        ]
        print(" ".join(cmd))
        output = poly_venv.run(cmd, capture=True)
        print(output)


class TestToolImports:
    # We should be able to at least launch tools with no dependencies installed.
    @pytest.mark.parametrize("tool, subtools", ALL_TOOLS.items())
    def test_can_run_tool_without_deps(self, poly_venv, tool, subtools):
        BASE_TOOL_CMD = [poly_venv.python, *POLYGRAPHY_CMD, tool, "-h"]

        def check_tool(tool):
            output = poly_venv.run(tool, capture=True)
            assert "This tool could not be loaded due to an error:" not in output
            assert "error:" not in output
            assert "could not be loaded" not in output

        check_tool(BASE_TOOL_CMD)

        for subtool in subtools:
            check_tool(BASE_TOOL_CMD + [subtool])


class TestAutoinstallDeps:
    @pytest.mark.parametrize(
        "cmd",
        [
            ["run", ONNX_MODELS["identity"].path, "--onnxrt"],
            ["run", ONNX_MODELS["identity"].path, "--trt"],
            [
                "surgeon",
                "sanitize",
                "--fold-constants",
                ONNX_MODELS["const_foldable"].path,
                "-o",
                util.NamedTemporaryFile().name,
            ],
        ],
    )
    def test_can_automatically_install_deps(self, poly_venv, cmd):
        # WAR an issue with newer versions of protobuf and ONNX
        poly_venv.run([poly_venv.python, "-m", "pip", "install", "protobuf==3.19.4", "onnx==1.10.0"])

        poly_venv.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"
        cmd = [poly_venv.python, *POLYGRAPHY_CMD] + cmd
        print(f"Running: {' '.join(cmd)}")
        output = poly_venv.run(cmd, capture=True)
        print(output)
        assert "is required, but not installed. Attempting to install now" in output

    @pytest.mark.parametrize(
        "new_ver, expected",
        [
            ("==1.4.2", "==1.4.2"),
            (mod.LATEST_VERSION, ">=1.4.2"),
        ],
    )
    def test_can_automatically_upgrade_deps(self, poly_venv, new_ver, expected):
        poly_venv.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"

        def get_colored_version():
            return poly_venv.installed_packages()["colored"].version

        poly_venv.run([poly_venv.python, "-m", "pip", "install", "colored==1.4.0"])
        assert get_colored_version() == "1.4.0"

        # Insert our own preferred version to make sure it upgrades.
        poly_venv.run(
            [
                poly_venv.python,
                "-c",
                f"from polygraphy import mod; colored = mod.lazy_import('colored{new_ver}'); print(colored.__version__)",
            ]
        )
        assert _version_ok(get_colored_version(), expected)

    # Make sure the `requires` parameter of `lazy_import` functions as we expect.
    @pytest.mark.parametrize("preinstall", [True, False])
    @pytest.mark.parametrize(
        "new_ver, expected",
        [
            ("==1.4.2", "==1.4.2"),
        ],
    )
    def test_can_automatically_install_requirements(self, poly_venv, new_ver, expected, preinstall):
        poly_venv.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"

        def get_colored_version():
            return poly_venv.installed_packages()["colored"].version

        if preinstall:
            poly_venv.run([poly_venv.python, "-m", "pip", "install", "colored==1.4.0"])
            assert get_colored_version() == "1.4.0"

        # Insert our own preferred version to make sure it upgrades.
        poly_venv.run(
            [
                poly_venv.python,
                "-c",
                f"from polygraphy import mod; "
                f"requests = mod.lazy_import('requests==1.0.0', requires=['colored{new_ver}']); "
                f"requests.__version__; "
                f"import colored; print(colored.__version__)",
            ]
        )
        assert _version_ok(get_colored_version(), expected)

    @pytest.mark.parametrize(
        "import_params",
        [
            "'colored'",
            "'colored', pkg_name='colored'",
            "'colored', install_flags=['--force-reinstall']",
        ],
    )
    def test_autoinstall(self, poly_venv, import_params):
        poly_venv.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"
        assert "colored" not in poly_venv.installed_packages()

        poly_venv.run(
            [
                poly_venv.python,
                "-c",
                f"from polygraphy import mod; colored = mod.lazy_import({import_params}); mod.autoinstall(colored)",
            ]
        )

        assert "colored" in poly_venv.installed_packages()

    @pytest.mark.parametrize(
        "response, should_install",
        [
            (" ", True),
            ("yes", True),
            ("Y", True),
            ("n", False),
        ],
    )
    def test_ask_before_autoinstall(self, response, should_install, poly_venv):
        poly_venv.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"
        poly_venv.env["POLYGRAPHY_ASK_BEFORE_INSTALL"] = "1"
        assert "colored" not in poly_venv.installed_packages()

        process = sp.Popen(
            [
                poly_venv.python,
                "-c",
                "from polygraphy import mod; " "colored = mod.lazy_import('colored'); " "mod.autoinstall(colored)",
            ],
            env=poly_venv.env,
            stdin=sp.PIPE,
        )
        process.communicate(input=response.encode())
        process.wait()

        assert ("colored" in poly_venv.installed_packages()) == should_install

    # We can import inner modules, and Polygraphy should still autoinstall the outermost one.
    def test_can_install_for_nested_import(self, poly_venv):
        # WAR an issue with newer versions of protobuf and ONNX
        poly_venv.run([poly_venv.python, "-m", "pip", "install", "protobuf==3.19.4", "onnx==1.10.0"])

        poly_venv.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"

        poly_venv.run(
            [
                poly_venv.python,
                "-c",
                "from polygraphy import mod; "
                "shape_inference = mod.lazy_import('onnx.shape_inference'); "
                "print(shape_inference.infer_shapes)",
            ]
        )

        assert "onnx" in poly_venv.installed_packages()

    def test_all_lazy_imports(self):
        # NOTE: If this test fails, it means a new lazy dependency has been
        # introduced. Please ensure that AUTOINSTALL continues to work with the
        # new dependency.
        expected = [
            "fcntl",
            "msvcrt",
            "numpy",
            "onnx_graphsurgeon",
            "onnx.external_data_helper",
            "onnx.numpy_helper",
            "onnx.shape_inference",
            "onnx",
            "onnxmltools",
            "onnxruntime.tools.symbolic_shape_infer",
            "onnxruntime",
            "tensorflow",
            "tensorrt",
            "tf2onnx",
            "uff",
        ]
        if sys.version_info < (3, 8):
            expected.append("importlib_metadata")

        assert mod.importer._all_external_lazy_imports == set(expected)
