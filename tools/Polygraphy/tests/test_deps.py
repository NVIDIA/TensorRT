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

import glob
import os

import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.mod.importer import _version_ok

from tests.models.meta import ONNX_MODELS

# The tests here ensure that no additional dependencies are introduced into
# the various modules under Polygraphy.
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


@pytest.fixture()
def virtualenv_with_poly(virtualenv):
    virtualenv.env["PYTHONPATH"] = ROOT_DIR
    yield virtualenv


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


class TestPublicImports(object):
    def test_no_extra_submodule_dependencies_required(self, virtualenv_with_poly):
        # Submodules should not require any extra dependencies to import.
        for submodule_path in SUBMODULE_PATHS:
            submodule_name = ".".join(submodule_path.split(os.path.sep))
            cmd = ["python3", "-c", "from {:} import *".format(submodule_name)]
            print(" ".join(cmd))
            output = virtualenv_with_poly.run(cmd, capture=True)
            print(output)

    def test_can_json_without_numpy(self, virtualenv_with_poly):
        cmd = ["python3", "-c", "from polygraphy.json import to_json, from_json; x = to_json(1); x = from_json(x)"]
        print(" ".join(cmd))
        output = virtualenv_with_poly.run(cmd, capture=True)
        print(output)


# CLI tools and all their subtools
TOOLS = {
    "run": [],
    "convert": [],
    "inspect": ["data", "model", "tactics", "capability"],
    "surgeon": ["extract", "insert", "sanitize"],
    "template": ["trt-network"],
    "debug": ["build", "precision", "diff-tactics", "reduce", "repeat"],
}


class TestToolImports(object):
    # We should be able to at least launch tools with no dependencies installed.
    @pytest.mark.parametrize("tool, subtools", TOOLS.items())
    def test_can_run_tool_without_deps(self, virtualenv_with_poly, tool, subtools):
        POLYGRAPHY_BIN = os.path.join(ROOT_DIR, "bin", "polygraphy")
        BASE_TOOL_CMD = ["python3", POLYGRAPHY_BIN, tool, "-h"]

        def check_tool(tool):
            output = virtualenv_with_poly.run(tool, capture=True)
            assert "This tool could not be loaded due to an error:" not in output
            assert "error:" not in output
            assert "could not be loaded" not in output

        check_tool(BASE_TOOL_CMD)

        for subtool in subtools:
            check_tool(BASE_TOOL_CMD + [subtool])


class TestAutoinstallDeps(object):
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
    def test_can_automatically_install_deps(self, virtualenv_with_poly, cmd):
        if "--trt" in cmd and mod.version(trt.__version__) < mod.version("7.0"):
            pytest.skip("TRT 6 container has an old version of CUDA")

        if "--trt" in cmd:
            pytest.xfail("TensorRT 8.0.1.6 wheels are currently broken")

        virtualenv_with_poly.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"
        POLYGRAPHY_BIN = os.path.join(ROOT_DIR, "bin", "polygraphy")
        cmd = ["python3", POLYGRAPHY_BIN] + cmd
        print("Running: {:}".format(" ".join(cmd)))
        output = virtualenv_with_poly.run(cmd, capture=True)
        print(output)
        assert "is required, but not installed. Attempting to install now" in output

    @pytest.mark.parametrize(
        "new_ver, expected",
        [
            ("==1.4.2", "==1.4.2"),
            (mod.LATEST_VERSION, ">=1.4.2"),
        ],
    )
    def test_can_automatically_upgrade_deps(self, virtualenv_with_poly, new_ver, expected):
        virtualenv_with_poly.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"

        def get_colored_version():
            return virtualenv_with_poly.installed_packages()["colored"].version

        virtualenv_with_poly.run(["python3", "-m", "pip", "install", "colored==1.4.0"])
        assert get_colored_version() == "1.4.0"

        # Insert our own preferred version to make sure it upgrades.
        virtualenv_with_poly.run(
            [
                "python3",
                "-c",
                "from polygraphy import mod; "
                "colored = mod.lazy_import('colored', version='{:}'); "
                "print(colored.__version__)".format(new_ver),
            ]
        )
        assert _version_ok(get_colored_version(), expected)

    # We can import inner modules, and Polygraphy should still autoinstall the outermost one.
    def test_can_install_for_nested_import(self, virtualenv_with_poly):
        virtualenv_with_poly.env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"

        virtualenv_with_poly.run(
            [
                "python3",
                "-c",
                "from polygraphy import mod; "
                "shape_inference = mod.lazy_import('onnx.shape_inference'); "
                "print(shape_inference.infer_shapes)",
            ]
        )

        assert "onnx" in virtualenv_with_poly.installed_packages()

    def test_all_lazy_imports(self):
        # NOTE: If this test fails, it means a new lazy dependency has been
        # introduced. Please ensure that AUTOINSTALL continues to work with the
        # new dependency.
        expected = [
            "numpy",
            "onnx_graphsurgeon",
            "onnx.external_data_helper",
            "onnx.numpy_helper",
            "onnx.shape_inference",
            "onnx",
            "onnxmltools",
            "onnxruntime",
            "tensorflow",
            "tensorrt",
            "tf2onnx",
        ]
        assert mod.importer._all_external_lazy_imports == set(expected)
