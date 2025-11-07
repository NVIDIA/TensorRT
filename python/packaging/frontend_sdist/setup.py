#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import os
import platform
import subprocess
import sys
import glob

from setuptools import setup
from setuptools.command.install import install

distribution_package_name = "##TENSORRT_MODULE##_cu##CUDA_MAJOR##"
import_package_name = "##TENSORRT_MODULE##"
plugin_import_package_name = f"{import_package_name}.plugin"
tensorrt_version = "##TENSORRT_PYTHON_VERSION##"
tensorrt_submodules = [
    "{}_libs=={}".format(distribution_package_name, tensorrt_version),
    "{}_bindings=={}".format(distribution_package_name, tensorrt_version),
]
nvidia_pip_index_url = os.environ.get("NVIDIA_PIP_INDEX_URL", "https://pypi.nvidia.com")

DISABLE_INTERNAL_PIP_FLAG = "NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP"
disable_internal_pip = os.environ.get(DISABLE_INTERNAL_PIP_FLAG, "1") == "1"


def run_pip_command(args, call_func):
    env = os.environ.copy()
    env["PYTHONPATH"] = sys.exec_prefix
    try:
        return call_func([sys.executable, "-m", "pip"] + args, env=env)
    except subprocess.CalledProcessError:

        def find_pip():
            pip_name = "pip"
            if sys.platform.startswith("win"):
                pip_name = "pip.exe"
            for path in glob.iglob(os.path.join(sys.exec_prefix, "**"), recursive=True):
                if os.path.isfile(path) and os.path.basename(path) == pip_name:
                    return path
            return None

        pip_path = find_pip()
        if pip_path is None:
            # Couldn't find `pip` in `sys.exec_prefix`, so we have no option but to abort.
            raise
        return call_func([pip_path] + args, env=env)


# check wheel availability using information from https://github.com/pypa/packaging/blob/23.1/src/packaging/markers.py#L175-L190
if sys.platform not in ("linux", "win32"):
    raise RuntimeError("TensorRT currently only builds wheels for Linux and Windows")
if sys.implementation.name != "cpython":
    raise RuntimeError("TensorRT currently only builds wheels for CPython")
if platform.machine() not in ("x86_64", "AMD64", "aarch64"):
    raise RuntimeError("TensorRT currently only builds wheels for x86_64 and ARM SBSA processors")
if "tegra" in platform.release():
    raise RuntimeError("TensorRT does not currently build wheels for Tegra systems")


class InstallCommand(install):
    def run(self):
        # pip-inside-pip hack ref #3080
        run_pip_command(
            [
                "install",
                "--extra-index-url",
                nvidia_pip_index_url,
                *tensorrt_submodules,
            ],
            subprocess.check_call,
        )

        super().run()


def pip_config_list():
    """Get the current pip config (env vars, config file, etc)."""
    with contextlib.suppress(subprocess.CalledProcessError, OSError, UnicodeDecodeError):
        return run_pip_command(["config", "list"], subprocess.check_output).decode()
    return ""


def parent_command_line():
    """Get the command line of the parent PID."""
    pid = os.getppid()

    # try retrieval using psutil
    with contextlib.suppress(ImportError, ModuleNotFoundError, Exception):
        import psutil
        return " ".join(psutil.Process(pid).cmdline())
    # fall back to shell
    with contextlib.suppress(subprocess.CalledProcessError, OSError, UnicodeDecodeError):
        return subprocess.check_output(["ps", "-p", str(pid), "-o", "command", "--no-headers"]).decode()
    return ""


# use pip-inside-pip hack only if the nvidia index is not set in the environment
install_requires = []
if disable_internal_pip or nvidia_pip_index_url in parent_command_line() or nvidia_pip_index_url in pip_config_list():
    install_requires.extend(tensorrt_submodules)
    cmdclass = {}
else:
    cmdclass = {"install": InstallCommand}


setup(
    name=distribution_package_name,
    version=tensorrt_version,
    description="A high performance deep learning inference library",
    long_description="""
NVIDIA TensorRT is an SDK that facilitates high-performance machine learning inference. It is designed to work in a complementary fashion with training frameworks such as TensorFlow, PyTorch, and MXNet. It focuses specifically on running an already-trained network quickly and efficiently on NVIDIA hardware.

If the dependencies of this package cannot be correctly installed from PyPI for any reason, you can try using the NVIDIA package index instead:
```
export {}=0
pip install tensorrt
```
""".format(
        DISABLE_INTERNAL_PIP_FLAG
    ),
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    license="Proprietary",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=[import_package_name, plugin_import_package_name],
    install_requires=install_requires,
    setup_requires=["wheel", "pip"],
    python_requires=">=3.6",  # ref https://pypi.nvidia.com/tensorrt-bindings/
    cmdclass=cmdclass,
    extras_require={"numpy": "numpy"},
    package_data={import_package_name: ["*.so*", "*.pyd", "*.pdb", "*.dll*"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://github.com/nvidia/tensorrt",
    download_url="https://developer.nvidia.com/tensorrt",
)
