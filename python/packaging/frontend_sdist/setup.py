#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess
import sys

from setuptools import setup
from setuptools.command.install import install

tensorrt_module = "##TENSORRT_MODULE##"
tensorrt_version = "##TENSORRT_PYTHON_VERSION##"
tensorrt_submodules = [
    "{}_libs=={}".format(tensorrt_module, tensorrt_version),
    "{}_bindings=={}".format(tensorrt_module, tensorrt_version),
]


class InstallCommand(install):
    def run(self):
        # pip-inside-pip hack ref #3080
        subprocess.check_call(
            [
                "{}/bin/pip".format(sys.exec_prefix),
                "install",
                "--extra-index-url",
                "https://pypi.nvidia.com",
                *tensorrt_submodules,
            ]
        )

        super().run()


def pip_config_list():
    """Get the current pip config (env vars, config file, etc)."""
    return subprocess.check_output(
        [
            "{}/bin/pip".format(sys.exec_prefix),
            "config",
            "list",
        ]
    ).decode()


def parent_command_line():
    """Get the command line of the parent PID."""
    pid = os.getppid()
    # try retrieval using psutil
    try:
        import psutil

        return " ".join(psutil.Process(pid).cmdline())
    except ModuleNotFoundError:
        pass
    # fall back to shell
    try:
        return (
            subprocess.check_output(["ps", "-p", str(pid), "-o", "command"])
            .decode()
            .split("\n")[1]
        )
    except subprocess.CalledProcessError:
        return ""


# use pip-inside-pip hack only if the nvidia index is not set in the environment
if "pypi.nvidia.com" in pip_config_list() or "pypi.nvidia.com" in parent_command_line():
    install_requires = tensorrt_submodules
    cmdclass = {}
else:
    install_requires = []
    cmdclass = {"install": InstallCommand}


setup(
    name=tensorrt_module,
    version=tensorrt_version,
    description="A high performance deep learning inference library",
    long_description="""A high performance deep learning inference library

To install, please execute the following:
```
pip install tensorrt --extra-index-url https://pypi.nvidia.com
```
Or put the index URL in the (comma-separated) PIP_EXTRA_INDEX_URL environment variable:
```
export PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
pip install tensorrt
```
When the extra index url is not found, a nested `pip install` will run with the extra index url hard-coded.
""",
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    license="Proprietary",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=[tensorrt_module],
    install_requires=install_requires,
    cmdclass=cmdclass,
    extras_require={"numpy": "numpy"},
    package_data={tensorrt_module: ["*.so*", "*.pyd", "*.pdb"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://developer.nvidia.com/tensorrt",
    download_url="https://github.com/nvidia/tensorrt/tags",
)
