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
nvidia_pip_index_url = os.environ.get("NVIDIA_PIP_INDEX_URL", "https://pypi.nvidia.com")
disable_internal_pip = os.environ.get("NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP", False)


def run_pip_command(args, call_func):
    try:
        return call_func([sys.executable, "-m", "pip"] + args)
    except subprocess.CalledProcessError:
        return call_func([os.path.join(sys.exec_prefix, "bin", "pip")] + args)


class InstallCommand(install):
    def run(self):
        def install_dep(package_name):
            status = sp.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "{:}==##TENSORRT_PYTHON_VERSION##".format(package_name),
                    "--index-url",
                    "https://pypi.nvidia.com",
                ]
            )
            status.check_returncode()

        install_dep("{:}_libs".format(tensorrt_module))
        install_dep("{:}_bindings".format(tensorrt_module))

        install.run(self)


setup(
    name=tensorrt_module,
    version=tensorrt_version,
    description="A high performance deep learning inference library",
    long_description="""A high performance deep learning inference library

To install, please execute the following:
```
pip install tensorrt --extra-index-url {}
```
Or add the index URL to the (space-separated) PIP_EXTRA_INDEX_URL environment variable:
```
export PIP_EXTRA_INDEX_URL='{}'
pip install tensorrt
```
When the extra index url does not contain `{}`, a nested `pip install` will run with the proper extra index url hard-coded.
""".format(
        nvidia_pip_index_url, nvidia_pip_index_url, nvidia_pip_index_url
    ),
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
    python_requires=">=3.6",  # ref https://pypi.nvidia.com/tensorrt-bindings/
    cmdclass=cmdclass,
    extras_require={"numpy": "numpy"},
    package_data={tensorrt_module: ["*.so*", "*.pyd", "*.pdb"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://developer.nvidia.com/tensorrt",
    download_url="https://github.com/nvidia/tensorrt/tags",
)
