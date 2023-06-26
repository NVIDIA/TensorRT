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
import platform
import sys

from setuptools import setup

tensorrt_module = "##TENSORRT_MODULE##"
tensorrt_version = "##TENSORRT_PYTHON_VERSION##"
index_url = "https://pypi.nvidia.com"

# cherry-pick information from `packaging.markers.default_environment()` needed to find the right wheel
# https://github.com/pypa/packaging/blob/23.1/src/packaging/markers.py#L175-L190
if sys.platform == "linux":
    platform_marker = "manylinux_2_17"
else:
    raise RuntimeError("TensorRT currently only builds wheels for linux")

if sys.implementation.name == "cpython":
    implementation_marker = "cp{}".format("".join(platform.python_version_tuple()[:2]))
else:
    raise RuntimeError("TensorRT currently only builds wheels for CPython")

machine_marker = platform.machine()
if machine_marker != "x86_64":
    raise RuntimeError("TensorRT currently only builds wheels for x86_64 processors")

libs_wheel = "{}/{}-libs/{}_libs-{}-py2.py3-none-{}_{}.whl".format(
    index_url,
    tensorrt_module,
    tensorrt_module,
    tensorrt_version,
    platform_marker,
    machine_marker,
)
bindings_wheel = "{}/{}-bindings/{}_bindings-{}-{}-none-{}_{}.whl".format(
    index_url,
    tensorrt_module,
    tensorrt_module,
    tensorrt_version,
    implementation_marker,
    platform_marker,
    machine_marker,
)

setup(
    name=tensorrt_module,
    version=tensorrt_version,
    description="A high performance deep learning inference library",
    long_description="A high performance deep learning inference library",
    author="NVIDIA Corporation",
    license="Proprietary",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=[tensorrt_module],
    install_requires=[
        "{}_libs @ {}".format(tensorrt_module, libs_wheel),
        "{}_bindings @ {}".format(tensorrt_module, bindings_wheel),
    ],
    python_requires=">=3.6",  # ref https://pypi.nvidia.com/tensorrt-bindings/
    extras_require={"numpy": "numpy"},
    package_data={tensorrt_module: ["*.so*", "*.pyd", "*.pdb"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://developer.nvidia.com/tensorrt",
    download_url="https://github.com/nvidia/tensorrt/tags",
)
