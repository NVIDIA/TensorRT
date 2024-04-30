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

from setuptools import setup

tensorrt_module = "##TENSORRT_MODULE##"
package_name = "##TENSORRT_MODULE##"

# This file expects the following to be passed from the environment when using standalone wheels:
# - STANDALONE: Whether we are building a standalone wheel
IS_STANDALONE = os.environ.get("STANDALONE") == "1"
if IS_STANDALONE:
    tensorrt_module += "-cu##CUDA_MAJOR##_bindings"
    package_name += "_bindings"

setup(
    name=tensorrt_module,
    version="##TENSORRT_PYTHON_VERSION##",
    description="A high performance deep learning inference library",
    long_description="A high performance deep learning inference library",
    author="NVIDIA Corporation",
    license="Proprietary",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=[package_name],
    extras_require={"numpy": "numpy"},
    package_data={package_name: ["*.so*", "*.pyd", "*.pdb", "*.dll*"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://developer.nvidia.com/tensorrt",
    download_url="https://github.com/nvidia/tensorrt/tags",
)
