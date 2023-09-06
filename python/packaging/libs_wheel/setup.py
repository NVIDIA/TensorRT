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

from setuptools import setup

module_name = "##TENSORRT_MODULE##_libs"


def get_requirements():
    def get_version_range():
        def get_vers(var):
            vers = os.environ.get(var).replace("cuda-", "")
            major, minor = map(int, vers.split("."))
            return major, minor

        cuda_major, _ = get_vers("CUDA")
        return "-cu{cuda_major}".format(cuda_major=cuda_major)

    reqs = ["nvidia-cuda-runtime" + get_version_range()]
    if "##TENSORRT_MODULE##" == "tensorrt":
        reqs += [
            "nvidia-cudnn" + get_version_range(),
            "nvidia-cublas" + get_version_range(),
        ]
    return reqs


setup(
    name=module_name,
    version="##TENSORRT_PYTHON_VERSION##",
    description="TensorRT Libraries",
    long_description="TensorRT Libraries",
    author="NVIDIA Corporation",
    license="Proprietary",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=[module_name],
    install_requires=get_requirements(),
    package_data={module_name: ["*.so*", "*.pyd", "*.pdb", "*.dll*"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://developer.nvidia.com/tensorrt",
    download_url="https://github.com/nvidia/tensorrt/tags",
)
