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


from setuptools import setup

distribution_package_name = "##TENSORRT_MODULE##_cu##CUDA_MAJOR##_libs"
import_package_name = "##TENSORRT_MODULE##_libs"


def get_requirements():
    reqs = [f"cuda-toolkit[cudart] >=##CUDA_MAJOR##,<{##CUDA_MAJOR## + 1}"]
    return reqs


setup(
    name=distribution_package_name,
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
    packages=[import_package_name],
    install_requires=get_requirements(),
    package_data={import_package_name: ["*.so*", "*.pyd", "*.pdb", "*.dll*"]},
    include_package_data=True,
    zip_safe=True,
    keywords="nvidia tensorrt deeplearning inference",
    url="https://github.com/nvidia/tensorrt",
    download_url="https://developer.nvidia.com/tensorrt",
)
