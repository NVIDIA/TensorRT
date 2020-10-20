#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


"""Simple setup script"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

abspath = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(abspath, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="pytorch_quantization",
    version="2.0.0",
    description="NVIDIA Pytorch quantization toolkit",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],

    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pytorch_quantization.cuda_ext",
            sources=[os.path.join(abspath, "src/tensor_quant.cpp"),
                     os.path.join(abspath, "src/tensor_quant_gpu.cu")])
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    zip_safe=False,

    long_description=open("README.md", "r", encoding="utf-8").read(),
    url="https://github.com/nvidia/tensorrt/tools/pytorch-quantization",
    author="NVIDIA",
    author_email="nvidia@nvidia.com",

)
