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

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
import os


def is_standalone():
    return os.environ.get("STANDALONE") == "1"


def is_dla():
    return os.environ.get("ENABLE_DLA") == "1"


def get_requirements():
    def get_version_range(envvar, needs_exact_minor=False):
        vers = os.environ.get(envvar).replace("cuda-", "")
        major, minor = map(int, vers.split("."))
        if needs_exact_minor:
            return ">={major}.{minor},<{major}.{minor_next}".format(major=major, minor=minor, minor_next=minor + 1)
        else:
            return ">={major},<{major_next}".format(major=major, major_next=major + 1)

    if is_standalone():
        return [
            "nvidia-cuda-runtime" + get_version_range("CUDA"),
            "nvidia-cudnn" + get_version_range("CUDNN"),
            "nvidia-cublas" + get_version_range("CUDA"),
            "nvidia-cuda-nvrtc" + get_version_range("CUDA", needs_exact_minor=True),
            ]
    return []


name = "tensorrt"
if is_standalone():
    name = "nvidia-{:}".format(name)
    # Only standalone wheels need to be disambiguated. Otherwise, the entire tar/deb/rpm is DLA/non-DLA.
    if is_dla():
        name += "-dla"

setup(
    name=name,
    version="##TENSORRT_VERSION##",
    description="A high performance deep learning inference library",
    long_description="A high performance deep learning inference library",
    author="NVIDIA",
    license="Proprietary",
    classifiers=[
        'License :: Other/Proprietary License',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require={"numpy": "numpy"},
    package_data={'tensorrt': ["*.so*"]},
    include_package_data=True,
    zip_safe=True,
)
