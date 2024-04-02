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
import sys
from setuptools import find_packages, setup


def no_publish():
    blacklist = ["register"]
    for cmd in blacklist:
        if cmd in sys.argv:
            raise RuntimeError('Command "{}" blacklisted'.format(cmd))


def main():
    no_publish()
    with open('requirements.txt','r') as req_file:
        required_pckgs = [line.strip() for line in req_file.readlines()]

    with open('requirements-notebook.txt','r') as notebook_req_file:
        extras_require_notebook = [line.strip() for line in notebook_req_file.readlines()]

    setup(
        name="trex",
        version="0.1.8",
        description="TREX: TensorRT Engine Exploration Toolkit",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        author="NVIDIA",
        author_email="svc_tensorrt@nvidia.com",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
        ],
        license="Apache 2.0",
        install_requires=required_pckgs,
        extras_require={"notebook": extras_require_notebook},
        packages=find_packages(exclude=("tests", "tests.*")),
        scripts=[os.path.join("bin", "trex")],
        zip_safe=True,
    )


if __name__ == "__main__":
    main()
