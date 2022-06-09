#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from setuptools import find_packages, setup

import polygraphy_reshape_destroyer

import os


def main():
    # We change to the project root directory so that `setup.py` is usable from any directory.
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT_DIR)

    setup(
        # NOTE: You will want to edit most of these fields for your custom extension module.
        name="polygraphy_reshape_destroyer",
        version=polygraphy_reshape_destroyer.__version__,
        description="Polygraphy Reshape Destroyer: Destroyer Of Reshapes",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        url="https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy",
        author="NVIDIA",
        author_email="svc_tensorrt@nvidia.com",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
        ],
        license="Apache 2.0",
        install_requires=[
            "polygraphy",
            # Our included loader needs ONNX-GraphSurgeon to modify the model.
            "onnx_graphsurgeon",
        ],
        packages=find_packages(exclude=("tests", "tests.*")),
        # The format of the entry_points is:
        # {
        #   "polygraphy.run.plugins"    # Polygraphy run entrypoint (this should not be changed)
        #      : ["<name-of-plugin>=module.<submodule...>:object"]
        # }
        entry_points={
            "polygraphy.run.plugins": [
                "reshape-destroyer=polygraphy_reshape_destroyer.export:export_argument_groups",
            ]
        },
        zip_safe=True,
        python_requires=">=3.6",
    )


if __name__ == "__main__":
    main()
