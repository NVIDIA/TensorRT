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

from setuptools import setup, find_packages
from pathlib import Path
import os

abspath = os.path.dirname(os.path.realpath(__file__))

license_header = """#
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
"""

# Generate version file
with open(os.path.join(abspath, "VERSION")) as f:
    version = f.read().strip()
with open(os.path.join(abspath, "tensorflow_quantization/version.py"), "w") as f:
    f.write(license_header)
    f.write(F"__version__ = \"{version}\"")

project_dir = Path(__file__).parent

# Setting up
setup(
    name="tensorflow_quantization",
    version=version,
    description="NVIDIA TensorFlow 2.x quantization toolkit",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=["tensorflow_quantization"],
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=["tensorflow-gpu==2.8.0", "tf2onnx==1.10.1"],
    author="NVIDIA",
    author_email="nvidia@nvidia.com",
    license="Apache 2.0",
)
