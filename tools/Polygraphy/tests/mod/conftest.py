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
import pytest
from tests.helper import ROOT_DIR


@pytest.fixture()
def poly_venv(virtualenv):
    virtualenv.env["PYTHONPATH"] = ROOT_DIR

    # Preserve CUDA-related library paths while clearing other paths
    # This is needed for TensorRT to initialize CUDA properly
    original_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_paths = [
        path
        for path in original_ld_path.split(os.pathsep)
        if path and ("cuda" in path.lower() or "tensorrt" in path.lower())
    ]
    virtualenv.env["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_paths)

    # Newer versions of setuptools break pytest-virtualenv
    virtualenv.run([virtualenv.python, "-m", "pip", "install", "setuptools==59.6.0"])

    return virtualenv
