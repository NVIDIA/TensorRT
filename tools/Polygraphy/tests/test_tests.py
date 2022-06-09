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

"""
Test-ception - this file includes tests for the tests.
For example, this includes things like custom fixtures.
"""


def test_sandboxed_install_run(sandboxed_install_run):
    status = sandboxed_install_run(["python3", "-c", "import colored; print(colored.__path__)"])
    assert status.success
    original_path = status.stdout

    # Once we install a package in the virtualenv, we should get that version of the package in all subsequent commands.
    status = sandboxed_install_run(["python3", "-m", "pip", "install", "colored"])
    assert status.success

    status = sandboxed_install_run(["python3", "-c", "import colored; print(colored.__path__)"])
    assert status.success
    venv_path = status.stdout

    assert original_path != venv_path
