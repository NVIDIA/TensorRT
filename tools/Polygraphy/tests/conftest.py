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

import copy
import glob
import os
import subprocess as sp

import pytest

from tests.helper import ROOT_DIR


@pytest.fixture()
def sandboxed_install_run(virtualenv, script_runner):
    """
    A special fixture that runs commands, but sandboxes any `pip install`s in a virtual environment.
    Packages from the test environment are still usable, but those in the virtual environment take precedence
    """

    VENV_PYTHONPATH = glob.glob(os.path.join(virtualenv.virtualenv, "lib", "python*", "site-packages"))[0]

    class StatusWrapper:
        def __init__(self, stdout=None, stderr=None, success=None) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.success = success

    def run_impl(command, cwd=None):
        env = copy.copy(os.environ)
        # Always prioritize our own copy of Polygraphy over anything in the venv.
        env["PYTHONPATH"] = ROOT_DIR + os.pathsep + VENV_PYTHONPATH

        print(f"Running command: {' '.join(command)}")

        status = StatusWrapper()
        if "pip" in command:
            virtualenv.run(command, cwd=cwd)
            status.success = True
        elif command[0] == "polygraphy":
            sr_status = script_runner.run(*command, cwd=cwd, env=env)
            status.stdout = sr_status.stdout
            status.success = sr_status.success
        else:
            sp_status = sp.run(command, cwd=cwd, env=env, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
            status.stdout = sp_status.stdout
            status.stderr = sp_status.stderr
            status.success = sp_status.returncode == 0

        return status

    return run_impl
