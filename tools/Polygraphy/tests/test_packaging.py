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

import glob
import os

import polygraphy
import pytest

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


class TestWheel:
    def test_install(self, virtualenv):
        with pytest.raises(Exception, match="returned non-zero exit"):
            virtualenv.run(["python3", "-c", "import polygraphy"])

        virtualenv.run(["make", "install"], cwd=ROOT_DIR)

        # Check Python package is installed
        assert "polygraphy" in virtualenv.installed_packages()
        poly_pkg = virtualenv.installed_packages()["polygraphy"]
        assert poly_pkg.version == polygraphy.__version__

        # Check that we only package things we actually want.
        # If tests are packaged, they'll end up in a higher-level directory.
        assert not os.path.exists(os.path.join(poly_pkg.source_path, "tests"))

        EXCLUDE_FILES = ["__pycache__"]
        all_poly_files = glob.glob(os.path.join(poly_pkg.source_path, "polygraphy", "*"))
        all_poly_files = [f for f in map(os.path.basename, all_poly_files) if f not in EXCLUDE_FILES]

        # NOTE: This should be updated when new files are added to the top-level package.
        EXPECTED_FILES = set(
            [
                "backend",
                "mod",
                "__init__.py",
                "cuda",
                "logger",
                "constants.py",
                "util",
                "comparator",
                "tools",
                "exception",
                "func",
                "common",
                "json",
                "config.py",
            ]
        )
        assert set(all_poly_files) == EXPECTED_FILES

        # Check CLI is installed
        bin_path = os.path.join(virtualenv.virtualenv, "bin")

        poly_path = os.path.join(bin_path, "polygraphy")
        assert os.path.exists(poly_path)
        assert polygraphy.__version__ in virtualenv.run([poly_path, "-v"], capture=True)

        lib_path = os.path.join(virtualenv.virtualenv, "lib")
        output = virtualenv.run(["polygraphy", "-v"], capture=True)
        assert polygraphy.__version__ in output
        assert lib_path in output  # Make sure we're using the binary from the venv.
