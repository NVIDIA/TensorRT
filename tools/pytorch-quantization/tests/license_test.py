#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""test the license of source files."""

import pytest
from pathlib import Path
from filecmp import cmp

# pylint:disable=missing-docstring, no-self-use

class TestLicense():

    def test_license(self):
        root = Path(__file__).parent.parent.absolute()
        root_len = len(str(root))

        # Collect files ending with relevant extensions
        file_list = []
        file_types = ['*.py', '*.cpp', '*.cu', '*.h', '*.hpp', '*.c', '*.sh']
        for ft in file_types:
            file_list += list(root.rglob(ft))

        # Trim files from build folders
        build_folders = ['build', 'dist', '.eggs', '.vscode']
        build_files = []
        for src_file in file_list:
            local_path = str(src_file.parents[0])[root_len : ]
            for folder in build_folders:
                if folder in local_path:
                    build_files.append(src_file)

        for bf in build_files:
            file_list.remove(bf)

        print (f"Found {len(file_list)} source files")

        cpp_header = (root / 'tests' / 'license_test_header_cpp.txt').open().readlines()
        py_header = (root / 'tests' / 'license_test_header_py.txt').open().readlines()
        sh_header = (root / 'tests' / 'license_test_header_sh.txt').open().readlines()

        invalid_files = []
        for f in file_list:
            with open(f) as src_file:
                src_lines = src_file.readlines()

            # Skip empty files
            if len(src_lines) == 0:
                continue

            if f.suffix == '.py':
                header = py_header
            elif f.suffix == '.sh':
                header = sh_header
            else:
                header = cpp_header

            num_lines = len(header)
            if len(src_lines) < num_lines:
                invalid_files.append(f)
                continue

            for i in range(num_lines):
                if src_lines[i] != header[i]:
                    invalid_files.append(f)
                    break

        if len(invalid_files) > 0:
            for f in invalid_files:
                print(f"The file {f} has an invalid header!")
            raise AssertionError("%d files have invalid headers!" % (len(invalid_files)))
