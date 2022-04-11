#!/usr/bin/env python3
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

import argparse
import os
import re

# Configuration

copyright_year = "2020"

extensions_p = (".py", ".sh", ".cmake", "CMakeLists")
extensions_c = (".c", ".cpp", ".h", ".hpp", ".cu")

pattern_p = """#\s*
# Copyright \(c\) ([1-2][0-9]{3}),* NVIDIA CORPORATION.*
#\s*
# Licensed under the Apache License, Version 2.0 \(the "License"\);\s*
# you may not use this file except in compliance with the License.\s*
# You may obtain a copy of the License at\s*
#\s*
#     http://www.apache.org/licenses/LICENSE-2.0\s*
#\s*
# Unless required by applicable law or agreed to in writing, software\s*
# distributed under the License is distributed on an "AS IS" BASIS,\s*
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\s*
# See the License for the specific language governing permissions and\s*
# limitations under the License.\s*
#
"""

header_p = """#
# Copyright (c) {year}, NVIDIA CORPORATION. All rights reserved.
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
""".format(year=copyright_year)

pattern_c = """/\*\s*
 \* Copyright \(c\) ([1-2][0-9]{3}),* NVIDIA CORPORATION.*
 \*\s*
 \* Licensed under the Apache License, Version 2.0 \(the "License"\);\s*
 \* you may not use this file except in compliance with the License.\s*
 \* You may obtain a copy of the License at\s*
 \*\s*
 \*     http://www.apache.org/licenses/LICENSE-2.0\s*
 \*\s*
 \* Unless required by applicable law or agreed to in writing, software\s*
 \* distributed under the License is distributed on an "AS IS" BASIS,\s*
 \* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\s*
 \* See the License for the specific language governing permissions and\s*
 \* limitations under the License.\s*
 \*/
"""

header_c = """/*
 * Copyright (c) {year}, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
""".format(year=copyright_year)

# Routines

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dir', type=str, required=True, help='Root directory to start the scan')
    parser.add_argument('-f','--force-update', action='store_true', help='Force the header writes for all files')
    parser.add_argument('--dry-run', action='store_true', help='Just perform a dry-run')
    parser.add_argument('--max-depth', type=int,default=100, help='Maximum depth to recurse while scanning files.')
    return parser

def update(filename, args):
    """
    Update copyright header for specified file
    """
    if filename.endswith(extensions_p):
        pattern = re.compile(pattern_p)
        header = header_p
        shebang = re.compile(r'^(\#\!.*\n)', re.MULTILINE)
    elif filename.endswith(extensions_c):
        pattern = re.compile(pattern_c)
        header = header_c
        shebang = None
    else:
        return

    with open(filename, "r+") as f:
        data = f.read()
        match = pattern.search(data)
        if match:
            year = match.group(1)
            if copyright_year == year:
                if args.force_update:
                    print(filename,": FORCED")
                    new_data = pattern.sub(header, data, count=1)
                else:
                    print(filename,": SKIP")
                    return
            else:
                print(filename,": UPDATE (",year,"->",copyright_year,")")
                new_data = pattern.sub(header, data, count=1)
        else:
            match = shebang.search(data) if shebang else None
            if match:
                print(filename,": ADD ( after",match.group(1),")")
                new_data = shebang.sub(match.group(1)+header, data, count=1)
            else:
                print(filename,": ADD ( top )")
                new_data = header+data

    if not args.dry_run:
        with open(filename, "w") as f:
            f.write(new_data)

def copyright_scan(directory, depth, args, exclude_dirs=[]):
    """
    Update copyright for TensorRT sources
    """
    if directory in exclude_dirs:
        return
    for f in os.listdir(directory):
        filename = os.path.join(directory,f)
        if os.path.isdir(filename) and (depth > 0):
            copyright_scan(filename, depth-1, args, exclude_dirs)
        elif filename.endswith(extensions_p + extensions_c):
            update(filename, args)

def main():
    parser = argparse.ArgumentParser(description='TensorRT copyright scan')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    exclude_dirs = ["./third_party","./build","./parsers/onnx", "./include"]
    copyright_scan(args.dir, args.max_depth, args, exclude_dirs)

if __name__ == '__main__':
    main()
