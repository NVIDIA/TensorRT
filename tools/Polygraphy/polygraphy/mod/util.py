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


def version(version_str):
    def process_version_part(num):
        suffix = None
        if "+" in num:
            num, suffix = num.split("+")

        try:
            num = int(num)
        except ValueError:
            VERSION_SUFFIXES = ["a", "b", "rc", "post", "dev"]
            # One version part can only contain one of the above suffixes
            for version_suffix in VERSION_SUFFIXES:
                if version_suffix in num:
                    num = num.partition(version_suffix)
                    break

        return [num, suffix] if suffix is not None else [num]

    ver_list = []
    for num in version_str.split("."):
        ver_list.extend(process_version_part(num))

    return tuple(ver_list)
