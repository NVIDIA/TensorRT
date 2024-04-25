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

import pytest
from polygraphy import mod


@pytest.mark.parametrize(
    "ver0, ver1, expected",
    [
        ("1.0.0", "2.0.0", False),
        ("1.0.0", "0.9.0", True),
        ("0.0b1", "0.1b1", False),
        ("0.1b1", "0.1b0", True),
        ("0.12b1", "0.1b0", True),
        ("0.1b0", "0.1a0", True),
        ("0.1rc0", "0.1b0", True),
        ("0.post1", "0.post0", True),
        ("0.post1", "0.post2", False),
        ("1.13.1+cu117", "1.13.0", True),
        ("1.13.1+cu117", "1.13.2", False),
    ],
)
def test_version(ver0, ver1, expected):
    assert (mod.version(ver0) > mod.version(ver1)) == expected
