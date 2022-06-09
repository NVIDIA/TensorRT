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
from polygraphy.logger import G_LOGGER

from polygraphy.util.format import FormatManager, DataFormat

import pytest


class FormatTestCase:
    def __init__(self, shape, format):
        self.shape = shape
        self.format = format


EXPECTED_FORMATS = [
    FormatTestCase((1, 3, 480, 960), DataFormat.NCHW),
    FormatTestCase((1, 3, 224, 224), DataFormat.NCHW),
    FormatTestCase((1, 224, 224, 3), DataFormat.NHWC),
    FormatTestCase((1, 9, 9, 3), DataFormat.NHWC),
]


@pytest.mark.parametrize("test_case", EXPECTED_FORMATS)
def test_format_deduction(test_case):
    assert test_case.format == FormatManager.determine_format(test_case.shape)
