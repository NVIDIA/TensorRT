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

class TestConstants:
    def test_config(self):
        from polygraphy import constants

        assert (constants.INTERNAL_CORRECTNESS_CHECKS, constants.AUTOINSTALL_DEPS)


class TestCompareFunc:
    def test_basic_compare_func(self):
        from polygraphy.comparator import CompareFunc

        CompareFunc.basic_compare_func(atol=1, rtol=1)
