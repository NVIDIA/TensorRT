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

import pytest

from polygraphy.common.interface import TypedDict, TypedList
from polygraphy.exception import PolygraphyException


@pytest.fixture()
def int_to_float():
    class IntToFloat(TypedDict(lambda: int, lambda: float)):
        pass

    return IntToFloat()


class TestTypedDict:
    def test_wrong_type_set_item_value(self, int_to_float):
        with pytest.raises(PolygraphyException, match="Unsupported value type"):
            int_to_float[0] = "hi"

    def test_wrong_type_set_item_key(self, int_to_float):
        with pytest.raises(PolygraphyException, match="Unsupported key type"):
            int_to_float["hi"] = 1.0

    def test_wrong_type_update(self, int_to_float):
        with pytest.raises(PolygraphyException, match="Unsupported key type"):
            int_to_float.update({"hi": 1.0})


@pytest.fixture()
def ints():
    class Ints(TypedList(lambda: int)):
        pass

    return Ints()


class TestTypedList:
    def test_wrong_type_append(self, ints):
        with pytest.raises(PolygraphyException, match="Unsupported element type"):
            ints.append(1.0)

    def test_wrong_type_extend(self, ints):
        with pytest.raises(PolygraphyException, match="Unsupported element type"):
            ints.extend([0, 1, 2, 3, "surprise"])

    def test_wrong_type_iadd(self, ints):
        with pytest.raises(PolygraphyException, match="Unsupported element type"):
            ints += [0, 1.0]

    def test_wrong_type_setitem(self, ints):
        ints.append(0)
        with pytest.raises(PolygraphyException, match="Unsupported element type"):
            ints[0] = 1.0
