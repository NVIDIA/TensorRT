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
from polygraphy import func
from polygraphy.exception import PolygraphyException, PolygraphyInternalException


class TestExtend:
    def test_override_rv(self):
        def x():
            return 1

        # Since y explicitly returns something, the return value of x is discarded.
        @func.extend(x)
        def y(elem):
            assert elem == 1
            return 2

        assert y() == 2

    def test_extend_named_parameters(self):
        def x(arg0, arg1):
            return arg0, arg1

        @func.extend(x)
        def y(elem0, elem1):
            pass

        arg0, arg1 = y(arg1=1, arg0=0)
        assert arg0 == 0
        assert arg1 == 1

    def test_extend_0_args_1_rv(self):
        def x():
            return 1

        @func.extend(x)
        def y(elem):
            assert elem == 1

        assert y() == 1

    def test_extend_0_args_2_rv(self):
        def x():
            return 1, 2

        @func.extend(x)
        def y(elem0, elem1):
            assert elem0 == 1
            assert elem1 == 2

        assert y() == (1, 2)

    def test_extend_1_args_0_rv(self):
        def x(arg0):
            pass

        @func.extend(x)
        def y():
            pass

        y(1)

    def test_extend_1_args_1_rv(self):
        def x(arg0):
            assert arg0 == 1
            return 3

        @func.extend(x)
        def y(elem):
            assert elem == 3

        assert y(1) == 3

    def test_extend_2_args_2_rv(self):
        def x(arg0, arg1):
            assert arg0 == -1
            assert arg1 == -1
            return 1, 2

        @func.extend(x)
        def y(elem0, elem1):
            assert elem0 == 1
            assert elem1 == 2

        assert y(-1, -1) == (1, 2)

    def test_extend_can_modify_rv(self):
        def x():
            return []

        @func.extend(x)
        def y(lst):
            lst.extend([1, 2, 3])

        assert x() == []
        assert y() == [1, 2, 3]

    def test_extend_can_modify_rv_objects(self):
        class ModifiableObj:
            def __init__(self):
                self.value = 0

        def x():
            return ModifiableObj()

        @func.extend(x)
        def y(mo):
            mo.value = 1

        assert x().value == 0
        assert y().value == 1

    def test_extend_incorrect_num_args(self):
        def x():
            return 1, 2

        with pytest.raises(
            PolygraphyException, match=r"Function: y accepts 1 parameter\(s\), but needs to accept 2 parameter\(s\)"
        ):

            @func.extend(x)
            def y(elem0):
                assert elem0 == 1

            y()

    @pytest.mark.parametrize("args_mode", ["kwargs", "args", "mixed"])
    def test_extend_forward_parameters(self, args_mode):
        def x(x_arg0, x_arg1):
            return x_arg0 + x_arg1

        @func.extend(x)
        def y(x_arg0, x_arg1, x_ret):
            assert x_ret == x_arg0 + x_arg1

        if args_mode == "kwargs":
            assert y(x_arg0=2, x_arg1=1) == 3
        elif args_mode == "args":
            assert y(2, 1) == 3
        else:
            assert args_mode == "mixed"
            assert y(2, x_arg1=1) == 3


class TestConstantMethod:
    def test_cannot_modify_attrs(self):
        class Dummy:
            def __init__(self):
                self.x = 1

            @func.constantmethod
            def modify_x(self):
                self.x = 2

        d = Dummy()
        with pytest.raises(PolygraphyInternalException, match="was mutated in a constant method"):
            d.modify_x()

    def test_cannot_add_attrs(self):
        class Dummy:
            @func.constantmethod
            def modify_x(self):
                self.x = 2

        d = Dummy()
        with pytest.raises(PolygraphyInternalException, match="was mutated in a constant method"):
            d.modify_x()
