#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from polygraphy.common import func


class TestExtend(object):
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
        class ModifiableObj(object):
            def __init__(self):
                self.value = 0


        def x():
            return ModifiableObj()

        @func.extend(x)
        def y(mo):
            mo.value = 1

        assert x().value == 0
        assert y().value == 1
