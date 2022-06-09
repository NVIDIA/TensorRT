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
from polygraphy import mod
from polygraphy.backend.base import BaseLoader

# For test_funcify_with_collision
functor2 = None


class TestExporter:
    def test_func(self):
        @mod.export()
        def test_func0():
            pass

        assert "test_func0" in __all__

    def test_class(self):
        @mod.export()
        class TestClass0:
            pass

        assert "TestClass0" in __all__

    def test_funcify_func_fails(self):
        with pytest.raises(AssertionError, match="must be a loader"):

            @mod.export(funcify=True)
            def test_func1():
                pass

    def test_funcify_non_base_loader_class(self):
        with pytest.raises(AssertionError, match="must derive from BaseLoader"):

            @mod.export(funcify=True)
            class NonFunctor0:
                def __init__(self, x):
                    self.x = x

    def test_funcify_duplicate_parameters_in_call_init(self):
        with pytest.raises(AssertionError, match="call_impl and __init__ have the same argument names"):

            @mod.export(funcify=True)
            class DupArgs(BaseLoader):
                def __init__(self, x):
                    self.x = x

                def call_impl(self, x):
                    self.x = x

    def test_funcify_takes_docstring(self):
        @mod.export(funcify=True)
        class DocstringFunctor(BaseLoader):
            """This is a docstring"""

            def __init__(self):
                pass

            def call_impl(self):
                pass

        assert "DocstringFunctor" in __all__
        assert "docstring_functor" in __all__

        assert docstring_functor.__doc__ == "Immediately evaluated functional variant of :class:`DocstringFunctor` .\n"

    def test_funcify_functor_no_call_args(self):
        @mod.export(funcify=True)
        class Functor0(BaseLoader):
            def __init__(self, x):
                self.x = x

            def call_impl(self):
                return self.x

        assert "Functor0" in __all__
        assert "functor0" in __all__
        assert functor0(0) == 0

    def test_funcify_functor_with_call_args(self):
        @mod.export(funcify=True)
        class Functor1(BaseLoader):
            def __init__(self, x):
                self.x = x

            def call_impl(self, y, z):
                return self.x, y, z

        assert "Functor1" in __all__
        assert "functor1" in __all__

        # __init__ arguments always precede __call__ arguments
        x, y, z = functor1(0, 1, -1)
        assert (x, y, z) == (0, 1, -1)

        # Keyword arguments should behave as expected
        x, y, z = functor1(y=1, x=0, z=-1)
        assert (x, y, z) == (0, 1, -1)

    def test_funcify_functor_with_call_args_defaults(self):
        @mod.export(funcify=True)
        class FunctorWithCallArgs(BaseLoader):
            def __init__(self, x=0):
                self.x = x

            def call_impl(self, y, z=-1):
                return self.x, y, z

        assert "FunctorWithCallArgs" in __all__
        assert "functor_with_call_args" in __all__

        # __init__ arguments always precede __call__ arguments
        x, y, z = functor_with_call_args(y=1)
        assert (x, y, z) == (0, 1, -1)

        # Keyword arguments should behave as expected
        x, y, z = functor_with_call_args(y=1)
        assert (x, y, z) == (0, 1, -1)

    def test_funcify_with_collision(self):
        with pytest.raises(AssertionError, match="symbol is already defined"):

            @mod.export(funcify=True)
            class Functor2(BaseLoader):
                def __init__(self, x):
                    self.x = x

                def call_impl(self, y, z):
                    return self.x, y, z

    def test_funcify_functor_with_dynamic_call_args_kwargs(self):
        @mod.export(funcify=True)
        class Functor3(BaseLoader):
            def __init__(self, f):
                self.f = f

            def call_impl(self, *args, **kwargs):
                return self.f(*args, **kwargs)

        assert "Functor3" in __all__
        assert "functor3" in __all__

        # We should be able to pass arbitrary arguments to call now.
        # __init__ arguments are always first.
        def func(arg0, arg1, arg2):
            return arg0 + arg1 + arg2

        assert functor3(func, 1, 2, arg2=4) == 7

    def test_funcify_with_inherited_init(self):
        class BaseFunctor4(BaseLoader):
            def __init__(self, x):
                self.x = x

        @mod.export(funcify=True)
        class Functor4(BaseFunctor4):
            def call_impl(self):
                return self.x

        assert "Functor4" in __all__
        assert "functor4" in __all__

        assert functor4(-1) == -1

    def test_funcify_functor_with_default_vals(self):
        @mod.export(funcify=True)
        class FunctorWithDefaults(BaseLoader):
            def __init__(self, w, x=1):
                self.w = w
                self.x = x

            def call_impl(self, y, z=3):
                return self.w, self.x, y, z

        assert "FunctorWithDefaults" in __all__
        assert "functor_with_defaults" in __all__

        # Since x and z have default values, the arguments will be interlaced into:
        # w, y, x, z
        # __init__ parameters take precedence, and call_impl parameters follow.
        w, x, y, z = functor_with_defaults(-1, -2)  # Set just w, y
        assert (w, x, y, z) == (-1, 1, -2, 3)

        w, x, y, z = functor_with_defaults(0, 1, 2, 3)  # Set all
        assert (w, x, y, z) == (0, 2, 1, 3)

    def test_funcify_functor_with_type_annotations(self):
        @mod.export(funcify=True)
        class FunctorWithTypeAnnotations(BaseLoader):
            def __init__(self, w: int, x: int = 1):
                self.w = w
                self.x = x

            def call_impl(self, y: int, z: int = 3):
                return self.w, self.x, y, z

        assert "FunctorWithTypeAnnotations" in __all__
        assert "functor_with_type_annotations" in __all__

        # Since x and z have default values, the arguments will be interlaced into:
        # w, y, x, z
        # __init__ parameters take precedence, and call_impl parameters follow.
        w, x, y, z = functor_with_type_annotations(-1, -2)  # Set just w, y
        assert (w, x, y, z) == (-1, 1, -2, 3)

        w, x, y, z = functor_with_type_annotations(0, 1, 2, 3)  # Set all
        assert (w, x, y, z) == (0, 2, 1, 3)
