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
import copy
import functools
import inspect

from polygraphy import config, mod
from polygraphy.logger import G_LOGGER


def make_iterable(obj):
    return obj if type(obj) == tuple else (obj,)


@mod.export()
def extend(extend_func):
    """
    A decorator that uses the function it decorates to extend the function
    provided as a parameter.

    This is best illustrated with an example:
    ::

        def x(a0, a1, a2):
            rv0 = [a0, a1, a2]
            rv1 = None
            return rv0, rv1

        @extend(x)
        def y(rv0, rv1):
            rv0.append(-1)

        # We can now call `y` as if it were `x`, and we will receive
        # the return values from `x` after any modifications by `y`
        rv0, rv1 = y(1, 2, 3)
        assert rv0 == [1, 2, 3, -1]
        assert rv1 is None

    In this case, ``extend`` is essentially syntactic sugar for:
    ::

        def y(a0, a1, a2):
            rv0, rv1 = x(a0, a1, a2)

            # Body of `y` from previous section
            rv0.append(-1)

            return rv0, rv1

    If ``y`` does not return anything, or returns ``None``, then ``extend`` will
    ensure that the return value of ``x`` is forwarded to the caller.
    This means that ``y`` will provide exactly the same interface as ``x``.

    If `y` returns something other than ``None``, then its return value will be
    provided to the caller, and the return value of ``x`` will be discarded.

    In some cases, it may be necessary to access the parameters of ``x``.
    If ``y`` takes all of ``x``'s parameters prior to its usual parameters
    (which are the return values of ``x``), then all arguments to ``x`` will be
    forwarded to ``y``.
    Note that if ``x`` modifies its arguments in place, then ``y`` will see the modified
    arguments, `not` the original ones.
    For example:
    ::

        def x(x_arg):
            return x_arg + 1

        def y(x_arg, x_ret):
            # `y` can now see both the input argument as well as the return value of `x`
            assert x_ret == x_arg + 1

        assert y(5) == 6


    NOTE: This function will automatically unpack tuples returned by the function
    being extended. Thus, the following implementation of ``x`` would behave just like
    the one mentioned above:
    ::

        def x(a0, a1, a2):
            ret = (rv0, rv1)
            return ret # Tuple will be unpacked, and `y` still sees 2 parameters

    NOTE: The decorated function must not use variadic parameters like ``*args`` or ``**kwargs``.

    Args:
        extend_func (Callable): A callable to extend.
    """

    def extend_decorator(func):
        @functools.wraps(func)
        def extended_func(*args, **kwargs):
            extend_func_retval = extend_func(*args, **kwargs)
            extend_func_ret_tuple = make_iterable(extend_func_retval)

            func_params = inspect.signature(func).parameters
            # Special case for when the extended function does not return anything
            if len(func_params) == 0 and len(extend_func_ret_tuple) == 1 and extend_func_ret_tuple[0] is None:
                func_retval = func()
            elif len(extend_func_ret_tuple) == len(func_params):
                func_retval = func(*extend_func_ret_tuple)
            elif len(func_params) == len(extend_func_ret_tuple) + len(args) + len(kwargs):
                # We need to turn `extend_func_ret_tuple` into keyword arguments so that it can
                # be ordered after `**kwargs`.
                ret_arg_names = [param.name for param in list(func_params.values())[-len(extend_func_ret_tuple) :]]
                ret_kwargs = dict(zip(ret_arg_names, extend_func_ret_tuple))
                func_retval = func(*args, **kwargs, **ret_kwargs)
            else:

                def try_get_name(fn):
                    try:
                        return fn.__name__
                    except:
                        return fn

                G_LOGGER.critical(
                    f"Function: {try_get_name(func)} accepts {len(func_params)} parameter(s), "
                    f"but needs to accept {len(extend_func_ret_tuple)} parameter(s) from: {try_get_name(extend_func)} instead."
                    f"\nNote: Parameters should be: {tuple(map(type, extend_func_ret_tuple))}"
                )

            if func_retval is not None:
                return func_retval
            return extend_func_retval

        return extended_func

    return extend_decorator


@mod.export()
def constantmethod(func):
    """
    A decorator that denotes constant methods.

    NOTE: This decorator does nothing if the POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS environment variable is not set to `1`

    Example:
    ::

        class Dummy:
            def __init__(self):
                self.x = 1

            @func.constantmethod
            def modify_x(self):
                self.x = 2

        d = Dummy()
        d.modify_x() # This will fail!


    This provides only minimal protection against accidental mutation of instance attributes.
    For example, if a class includes references (e.g. a numpy array member), this function cannot
    ensure that the contents of that member (e.g. the values in a numpy array) will remain unchanged.
    """
    if not config.INTERNAL_CORRECTNESS_CHECKS:
        return func

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        old_dict = copy.copy(vars(self))
        ret = None
        try:
            ret = func(self, *args, **kwargs)
        finally:
            if vars(self) != old_dict:
                G_LOGGER.internal_error(
                    f"{self} was mutated in a constant method! Note:\nOld state: {old_dict}\nNew state: {vars(self)}"
                )
        return ret

    return wrapper
