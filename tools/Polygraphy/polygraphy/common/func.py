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
import inspect

from polygraphy.logger import G_LOGGER


def make_iterable(obj):
    return obj if type(obj) == tuple else (obj, )


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

    NOTE: This function will automatically unpack tuples returned by the function
    being extended. Thus, the following implementation of ``x`` would behave just like
    the one mentioned above:
    ::

        def x(a0, a1, a2):
            ret = (rv0, rv1)
            return ret # Tuple will be unpacked, and `y` still sees 2 parameters

    Args:
        extend_func (Callable): A callable to extend.
    """
    def extend_decorator(func):
        def extended_func(*args, **kwargs):
            extend_func_retval = extend_func(*args, **kwargs)
            extend_func_ret_tuple = make_iterable(extend_func_retval)

            func_args = inspect.getfullargspec(func).args
            # Special case for when the extended function does not return anything
            if len(func_args) == 0 and len(extend_func_ret_tuple) == 1 and extend_func_ret_tuple[0] is None:
                func_retval = func()
            elif len(extend_func_ret_tuple) == len(func_args):
                func_retval = func(*extend_func_ret_tuple)
            else:
                G_LOGGER.critical("Function: {:} expected to receive {:} parameters from function: {:}, but "
                                  "received: {:} instead".format(func.__name__, len(func_args), extend_func.__name__, extend_func_ret_tuple))

            if func_retval is not None:
                return func_retval
            return extend_func_retval

        return extended_func
    return extend_decorator
