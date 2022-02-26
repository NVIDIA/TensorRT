#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import sys
import warnings
from textwrap import dedent

import polygraphy
from polygraphy import config
from polygraphy.logger import G_LOGGER
from polygraphy.mod.util import version


def _add_to_all(symbol, module):
    if hasattr(module, "__all__"):
        module.__all__.append(symbol)
    else:
        module.__all__ = [symbol]


def _define_in_module(name, symbol, module):
    assert name not in vars(module), "This symbol is already defined!"
    vars(module)[name] = symbol
    _add_to_all(name, module)


def export(funcify=False, func_name=None):
    """
    Decorator that exports a symbol into the ``__all__`` attribute of
    the caller's module. This makes the symbol visible in a ``*`` import
    (e.g. ``from module import *``) and hides other symbols unless they are
    also present in ``__all__``.

    Args:
        funcify (bool):
                Whether to create and export a function that will call a decorated Polygraphy loader.
                The decorated type *must* be a subclass of ``BaseLoader`` if ``funcify=True``.

                This is useful to provide convenient short-hands to immediately evaluate loaders.
                For example:
                ::

                    @mod.export(funcify=True)
                    class SuperCoolModelFromPath(BaseLoader):
                        def __init__(self, init_params):
                            ...

                        def call_impl(self, call_params):
                            ...

                    # We can now magically access an immediately evaluated functional
                    # variant of the loader:
                    model = super_cool_model_from_path(init_params, call_params)

                    # Which is equivalent to:
                    load_model = SuperCoolModelFromPath(init_params)
                    model = load_model(call_params)


                The signature of the generated function is a combination of the signatures
                of ``__init__`` and ``call_impl``. Specifically, parameters without defaults will
                precede those with defaults, and ``__init__`` parameters will precede ``call_impl``
                parameters. Special parameters like ``*args`` and ``**kwargs`` will always be the last
                parameters in the generated signature if they are present in the loader method signatures.
                The return value(s) will always come from ``call_impl``.

                For example:
                ::

                    # With __init__ signature:
                    def __init__(a, b=0) -> None:

                    # And call_impl signature:
                    def call_impl(c, d=0) -> z:

                    # The generated function will have a signature:
                    def generated(a, c, b=0, d=0) -> z:

        func_name (str):
                If funcify is True, this controls the name of the generated function.
                By default, the exported function will use the same name as the loader, but
                ``snake_case`` instead of ``PascalCase``.
    """
    module = inspect.getmodule(sys._getframe(1))

    # Find a method by wallking the inheritance hierarchy of a type:
    def find_method(symbol, method):
        hierarchy = inspect.getmro(symbol)
        for ancestor in hierarchy:
            if method in vars(ancestor):
                return vars(ancestor)[method]

        assert False, "Could not find method: {:} in the inheritance hierarcy of: {:}".format(method, symbol)

    def export_impl(func_or_cls):
        _add_to_all(func_or_cls.__name__, module)

        if funcify:
            # We only support funcify-ing BaseLoaders, and only if __init__ and call_impl
            # have no overlapping parameters.
            from polygraphy.backend.base import BaseLoader

            assert inspect.isclass(func_or_cls), "Decorated type must be a loader to use funcify=True"
            assert BaseLoader in inspect.getmro(
                func_or_cls
            ), "Decorated type must derive from BaseLoader to use funcify=True"
            loader = func_or_cls

            def get_params(method):
                return [
                    p
                    for p in inspect.signature(find_method(func_or_cls, method)).parameters.values()
                    if p.name != "self"
                ]

            init_params = get_params("__init__")
            call_impl_params = get_params("call_impl")

            def param_names(params):
                return list(str(p).partition("=")[0] for p in params)

            assert (set(param_names(call_impl_params)) - set(param_names(init_params))) == set(
                param_names(call_impl_params)
            ), "Cannot funcify a type where call_impl and __init__ have the same argument names!"

            # Dynamically generate a function with the right signature.

            # To generate the signature, we use the init and call_impl arguments,
            # but move required arguments (i.e. without default values) to the front.
            def is_special(param):
                return "*" in str(param)

            def has_default(param):  # Non special arguments that have default values
                return "=" in str(param)

            def build_arg_list(should_include):
                arg_list = [str(p) for p in init_params if should_include(p)]
                arg_list += [str(p) for p in call_impl_params if should_include(p)]
                return arg_list

            non_default_args = build_arg_list(should_include=lambda p: not is_special(p) and not has_default(p))
            default_args = build_arg_list(should_include=lambda p: not is_special(p) and has_default(p))
            special_args = build_arg_list(should_include=is_special)

            signature = ", ".join(non_default_args + default_args + special_args)

            init_args = ", ".join(param_names(init_params))
            call_impl_args = ", ".join(param_names(call_impl_params))

            func_code = dedent(
                """
                def func_impl({signature}):
                    return loader_binding({init_args})({call_impl_args})

                func_var = func_impl
                """.format(
                    signature=signature, init_args=init_args, call_impl_args=call_impl_args
                )
            )

            exec(
                func_code, {"loader_binding": loader}, locals()
            )  # Need to bind the loader this way, or it won't be accesible from func_code.
            func = locals()["func_var"]

            # Next we setup the docstring so that it is a combination of the __init__
            # and call_impl docstrings.
            func.__doc__ = "Immediately evaluated functional variant of :class:`{}` .\n".format(loader.__name__)

            def try_add_method_doc(method):
                call_impl = find_method(loader, method)
                if call_impl.__doc__:
                    func.__doc__ += dedent(call_impl.__doc__)

            try_add_method_doc("__init__")
            try_add_method_doc("call_impl")

            # Now that the function has been defined, we just need to add it into the module's
            # __dict__ so it is accessible like a normal symbol.
            def pascal_to_snake(name):
                return "".join("_{:}".format(c.lower()) if c.isupper() else c for c in name).lstrip("_")

            nonlocal func_name
            func_name = func_name or pascal_to_snake(loader.__name__)

            _define_in_module(func_name, func, module)

        # We don't actually want to modify the decorated object.
        return func_or_cls

    return export_impl


def warn_deprecated(name, use_instead, remove_in, module_name=None):
    if config.INTERNAL_CORRECTNESS_CHECKS and version(polygraphy.__version__) >= version(remove_in):
        G_LOGGER.internal_error("{:} should have been removed in version: {:}".format(name, remove_in))

    full_obj_name = "{:}.{:}".format(module_name, name) if module_name else name
    warnings.warn(
        "{:} is deprecated and will be removed in Polygraphy {:}. "
        "Use {:} instead.".format(full_obj_name, remove_in, use_instead),
        DeprecationWarning,
        stacklevel=3,
    )


def deprecate(remove_in, use_instead, module_name=None, name=None):
    """
    Decorator that marks a function or class as deprecated.
    When the function or class is used, a warning will be issued.

    Args:
        remove_in (str):
                The version in which the decorated type will be removed.
        use_instead (str):
                The function or class to use instead.
        module_name (str):
                The name of the containing module. This will be used to
                generate more informative warnings.
                Defaults to None.
        name (str):
                The name of the object being deprecated.
                If not provided, this is automatically determined based on the decorated type.
                Defaults to None.
    """

    def deprecate_impl(obj):
        if config.INTERNAL_CORRECTNESS_CHECKS and version(polygraphy.__version__) >= version(remove_in):
            G_LOGGER.internal_error("{:} should have been removed in version: {:}".format(obj, remove_in))

        nonlocal name
        name = name or obj.__name__

        if inspect.ismodule(obj):

            class DeprecatedModule(object):
                def __getattr__(self, attr_name):
                    warn_deprecated(name, use_instead, remove_in, module_name)
                    self = obj
                    return getattr(self, attr_name)

                def __setattr__(self, attr_name, value):
                    warn_deprecated(name, use_instead, remove_in, module_name)
                    self = obj
                    return setattr(self, attr_name, value)

            DeprecatedModule.__doc__ = "Deprecated: Use {:} instead".format(use_instead)
            return DeprecatedModule()
        elif inspect.isclass(obj):

            class Deprecated(obj):
                def __init__(self, *args, **kwargs):
                    warn_deprecated(name, use_instead, remove_in, module_name)
                    super().__init__(*args, **kwargs)

            Deprecated.__doc__ = "Deprecated: Use {:} instead".format(use_instead)
            return Deprecated
        elif inspect.isfunction(obj):

            def wrapped(*args, **kwargs):
                warn_deprecated(name, use_instead, remove_in, module_name)
                return obj(*args, **kwargs)

            wrapped.__doc__ = "Deprecated: Use {:} instead".format(use_instead)
            return wrapped
        else:
            G_LOGGER.internal_error("deprecate is not implemented for: {:}".format(obj))

    return deprecate_impl


def export_deprecated_alias(name, remove_in, use_instead=None):
    """
    Decorator that creates and exports a deprecated alias for
    the decorated class or function.

    The alias will behave like the decorated type, except it will
    issue a deprecation warning when used.

    To create a deprecated alias for an entire module, invoke the
    function manually within the module like so:
    ::

        mod.export_deprecated_alias("old_mod_name", remove_in="0.0.0")(sys.modules[__name__])

    Args:
        name (str):
                The name of the deprecated alias.
        remove_in (str):
                The version, as a string, in which the deprecated alias will be removed.
        use_instead (str):
                The name of the function, class, or module to use instead.
                If this is ``None``, the new name will be automatically determined.
                Defaults to None.
    """
    module = inspect.getmodule(sys._getframe(1))

    def export_deprecated_alias_impl(obj):
        new_obj = deprecate(remove_in, use_instead=use_instead or obj.__name__, module_name=module.__name__, name=name)(
            obj
        )
        _define_in_module(name, new_obj, module)
        _add_to_all(name, module)
        return obj

    return export_deprecated_alias_impl
