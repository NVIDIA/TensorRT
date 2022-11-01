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
import os
import sys
import time
from collections import OrderedDict, defaultdict

import polygraphy
from polygraphy import constants, mod, util
from polygraphy.logger import G_LOGGER


def inline_identifier(ident):
    """
    Returns an inline safe string if the provided string is an identifier and raises an exception otherwise.

    Args:
        ident (str): The string supposed to contain an identifier.

    Returns:
        Script.String: An inline safe string

    Raises:
        PolygraphyException: if the input string is not an identifier.
    """
    if not ident.isidentifier():
        G_LOGGER.critical(
            f"This argument must be a valid identifier. Provided argument cannot be a Python identifier: {ident}"
        )
    return inline(safe(ident))


@mod.export()
def safe(base_str, *args, **kwargs):
    """
    Indicates a string is safe to use and will not compromise security.

    NOTE: The caller is reponsible for checking that the string is actually safe.
    This function serves only to mark the string as such.

    Can work with format strings as well. For example:
    ::

        >>> safe("{} is my name", "polygraphy")
        "'polygraphy' is my name"
    """
    args = [repr(arg) for arg in args]
    kwargs = {key: repr(val) for key, val in kwargs.items()}
    return Script.String(base_str.format(*args, **kwargs), safe=True)


def ensure_safe(inp):
    """
    Ensures that the input is marked as a safe string (i.e. Script.String(safe=True)).
    """
    if not isinstance(inp, Script.String):
        G_LOGGER.internal_error(f"Input to ensure_safe must be of type Script.String, but was: {inp}")
    elif not inp.safe:
        G_LOGGER.internal_error(f"Input string: {inp} was not checked for safety. This is a potential security risk!")
    return inp


@mod.export()
def inline(inp):
    """
    Marks a safe string as being inline when used with other Script APIs.
    Non-inlined strings will remain strings when used with any of these APIs.
    For example:
    ::

        >>> make_invocable("print", safe("example"))
        "print('example')"

        >>> make_invocable("print", inline(safe("example")))
        "print(example)"

    Args:
        inp (Script.String): The safe string to inline.
    """
    inp = ensure_safe(inp)
    inp.inline = True
    return inp


def make_invocable_impl(type_str, *args, **kwargs):
    """
    Generates a string that would invoke the type specified in
    type_str with the specified arguments.
    Skips keyword arguments that are set to ``None``.

    For example:
    ::
        >>> make_invocable_impl("Example", None, "string", w=None, x=2, y=inline("test"))
        "Example(None, 'string', x=2, y=test)"

    Args:
        type_str (str): The type to invoke.

    Returns:
        Tuple[str, bool]:
                A tuple including the `invoke` string and a boolean
                indicating whether all the arguments were default (i.e. None).
    """
    # We don't need to check obj_str for safety since we know that any inline
    # args/kwargs are already safe - other types need no checks
    obj_str, all_args_default, all_kwargs_default = util.make_repr(type_str, *args, **kwargs)
    return Script.String(obj_str, safe=True, inline=True), all_args_default, all_kwargs_default


@mod.export()
def make_invocable(type_str, *args, **kwargs):
    """
    Creates a string representation that will invoke the specified object,
    with the specified arguments.

    Args:
        type_str (str): A string representing the object that should be invoked.
        args, kwargs:
                Arguments to pass along to the object. If a keyword argument
                is set to None, it will be omitted.

    Returns:
        str: A string representation that invokes the object specified.

    For example:
    ::

        >>> make_invocable("MyClass", 0, 1, last=3)
        "MyClass(0, 1, last=3)"

        >>> make_invocable("my_func", 0, 1, last=None)
        "my_func(0, 1)"
    """
    return make_invocable_impl(type_str, *args, **kwargs)[0]


@mod.export()
def make_invocable_if_nondefault(type_str, *args, **kwargs):
    """
    Similar to `make_invocable`, but will return ``None`` if all arguments are ``None``.

    For example:
    ::

        >>> make_invocable_if_nondefault("MyClass", 0, 1, last=3)
        "MyClass(0, 1, last=3)"

        >>> make_invocable_if_nondefault("my_func", None, None, last=None)
        None
    """
    obj_str, all_args_default, all_kwargs_default = make_invocable_impl(type_str, *args, **kwargs)
    if all_args_default and all_kwargs_default:
        return None
    return obj_str


@mod.export()
def make_invocable_if_nondefault_kwargs(type_str, *args, **kwargs):
    """
    Similar to `make_invocable_if_nondefault`, but will return ``None``
    if all keyword arguments are ``None``, even if positional arguments are not.

    For example:
    ::

        >>> make_invocable_if_nondefault("MyClass", 0, 1, last=3)
        "MyClass(0, 1, last=3)"

        >>> make_invocable_if_nondefault("my_func", 0, 1, last=None)
        None
    """
    obj_str, _, all_kwargs_default = make_invocable_impl(type_str, *args, **kwargs)
    if all_kwargs_default:
        return None
    return obj_str


################################# SCRIPT ##################################
# Used to generate a script that uses the Polygraphy API.


@mod.export()
class Script:
    """
    Represents a Python script that uses the Polygraphy API.
    """

    class String:
        """
        Represents a string that has passed Polygraphy's security checks.
        """

        def __init__(self, s, safe=False, inline=False):
            self.s = s
            self.safe = safe
            self.inline = inline

        def __str__(self):
            return str(self.s)

        def __repr__(self):
            if self.inline:
                # Since only safe strings can be marked inline, self.safe is always
                # True in this branch, so no need to check it.
                return str(self.s)
            return repr(self.s)

        def __iadd__(self, other):
            if not isinstance(other, Script.String):
                G_LOGGER.internal_error(f"Cannot concatenate str and Script.String. Note: str was: {other}")
            elif self.safe != other.safe:
                G_LOGGER.internal_error(f"Cannot concatenate unsafe string ({other}) to safe string ({self.s})!")
            self.s += other.s
            return self

        def unwrap(self):
            """
            Returns the underlying string object.

            Returns:
                str
            """
            return self.s

    DATA_LOADER_NAME = String("data_loader", safe=True, inline=True)

    def __init__(self, summary=None, always_create_runners=True):
        """
        Args:
            summary (str):
                    A summary of what the script does, which will be included in the script as a comment.
            always_create_runners (bool):
                    Whether to create the list of runners even if it would be empty.
        """
        self.imports = {}  # Dict[str, Set[str, str]]: Maps from: {(import, as), ...}
        self.loaders = OrderedDict()  # Dict[str, str] Maps a string constructing a loader to a name.
        self.loader_count = defaultdict(int)  # Dict[str, int] Maps loader_id to the number of loaders sharing that ID
        self.runners = []  # List[str]
        self.preimport = []  # List[str]
        self.suffix = []  # List[str]
        self.data_loader = ""  # str Contains the DataLoader constructor
        self.summary = summary
        self.always_create_runners = always_create_runners

    def add_import(self, imports, frm=None, imp_as=None):
        """
        Adds imports to this script.

        For example:
        ::

            script.add_import("numpy") # Equivalent to: import numpy
            script.add_import("numpy", imp_as="np") # Equivalent to: import numpy as np
            script.add_import("TrtRunner", frm="polygraphy.backend.trt") # Equivalent to: from polygraphy.backend.trt import TrtRunner

        Args:
            imports (Union[str, List[str]]): Object/submodule or list of objects/submodules to import.
            frm (str): Module from which to import.
            imp_as (str): Name to import as. When this is specified, ``imports`` must be a string and not a list.
        """
        if isinstance(imports, str):
            imports = {imports}

        if imp_as and len(imports) > 1:
            G_LOGGER.internal_error("When `imp_as` is specified, `imports` must be a string and not a list")

        if frm not in self.imports:
            self.imports[frm] = set()
        self.imports[frm].update({(imp, imp_as) for imp in imports})

    def set_data_loader(self, data_loader_str):
        """
        Adds a data loader to this script, overwriting any previous data loader.

        Args:
            data_loader_str (str): A string constructing the data loader.

        Returns:
            str:
                The name of the data loader in the script, or None if the
                provided data loader is empty.
        """
        if not data_loader_str:
            return None
        data_loader_str = ensure_safe(data_loader_str).unwrap()

        self.data_loader = data_loader_str
        return Script.DATA_LOADER_NAME

    def add_loader(self, loader_str, loader_id):
        """
        Adds a loader to the script.
        If the loader is a duplicate, returns the existing loader instead.

        Args:
            loader_str (str):
                    A string constructing the loader.
                    For security reasons, this must be generated using
                    ``make_invocable`` or ``make_invocable_if_nondefault``.
            loader_id (str):
                    A short human-readable identifier for the loader.

        Returns:
            str: The name of the loader added.
        """
        loader_str = ensure_safe(loader_str).unwrap()

        if loader_str in self.loaders:
            return self.loaders[loader_str]

        unique_name = loader_id
        if self.loader_count[unique_name]:
            unique_name = f"{unique_name}_{self.loader_count[loader_id]}"
        unique_name = Script.String(unique_name, safe=True, inline=True)

        self.loader_count[loader_id] += 1
        self.loaders[loader_str] = unique_name
        return unique_name

    def get_runners(self):
        return Script.String("runners", safe=True, inline=True)

    def add_runner(self, runner_str):
        """
        Adds a runner to the script.

        Args:
            runner_str (str):
                    A string constructing the runner.
                    For example, this may be generated by ``make_invocable``.
        """
        runner_str = ensure_safe(runner_str).unwrap()
        self.runners.append(runner_str)

    def append_preimport(self, line):
        """
        Append a line to the pre-import prefix of the script.

        Args:
            line (str): The line to append.
        """
        line = ensure_safe(line).unwrap()
        self.preimport.append(line)

    def append_suffix(self, line):
        """
        Append a line to the suffix of the script

        Args:
            line (str): The line to append.
        """
        line = ensure_safe(line).unwrap()
        self.suffix.append(line)

    def __str__(self):
        script = "#!/usr/bin/env python3\n"
        script += f"# Template auto-generated by polygraphy [v{polygraphy.__version__}] on {time.strftime('%D')} at {time.strftime('%H:%M:%S')}\n"
        script += f"# Generation Command: {' '.join(sys.argv)}\n"
        if self.summary:
            script += "# " + "\n# ".join(self.summary.splitlines()) + "\n"
        script += ("\n" if self.preimport else "") + "\n".join(self.preimport) + ("\n\n" if self.preimport else "")

        has_external_import = False
        imports = []
        for frm, imps in self.imports.items():
            imps = sorted(imps)
            is_external_import = False
            if frm is not None:
                # NOTE: We do not currently translate 'from' imports to `lazy_import`.
                imps = [f"{imp}" if imp_as is None else f"{imp} as {imp_as}" for imp, imp_as in imps]
                imports.append(f"from {frm} import {', '.join(imps)}")
            else:
                # When `frm` is None, we want to treat each import separately.
                # Translate external imports to use `mod.lazy_import()`.
                for imp, imp_as in imps:
                    is_external_import = "polygraphy" not in imp
                    if is_external_import:
                        imp_as = imp_as or imp
                        imports.append(f"{imp_as} = mod.lazy_import({repr(imp)})")
                    else:
                        imports.append(f"import {imp}{'' if imp_as is None else f' as {imp_as}'}")
            has_external_import |= is_external_import

        if has_external_import:
            script += "from polygraphy import mod" + "\n"
        script += "\n".join(sorted(imports)) + "\n"

        if self.data_loader:
            script += "\n# Data Loader\n"
            script += f"{Script.DATA_LOADER_NAME} = {self.data_loader}\n"
        script += "\n"

        if self.loaders:
            script += "# Loaders\n"
        for loader, loader_name in self.loaders.items():
            script += f"{loader_name} = {loader}\n"
        script += "\n"

        if self.runners or self.always_create_runners:
            script += "# Runners\n"
            script += f"{self.get_runners()} = ["
            for runner in self.runners:
                script += f"\n{constants.TAB}{runner},"
            if self.runners:
                script += "\n"
            script += "]\n"

        script += "\n".join(self.suffix) + "\n"
        script = script.replace("\n\n\n", "\n\n")

        G_LOGGER.super_verbose(f"Created script:\n{script}")
        return script

    def save(self, dest):
        """
        Save this script to the specified destination.

        Args:
            dest (file-like):
                    A file-like object that defines ``write()``, ``flush()``, ``isatty``, and has a ``name`` attribute.
        """
        path = dest.name
        # Somehow, piping fools isatty, e.g. `polygraphy run --gen-script - | cat`
        is_file = not dest.isatty() and path not in ["<stdout>", "<stderr>"]

        dest.write(str(self))
        dest.flush()

        if is_file:
            G_LOGGER.info(f"Writing script to: {path}")
            # Make file executable
            os.chmod(path, os.stat(path).st_mode | 0o111)
