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
import os
import sys
import time
from collections import OrderedDict, defaultdict

import polygraphy
from polygraphy import config, constants, util
from polygraphy.logger import G_LOGGER


def assert_identifier(inp):
    """
    Checks if the argument can be a valid Python identifier.

    Raises a PolygraphyException if it can't.
    """
    if not inp.isidentifier():
        G_LOGGER.critical(
            "This argument must be a valid identifier. "
            "Provided argument cannot be a Python identifier: {:}".format(inp)
        )
    return inp


def safe(base_str, *args, **kwargs):
    """
    Marks a string as being safe.

    NOTE: The caller is reponsible for checking that the string is actually safe.

    Can work with format strings as well. For example:
    ::

        safe("{:} is my name", "polygraphy")
            -> "'polygraphy' is my name"

        safe("{:} is my name", inline("polygraphy"))
            -> "polygraphy is my name"
    """
    args = [repr(arg) for arg in args]
    kwargs = {key: repr(val) for key, val in kwargs.items()}
    return Script.String(base_str.format(*args, **kwargs), safe=True)


def ensure_safe(inp):
    """
    Ensures that the input is marked as a safe string (i.e. Script.String(safe=True)).
    """
    if not isinstance(inp, Script.String):
        G_LOGGER.internal_error("Input to ensure_safe must be of type Script.String, but was: {:}".format(inp))
    elif not inp.safe:
        G_LOGGER.internal_error(
            "Input string: {:} was not checked for safety. " "This is a potential security risk!".format(inp)
        )
    return inp


def inline(inp):
    """
    Marks a safe string as being inline. See Script.Inline for details
    on what this means.

    Args:
        inp (Script.String):
                The safe string to inline..
    """
    inp = ensure_safe(inp)
    inp.inline = True
    return inp


def make_invocable_impl(type_str, *args, **kwargs):
    """
    Generates a string that would invoke the type specified in
    type_str with the specified arguments.
    Skips keyword arguments that are set to ``None``.

    For example, ``make_invocable_impl("Example", None, "string", w=None, x=2, y=inline("test"))``
    would return a string: ``"Example(None, 'string', x=2, y=test)"``

    Args:
        type_str (str):
                The type to invoke.

    Returns:
        Tuple[str, bool]:
                A tuple including the `invoke` string and a boolean
                indicating whether all the arguments were default (i.e. None).
    """
    # We don't need to check obj_str for safety since we know that any inline
    # args/kwargs are already safe - other types need no checks
    obj_str, all_defaults = util.make_repr(type_str, *args, **kwargs)
    return Script.String(obj_str, safe=True, inline=True), all_defaults


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

    Examples:
        make_invocable("MyClass", 0, 1, last=3)
            -> "MyClass(0, 1, last=3)"

        make_invocable("my_func", 0, 1, last=None)
            -> "my_func(0, 1)"
    """
    return make_invocable_impl(type_str, *args, **kwargs)[0]


def make_invocable_if_nondefault(type_str, *args, **kwargs):
    """
    Similar to `make_invocable`, but will return None if all arguments are None.

    Examples:
        make_invocable_if_nondefault("MyClass", 0, 1, last=3)
            -> "MyClass(0, 1, last=3)"

        make_invocable_if_nondefault("my_func", None, None, last=None)
            -> None
    """
    obj_str, all_defaults = make_invocable_impl(type_str, *args, **kwargs)
    if all_defaults:
        return None
    return obj_str


################################# SCRIPT ##################################
# Used to generate a script that uses the Polygraphy API.


class Script(object):
    class String(object):
        """
        Represents a string that has passed security checks.

        This can be spoofed easily - the purpose is to check Polygraphy's implementations,
        not external ones.
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
                G_LOGGER.internal_error("Cannot concatenate str and Script.String. Note: str was: {:}".format(other))
            elif self.safe != other.safe:
                G_LOGGER.internal_error(
                    "Cannot concatenate unsafe string ({:}) to safe string ({:})!".format(other, self.s)
                )
            self.s += other.s
            return self

        def unwrap(self):
            return self.s

    DATA_LOADER_NAME = String("data_loader", safe=True, inline=True)

    def __init__(self, summary=None, always_create_runners=True):
        """
        Represents a Python script that uses the Polygraphy API.

        Args:
            summary (str):
                    A summary of what the script does, which will be included in the script as a comment.
            always_create_runners (bool):
                    Whether to create the list of runners even if it would be empty.
        """
        self.imports = set()
        self.from_imports = defaultdict(set)  # Dict[str, List[str]] Maps from module to imported components
        self.loaders = OrderedDict()  # Dict[str, str] Maps a string constructing a loader to a name.
        self.loader_count = defaultdict(int)  # Dict[str, int] Maps loader_id to the number of loaders sharing that ID
        self.runners = []  # List[str]
        self.preimport = []  # List[str]
        self.suffix = []  # List[str]
        self.data_loader = ""  # str Contains the DataLoader constructor
        self.summary = summary
        self.always_create_runners = always_create_runners

    def add_import(self, imports, frm=None):
        """
        Adds imports to this script

        Args:
            imports (List[str]): List of components to import
            frm (str): Module frm which to import
        """
        if frm:
            self.from_imports[frm].update(imports)
        else:
            self.imports.update(imports)

    def set_data_loader(self, data_loader_str):
        """
        Adds a data loader to this script, overwriting
        any previous data loader.

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

    def add_loader(self, loader_str, loader_id, suffix=None):
        """
        Adds a loader to the script.
        If the loader is a duplicate, returns the existing loader instead.

        Args:
            loader_str (str):
                    A string constructing the loader.
                    For security reasons, this must be generated using
                    `make_invocable` or `Script.invoke_if_non_default`.
            loader_id (str): A short human-friendly identifier for the loader

        Returns:
            str: The name of the loader added.
        """
        suffix = util.default(suffix, "")
        loader_str = ensure_safe(loader_str).unwrap()

        if loader_str in self.loaders:
            return self.loaders[loader_str]

        unique_name = loader_id + suffix
        if self.loader_count[unique_name]:
            unique_name = "{:}_{:}".format(unique_name, self.loader_count[loader_id])
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
            runner_str (str): A string constructing the runner
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
        script += "# Template auto-generated by polygraphy [v{:}] on {:} at {:}\n".format(
            polygraphy.__version__, time.strftime("%D"), time.strftime("%H:%M:%S")
        )
        script += "# Generation Command: {:}\n".format(" ".join(sys.argv))
        if self.summary:
            script += "# " + "\n# ".join(self.summary.splitlines()) + "\n"
        script += ("\n" if self.preimport else "") + "\n".join(self.preimport) + ("\n\n" if self.preimport else "")

        imports = []
        for imp in self.imports:
            imports.append("import {:}".format(imp))
        for frm, imps in self.from_imports.items():
            imps = sorted(imps)
            imports.append("from {:} import {:}".format(frm, ", ".join(imps)))
        script += "\n".join(sorted(imports)) + "\n"

        if self.data_loader:
            script += "\n# Data Loader\n"
            script += "{:} = {:}\n".format(Script.DATA_LOADER_NAME, self.data_loader)
        script += "\n"

        if self.loaders:
            script += "# Loaders\n"
        for loader, loader_name in self.loaders.items():
            script += "{:} = {:}\n".format(loader_name, loader)
        script += "\n"

        if self.runners or self.always_create_runners:
            script += "# Runners\n"
            script += "{:} = [".format(self.get_runners())
            for runner in self.runners:
                script += "\n\t{:},".format(runner)
            if self.runners:
                script += "\n"
            script += "]\n"

        script += "\n".join(self.suffix) + "\n"
        script = script.replace("\t", constants.TAB).replace("\n\n\n", "\n\n")

        G_LOGGER.super_verbose("Created script:\n{:}".format(script))
        return script

    def save(self, dest):
        """
        Save this script to the specified destination.

        Args:
            dest (file-like):
                    A file-like object that defines ``write()``, ``isatty``, and has a `name` attribute.
        """
        with dest:
            dest.write(str(self))

            path = dest.name
            # Somehow, piping fools isatty, e.g. `polygraphy run --gen-script - | cat`
            if not dest.isatty() and path not in ["<stdout>", "<stderr>"]:
                G_LOGGER.info("Writing script to: {:}".format(path))
                # Make file executable
                os.chmod(path, os.stat(path).st_mode | 0o111)
