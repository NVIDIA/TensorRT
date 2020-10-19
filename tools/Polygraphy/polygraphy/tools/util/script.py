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
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.util import misc
from polygraphy.common import constants
import polygraphy

from collections import OrderedDict, defaultdict
import time
import sys
import os

# Special marker that indicates a string should be literally copied into the script, not wrapped in quotes
class Inline(str):
    pass

# Processes arguments for an object in the generated script.
def process_args(args, kwargs):
    def process_arg(arg):
        if arg is not None:
            if isinstance(arg, str) and not isinstance(arg, Inline):
                return "'{:}'".format(arg)
            if isinstance(arg, OrderedDict):
                return dict(arg)
        return arg
    args = [process_arg(arg) for arg in args]
    kwargs = {key: process_arg(val) for key, val in kwargs.items()}
    return args, kwargs


def invoke_impl(type_str, *args, **kwargs):
    args, kwargs = process_args(args, kwargs)
    obj_str = "{:}(".format(type_str)
    obj_str += ", ".join(args)

    all_defaults = all([arg is None for arg in args])
    is_first = len(args) == 0

    for key, val in kwargs.items():
        if val is not None:
            all_defaults = False
            if not is_first:
                obj_str += ", "
            obj_str += "{:}={:}".format(key, val)
            is_first = False
    obj_str += ")"
    return obj_str, all_defaults


################################# SCRIPT ##################################
# Used to generate a script that uses the Polygraphy API.

class Script(object):
    @staticmethod
    def format_str(base_str, *args, **kwargs):
        """
        Like str.format(), but includes string arguments with quotes, unless
        they are marked Inline.

        Examples:
            format_str("{:} is my name", "polygraphy")
                -> "'polygraphy' is my name"

            format_str("{:} is my name", Inline("polygraphy"))
                -> "polygraphy is my name"
        """
        args, kwargs = process_args(args, kwargs)
        return base_str.format(*args, **kwargs)


    @staticmethod
    def invoke(type_str, *args, **kwargs):
        """
        Creates a string representation that will invoke the specified object,
        with the specified arguments.

        Args:
            type_str (str): A string representing the object that should be invoked.
            *args and **kwargs:
                    Arguments to pass along to the object. If a keyword argument
                    is set to None, it will be omitted.

        Returns:
            str: A string representation that invokes the object specified.

        Examples:
            Script.invoke("MyClass", 0, 1, last=3)
                -> "MyClass(0, 1, last=3)"

            Script.invoke("my_func", 0, 1, last=None)
                -> "my_func(0, 1)"
        """
        return invoke_impl(type_str, *args, **kwargs)[0]


    @staticmethod
    def invoke_if_nondefault(type_str, *args, **kwargs):
        """
        Similar to `invoke`, but will return None if all arguments are None.

        Examples:
            Script.invoke("MyClass", 0, 1, last=3)
                -> "MyClass(0, 1, last=3)"

            Script.invoke("my_func", None, None, last=None)
                -> None
        """
        obj_str, all_defaults = invoke_impl(type_str, *args, **kwargs)
        if all_defaults:
            return None
        return obj_str


    def __init__(self, summary=None):
        """
        Represents a Python script that uses the Polygraphy API.


            summary (str):
                    A summary of what the script does, which will be included in the script as a comment.
        """
        self.imports = set()
        self.from_imports = defaultdict(set) # Dict[str, List[str]] Maps from module to imported components (e.g. {"polygraphy.util": ["misc", "cli"]})
        self.loaders = OrderedDict() # Dict[str, str] Maps a string constructing a loader to a name.
        self.loader_count = defaultdict(int) # Dict[str, int] Maps loader_id to the number of loaders sharing that ID
        self.runners = [] # List[str]
        self.preimport = [] # List[str]
        self.prefix = [] # List[str]
        self.suffix = [] # List[str]
        self.summary = summary


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


    def add_loader(self, loader_str, loader_id, suffix=None):
        """
        Adds a loader to the script.
        If the loader is a duplicate, returns the existing loader instead.

        Args:
            loader_str (str): A string constructing the loader.
            loader_id (str): A short human-friendly identifier for the loader

        Returns:
            str: The name of the loader added.
        """
        suffix = misc.default_value(suffix, "")

        if loader_str in self.loaders:
            return self.loaders[loader_str]

        unique_name = loader_id + suffix
        if self.loader_count[unique_name]:
            unique_name = "{:}_{:}".format(unique_name, self.loader_count[loader_id])
        unique_name = Inline(unique_name)

        self.loader_count[loader_id] += 1
        self.loaders[loader_str] = unique_name
        return unique_name


    def get_runners(self):
        return Inline("runners")


    def add_runner(self, runner_str):
        """
        Adds a runner to the script.

        Args:
            runner_str (str): A string constructing the runner
        """
        self.runners.append(runner_str)


    def append_preimport(self, line):
        """
        Append a line to the pre-import prefix of the script

        Args:
            line (str): The line to append.
        """
        self.preimport.append(line)


    def append_prefix(self, line):
        """
        Append a line to the prefix of the script

        Args:
            line (str): The line to append.
        """
        self.prefix.append(line)


    def append_suffix(self, line):
        """
        Append a line to the suffix of the script

        Args:
            line (str): The line to append.
        """
        self.suffix.append(line)


    def __str__(self):
        script = "#!/usr/bin/env python3\n"
        script += "# Template auto-generated by polygraphy [v{:}] on {:} at {:}\n".format(
                    polygraphy.__version__, time.strftime("%D"), time.strftime("%H:%M:%S"))
        script += "# Generation Command: {:}\n".format(" ".join(sys.argv))
        if self.summary:
            script += "# " + "\n# ".join(self.summary.splitlines()) + "\n"
        script += "\n".join(self.preimport) + ("\n\n" if self.preimport else "")

        for imp in sorted(self.imports):
            script += "import {:}\n".format(imp)
        for frm, imps in sorted(self.from_imports.items()):
            imps = sorted(imps)
            script += "from {:} import {:}\n".format(frm, ", ".join(imps))
        script += "\n"

        script += "\n".join(self.prefix) + ("\n" if self.prefix else "")

        if self.loaders:
            script += "# Loaders\n"
        for loader, loader_name in self.loaders.items():
            script += "{:} = {:}\n".format(loader_name, loader)
        script += "\n"

        script += "# Runners\n"
        script += "{:} = [\n".format(self.get_runners())
        for runner in self.runners:
            script += "{:}{:},\n".format(constants.TAB, runner)
        script += "]\n"

        script += "\n".join(self.suffix) + "\n"

        G_LOGGER.super_verbose("Created script:\n{:}".format(script))
        return script
