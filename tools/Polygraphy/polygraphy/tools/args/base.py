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

from typing import Tuple

from polygraphy import util
from polygraphy.common.interface import TypedDict
from polygraphy.logger import G_LOGGER


class ArgGroups(TypedDict(lambda: type, lambda: BaseArgs)):
    """
    Maps argument group types to argument groups.
    """

    pass


class BaseArgs:
    """
    Adds a arguments to a command-line parser, and provides capabilities to create
    Polygraphy objects based on the arguments.

    Child classes that add options must define a docstring that includes a section header for the argument group,
    a brief description which should complete the sentence: "Options related to ...", and finally, any dependencies:
    ::

        Section Header: Description

        Depends on:

            - OtherArgs0
            - OtherArgs1: <additional info: condition under which it is needed, or reason for dependency>
            - OtherArgs2: [Optional] <behavior if available>

        <Optional Additional Documentation>

    For example:
    ::

        TensorRT Engine: loading TensorRT engines.

        Depends on:

            - ModelArgs
            - TrtLoadPluginsArgs
            - TrtLoadNetworkArgs: if building engines
            - TrtConfigArgs: if building engines
            - TrtSaveEngineArgs: if allow_saving == True

    The section header and description will be used to popluate the tool's help output.
    """

    def __init__(self):
        self.group = None
        """The ``argparse`` argument group associated with this argument group"""

        # This is populated by the tool base class via ``register()``.
        self.arg_groups = None
        """ArgGroups: Maps argument group types to argument groups"""

    # Implementation for `allows_abbreviation`. Derived classes should override this instead of `allows_abbreviation`.
    def allows_abbreviation_impl(self):
        return True

    def allows_abbreviation(self):
        """
        Whether to allow abbreviated options. When this is enabled, a prefix of an option can be used instead of
        specifying the entire option. For example, an ``--iterations`` could be specified with just ``--iter``.
        This breaks ``argparse.REMAINDER``, so any argument groups using that should disable this.
        The default implementation returns True.

        Returns:
            bool
        """
        return self.allows_abbreviation_impl()

    def register(self, arg_groups):
        """
        Registers a dictionary of all available argument groups with this argument group.

        Args:
            arg_groups (ArgGroups): Maps argument group types to argument groups.
        """
        self.arg_groups = arg_groups

    # Implementation for `add_parser_args`. Derived classes should override this instead of `add_parser_args`.
    # The `self.group` attribute will be populated with an argparse argument group before this is called.
    def add_parser_args_impl(self):
        pass

    def add_parser_args(self, parser):
        """
        Add arguments to a command-line parser.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
        """
        title, _, desc = self.__doc__.strip().splitlines()[0].rpartition(":")
        if not title or not desc:
            G_LOGGER.internal_error(
                "Incorrect docstring format, expected 'Title: Description'.\n"
                f"Note: Docstring was:\n{self.__doc__}.\n"
                "See BaseArgs documentation for details."
            )

        self.group = parser.add_argument_group(title.strip(), f"Options related to {desc.strip()}")

        self.add_parser_args_impl()

        # Remove empty groups from the parser.
        if self.group._action_groups:
            G_LOGGER.internal_error("Argument groups should not create subgroups!")

        if not self.group._actions:
            parser._action_groups.remove(self.group)
            self.group = None

    # Implementation for `parse`. Derived classes should override this instead of `parse`.
    def parse_impl(self, args):
        pass

    def parse(self, args):
        """
        Parses relevant arguments from command-line arguments and populates corresponding
        attributes of this argument group.

        Args:
            args: Arguments provided by argparse.
        """
        self.parse_impl(args)

    # Implementation for `add_to_script`. Derived classes should override this instead of `add_to_script`.
    def add_to_script_impl(self, script, *args, **kwargs):
        raise NotImplementedError()

    def add_to_script(self, script, *args, **kwargs) -> str:
        """
        Adds code to the given script that performs the functionality provided by this argument group.

        For example, ``TrtConfigArgs`` would add a call to ``CreateConfig``.

        Args:
            script (polygraphy.tools.script.Script):
                    A script to which code should be added.

        Returns:
            str: The name of the variable that was modified or added in the script.
        """
        return self.add_to_script_impl(script, *args, **kwargs)


class BaseRunnerArgs(BaseArgs):
    """
    Similar to BaseArgs, but meant specifically for argument groups dealing with runners.
    """

    def add_to_script(self, script) -> str:
        """
        Returns:
            str: The name of the list of runners in the script.
        """
        self.add_to_script_impl(script)
        return script.get_runners()

    # Implementation for `get_name_opt`. Derived classes should override this instead of `get_name_opt`.
    def get_name_opt_impl(self):
        raise NotImplementedError()

    def get_name_opt(self) -> Tuple[str, str]:
        """
        Returns a tuple containing a human readable name of the runner and the name of the command-line option (*without* leading dashes)
        that should be used to select the runner controlled by this argument group.

        For example: ``("TensorRT", "trt")``.
        """
        return self.get_name_opt_impl()

    # Implementation for `get_extra_help_text`. Derived classes should override this instead of `get_extra_help_text`.
    def get_extra_help_text_impl(self):
        return ""

    def get_extra_help_text(self) -> str:
        """
        Returns any extra help text to display in the tool help output for this runner.
        """
        return self.get_extra_help_text_impl()
