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
import argparse
import sys
from textwrap import dedent

import polygraphy
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import LoggerArgs
from polygraphy.tools.args.base import ArgGroups


@mod.export()
class Tool:
    """
    Base class for CLI Tools.
    """

    def __init__(self, name=None):
        self.name = name
        self.arg_groups = ArgGroups()  # Populated by setup_parser based on get_subscriptions()

    def setup_parser(self, subparsers=None):
        """
        Set up a command-line argument parser.

        Args:
            subparsers (argparse.SubParsers):
                    A subparser group from argparse, like that returned by ``ArgumentParser.add_subparsers()``.
                    If this is omitted, this function will generate a new ``ArgumentParser`` instance.
                    Defaults to None.

        Returns:
            argparse.ArgumentParser:
                    The newly created parser if ``subparsers`` is not provided, or the newly created subparser otherwise.
        """
        assert self.__doc__, "No help output was provided for this tool!"

        subscriptions = self.get_subscriptions()
        # Always subscribe to the logger arguments first.
        for arg_group in [LoggerArgs()] + subscriptions:
            m_type = type(arg_group)
            self.arg_groups[m_type] = arg_group

        allow_abbrev = all(arg_group.allows_abbreviation() for arg_group in self.arg_groups.values())

        description = dedent(self.__doc__)
        if subparsers is not None:
            summary = []
            for line in description.strip().splitlines():
                if line.isspace() or not line:
                    break
                summary.append(line)
            summary = "\n".join(summary)

            parser = subparsers.add_parser(
                self.name,
                help=summary,
                add_help=True,
                description=description,
                allow_abbrev=allow_abbrev,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            parser.set_defaults(subcommand=self)
        else:
            parser = argparse.ArgumentParser(add_help=True, description=description, allow_abbrev=allow_abbrev)

        for arg_group in self.arg_groups.values():
            arg_group.register(self.arg_groups)
            # This must be done after registration, since some argument groups
            # may conditionally define arguments based on what other groups are present.
            arg_group.add_parser_args(parser)

        try:
            self.add_parser_args(parser)
        except Exception as err:
            G_LOGGER.internal_error(f"Could not register tool argument parser for: {self.name}\nNote: Error was: {err}")
        return parser

    # Implementation for `get_subscriptions`. This should be implemented by child classes instead of `get_subscriptions`
    def get_subscriptions_impl(self):
        return []

    def get_subscriptions(self):
        """
        Returns the list of argument groups this tools wishes to subscribe to.

        Returns:
            List[BaseArgs]: The list of argument groups to subscribe to.
        """
        return self.get_subscriptions_impl()

    # Implementation for `get_subtools`. This should be implemented by child classes instead of `get_subtools`
    def get_subtools_impl(self):
        return "", []

    def get_subtools(self):
        """
        Returns the subtools of this tool.

        Returns:
            Tuple[str, List[Tool]]:
                    A tuple containing the title to use for the subtools
                    and a list of subtools.
        """
        return self.get_subtools_impl()

    # Implementation for `add_parser_args`. This should be implemented by child classes instead of `add_parser_args`
    def add_parser_args_impl(self, parser):
        pass

    def add_parser_args(self, parser):
        """
        Add arguments to the command-line parser.
        This will also add arguments for any subtools reported by `get_subtools()`.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
        """
        self.add_parser_args_impl(parser)

        title, subtools = self.get_subtools()
        if subtools:
            subparsers = parser.add_subparsers(title=title, dest="subtool")
            subparsers.required = True

            for subtool in subtools:
                subtool.setup_parser(subparsers)

    def parse(self, args):
        """
        Run argument parsing for all argument groups.

        Args:
            args (Namespace):
                    The namespace returned by ``parse_args()`` or ``parse_known_args()``.
        """
        for arg_group in self.arg_groups.values():
            arg_group.parse(args)

    # Implementation for `run`. This should be implemented by child classes instead of `run`
    def run_impl(self, args):
        raise NotImplementedError("run_impl() must be implemented by child classes")

    def run(self, args):
        """
        Runs this tool with the specified arguments.
        This must be called only after calling `parse()`.

        Args:
            args (Namespace):
                    The namespace returned by ``parse_args()`` or ``parse_known_args()``.
        """
        G_LOGGER.module_info(polygraphy)

        status = self.run_impl(args)
        if status is None:
            status = 0
        return status

    # Implementation for `show_start_end_logging`. This should be implemented by child classes instead of `show_start_end_logging`
    def show_start_end_logging_impl(self, args):
        return False

    def show_start_end_logging(self, args):
        """
        Whether to display start and end times before and after running the tools.

        Args:
            args (Namespace):
                    The namespace returned by ``parse_args()`` or ``parse_known_args()``.

        Returns:
            bool: Whether to display start/end times.
        """
        return self.show_start_end_logging_impl(args)

    def main(self):
        """
        Set up and run this tool. This function serves as a replacement for a manually
        defined ``main`` method.

        Runs ``sys.exit()`` with the status code returned by ``run``. If ``run`` does
        not return anything, always exits with ``0`` (success).
        """
        parser = self.setup_parser()
        args = parser.parse_args()
        self.parse(args)
        sys.exit(self.run(args))
