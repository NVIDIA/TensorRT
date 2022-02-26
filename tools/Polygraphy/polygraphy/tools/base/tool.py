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
import argparse
import sys
from collections import OrderedDict

import polygraphy
from polygraphy import mod
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.logger import LoggerArgs


@mod.export()
class Tool(object):
    """
    Base class for CLI Tools.
    """

    def __init__(self, name=None):
        self.name = name

        # makers is a Dict[type, BaseArgs] - Maps names to subtypes of BaseArgs.
        # This will be populated with instances of BaseArgs, and parsing will
        # happen in __call__. Child classes can then access the instances directly
        # instead of reimplementing argument parsing.
        self.arg_groups = OrderedDict()
        self.subscribe_args(LoggerArgs())

    def subscribe_args(self, maker):
        """
        Subscribe to an argument group. The argument group's arguments will be added
        to the argument parser, and will be parsed prior to ``run``.

        Args:
            maker (BaseArgs): The argument group to register.
        """
        assert isinstance(maker, BaseArgs)
        m_type = type(maker)
        self.arg_groups[m_type] = maker

    def add_parser_args(self, parser):
        # Should be implemented by child classes to add custom arguments.
        pass

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

        allow_abbrev = all(not maker.disable_abbrev for maker in self.arg_groups.values())
        if subparsers is not None:
            parser = subparsers.add_parser(
                self.name, help=self.__doc__, add_help=True, description=self.__doc__, allow_abbrev=allow_abbrev
            )
            parser.set_defaults(subcommand=self)
        else:
            parser = argparse.ArgumentParser(add_help=True, description=self.__doc__, allow_abbrev=allow_abbrev)

        for maker in self.arg_groups.values():
            # Register each maker with every other maker
            for other_maker in self.arg_groups.values():
                maker.register(other_maker)

            maker.check_registered()

        # This must be done after registration, since some argument groups
        # may conditionally define arguments based on what other groups are present.
        for maker in self.arg_groups.values():
            maker.add_to_parser(parser)

        try:
            self.add_parser_args(parser)
        except Exception as err:
            G_LOGGER.internal_error(
                "Could not register tool argument parser for: {:}\nNote: Error was: {:}".format(self.name, err)
            )
        return parser

    def run(self, args):
        raise NotImplementedError("run() must be implemented by child classes")

    def __call__(self, args):
        """
        Calls this tool with the specified arguments.

        Args:
            args (Namespace):
                    The namespace returned by ``parse_args()`` or ``parse_known_args()``.
        """
        for maker in self.arg_groups.values():
            maker.parse(args)

        G_LOGGER.module_info(polygraphy)
        return self.run(args)

    def main(self):
        """
        Set up and run this tool. This function serves as a replacement for a manually
        defined ``main`` method.

        Runs ``sys.exit()`` with the status code returned by ``run``. If ``run`` does
        not return anything, always exits with ``0`` (success).
        """
        parser = self.setup_parser()
        args = parser.parse_args()
        sys.exit(self.__call__(args))
