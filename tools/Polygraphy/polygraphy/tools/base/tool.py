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
from collections import OrderedDict
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.logger import LoggerArgs


class Tool(object):
    """
    """
    def __init__(self, name):
        self.name = name

        # makers is a Dict[type, BaseArgs] - Maps names to subtypes of BaseArgs.
        # This will be populated with instances of BaseArgs, and parsing will
        # happen in __call__. Child classes can then access the instances directly
        # instead of reimplementing argument parsing.
        self.makers = OrderedDict()
        self.subscribe_args(LoggerArgs())


    def subscribe_args(self, maker):
        assert isinstance(maker, BaseArgs)
        m_type = type(maker)
        self.makers[m_type] = maker
        return self.makers[m_type]


    def add_parser_args(self, parser):
        # Should be implemented by child classes to add custom arguments.
        pass


    def setup_parser(self, subparsers):
        add_help = True
        tool_help = self.__doc__ if self.__doc__ is not None else ""
        if not tool_help.strip():
            add_help = False
            tool_help = "<No help output provided>"

        parser = subparsers.add_parser(self.name, help=tool_help, add_help=add_help, description=tool_help)
        parser.set_defaults(subcommand=self)

        self.add_parser_args(parser)

        for maker in self.makers.values():
            maker.add_to_parser(parser)
            # Register each maker with every other maker
            for other_maker in self.makers.values():
                maker.register(other_maker)

            maker.check_registered()


    def run(self, args):
        raise NotImplementedError("run() must be implemented by child classes")


    def __call__(self, args):
        for maker in self.makers.values():
            maker.parse(args)
        self.run(args)
