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

from polygraphy import util


class BaseArgs(object):
    """
    Adds a arguments to a command-line parser, and provides capabilities to create
    Polygraphy objects based on the arguments.
    """

    def __init__(self, disable_abbrev=None):
        self.disable_abbrev = util.default(disable_abbrev, False)

    def add_to_parser(self, parser):
        """
        Add arguments to a command-line parser.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
        """
        pass

    def parse(self, args):
        """
        Parses relevant arguments from command-line arguments.

        Args:
            args: Arguments provided by argparse.
        """
        pass

    def register(self, maker):
        """
        Registers another argument group with this one.
        This can be used to pick up dependencies for example.

        Args:
            maker (BaseArgs): Another argument group.
        """
        pass

    def check_registered(self):
        """
        Called after all `register()` calls to make dependency checks easier.
        """
        pass
