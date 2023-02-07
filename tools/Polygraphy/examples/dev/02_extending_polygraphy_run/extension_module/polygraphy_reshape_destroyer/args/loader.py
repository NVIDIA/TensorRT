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

"""
This file defines the `ReplaceReshapeArgs` argument group, which manages
command-line options that control the `ReplaceReshape` loader.

The argument group implements the standard `BaseArgs` interface.
"""


from polygraphy import mod
from polygraphy.tools.args import OnnxLoadArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable


# NOTE: Our argument groups can depend on any argument groups that `polygraphy run` subscribes to.
#       See `polygraphy/tools/run/run.py` for the complete list.
#       In this case, we'll take advantage of OnnxLoadArgs to load the ONNX model for us.
@mod.export()
class ReplaceReshapeArgs(BaseArgs):
    # Argument groups employ a standardized format for their docstrings:
    #
    #  - The first line must include a title and description separated by a colon (':').
    #   The description should answer the question: "What is this argument group responsible for?".
    #
    # - If our argument group depends on other argument groups, we must also add a `Depends on:` section
    #   enumerating our dependencies.
    #
    # See the `BaseArgs` docstring for more details on the expected format.
    #
    """
    ONNX Reshape Replacement: replacing no-op Reshape nodes with Identity in ONNX models

    Depends on:

        - OnnxLoadArgs
    """

    # Add any command-line options we want for our loader.
    def add_parser_args_impl(self):
        # The `BaseArgs` constructor will automatically set `self.group` to an `argparse`
        # argument group to which we can add our command-line options.
        #
        # NOTE: In order to prevent collisions with other Polygraphy options, we'll prefix all the options
        #       we add with `--res-des`, short for `REShape DEStroyer`.
        self.group.add_argument(
            "--res-des-rename-nodes",
            help="Whether to rename nodes if they are replaced",
            action="store_true",
            default=None,
        )

    # Next, we'll implement parsing code for the arguments we added.
    # This will allow our argument group to be used by other argument groups.
    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            rename_nodes (bool): Whether to rename nodes if they are replaced.
        """
        # We'll use `args_util.get` to retrieve attributes from `args`, which will return `None` if the attribute is not found.
        # This will ensure that our argument group will continue to work even if a command-line option is disabled in the code.
        self.rename_nodes = args_util.get(args, "res_des_rename_nodes")

    # Finally, we can implement the logic which will add code to the script.
    def add_to_script_impl(self, script):
        # First, we'll use `OnnxLoadArgs` to add code to load the ONNX model.
        # This will ensure that any options related to ONNX model loading are respected by our argument group.
        # `OnnxLoadArgs`'s `add_to_script` method will return the name of a loader that loads an ONNX model.
        loader_name = self.arg_groups[OnnxLoadArgs].add_to_script(script)

        # Next, we'll add Polygraphy's `GsFromOnnx` loader so that we can convert the ONNX model to an
        # ONNX-GraphSurgeon graph that can be fed to our custom loader.
        #
        # First, import the loader from Polygraphy:
        script.add_import(imports=["GsFromOnnx"], frm="polygraphy.backend.onnx")
        # Next, invoke the loader with arguments (in this case, the ONNX model loader name), and add it to the script.
        loader_name = script.add_loader(make_invocable("GsFromOnnx", loader_name), loader_id="gs_from_onnx")

        # Finally, add the ReplaceReshapeArgs loader.
        # Unlike the Polygraphy loaders, we'll need to import our loader from the extension module.
        script.add_import(imports=["ReplaceReshapes"], frm="polygraphy_reshape_destroyer.backend")
        # Add the loader and return the ID so that it can be used by subsequent loaders or runners.
        # NOTE: We can provide additional positional and keyword arguments to `make_invocable` to pass them on to the loader.
        return script.add_loader(
            make_invocable("ReplaceReshapes", loader_name, rename_nodes=self.rename_nodes),
            loader_id="replace_reshapes",
        )
