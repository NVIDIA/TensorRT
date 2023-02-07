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
import importlib

from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool

TOOL_REGISTRY = []


class MissingTool(Tool):
    def __init__(self, name, err):
        super().__init__(name)
        self.err = err
        # NOTE: When modifying this error message, make sure to update the checks in
        # tests/test_dependencies.py so that we don't miss errors!
        self.__doc__ = (
            f"[!] This tool could not be loaded due to an error:\n{self.err}\nRun 'polygraphy {self.name}' for details."
        )

    def __call__(self, args):
        G_LOGGER.critical(f"Encountered an error when loading this tool:\n{self.err}")


def try_register_tool(module, tool_class):
    global TOOL_REGISTRY

    try:
        toolmod = importlib.import_module(module)
        ToolClass = getattr(toolmod, tool_class)
        TOOL_REGISTRY.append(ToolClass())
    except Exception as err:
        G_LOGGER.internal_error(
            f"Could not load command-line tool: {tool_class.lower()}.\nNote: Error was: {err}"
        )
        TOOL_REGISTRY.append(MissingTool(tool_class.lower(), err=err))


try_register_tool("polygraphy.tools.run", "Run")
try_register_tool("polygraphy.tools.convert", "Convert")
try_register_tool("polygraphy.tools.inspect", "Inspect")
try_register_tool("polygraphy.tools.surgeon", "Surgeon")
try_register_tool("polygraphy.tools.template", "Template")
try_register_tool("polygraphy.tools.debug", "Debug")
try_register_tool("polygraphy.tools.data", "Data")

# Check that tool names are unique
tool_names = [tool.name for tool in TOOL_REGISTRY]
duplicates = {name for name in tool_names if tool_names.count(name) > 1}
if duplicates:
    G_LOGGER.internal_error(f"Multiple tools have the same name. Duplicate tool names found: {duplicates}")
