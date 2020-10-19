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
from polygraphy.tools.base import Tool
from polygraphy.logger import G_LOGGER
import importlib

try:
    ModuleNotFoundError
except:
    ModuleNotFoundError = ImportError

TOOL_REGISTRY = []


class MissingTool(Tool):
    def __init__(self, name, err):
        self.name = name
        self.__doc__ = "Error: Tool could not be loaded. Run 'polygraphy {:}' for details".format(self.name)
        self.err = err


    def __call__(self, args):
        G_LOGGER.critical("Encountered an error when loading this tool:\n{:}".format(self.err))


def try_register_tool(module, tool_class):
    global TOOL_REGISTRY

    try:
        toolmod = importlib.import_module(module)
        tool = getattr(toolmod, tool_class)()
        TOOL_REGISTRY.append(tool)
    except Exception as err:
        TOOL_REGISTRY.append(MissingTool(tool_class.lower(), err=err))


try_register_tool("polygraphy.tools.run", "Run")
try_register_tool("polygraphy.tools.inspect", "Inspect")
try_register_tool("polygraphy.tools.surgeon", "Surgeon")
try_register_tool("polygraphy.tools.precision", "Precision")
