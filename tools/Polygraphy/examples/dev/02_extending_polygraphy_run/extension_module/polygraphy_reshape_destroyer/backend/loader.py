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
This file defines the `ReplaceReshapes` loader, which takes an ONNX-GraphSurgeon graph
and replaces any no-op Reshapes with Identity nodes.

The loader implements the standard `BaseLoader` interface.
"""

from typing import Callable, Union

from polygraphy import mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.logger import G_LOGGER

# For external dependencies besides `polygraphy` or any Polygraphy backends
# (any backend besides `polygraphy.backend.base`), you should use `mod.lazy_import`.
#
# This will enable Polygraphy to automatically install dependencies at runtime if required, and
# will avoid creating a hard dependency on external packages.
#
# NOTE: As the name implies, `lazy_import` does *not* import the module until the first time it is
#       accessed. Thus, you should be careful to avoid an antipattern like:
#
#   my_module = mod.lazy_import("my_module")
#   submodule = my_module.submodule
#
# The second line will trigger an immediate import of `my_module`.
# Instead, use something like:
#
#   submodule = mod.lazy_import("my_module.submodule")
#
gs = mod.lazy_import("onnx_graphsurgeon")


# `mod.export()` adds the decorated class or function to this module's __all__ attribute.
# When we do an `import *` from the `__init__.py` file in this submodule, this will ensure
# that only the decorated objects are exported.
#
# NOTE: We use `funcify=True` so that an immediately evaluated functional loader (called `replace_reshapes`)
#       will be automatically generated for us. This won't be used by the command-line toolkit, but could
#       be useful if this module is ever used via the Python API.
#
@mod.export(funcify=True)
class ReplaceReshapes(BaseLoader):
    """
    Functor that replaces no-op Reshape nodes in an ONNX-GraphSurgeon graph with Identity.
    """

    def __init__(self, graph: Union[gs.Graph, Callable[[], gs.Graph]], rename_nodes: bool = None):
        """
        Replaces no-op Reshape nodes in an ONNX-GraphSurgeon graph with Identity.

        Args:
            graph (Union[gs.Graph, Callable() -> gs.Graph]):
                    An ONNX-GraphSurgeon graph or a callable that returns one.
            rename_nodes (bool):
                    Whether to rename Reshape nodes when we convert them to Identity.
                    Defaults to False.
        """
        # In addition to accepting a `gs.Graph` directly, we will also support callables, e.g. Polygraphy loaders.
        # This will allow our loader to be composed together with other Polygraphy loaders.
        #
        # Since the `graph` parameter may be a callable, we'll assign it to a "private" member, i.e. prefixed with '_',
        # to avoid conflating it with an actual `gs.Graph`.
        #
        self._graph = graph

        # See the comment in `util.default` for details on why we use this approach rather than standard Python default parameters.
        self.rename_nodes = util.default(rename_nodes, False)

    # The `call_impl` method is responsible for doing the actual work of the loader.
    def call_impl(self):
        """
        Returns:
            gs.Graph: The graph with no-op Reshape nodes replaced by Identity.
        """
        # As mentioned before, `self._graph` could be a callable, so we invoke it if needed here.
        #
        # TIP: The second value returned by `invoke_if_callable` (unused here) is a boolean indicating
        #      whether the argument was indeed a callable.
        #
        graph, _ = util.invoke_if_callable(self._graph)

        for node in graph.nodes:
            if node.op != "Reshape":
                continue

            # We can't determine that a Reshape is a no-op unless the new shape is known
            # prior to inference-time, i.e. a constant.
            if not isinstance(node.inputs[1], gs.Constant):
                continue

            # Reshape is only a no-op when the new shape is the same as the old shape.
            new_shape = node.inputs[1].values
            if list(node.inputs[0].shape) != list(new_shape):
                continue

            # Replace no-op reshape with an Identity. We can simply edit the operator name,
            # clear any attributes, and then delete the second input.
            G_LOGGER.info(f"Replacing no-op reshape: {node.name} with an Identity node")
            if self.rename_nodes:
                node.name += "_destroyed"
                G_LOGGER.info(f"Renamed Identity node to: {node.name}")

            node.op = "Identity"
            node.attrs.clear()
            del node.inputs[1]

        # Finally, clean up the graph to remove any dangling tensors and return it.
        graph.cleanup()
        return graph
