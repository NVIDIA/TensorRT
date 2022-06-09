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

import copy
import time
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.pluginref.references import OP_REGISTRY
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")


@mod.export()
class PluginRefRunner(BaseRunner):
    """
    Runs inference using custom CPU reference implementations
    """

    def __init__(self, graph, name=None):
        """
        Args:
            graph (Union[onnx_graphsurgeon.Graph, Callable() -> onnx_graphsurgeon.Graph]):
                    An ONNX-GraphSurgeon graph or a callable that returns one.
            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="pluginref-runner")
        self._graph = graph

    def activate_impl(self):
        self.graph, _ = util.invoke_if_callable(self._graph)

    def get_input_metadata_impl(self):
        return onnx_util.meta_from_gs_tensors(self.graph.inputs)

    def infer_impl(self, feed_dict):
        start = time.time()

        intermediate_tensors = copy.copy(feed_dict)
        for node in self.graph.nodes:
            if node.op not in OP_REGISTRY:
                G_LOGGER.critical(f"Op: {node.op} does not have a reference implementation registered!")

            intermediate_tensors.update(OP_REGISTRY[node.op](node, intermediate_tensors))

        outputs = OrderedDict()
        for out in self.graph.outputs:
            outputs[out.name] = intermediate_tensors[out.name]

        end = time.time()

        self.inference_time = end - start
        return outputs

    def deactivate_impl(self):
        del self.graph
