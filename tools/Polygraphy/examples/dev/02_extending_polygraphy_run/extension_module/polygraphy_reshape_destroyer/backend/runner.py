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
This file defines the `IdentityOnlyRunner` runner, which takes an ONNX-GraphSurgeon graph
containing only Identity nodes and runs inference.

The runner implements the standard `BaseRunner` interface.
"""

import copy
import time
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER


@mod.export()
class IdentityOnlyRunner(BaseRunner):
    """
    Runs inference using custom Python code.
    Only supports models containing only Identity nodes.
    """

    def __init__(self, graph, name=None, speed: str = None):
        """
        Args:
            graph (Union[onnx_graphsurgeon.Graph, Callable() -> onnx_graphsurgeon.Graph]):
                    An ONNX-GraphSurgeon graph or a callable that returns one.
            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
            speed (str):
                    How fast to run inference. Should be one of: ["slow", "medium", "fast"].
                    Defaults to "fast".
        """
        super().__init__(name=name, prefix="pluginref-runner")
        self._graph = graph

        self.speed = util.default(speed, "fast")

        VALID_SPEEDS = ["slow", "medium", "fast"]
        if self.speed not in VALID_SPEEDS:
            # Like Polygraphy, extension modules should use `G_LOGGER.critical()` for any unrecoverable errors.
            G_LOGGER.critical(f"Invalid speed: {self.speed}. Note: Valid speeds are: {VALID_SPEEDS}")

    def activate_impl(self):
        # As with the loader, the `graph` argument could be either a `gs.Graph` or a callable that
        # returns one, such as a loader, so we try to call it.
        self.graph, _ = util.invoke_if_callable(self._graph)

    #
    # All the methods from this point forward are guaranteed to be called only after `activate()`,
    # so we can assume that `self.graph` will be available.
    #

    def get_input_metadata_impl(self):
        # Input metadata is used by Polygraphy's default data loader to determine the required
        # shapes and datatypes of the input buffers.
        meta = TensorMetadata()
        for tensor in self.graph.inputs:
            meta.add(tensor.name, tensor.dtype, tensor.shape)
        return meta

    def infer_impl(self, feed_dict):
        start = time.time()

        # Since our runner only supports Identity, all we need to do for inference is bind node outputs to their inputs.
        # We'll begin with a copy of the input tensors:
        tensor_values = copy.copy(feed_dict)

        for node in self.graph.nodes:
            # We don't support non-Identity nodes, so we'll report an error if we see one
            if node.op != "Identity":
                G_LOGGER.critical(
                    f"Encountered an unsupported type of node: {node.op}."
                    "Note: This runner only supports Identity nodes!"
                )

            inp_tensor = node.inputs[0]
            out_tensor = node.outputs[0]
            # The output of an Identity node should be identical to its input.
            tensor_values[out_tensor.name] = tensor_values[inp_tensor.name]

        # Find the output tensors based on `self.graph.outputs` and create a dictionary that we can return:
        outputs = OrderedDict()
        for out in self.graph.outputs:
            outputs[out.name] = tensor_values[out.name]

        # Next we'll implement our artifical delay so that we can see amazing performance gains in "fast" mode!
        delay = {"slow": 1.0, "medium": 0.5, "fast": 0.0}[self.speed]
        time.sleep(delay)

        end = time.time()

        # In order to allow Polygraphy to report inference times accurately, runners are responsible for reporting
        # their own inference time. This is done by setting the `self.inference_time` attribute.
        self.inference_time = end - start
        return outputs

    def deactivate_impl(self):
        del self.graph
