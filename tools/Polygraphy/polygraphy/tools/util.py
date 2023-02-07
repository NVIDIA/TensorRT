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

from polygraphy import mod
from polygraphy.logger import G_LOGGER

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")
gs = mod.lazy_import("onnx_graphsurgeon")


@mod.export()
def override_input_shapes(graph, user_input_metadata):
    """
    Overrides input shapes in the model according to the provided input metadata.
    Inputs omitted from user_input_metadata are not changed.

    Shapes of intermediate tensors are cleared.
    """
    # We can leverage extract_subgraph if we make sure all the current graph inputs are preserved.
    # We need to be careful to preserve the order of graph inputs here.
    input_metadata = onnx_util.meta_from_gs_tensors(graph.inputs)
    input_metadata.update(user_input_metadata)
    graph = onnx_backend.extract_subgraph(graph, input_metadata)

    G_LOGGER.info(f"Overriding input shapes to:\n{onnx_util.meta_from_gs_tensors(graph.inputs)}")

    # Have to unset intermediate shapes as they may cause problems.
    tensors = graph.tensors()
    for tensor in tensors.values():
        if tensor not in graph.inputs and isinstance(tensor, gs.Variable):
            tensor.shape = None

    return graph
