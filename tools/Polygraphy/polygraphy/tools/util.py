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

from polygraphy import mod
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
gs = mod.lazy_import("onnx_graphsurgeon")


@mod.export()
def meta_from_gs_tensors(tensors):
    """Get TensorMetadata from a list of ONNX-GraphSurgeon tensors"""
    meta = TensorMetadata()
    for tensor in tensors:
        meta.add(tensor.name, tensor.dtype, tensor.shape)
    return meta


@mod.export()
def override_input_shapes(graph, user_input_metadata):
    """
    Overrides input shapes in the model according to the provided input metadata.
    Inputs omitted from user_input_metadata are not changed.

    Shapes of intermediate tensors are cleared.
    """
    # We can leverage extract_subgraph if we make sure all the current graph inputs are preserved.
    # We need to be careful to preserve the order of graph inputs here.
    input_metadata = meta_from_gs_tensors(graph.inputs)
    input_metadata.update(user_input_metadata)
    graph = onnx_backend.extract_subgraph(graph, input_metadata)

    G_LOGGER.info("Overriding input shapes to:\n{:}".format(meta_from_gs_tensors(graph.inputs)))

    # Have to unset intermediate shapes as they may cause problems.
    tensors = graph.tensors()
    for tensor in tensors.values():
        if tensor not in graph.inputs and isinstance(tensor, gs.Variable):
            tensor.shape = None

    return graph


@mod.export()
def set_shapes_from_layerwise_meta(graph, layerwise_meta):
    for tensor in graph.tensors().values():
        if isinstance(tensor, gs.Variable) and tensor.name in layerwise_meta:
            tensor.shape = layerwise_meta[tensor.name].shape
            tensor.dtype = layerwise_meta[tensor.name].dtype
