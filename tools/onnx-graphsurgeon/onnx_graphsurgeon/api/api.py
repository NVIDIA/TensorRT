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

# Contains high-level API functions.
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.graph import Graph


def import_onnx(onnx_model: "onnx.ModelProto") -> Graph:
    """
    Import an onnx-graphsurgeon Graph from the provided ONNX model.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.

    Returns:
        Graph: A corresponding onnx-graphsurgeon Graph.
    """
    from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter

    return OnnxImporter.import_graph(onnx_model.graph, opset=OnnxImporter.get_opset(onnx_model))


def export_onnx(graph: Graph, do_type_check=True, **kwargs) -> "onnx.ModelProto":
    """
    Exports an onnx-graphsurgeon Graph to an ONNX model.

    Args:
        graph (Graph): The graph to export

    Optional Args:
        do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
        **kwargs: Additional arguments to onnx.helper.make_model

    Returns:
        onnx.ModelProto: A corresponding ONNX model.
    """
    from onnx_graphsurgeon.exporters.onnx_exporter import OnnxExporter
    import onnx

    onnx_graph = OnnxExporter.export_graph(graph, do_type_check=do_type_check)

    if "opset_imports" not in kwargs:
        kwargs["opset_imports"] = [onnx.helper.make_opsetid("", graph.opset)]

    return onnx.helper.make_model(onnx_graph, **kwargs)
