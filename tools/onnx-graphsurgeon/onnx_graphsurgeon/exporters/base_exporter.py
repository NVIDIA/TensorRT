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

from onnx_graphsurgeon.ir.graph import Graph

class BaseExporter(object):
    @staticmethod
    def export_graph(graph: Graph):
        """
        Export a graph to some destination graph.

        Args:
            graph (Graph): The source graph to export.

        Returns:
            object: The exported graph. For example, this might be an onnx.GraphProto
        """
        raise NotImplementedError("BaseExporter is an abstract class")
