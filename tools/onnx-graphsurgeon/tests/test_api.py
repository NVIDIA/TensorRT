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

from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter
import onnx_graphsurgeon as gs

from onnx_models import identity_model

import tempfile
import onnx


class TestApi(object):
    def setup_method(self):
        self.imported_graph = OnnxImporter.import_graph(identity_model().load().graph)

    def test_import(self):
        graph = gs.import_onnx(onnx.load(identity_model().path))
        assert graph == self.imported_graph

    def test_export(self):
        with tempfile.NamedTemporaryFile() as f:
            onnx_model = gs.export_onnx(self.imported_graph)
            assert onnx_model
            assert OnnxImporter.import_graph(onnx_model.graph) == self.imported_graph
