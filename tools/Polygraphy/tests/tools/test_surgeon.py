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
import tempfile

from tests.models.meta import ONNX_MODELS
from tests.tools.common import run_polygraphy_run, run_polygraphy_surgeon

import onnx


class TestSurgeonExtract(object):
    def test_sanity(self):
        with tempfile.NamedTemporaryFile() as modelpath:
            run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs", "identity_out_0,auto,auto"])
            run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


    def test_fallback_shape_inference(self):
        with tempfile.NamedTemporaryFile() as modelpath:
            # Force fallback shape inference by disabling ONNX shape inference
            run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs",
                                "identity_out_0,auto,auto", "--outputs", "identity_out_2,auto", "--no-shape-inference"])
            run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


    def test_sanity_dim_param(self):
        with tempfile.NamedTemporaryFile() as modelpath:
            run_polygraphy_surgeon(["extract", ONNX_MODELS["dim_param"].path, "-o", modelpath.name])
            run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


class TestSurgeonInsert(object):
    def check_insert_model(self, path, expected_node_ops, expected_graph_input_names, expected_graph_output_names):
        model = onnx.load(path)
        for node, op in zip(model.graph.node, expected_node_ops):
            assert node.op_type == op

        graph_output_names = set([out.name for out in model.graph.output])
        assert graph_output_names == set(expected_graph_output_names)

        graph_input_names = set([out.name for out in model.graph.input])
        assert graph_input_names == set(expected_graph_input_names)


    def test_input_is_output(self):
        with tempfile.NamedTemporaryFile() as modelpath:
            run_polygraphy_surgeon(["insert", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs=identity_out_0",
                                    "--outputs=identity_out_0", "--op=FakeOp"])
            self.check_insert_model(modelpath.name, ["Identity", "FakeOp", "Identity"], ["X"], ["identity_out_2"])


    def test_graph_output(self):
        # FakeOp output tensor should be marked as a graph output since identity_out_2 was.
        with tempfile.NamedTemporaryFile() as modelpath:
            run_polygraphy_surgeon(["insert", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs=identity_out_2",
                                    "--outputs=identity_out_2", "--op=FakeOp"])
            self.check_insert_model(modelpath.name, ["Identity", "Identity", "FakeOp"], ["X"], ["identity_out_2_polygraphy_surgeon_insert_output"])


    def test_at_graph_input(self):
        with tempfile.NamedTemporaryFile() as modelpath:
            run_polygraphy_surgeon(["insert", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs=X",
                                    "--outputs=X", "--op=FakeOp"])
            self.check_insert_model(modelpath.name, ["FakeOp", "Identity", "Identity"], ["X"], ["identity_out_2"])


class TestSurgeonSanitize(object):
    def test_sanity(self):
        with tempfile.NamedTemporaryFile() as modelpath:
            run_polygraphy_surgeon(["sanitize", ONNX_MODELS["const_foldable"].path, "-o", modelpath.name, "--fold-constants"])

            model = onnx.load(modelpath.name)
            assert len(model.graph.node) == 1
