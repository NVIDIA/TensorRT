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
import os
import tempfile

import numpy as np
import onnx_graphsurgeon as gs
import pytest
from polygraphy import constants, util
from polygraphy.backend.onnx import (
    ConvertToFp16,
    FoldConstants,
    ModifyOutputs,
    OnnxFromBytes,
    OnnxFromPath,
    OnnxFromTfGraph,
    SaveOnnx,
    extract_subgraph,
    gs_from_onnx,
    infer_shapes,
    fold_constants,
    onnx_from_path,
)
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER
from tests.helper import is_file_non_empty
from tests.models.meta import ONNX_MODELS, TF_MODELS

import onnx


class TestLoggerCallbacks:
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.module_severity = sev


class TestOnnxFromPath:
    def test_basic(self):
        loader = OnnxFromPath(ONNX_MODELS["identity"].path)
        model = loader()
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) == 1

    def test_external_data(self):
        model = ONNX_MODELS["ext_weights"]
        loader = OnnxFromPath(model.path, model.ext_data)
        assert isinstance(loader(), onnx.ModelProto)

    def test_ignore_external_data(self):
        model = ONNX_MODELS["ext_weights"]
        loader = OnnxFromPath(model.path, ignore_external_data=True)
        onnx_model = loader()

        assert isinstance(onnx_model, onnx.ModelProto)
        assert all(init.data_location == 1 for init in onnx_model.graph.initializer)


class TestOnnxFromBytes:
    def test_basic(self):
        loader = OnnxFromBytes(ONNX_MODELS["identity"].loader)
        model = loader()
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) == 1


class TestGsFromOnnx:
    def test_basic(self):
        graph = gs_from_onnx(OnnxFromPath(ONNX_MODELS["identity"].path))
        assert isinstance(graph, gs.Graph)


class TestExportOnnxFromTf:
    pytest.importorskip("tensorflow")

    def test_no_optimize(self):
        loader = OnnxFromTfGraph(TF_MODELS["identity"].loader, optimize=False)
        model = loader()

    def test_opset(self):
        loader = OnnxFromTfGraph(TF_MODELS["identity"].loader, opset=9)
        model = loader()
        assert model.opset_import[0].version == 9


class TestModifyOnnx:
    @pytest.mark.parametrize("copy", [True, False])
    def test_layerwise(self, copy):
        original_model = onnx_from_path(ONNX_MODELS["identity_identity"].path)
        loader = ModifyOutputs(original_model, outputs=constants.MARK_ALL, copy=copy)
        model = loader()
        assert len(original_model.graph.output) == 1 or not copy
        assert len(model.graph.output) == 2

    @pytest.mark.parametrize("output", ["identity_out_0", "identity_out_2"])
    def test_custom_outputs(self, output):
        loader = ModifyOutputs(OnnxFromPath(ONNX_MODELS["identity_identity"].path), outputs=[output])
        model = loader()
        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == output

    def test_exclude_outputs_with_layerwise(self):
        loader = ModifyOutputs(
            OnnxFromPath(ONNX_MODELS["identity_identity"].path),
            outputs=constants.MARK_ALL,
            exclude_outputs=["identity_out_2"],
        )
        model = loader()
        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == "identity_out_0"


@pytest.mark.parametrize("allow_onnxruntime", [True, False])
class TestInferShapes:
    def check_model(self, model):
        # Find all intermediate tensors to check if they have shapes.
        tensors = set()
        for node in model.graph.node:
            tensors.update(node.output)
        tensors -= {out.name for out in model.graph.output}

        assert len(model.graph.value_info) >= len(tensors)
        for val in model.graph.value_info:
            assert val.type.tensor_type.HasField("shape")

    def test_model(self, allow_onnxruntime):
        original_model = onnx_from_path(ONNX_MODELS["identity_identity"].path)
        model = infer_shapes(original_model, allow_onnxruntime=allow_onnxruntime)
        self.check_model(model)

    def test_path(self, allow_onnxruntime):
        model = infer_shapes(ONNX_MODELS["identity_identity"].path, allow_onnxruntime=allow_onnxruntime)
        self.check_model(model)

    @pytest.mark.parametrize("set_data_dir", [True, False])
    def test_external_data(self, set_data_dir, allow_onnxruntime):
        model = ONNX_MODELS["ext_weights_same_dir"]
        model = infer_shapes(
            model.path,
            external_data_dir=model.ext_data if set_data_dir else None,
            allow_onnxruntime=allow_onnxruntime,
        )
        self.check_model(model)

    def test_save_to_disk_on_size_threshold(self, allow_onnxruntime):
        model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        model = infer_shapes(model, save_to_disk_threshold_bytes=0, allow_onnxruntime=allow_onnxruntime)
        self.check_model(model)


class TestConvertToFp16:
    @pytest.mark.parametrize("copy", [True, False])
    def test_basic(self, copy):
        original_model = onnx_from_path(ONNX_MODELS["identity_identity"].path)
        loader = ConvertToFp16(original_model, copy=copy)
        model = loader()

        assert original_model.graph.input[0].type.tensor_type.elem_type == 1 or not copy
        assert model.graph.input[0].type.tensor_type.elem_type == 1
        assert model.graph.node[2].op_type == "Cast"
        assert model.graph.node[0].op_type == "Identity"
        assert model.graph.node[1].op_type == "Identity"
        assert model.graph.node[3].op_type == "Cast"
        assert model.graph.output[0].type.tensor_type.elem_type == 1


class TestFoldConstants:
    @pytest.mark.parametrize("fold_shapes", [True, False])
    @pytest.mark.parametrize("partitioning", [None, "basic", "recursive"])
    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("allow_onnxruntime_shape_inference", [True, False])
    def test_basic(self, partitioning, fold_shapes, copy, allow_onnxruntime_shape_inference):
        original_model = onnx_from_path(ONNX_MODELS["const_foldable"].path)
        loader = FoldConstants(
            original_model,
            partitioning=partitioning,
            fold_shapes=fold_shapes,
            copy=copy,
            error_ok=False,
            allow_onnxruntime_shape_inference=allow_onnxruntime_shape_inference,
        )
        model = loader()
        assert len(original_model.graph.node) != 1 or not copy
        assert len(model.graph.node) == 1

    @pytest.mark.parametrize(
        "size_threshold, expect_folding",
        [
            (None, True),
            (0, False),
            (10 << 20, True),
            (10 << 20 - 1, False),
        ],
    )
    def test_size_threshold(self, size_threshold, expect_folding):
        model = onnx_from_path(ONNX_MODELS["constant_fold_bloater"].path)
        model = fold_constants(model, size_threshold=size_threshold)

        if expect_folding:
            assert len(model.graph.node) == 0
        else:
            assert len(model.graph.node) == 1
            assert model.graph.node[0].op_type == "Tile"


class TestSaveOnnx:
    def test_save_onnx(self):
        with tempfile.TemporaryDirectory() as outdir:
            outpath = os.path.join(outdir, "test", "nested")
            loader = SaveOnnx(OnnxFromPath(ONNX_MODELS["identity"].path), path=outpath)
            loader()
            assert is_file_non_empty(outpath)

    def test_external_data(self):
        with util.NamedTemporaryFile() as path, util.NamedTemporaryFile() as data:
            model = OnnxFromPath(ONNX_MODELS["const_foldable"].path)
            loader = SaveOnnx(model, path.name, external_data_path=data.name, size_threshold=0)
            loader()
            assert is_file_non_empty(path.name)
            assert is_file_non_empty(data.name)


@pytest.fixture()
def extract_model():
    input_metadata = TensorMetadata().add("X", dtype=np.float32, shape=(64, 64))
    output_metadata = TensorMetadata().add("identity_out_0", dtype=np.float32, shape=None)
    return onnx_from_path(ONNX_MODELS["identity_identity"].path), input_metadata, output_metadata


class TestExtractSubgraph:
    def check_model(self, model):
        graph = gs_from_onnx(model)
        assert len(graph.nodes) == 1

        assert len(graph.inputs) == 1
        assert graph.inputs[0].name == "X"
        assert graph.inputs[0].shape is not None
        assert graph.inputs[0].dtype is not None

        assert len(graph.outputs) == 1
        assert graph.outputs[0].name == "identity_out_0"
        assert graph.outputs[0].dtype is not None

    def test_extract_onnx_model(self, extract_model):
        original_model, input_meta, output_meta = extract_model
        model = extract_subgraph(original_model, input_meta, output_meta)

        assert original_model.graph.output[0].name == "identity_out_2"
        self.check_model(model)

    def test_extract_onnx_model_no_input_meta(self, extract_model):
        model, _, output_meta = extract_model
        model = extract_subgraph(model, output_metadata=output_meta)
        self.check_model(model)

    def test_extract_onnx_model_no_output_meta(self, extract_model):
        model, input_meta, _ = extract_model
        model = extract_subgraph(model, input_metadata=input_meta)
        assert model.graph.output[0].name == "identity_out_2"

    def test_extract_onnx_gs_graph(self, extract_model):
        model, input_meta, output_meta = extract_model
        graph = gs_from_onnx(model)
        subgraph = extract_subgraph(graph, input_meta, output_meta)
        # Make sure original graph isn't modified.
        assert len(graph.nodes) == 2

        assert isinstance(subgraph, gs.Graph)
        assert len(subgraph.nodes) == 1

        assert len(subgraph.inputs) == 1
        assert subgraph.inputs[0].name == "X"

        assert len(subgraph.outputs) == 1
        assert subgraph.outputs[0].name == "identity_out_0"

    def test_extract_passes_no_input_shape(self, extract_model):
        model, input_meta, output_meta = extract_model
        input_meta["X"].shape = None
        model = extract_subgraph(model, input_meta, output_meta)
        self.check_model(model)

    def test_extract_passes_no_input_dtype(self, extract_model):
        model, input_meta, output_meta = extract_model
        input_meta["X"].dtype = None
        model = extract_subgraph(model, input_meta, output_meta)
        self.check_model(model)

    def test_extract_passes_no_output_shape(self, extract_model):
        model, input_meta, output_meta = extract_model
        output_meta["identity_out_0"].shape = None
        model = extract_subgraph(model, input_meta, output_meta)
        self.check_model(model)
