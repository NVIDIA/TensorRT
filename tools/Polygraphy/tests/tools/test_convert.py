#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from textwrap import dedent

import onnx
import pytest
import tensorrt as trt

from polygraphy import util
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.onnx import gs_from_onnx
from polygraphy.backend.trt import EngineFromBytes
from tests.models.meta import ONNX_MODELS, TF_MODELS


class TestConvertToOnnx:
    def test_tf2onnx(self, poly_convert):
        pytest.importorskip("tensorflow")

        with util.NamedTemporaryFile(suffix=".onnx") as outmodel:
            poly_convert(
                [TF_MODELS["identity"].path, "--model-type=frozen", "-o", outmodel.name]
            )
            assert onnx.load(outmodel.name)

    def test_fp_to_fp16(self, poly_convert):
        with util.NamedTemporaryFile() as outmodel:
            poly_convert(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--convert-to=onnx",
                    "--fp-to-fp16",
                    "-o",
                    outmodel.name,
                ]
            )
            # I/O types should be unchanged
            model = onnx.load(outmodel.name)
            graph = gs_from_onnx(model)
            graph.toposort()

            assert graph.inputs[0].dtype == "float32"
            assert graph.nodes[0].op == "Cast"
            assert graph.nodes[1].op == "Identity"
            assert graph.nodes[2].op == "Identity"
            assert graph.nodes[3].op == "Cast"
            assert graph.outputs[0].dtype == "float32"


class TestConvertToTrt:
    def check_engine(self, path):
        loader = EngineFromBytes(BytesFromPath(path))
        with loader() as engine:
            assert isinstance(engine, trt.ICudaEngine)

    def test_onnx_to_trt(self, poly_convert):
        with util.NamedTemporaryFile(suffix=".engine") as outmodel:
            poly_convert(
                [ONNX_MODELS["identity"].path, "--model-type=onnx", "-o", outmodel.name]
            )
            self.check_engine(outmodel.name)

    def test_tf_to_onnx_to_trt(self, poly_convert):
        pytest.importorskip("tensorflow")

        with util.NamedTemporaryFile() as outmodel:
            poly_convert(
                [
                    TF_MODELS["identity"].path,
                    "--model-type=frozen",
                    "--convert-to=trt",
                    "-o",
                    outmodel.name,
                ]
            )
            self.check_engine(outmodel.name)

    def test_trt_network_config_script_to_engine(self, poly_convert):
        script = dedent(
            """
        from polygraphy.backend.trt import CreateNetwork, CreateConfig
        from polygraphy import func
        import tensorrt as trt

        @func.extend(CreateNetwork())
        def my_load_network(builder, network):
            inp = network.add_input("input", dtype=trt.float32, shape=(1, 1))
            out = network.add_identity(inp).get_output(0)
            network.mark_output(out)

        @func.extend(CreateConfig())
        def load_config(config):
            config.set_flag(trt.BuilderFlag.FP16)
        """
        )

        with util.NamedTemporaryFile(
            "w+", suffix=".py"
        ) as f, util.NamedTemporaryFile() as outmodel:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())

            poly_convert(
                [
                    f"{f.name}:my_load_network",
                    "--model-type=trt-network-script",
                    "--trt-config-script",
                    f.name,
                    "--convert-to=trt",
                    "-o",
                    outmodel.name,
                ]
            )
            self.check_engine(outmodel.name)

    def test_modify_onnx_outputs(self, poly_convert):
        with util.NamedTemporaryFile(suffix=".onnx") as outmodel:
            poly_convert(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "-o",
                    outmodel.name,
                    "--onnx-outputs",
                    "mark",
                    "all",
                ]
            )

            model = onnx.load(outmodel.name)
            assert len(model.graph.output) == 2


class TestConvertToOnnxLikeTrt:
    @pytest.mark.parametrize(
        "model_name",
        [
            "identity",
            "empty_tensor_expand",
            "const_foldable",
            "and",
            "scan",
            "dim_param",
            "tensor_attr",
        ],
    )
    def test_onnx_to_trt_to_onnx_like(self, poly_convert, model_name):
        with util.NamedTemporaryFile() as outmodel:
            poly_convert(
                [
                    ONNX_MODELS[model_name].path,
                    "--convert-to=onnx-like-trt-network",
                    "-o",
                    outmodel.name,
                ]
            )
