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

import numpy as np
import pytest
import tensorrt as trt

from polygraphy import mod, util
from polygraphy.backend.trt import (
    CreateConfig,
    TrtRunner,
    create_network,
    engine_bytes_from_network,
    network_from_onnx_path,
)
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import (
    ModelArgs,
    OnnxLoadArgs,
    TrtConfigArgs,
    TrtLoadEngineArgs,
    TrtLoadEngineBytesArgs,
    TrtLoadNetworkArgs,
    TrtLoadPluginsArgs,
    TrtOnnxFlagArgs,
)
from polygraphy.tools.args.backend.trt.helper import make_trt_enum_val
from tests.models.meta import ONNX_MODELS
from tests.tools.args.helper import ArgGroupTestHelper


class TestTrtLoadNetworkArgs:
    @pytest.mark.parametrize("force_onnx_loader", [True, False])
    @pytest.mark.parametrize(
        "opts,expected_flag",
        (
            [([], None)]
            + [(["--strongly-typed"], trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)]
            if mod.version(trt.__version__) >= mod.version("8.7")
            else []
        ),
    )
    def test_load_network(self, force_onnx_loader, opts, expected_flag):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )

        args = [ONNX_MODELS["identity_identity"].path]
        if force_onnx_loader:
            # We can force Polygraphy to use NetworkFromOnnxBytes instead of NetworkFromOnnxPath by requiring
            # changes to the model.
            args.append("--trt-outputs=identity_out_0")

        args += opts
        arg_group.parse_args(args)

        builder, network, _ = arg_group.load_network()
        with builder, network:
            assert network.num_outputs == 1
            assert network.get_output(0).name == (
                "identity_out_0" if force_onnx_loader else "identity_out_2"
            )
            if expected_flag is not None:
                assert network.get_flag(expected_flag)

    @pytest.mark.parametrize("func_name", ["postprocess", "custom_func"])
    def test_postprocess_network(self, func_name):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )
        script = dedent(
            f"""
            def {func_name}(network):
                for layer in network:
                    print(layer.name)
                network.get_output(0).name = "modified_output"
            """
        )
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())

            if func_name == "postprocess":
                pps_arg = f"{f.name}"
            else:
                pps_arg = f"{f.name}:{func_name}"

            arg_group.parse_args(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--trt-network-postprocess-script",
                    pps_arg,
                ]
            )

            builder, network, _ = arg_group.load_network()
            with builder, network:
                assert network.num_outputs == 1
                assert network.get_output(0).name == "modified_output"

    def test_set_layer_precisions(self):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )
        arg_group.parse_args(
            [
                ONNX_MODELS["identity_identity"].path,
                "--layer-precisions",
                "onnx_graphsurgeon_node_1:float16",
                "onnx_graphsurgeon_node_3:int8",
            ]
        )

        builder, network, _ = arg_group.load_network()
        with builder, network:
            assert network[0].precision == trt.float16
            assert network[1].precision == trt.int8

    def test_set_layer_precisions_default_disallowed(self):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )
        with pytest.raises(PolygraphyException, match="Could not parse argument"):
            arg_group.parse_args(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--layer-precisions",
                    "float16",
                ]
            )

    def test_set_tensor_datatypes(self):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )
        arg_group.parse_args(
            [
                ONNX_MODELS["identity_identity"].path,
                "--tensor-datatypes",
                "X:float16",
                "identity_out_2:float16",
            ]
        )

        builder, network, _ = arg_group.load_network()
        with builder, network:
            assert network.get_input(0).dtype == trt.float16
            assert network.get_output(0).dtype == trt.float16

    def test_set_tensor_datatypes_default_disallowed(self):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )
        with pytest.raises(PolygraphyException, match="Could not parse argument"):
            arg_group.parse_args(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--tensor-datatypes",
                    "float16",
                ]
            )

    def test_set_tensor_formats(self):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(allow_tensor_formats=True),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )
        arg_group.parse_args(
            [
                ONNX_MODELS["identity_identity"].path,
                "--tensor-formats",
                # Should be case-insensitive
                "X:[liNEar,chw4]",
                "identity_out_2:[hWc8]",
            ]
        )

        builder, network, _ = arg_group.load_network()
        with builder, network:
            assert network.get_input(0).allowed_formats == (
                1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW4)
            )
            assert network.get_output(0).allowed_formats == 1 << int(
                trt.TensorFormat.HWC8
            )

    def test_set_tensor_formats_default_disallowed(self):
        arg_group = ArgGroupTestHelper(
            TrtLoadNetworkArgs(allow_tensor_formats=True),
            deps=[
                ModelArgs(),
                OnnxLoadArgs(allow_shape_inference=False),
                TrtLoadPluginsArgs(),
                TrtConfigArgs(),
                TrtOnnxFlagArgs(),
            ],
        )
        with pytest.raises(PolygraphyException, match="Could not parse argument"):
            arg_group.parse_args(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--tensor-formats",
                    "[linear]",
                ]
            )

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"),
        reason="API was added in TRT 8.6",
    )
    @pytest.mark.parametrize(
        "args",
        [["--hardware-compatibility-level=ampere_plus"], ["--version-compatible"]],
    )
    def test_onnx_flags_autoenabled_for_vc_or_hc(self, args):
        arg_group = ArgGroupTestHelper(
            TrtOnnxFlagArgs(), deps=[ModelArgs(), TrtConfigArgs()]
        )
        arg_group.parse_args([ONNX_MODELS["identity_identity"].path] + args)

        assert arg_group.get_flags()[0] == [
            make_trt_enum_val("OnnxParserFlag", "NATIVE_INSTANCENORM")
        ]


@pytest.fixture()
def engine_loader_args():
    return ArgGroupTestHelper(
        TrtLoadEngineArgs(),
        deps=[
            ModelArgs(),
            OnnxLoadArgs(allow_shape_inference=False),
            TrtConfigArgs(),
            TrtLoadPluginsArgs(),
            TrtLoadEngineBytesArgs(),
            TrtLoadNetworkArgs(),
            TrtOnnxFlagArgs(),
        ],
    )


class TestTrtEngineLoaderArgs:
    def test_build_engine(self, engine_loader_args):
        engine_loader_args.parse_args(
            [ONNX_MODELS["identity_identity"].path, "--trt-outputs=identity_out_0"]
        )

        with engine_loader_args.load_engine() as engine:
            assert isinstance(engine, trt.ICudaEngine)
            assert engine[1] == "identity_out_0"

    def test_build_engine_custom_network(self, engine_loader_args):
        engine_loader_args.parse_args([])

        builder, network = create_network()
        inp = network.add_input("input", dtype=trt.float32, shape=(1, 1))
        out = network.add_identity(inp).get_output(0)
        out.name = "output"
        network.mark_output(out)

        with builder, network, engine_loader_args.load_engine(
            network=(builder, network)
        ) as engine:
            assert isinstance(engine, trt.ICudaEngine)
            assert engine[0] == "input"
            assert engine[1] == "output"

    def test_load_serialized_engine(self, engine_loader_args):
        with util.NamedTemporaryFile() as f, engine_bytes_from_network(
            network_from_onnx_path(ONNX_MODELS["identity"].path)
        ) as engine_bytes:
            f.write(engine_bytes)
            f.flush()
            os.fsync(f.fileno())

            engine_loader_args.parse_args([f.name, "--model-type=engine"])
            with engine_loader_args.load_engine() as engine:
                assert isinstance(engine, trt.ICudaEngine)

                assert engine[0] == "x"
                assert engine[1] == "y"

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"),
        reason="API was added in TRT 8.6",
    )
    def test_load_engine_with_custom_runtime(
        self, engine_loader_args, nvinfer_lean_path
    ):
        with util.NamedTemporaryFile() as f, engine_bytes_from_network(
            network_from_onnx_path(ONNX_MODELS["identity"].path),
            CreateConfig(version_compatible=True, exclude_lean_runtime=True),
        ) as engine_bytes:
            f.write(engine_bytes)
            f.flush()
            os.fsync(f.fileno())

            engine_loader_args.parse_args(
                [f.name, "--model-type=engine", "--load-runtime", nvinfer_lean_path]
            )
            assert engine_loader_args.load_runtime == nvinfer_lean_path
            with engine_loader_args.load_engine() as engine:
                assert isinstance(engine, trt.ICudaEngine)

                with TrtRunner(engine) as runner:
                    assert runner.infer({"x": np.ones((1, 1, 2, 2), dtype=np.float32)})
