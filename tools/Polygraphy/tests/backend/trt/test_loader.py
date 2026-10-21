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
from __future__ import annotations

import sys

import pytest

import polygraphy.backend.trt.loader as trt_loader
from polygraphy import config, constants, mod, util
from polygraphy.backend.trt import (
    Calibrator,
    EngineBytesFromNetwork,
    EngineFromBytes,
    EngineFromNetwork,
    EngineFromPath,
    LoadPlugins,
    LoadRuntime,
    ModifyNetworkOutputs,
    NetworkFromOnnxBytes,
    Profile,
    SaveEngine,
    buffer_from_engine,
    bytes_from_engine,
    create_config,
    create_network,
    engine_from_network,
    get_trt_logger,
    modify_network_outputs,
    network_from_onnx_bytes,
    network_from_onnx_path,
    onnx_like_from_network,
    postprocess_network,
    set_layer_precisions,
    set_tensor_datatypes,
    set_tensor_formats,
)
from polygraphy.common.struct import BoundedShape
from polygraphy.comparator import DataLoader
from polygraphy.datatype import DataType
from polygraphy.exception import PolygraphyException
from tests.helper import get_file_size, is_file_non_empty
from tests.models.meta import ONNX_MODELS

# Import CreateConfigRTX conditionally for TensorRT-RTX builds
if config.USE_TENSORRT_RTX:
    import tensorrt_rtx as trt
    from polygraphy.backend.tensorrt_rtx import CreateConfigRTX as CreateConfig
else:
    import tensorrt as trt
    from polygraphy.backend.trt import CreateConfig

# Previous versions of TRT segfault when using `parser.set_builder_config`
_SKIP_REPORT_CAPABILITY_DLA = mod.version(trt.__version__) < mod.version("10.16")


##
## Fixtures
##


@pytest.fixture(scope="session")
def identity_engine():
    network_loader = NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader)
    engine_loader = EngineFromNetwork(network_loader)
    with engine_loader() as engine:
        yield engine


@pytest.fixture(scope="session")
def identity_vc_engine_bytes():
    flags = [trt.OnnxParserFlag.NATIVE_INSTANCENORM]
    config = CreateConfig(version_compatible=True)
    network_loader = NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader, flags=flags)
    engine_loader = EngineBytesFromNetwork(network_loader, config=config)
    with engine_loader() as engine_bytes:
        yield engine_bytes


@pytest.fixture(scope="session")
def identity_builder_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader)
    yield builder, network


@pytest.fixture(scope="session")
def identity_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader)
    yield builder, network, parser


@pytest.fixture(scope="session")
def identity_identity_network():
    builder, network, parser = network_from_onnx_bytes(
        ONNX_MODELS["identity_identity"].loader
    )
    yield builder, network, parser


@pytest.fixture(scope="session")
def reshape_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["reshape"].loader)
    yield builder, network, parser


@pytest.fixture(scope="session")
def modifiable_network():
    # Must return a loader since the network will be modified each time it's loaded.
    return NetworkFromOnnxBytes(ONNX_MODELS["identity_identity"].loader)


@pytest.fixture(scope="session")
def modifiable_reshape_network():
    # Must return a loader since the network will be modified each time it's loaded.
    return NetworkFromOnnxBytes(ONNX_MODELS["reshape"].loader)


##
## Tests
##


class TestLoadPlugins:
    @pytest.mark.skipif(
        config.USE_TENSORRT_RTX,
        reason="Plugin tests are not compatible with TensorRT-RTX",
    )
    def test_can_load_libnvinfer_plugins(self):
        def get_plugin_names():
            registry = trt.get_plugin_registry()
            # `plugin_creator_list` was renamed to `all_creators` in TRT 11.
            creators = (
                registry.plugin_creator_list
                if hasattr(registry, "plugin_creator_list")
                else registry.all_creators
            )
            return [pc.name for pc in creators]

        loader = LoadPlugins(
            plugins=[
                (
                    "nvinfer_plugin.dll"
                    if sys.platform.startswith("win")
                    # The TRT 11 pip wheels ship only the major-versioned soname
                    # (libnvinfer_plugin.so.11); there is no unversioned symlink.
                    else (
                        f"libnvinfer_plugin.so.{trt.__version__.split('.')[0]}"
                        if mod.version(trt.__version__) >= mod.version("11.0")
                        else "libnvinfer_plugin.so"
                    )
                )
            ]
        )
        loader()
        assert get_plugin_names()


class TestSerializedEngineLoader:
    def test_serialized_engine_loader_from_lambda(self, identity_engine):
        with util.NamedTemporaryFile() as outpath:
            with open(outpath.name, "wb") as f, identity_engine.serialize() as buffer:
                f.write(buffer)

            loader = EngineFromBytes(lambda: open(outpath.name, "rb").read())
            with loader() as engine:
                assert isinstance(engine, trt.ICudaEngine)

    def test_serialized_engine_loader_from_buffer(self, identity_engine):
        with identity_engine.serialize() as buffer:
            loader = EngineFromBytes(buffer)
            with loader() as engine:
                assert isinstance(engine, trt.ICudaEngine)

    def test_serialized_engine_loader_custom_runtime(self, identity_engine):
        with identity_engine.serialize() as buffer:
            loader = EngineFromBytes(buffer, runtime=trt.Runtime(get_trt_logger()))
            with loader() as engine:
                assert isinstance(engine, trt.ICudaEngine)

    @pytest.mark.parametrize("from_path", [False, True])
    def test_sets_dla_workspace_strategy_before_deserialization(
        self, from_path, monkeypatch
    ):
        strategy = object()
        engine = object()
        events = []

        class DummyRuntime:
            @property
            def dla_workspace_allocation_strategy(self):
                return self._dla_workspace_allocation_strategy

            @dla_workspace_allocation_strategy.setter
            def dla_workspace_allocation_strategy(self, value):
                self._dla_workspace_allocation_strategy = value
                events.append(("set_strategy", value))

            def deserialize_cuda_engine(self, buffer):
                events.append(("deserialize", self.dla_workspace_allocation_strategy))
                return engine

        class DummyTrt:
            @staticmethod
            def init_libnvinfer_plugins(*args):
                pass

        monkeypatch.setattr(trt_loader, "trt", DummyTrt)
        monkeypatch.setattr(trt_loader.trt_util, "get_trt_logger", lambda: None)
        monkeypatch.setattr(trt_loader, "FileReader", lambda path: path)

        runtime = DummyRuntime()
        if from_path:
            with util.NamedTemporaryFile() as engine_file:
                engine_file.write(b"engine")
                engine_file.flush()
                loader = EngineFromPath(
                    engine_file.name,
                    runtime=runtime,
                    dla_workspace_allocation_strategy=strategy,
                )
                actual_engine = loader()
        else:
            loader = EngineFromBytes(
                b"engine",
                runtime=runtime,
                dla_workspace_allocation_strategy=strategy,
            )
            actual_engine = loader()

        assert actual_engine is engine
        assert events == [("set_strategy", strategy), ("deserialize", strategy)]


@pytest.mark.skipif(
    mod.version(trt.__version__) < mod.version("10.0") and not config.USE_TENSORRT_RTX,
    reason="API was added in TRT 10.0",
)
class TestSerializedEngineLoaderFromDisk:
    def test_serialized_engine_loader_from_lambda(self, identity_engine):
        with util.NamedTemporaryFile() as outpath:
            with open(outpath.name, "wb") as f, identity_engine.serialize() as buffer:
                f.write(buffer)

            loader = EngineFromPath(lambda: outpath.name)
            with loader() as engine:
                assert isinstance(engine, trt.ICudaEngine)

    def test_serialized_engine_loader_custom_runtime(self, identity_engine):
        with util.NamedTemporaryFile() as outpath:
            with open(outpath.name, "wb") as f, identity_engine.serialize() as buffer:
                f.write(buffer)

            loader = EngineFromPath(
                lambda: outpath.name, runtime=trt.Runtime(get_trt_logger())
            )
            with loader() as engine:
                assert isinstance(engine, trt.ICudaEngine)


@pytest.mark.skipif(
    mod.version(trt.__version__) < mod.version("8.6"), reason="API was added in TRT 8.6"
)
@pytest.mark.skipif(
    config.USE_TENSORRT_RTX,
    reason="TensorRT-RTX does not have lean runtime shared objects",
)
class TestLoadRuntime:
    def test_load_lean_runtime(self, nvinfer_lean_path):
        loader = LoadRuntime(nvinfer_lean_path)
        with loader() as runtime:
            assert isinstance(runtime, trt.Runtime)


@pytest.mark.skipif(
    mod.version(trt.__version__) < mod.version("8.6") and not config.USE_TENSORRT_RTX,
    reason="API was added in TRT 8.6",
)
@pytest.mark.skipif(
    config.USE_TENSORRT_RTX, reason="TensorRT-RTX does not have libnvinfer_lean.so.1"
)
class TestSerializedVCEngineLoader:
    def test_serialized_vc_engine_loader_from_lambda(self, identity_vc_engine_bytes):
        with util.NamedTemporaryFile() as outpath:
            with open(outpath.name, "wb") as f:
                f.write(identity_vc_engine_bytes)

            loader = EngineFromBytes(lambda: open(outpath.name, "rb").read())
            with loader() as engine:
                assert isinstance(engine, trt.ICudaEngine)

    def test_serialized_engine_loader_from_buffer(self, identity_vc_engine_bytes):
        loader = EngineFromBytes(identity_vc_engine_bytes)
        with loader() as engine:
            assert isinstance(engine, trt.ICudaEngine)


class TestNetworkFromOnnxBytes:
    def test_loader(self):
        builder, network, parser = network_from_onnx_bytes(
            ONNX_MODELS["identity"].loader
        )
        if not config.USE_TENSORRT_RTX:
            assert not network.has_implicit_batch_dimension

    @pytest.mark.parametrize(
        "kwargs, flag",
        (
            [
                (
                    {"strongly_typed": True},
                    trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED,
                )
            ]
            if mod.version(trt.__version__) >= mod.version("8.7")
            and not config.USE_TENSORRT_RTX
            else []
        ),
    )
    def test_network_flags(self, kwargs, flag):
        builder, network, parser = network_from_onnx_bytes(
            ONNX_MODELS["identity"].loader, **kwargs
        )
        assert network.get_flag(flag)

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6")
        and not config.USE_TENSORRT_RTX,
        reason="OnnxParserFlag operations require TRT >= 8.6",
    )
    @pytest.mark.parametrize(
        "kwargs, flag, should_be_set",
        [
            (
                {"plugin_instancenorm": True},
                trt.OnnxParserFlag.NATIVE_INSTANCENORM,
                False,  # plugin_instancenorm clears the flag
            ),
        ]
        + (
            [
                (
                    {"enable_uint8_asymmetric_quantization_dla": True},
                    trt.OnnxParserFlag.ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA,
                    True,  # this sets the flag
                ),
            ]
            if hasattr(
                trt.OnnxParserFlag, "ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA"
            )
            else []
        )
        + (
            [
                (
                    {
                        "flags": [trt.OnnxParserFlag.REPORT_CAPABILITY_DLA],
                        "config": CreateConfig(),
                    },
                    trt.OnnxParserFlag.REPORT_CAPABILITY_DLA,
                    True,  # this sets the flag (requires config to be set)
                ),
            ]
            if hasattr(trt.OnnxParserFlag, "REPORT_CAPABILITY_DLA")
            and not _SKIP_REPORT_CAPABILITY_DLA
            else []
        ),
    )
    def test_parser_flags(self, kwargs, flag, should_be_set, capfd):
        """Test that parser flags are properly set or cleared."""
        if (
            hasattr(trt.OnnxParserFlag, "REPORT_CAPABILITY_DLA")
            and flag == trt.OnnxParserFlag.REPORT_CAPABILITY_DLA
        ):
            # Our test machines don't have DLA hardware; expect parsing to fail.
            with pytest.raises(PolygraphyException):
                network_from_onnx_bytes(ONNX_MODELS["identity"].loader, **kwargs)

            captured = capfd.readouterr()
            combined = captured.out + captured.err
            # The TensorRT C++ logger's DLA-validation message is not reliably
            # captured by capfd under xdist on TRT 11; the pytest.raises above
            # already confirms REPORT_CAPABILITY_DLA is honored (parsing fails
            # without DLA hardware), so only assert the message where capture is
            # reliable.
            if mod.version(trt.__version__) < mod.version("11.0"):
                assert "DLA validation failed for layer:" in combined
            return

        builder, network, parser = network_from_onnx_bytes(
            ONNX_MODELS["identity"].loader, **kwargs
        )
        assert parser.get_flag(flag) == should_be_set


class TestNetworkFromOnnxPath:
    def test_loader(self):
        builder, network, parser = network_from_onnx_path(ONNX_MODELS["identity"].path)
        if not config.USE_TENSORRT_RTX:
            assert not network.has_implicit_batch_dimension

    @pytest.mark.parametrize(
        "kwargs, flag",
        (
            [
                (
                    {"strongly_typed": True},
                    trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED,
                )
            ]
            if mod.version(trt.__version__) >= mod.version("8.7")
            and not config.USE_TENSORRT_RTX
            else []
        ),
    )
    def test_network_flags(self, kwargs, flag):
        builder, network, parser = network_from_onnx_path(
            ONNX_MODELS["identity"].path, **kwargs
        )
        assert network.get_flag(flag)

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6")
        and not config.USE_TENSORRT_RTX,
        reason="OnnxParserFlag operations require TRT >= 8.6",
    )
    @pytest.mark.parametrize(
        "kwargs, flag, should_be_set",
        [
            (
                {"plugin_instancenorm": True},
                trt.OnnxParserFlag.NATIVE_INSTANCENORM,
                False,  # plugin_instancenorm clears the flag
            ),
        ]
        + (
            [
                (
                    {"enable_uint8_asymmetric_quantization_dla": True},
                    trt.OnnxParserFlag.ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA,
                    True,  # this sets the flag
                ),
            ]
            if hasattr(
                trt.OnnxParserFlag, "ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA"
            )
            else []
        )
        + (
            [
                (
                    {
                        "flags": [trt.OnnxParserFlag.REPORT_CAPABILITY_DLA],
                        "config": CreateConfig(),
                    },
                    trt.OnnxParserFlag.REPORT_CAPABILITY_DLA,
                    True,  # this sets the flag (requires config to be set)
                ),
            ]
            if hasattr(trt.OnnxParserFlag, "REPORT_CAPABILITY_DLA")
            and not _SKIP_REPORT_CAPABILITY_DLA
            else []
        ),
    )
    def test_parser_flags(self, kwargs, flag, should_be_set, capfd):
        """Test that parser flags are properly set or cleared."""
        if (
            hasattr(trt.OnnxParserFlag, "REPORT_CAPABILITY_DLA")
            and flag == trt.OnnxParserFlag.REPORT_CAPABILITY_DLA
        ):
            # Our test machines don't have DLA hardware; expect parsing to fail.
            with pytest.raises(PolygraphyException):
                network_from_onnx_path(ONNX_MODELS["identity"].path, **kwargs)

            captured = capfd.readouterr()
            combined = captured.out + captured.err
            # The TensorRT C++ logger's DLA-validation message is not reliably
            # captured by capfd under xdist on TRT 11; the pytest.raises above
            # already confirms REPORT_CAPABILITY_DLA is honored (parsing fails
            # without DLA hardware), so only assert the message where capture is
            # reliable.
            if mod.version(trt.__version__) < mod.version("11.0"):
                assert "DLA validation failed for layer:" in combined
            return

        builder, network, parser = network_from_onnx_path(
            ONNX_MODELS["identity"].path, **kwargs
        )
        assert parser.get_flag(flag) == should_be_set


class TestModifyNetwork:
    def test_mark_layerwise(self, modifiable_network):
        load_network = ModifyNetworkOutputs(
            modifiable_network, outputs=constants.MARK_ALL
        )
        builder, network, parser = load_network()

        for layer in network:
            for index in range(layer.num_outputs):
                assert layer.get_output(index).is_network_output

    def test_mark_custom_outputs(self, modifiable_network):
        builder, network, parser = modify_network_outputs(
            modifiable_network, outputs=["identity_out_0"]
        )

        assert network.num_outputs == 1
        assert network.get_output(0).name == "identity_out_0"

    def test_exclude_outputs_with_mark_layerwise(self, modifiable_network):
        builder, network, parser = modify_network_outputs(
            modifiable_network,
            outputs=constants.MARK_ALL,
            exclude_outputs=["identity_out_2"],
        )

        assert network.num_outputs == 1
        assert network.get_output(0).name == "identity_out_0"

    def test_mark_shape_outputs(self, modifiable_reshape_network):
        builder, network, parser = modify_network_outputs(
            modifiable_reshape_network, outputs=["output", "reduce_prod_out_gs_2"]
        )

        assert network.num_outputs == 2
        assert network.get_output(1).name == "reduce_prod_out_gs_2"

    def test_unmark_shape_outputs(self, modifiable_reshape_network):
        builder, network, parser = modify_network_outputs(
            modifiable_reshape_network,
            outputs=constants.MARK_ALL,
            exclude_outputs=["shape_out_gs_0", "reduce_prod_out_gs_2"],
        )

        assert network.num_outputs == 1

    def test_mark_outputs_layer_with_optional_inputs(self):
        builder, network = create_network()
        inp = network.add_input("input", shape=(1, 3, 224, 224), dtype=trt.float32)
        slice_layer = network.add_slice(
            inp, (0, 0, 0, 0), (1, 3, 224, 224), (1, 1, 1, 1)
        )

        # Set a tensor for `stride` to increment `num_inputs` so we have some inputs
        # which are `None` in between.
        slice_layer.set_input(3, inp)
        assert slice_layer.num_inputs == 4

        slice = slice_layer.get_output(0)
        slice.name = "Slice"

        builder, network = modify_network_outputs((builder, network), outputs=["Slice"])
        assert network.num_outputs == 1
        assert network.get_output(0).name == "Slice"
        assert network.get_output(0) == slice

    def test_mark_by_layer_type(self, modifiable_network):
        # identity model has IDENTITY-type layers
        load_network = ModifyNetworkOutputs(
            modifiable_network, mark_layer_types=["IDENTITY"]
        )
        builder, network, parser = load_network()
        assert network.num_outputs >= 1
        for i in range(network.num_outputs):
            assert network.get_output(i) is not None

    def test_mark_by_layer_type_case_insensitive(self, modifiable_network):
        load_network = ModifyNetworkOutputs(
            modifiable_network, mark_layer_types=["identity"]  # lowercase
        )
        builder, network, parser = load_network()
        assert network.num_outputs >= 1

    def test_mark_by_layer_type_unknown_raises(self, modifiable_network):
        load_network = ModifyNetworkOutputs(
            modifiable_network, mark_layer_types=["NONEXISTENT_TYPE"]
        )
        # G_LOGGER.critical raises
        with pytest.raises(Exception):
            load_network()


class TestPostprocessNetwork:
    def test_basic(self, modifiable_network):
        """Tests that the callback is actually invoked by Polygraphy."""
        func_called = False

        def func(network):
            nonlocal func_called
            func_called = True
            assert isinstance(network, trt.INetworkDefinition)

        builder, network, parser = postprocess_network(modifiable_network, func)
        assert func_called

    def test_kwargs(self, modifiable_network):
        """Tests that callbacks that use **kwargs work as expected."""
        func_called = False

        def func(**kwargs):
            nonlocal func_called
            func_called = True
            assert isinstance(kwargs["network"], trt.INetworkDefinition)

        builder, network, parser = postprocess_network(modifiable_network, func)
        assert func_called

    def test_modify_network(self, modifiable_network):
        """Tests that the network passed in is properly modified by the callback."""

        # Setting allowed_formats is a cross-version mutation (works on TRT 8.6/10/11
        # and RTX); weak-typing per-layer precision was removed in TRT 11.
        def func(network):
            network.get_input(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)
            network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.HWC8)

        builder, network, parser = postprocess_network(modifiable_network, func)

        assert network.get_input(0).allowed_formats == 1 << int(trt.TensorFormat.CHW4)
        assert network.get_output(0).allowed_formats == 1 << int(trt.TensorFormat.HWC8)

    def test_negative_non_callable(self, modifiable_network):
        """Tests that PostprocessNetwork properly rejects `func` objects that
        are not callable."""

        with pytest.raises(PolygraphyException, match=r"Object .* is not a callable"):
            builder, network, parser = postprocess_network(modifiable_network, None)


class TestSetLayerPrecisions:
    @pytest.mark.skipif(
        config.USE_TENSORRT_RTX or mod.version(trt.__version__) >= mod.version("11.0"),
        reason="Layer precision API not available",
    )
    def test_basic(self, modifiable_network):
        builder, network, parser = set_layer_precisions(
            modifiable_network,
            layer_precisions={
                "onnx_graphsurgeon_node_1": trt.float16,
                "onnx_graphsurgeon_node_3": trt.int8,
            },
        )

        assert network[0].precision == trt.float16
        assert network[1].precision == trt.int8


class TestSetTensorDatatypes:
    @pytest.mark.skipif(
        config.USE_TENSORRT_RTX or mod.version(trt.__version__) >= mod.version("11.0"),
        reason="TensorRT-RTX and TRT 11 use strongly typed networks where tensor datatypes cannot be set",
    )
    def test_basic(self, modifiable_network):
        builder, network, parser = set_tensor_datatypes(
            modifiable_network,
            tensor_datatypes={
                "X": trt.float16,
                "identity_out_2": trt.float16,
            },
        )

        assert network.get_input(0).dtype == trt.float16
        assert network.get_output(0).dtype == trt.float16

    @pytest.mark.skipif(
        config.USE_TENSORRT_RTX or mod.version(trt.__version__) >= mod.version("11.0"),
        reason="TensorRT-RTX and TRT 11 use strongly typed networks where tensor datatypes cannot be set",
    )
    def test_all(self, modifiable_network):
        builder, network, parser = set_tensor_datatypes(
            modifiable_network, tensor_datatypes=trt.float16
        )

        assert network.get_input(0).dtype == trt.float16
        assert network.get_output(0).dtype == trt.float16


class TestSetTensorFormats:
    def test_basic(self, modifiable_network):
        builder, network, parser = set_tensor_formats(
            modifiable_network,
            tensor_formats={
                "X": [trt.TensorFormat.LINEAR, trt.TensorFormat.CHW4],
                "identity_out_2": [trt.TensorFormat.HWC8],
            },
        )

        assert network.get_input(0).allowed_formats == (
            1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW4)
        )
        assert network.get_output(0).allowed_formats == 1 << int(trt.TensorFormat.HWC8)

    def test_all(self, modifiable_network):
        builder, network, parser = set_tensor_formats(
            modifiable_network,
            tensor_formats=[trt.TensorFormat.LINEAR, trt.TensorFormat.CHW4],
        )

        assert network.get_input(0).allowed_formats == (
            1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW4)
        )
        assert network.get_output(0).allowed_formats == (
            1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW4)
        )


class TestEngineBytesFromNetwork:
    def test_can_build(self, identity_network):
        loader = EngineBytesFromNetwork(identity_network)
        with loader() as serialized_engine:
            assert isinstance(serialized_engine, trt.IHostMemory)


class TestEngineFromNetwork:
    def test_defaults(self, identity_network):
        loader = EngineFromNetwork(identity_network)
        assert loader.timing_cache_path is None

    def test_forwards_dla_workspace_allocation_strategy(self, monkeypatch):
        expected_runtime = object()
        expected_strategy = object()
        expected_engine = object()

        monkeypatch.setattr(
            trt_loader.EngineBytesFromNetwork,
            "call_impl",
            lambda self: b"engine",
        )

        def fake_engine_from_bytes(
            serialized_engine, runtime=None, dla_workspace_allocation_strategy=None
        ):
            assert serialized_engine() == b"engine"
            assert runtime is expected_runtime
            assert dla_workspace_allocation_strategy is expected_strategy
            return expected_engine

        monkeypatch.setattr(trt_loader, "engine_from_bytes", fake_engine_from_bytes)
        loader = EngineFromNetwork(
            object(),
            runtime=expected_runtime,
            dla_workspace_allocation_strategy=expected_strategy,
        )
        assert loader() is expected_engine

    def test_can_build_with_parser_owning(self, identity_network):
        loader = EngineFromNetwork(identity_network)
        with loader() as engine:
            assert isinstance(engine, trt.ICudaEngine)

    def test_can_build_without_parser_non_owning(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = EngineFromNetwork((builder, network))
        with loader() as engine:
            assert isinstance(engine, trt.ICudaEngine)

    def test_custom_runtime(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = EngineFromNetwork(
            (builder, network), runtime=trt.Runtime(get_trt_logger())
        )
        with loader() as engine:
            assert isinstance(engine, trt.ICudaEngine)

    @pytest.mark.skipif(
        config.USE_TENSORRT_RTX or mod.version(trt.__version__) >= mod.version("11.0"),
        reason="TensorRT-RTX and TRT 11 do not support INT8 calibrators",
    )
    @pytest.mark.parametrize(
        "use_config_loader, set_calib_profile",
        [(True, None), (False, False), (False, True)],
    )
    def test_can_build_with_calibrator(
        self, identity_builder_network, use_config_loader, set_calib_profile
    ):
        builder, network = identity_builder_network
        calibrator = Calibrator(DataLoader())

        def check_calibrator():
            # CreateConfig and EngineFromNetwork should set the input metadata for the calibrator,
            # which in turn should be passed to the data loader.
            assert calibrator.input_metadata is not None
            assert "x" in calibrator.data_loader.input_metadata
            meta = calibrator.data_loader.input_metadata["x"]
            assert meta.shape == BoundedShape((1, 1, 2, 2))
            assert meta.dtype == DataType.FLOAT32

        if use_config_loader:
            config = create_config(builder, network, int8=True, calibrator=calibrator)
            check_calibrator()
        else:
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            # Since this network has static shapes, we shouldn't need to set a calibration profile.
            if set_calib_profile:
                calib_profile = (
                    Profile().fill_defaults(network).to_trt(builder, network)
                )
                config.add_optimization_profile(calib_profile)
                config.set_calibration_profile(calib_profile)

        loader = EngineFromNetwork((builder, network), config)
        with loader():
            pass
        check_calibrator()

        # Calibrator buffers should be freed after the build
        assert all(
            [buf.allocated_nbytes == 0 for buf in calibrator.device_buffers.values()]
        )

    @pytest.mark.parametrize("path_mode", [True, False], ids=["path", "file-like"])
    def test_timing_cache_generate_and_append(self, path_mode):
        with util.NamedTemporaryFile() as total_cache, util.NamedTemporaryFile() as identity_cache:

            def build_engine(model, cache):
                if not path_mode:
                    cache.seek(0)
                network_loader = NetworkFromOnnxBytes(ONNX_MODELS[model].loader)
                # In non-path_mode, use the file-like object directly.
                # Must load the cache with CreateConfig so that new data is appended
                # instead of overwriting the previous cache.
                loader = EngineFromNetwork(
                    network_loader,
                    CreateConfig(load_timing_cache=cache.name),
                    save_timing_cache=cache.name if path_mode else cache,
                )
                with loader():
                    pass
                if not path_mode:
                    cache.seek(0)

            assert not total_cache.read()

            build_engine("const_foldable", total_cache)
            const_foldable_cache_size = get_file_size(total_cache.name)

            # Build this network twice. Once with a fresh cache so we can determine its size.
            assert get_file_size(identity_cache.name) == 0
            build_engine("identity", identity_cache)
            identity_cache_size = get_file_size(identity_cache.name)

            build_engine("identity", total_cache)
            total_cache_size = get_file_size(total_cache.name)

            # The total cache should be larger than either of the individual caches.
            assert (
                total_cache_size >= const_foldable_cache_size
                and total_cache_size >= identity_cache_size
            )
            # The total cache should also be smaller than or equal to the sum of the individual caches since
            # header information should not be duplicated.
            assert total_cache_size <= (const_foldable_cache_size + identity_cache_size)


class TestBytesFromEngine:
    def test_serialize_engine(self, identity_network):
        with engine_from_network(identity_network) as engine:
            serialized_engine = bytes_from_engine(engine)
            assert isinstance(serialized_engine, bytes)


class TestBufferFromEngine:

    def test_should_return_IHostMemory(self, identity_engine: trt.ICudaEngine) -> None:
        # Precondition.
        engine = identity_engine

        # Under test.
        buffer = buffer_from_engine(engine)

        # Postcondition.
        assert isinstance(buffer, trt.IHostMemory)

    def test_should_content_match_engine(
        self, identity_engine: trt.ICudaEngine
    ) -> None:
        """Test that `BufferFromEngine` returns a buffer with the same content as the engine."""
        # Precondition.
        engine = identity_engine

        # Under test.
        buffer = buffer_from_engine(engine)

        # Postcondition.
        assert bytes(buffer) == bytes(engine.serialize())


class TestSaveEngine:

    def test_should_write_serialized_engine_to_file(
        self, identity_network: trt.ICudaEngine
    ) -> None:
        # Precondition.
        with util.NamedTemporaryFile(mode="wb+") as out_file:
            name = out_file.name
            engine = engine_from_network(identity_network)

            # Under test.
            save_engine = SaveEngine(engine, path=out_file)
            save_engine()
            out_file.flush()

            # Postcondition.
            assert is_file_non_empty(out_file.name)
            out_file.seek(0)
            assert bytes(engine.serialize()) == bytes(out_file.read())


class TestOnnxLikeFromNetwork:
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
    def test_onnx_like_from_network(self, model_name):
        assert onnx_like_from_network(
            NetworkFromOnnxBytes(ONNX_MODELS[model_name].loader)
        )


class TestDefaultPlugins:
    @pytest.mark.skipif(
        config.USE_TENSORRT_RTX,
        reason="Plugin tests are not compatible with TensorRT-RTX",
    )
    @pytest.mark.skipif(
        mod.version(trt.__version__) >= mod.version("11.0"),
        # ROIAlign became a native layer in TRT 11 so this model no longer exercises
        # a default plugin, and the obvious alternative (InstanceNormalization_TRT via
        # plugin_instancenorm) cannot load on TRT 11 because it depends on cuDNN 8 while
        # TRT 11 ships cuDNN 9. No simple ONNX op maps to a buildable default plugin here.
        reason="No ONNX op maps to a buildable default plugin on TensorRT 11 (ROIAlign is native; the InstanceNorm plugin needs cuDNN 8)",
    )
    def test_default_plugins(self):
        network_loader = NetworkFromOnnxBytes(ONNX_MODELS["roialign"].loader)
        engine_loader = EngineFromNetwork(network_loader)
        engine = engine_loader()
        assert engine is not None
