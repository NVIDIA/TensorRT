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
import sys

import numpy as np
import pytest
import tensorrt as trt
from polygraphy import constants, mod, util
from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    EngineBytesFromNetwork,
    EngineFromBytes,
    EngineFromNetwork,
    LoadPlugins,
    ModifyNetworkOutputs,
    NetworkFromOnnxBytes,
    Profile,
    SaveEngine,
    bytes_from_engine,
    engine_from_network,
    modify_network_outputs,
    network_from_onnx_bytes,
    network_from_onnx_path,
    onnx_like_from_network,
    set_layer_precisions,
    create_config,
    postprocess_network,
    set_tensor_datatypes,
    set_tensor_formats,
    create_network,
)
from polygraphy.common.struct import MetadataTuple, BoundedShape
from polygraphy.comparator import DataLoader
from polygraphy.exception import PolygraphyException
from tests.helper import get_file_size, is_file_non_empty
from tests.models.meta import ONNX_MODELS

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
def identity_builder_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader)
    with builder, network, parser:
        yield builder, network


@pytest.fixture(scope="session")
def identity_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader)
    with builder, network, parser:
        yield builder, network, parser


@pytest.fixture(scope="session")
def identity_identity_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity_identity"].loader)
    with builder, network, parser:
        yield builder, network, parser


@pytest.fixture(scope="session")
def reshape_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["reshape"].loader)
    with builder, network, parser:
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
    def test_can_load_libnvinfer_plugins(self):
        def get_plugin_names():
            return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]

        loader = LoadPlugins(
            plugins=["nvinfer_plugin.dll" if sys.platform.startswith("win") else "libnvinfer_plugin.so"]
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


class TestOnnxNetworkLoader:
    def test_loader(self):
        builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader)
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            assert not network.has_explicit_precision


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.1.0.0"), reason="API was added in TRT 7.1")
class TestNetworkFromOnnxPath:
    def test_loader(self):
        builder, network, parser = network_from_onnx_path(ONNX_MODELS["identity"].path)
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            assert not network.has_explicit_precision


class TestModifyNetwork:
    def test_mark_layerwise(self, modifiable_network):
        load_network = ModifyNetworkOutputs(modifiable_network, outputs=constants.MARK_ALL)
        builder, network, parser = load_network()
        with builder, network, parser:
            for layer in network:
                for index in range(layer.num_outputs):
                    assert layer.get_output(index).is_network_output

    def test_mark_custom_outputs(self, modifiable_network):
        builder, network, parser = modify_network_outputs(modifiable_network, outputs=["identity_out_0"])
        with builder, network, parser:
            assert network.num_outputs == 1
            assert network.get_output(0).name == "identity_out_0"

    def test_exclude_outputs_with_mark_layerwise(self, modifiable_network):
        builder, network, parser = modify_network_outputs(
            modifiable_network, outputs=constants.MARK_ALL, exclude_outputs=["identity_out_2"]
        )
        with builder, network, parser:
            assert network.num_outputs == 1
            assert network.get_output(0).name == "identity_out_0"

    def test_mark_shape_outputs(self, modifiable_reshape_network):
        builder, network, parser = modify_network_outputs(
            modifiable_reshape_network, outputs=["output", "reduce_prod_out_gs_2"]
        )
        with builder, network, parser:
            assert network.num_outputs == 2
            assert network.get_output(1).name == "reduce_prod_out_gs_2"
            assert network.get_output(1).is_shape_tensor

    def test_unmark_shape_outputs(self, modifiable_reshape_network):
        builder, network, parser = modify_network_outputs(
            modifiable_reshape_network, outputs=constants.MARK_ALL, exclude_outputs=["reduce_prod_out_gs_2"]
        )
        with builder, network, parser:
            assert network.num_outputs == 1

    def test_mark_outputs_layer_with_optional_inputs(self):
        builder, network = create_network()
        with builder, network:
            inp = network.add_input("input", shape=(1, 3, 224, 224), dtype=trt.float32)
            slice_layer = network.add_slice(inp, (0, 0, 0, 0), (1, 3, 224, 224), (1, 1, 1, 1))

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

        # Performs the equivalent of set_layer_precisions
        def func(network):
            for layer in network:
                if layer.name == "onnx_graphsurgeon_node_1":
                    layer.precision = trt.float16
                if layer.name == "onnx_graphsurgeon_node_3":
                    layer.precision = trt.int8

        builder, network, parser = postprocess_network(modifiable_network, func)

        with builder, network, parser:
            assert network[0].precision == trt.float16
            assert network[1].precision == trt.int8

    def test_negative_non_callable(self, modifiable_network):
        """Tests that PostprocessNetwork properly rejects `func` objects that
        are not callable."""

        with pytest.raises(PolygraphyException, match=r"Object .* is not a callable"):
            builder, network, parser = postprocess_network(modifiable_network, None)


class TestSetLayerPrecisions:
    def test_basic(self, modifiable_network):
        builder, network, parser = set_layer_precisions(
            modifiable_network,
            layer_precisions={"onnx_graphsurgeon_node_1": trt.float16, "onnx_graphsurgeon_node_3": trt.int8},
        )
        with builder, network, parser:
            assert network[0].precision == trt.float16
            assert network[1].precision == trt.int8

    def test_non_existent_layer(self, modifiable_network):
        with pytest.raises(PolygraphyException, match="The following layers were not found"):
            set_layer_precisions(
                modifiable_network,
                layer_precisions={"fake_layer": trt.float16},
            )


class TestSetTensorDatatypes:
    def test_basic(self, modifiable_network):
        builder, network, parser = set_tensor_datatypes(
            modifiable_network,
            tensor_datatypes={
                "X": trt.float16,
                "identity_out_0": trt.float32,
                "identity_out_2": trt.float16,
            },
        )
        with builder, network, parser:
            assert network[0].get_input(0).dtype == trt.float16
            assert network[0].get_output(0).dtype == trt.float32
            assert network[1].get_input(0).dtype == trt.float32
            assert network[1].get_output(0).dtype == trt.float16

    def test_non_existent_tensor(self, modifiable_network):
        with pytest.raises(PolygraphyException, match="The following tensors were not found"):
            set_tensor_datatypes(
                modifiable_network,
                tensor_datatypes={"fake_tensor": trt.float16},
            )


class TestSetTensorFormats:
    def test_basic(self, modifiable_network):
        builder, network, parser = set_tensor_formats(
            modifiable_network,
            tensor_formats={
                "X": [trt.TensorFormat.LINEAR, trt.TensorFormat.CHW4],
                "identity_out_2": [trt.TensorFormat.HWC8],
            },
        )
        with builder, network, parser:
            assert network[0].get_input(0).allowed_formats == (
                1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW4)
            )
            assert network[1].get_output(0).allowed_formats == 1 << int(trt.TensorFormat.HWC8)

    def test_non_existent_tensor(self, modifiable_network):
        with pytest.raises(PolygraphyException, match="The following tensors were not found"):
            set_tensor_formats(
                modifiable_network,
                tensor_formats={"fake_tensor": trt.float16},
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

    def test_can_build_with_parser_owning(self, identity_network):
        loader = EngineFromNetwork(identity_network)
        with loader():
            pass

    def test_can_build_without_parser_non_owning(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = EngineFromNetwork((builder, network))
        with loader():
            pass

    @pytest.mark.parametrize("use_config_loader, set_calib_profile", [(True, None), (False, False), (False, True)])
    def test_can_build_with_calibrator(self, identity_builder_network, use_config_loader, set_calib_profile):
        builder, network = identity_builder_network
        calibrator = Calibrator(DataLoader())

        def check_calibrator():
            # CreateConfig and EngineFromNetwork should set the input metadata for the calibrator,
            # which in turn should be passed to the data loader.
            assert calibrator.input_metadata is not None
            assert "x" in calibrator.data_loader.input_metadata
            assert calibrator.data_loader.input_metadata["x"] == MetadataTuple(
                shape=BoundedShape((1, 1, 2, 2)), dtype=np.dtype(np.float32)
            )

        if use_config_loader:
            config = create_config(builder, network, int8=True, calibrator=calibrator)
            check_calibrator()
        else:
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            # Since this network has static shapes, we shouldn't need to set a calibration profile.
            if set_calib_profile:
                calib_profile = Profile().fill_defaults(network).to_trt(builder, network)
                config.add_optimization_profile(calib_profile)
                config.set_calibration_profile(calib_profile)

        loader = EngineFromNetwork((builder, network), config)
        with loader():
            pass
        check_calibrator()

        # Calibrator buffers should be freed after the build
        assert all([buf.allocated_nbytes == 0 for buf in calibrator.device_buffers.values()])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
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
            assert total_cache_size >= const_foldable_cache_size and total_cache_size >= identity_cache_size
            # The total cache should also be smaller than or equal to the sum of the individual caches since
            # header information should not be duplicated.
            assert total_cache_size <= (const_foldable_cache_size + identity_cache_size)


class TestBytesFromEngine:
    def test_serialize_engine(self, identity_network):
        with engine_from_network(identity_network) as engine:
            serialized_engine = bytes_from_engine(engine)
            assert isinstance(serialized_engine, bytes)


class TestSaveEngine:
    def test_save_engine(self, identity_network):
        with util.NamedTemporaryFile() as outpath:
            engine_loader = SaveEngine(EngineFromNetwork(identity_network), path=outpath.name)
            with engine_loader():
                assert is_file_non_empty(outpath.name)


class TestOnnxLikeFromNetwork:
    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.2"), reason="Unsupported for TRT 7.1 and older")
    @pytest.mark.parametrize(
        "model_name", ["identity", "empty_tensor_expand", "const_foldable", "and", "scan", "dim_param", "tensor_attr"]
    )
    def test_onnx_like_from_network(self, model_name):
        assert onnx_like_from_network(NetworkFromOnnxBytes(ONNX_MODELS[model_name].loader))
