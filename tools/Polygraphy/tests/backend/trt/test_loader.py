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
from polygraphy.backend.trt import EngineFromBytes, EngineFromNetwork, CreateConfig, NetworkFromOnnxBytes, NetworkFromOnnxPath, ModifyNetwork, Calibrator, Profile, SaveEngine, LoadPlugins
from polygraphy.backend.trt import util as trt_util
from polygraphy.common import PolygraphyException, constants
from polygraphy.comparator import DataLoader

from tests.models.meta import ONNX_MODELS
from tests.common import version, check_file_non_empty

import tensorrt as trt
import numpy as np
import tempfile
import pytest
import os


@pytest.fixture(scope="session")
def identity_engine():
    network_loader = NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader)
    engine_loader = EngineFromNetwork(network_loader, CreateConfig())
    with engine_loader() as engine:
        yield engine


@pytest.fixture(scope="session")
def identity_builder_network():
    builder, network, parser = NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader)()
    with builder, network, parser:
        yield builder, network


@pytest.fixture(scope="session")
def load_identity():
    return NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader)


@pytest.fixture(scope="session")
def load_identity_identity():
    return NetworkFromOnnxBytes(ONNX_MODELS["identity_identity"].loader)


class TestLoadPlugins(object):
    def test_can_load_libnvinfer_plugins(self):
        def get_plugin_names():
            return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]

        loader = LoadPlugins(plugins=["libnvinfer_plugin.so"])
        loader()
        assert get_plugin_names()


class TestSerializedEngineLoader(object):
    def test_serialized_engine_loader_from_lambda(self, identity_engine):
        with tempfile.NamedTemporaryFile() as outpath:
            with open(outpath.name, "wb") as f:
                f.write(identity_engine.serialize())

            loader = EngineFromBytes(lambda: open(outpath.name, "rb").read())
            with loader() as engine:
                assert isinstance(engine, trt.ICudaEngine)


    def test_serialized_engine_loader_from_buffer(self, identity_engine):
        loader = EngineFromBytes(identity_engine.serialize())
        with loader() as engine:
            assert isinstance(engine, trt.ICudaEngine)


class TestOnnxNetworkLoader(object):
    def test_loader(self):
        builder, network, parser = NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader)()
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            assert not network.has_explicit_precision


    def test_loader_explicit_precision(self):
        builder, network, parser = NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader, explicit_precision=True)()
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            assert network.has_explicit_precision


@pytest.mark.skipif(version(trt.__version__) < version("7.1.0.0"), reason="API was added in TRT 7.1")
class TestNetworkFromOnnxPath(object):
    def test_loader(self):
        builder, network, parser = NetworkFromOnnxPath(ONNX_MODELS["identity"].path)()
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            assert not network.has_explicit_precision


    def test_loader_explicit_precision(self):
        builder, network, parser = NetworkFromOnnxPath(ONNX_MODELS["identity"].path, explicit_precision=True)()
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            assert network.has_explicit_precision


class TestModifyNetwork(object):
    def test_layerwise(self, load_identity_identity):
        load_network = ModifyNetwork(load_identity_identity, outputs=constants.MARK_ALL)
        builder, network, parser = load_network()
        with builder, network, parser:
            for layer in network:
                for index in range(layer.num_outputs):
                    assert layer.get_output(index).is_network_output


    def test_custom_outputs(self, load_identity_identity):
        builder, network, parser = ModifyNetwork(load_identity_identity, outputs=["identity_out_0"])()
        with builder, network, parser:
            assert network.num_outputs == 1
            assert network.get_output(0).name == "identity_out_0"


    def test_exclude_outputs_with_layerwise(self, load_identity_identity):
        builder, network, parser = ModifyNetwork(load_identity_identity, outputs=constants.MARK_ALL, exclude_outputs=["identity_out_2"])()
        with builder, network, parser:
            assert network.num_outputs == 1
            assert network.get_output(0).name == "identity_out_0"


class TestProfile(object):
    def test_can_add(self):
        profile = Profile()
        min, opt, max = (1, 1), (2, 2), (4, 4)
        profile.add("input", min=min, opt=opt, max=max)
        shape_tuple = profile["input"]
        assert shape_tuple.min == min
        assert shape_tuple.opt == opt
        assert shape_tuple.max == max


class TestConfigLoader(object):
    def test_defaults(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig()
        config = loader(builder, network)
        assert config.max_workspace_size == 1 << 24
        if version(trt.__version__) > version("7.1.0.0"):
            assert not config.get_flag(trt.BuilderFlag.TF32)
        assert not config.get_flag(trt.BuilderFlag.FP16)
        assert not config.get_flag(trt.BuilderFlag.INT8)
        assert config.num_optimization_profiles == 1
        assert config.int8_calibrator is None


    def test_workspace_size(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig(max_workspace_size=0)
        config = loader(builder, network)
        assert config.max_workspace_size == 0


    @pytest.mark.parametrize("flag", [True, False])
    def test_strict_types(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(strict_types=flag)
        config = loader(builder, network)
        assert config.get_flag(trt.BuilderFlag.STRICT_TYPES) == flag


    @pytest.mark.parametrize("flag", [True, False])
    def test_tf32(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(tf32=flag)
        config = loader(builder, network)
        if version(trt.__version__) > version("7.1.0.0"):
            assert config.get_flag(trt.BuilderFlag.TF32) == flag


    @pytest.mark.parametrize("flag", [True, False])
    def test_fp16(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(fp16=flag)
        config = loader(builder, network)
        assert config.get_flag(trt.BuilderFlag.FP16) == flag


    @pytest.mark.parametrize("flag", [True, False])
    def test_int8(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(int8=flag)
        config = loader(builder, network)
        assert config.get_flag(trt.BuilderFlag.INT8) == flag


    def test_calibrator_metadata_set(self, identity_builder_network):
        builder, network = identity_builder_network
        calibrator = Calibrator(DataLoader())
        loader = CreateConfig(int8=True, calibrator=calibrator)
        config = loader(builder, network)
        assert config.int8_calibrator
        assert "x" in calibrator.data_loader.input_metadata


    def test_multiple_profiles(self, identity_builder_network):
        builder, network = identity_builder_network
        profiles = [
            Profile().add("x", (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4)),
            Profile().add("x", (1, 2, 4, 4), (1, 2, 8, 8), (1, 2, 16, 16)),
        ]
        loader = CreateConfig(profiles=profiles)
        config = loader(builder, network)
        assert config.num_optimization_profiles == 2


class TestEngineFromNetwork(object):
    def test_can_build_with_parser_owning(self):
        loader = EngineFromNetwork(NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader))
        with loader():
            pass


    def test_can_build_without_parser_non_owning(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = EngineFromNetwork((builder, network))
        with loader():
            pass


    def test_can_build_with_calibrator(self, identity_builder_network):
        builder, network = identity_builder_network
        calibrator = Calibrator(DataLoader())
        create_config = CreateConfig(int8=True, calibrator=calibrator)
        loader = EngineFromNetwork((builder, network), create_config)
        with loader():
            pass
        # Calibrator buffers should be freed after the build
        assert all([buf.allocated_nbytes == 0 for buf in calibrator.device_buffers.values()])


class TestSaveEngine(object):
    def test_save_engine(self, load_identity):
        with tempfile.NamedTemporaryFile() as outpath:
            engine_loader = SaveEngine(EngineFromNetwork(load_identity), path=outpath.name)
            with engine_loader() as engine:
                check_file_non_empty(outpath.name)
