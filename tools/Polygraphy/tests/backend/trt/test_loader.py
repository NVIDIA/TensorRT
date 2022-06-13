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
import contextlib
import sys

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
)
from polygraphy.comparator import DataLoader
from tests.helper import get_file_size, has_dla, is_file_non_empty
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

    def test_loader_explicit_precision(self):
        builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader, explicit_precision=True)
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            if mod.version(trt.__version__) < mod.version("8.0"):
                assert network.has_explicit_precision


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.1.0.0"), reason="API was added in TRT 7.1")
class TestNetworkFromOnnxPath:
    def test_loader(self):
        builder, network, parser = network_from_onnx_path(ONNX_MODELS["identity"].path)
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            assert not network.has_explicit_precision

    def test_loader_explicit_precision(self):
        builder, network, parser = network_from_onnx_path(ONNX_MODELS["identity"].path, explicit_precision=True)
        with builder, network, parser:
            assert not network.has_implicit_batch_dimension
            if mod.version(trt.__version__) < mod.version("8.0"):
                assert network.has_explicit_precision


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

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_mark_shape_outputs(self, modifiable_reshape_network):
        builder, network, parser = modify_network_outputs(
            modifiable_reshape_network, outputs=["output", "reduce_prod_out_gs_2"]
        )
        with builder, network, parser:
            assert network.num_outputs == 2
            assert network.get_output(0).name == "reduce_prod_out_gs_2"
            assert network.get_output(0).is_shape_tensor

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_unmark_shape_outputs(self, modifiable_reshape_network):
        builder, network, parser = modify_network_outputs(
            modifiable_reshape_network, outputs=constants.MARK_ALL, exclude_outputs=["reduce_prod_out_gs_2"]
        )
        with builder, network, parser:
            assert network.num_outputs == 1


class TestCreateConfig:
    def test_defaults(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig()
        assert loader.timing_cache_path is None

        with loader(builder, network) as config:
            assert config.max_workspace_size == 1 << 24
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.TF32)
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            assert not config.get_flag(trt.BuilderFlag.FP16)
            assert not config.get_flag(trt.BuilderFlag.INT8)
            assert config.num_optimization_profiles == 1
            assert config.int8_calibrator is None
            with contextlib.suppress(AttributeError):
                if mod.version(trt.__version__) >= mod.version("8.4"):
                    assert config.get_tactic_sources() == 15
                elif mod.version(trt.__version__) >= mod.version("8.0"):
                    assert config.get_tactic_sources() == 7
                else:
                    assert config.get_tactic_sources() == 3
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    def test_workspace_size(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig(max_workspace_size=0)
        with loader(builder, network) as config:
            assert config.max_workspace_size == 0

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.2"), reason="Unsupported before TRT 8.2")
    @pytest.mark.parametrize("flag", [True, False])
    def test_obey_precision_constraints(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(obey_precision_constraints=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS) == flag

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.2"), reason="Unsupported before TRT 8.2")
    @pytest.mark.parametrize("flag", ["obey", "prefer", None])
    def test_precision_constraints(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(precision_constraints=flag)
        with loader(builder, network) as config:
            obey_set = config.get_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            prefer_set = config.get_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            if flag == "obey":
                assert obey_set and not prefer_set
            elif flag == "prefer":
                assert not obey_set and prefer_set
            else:
                assert not obey_set and not prefer_set

    @pytest.mark.parametrize("flag", [True, False])
    def test_strict_types(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(strict_types=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.STRICT_TYPES) == flag

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0.0.0"), reason="API was added in TRT 8.0")
    @pytest.mark.parametrize("flag", [True, False])
    def test_restricted(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(restricted=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.SAFETY_SCOPE) == flag

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.1.0.0"), reason="API was added in TRT 7.1")
    @pytest.mark.parametrize("flag", [True, False])
    def test_tf32(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(tf32=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.TF32) == flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_fp16(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(fp16=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.FP16) == flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_int8(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(int8=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.INT8) == flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_allow_gpu_fallback(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(allow_gpu_fallback=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.GPU_FALLBACK) == flag

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.0"), reason="API was not available in 7.2 and older"
    )
    @pytest.mark.parametrize("flag", [True, False])
    def test_sparse_weights(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(sparse_weights=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.SPARSE_WEIGHTS) == flag

    def test_use_dla(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig(use_dla=True)
        with loader(builder, network) as config:
            assert config.default_device_type == trt.DeviceType.DLA
            if has_dla():
                assert config.DLA_core == 0

    with contextlib.suppress(AttributeError):
        if mod.version(trt.__version__) >= mod.version("8.0"):
            TACTIC_SOURCES_CASES = [
                (None, 7),  # By default, all sources are enabled.
                ([], 0),
                ([trt.TacticSource.CUBLAS], 1),
                ([trt.TacticSource.CUBLAS_LT], 2),
                ([trt.TacticSource.CUDNN], 4),
                ([trt.TacticSource.CUBLAS, trt.TacticSource.CUBLAS_LT], 3),
                ([trt.TacticSource.CUBLAS, trt.TacticSource.CUDNN], 5),
                ([trt.TacticSource.CUBLAS_LT, trt.TacticSource.CUDNN], 6),
                ([trt.TacticSource.CUDNN, trt.TacticSource.CUBLAS, trt.TacticSource.CUBLAS_LT], 7),
            ]
        else:
            TACTIC_SOURCES_CASES = [
                (None, 3),  # By default, all sources are enabled.
                ([], 0),
                ([trt.TacticSource.CUBLAS], 1),
                ([trt.TacticSource.CUBLAS_LT], 2),
                ([trt.TacticSource.CUBLAS, trt.TacticSource.CUBLAS_LT], 3),
            ]

        if mod.version(trt.__version__) >= mod.version("8.4"):
            TACTIC_SOURCES_CASES[0] = (None, 15)
            TACTIC_SOURCES_CASES.extend(
                [
                    (
                        [
                            trt.TacticSource.CUDNN,
                            trt.TacticSource.CUBLAS,
                            trt.TacticSource.CUBLAS_LT,
                            trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
                        ],
                        15,
                    )
                ]
            )

        @pytest.mark.parametrize("sources, expected", TACTIC_SOURCES_CASES)
        def test_tactic_sources(self, identity_builder_network, sources, expected):
            builder, network = identity_builder_network
            loader = CreateConfig(tactic_sources=sources)
            with loader(builder, network) as config:
                assert config.get_tactic_sources() == expected

    def test_calibrator_metadata_set(self, identity_builder_network):
        builder, network = identity_builder_network
        calibrator = Calibrator(DataLoader())
        loader = CreateConfig(int8=True, calibrator=calibrator)
        with loader(builder, network) as config:
            assert config.int8_calibrator
            assert "x" in calibrator.data_loader.input_metadata

    def test_multiple_profiles(self, identity_builder_network):
        builder, network = identity_builder_network
        profiles = [
            Profile().add("x", (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4)),
            Profile().add("x", (1, 2, 4, 4), (1, 2, 8, 8), (1, 2, 16, 16)),
        ]
        loader = CreateConfig(profiles=profiles)
        with loader(builder, network) as config:
            assert config.num_optimization_profiles == 2

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
    @pytest.mark.parametrize("path_mode", [True, False], ids=["path", "file-like"])
    def test_timing_cache(self, identity_builder_network, path_mode):
        builder, network = identity_builder_network
        with util.NamedTemporaryFile() as cache:
            loader = CreateConfig(load_timing_cache=cache.name if path_mode else cache)
            with loader(builder, network) as config:
                assert config.get_timing_cache()

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
    def test_empty_timing_cache_when_default(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig()
        with loader(builder, network) as config:
            cache = config.get_timing_cache()
            with cache.serialize() as buffer:
                cache_size = len(bytes(buffer))

            cache.reset()
            with cache.serialize() as buffer:
                new_cache_size = len(bytes(buffer))
            assert cache_size == new_cache_size

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
    def test_profiling_verbosity(self, identity_builder_network):
        builder, network = identity_builder_network
        expected = trt.ProfilingVerbosity.NONE
        loader = CreateConfig(profiling_verbosity=expected)
        with loader(builder, network) as config:
            assert config.profiling_verbosity == expected

    with contextlib.suppress(AttributeError):
        POOL_LIMITS = [
            {trt.MemoryPoolType.WORKSPACE: 25},
            {trt.MemoryPoolType.DLA_MANAGED_SRAM: 25},
            {trt.MemoryPoolType.DLA_LOCAL_DRAM: 25},
            {trt.MemoryPoolType.DLA_GLOBAL_DRAM: 25},
            # Multiple limits
            {
                trt.MemoryPoolType.DLA_LOCAL_DRAM: 20,
                trt.MemoryPoolType.DLA_GLOBAL_DRAM: 25,
                trt.MemoryPoolType.WORKSPACE: 39,
            },
        ]

        @pytest.mark.skipif(
            mod.version(trt.__version__) < mod.version("8.3"), reason="Unsupported for TRT versions prior to 8.3"
        )
        @pytest.mark.parametrize("pool_limits", POOL_LIMITS)
        def test_memory_pool_limits(self, pool_limits, identity_builder_network):
            if any("dla" in key.name.lower() for key in pool_limits) and not has_dla():
                pytest.skip("DLA is not available on this system")

            builder, network = identity_builder_network
            loader = CreateConfig(memory_pool_limits=pool_limits)
            with loader(builder, network) as config:
                for pool_type, pool_size in pool_limits.items():
                    assert config.get_memory_pool_limit(pool_type) == pool_size


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

    def test_can_build_with_calibrator(self, identity_builder_network):
        builder, network = identity_builder_network
        calibrator = Calibrator(DataLoader())
        create_config = CreateConfig(int8=True, calibrator=calibrator)
        loader = EngineFromNetwork((builder, network), create_config)
        with loader():
            pass
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
            assert total_cache_size > const_foldable_cache_size and total_cache_size > identity_cache_size
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
