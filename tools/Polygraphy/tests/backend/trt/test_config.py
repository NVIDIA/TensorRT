import contextlib
import os
import tempfile

import numpy as np
import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.backend.trt import Calibrator, CreateConfig, Profile, network_from_onnx_bytes, postprocess_config
from polygraphy.common.struct import MetadataTuple, BoundedShape
from polygraphy.comparator import DataLoader
from tests.helper import has_dla
from tests.models.meta import ONNX_MODELS


@pytest.fixture(scope="session")
def identity_builder_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader)
    with builder, network, parser:
        yield builder, network


class TestCreateConfig:
    def test_defaults(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig()
        assert loader.timing_cache_path is None

        with loader(builder, network) as config:
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.TF32)
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            assert not config.get_flag(trt.BuilderFlag.FP16)
            assert not config.get_flag(trt.BuilderFlag.INT8)
            if mod.version(trt.__version__) >= mod.version("8.6"):
                assert not config.get_flag(trt.BuilderFlag.FP8)
            assert config.num_optimization_profiles == 1
            assert config.int8_calibrator is None
            with contextlib.suppress(AttributeError):
                if mod.version(trt.__version__) >= mod.version("8.5"):
                    assert config.get_tactic_sources() == 31
                elif mod.version(trt.__version__) >= mod.version("8.4"):
                    assert config.get_tactic_sources() == 15
                elif mod.version(trt.__version__) >= mod.version("8.0"):
                    assert config.get_tactic_sources() == 7
                else:
                    assert config.get_tactic_sources() == 3
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            with contextlib.suppress(AttributeError):
                assert config.engine_capability == trt.EngineCapability.STANDARD
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.DIRECT_IO)

    def test_workspace_size(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig(max_workspace_size=0)
        with loader(builder, network) as config:
            assert config.max_workspace_size == 0

    if mod.version(trt.__version__) >= mod.version("8.0"):

        @pytest.mark.parametrize(
            "engine_capability",
            [trt.EngineCapability.STANDARD, trt.EngineCapability.SAFETY, trt.EngineCapability.DLA_STANDALONE],
        )
        def test_engine_capability(self, identity_builder_network, engine_capability):
            builder, network = identity_builder_network
            loader = CreateConfig(engine_capability=engine_capability)
            with loader(builder, network) as config:
                assert config.engine_capability == engine_capability

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

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.2"), reason="Unsupported before TRT 8.2")
    def test_direct_io(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig(direct_io=True)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.DIRECT_IO)

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

    @pytest.mark.parametrize("flag", [True, False])
    def test_refittable(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(refittable=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.REFIT) == flag

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

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.6"), reason="API was added in TRT 8.6")
    @pytest.mark.parametrize("flag", [True, False])
    def test_fp8(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(fp8=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.FP8) == flag

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
        if mod.version(trt.__version__) < mod.version("8.0"):
            TACTIC_SOURCES_CASES = [
                (None, 3),  # By default, all sources are enabled.
                ([], 0),
                ([trt.TacticSource.CUBLAS], 1),
                ([trt.TacticSource.CUBLAS_LT], 2),
                ([trt.TacticSource.CUBLAS, trt.TacticSource.CUBLAS_LT], 3),
            ]

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

        if mod.version(trt.__version__) >= mod.version("8.5"):
            TACTIC_SOURCES_CASES[0] = (None, 31)
            TACTIC_SOURCES_CASES.extend(
                [
                    (
                        [
                            trt.TacticSource.CUDNN,
                            trt.TacticSource.CUBLAS,
                            trt.TacticSource.CUBLAS_LT,
                            trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
                            trt.TacticSource.JIT_CONVOLUTIONS,
                        ],
                        31,
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
            assert calibrator.data_loader.input_metadata["x"] == MetadataTuple(
                shape=BoundedShape((1, 1, 2, 2)), dtype=np.dtype(np.float32)
            )

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
    def test_fall_back_to_empty_timing_cache(self, identity_builder_network):
        """Tests that passing in a nonexistent timing cache path is non-fatal"""
        builder, network = identity_builder_network
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_name = os.path.join(tmpdir, "casper")
            loader = CreateConfig(load_timing_cache=cache_name)
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

    if mod.version(trt.__version__) >= mod.version("8.5"):

        @pytest.mark.parametrize(
            "preview_features",
            [
                [],
                [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805],
                [
                    trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805,
                    trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805,
                ],
            ],
        )
        def test_preview_features(self, identity_builder_network, preview_features):
            builder, network = identity_builder_network
            loader = CreateConfig(preview_features=preview_features)
            with loader(builder, network) as config:
                # Check that only the enabled preview features are on.
                for pf in trt.PreviewFeature.__members__.values():
                    assert config.get_preview_feature(pf) == (pf in preview_features)

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"), reason="Unsupported for TRT versions prior to 8.6"
    )
    @pytest.mark.parametrize("level", range(5))
    def test_builder_optimization_level(self, identity_builder_network, level):
        builder, network = identity_builder_network
        loader = CreateConfig(builder_optimization_level=level)
        with loader(builder, network) as config:
            assert config.builder_optimization_level == level

    if mod.version(trt.__version__) >= mod.version("8.6"):

        @pytest.mark.parametrize(
            "level",
            [
                trt.HardwareCompatibilityLevel.NONE,
                trt.HardwareCompatibilityLevel.AMPERE_PLUS,
            ],
        )
        def test_hardware_compatibility_level(self, identity_builder_network, level):
            builder, network = identity_builder_network
            loader = CreateConfig(hardware_compatibility_level=level)
            with loader(builder, network) as config:
                assert config.hardware_compatibility_level == level


class TestPostprocessConfig:
    def test_with_config(self, identity_builder_network):
        builder, network = identity_builder_network
        config = CreateConfig()(builder, network)
        assert not config.get_flag(trt.BuilderFlag.INT8)

        config = postprocess_config(
            config,
            func=lambda builder, network, config: config.set_flag(trt.BuilderFlag.INT8),
            builder=builder,
            network=network,
        )
        assert config.get_flag(trt.BuilderFlag.INT8)

    def test_with_config_callable(self, identity_builder_network):
        builder, network = identity_builder_network
        config = CreateConfig()

        config = postprocess_config(
            config,
            func=lambda builder, network, config: config.set_flag(trt.BuilderFlag.INT8),
            builder=builder,
            network=network,
        )
        assert config.get_flag(trt.BuilderFlag.INT8)
