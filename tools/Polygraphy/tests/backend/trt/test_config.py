import contextlib
import os
import tempfile

import pytest

from polygraphy import config as polygraphy_config
from polygraphy import mod, util
from polygraphy.backend.trt import (
    Calibrator,
    Profile,
    network_from_onnx_bytes,
    postprocess_config,
)
from polygraphy.common.struct import BoundedShape
from polygraphy.comparator import DataLoader
from polygraphy.datatype import DataType
from tests.helper import has_dla
from tests.models.meta import ONNX_MODELS

# Import CreateConfigRTX conditionally for TensorRT-RTX builds
if polygraphy_config.USE_TENSORRT_RTX:
    import tensorrt_rtx as trt

    from polygraphy.backend.tensorrt_rtx import CreateConfigRTX as CreateConfig
else:
    import tensorrt as trt

    from polygraphy.backend.trt import CreateConfig


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
            assert not config.get_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            with contextlib.suppress(AttributeError):
                if polygraphy_config.USE_TENSORRT_RTX:
                    assert config.get_flag(trt.BuilderFlag.TF32)
                else:
                    assert not config.get_flag(trt.BuilderFlag.TF32)
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            # FP16/INT8/BF16/FP8 precision flags were removed in TRT 11 (strong
            # typing); suppress mirrors the optional-flag handling used above.
            if not polygraphy_config.USE_TENSORRT_RTX:
                with contextlib.suppress(AttributeError):
                    assert not config.get_flag(trt.BuilderFlag.FP16)
                    assert not config.get_flag(trt.BuilderFlag.INT8)
            if mod.version(trt.__version__) >= mod.version("8.7"):
                with contextlib.suppress(AttributeError):
                    assert not config.get_flag(trt.BuilderFlag.BF16)
            if mod.version(trt.__version__) >= mod.version("8.6"):
                with contextlib.suppress(AttributeError):
                    assert not config.get_flag(trt.BuilderFlag.FP8)
                assert not config.get_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
                assert not config.get_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
                if not polygraphy_config.USE_TENSORRT_RTX:
                    assert (
                        config.hardware_compatibility_level
                        == trt.HardwareCompatibilityLevel.NONE
                    )
            if (
                mod.version(trt.__version__) >= mod.version("10.2")
                and not polygraphy_config.USE_TENSORRT_RTX
            ):
                assert config.runtime_platform == trt.RuntimePlatform.SAME_AS_BUILD
            assert config.num_optimization_profiles == 1
            if not polygraphy_config.USE_TENSORRT_RTX and mod.version(
                trt.__version__
            ) < mod.version("11.0"):
                assert config.int8_calibrator is None
            with contextlib.suppress(AttributeError):
                if polygraphy_config.USE_TENSORRT_RTX:
                    assert config.get_tactic_sources() == 3
                elif mod.version(trt.__version__) >= mod.version("11.0"):
                    # Only EDGE_MASK_CONVOLUTIONS + JIT_CONVOLUTIONS remain in TRT 11.
                    assert config.get_tactic_sources() == 3
                elif mod.version(trt.__version__) >= mod.version("10.0"):
                    assert config.get_tactic_sources() == 24
                elif mod.version(trt.__version__) >= mod.version("8.7"):
                    assert config.get_tactic_sources() == 29
                elif mod.version(trt.__version__) >= mod.version("8.5"):
                    assert config.get_tactic_sources() == 31
            if mod.version(trt.__version__) >= mod.version("8.7"):
                assert not config.get_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
            if mod.version(trt.__version__) >= mod.version("8.7"):
                assert not config.get_flag(trt.BuilderFlag.DISABLE_COMPILATION_CACHE)
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            with contextlib.suppress(AttributeError):
                if polygraphy_config.USE_TENSORRT_RTX:
                    assert config.engine_capability == trt.EngineCapability.STANDARD
                else:
                    assert config.engine_capability == trt.EngineCapability.DEFAULT
            with contextlib.suppress(AttributeError):
                assert not config.get_flag(trt.BuilderFlag.DIRECT_IO)

    @pytest.mark.parametrize(
        "engine_capability",
        [
            trt.EngineCapability.STANDARD,
            trt.EngineCapability.SAFETY,
            trt.EngineCapability.DLA_STANDALONE,
        ],
    )
    def test_engine_capability(self, identity_builder_network, engine_capability):
        builder, network = identity_builder_network
        loader = CreateConfig(engine_capability=engine_capability)
        with loader(builder, network) as config:
            assert config.engine_capability == engine_capability

    @pytest.mark.skipif(
        mod.version(trt.__version__) >= mod.version("11.0")
        or polygraphy_config.USE_TENSORRT_RTX,
        reason="Precision constraints flags not available in this TRT version",
    )
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

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6")
        and not polygraphy_config.USE_TENSORRT_RTX,
        reason="Unsupported before TRT 8.6",
    )
    @pytest.mark.parametrize(
        "kwargs, expected_flag",
        [
            ({"version_compatible": True}, "VERSION_COMPATIBLE"),
            (
                {"version_compatible": True, "exclude_lean_runtime": True},
                "EXCLUDE_LEAN_RUNTIME",
            ),
        ],
    )
    def test_version_compatibility_flags(
        self, identity_builder_network, kwargs, expected_flag
    ):
        builder, network = identity_builder_network
        loader = CreateConfig(**kwargs)
        with loader(builder, network) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, expected_flag))

    def test_direct_io(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig(direct_io=True)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.DIRECT_IO)

    @pytest.mark.parametrize("flag", [True, False])
    def test_restricted(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(restricted=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.SAFETY_SCOPE) == flag

    @pytest.mark.parametrize(
        "arg_name, flag_type",
        [
            ("refittable", trt.BuilderFlag.REFIT),
        ]
        + (
            [
                (
                    "disable_compilation_cache",
                    trt.BuilderFlag.DISABLE_COMPILATION_CACHE,
                ),
            ]
            if mod.version(trt.__version__) >= mod.version("9.0")
            else []
        )
        + (
            [
                ("strip_plan", trt.BuilderFlag.STRIP_PLAN),
            ]
            if mod.version(trt.__version__) >= mod.version("10.0")
            else []
        )
        + (
            (
                # The precision-enable BuilderFlags (FP16/INT8/BF16/FP8) were removed
                # in TensorRT 11 in favor of strongly typed networks.
                [
                    ("fp16", trt.BuilderFlag.FP16),
                    ("int8", trt.BuilderFlag.INT8),
                ]
                + (
                    [
                        ("bf16", trt.BuilderFlag.BF16),
                    ]
                    if mod.version(trt.__version__) >= mod.version("8.7")
                    else []
                )
                + (
                    [
                        ("fp8", trt.BuilderFlag.FP8),
                    ]
                    if mod.version(trt.__version__) >= mod.version("8.6")
                    else []
                )
                if mod.version(trt.__version__) < mod.version("11.0")
                else []
            )
            # GPU_FALLBACK and TF32 still exist on TRT 11, but not on TensorRT-RTX.
            + [
                ("allow_gpu_fallback", trt.BuilderFlag.GPU_FALLBACK),
                ("tf32", trt.BuilderFlag.TF32),
            ]
            if not polygraphy_config.USE_TENSORRT_RTX
            else []
        ),
    )
    @pytest.mark.parametrize("value", [True, False])
    def test_flags(self, identity_builder_network, arg_name, flag_type, value):
        builder, network = identity_builder_network
        # allow_gpu_fallback is only supported if use_dla is also set
        if arg_name == "allow_gpu_fallback":
            # False test requires DLA hardware, so check for it here.
            if value == False and not has_dla():
                pytest.skip("DLA hardware is not available on this system")
            loader = CreateConfig(use_dla=True, allow_gpu_fallback=value)
        else:
            loader = CreateConfig(**{arg_name: value})

        with loader(builder, network) as config:
            assert config.get_flag(flag_type) == value

    @pytest.mark.parametrize("flag", [True, False])
    def test_sparse_weights(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(sparse_weights=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.SPARSE_WEIGHTS) == flag

    @pytest.mark.skipif(
        polygraphy_config.USE_TENSORRT_RTX, reason="TensorRT-RTX does not support DLA"
    )
    def test_use_dla(self, identity_builder_network):
        builder, network = identity_builder_network
        loader = CreateConfig(use_dla=True, allow_gpu_fallback=True)
        with loader(builder, network) as config:
            assert config.default_device_type == trt.DeviceType.DLA
            if has_dla():
                assert config.DLA_core == 0

    @pytest.mark.skipif(
        polygraphy_config.USE_TENSORRT_RTX, reason="TensorRT-RTX does not support DLA"
    )
    @pytest.mark.parametrize("allow_gpu_fallback", [True, False])
    def test_use_dla_implicit_flags(self, identity_builder_network, allow_gpu_fallback):
        # PREFER_PRECISION_CONSTRAINTS and DIRECT_IO are only set implicitly before TRT 11.
        if not allow_gpu_fallback and not has_dla():
            pytest.skip("DLA hardware is not available on this system")

        builder, network = identity_builder_network
        loader = CreateConfig(use_dla=True, allow_gpu_fallback=allow_gpu_fallback)
        pre_trt_11 = mod.version(trt.__version__) < mod.version("11.0")

        with loader(builder, network) as config:
            with contextlib.suppress(AttributeError):
                assert (
                    config.get_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                    == pre_trt_11
                )
            with contextlib.suppress(AttributeError):
                assert config.get_flag(trt.BuilderFlag.DIRECT_IO) == (
                    pre_trt_11 and not allow_gpu_fallback
                )

    DEFAULT_TACTIC_SOURCES = 31
    if polygraphy_config.USE_TENSORRT_RTX:
        DEFAULT_TACTIC_SOURCES = 3
    elif mod.version(trt.__version__) >= mod.version("11.0"):
        # CUBLAS/CUBLAS_LT/CUDNN tactic sources were removed in TRT 11; only
        # EDGE_MASK_CONVOLUTIONS (bit 0) and JIT_CONVOLUTIONS (bit 1) remain.
        DEFAULT_TACTIC_SOURCES = 3
    elif mod.version(trt.__version__) >= mod.version("10.0"):
        DEFAULT_TACTIC_SOURCES = 24
    elif mod.version(trt.__version__) >= mod.version("8.7"):
        DEFAULT_TACTIC_SOURCES = 29

    def add_tactic_sources_case(cases, expected, *source_names):
        try:
            sources = [getattr(trt.TacticSource, name) for name in source_names]
        except AttributeError:
            return

        cases.append((sources, expected))

    TACTIC_SOURCES_CASES = []
    if hasattr(trt, "TacticSource"):
        TACTIC_SOURCES_CASES = [
            (None, DEFAULT_TACTIC_SOURCES),
            ([], 0),
        ]

        for source_names, expected in [
            (("CUBLAS",), 1),
            (("CUBLAS_LT",), 2),
            (("CUDNN",), 4),
            (("CUBLAS", "CUBLAS_LT"), 3),
            (("CUBLAS", "CUDNN"), 5),
            (("CUBLAS_LT", "CUDNN"), 6),
            (("CUDNN", "CUBLAS", "CUBLAS_LT"), 7),
            (("CUDNN", "CUBLAS", "CUBLAS_LT", "EDGE_MASK_CONVOLUTIONS"), 15),
            (
                (
                    "CUDNN",
                    "CUBLAS",
                    "CUBLAS_LT",
                    "EDGE_MASK_CONVOLUTIONS",
                    "JIT_CONVOLUTIONS",
                ),
                31,
            ),
        ]:
            add_tactic_sources_case(TACTIC_SOURCES_CASES, expected, *source_names)

    @pytest.mark.parametrize("sources, expected", TACTIC_SOURCES_CASES)
    def test_tactic_sources(self, identity_builder_network, sources, expected):
        builder, network = identity_builder_network
        loader = CreateConfig(tactic_sources=sources)
        with loader(builder, network) as config:
            assert config.get_tactic_sources() == expected

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.7")
        and not polygraphy_config.USE_TENSORRT_RTX,
        reason="API was added in TRT 8.7",
    )
    @pytest.mark.parametrize("flag", [True, False])
    def test_error_on_timing_cache_miss(self, identity_builder_network, flag):
        builder, network = identity_builder_network
        loader = CreateConfig(error_on_timing_cache_miss=flag)
        with loader(builder, network) as config:
            assert config.get_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS) == flag

    @pytest.mark.skipif(
        polygraphy_config.USE_TENSORRT_RTX
        or mod.version(trt.__version__) >= mod.version("11.0"),
        reason="TensorRT-RTX and TRT 11 do not support INT8 calibrators",
    )
    def test_calibrator_metadata_set(self, identity_builder_network):
        builder, network = identity_builder_network
        calibrator = Calibrator(DataLoader())
        loader = CreateConfig(int8=True, calibrator=calibrator)
        with loader(builder, network) as config:
            assert config.int8_calibrator
            assert "x" in calibrator.data_loader.input_metadata
            meta = calibrator.data_loader.input_metadata["x"]
            assert meta.shape == BoundedShape((1, 1, 2, 2))
            assert meta.dtype == DataType.FLOAT32

    def test_multiple_profiles(self, identity_builder_network):
        builder, network = identity_builder_network
        profiles = [
            Profile().add("x", (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4)),
            Profile().add("x", (1, 2, 4, 4), (1, 2, 8, 8), (1, 2, 16, 16)),
        ]
        loader = CreateConfig(profiles=profiles)
        with loader(builder, network) as config:
            assert config.num_optimization_profiles == 2

    @pytest.mark.parametrize("path_mode", [True, False], ids=["path", "file-like"])
    def test_timing_cache(self, identity_builder_network, path_mode):
        builder, network = identity_builder_network
        with util.NamedTemporaryFile() as cache:
            loader = CreateConfig(load_timing_cache=cache.name if path_mode else cache)
            with loader(builder, network) as config:
                assert config.get_timing_cache()

    def test_fall_back_to_empty_timing_cache(self, identity_builder_network):
        """Tests that passing in a nonexistent timing cache path is non-fatal"""
        builder, network = identity_builder_network
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_name = os.path.join(tmpdir, "casper")
            loader = CreateConfig(load_timing_cache=cache_name)
            with loader(builder, network) as config:
                assert config.get_timing_cache()

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
            polygraphy_config.USE_TENSORRT_RTX,
            reason="TensorRT-RTX does not support DLA memory pools",
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

    @pytest.mark.parametrize(
        "preview_features",
        [
            (
                # MULTIDEVICE_RUNTIME_10_16 graduated out of preview in TRT 11;
                # use a preview feature that still exists there.
                [trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03]
                if (mod.version(trt.__version__) >= mod.version("11.0"))
                else (
                    [trt.PreviewFeature.PROFILE_SHARING_0806]
                    if mod.version(trt.__version__) >= mod.version("10.0")
                    else (
                        [trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03]
                        if polygraphy_config.USE_TENSORRT_RTX
                        else [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]
                    )
                )
            ),
        ],
    )
    def test_preview_features(self, identity_builder_network, preview_features):
        builder, network = identity_builder_network
        loader = CreateConfig(preview_features=preview_features)
        with loader(builder, network) as config:
            # Check that only the enabled preview features are on.
            for pf in trt.PreviewFeature.__members__.values():
                expected = pf in preview_features
                assert config.get_preview_feature(pf) == expected

    @pytest.mark.skipif(
        polygraphy_config.USE_TENSORRT_RTX
        or mod.version(trt.__version__) >= mod.version("11.0"),
        reason="TensorRT-RTX and TRT 11.0 and newer do not support quantization_flag API",
    )
    @pytest.mark.parametrize(
        "calibrate_before_fusion",
        [True, False],
    )
    def test_quantization_flags(
        self, identity_builder_network, calibrate_before_fusion
    ):
        builder, network = identity_builder_network
        loader = CreateConfig(
            quantization_flags=(
                [trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION]
                if calibrate_before_fusion
                else []
            )
        )
        with loader(builder, network) as config:
            # Check that only the enabled quantization flags are on.
            for qf in trt.QuantizationFlag.__members__.values():
                if qf == trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION:
                    assert config.get_quantization_flag(qf) == calibrate_before_fusion
                else:
                    assert config.get_quantization_flag(qf) == False

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6")
        and not polygraphy_config.USE_TENSORRT_RTX,
        reason="Unsupported for TRT versions prior to 8.6",
    )
    @pytest.mark.parametrize("level", range(6))
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

    if mod.version(trt.__version__) >= mod.version("10.2"):

        @pytest.mark.parametrize(
            "platform",
            [
                trt.RuntimePlatform.SAME_AS_BUILD,
                pytest.param(
                    trt.RuntimePlatform.WINDOWS_AMD64,
                    marks=pytest.mark.skipif(
                        not polygraphy_config.USE_TENSORRT_RTX,
                        reason="TensorRT-RTX not enabled",
                    ),
                ),
            ],
        )
        def test_runtime_platform(self, identity_builder_network, platform):
            builder, network = identity_builder_network
            loader = CreateConfig(runtime_platform=platform)
            with loader(builder, network) as config:
                assert config.runtime_platform == platform

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6")
        and not polygraphy_config.USE_TENSORRT_RTX,
        reason="Unsupported for TRT versions prior to 8.6",
    )
    @pytest.mark.parametrize("num_streams", range(3))
    def test_max_aux_streams(self, identity_builder_network, num_streams):
        builder, network = identity_builder_network
        loader = CreateConfig(max_aux_streams=num_streams)
        with loader(builder, network) as config:
            assert config.max_aux_streams == num_streams

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("9.0")
        and not polygraphy_config.USE_TENSORRT_RTX,
        reason="API was added in TRT 9.0",
    )
    def test_progress_monitor(self, identity_builder_network):
        class DummyProgressMonitor(trt.IProgressMonitor):
            def __init__(self):
                trt.IProgressMonitor.__init__(self)

            def phase_start(self, phase_name, parent_phase, num_steps):
                pass

            def phase_finish(self, phase_name):
                pass

            def step_complete(self, phase_name, step):
                return True

        builder, network = identity_builder_network
        progress_monitor = DummyProgressMonitor()
        loader = CreateConfig(progress_monitor=progress_monitor)
        with loader(builder, network) as config:
            assert config.progress_monitor == progress_monitor

    if (
        mod.version(trt.__version__) >= mod.version("10.8")
        and not polygraphy_config.USE_TENSORRT_RTX
    ):

        @pytest.mark.parametrize(
            "level",
            [
                trt.TilingOptimizationLevel.NONE,
                trt.TilingOptimizationLevel.FAST,
                trt.TilingOptimizationLevel.MODERATE,
                trt.TilingOptimizationLevel.FULL,
            ],
        )
        def test_tiling_optimization_level(self, identity_builder_network, level):
            builder, network = identity_builder_network
            loader = CreateConfig(tiling_optimization_level=level)
            with loader(builder, network) as config:
                assert config.tiling_optimization_level == level


class TestPostprocessConfig:
    def test_with_config(self, identity_builder_network):
        builder, network = identity_builder_network
        config = CreateConfig()(builder, network)
        assert not config.get_flag(trt.BuilderFlag.REFIT)

        config = postprocess_config(
            config,
            func=lambda builder, network, config: config.set_flag(
                trt.BuilderFlag.REFIT
            ),
            builder=builder,
            network=network,
        )
        assert config.get_flag(trt.BuilderFlag.REFIT)

    def test_with_config_callable(self, identity_builder_network):
        builder, network = identity_builder_network
        config = CreateConfig()

        config = postprocess_config(
            config,
            func=lambda builder, network, config: config.set_flag(
                trt.BuilderFlag.REFIT
            ),
            builder=builder,
            network=network,
        )
        assert config.get_flag(trt.BuilderFlag.REFIT)
