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

import contextlib
import os
from textwrap import dedent

import polygraphy.tools.args.util as args_util
import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.backend.trt import (
    TacticRecorder,
    TacticReplayData,
    TacticReplayer,
    create_network,
)
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, TrtConfigArgs
from tests.helper import has_dla
from tests.tools.args.helper import ArgGroupTestHelper


@pytest.fixture()
def trt_config_args():
    return ArgGroupTestHelper(
        TrtConfigArgs(allow_engine_capability=True, allow_tensor_formats=True),
        deps=[ModelArgs(), DataLoaderArgs()],
    )


class TestTrtConfigArgs:
    def test_defaults(self, trt_config_args):
        trt_config_args.parse_args([])

    def test_create_config(self, trt_config_args):
        trt_config_args.parse_args([])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert isinstance(config, trt.IBuilderConfig)

    @pytest.mark.parametrize(
        "args, flag",
        [
            (["--int8"], "INT8"),
            (["--fp16"], "FP16"),
            (["--bf16"], "BF16"),
            (["--fp8"], "FP8"),
            (["--tf32"], "TF32"),
            (["--allow-gpu-fallback"], "GPU_FALLBACK"),
            (["--precision-constraints", "obey"], "OBEY_PRECISION_CONSTRAINTS"),
            (["--precision-constraints", "prefer"], "PREFER_PRECISION_CONSTRAINTS"),
            (["--direct-io"], "DIRECT_IO"),
            (["--disable-compilation-cache"], "DISABLE_COMPILATION_CACHE"),
        ],
    )
    def test_flags(self, trt_config_args, args, flag):
        if flag == "FP8" and mod.version(trt.__version__) < mod.version("8.6"):
            pytest.skip("FP8 support was added in 8.6")
        if flag == "BF16" and mod.version(trt.__version__) < mod.version("8.7"):
            pytest.skip("BF16 support was added in 8.7")
        if flag == "DISABLE_COMPILATION_CACHE" and mod.version(
            trt.__version__
        ) < mod.version("9.0"):
            pytest.skip("BF16 support was added in 9.0")

        trt_config_args.parse_args(args)

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, flag))

    @pytest.mark.parametrize(
        "engine_capability, expected",
        [
            ("Standard", trt.EngineCapability.STANDARD),
            ("SaFETY", trt.EngineCapability.SAFETY),
            ("DLA_STANDALONE", trt.EngineCapability.DLA_STANDALONE),
        ],
    )
    def test_engine_capability(self, trt_config_args, engine_capability, expected):
        trt_config_args.parse_args(["--engine-capability", engine_capability])
        assert (
            str(trt_config_args.engine_capability)
            == f"trt.EngineCapability.{engine_capability.upper()}"
        )

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.engine_capability == expected

    def test_dla(self, trt_config_args):
        trt_config_args.parse_args(["--use-dla"])
        assert trt_config_args.use_dla

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.default_device_type == trt.DeviceType.DLA
            if has_dla():
                assert config.DLA_core == 0

    def test_calibrator_when_dla(self, trt_config_args):
        trt_config_args.parse_args(["--use-dla", "--int8"])

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert isinstance(config.int8_calibrator, trt.IInt8EntropyCalibrator2)

    def test_restricted_flags(self, trt_config_args):
        trt_config_args.parse_args(["--trt-safety-restricted"])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, "SAFETY_SCOPE"))

    def test_refittable_flags(self, trt_config_args):
        trt_config_args.parse_args(["--refittable"])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, "REFIT"))

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("10.0"),
        reason="Feature not present before 10.0",
    )
    def test_weight_streaming_flags(self, trt_config_args):
        trt_config_args.parse_args(["--weight-streaming"])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, "WEIGHT_STREAMING"))

    @pytest.mark.parametrize(
        "opt, cls",
        [
            ("--save-tactics", TacticRecorder),
            ("--load-tactics", TacticReplayer),
        ],
    )
    def test_tactics(self, trt_config_args, opt, cls):
        with util.NamedTemporaryFile("w+", suffix=".json") as f:
            if opt == "--load-tactics":
                TacticReplayData().save(f)

            trt_config_args.parse_args([opt, f.name])
            builder, network = create_network()
            with builder, network, trt_config_args.create_config(
                builder, network=network
            ) as config:
                selector = config.algorithm_selector
                assert selector.make_func == cls
                assert selector.path == f.name

    TACTIC_SOURCES_CASES = [
        ([], 31),  # By default, all sources are enabled.
        (["--tactic-sources"], 0),
        (["--tactic-sources", "CUBLAS"], 1),
        (["--tactic-sources", "CUBLAS_LT"], 2),
        (["--tactic-sources", "CUDNN"], 4),
        (["--tactic-sources", "CUblAS", "cublas_lt"], 3),  # Not case sensitive
        (["--tactic-sources", "CUBLAS", "cuDNN"], 5),
        (["--tactic-sources", "CUBLAS_LT", "CUDNN"], 6),
        (["--tactic-sources", "CUDNN", "cuBLAS", "CUBLAS_LT"], 7),
        (
            [
                "--tactic-sources",
                "CUDNN",
                "cuBLAS",
                "CUBLAS_LT",
                "edge_mask_convolutions",
            ],
            15,
        ),
        (
            [
                "--tactic-sources",
                "CUDNN",
                "cuBLAS",
                "CUBLAS_LT",
                "edge_mask_convolutions",
                "jit_convolutions",
            ],
            31,
        ),
    ]

    if mod.version(trt.__version__) >= mod.version("10.0"):
        TACTIC_SOURCES_CASES[0] = ([], 24)
    elif mod.version(trt.__version__) >= mod.version("8.7"):
        TACTIC_SOURCES_CASES[0] = ([], 29)

    @pytest.mark.parametrize("opt, expected", TACTIC_SOURCES_CASES)
    def test_tactic_sources(self, trt_config_args, opt, expected):
        trt_config_args.parse_args(opt)
        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.get_tactic_sources() == expected

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.7"),
        reason="ERROR_ON_TIMING_CACHE_MISS support was added in 8.7",
    )
    def test_error_on_timing_cache_miss(self, trt_config_args):
        trt_config_args.parse_args(["--error-on-timing-cache-miss"])
        builder, network = create_network()

        assert trt_config_args.error_on_timing_cache_miss
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.get_flag(
                getattr(trt.BuilderFlag, "ERROR_ON_TIMING_CACHE_MISS")
            )

    @pytest.mark.parametrize(
        "base_class", ["IInt8LegacyCalibrator", "IInt8EntropyCalibrator2"]
    )
    def test_calibration_base_class(self, trt_config_args, base_class):
        trt_config_args.parse_args(["--int8", "--calibration-base-class", base_class])
        assert trt_config_args.calibration_base_class.unwrap() == f"trt.{base_class}"

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert isinstance(config.int8_calibrator, getattr(trt, base_class))

    def test_legacy_calibrator_params(self, trt_config_args):
        quantile = 0.25
        regression_cutoff = 0.9
        trt_config_args.parse_args(
            [
                "--int8",
                "--calibration-base-class=IInt8LegacyCalibrator",
                "--quantile",
                str(quantile),
                "--regression-cutoff",
                str(regression_cutoff),
            ]
        )
        assert trt_config_args._quantile == quantile
        assert trt_config_args._regression_cutoff == regression_cutoff

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.int8_calibrator.get_quantile() == quantile
            assert config.int8_calibrator.get_regression_cutoff() == regression_cutoff

    def test_no_deps_profiles_int8(self, trt_config_args):
        trt_config_args.parse_args(
            [
                "--trt-min-shapes=input:[1,25,25]",
                "--trt-opt-shapes=input:[2,25,25]",
                "--trt-max-shapes=input:[4,25,25]",
                "--int8",
            ]
        )

        for profile in trt_config_args.profile_dicts:
            assert profile["input"][0] == [1, 25, 25]
            assert profile["input"][1] == [2, 25, 25]
            assert profile["input"][2] == [4, 25, 25]

        builder, network = create_network()
        network.add_input("input", shape=(-1, 25, 25), dtype=trt.float32)

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert isinstance(config, trt.IBuilderConfig)
            # Unfortunately there is no API to check the contents of the profile in a config.
            # The checks above will have to do.
            assert config.num_optimization_profiles == 1
            assert config.get_calibration_profile().get_shape("input") == [
                tuple(s) for s in trt_config_args.profile_dicts[0]["input"]
            ]
            assert config.get_flag(trt.BuilderFlag.INT8)

    def test_config_script_default_func(self, trt_config_args):
        trt_config_args.parse_args(["--trt-config-script", "example.py"])
        assert trt_config_args.trt_config_func_name == "load_config"

    def test_config_script(self, trt_config_args):
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(
                dedent(
                    """
                    from polygraphy.backend.trt import CreateConfig
                    from polygraphy import func
                    import tensorrt as trt

                    @func.extend(CreateConfig())
                    def my_load_config(config):
                        config.set_flag(trt.BuilderFlag.FP16)
                    """
                )
            )
            f.flush()
            os.fsync(f.fileno())

            trt_config_args.parse_args(
                ["--trt-config-script", f"{f.name}:my_load_config"]
            )
            assert trt_config_args.trt_config_script == f.name
            assert trt_config_args.trt_config_func_name == "my_load_config"

            builder, network = create_network()
            with builder, network, trt_config_args.create_config(
                builder, network
            ) as config:
                assert isinstance(config, trt.IBuilderConfig)
                assert config.get_flag(trt.BuilderFlag.FP16)

    def test_config_postprocess_script_default_func(self, trt_config_args):
        trt_config_args.parse_args(["--trt-config-postprocess-script", "example.py"])
        assert trt_config_args.trt_config_postprocess_func_name == "postprocess_config"

    def test_config_postprocess_script(self, trt_config_args):
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(
                dedent(
                    """
                    import tensorrt as trt

                    def my_postprocess_config(builder, network, config):
                        config.set_flag(trt.BuilderFlag.FP16)
                    """
                )
            )
            f.flush()
            os.fsync(f.fileno())

            trt_config_args.parse_args(
                [
                    "--trt-config-postprocess-script",
                    f"{f.name}:my_postprocess_config",
                    "--int8",
                ]
            )
            assert trt_config_args.trt_config_postprocess_script == f.name
            assert (
                trt_config_args.trt_config_postprocess_func_name
                == "my_postprocess_config"
            )

            builder, network = create_network()
            with builder, network, trt_config_args.create_config(
                builder, network
            ) as config:
                assert isinstance(config, trt.IBuilderConfig)
                assert config.get_flag(trt.BuilderFlag.FP16)
                assert config.get_flag(trt.BuilderFlag.INT8)

    @pytest.mark.parametrize(
        "args",
        [
            ["--int8", "--calibration-base-class", "IInt8LegacyCalibrator'"],
            ["--int8", "--calibration-base-class", 'IInt8LegacyCalibrator"'],
            ["--int8", "--calibration-base-class", "IInt8LegacyCalibrator)"],
            ["--int8", "--calibration-base-class", "IInt8LegacyCalibrator}"],
            ["--int8", "--calibration-base-class", "IInt8LegacyCalibrator]"],
            [
                "--int8",
                "--calibration-base-class",
                "IInt8LegacyCalibrator));print(('hi'",
            ],
            [
                "--int8",
                "--calibration-base-class",
                "IInt8LegacyCalibrator;print(('hi')",
            ],
            [
                "--int8",
                "--calibration-base-class",
                "IInt8LegacyCalibrator';print('hi')",
            ],
            ["--tactic-sources", "CUBLAS, fp16=True"],
        ],
    )
    def test_code_injection_checks(self, trt_config_args, args):
        with pytest.raises(PolygraphyException):
            trt_config_args.parse_args(args)

    with contextlib.suppress(AttributeError):

        @pytest.mark.parametrize(
            "args,expected",
            [
                (
                    ["--pool-limit", "workspace:250"],
                    {trt.MemoryPoolType.WORKSPACE: 250},
                ),
                (
                    ["--pool-limit", "dla_managed_sram:250"],
                    {trt.MemoryPoolType.DLA_MANAGED_SRAM: 250},
                ),
                (
                    ["--pool-limit", "dla_local_dram:250"],
                    {trt.MemoryPoolType.DLA_LOCAL_DRAM: 250},
                ),
                (
                    ["--pool-limit", "dla_global_dram:250"],
                    {trt.MemoryPoolType.DLA_GLOBAL_DRAM: 250},
                ),
                # Test case insensitivity
                (
                    ["--pool-limit", "wOrkSpaCE:250"],
                    {trt.MemoryPoolType.WORKSPACE: 250},
                ),
                # Test works with K/M/G suffixes
                (
                    ["--pool-limit", "workspace:2M"],
                    {trt.MemoryPoolType.WORKSPACE: 2 << 20},
                ),
                # Test works with scientific notation
                (
                    ["--pool-limit", "workspace:2e3"],
                    {trt.MemoryPoolType.WORKSPACE: 2e3},
                ),
            ],
        )
        def test_memory_pool_limits(self, args, expected, trt_config_args):
            trt_config_args.parse_args(args)
            builder, network = create_network()
            loader = args_util.run_script(trt_config_args.add_to_script)
            assert loader.memory_pool_limits == expected
            with builder, network, loader(builder, network=network) as config:
                for pool_type, pool_size in expected.items():
                    if "dla" in pool_type.name.lower() and not has_dla():
                        pytest.skip("DLA is not available on this system")
                    config.get_memory_pool_limit(pool_type) == pool_size

        @pytest.mark.parametrize(
            "args",
            [
                ["--pool-limit", "250"],
            ],
        )
        def test_memory_pool_limits_empty_key_not_allowed(self, args, trt_config_args):
            with pytest.raises(PolygraphyException, match="Could not parse argument"):
                trt_config_args.parse_args(args)

    @pytest.mark.parametrize(
        "preview_features",
        [
            (
                ["PROFILE_SHAriNG_0806"]
                if mod.version(trt.__version__) >= mod.version("10.0")
                else ["FASter_DYNAMIC_ShAPeS_0805"]
            ),
        ],
    )
    def test_preview_features(self, trt_config_args, preview_features):
        # Flag should be case-insensitive
        trt_config_args.parse_args(["--preview-features"] + preview_features)
        builder, network = create_network()

        sanitized_preview_features = [pf.upper() for pf in preview_features]

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            # Check that only the enabled preview features are on.
            for name, pf in trt.PreviewFeature.__members__.items():
                assert config.get_preview_feature(pf) == (
                    name in sanitized_preview_features
                )

    @pytest.mark.parametrize(
        "quantization_flags",
        [
            [],
            ["CALIBRATE_BEFORE_FUSION"],
            ["cAlIBRaTE_BEFORE_fUSIoN"],
        ],
    )
    def test_quantization_flags(self, trt_config_args, quantization_flags):
        # Flag should be case-insensitive
        trt_config_args.parse_args(["--quantization-flags"] + quantization_flags)
        builder, network = create_network()

        sanitized_quantization_flags = [pf.upper() for pf in quantization_flags]

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            # Check that only the enabled quantization flags are on.
            for name, qf in trt.QuantizationFlag.__members__.items():
                assert config.get_quantization_flag(qf) == (
                    name in sanitized_quantization_flags
                )

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"),
        reason="Unsupported for TRT versions prior to 8.6",
    )
    @pytest.mark.parametrize("level", range(6))
    def test_builder_optimization_level(self, trt_config_args, level):
        trt_config_args.parse_args(["--builder-optimization-level", str(level)])
        assert trt_config_args.builder_optimization_level == level

        builder, network = create_network()

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.builder_optimization_level == level

    if mod.version(trt.__version__) >= mod.version("8.6"):

        @pytest.mark.parametrize(
            "level, expected",
            [
                ("none", trt.HardwareCompatibilityLevel.NONE),
                ("AMPERe_plUS", trt.HardwareCompatibilityLevel.AMPERE_PLUS),
                ("ampere_plus", trt.HardwareCompatibilityLevel.AMPERE_PLUS),
            ],
        )
        def test_hardware_compatibility_level(self, trt_config_args, level, expected):
            trt_config_args.parse_args(["--hardware-compatibility-level", str(level)])
            assert (
                str(trt_config_args.hardware_compatibility_level)
                == f"trt.HardwareCompatibilityLevel.{expected.name}"
            )

            builder, network = create_network()

            with builder, network, trt_config_args.create_config(
                builder, network=network
            ) as config:
                assert config.hardware_compatibility_level == expected

    if mod.version(trt.__version__) >= mod.version("10.2"):

        @pytest.mark.parametrize(
            "platform, expected",
            [
                ("same_as_build", trt.RuntimePlatform.SAME_AS_BUILD),
                ("windows_amd64", trt.RuntimePlatform.WINDOWS_AMD64),
                ("Windows_AMD64", trt.RuntimePlatform.WINDOWS_AMD64),
            ],
        )
        def test_runtime_platform(self, trt_config_args, platform, expected):
            trt_config_args.parse_args(["--runtime-platform", str(platform)])
            assert (
                str(trt_config_args.runtime_platform)
                == f"trt.RuntimePlatform.{expected.name}"
            )

            builder, network = create_network()

            with builder, network, trt_config_args.create_config(
                builder, network=network
            ) as config:
                assert config.runtime_platform == expected

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"),
        reason="Unsupported for TRT versions prior to 8.6",
    )
    @pytest.mark.parametrize("num_streams", range(5))
    def test_max_aux_streams(self, trt_config_args, num_streams):
        trt_config_args.parse_args(["--max-aux-streams", str(num_streams)])
        assert trt_config_args.max_aux_streams == num_streams

        builder, network = create_network()

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.max_aux_streams == num_streams

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"),
        reason="Unsupported before TRT 8.6",
    )
    @pytest.mark.parametrize(
        "args, attr, expected_flag",
        [
            (["--version-compatible"], "version_compatible", "VERSION_COMPATIBLE"),
            (
                ["--version-compatible", "--exclude-lean-runtime"],
                "exclude_lean_runtime",
                "EXCLUDE_LEAN_RUNTIME",
            ),
        ],
    )
    def test_version_compatibility(self, trt_config_args, args, attr, expected_flag):
        trt_config_args.parse_args(args)
        assert getattr(trt_config_args, attr)

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, expected_flag))

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"),
        reason="Unsupported before 8.6",
    )
    @pytest.mark.parametrize(
        "level, expected",
        [
            ("none", trt.ProfilingVerbosity.NONE),
            ("detailed", trt.ProfilingVerbosity.DETAILED),
            ("layer_names_only", trt.ProfilingVerbosity.LAYER_NAMES_ONLY),
        ],
    )
    def test_profiling_verbosity(self, trt_config_args, level, expected):
        trt_config_args.parse_args(["--profiling-verbosity", str(level)])
        assert (
            str(trt_config_args.profiling_verbosity)
            == f"trt.ProfilingVerbosity.{expected.name}"
        )

        builder, network = create_network()

        with builder, network, trt_config_args.create_config(
            builder, network=network
        ) as config:
            assert config.profiling_verbosity == expected
    
    if mod.version(trt.__version__) >= mod.version("10.8"):
        @pytest.mark.parametrize(
            "level, expected",
            [
                ("none", trt.TilingOptimizationLevel.NONE),
                ("fast", trt.TilingOptimizationLevel.FAST),
                ("moderate", trt.TilingOptimizationLevel.MODERATE),
                ("full", trt.TilingOptimizationLevel.FULL),
            ],
        )
        def test_tiling_optimization_level(self, trt_config_args, level, expected):
            trt_config_args.parse_args(["--tiling-optimization-level", str(level)])
            assert (
                str(trt_config_args.tiling_optimization_level)
                == f"trt.TilingOptimizationLevel.{expected.name}"
            )

            builder, network = create_network()

            with builder, network, trt_config_args.create_config(
                builder, network=network
            ) as config:
                assert config.tiling_optimization_level == expected