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
import os
from textwrap import dedent

import polygraphy.tools.args.util as args_util
import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.backend.trt import TacticRecorder, TacticReplayData, TacticReplayer, create_network
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, TrtConfigArgs
from tests.helper import has_dla
from tests.tools.args.helper import ArgGroupTestHelper


@pytest.fixture()
def trt_config_args():
    return ArgGroupTestHelper(
        TrtConfigArgs(allow_engine_capability=True, allow_tensor_formats=True), deps=[ModelArgs(), DataLoaderArgs()]
    )


class TestTrtConfigArgs:
    def test_defaults(self, trt_config_args):
        trt_config_args.parse_args([])
        assert trt_config_args._workspace is None

    def test_create_config(self, trt_config_args):
        trt_config_args.parse_args([])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert isinstance(config, trt.IBuilderConfig)

    @pytest.mark.parametrize(
        "args, flag",
        [
            (["--int8"], "INT8"),
            (["--fp16"], "FP16"),
            (["--fp8"], "FP8"),
            (["--tf32"], "TF32"),
            (["--allow-gpu-fallback"], "GPU_FALLBACK"),
            (["--precision-constraints", "obey"], "OBEY_PRECISION_CONSTRAINTS"),
            (["--precision-constraints", "prefer"], "PREFER_PRECISION_CONSTRAINTS"),
            (["--direct-io"], "DIRECT_IO"),
        ],
    )
    def test_flags(self, trt_config_args, args, flag):
        if flag == "TF32" and mod.version(trt.__version__) < mod.version("7.1"):
            pytest.skip("TF32 support was added in 7.1")
        if flag == "FP8" and mod.version(trt.__version__) < mod.version("8.6"):
            pytest.skip("FP8 support was added in 8.6")

        if (
            flag == "OBEY_PRECISION_CONSTRAINTS"
            or flag == "PREFER_PRECISION_CONSTRAINTS"
            or flag == "DIRECT_IO"
            and mod.version(trt.__version__) < mod.version("8.2")
        ):
            pytest.skip("OBEY_PRECISION_CONSTRAINTS/PREFER_PRECISION_CONSTRAINTS/DIRECT_IO support was added in 8.2")

        trt_config_args.parse_args(args)

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, flag))

    @pytest.mark.parametrize(
        "workspace, expected",
        [
            ("16", 16),
            ("1e9", 1e9),
            ("2M", 2 << 20),
        ],
    )
    def test_workspace(self, trt_config_args, workspace, expected):
        trt_config_args.parse_args(["--workspace", workspace])
        assert trt_config_args._workspace == expected

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.max_workspace_size == expected

    if mod.version(trt.__version__) >= mod.version("8.0"):

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
            assert str(trt_config_args.engine_capability) == f"trt.EngineCapability.{engine_capability.upper()}"

            builder, network = create_network()
            with builder, network, trt_config_args.create_config(builder, network=network) as config:
                assert config.engine_capability == expected

    def test_dla(self, trt_config_args):
        trt_config_args.parse_args(["--use-dla"])
        assert trt_config_args.use_dla

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.default_device_type == trt.DeviceType.DLA
            if has_dla():
                assert config.DLA_core == 0

    def test_calibrator_when_dla(self, trt_config_args):
        trt_config_args.parse_args(["--use-dla", "--int8"])

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert isinstance(config.int8_calibrator, trt.IInt8EntropyCalibrator2)

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="SAFETY_SCOPE was added in TRT 8")
    def test_restricted_flags(self, trt_config_args):
        trt_config_args.parse_args(["--trt-safety-restricted"])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, "SAFETY_SCOPE"))

    def test_refittable_flags(self, trt_config_args):
        trt_config_args.parse_args(["--refittable"])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, "REFIT"))

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Bugged before TRT 8")
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
            with builder, network, trt_config_args.create_config(builder, network=network) as config:
                selector = config.algorithm_selector
                assert selector.make_func == cls
                assert selector.path == f.name

    if mod.version(trt.__version__) < mod.version("8.0"):
        TACTIC_SOURCES_CASES = [
            ([], 3),  # By default, all sources are enabled.
            (["--tactic-sources"], 0),
            (["--tactic-sources", "CUBLAS"], 1),
            (["--tactic-sources", "CUBLAS_LT"], 2),
            (["--tactic-sources", "CUblAS", "cublas_lt"], 3),  # Not case sensitive
        ]

    if mod.version(trt.__version__) >= mod.version("8.0"):
        TACTIC_SOURCES_CASES = [
            ([], 7),  # By default, all sources are enabled.
            (["--tactic-sources"], 0),
            (["--tactic-sources", "CUBLAS"], 1),
            (["--tactic-sources", "CUBLAS_LT"], 2),
            (["--tactic-sources", "CUDNN"], 4),
            (["--tactic-sources", "CUblAS", "cublas_lt"], 3),  # Not case sensitive
            (["--tactic-sources", "CUBLAS", "cuDNN"], 5),
            (["--tactic-sources", "CUBLAS_LT", "CUDNN"], 6),
            (["--tactic-sources", "CUDNN", "cuBLAS", "CUBLAS_LT"], 7),
        ]

    if mod.version(trt.__version__) >= mod.version("8.4"):
        TACTIC_SOURCES_CASES[0] = ([], 15)
        TACTIC_SOURCES_CASES.extend(
            [(["--tactic-sources", "CUDNN", "cuBLAS", "CUBLAS_LT", "edge_mask_convolutions"], 15)]
        )

    if mod.version(trt.__version__) >= mod.version("8.5"):
        TACTIC_SOURCES_CASES[0] = ([], 31)
        TACTIC_SOURCES_CASES.extend(
            [(["--tactic-sources", "CUDNN", "cuBLAS", "CUBLAS_LT", "edge_mask_convolutions", "jit_convolutions"], 31)]
        )

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.2"), reason="Not available before 7.2")
    @pytest.mark.parametrize("opt, expected", TACTIC_SOURCES_CASES)
    def test_tactic_sources(self, trt_config_args, opt, expected):
        trt_config_args.parse_args(opt)
        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.get_tactic_sources() == expected

    @pytest.mark.parametrize("base_class", ["IInt8LegacyCalibrator", "IInt8EntropyCalibrator2"])
    def test_calibration_base_class(self, trt_config_args, base_class):
        trt_config_args.parse_args(["--int8", "--calibration-base-class", base_class])
        assert trt_config_args.calibration_base_class.unwrap() == f"trt.{base_class}"

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
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
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
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

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
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

            trt_config_args.parse_args(["--trt-config-script", f"{f.name}:my_load_config"])
            assert trt_config_args.trt_config_script == f.name
            assert trt_config_args.trt_config_func_name == "my_load_config"

            builder, network = create_network()
            with builder, network, trt_config_args.create_config(builder, network) as config:
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

            trt_config_args.parse_args(["--trt-config-postprocess-script", f"{f.name}:my_postprocess_config", "--int8"])
            assert trt_config_args.trt_config_postprocess_script == f.name
            assert trt_config_args.trt_config_postprocess_func_name == "my_postprocess_config"

            builder, network = create_network()
            with builder, network, trt_config_args.create_config(builder, network) as config:
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
            ["--int8", "--calibration-base-class", "IInt8LegacyCalibrator));print(('hi'"],
            ["--int8", "--calibration-base-class", "IInt8LegacyCalibrator;print(('hi')"],
            ["--int8", "--calibration-base-class", "IInt8LegacyCalibrator';print('hi')"],
            ["--tactic-sources", "CUBLAS, fp16=True"],
        ],
    )
    def test_code_injection_checks(self, trt_config_args, args):
        with pytest.raises(PolygraphyException):
            trt_config_args.parse_args(args)

    with contextlib.suppress(AttributeError):

        @pytest.mark.skipif(
            mod.version(trt.__version__) < mod.version("8.3"), reason="Unsupported for TRT versions prior to 8.3"
        )
        @pytest.mark.parametrize(
            "args,expected",
            [
                (["--pool-limit", "workspace:250"], {trt.MemoryPoolType.WORKSPACE: 250}),
                (["--pool-limit", "dla_managed_sram:250"], {trt.MemoryPoolType.DLA_MANAGED_SRAM: 250}),
                (["--pool-limit", "dla_local_dram:250"], {trt.MemoryPoolType.DLA_LOCAL_DRAM: 250}),
                (["--pool-limit", "dla_global_dram:250"], {trt.MemoryPoolType.DLA_GLOBAL_DRAM: 250}),
                # Test case insensitivity
                (["--pool-limit", "wOrkSpaCE:250"], {trt.MemoryPoolType.WORKSPACE: 250}),
                # Test works with K/M/G suffixes
                (["--pool-limit", "workspace:2M"], {trt.MemoryPoolType.WORKSPACE: 2 << 20}),
                # Test works with scientific notation
                (["--pool-limit", "workspace:2e3"], {trt.MemoryPoolType.WORKSPACE: 2e3}),
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

        @pytest.mark.skipif(
            mod.version(trt.__version__) < mod.version("8.3"), reason="Unsupported for TRT versions prior to 8.3"
        )
        @pytest.mark.parametrize(
            "args",
            [
                ["--pool-limit", "250"],
            ],
        )
        def test_memory_pool_limits_empty_key_not_allowed(self, args, trt_config_args):
            with pytest.raises(PolygraphyException, match="Could not parse argument"):
                trt_config_args.parse_args(args)

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.5"), reason="Unsupported for TRT versions prior to 8.5"
    )
    @pytest.mark.parametrize(
        "preview_features",
        [
            [],
            ["FASter_DYNAMIC_ShAPeS_0805"],
            ["FASter_DYNAMIC_ShAPeS_0805", "DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805"],
        ],
    )
    def test_preview_features(self, trt_config_args, preview_features):
        # Flag should be case-insensitive
        trt_config_args.parse_args(["--preview-features"] + preview_features)
        builder, network = create_network()

        sanitized_preview_features = [pf.upper() for pf in preview_features]

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            # Check that only the enabled preview features are on.
            for name, pf in trt.PreviewFeature.__members__.items():
                assert config.get_preview_feature(pf) == (name in sanitized_preview_features)

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.6"), reason="Unsupported for TRT versions prior to 8.6"
    )
    @pytest.mark.parametrize("level", range(5))
    def test_builder_optimization_level(self, trt_config_args, level):
        trt_config_args.parse_args(["--builder-optimization-level", str(level)])
        assert trt_config_args.builder_optimization_level == level

        builder, network = create_network()

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
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
                str(trt_config_args.hardware_compatibility_level) == f"trt.HardwareCompatibilityLevel.{expected.name}"
            )

            builder, network = create_network()

            with builder, network, trt_config_args.create_config(builder, network=network) as config:
                assert config.hardware_compatibility_level == expected
