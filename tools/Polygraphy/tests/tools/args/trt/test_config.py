#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.backend.trt import TacticRecorder, TacticReplayData, TacticReplayer, create_network
from polygraphy.exception import PolygraphyException
from polygraphy.tools.args import DataLoaderArgs, ModelArgs, TrtConfigArgs
from tests.tools.args.helper import ArgGroupTestHelper


@pytest.fixture()
def trt_config_args():
    return ArgGroupTestHelper(TrtConfigArgs(), deps=[ModelArgs(), DataLoaderArgs()])


class TestTrtConfigArgs(object):
    def test_defaults(self, trt_config_args):
        trt_config_args.parse_args([])
        assert trt_config_args.workspace is None

    def test_create_config(self, trt_config_args):
        trt_config_args.parse_args([])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert isinstance(config, trt.IBuilderConfig)

    @pytest.mark.parametrize(
        "arg, flag",
        [
            ("--int8", "INT8"),
            ("--fp16", "FP16"),
            ("--tf32", "TF32"),
            ("--allow-gpu-fallback", "GPU_FALLBACK"),
        ],
    )
    def test_precision_flags(self, trt_config_args, arg, flag):
        if flag == "TF32" and mod.version(trt.__version__) < mod.version("7.1"):
            pytest.skip("TF32 support was added in 7.1")

        trt_config_args.parse_args([arg])

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
        assert trt_config_args.workspace == expected

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.max_workspace_size == expected

    def test_dla(self, trt_config_args):
        trt_config_args.parse_args(["--use-dla"])
        assert trt_config_args.use_dla

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.default_device_type == trt.DeviceType.DLA
            assert config.DLA_core == 0

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="SAFETY_SCOPE was added in TRT 8")
    def test_restricted_flags(self, trt_config_args):
        trt_config_args.parse_args(["--trt-safety-restricted"])
        builder, network = create_network()

        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.get_flag(getattr(trt.BuilderFlag, "SAFETY_SCOPE"))

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Bugged before TRT 8")
    def test_tactic_replay(self, trt_config_args):
        with util.NamedTemporaryFile(suffix=".json") as f:
            trt_config_args.parse_args(["--tactic-replay", f.name])
            builder, network = create_network()

            with builder, network, trt_config_args.create_config(builder, network=network) as config:
                recorder = config.algorithm_selector
                assert recorder.make_func == TacticRecorder
                assert recorder.path == f.name

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
                recorder = config.algorithm_selector
                assert recorder.make_func == cls
                assert recorder.path == f.name

    if mod.version(trt.__version__) < mod.version("8.0"):
        TACTIC_SOURCES_CASES = [
            ([], 3),  # By default, all sources are enabled.
            (["--tactic-sources"], 0),
            (["--tactic-sources", "CUBLAS"], 1),
            (["--tactic-sources", "CUBLAS_LT"], 2),
            (["--tactic-sources", "CUblAS", "cublas_lt"], 3),  # Not case sensitive
        ]
    else:
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
        assert trt_config_args.calibration_base_class.unwrap() == "trt.{:}".format(base_class)

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert isinstance(config.int8_calibrator, getattr(trt, base_class))

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
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
        assert trt_config_args.quantile == quantile
        assert trt_config_args.regression_cutoff == regression_cutoff

        builder, network = create_network()
        with builder, network, trt_config_args.create_config(builder, network=network) as config:
            assert config.int8_calibrator.get_quantile() == quantile
            assert config.int8_calibrator.get_regression_cutoff() == regression_cutoff

    def test_no_deps_profiles_int8(self):
        arg_group = ArgGroupTestHelper(TrtConfigArgs())
        arg_group.parse_args(
            [
                "--trt-min-shapes=input:[1,25,25]",
                "--trt-opt-shapes=input:[2,25,25]",
                "--trt-max-shapes=input:[4,25,25]",
                "--int8",
            ]
        )

        for (min_shapes, opt_shapes, max_shapes) in arg_group.profile_dicts:
            assert min_shapes["input"] == [1, 25, 25]
            assert opt_shapes["input"] == [2, 25, 25]
            assert max_shapes["input"] == [4, 25, 25]

        builder, network = create_network()

        with builder, network, arg_group.create_config(builder, network=network) as config:
            assert isinstance(config, trt.IBuilderConfig)
            # Unfortunately there is no API to check the contents of the profile in a config.
            # The checks above will have to do.
            assert config.num_optimization_profiles == 1
            assert config.get_flag(trt.BuilderFlag.INT8)

    def test_config_script(self):
        arg_group = ArgGroupTestHelper(TrtConfigArgs())

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

            arg_group.parse_args(["--trt-config-script", f.name, "--trt-config-func-name=my_load_config"])
            assert arg_group.trt_config_script == f.name
            assert arg_group.trt_config_func_name == "my_load_config"

            builder, network = create_network()
            with builder, network, arg_group.create_config(builder, network) as config:
                assert isinstance(config, trt.IBuilderConfig)
                assert config.get_flag(trt.BuilderFlag.FP16)

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
