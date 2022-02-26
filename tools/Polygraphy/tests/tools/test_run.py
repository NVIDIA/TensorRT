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
import copy
import glob
import os
import subprocess as sp
import sys
import tempfile
from textwrap import dedent

import onnx
import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.json import load_json
from tests.helper import get_file_size, is_file_non_empty
from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.tools.common import ROOT_DIR, check_subprocess, run_polygraphy_run


class TestGen(object):
    def test_polygraphy_run_gen_script(self):
        with util.NamedTemporaryFile(mode="w") as f:
            run_polygraphy_run(["--gen-script={:}".format(f.name), ONNX_MODELS["identity"].path])
            with open(f.name, "r") as script:
                print(script.read())
            env = copy.deepcopy(os.environ)
            env.update({"PYTHONPATH": ROOT_DIR})
            check_subprocess(sp.run([sys.executable, f.name], env=env))


class TestLogging(object):
    def test_logger_verbosity(self):
        run_polygraphy_run(["--silent"])

    @pytest.mark.parametrize(
        "log_path",
        [
            os.path.join("example", "example.log"),
            "example.log",
        ],
    )
    def test_log_file(self, log_path):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_run(["--log-file", log_path], cwd=outdir)
            assert open(os.path.join(outdir, log_path)).read()


class TestTrtLegacy(object):
    def test_uff(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--trt-legacy"])

    @pytest.mark.skipif(mod.version(trt.__version__) >= mod.version("7.0"), reason="Unsupported in TRT 7.0 and later")
    def test_onnx(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt-legacy"])


class TestTrt(object):
    def test_basic(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt"])

    def test_plugins(self):
        run_polygraphy_run(
            [
                ONNX_MODELS["identity"].path,
                "--trt",
                "--plugins",
                "nvinfer_plugin.dll" if sys.platform.startswith("win") else "libnvinfer_plugin.so",
            ]
        )

    def test_custom_outputs(self):
        run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--trt", "--trt-outputs", "identity_out_0"])

    def test_layerwise_outputs(self):
        with util.NamedTemporaryFile() as outfile0:
            run_polygraphy_run(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--trt",
                    "--trt-outputs",
                    "mark",
                    "all",
                    "--save-outputs",
                    outfile0.name,
                ]
            )
            results = load_json(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 2
            assert "identity_out_0" in result
            assert "identity_out_2" in result

    def test_exclude_outputs_with_layerwise(self):
        with util.NamedTemporaryFile() as outfile0:
            run_polygraphy_run(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--trt",
                    "--trt-outputs",
                    "mark",
                    "all",
                    "--trt-exclude-outputs",
                    "identity_out_2",
                    "--save-outputs",
                    outfile0.name,
                ]
            )
            results = load_json(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 1
            assert "identity_out_0" in result

    def test_int8(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--int8"])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="API was added after TRT 7.2")
    def test_sparse_weights(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--sparse-weights"])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_input_shape(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--input-shapes", "X:[1,2,4,4]"])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_dynamic_input_shape(self):
        run_polygraphy_run(
            [ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--input-shapes", "X:[1,2,-1,4]"]
        )

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_dynamic_input_shape(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--input-shapes", "X,1x2x-1x4"])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_explicit_profile(self):
        run_polygraphy_run(
            [
                ONNX_MODELS["dynamic_identity"].path,
                "--trt",
                "--onnxrt",
                "--input-shapes",
                "X:[1,2,1,1]",
                "--trt-min-shapes",
                "X:[1,2,1,1]",
                "--trt-opt-shapes",
                "X:[1,2,1,1]",
                "--trt-max-shapes",
                "X:[1,2,1,1]",
            ]
        )

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_explicit_profile_implicit_runtime_shape(self):
        run_polygraphy_run(
            [
                ONNX_MODELS["dynamic_identity"].path,
                "--trt",
                "--onnxrt",
                "--trt-min-shapes",
                "X:[1,2,1,1]",
                "--trt-opt-shapes",
                "X:[1,2,1,1]",
                "--trt-max-shapes",
                "X:[1,2,1,1]",
            ]
        )

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_explicit_profile_opt_runtime_shapes_differ(self):
        run_polygraphy_run(
            [
                ONNX_MODELS["dynamic_identity"].path,
                "--trt",
                "--onnxrt",
                "--input-shapes",
                "X:[1,2,2,2]",
                "--trt-min-shapes",
                "X:[1,2,1,1]",
                "--trt-opt-shapes",
                "X:[1,2,3,3]",
                "--trt-max-shapes",
                "X:[1,2,4,4]",
            ]
        )

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_multiple_profiles(self):
        run_polygraphy_run(
            [
                ONNX_MODELS["dynamic_identity"].path,
                "--trt",
                "--onnxrt",
                "--trt-min-shapes",
                "X:[1,2,1,1]",
                "--trt-opt-shapes",
                "X:[1,2,1,1]",
                "--trt-max-shapes",
                "X:[1,2,1,1]",
                "--trt-min-shapes",
                "X:[1,2,4,4]",
                "--trt-opt-shapes",
                "X:[1,2,4,4]",
                "--trt-max-shapes",
                "X:[1,2,4,4]",
            ]
        )

    def test_int8_calibration_cache(self):
        with util.NamedTemporaryFile() as outpath:
            cmd = [ONNX_MODELS["identity"].path, "--trt", "--int8", "--calibration-cache", outpath.name]
            if mod.version(trt.__version__) >= mod.version("7.0"):
                cmd += ["--onnxrt"]
            run_polygraphy_run(cmd)
            assert is_file_non_empty(outpath.name)

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    @pytest.mark.parametrize("base_class", ["IInt8LegacyCalibrator", "IInt8EntropyCalibrator2"])
    def test_int8_calibration_base_class(self, base_class):
        cmd = [ONNX_MODELS["identity"].path, "--trt", "--int8", "--calibration-base-class", base_class]
        if mod.version(trt.__version__) >= mod.version("7.0"):
            cmd += ["--onnxrt"]
        run_polygraphy_run()

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
    def test_timing_cache(self):
        with tempfile.TemporaryDirectory() as dir:
            # Test with files that haven't already been created instead of using NamedTemporaryFile().
            total_cache = os.path.join(dir, "total.cache")
            identity_cache = os.path.join(dir, "identity.cache")

            run_polygraphy_run([ONNX_MODELS["const_foldable"].path, "--trt", "--timing-cache", total_cache])
            assert is_file_non_empty(total_cache)
            const_foldable_cache_size = get_file_size(total_cache)

            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--timing-cache", identity_cache])
            identity_cache_size = get_file_size(identity_cache)

            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--timing-cache", total_cache])
            total_cache_size = get_file_size(total_cache)

            # The total cache should be larger than either of the individual caches.
            assert total_cache_size > const_foldable_cache_size and total_cache_size > identity_cache_size
            # The total cache should also be smaller than or equal to the sum of the individual caches since
            # header information should not be duplicated.
            assert total_cache_size <= (const_foldable_cache_size + identity_cache_size)

    def test_save_load_engine(self):
        with util.NamedTemporaryFile() as outpath:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name])
            assert is_file_non_empty(outpath.name)
            run_polygraphy_run(["--trt", outpath.name, "--model-type=engine"])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
    def test_tactic_replay(self):
        with util.NamedTemporaryFile() as tactic_replay:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-tactics", tactic_replay.name])
            assert is_file_non_empty(tactic_replay.name)
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--load-tactics", tactic_replay.name])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.2"), reason="Unsupported before TRT 7.2")
    def test_tactic_sources(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--tactic-sources", "CUBLAS", "CUBLAS_LT"])

    def test_data_loader_script_calibration(self):
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(
                dedent(
                    """
                    import numpy as np

                    def load_data():
                        for _ in range(5):
                            yield {"x": np.ones((1, 1, 2, 2), dtype=np.float32) * 6.4341}
                    """
                )
            )
            f.flush()

            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--int8", "--data-loader-script", f.name])


class TestTf(object):
    def test_tf(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5"])

    def test_tf_save_pb(self):
        with util.NamedTemporaryFile() as outpath:
            run_polygraphy_run(
                [TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-pb", outpath.name]
            )
            assert is_file_non_empty(outpath.name)

    def test_tf_save_tensorboard(self):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_run(
                [TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-tensorboard", outdir]
            )
            files = glob.glob("{:}{:}*".format(outdir, os.path.sep))
            assert len(files) == 1

    @pytest.mark.skip(reason="Non-trivial to set up - requires CUPTI")
    def test_tf_save_timeline(self):
        with util.NamedTemporaryFile() as outpath:
            run_polygraphy_run(
                [TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-timeline", outpath.name]
            )
            timelines = glob.glob(os.path.join(outpath.name, "*"))
            for timeline in timelines:
                assert is_file_non_empty(timeline)

    @pytest.mark.skip(reason="Non-trivial to set up")
    def test_tftrt(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--tftrt"])


class TestOnnxrt(object):
    def test_tf2onnxrt(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--onnxrt", "--model-type=frozen"])

    def test_tf2onnx_save_onnx(self):
        with util.NamedTemporaryFile() as outpath:
            run_polygraphy_run(
                [TF_MODELS["identity"].path, "--onnxrt", "--model-type=frozen", "--save-onnx", outpath.name]
            )
            assert is_file_non_empty(outpath.name)
            assert onnx.load(outpath.name)

    def test_onnx_rt(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt"])

    def test_onnx_rt_save_onnx(self):
        with util.NamedTemporaryFile() as outpath:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-onnx", outpath.name])
            assert is_file_non_empty(outpath.name)
            assert onnx.load(outpath.name)

    def test_onnx_rt_custom_outputs(self):
        run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--onnxrt", "--onnx-outputs", "identity_out_0"])

    def test_onnx_rt_layerwise_outputs(self):
        with util.NamedTemporaryFile() as outfile0:
            run_polygraphy_run(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--onnxrt",
                    "--onnx-outputs",
                    "mark",
                    "all",
                    "--save-outputs",
                    outfile0.name,
                ]
            )
            results = load_json(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 2
            assert "identity_out_0" in result
            assert "identity_out_2" in result

    def test_onnx_rt_exclude_outputs_with_layerwise(self):
        with util.NamedTemporaryFile() as outfile0:
            run_polygraphy_run(
                [
                    ONNX_MODELS["identity_identity"].path,
                    "--onnxrt",
                    "--onnx-outputs",
                    "mark",
                    "all",
                    "--onnx-exclude-outputs",
                    "identity_out_2",
                    "--save-outputs",
                    outfile0.name,
                ]
            )
            results = load_json(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 1
            assert "identity_out_0" in result

    def test_external_data(self):
        model = ONNX_MODELS["ext_weights"]
        assert run_polygraphy_run([model.path, "--onnxrt", "--external-data-dir", model.ext_data])


class TestOther(object):
    def test_0_iterations(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--iterations=0"])

    def test_subprocess_sanity(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--use-subprocess"])

    def test_custom_tolerance(self):
        run_polygraphy_run(
            [ONNX_MODELS["identity"].path, "--onnxrt", "--onnxrt", "--iterations=0", "--atol=1.0", "--rtol=1.0"]
        )

    def test_custom_per_output_tolerance(self):
        run_polygraphy_run(
            [
                ONNX_MODELS["identity_identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--onnx-outputs",
                "mark",
                "all",
                "--atol",
                "identity_out_0:1.0",
                "identity_out_2:3.0",
                "0.5",
                "--rtol",
                "identity_out_0:1.0",
                "identity_out_2:3.0",
                "0.5",
            ]
        )

    def test_custom_input_ranges(self):
        run_polygraphy_run(
            [ONNX_MODELS["identity_identity"].path, "--onnxrt", "--val-range", "X:[1.0,2.0]", "[0.5,1.5]"]
        )

    def test_top_k(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--top-k=5"])

    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean"])
    def test_check_error_stat(self, check_error_stat):
        run_polygraphy_run(
            [ONNX_MODELS["identity"].path, "--onnxrt", "--onnxrt", "--check-error-stat", check_error_stat]
        )

    def test_save_load_outputs(self, tmp_path):
        OUTFILE0 = os.path.join(tmp_path, "outputs0.json")
        OUTFILE1 = os.path.join(tmp_path, "outputs1.json")
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", OUTFILE0])
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", OUTFILE1])

        status = run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--load-outputs", OUTFILE0, OUTFILE1])
        assert (
            "Difference is within tolerance" in status.stdout + status.stderr
        )  # Make sure it actually compared stuff.

        # Should work with only one file
        status = run_polygraphy_run([ONNX_MODELS["identity"].path, "--load-outputs", OUTFILE0])
        assert (
            "Difference is within tolerance" not in status.stdout + status.stderr
        )  # Make sure it DIDN'T compare stuff.

        # Should work even with no runners specified
        status = run_polygraphy_run([ONNX_MODELS["identity"].path, "--load-outputs", OUTFILE0, OUTFILE1])
        assert (
            "Difference is within tolerance" in status.stdout + status.stderr
        )  # Make sure it actually compared stuff.

        # Should work even when comparing a single runner to itself.
        status = run_polygraphy_run([ONNX_MODELS["identity"].path, "--load-outputs", OUTFILE0, OUTFILE0])
        assert (
            "Difference is within tolerance" in status.stdout + status.stderr
        )  # Make sure it actually compared stuff.

    def test_save_load_inputs(self):
        with util.NamedTemporaryFile() as infile0, util.NamedTemporaryFile() as infile1:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-input-data", infile0.name])
            run_polygraphy_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--load-input-data",
                    infile0.name,
                    "--save-input-data",
                    infile1.name,
                ]
            )  # Copy
            run_polygraphy_run(
                [ONNX_MODELS["identity"].path, "--onnxrt", "--load-input-data", infile0.name, infile1.name]
            )

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_runner_coexistence(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--model-type=frozen", "--tf", "--onnxrt", "--trt"])


class TestPluginRef(object):
    def test_basic(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--pluginref"])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    @pytest.mark.parametrize("model", ["identity", "instancenorm"])
    def test_ref_implementations(self, model):
        run_polygraphy_run([ONNX_MODELS[model].path, "--pluginref", "--onnxrt", "--trt"])
