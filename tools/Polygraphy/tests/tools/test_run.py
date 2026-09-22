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
import copy
import glob
import os
import subprocess as sp
import sys
import tempfile
from textwrap import dedent
import tensorrt as trt

import onnx
import pytest
import torch

from polygraphy import mod, util
from polygraphy.json import load_json, save_json
from tests.helper import ROOT_DIR, get_file_size, is_file_non_empty
from tests.models.meta import ONNX_MODELS, TF_MODELS

# INT8 mode and the INT8 calibration APIs were removed in TensorRT 11.
skip_if_trt_11 = pytest.mark.skipif(
    mod.version(trt.__version__) >= mod.version("11.0"),
    reason="INT8 mode and calibration were removed in TRT 11",
)


class TestGen:
    def test_polygraphy_run_gen_script(self, poly_run):
        with util.NamedTemporaryFile(mode="w") as f:
            poly_run([f"--gen-script={f.name}", ONNX_MODELS["identity"].path])
            with open(f.name) as script:
                print(script.read())
            env = copy.deepcopy(os.environ)
            env.update({"PYTHONPATH": ROOT_DIR})
            status = sp.run([sys.executable, f.name], env=env)
            assert status.returncode == 0


class TestLogging:
    def test_logger_verbosity(self, poly_run):
        poly_run(["--silent"])

    @pytest.mark.parametrize(
        "log_path",
        [
            os.path.join("example", "example.log"),
            "example.log",
        ],
    )
    def test_log_file(self, poly_run, log_path):
        with tempfile.TemporaryDirectory() as outdir:
            poly_run(["--log-file", log_path], cwd=outdir)
            assert open(os.path.join(outdir, log_path)).read()


class TestTrt:
    def test_basic(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--trt"])

    def test_plugins(self, poly_run):
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--trt",
                "--plugins",
                (
                    "nvinfer_plugin.dll"
                    if sys.platform.startswith("win")
                    # TRT 11 pip wheels ship only the major-versioned soname.
                    else (
                        f"libnvinfer_plugin.so.{trt.__version__.split('.')[0]}"
                        if mod.version(trt.__version__) >= mod.version("11.0")
                        else "libnvinfer_plugin.so"
                    )
                ),
            ]
        )

    def test_custom_outputs(self, poly_run):
        poly_run(
            [
                ONNX_MODELS["identity_identity"].path,
                "--trt",
                "--trt-outputs",
                "identity_out_0",
            ]
        )

    def test_layerwise_outputs(self, poly_run):
        with util.NamedTemporaryFile(suffix=".json") as outfile0:
            poly_run(
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

    def test_exclude_outputs_with_layerwise(self, poly_run):
        with util.NamedTemporaryFile(suffix=".json") as outfile0:
            poly_run(
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

    @skip_if_trt_11
    def test_int8(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--trt", "--int8"])

    def test_sparse_weights(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--trt", "--sparse-weights"])

    def test_input_shape(self, poly_run):
        poly_run(
            [
                ONNX_MODELS["dynamic_identity"].path,
                "--trt",
                "--onnxrt",
                "--input-shapes",
                "X:[1,2,4,4]",
            ]
        )

    def test_dynamic_input_shape(self, poly_run):
        poly_run(
            [
                ONNX_MODELS["dynamic_identity"].path,
                "--trt",
                "--onnxrt",
                "--input-shapes",
                "X:[1,2,-1,4]",
            ]
        )

    def test_explicit_profile(self, poly_run):
        poly_run(
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

    def test_explicit_profile_implicit_runtime_shape(self, poly_run):
        poly_run(
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

    def test_explicit_profile_opt_runtime_shapes_differ(self, poly_run):
        poly_run(
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

    @pytest.mark.parametrize("optimization_profile", [None, 0, 1])
    def test_multiple_profiles(self, poly_run, optimization_profile):
        cmd = [
            ONNX_MODELS["dynamic_identity"].path,
            "--trt",
            "--onnxrt",
            # Profile 0
            "--trt-min-shapes",
            "X:[1,2,1,1]",
            "--trt-opt-shapes",
            "X:[1,2,1,1]",
            "--trt-max-shapes",
            "X:[1,2,1,1]",
            # Profile 1
            "--trt-min-shapes",
            "X:[1,2,4,4]",
            "--trt-opt-shapes",
            "X:[1,2,4,4]",
            "--trt-max-shapes",
            "X:[1,2,4,4]",
            # Input shapes
            "--input-shapes",
            "X:[1,2,4,4]" if optimization_profile == 1 else "X:[1,2,1,1]",
        ]
        if optimization_profile is not None:
            cmd += [f"--optimization-profile={optimization_profile}"]

        poly_run(cmd)

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("10.0"),
        reason="Feature not present before 10.0",
    )
    @pytest.mark.parametrize(
        "allocation_strategy", [None, "static", "profile", "runtime"]
    )
    def test_allocation_strategies(self, poly_run, allocation_strategy):
        cmd = [
            ONNX_MODELS["residual_block"].path,
            "--trt",
            "--onnxrt",
            # Profile 0
            "--trt-min-shapes",
            "gpu_0/data_0:[1,3,224,224]",
            "--trt-opt-shapes",
            "gpu_0/data_0:[1,3,224,224]",
            "--trt-max-shapes",
            "gpu_0/data_0:[2,3,224,224]",
            # Profile 1
            "--trt-min-shapes",
            "gpu_0/data_0:[1,3,224,224]",
            "--trt-opt-shapes",
            "gpu_0/data_0:[1,3,224,224]",
            "--trt-max-shapes",
            "gpu_0/data_0:[4,3,224,224]",
            # Input shapes
            "--input-shapes",
            "gpu_0/data_0:[2,3,224,224]",
            "--optimization-profile",
            "1",
        ]
        if allocation_strategy is not None:
            cmd += ["--allocation-strategy", allocation_strategy]

        poly_run(cmd)

    @skip_if_trt_11
    def test_int8_calibration_cache(self, poly_run):
        with util.NamedTemporaryFile() as outpath:
            cmd = [
                ONNX_MODELS["identity"].path,
                "--trt",
                "--int8",
                "--calibration-cache",
                outpath.name,
            ]
            cmd += ["--onnxrt"]
            poly_run(cmd)
            assert is_file_non_empty(outpath.name)

    @skip_if_trt_11
    @pytest.mark.parametrize(
        "base_class", ["IInt8LegacyCalibrator", "IInt8EntropyCalibrator2"]
    )
    def test_int8_calibration_base_class(self, poly_run, base_class):
        cmd = [
            ONNX_MODELS["identity"].path,
            "--trt",
            "--int8",
            "--calibration-base-class",
            base_class,
        ]
        cmd += ["--onnxrt"]
        poly_run()

    def test_timing_cache(self, poly_run):
        with tempfile.TemporaryDirectory() as dir:
            # Test with files that haven't already been created instead of using NamedTemporaryFile().
            total_cache = os.path.join(dir, "total.cache")
            identity_cache = os.path.join(dir, "identity.cache")

            poly_run(
                [
                    ONNX_MODELS["const_foldable"].path,
                    "--trt",
                    "--save-timing-cache",
                    total_cache,
                ]
            )
            assert is_file_non_empty(total_cache)
            const_foldable_cache_size = get_file_size(total_cache)

            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--trt",
                    "--save-timing-cache",
                    identity_cache,
                ]
            )
            identity_cache_size = get_file_size(identity_cache)

            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--trt",
                    "--save-timing-cache",
                    total_cache,
                ]
            )
            total_cache_size = get_file_size(total_cache)

            # The total cache should be larger than either of the individual caches.
            assert (
                total_cache_size >= const_foldable_cache_size
                and total_cache_size >= identity_cache_size
            )
            # The total cache should also be smaller than or equal to the sum of the individual caches since
            # header information should not be duplicated.
            assert total_cache_size <= (const_foldable_cache_size + identity_cache_size)

    def test_save_load_engine(self, poly_run):
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name]
            )
            assert is_file_non_empty(outpath.name)
            poly_run(["--trt", outpath.name, "--model-type=engine"])

    @pytest.mark.skipif(
        mod.version(trt.__version__) >= mod.version("11.0"),
        reason="--save/--load-tactics rely on the algorithm selector API, removed in TRT 11",
    )
    def test_tactic_replay(self, poly_run):
        with util.NamedTemporaryFile() as tactic_replay:
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--trt",
                    "--save-tactics",
                    tactic_replay.name,
                ]
            )
            assert is_file_non_empty(tactic_replay.name)
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--trt",
                    "--load-tactics",
                    tactic_replay.name,
                ]
            )

    def test_tactic_sources(self, poly_run):
        # EDGE_MASK_CONVOLUTIONS / JIT_CONVOLUTIONS are the tactic sources that
        # survive on every supported TRT version (CUBLAS/CUBLAS_LT/CUDNN were
        # removed in TRT 11).
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--trt",
                "--tactic-sources",
                "EDGE_MASK_CONVOLUTIONS",
                "JIT_CONVOLUTIONS",
            ]
        )

    def test_pool_limits(self, poly_run):
        poly_run(
            [ONNX_MODELS["identity"].path, "--trt", "--pool-limit", "workspace:32M"]
        )

    @skip_if_trt_11
    def test_data_loader_script_calibration(self, poly_run):
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
            os.fsync(f.fileno())

            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--trt",
                    "--int8",
                    "--data-loader-script",
                    f.name,
                ]
            )


class TestTf:
    def test_tf(self, poly_run):
        pytest.importorskip("tensorflow")
        poly_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5"])

    def test_tf_save_pb(self, poly_run):
        pytest.importorskip("tensorflow")
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [
                    TF_MODELS["identity"].path,
                    "--tf",
                    "--gpu-memory-fraction=0.5",
                    "--save-pb",
                    outpath.name,
                ]
            )
            assert is_file_non_empty(outpath.name)

    def test_tf_save_tensorboard(self, poly_run):
        pytest.importorskip("tensorflow")
        with tempfile.TemporaryDirectory() as outdir:
            poly_run(
                [
                    TF_MODELS["identity"].path,
                    "--tf",
                    "--gpu-memory-fraction=0.5",
                    "--save-tensorboard",
                    outdir,
                ]
            )
            files = glob.glob(f"{outdir}{os.path.sep}*")
            assert len(files) == 1

    @pytest.mark.skip(reason="Non-trivial to set up - requires CUPTI")
    def test_tf_save_timeline(self, poly_run):
        pytest.importorskip("tensorflow")
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [
                    TF_MODELS["identity"].path,
                    "--tf",
                    "--gpu-memory-fraction=0.5",
                    "--save-timeline",
                    outpath.name,
                ]
            )
            timelines = glob.glob(os.path.join(outpath.name, "*"))
            for timeline in timelines:
                assert is_file_non_empty(timeline)

    @pytest.mark.skip(reason="Non-trivial to set up")
    def test_tftrt(self, poly_run):
        pytest.importorskip("tensorflow")
        poly_run([TF_MODELS["identity"].path, "--tf", "--tftrt"])


class TestOnnxrt:
    def test_onnx_rt(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt"])

    def test_onnx_rt_save_onnx(self, poly_run):
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [ONNX_MODELS["identity"].path, "--onnxrt", "--save-onnx", outpath.name]
            )
            assert is_file_non_empty(outpath.name)
            assert onnx.load(outpath.name)

    def test_onnx_rt_custom_outputs(self, poly_run):
        poly_run(
            [
                ONNX_MODELS["identity_identity"].path,
                "--onnxrt",
                "--onnx-outputs",
                "identity_out_0",
            ]
        )

    def test_onnx_rt_layerwise_outputs(self, poly_run):
        with util.NamedTemporaryFile(suffix=".json") as outfile0:
            poly_run(
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

    def test_onnx_rt_exclude_outputs_with_layerwise(self, poly_run):
        with util.NamedTemporaryFile(suffix=".json") as outfile0:
            poly_run(
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

    def test_external_data(self, poly_run):
        model = ONNX_MODELS["ext_weights"]
        assert poly_run([model.path, "--onnxrt", "--external-data-dir", model.ext_data])

    def test_providers(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--providers", "cpu"])


class TestOther:
    def test_0_iterations(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--iterations=0"])

    def test_subprocess_sanity(self, poly_run):
        # --use-subprocess requires --sequential-runners (streaming runs in one process).
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--use-subprocess",
                "--sequential-runners",
            ]
        )

    @pytest.mark.parametrize(
        "save_name",
        # --load-outputs reads back either a single file or a per-iteration directory (default).
        ["outputs0.json", "golden"],
    )
    def test_exit_status_on_fail_comparison(self, poly_run, tmp_path, save_name):
        # An accuracy mismatch against a saved reference run must yield a non-zero exit status.
        outputs = os.path.join(tmp_path, save_name)
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--save-outputs",
                outputs,
                "--seed=1",
            ]
        )
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--load-outputs",
                outputs,
                "--seed=2",
            ],
            expect_error=True,
        )

    def test_custom_tolerance(self, poly_run):
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--iterations=0",
                "--atol=1.0",
                "--rtol=1.0",
            ]
        )

    def test_custom_per_output_tolerance(self, poly_run):
        poly_run(
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

    def test_custom_input_ranges(self, poly_run):
        poly_run(
            [
                ONNX_MODELS["identity_identity"].path,
                "--onnxrt",
                "--val-range",
                "X:[1.0,2.0]",
                "[0.5,1.5]",
            ]
        )

    def test_index_comparison(self, poly_run):
        # Two identical runners with top-1 postprocessing and indices comparison must match fully.
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--postprocess",
                "top-1",
                "--compare-func=indices",
            ]
        )
        out = status.stdout + status.stderr
        assert "PASSED | Difference is within index tolerance" in out
        assert "Accuracy Summary" in out
        assert "Pass Rate: 100.00%" in out

    def test_postprocess_func_script(self, poly_run):
        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(
                dedent(
                    """
                    def postprocess_outputs(iter_result):
                        raise RuntimeError("postprocess hook invoked")
                    """
                )
            )
            f.flush()
            os.fsync(f.fileno())

            status = poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--postprocess-func-script",
                    f.name,
                ],
                expect_error=True,
            )
            assert "postprocess hook invoked" in status.stderr

    @pytest.mark.parametrize(
        "compare_func",
        ["l2", "cosine_similarity", "psnr", "snr"],
    )
    def test_atomic_compare_funcs(self, poly_run, compare_func):
        # Each single-metric comparison function passes its threshold for two identical runners.
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--compare",
                compare_func,
            ]
        )
        assert "Accuracy Summary" in status.stdout + status.stderr

    @pytest.mark.parametrize("check_error_stat", ["max", "median", "mean", "quantile"])
    def test_check_error_stat(self, poly_run, check_error_stat):
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--check-error-stat",
                check_error_stat,
            ]
        )

    def test_check_average(self, poly_run):
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--check-error-stat",
                "mean",
                "--check-average",
            ]
        )
        assert "Accuracy Summary" in status.stdout + status.stderr
        # Average mode summarizes per-output; per-iteration mode would emit "Pass Rate".
        assert "Pass Rate" not in status.stdout + status.stderr

    def test_check_average_rejects_elemwise(self, poly_run):
        # Regression test: --check-average must be rejected when 'simple' uses the default
        # 'elemwise' stat (which has no scalar to average). The validation reads check_error_stat
        # from another argument group, so it must run after all parsing; exercising the real tool
        # here guards against that ordering bug.
        status = poly_run(
            [ONNX_MODELS["identity"].path, "--onnxrt", "--onnxrt", "--check-average"],
            expect_error=True,
        )
        assert (
            "--check-average is not supported with check_error_stat='elemwise'"
            in status.stdout + status.stderr
        )

    def test_check_average_rejects_indices(self, poly_run):
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--check-average",
                "--compare",
                "indices",
            ],
            expect_error=True,
        )
        assert "does not produce averageable metrics" in status.stdout + status.stderr

    def test_save_load_outputs(self, poly_run, tmp_path):
        OUTFILE0 = os.path.join(tmp_path, "outputs0.json")
        OUTFILE1 = os.path.join(tmp_path, "outputs1.json")
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", OUTFILE0])
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", OUTFILE1])

        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--load-outputs",
                OUTFILE0,
                OUTFILE1,
            ]
        )
        assert (
            "Difference is within tolerance" in status.stdout + status.stderr
        )  # Make sure it actually compared stuff.

        # Should work with only one file
        status = poly_run([ONNX_MODELS["identity"].path, "--load-outputs", OUTFILE0])
        assert (
            "Difference is within tolerance" not in status.stdout + status.stderr
        )  # Make sure it DIDN'T compare stuff.

        # Should work even with no runners specified
        status = poly_run(
            [ONNX_MODELS["identity"].path, "--load-outputs", OUTFILE0, OUTFILE1]
        )
        assert (
            "Difference is within tolerance" in status.stdout + status.stderr
        )  # Make sure it actually compared stuff.

        # Should work even when comparing a single runner to itself.
        status = poly_run(
            [ONNX_MODELS["identity"].path, "--load-outputs", OUTFILE0, OUTFILE0]
        )
        assert (
            "Difference is within tolerance" in status.stdout + status.stderr
        )  # Make sure it actually compared stuff.

    def test_save_accuracy_results_and_recheck(
        self, poly_run, poly_check_accuracy, tmp_path
    ):
        # Save two runs, compare them while saving the accuracy results, then re-check those results
        # against new thresholds with `check accuracy` -- without re-running inference.
        OUTFILE0 = os.path.join(tmp_path, "outputs0.json")
        OUTFILE1 = os.path.join(tmp_path, "outputs1.json")
        ACC = os.path.join(tmp_path, "accuracy.json")
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", OUTFILE0])
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", OUTFILE1])

        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--load-outputs",
                OUTFILE0,
                OUTFILE1,
                "--check-error-stat",
                "max",
                "--save-accuracy-results",
                ACC,
            ]
        )
        assert os.path.exists(ACC)

        # The two runs are identical, so any tolerance passes.
        poly_check_accuracy([ACC, "--check-error-stat", "max", "--atol", "1e-8"])

    def test_save_load_inputs(self, poly_run):
        with util.NamedTemporaryFile(
            suffix=".json"
        ) as infile0, util.NamedTemporaryFile(suffix=".json") as infile1:
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--save-input-data",
                    infile0.name,
                ]
            )
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--load-input-data",
                    infile0.name,
                    "--save-input-data",
                    infile1.name,
                ]
            )  # Copy
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--load-input-data",
                    infile0.name,
                    infile1.name,
                ]
            )

    def test_load_torch_inputs(self, poly_run):
        with util.NamedTemporaryFile() as infile:
            inp = torch.ones((1, 1, 2, 2), dtype=torch.float32)
            feed_dict = [{"x": inp}]
            save_json(feed_dict, infile.name)
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--onnxrt",
                    "--load-inputs",
                    infile.name,
                ]
            )

    def test_runner_coexistence(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--trt"])

    def test_tf2onnxrt(self, poly_run):
        pytest.importorskip("tensorflow")
        poly_run([TF_MODELS["identity"].path, "--onnxrt", "--model-type=frozen"])

    def test_tf2onnx_save_onnx(self, poly_run):
        pytest.importorskip("tensorflow")
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [
                    TF_MODELS["identity"].path,
                    "--onnxrt",
                    "--model-type=frozen",
                    "--save-onnx",
                    outpath.name,
                ]
            )
            assert is_file_non_empty(outpath.name)
            assert onnx.load(outpath.name)


class TestPluginRef:
    def test_basic(self, poly_run):
        poly_run([ONNX_MODELS["identity"].path, "--pluginref"])

    @pytest.mark.parametrize("model", ["identity", "instancenorm"])
    def test_ref_implementations(self, poly_run, model):
        poly_run([ONNX_MODELS[model].path, "--pluginref", "--onnxrt", "--trt"])


class TestStreamData:
    def test_gen_script(self, poly_run):
        with util.NamedTemporaryFile(mode="w") as f:
            poly_run(
                [
                    f"--gen-script={f.name}",
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--onnxrt",
                    "--validate",
                ]
            )
            with open(f.name) as script_file:
                script = script_file.read()
            # Streaming (the default) emits run(..., streaming=True) feeding compare_accuracy.
            assert "streaming=True" in script
            assert "Comparator.compare_accuracy(" in script
            # In streaming mode, --validate wraps the runs in a lazy validation pass-through.
            assert "Comparator.validate(" in script

    def test_postprocess_applies_to_loaded_outputs(self, poly_run, tmp_path):
        # --postprocess must apply to --load-outputs goldens too, not just the live run:
        # top-1 + indices comparison only matches if the loaded golden is also reduced to indices.
        golden = os.path.join(tmp_path, "golden")
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--iterations",
                "3",
                "--save-outputs",
                golden,
            ]
        )
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--load-outputs",
                golden,
                "--postprocess",
                "top-1",
                "--compare-func",
                "indices",
            ]
        )
        assert "Pass Rate: 100.00%" in status.stdout + status.stderr

    @pytest.mark.parametrize("extra_save_args", [[], ["--sequential-runners"]])
    def test_save_writes_per_iteration_directory(
        self, poly_run, tmp_path, extra_save_args
    ):
        # An extensionless --save-* path writes one <i>.json per iteration in both the streaming
        # and --sequential-runners paths (both should produce the same directory layout).
        in_dir = os.path.join(tmp_path, "inputs")
        out_dir = os.path.join(tmp_path, "outputs")
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                *extra_save_args,
                "--iterations",
                "3",
                "--save-inputs",
                in_dir,
                "--save-outputs",
                out_dir,
            ]
        )
        assert sorted(os.listdir(in_dir)) == [
            "0000000000.json",
            "0000000001.json",
            "0000000002.json",
        ]
        assert sorted(os.listdir(out_dir)) == [
            "0000000000.json",
            "0000000001.json",
            "0000000002.json",
        ]

    def test_sequential_runners_load_outputs_rejects_directory(
        self, poly_run, tmp_path
    ):
        # With --sequential-runners, --load-outputs reads a materialized run (RunResults.load), so a
        # per-iteration directory is rejected at runtime. (Streaming accepts it; see round-trip.)
        out_dir = os.path.join(tmp_path, "outputs")
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--iterations",
                "3",
                "--save-outputs",
                out_dir,
            ]
        )
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--sequential-runners",
                "--load-outputs",
                out_dir,
            ],
            expect_error=True,
        )
        assert "directory" in (status.stdout + status.stderr).lower()

    def test_gen_script_does_not_inspect_disk(self, poly_run, tmp_path):
        # --gen-script must not inspect disk state: the files-only contract for a --sequential-runners
        # --load-outputs directory is enforced at runtime, not during script generation.
        existing_dir = os.path.join(tmp_path, "some_dir")
        os.makedirs(existing_dir)
        gen_script = tmp_path / "gen.py"
        poly_run(
            [
                f"--gen-script={gen_script}",
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--sequential-runners",
                "--load-outputs",
                existing_dir,
            ]
        )
        assert gen_script.stat().st_size > 0

    @pytest.mark.parametrize(
        "extra_args, expected",
        [
            (["--use-subprocess"], "--use-subprocess requires --sequential-runners"),
            (["--warm-up", "2"], "--warm-up requires --sequential-runners"),
        ],
        ids=["use-subprocess", "warm-up"],
    )
    def test_streaming_rejects_sequential_only_args(
        self, poly_run, extra_args, expected
    ):
        # These flags are not supported in the default streaming mode; they require
        # --sequential-runners. The guard fires on streaming mode regardless of runner count.
        status = poly_run(
            [ONNX_MODELS["identity"].path, "--onnxrt", *extra_args],
            expect_error=True,
        )
        assert expected in status.stdout + status.stderr

    def test_allows_load_and_save_outputs(self, poly_run, tmp_path):
        # Loading a reference run while saving the live run to a different path is allowed
        # (only overlapping load/save paths are rejected).
        golden = os.path.join(tmp_path, "golden")
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--save-outputs",
                golden,
            ]
        )
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--load-outputs",
                golden,
                "--save-outputs",
                os.path.join(tmp_path, "newsave"),
            ]
        )
        assert "Accuracy Summary" in status.stdout + status.stderr

    @pytest.mark.parametrize("load_is_file_inside", [False, True])
    def test_rejects_load_overlapping_save_path(
        self, poly_run, tmp_path, load_is_file_inside
    ):
        # A load path that is the same directory as a save path -- or a file directly inside it --
        # is rejected, since the per-iteration save directory (which must be empty) would conflict
        # with the data being loaded.
        save_dir = tmp_path / "data"
        if load_is_file_inside:
            save_dir.mkdir()
            (save_dir / "0.json").write_text("{}")
            load_path = str(save_dir / "0.json")
        else:
            load_path = str(save_dir)
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--load-inputs",
                load_path,
                "--save-inputs",
                str(save_dir),
            ],
            expect_error=True,
        )
        assert "overlaps the save path" in status.stdout + status.stderr

    def test_multiple_compare_funcs_distinct_in_script(self, poly_run, tmp_path):
        gen_script = tmp_path / "gen.py"
        poly_run(
            [
                f"--gen-script={gen_script}",
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--compare",
                "simple",
                "distance_metrics",
                "--check-error-stat",
                "mean",
                "--l2-threshold",
                "0.0",
            ]
        )
        script = gen_script.read_text()
        # Each compare function must be its own variable so they don't alias each other.
        # 'distance_metrics' expands into the atomic L2 + cosine-similarity functions.
        assert "simple_compare_func = SimpleCompareFunc(" in script
        assert "l2_compare_func = L2CompareFunc(" in script
        assert "cosine_similarity_compare_func = CosineSimilarityCompareFunc(" in script
        assert (
            "compare_func=[simple_compare_func, l2_compare_func, cosine_similarity_compare_func]"
            in script
        )

    def test_multiple_compare_funcs_all_applied(self, poly_run):
        # distance_metrics with an impossible cosine threshold must fail the run, even though the
        # 'simple' comparison passes -- proving both compare functions actually take effect.
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--onnxrt",
                "--compare",
                "distance_metrics",
                "simple",
                "--cosine-similarity-threshold",
                "2.0",
            ],
            expect_error=True,
        )
        assert "FAILED" in status.stdout + status.stderr

    def test_fail_fast_stops_mid_stream(self, poly_run, tmp_path):
        # A mismatch on the first iteration with --fail-fast must produce a non-zero exit and a
        # FAILED marker without waiting for subsequent iterations to complete.
        golden = os.path.join(tmp_path, "golden")
        poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--save-outputs",
                golden,
                "--seed=1",
            ]
        )
        status = poly_run(
            [
                ONNX_MODELS["identity"].path,
                "--onnxrt",
                "--load-outputs",
                golden,
                "--seed=2",
                "--check-error-stat",
                "mean",
                "--atol=0",
                "--rtol=0",
                "--fail-fast",
            ],
            expect_error=True,
        )
        assert "FAILED" in status.stdout + status.stderr
