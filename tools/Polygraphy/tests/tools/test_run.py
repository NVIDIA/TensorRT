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
import copy
import glob
import os
import subprocess as sp
import sys
import tempfile

import pytest
from polygraphy.logger import G_LOGGER
from polygraphy.util import misc

import tensorrt as trt
from tests.common import check_file_non_empty, version
from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.tools.common import (ROOT_DIR, check_subprocess, run_polygraphy_run,
                                run_subtool)


class TestGen(object):
    def test_polygraphy_run_gen_script(self):
        with tempfile.NamedTemporaryFile(mode="w") as f:
            run_polygraphy_run(["--gen-script={:}".format(f.name), ONNX_MODELS["identity"].path])
            with open(f.name, "r") as script:
                print(script.read())
            env = copy.deepcopy(os.environ)
            env.update({"PYTHONPATH": ROOT_DIR})
            check_subprocess(sp.run([sys.executable, f.name], env=env))


class TestLogging(object):
    def test_logger_verbosity(self):
        run_polygraphy_run(["--silent"])


class TestTrtLegacy(object):
    def test_trt_legacy_uff(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--trt-legacy"])


    @pytest.mark.skipif(version(trt.__version__) >= version("7.0"), reason="Unsupported in TRT 7.0 and later")
    def test_trt_legacy_onnx(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt-legacy"])


class TestTrt(object):
    def test_trt(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt"])


    def test_trt_plugins(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--plugins", "libnvinfer_plugin.so"])


    def test_trt_custom_outputs(self):
        run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--trt", "--trt-outputs", "identity_out_0"])


    def test_trt_layerwise_outputs(self):
        with tempfile.NamedTemporaryFile() as outfile0:
            run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--trt", "--trt-outputs", "mark", "all", "--save-results", outfile0.name])
            results = misc.pickle_load(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 2
            assert "identity_out_0" in result
            assert "identity_out_2" in result


    def test_trt_exclude_outputs_with_layerwise(self):
        with tempfile.NamedTemporaryFile() as outfile0:
            run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--trt", "--trt-outputs", "mark", "all", "--trt-exclude-outputs", "identity_out_2", "--save-results", outfile0.name])
            results = misc.pickle_load(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 1
            assert "identity_out_0" in result


    @pytest.mark.skipif(version(trt.__version__) < version("7.1.0.0"), reason="API was added in TRT 7.1")
    def test_trt_onnx_ext(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--ext"])


    def test_trt_int8(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--int8"])


    @pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
    def test_trt_input_shape(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--input-shapes", "X,1x2x4x4"])


    @pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
    def test_trt_dynamic_input_shape(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--input-shapes", "X,1x2x-1x4"])


    @pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
    def test_trt_explicit_profile(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--input-shapes", "X,1x2x1x1", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x1x1", "--trt-max-shapes", "X,1x2x1x1"])


    @pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
    def test_trt_explicit_profile_implicit_runtime_shape(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x1x1", "--trt-max-shapes", "X,1x2x1x1"])


    @pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
    def test_trt_explicit_profile_opt_runtime_shapes_differ(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--input-shapes", "X,1x2x2x2", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x3x3", "--trt-max-shapes", "X,1x2x4x4"])


    @pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
    def test_trt_multiple_profiles(self):
        run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x1x1", "--trt-max-shapes", "X,1x2x1x1", "--trt-min-shapes", "X,1x2x4x4", "--trt-opt-shapes", "X,1x2x4x4", "--trt-max-shapes", "X,1x2x4x4"])


    def test_trt_int8_calibration_cache(self):
        with tempfile.NamedTemporaryFile() as outpath:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--int8", "--calibration-cache", outpath.name])
            check_file_non_empty(outpath.name)


    def test_trt_save_load_engine(self):
        with tempfile.NamedTemporaryFile() as outpath:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name])
            check_file_non_empty(outpath.name)
            run_polygraphy_run(["--trt", outpath.name, "--model-type=engine"])


class TestTf(object):
    def test_tf(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5"])


    def test_tf_save_pb(self):
        with tempfile.NamedTemporaryFile() as outpath:
            run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-pb", outpath.name])
            check_file_non_empty(outpath.name)


    def test_tf_save_tensorboard(self):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-tensorboard", outdir])
            files = glob.glob("{:}{:}*".format(outdir, os.path.sep))
            assert len(files) == 1


    @pytest.mark.skip(reason="Non-trivial to set up - requires CUPTI")
    def test_tf_save_timeline(self):
        with tempfile.NamedTemporaryFile() as outpath:
            run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-timeline", outpath.name])
            timelines = glob.glob(misc.insert_suffix(outpath.name, "*"))
            for timeline in timelines:
                check_file_non_empty(timeline)


    @pytest.mark.skip(reason="Non-trivial to set up")
    def test_tftrt(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--tftrt"])


class TestOnnxTf(object):
    def test_onnx_tf(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxtf"])


class TestOnnxrt(object):
    def test_tf2onnxrt(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--onnxrt", "--model-type=frozen"])


    def test_tf2onnx_save_onnx(self):
        with tempfile.NamedTemporaryFile() as outpath:
            run_polygraphy_run([TF_MODELS["identity"].path, "--onnxrt", "--model-type=frozen", "--save-onnx", outpath.name])
            check_file_non_empty(outpath.name)
            import onnx
            assert onnx.load(outpath.name)


    def test_onnx_rt(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt"])


    def test_onnx_rt_custom_outputs(self):
        run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--onnxrt", "--onnx-outputs", "identity_out_0"])


    def test_onnx_rt_layerwise_outputs(self):
        with tempfile.NamedTemporaryFile() as outfile0:
            run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--onnxrt", "--onnx-outputs", "mark", "all", "--save-results", outfile0.name])
            results = misc.pickle_load(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 2
            assert "identity_out_0" in result
            assert "identity_out_2" in result


    def test_onnx_rt_exclude_outputs_with_layerwise(self):
        with tempfile.NamedTemporaryFile() as outfile0:
            run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--onnxrt", "--onnx-outputs", "mark", "all", "--onnx-exclude-outputs", "identity_out_2", "--save-results", outfile0.name])
            results = misc.pickle_load(outfile0.name)
            [result] = list(results.values())[0]
            assert len(result) == 1
            assert "identity_out_0" in result


class TestOther(object):
    def test_0_iterations(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--iterations=0"])


    def test_custom_tolerance(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--iterations=0", "--atol=1.0", "--rtol=1.0"])


    def test_top_k(self):
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--top-k=5"])


    def test_save_load_outputs(self):
        with tempfile.NamedTemporaryFile() as outfile0, tempfile.NamedTemporaryFile() as outfile1:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outfile0.name])
            run_polygraphy_run(["--load-results", outfile0.name, "--save-results", outfile1.name]) # Copy
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--load-results", outfile0.name, outfile1.name])
            # Should work even with no runners specified
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--load-results", outfile0.name, outfile1.name])
            # Should work with only one file
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--load-results", outfile0.name])


    @pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
    def test_runner_coexistence(self):
        run_polygraphy_run([TF_MODELS["identity"].path, "--model-type=frozen", "--tf", "--onnxrt", "--trt"])
