from polygraphy.util import misc
from polygraphy.logger import G_LOGGER

from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.common import version, check_file_non_empty

import subprocess as sp
import tensorrt as trt
import tempfile
import pytest
import copy
import glob
import sys
import os


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
BIN_DIR = os.path.join(ROOT_DIR, "bin")
Polygraphy = os.path.join(BIN_DIR, "polygraphy")



def check_subprocess(status):
    assert not status.returncode


def run_subtool(subtool, additional_opts):
    cmd = [sys.executable, Polygraphy, subtool] + additional_opts
    G_LOGGER.info("Running command: {:}".format(" ".join(cmd)))
    check_subprocess(sp.run(cmd))


def run_polygraphy_run(additional_opts=[]):
    run_subtool("run", additional_opts)


def run_polygraphy_inspect(additional_opts=[]):
    run_subtool("inspect", additional_opts)


def run_polygraphy_precision(additional_opts=[]):
    run_subtool("precision", additional_opts)


def run_polygraphy_surgeon(additional_opts=[]):
    run_subtool("surgeon", additional_opts)


#
# INSPECT MODEL
#

@pytest.fixture(scope="module", params=[None, "basic", "full"])
def run_inspect_model(request):
    flags = ["--layer-info={:}".format(request.param)] if request.param else []
    yield lambda additional_opts: run_polygraphy_inspect(["model"] + flags + additional_opts)


def test_polygraphy_inspect_model_trt_sanity(run_inspect_model):
    run_inspect_model([ONNX_MODELS["identity"].path, "--display-as=trt"])


def test_polygraphy_inspect_model_trt_engine_sanity(run_inspect_model):
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name])
        run_inspect_model([outpath.name, "--model-type=engine"])


def test_polygraphy_inspect_model_onnx_sanity(run_inspect_model):
    run_inspect_model([ONNX_MODELS["identity"].path])


def test_polygraphy_inspect_model_onnx_subgraphs(run_inspect_model):
    run_inspect_model([ONNX_MODELS["scan"].path])


def test_polygraphy_inspect_model_tf_sanity(run_inspect_model):
    run_inspect_model([TF_MODELS["identity"].path, "--model-type=frozen"])


#
# INSPECT RESULTS
#

def test_polygraphy_inspect_results_sanity():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name])
        run_polygraphy_inspect(["results", outpath.name])


#
# PRECISION
#

@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_polygraphy_precision_bisect_sanity():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name])
        run_polygraphy_precision(["bisect", ONNX_MODELS["identity"].path, "--golden", outpath.name, "--int8"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_polygraphy_precision_linear_sanity():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name])
        run_polygraphy_precision(["linear", ONNX_MODELS["identity"].path, "--golden", outpath.name, "--int8"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_polygraphy_precision_worst_first_sanity():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name, "--onnx-outputs", "mark", "all"])
        run_polygraphy_precision(["worst-first", ONNX_MODELS["identity"].path, "--golden", outpath.name, "--int8", "--trt-outputs", "mark", "all"])


#
# SURGEON
#

def test_polygraphy_surgeon_sanity():
    with tempfile.NamedTemporaryFile() as configpath, tempfile.NamedTemporaryFile() as modelpath:
        run_polygraphy_surgeon(["prepare", ONNX_MODELS["identity"].path, "-o", configpath.name])
        run_polygraphy_surgeon(["operate", ONNX_MODELS["identity"].path, "-c", configpath.name, "-o", modelpath.name])
        run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


def test_polygraphy_surgeon_extract_sanity():
    with tempfile.NamedTemporaryFile() as modelpath:
        run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs", "identity_out_0,auto,auto"])
        run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


def test_polygraphy_surgeon_extract_fallback_shape_inference():
    with tempfile.NamedTemporaryFile() as modelpath:
        # Force fallback shape inference by disabling ONNX shape inference
        run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs",
                             "identity_out_0,auto,auto", "--outputs", "identity_out_2,auto", "--no-shape-inference"])
        run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


#
# RUN
#

def test_polygraphy_run_gen_script():
    with tempfile.NamedTemporaryFile(mode="w") as f:
        run_polygraphy_run(["--gen-script={:}".format(f.name), ONNX_MODELS["identity"].path])
        with open(f.name, "r") as script:
            print(script.read())
        env = copy.deepcopy(os.environ)
        env.update({"PYTHONPATH": ROOT_DIR})
        check_subprocess(sp.run([sys.executable, f.name], env=env))


def test_logger_verbosity():
    run_polygraphy_run(["--silent"])


def test_trt_legacy_uff():
    run_polygraphy_run([TF_MODELS["identity"].path, "--trt-legacy"])


@pytest.mark.skipif(version(trt.__version__) >= version("7.0"), reason="Unsupported in TRT 7.0 and later")
def test_trt_legacy_onnx():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt-legacy"])


def test_trt():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt"])


def test_trt_plugins():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--plugins", "libnvinfer_plugin.so"])


def test_trt_custom_outputs():
    run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--trt", "--trt-outputs", "identity_out_0"])


def test_trt_layerwise_outputs():
    with tempfile.NamedTemporaryFile() as outfile0:
        run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--trt", "--trt-outputs", "mark", "all", "--save-results", outfile0.name])
        results = misc.pickle_load(outfile0.name)
        [result] = list(results.values())[0]
        assert len(result) == 2


@pytest.mark.skipif(version(trt.__version__) < version("7.1.0.0"), reason="API was added in TRT 7.1")
def test_trt_onnx_ext():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--ext"])


def test_trt_int8():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--int8"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_trt_input_shape():
    run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--inputs", "X,1x2x4x4"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_trt_dynamic_input_shape():
    run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--inputs", "X,1x2x-1x4"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_trt_explicit_profile():
    run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--inputs", "X,1x2x1x1", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x1x1", "--trt-max-shapes", "X,1x2x1x1"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_trt_explicit_profile_implicit_runtime_shape():
    run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x1x1", "--trt-max-shapes", "X,1x2x1x1"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_trt_explicit_profile_opt_runtime_shapes_differ():
    run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--inputs", "X,1x2x2x2", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x3x3", "--trt-max-shapes", "X,1x2x4x4"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_trt_multiple_profiles():
    run_polygraphy_run([ONNX_MODELS["dynamic_identity"].path, "--trt", "--onnxrt", "--trt-min-shapes", "X,1x2x1x1", "--trt-opt-shapes", "X,1x2x1x1", "--trt-max-shapes", "X,1x2x1x1", "--trt-min-shapes", "X,1x2x4x4", "--trt-opt-shapes", "X,1x2x4x4", "--trt-max-shapes", "X,1x2x4x4"])


def test_trt_int8_calibration_cache():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--int8", "--calibration-cache", outpath.name])
        check_file_non_empty(outpath.name)


def test_trt_save_load_engine():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name])
        check_file_non_empty(outpath.name)
        run_polygraphy_run(["--trt", outpath.name, "--model-type=engine"])


def test_tf():
    run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5"])


def test_tf_save_pb():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-pb", outpath.name])
        check_file_non_empty(outpath.name)


def test_tf_save_tensorboard():
    with tempfile.TemporaryDirectory() as outdir:
        run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-tensorboard", outdir])
        files = glob.glob("{:}{:}*".format(outdir, os.path.sep))
        assert len(files) == 1


@pytest.mark.skip(reason="Non-trivial to set up - requires CUPTI")
def test_tf_save_timeline():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--gpu-memory-fraction=0.5", "--save-timeline", outpath.name])
        timelines = glob.glob(misc.insert_suffix(outpath.name, "*"))
        for timeline in timelines:
            check_file_non_empty(timeline)


@pytest.mark.skip(reason="Non-trivial to set up")
def test_tftrt():
    run_polygraphy_run([TF_MODELS["identity"].path, "--tf", "--tftrt"])


def test_tf2onnxrt():
    run_polygraphy_run([TF_MODELS["identity"].path, "--onnxrt", "--model-type=frozen"])


def test_tf2onnx_save_onnx():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([TF_MODELS["identity"].path, "--onnxrt", "--model-type=frozen", "--save-onnx", outpath.name])
        check_file_non_empty(outpath.name)
        import onnx
        assert onnx.load(outpath.name)


def test_onnx_tf():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxtf"])


def test_onnx_rt():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt"])


def test_onnx_rt_custom_outputs():
    run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--onnxrt", "--onnx-outputs", "identity_out_0"])


def test_onnx_rt_layerwise_outputs():
    with tempfile.NamedTemporaryFile() as outfile0:
        run_polygraphy_run([ONNX_MODELS["identity_identity"].path, "--onnxrt", "--onnx-outputs", "mark", "all", "--save-results", outfile0.name])
        results = misc.pickle_load(outfile0.name)
        [result] = list(results.values())[0]
        assert len(result) == 2


def test_0_iterations():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--iterations=0"])


def test_custom_tolerance():
    run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--iterations=0", "--atol=1.0", "--rtol=1.0"])


def test_save_load_outputs():
    with tempfile.NamedTemporaryFile() as outfile0, tempfile.NamedTemporaryFile() as outfile1:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outfile0.name])
        run_polygraphy_run(["--load-results", outfile0.name, "--save-results", outfile1.name]) # Copy
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--load-results", outfile0.name, outfile1.name])
        # Should work even with no runners specified
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--load-results", outfile0.name, outfile1.name])
        # Should work with only one file
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--load-results", outfile0.name])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_runner_coexistence():
    run_polygraphy_run([TF_MODELS["identity"].path, "--model-type=frozen", "--tf", "--onnxrt", "--trt"])
