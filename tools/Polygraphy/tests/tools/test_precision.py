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
from tests.tools.common import run_subtool, run_polygraphy_precision, run_polygraphy_run


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
