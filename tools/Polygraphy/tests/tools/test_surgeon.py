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
import tempfile

from tests.models.meta import ONNX_MODELS
from tests.tools.common import run_polygraphy_run, run_polygraphy_surgeon


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


def test_polygraphy_surgeon_extract_sanity_dim_param():
    with tempfile.NamedTemporaryFile() as modelpath:
        run_polygraphy_surgeon(["extract", ONNX_MODELS["dim_param"].path, "-o", modelpath.name])
        run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])
