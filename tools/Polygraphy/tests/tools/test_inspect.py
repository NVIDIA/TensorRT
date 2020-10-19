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
from textwrap import dedent

import pytest
from polygraphy.logger import G_LOGGER
from polygraphy.util import misc
from tests.common import check_file_non_empty, version
from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.tools.common import (run_polygraphy_inspect, run_polygraphy_run,
                                run_subtool)

#
# INSPECT MODEL
#

@pytest.fixture(scope="module", params=["none", "basic", "attrs", "full"])
def run_inspect_model(request):
    yield lambda additional_opts: run_polygraphy_inspect(["model"] + ["--mode={:}".format(request.param)] + additional_opts)


# ONNX cases
ONNX_CASES = [
    ["identity", "none",
        """
        [I] ==== ONNX Model ====
            Name: test_identity | Opset: 8

            ---- 1 Graph Inputs ----
            {x [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 1 Graph Outputs ----
            {y [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 0 Initializers ----
            (Use --mode to display)

            ---- 1 Nodes ----
            (Use --mode to display)
        """
    ],
    ["identity", "basic",
        """
        [I] ==== ONNX Model ====
            Name: test_identity | Opset: 8

            ---- 1 Graph Inputs ----
            {x [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 1 Graph Outputs ----
            {y [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 0 Initializers ----
            {}

            ---- 1 Nodes ----
            Node 0    |  [Op: Identity]
                {x [dtype=float32, shape=(1, 1, 2, 2)]}
                -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
        """
    ],
    ["identity_with_initializer", "basic",
        """
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Inputs ----
            {}

            ---- 1 Graph Outputs ----
            {Y [dtype=float32, shape=(2, 2)]}

            ---- 1 Initializers ----
            {X [dtype=float32, shape=(2, 2)]}

            ---- 1 Nodes ----
            Node 0    |  [Op: Identity]
                {Initializer | X [dtype=float32, shape=(2, 2)]}
                -> {Y [dtype=float32, shape=(2, 2)]}
        """
    ],
    ["identity_with_initializer", "full",
        """
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Inputs ----
            {}

            ---- 1 Graph Outputs ----
            {Y [dtype=float32, shape=(2, 2)]}

            ---- 1 Initializers ----
            Initializer | X [dtype=float32, shape=[2, 2]] | Values:
                [[1. 1.]
                 [1. 1.]]

            ---- 1 Nodes ----
            Node 0    |  [Op: Identity]
                {Initializer | X [dtype=float32, shape=(2, 2)]}
                -> {Y [dtype=float32, shape=(2, 2)]}
        """
    ],
    ["tensor_attr", "basic",
        """
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Inputs ----
            {}

            ---- 1 Graph Outputs ----
            {const_out [dtype=float32, shape=(14, 14)]}

            ---- 0 Initializers ----
            {}

            ---- 1 Nodes ----
            Node 0    |  [Op: Constant]
                {} -> {const_out [dtype=float32, shape=(14, 14)]}
        """
    ],
    ["tensor_attr", "attrs",
        """
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Inputs ----
            {}

            ---- 1 Graph Outputs ----
            {const_out [dtype=float32, shape=(14, 14)]}

            ---- 0 Initializers ----
            {}

            ---- 1 Nodes ----
            Node 0    |  [Op: Constant]
                {} -> {const_out [dtype=float32, shape=(14, 14)]}
                ---- Attributes ----
                value = Tensor: [dtype=float32, shape=[14, 14]]
        """
    ],
    ["tensor_attr", "full",
        """
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Inputs ----
            {}

            ---- 1 Graph Outputs ----
            {const_out [dtype=float32, shape=(14, 14)]}

            ---- 0 Initializers ----

            ---- 1 Nodes ----
            Node 0    |  [Op: Constant]
                {} -> {const_out [dtype=float32, shape=(14, 14)]}
                ---- Attributes ----
                value = Tensor: [dtype=float32, shape=[14, 14]] | Values:
                    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
                     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
        """
    ],
    ["scan", "full",
        """
        [I] ==== ONNX Model ====
            Name: graph | Opset: 10

            ---- 2 Graph Inputs ----
            {initial [dtype=float32, shape=(2,)], x [dtype=float32, shape=(3, 2)]}

            ---- 2 Graph Outputs ----
            {y [dtype=float32, shape=(2,)], z [dtype=float32, shape=(3, 2)]}

            ---- 0 Initializers ----

            ---- 1 Nodes ----
            Node 0    |  [Op: Scan]
                {initial [dtype=float32, shape=(2,)], x [dtype=float32, shape=(3, 2)]}
                -> {y [dtype=float32, shape=(2,)], z [dtype=float32, shape=(3, 2)]}
                ---- Attributes ----
                body =
                        ---- 2 Subgraph Inputs ----
                        {sum_in [dtype=float32, shape=(2,)], next [dtype=float32, shape=(2,)]}

                        ---- 2 Subgraph Outputs ----
                        {sum_out [dtype=float32, shape=(2,)], scan_out [dtype=float32, shape=(2,)]}

                        ---- 0 Initializers ----

                        ---- 2 Nodes ----
                        Node 0    |  [Op: Add]
                            {sum_in [dtype=float32, shape=(2,)], next [dtype=float32, shape=(2,)]}
                            -> {sum_out [dtype=float32, shape=(2,)]}

                        Node 1    |  [Op: Identity]
                            {sum_out [dtype=float32, shape=(2,)]}
                            -> {scan_out [dtype=float32, shape=(2,)]}

                num_scan_inputs = 1
        """
    ],
]

@pytest.mark.parametrize("case", ONNX_CASES, ids=lambda case: "{:}-{:}".format(case[0], case[1]))
def test_polygraphy_inspect_model_onnx(run_inspect_model, case):
    model, mode, expected = case
    status = run_polygraphy_inspect(["model", ONNX_MODELS[model].path, "--mode={:}".format(mode)], disable_verbose=True)

    expected = dedent(expected).strip()
    actual = status.stdout.decode()

    print("Actual output:\n{:}".format(actual))
    for acline, exline in zip(actual.splitlines(), expected.splitlines()):
        acline = acline.rstrip()
        exline = exline.rstrip()
        print("Checking line : {:}".format(acline))
        print("Expecting line: {:}".format(exline))
        assert acline == exline


@pytest.mark.parametrize("model", ["identity", "scan", "tensor_attr"])
def test_polygraphy_inspect_model_trt_sanity(run_inspect_model, model):
    import tensorrt as trt

    if model == "tensor_attr" and version(trt.__version__) < version("7.2"):
        pytest.skip("Models with constant outputs were not supported before 7.2")

    if model == "scan" and version(trt.__version__) < version("7.0"):
        pytest.skip("Scan was not supported until 7.0")

    run_inspect_model([ONNX_MODELS[model].path, "--display-as=trt"])


def test_polygraphy_inspect_model_trt_engine_sanity(run_inspect_model):
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name])
        run_inspect_model([outpath.name, "--model-type=engine"])


def test_polygraphy_inspect_model_tf_sanity(run_inspect_model):
    run_inspect_model([TF_MODELS["identity"].path, "--model-type=frozen"])


#
# INSPECT RESULTS
#

@pytest.mark.parametrize("opts", [[], ["--show-values"]])
def test_polygraphy_inspect_results(opts):
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name])
        run_polygraphy_inspect(["results", outpath.name] + opts)
