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
import glob
import os
import tempfile
from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import mod, util
from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.tools.common import run_polygraphy_inspect, run_polygraphy_run


@pytest.fixture(scope="session", params=["none", "basic", "attrs", "full"])
def run_inspect_model(request):
    yield lambda additional_opts: run_polygraphy_inspect(
        ["model"] + ["--mode={:}".format(request.param)] + additional_opts
    )


@pytest.fixture(scope="session")
def identity_engine():
    with util.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name])
        yield outpath.name


def check_lines_match(actual, expected, should_check_line=lambda x: True):
    print("Actual output:\n{:}".format(actual))

    actual = [line for line in actual.splitlines() if "Loading" not in line]
    expected = expected.splitlines()
    assert len(actual) == len(expected)

    for acline, exline in zip(actual, expected):
        acline = acline.rstrip()
        exline = exline.rstrip()
        print("Checking line : {:}".format(acline))
        print("Expecting line: {:}".format(exline))
        if should_check_line(exline):
            assert acline == exline


# ONNX cases
ONNX_CASES = [
    [
        "identity",
        "none",
        r"""
        [I] ==== ONNX Model ====
            Name: test_identity | Opset: 8

            ---- 1 Graph Input(s) ----
            {x [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 1 Graph Output(s) ----
            {y [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 0 Initializer(s) ----

            ---- 1 Node(s) ----
        """,
    ],
    [
        "identity",
        "basic",
        r"""
        [I] ==== ONNX Model ====
            Name: test_identity | Opset: 8

            ---- 1 Graph Input(s) ----
            {x [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 1 Graph Output(s) ----
            {y [dtype=float32, shape=(1, 1, 2, 2)]}

            ---- 0 Initializer(s) ----
            {}

            ---- 1 Node(s) ----
            Node 0    |  [Op: Identity]
                {x [dtype=float32, shape=(1, 1, 2, 2)]}
                 -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
        """,
    ],
    [
        "identity_with_initializer",
        "basic",
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Input(s) ----
            {}

            ---- 1 Graph Output(s) ----
            {Y [dtype=float32, shape=(2, 2)]}

            ---- 1 Initializer(s) ----
            {X [dtype=float32, shape=(2, 2)]}

            ---- 1 Node(s) ----
            Node 0    |  [Op: Identity]
                {Initializer | X [dtype=float32, shape=(2, 2)]}
                 -> {Y [dtype=float32, shape=(2, 2)]}
        """,
    ],
    [
        "identity_with_initializer",
        "full",
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Input(s) ----
            {}

            ---- 1 Graph Output(s) ----
            {Y [dtype=float32, shape=(2, 2)]}

            ---- 1 Initializer(s) ----
            Initializer | X [dtype=float32, shape=[2, 2]] | Values:
                [[1. 1.]
                 [1. 1.]]

            ---- 1 Node(s) ----
            Node 0    |  [Op: Identity]
                {Initializer | X [dtype=float32, shape=(2, 2)]}
                 -> {Y [dtype=float32, shape=(2, 2)]}
        """,
    ],
    [
        "tensor_attr",
        "basic",
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Input(s) ----
            {}

            ---- 1 Graph Output(s) ----
            {const_out [dtype=float32, shape=(14, 14)]}

            ---- 0 Initializer(s) ----
            {}

            ---- 1 Node(s) ----
            Node 0    |  [Op: Constant]
                {} -> {const_out [dtype=float32, shape=(14, 14)]}
        """,
    ],
    [
        "tensor_attr",
        "attrs",
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Input(s) ----
            {}

            ---- 1 Graph Output(s) ----
            {const_out [dtype=float32, shape=(14, 14)]}

            ---- 0 Initializer(s) ----
            {}

            ---- 1 Node(s) ----
            Node 0    |  [Op: Constant]
                {} -> {const_out [dtype=float32, shape=(14, 14)]}
                ---- Attributes ----
                value = Tensor: [dtype=float32, shape=[14, 14]]
        """,
    ],
    [
        "tensor_attr",
        "full",
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | Opset: 11

            ---- 0 Graph Input(s) ----
            {}

            ---- 1 Graph Output(s) ----
            {const_out [dtype=float32, shape=(14, 14)]}

            ---- 0 Initializer(s) ----
            {}

            ---- 1 Node(s) ----
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
        """,
    ],
    [
        "scan",
        "full",
        r"""
        [I] ==== ONNX Model ====
            Name: graph | Opset: 10

            ---- 2 Graph Input(s) ----
            {initial [dtype=float32, shape=(2,)],
             x [dtype=float32, shape=(3, 2)]}

            ---- 2 Graph Output(s) ----
            {y [dtype=float32, shape=(2,)],
             z [dtype=float32, shape=(3, 2)]}

            ---- 0 Initializer(s) ----
            {}

            ---- 1 Node(s) ----
            Node 0    |  [Op: Scan]
                {initial [dtype=float32, shape=(2,)],
                 x [dtype=float32, shape=(3, 2)]}
                 -> {y [dtype=float32, shape=(2,)],
                     z [dtype=float32, shape=(3, 2)]}
                ---- Attributes ----
                body =
                        ---- 2 Subgraph Input(s) ----
                        {sum_in [dtype=float32, shape=(2,)],
                         next [dtype=float32, shape=(2,)]}

                        ---- 2 Subgraph Output(s) ----
                        {sum_out [dtype=float32, shape=(2,)],
                         scan_out [dtype=float32, shape=(2,)]}

                        ---- 0 Initializer(s) ----
                        {}

                        ---- 2 Node(s) ----
                        Node 0    |  [Op: Add]
                            {sum_in [dtype=float32, shape=(2,)],
                             next [dtype=float32, shape=(2,)]}
                             -> {sum_out [dtype=float32, shape=(2,)]}

                        Node 1    |  [Op: Identity]
                            {sum_out [dtype=float32, shape=(2,)]}
                             -> {scan_out [dtype=float32, shape=(2,)]}

                num_scan_inputs = 1
        """,
    ],
    [
        "dim_param",
        "basic",
        r"""
        [I] ==== ONNX Model ====
            Name: tf2onnx | Opset: 10

            ---- 1 Graph Input(s) ----
            {Input:0 [dtype=float32, shape=('dim0', 16, 128)]}

            ---- 1 Graph Output(s) ----
            {Output:0 [dtype=float32, shape=('dim0', 16, 128)]}

            ---- 0 Initializer(s) ----
            {}

            ---- 1 Node(s) ----
            Node 0    |  [Op: Identity]
                {Input:0 [dtype=float32, shape=('dim0', 16, 128)]}
                 -> {Output:0 [dtype=float32, shape=('dim0', 16, 128)]}
        """,
    ],
]


# List[model, expected_files, expected_output]
TEST_CAPABILITY_CASES = [
    (
        "capability",
        [
            "results.txt",
            "supported_subgraph-nodes-1-1.onnx",
            "supported_subgraph-nodes-3-3.onnx",
            "unsupported_subgraph-nodes-0-0.onnx",
            "unsupported_subgraph-nodes-2-2.onnx",
            "unsupported_subgraph-nodes-4-4.onnx",
        ],
        """
        [I] ===== Summary =====
            Operator | Count   | Reason                                                                                                                                                            | Nodes
            -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            FAKE!    |       2 | In node 0 (importFallbackPluginImporter): UNSUPPORTED_NODE: Assertion failed: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[0, 1], [2, 3]]
            FAKER!   |       1 | In node 0 (importFallbackPluginImporter): UNSUPPORTED_NODE: Assertion failed: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[4, 5]]
        """,
    ),
    (
        "identity_identity",
        [],
        """
        Graph is fully supported by TensorRT; Will not generate subgraphs.
        """,
    ),
]


class TestCapability(object):
    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.0"), reason="supports_model API not available before TRT 8.0"
    )
    @pytest.mark.parametrize("case", TEST_CAPABILITY_CASES, ids=lambda case: case[0])
    def test_capability(self, case):
        model, expected_files, expected_summary = case
        with tempfile.TemporaryDirectory() as outdir:
            status = run_polygraphy_inspect(
                ["capability", ONNX_MODELS[model].path, "-o", os.path.join(outdir, "subgraphs")],
            )
            assert sorted(map(os.path.basename, glob.glob(os.path.join(outdir, "subgraphs", "**")))) == sorted(
                expected_files
            )
            assert dedent(expected_summary).strip() in status.stdout


class TestInspectModel(object):
    @pytest.mark.parametrize("case", ONNX_CASES, ids=lambda case: "{:}-{:}".format(case[0], case[1]))
    def test_model_onnx(self, case):
        model, mode, expected = case
        status = run_polygraphy_inspect(
            ["model", ONNX_MODELS[model].path, "--mode={:}".format(mode)], disable_verbose=True
        )

        expected = dedent(expected).strip()
        actual = "\n".join(status.stdout.splitlines()[1:])  # Ignore loading message

        check_lines_match(actual, expected)

    @pytest.mark.parametrize("model", ["identity", "scan", "tensor_attr"])
    def test_model_trt_sanity(self, run_inspect_model, model):
        import tensorrt as trt

        if model == "tensor_attr" and mod.version(trt.__version__) < mod.version("7.2"):
            pytest.skip("Models with constant outputs were not supported before 7.2")

        if model == "scan" and mod.version(trt.__version__) < mod.version("7.0"):
            pytest.skip("Scan was not supported until 7.0")

        run_inspect_model([ONNX_MODELS[model].path, "--display-as=trt"])

    def test_model_trt_network_script(self):
        script = dedent(
            """
            from polygraphy.backend.trt import CreateNetwork
            from polygraphy import func
            import tensorrt as trt

            @func.extend(CreateNetwork())
            def load_network(builder, network):
                inp = network.add_input("input", dtype=trt.float32, shape=(1, 1))
                out = network.add_identity(inp).get_output(0)
                network.mark_output(out)
        """
        )

        with util.NamedTemporaryFile("w+", suffix=".py") as f:
            f.write(script)
            f.flush()

            run_polygraphy_inspect(["model", f.name])

    def test_model_trt_engine_sanity(self, run_inspect_model, identity_engine):
        run_inspect_model([identity_engine, "--model-type=engine"])

    def test_model_tf_sanity(self, run_inspect_model):
        run_inspect_model([TF_MODELS["identity"].path, "--model-type=frozen"])


class TestInspectData(object):
    @pytest.mark.parametrize("opts", [[], ["--show-values"]])
    def test_outputs(self, opts):
        with util.NamedTemporaryFile() as outpath:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", outpath.name])
            run_polygraphy_inspect(["data", outpath.name] + opts)

    @pytest.mark.parametrize("opts", [[], ["--show-values"]])
    def test_inputs(self, opts):
        with util.NamedTemporaryFile() as outpath:
            run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-inputs", outpath.name])
            run_polygraphy_inspect(["data", outpath.name] + opts)


TACTIC_REPLAY_CASES = [
    [
        "identity",
        r"""
        [I] Layer: node_of_y
                Algorithm: (Implementation: -2147483642, Tactic: 0) | Inputs: (('TensorFormat.LINEAR', 'DataType.FLOAT'),) | Outputs: (('TensorFormat.LINEAR', 'DataType.FLOAT'),)
        """,
    ],
]


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
class TestInspectTactics(object):
    @pytest.mark.parametrize("case", TACTIC_REPLAY_CASES, ids=lambda case: case[0])
    def test_show_tactics(self, case):
        with util.NamedTemporaryFile() as replay:
            model_name, expected = case

            run_polygraphy_run([ONNX_MODELS[model_name].path, "--trt", "--save-tactics", replay.name])
            status = run_polygraphy_inspect(["tactics", replay.name], disable_verbose=True)

            expected = dedent(expected).strip()
            actual = status.stdout

            check_lines_match(actual, expected, should_check_line=lambda line: "Algorithm: " not in line)
