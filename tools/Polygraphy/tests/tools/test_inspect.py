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
import glob
import os
import tempfile
from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_bytes, save_engine
from tests.models.meta import ONNX_MODELS, TF_MODELS


@pytest.fixture(
    params=[
        [],
        ["layers"],
        ["layers", "attrs"],
        ["layers", "attrs", "weights"],
        ["weights"],
    ],
)
def run_inspect_model(request, poly_inspect):
    show_args = (["--show"] if request.param else []) + request.param
    yield lambda additional_opts: poly_inspect(["model"] + additional_opts + show_args)


@pytest.fixture(scope="session")
def dynamic_identity_engine():
    with util.NamedTemporaryFile(suffix=".engine") as outpath:
        engine = engine_from_network(
            network_from_onnx_bytes(ONNX_MODELS["dynamic_identity"].loader),
            CreateConfig(
                profiles=[
                    Profile().add("X", (1, 2, 1, 1), (1, 2, 3, 3), (1, 2, 5, 5)),
                    Profile().add("X", (1, 2, 2, 2), (1, 2, 4, 4), (1, 2, 6, 6)),
                ]
            ),
        )
        save_engine(engine, path=outpath)
        yield outpath.name


def check_lines_match(actual, expected, should_check_line=lambda x: True):
    print(f"Actual output:\n{actual}")

    actual = [line for line in actual.splitlines() if "Loading" not in line and "[V]" not in line]
    expected = expected.splitlines()
    assert len(actual) == len(expected)

    for acline, exline in zip(actual, expected):
        acline = acline.rstrip()
        exline = exline.rstrip()
        print(f"Checking line : {acline}")
        print(f"Expecting line: {exline}")
        if should_check_line(exline):
            assert acline == exline


ONNX_CASES = [
    [
        "identity",
        [],
        r"""
        [I] ==== ONNX Model ====
            Name: test_identity | ONNX Opset: 8

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
        ["layers"],
        r"""
        [I] ==== ONNX Model ====
            Name: test_identity | ONNX Opset: 8

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
        ["layers"],
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | ONNX Opset: 11

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
        ["layers", "attrs", "weights"],
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | ONNX Opset: 11

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
        "add_with_dup_inputs",
        ["layers"],
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon_graph | ONNX Opset: 11

            ---- 1 Graph Input(s) ----
            {inp [dtype=float32, shape=(4, 4)]}

            ---- 1 Graph Output(s) ----
            {add_out_0 [dtype=float32, shape=()]}

            ---- 0 Initializer(s) ----
            {}

            ---- 1 Node(s) ----
            Node 0    | onnx_graphsurgeon_node_1 [Op: Add]
                {inp [dtype=float32, shape=(4, 4)],
                 inp [dtype=float32, shape=(4, 4)]}
                 -> {add_out_0 [dtype=float32, shape=()]}
        """,
    ],
    [
        "tensor_attr",
        ["layers"],
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | ONNX Opset: 11

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
        ["layers", "attrs"],
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | ONNX Opset: 11

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
        ["layers", "attrs", "weights"],
        r"""
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | ONNX Opset: 11

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
        ["layers", "attrs", "weights"],
        r"""
        [I] ==== ONNX Model ====
            Name: graph | ONNX Opset: 10

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
        ["layers"],
        r"""
        [I] ==== ONNX Model ====
            Name: tf2onnx | ONNX Opset: 10

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

# Format: List[Tuple[show_opts, expected]]
ENGINE_CASES = [
    (
        [],
        r"""
        [I] ==== TensorRT Engine ====
            Name: Unnamed Network 0 | Explicit Batch Engine

            ---- 1 Engine Input(s) ----
            {X [dtype=float32, shape=(1, 2, -1, -1)]}

            ---- 1 Engine Output(s) ----
            {Y [dtype=float32, shape=(1, 2, -1, -1)]}

            ---- Memory ----
            Device Memory: 0 bytes

            ---- 2 Profile(s) (2 Binding(s) Each) ----
            - Profile: 0
                Binding Index: 0 (Input)  [Name: X]             | Shapes: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
                Binding Index: 1 (Output) [Name: Y]             | Shape: (1, 2, -1, -1)

            - Profile: 1
                Binding Index: 2 (Input)  [Name: X [profile 1]] | Shapes: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
                Binding Index: 3 (Output) [Name: Y [profile 1]] | Shape: (1, 2, -1, -1)

            ---- 1 Layer(s) Per Profile ----
        """,
    ),
    (
        ["layers"],
        r"""
        [I] ==== TensorRT Engine ====
            Name: Unnamed Network 0 | Explicit Batch Engine

            ---- 1 Engine Input(s) ----
            {X [dtype=float32, shape=(1, 2, -1, -1)]}

            ---- 1 Engine Output(s) ----
            {Y [dtype=float32, shape=(1, 2, -1, -1)]}

            ---- Memory ----
            Device Memory: 0 bytes

            ---- 2 Profile(s) (2 Binding(s) Each) ----
            - Profile: 0
                Binding Index: 0 (Input)  [Name: X]             | Shapes: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
                Binding Index: 1 (Output) [Name: Y]             | Shape: (1, 2, -1, -1)

            - Profile: 1
                Binding Index: 2 (Input)  [Name: X [profile 1]] | Shapes: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
                Binding Index: 3 (Output) [Name: Y [profile 1]] | Shape: (1, 2, -1, -1)

            ---- 1 Layer(s) Per Profile ----
            - Profile: 0
                Layer 0    | node_of_Y [Op: Reformat]
                    {X [shape=(1, 2, -1, -1)]}
                     -> {Y [shape=(1, 2, -1, -1)]}

            - Profile: 1
                Layer 0    | node_of_Y [profile 1] [Op: Reformat]
                    {X [profile 1] [shape=(1, 2, -1, -1)]}
                     -> {Y [profile 1] [shape=(1, 2, -1, -1)]}
        """,
    ),
    (
        ["layers", "attrs"],
        r"""
        [I] ==== TensorRT Engine ====
            Name: Unnamed Network 0 | Explicit Batch Engine

            ---- 1 Engine Input(s) ----
            {X [dtype=float32, shape=(1, 2, -1, -1)]}

            ---- 1 Engine Output(s) ----
            {Y [dtype=float32, shape=(1, 2, -1, -1)]}

            ---- Memory ----
            Device Memory: 0 bytes

            ---- 2 Profile(s) (2 Binding(s) Each) ----
            - Profile: 0
                Binding Index: 0 (Input)  [Name: X]             | Shapes: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
                Binding Index: 1 (Output) [Name: Y]             | Shape: (1, 2, -1, -1)

            - Profile: 1
                Binding Index: 2 (Input)  [Name: X [profile 1]] | Shapes: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
                Binding Index: 3 (Output) [Name: Y [profile 1]] | Shape: (1, 2, -1, -1)

            ---- 1 Layer(s) Per Profile ----
            - Profile: 0
                Layer 0    | node_of_Y [Op: Reformat]
                    {X [shape=(1, 2, -1, -1)]}
                     -> {Y [shape=(1, 2, -1, -1)]}
                    ---- Attributes ----
                    Origin = IDENTITY
                    Tactic = 0x0

            - Profile: 1
                Layer 0    | node_of_Y [profile 1] [Op: Reformat]
                    {X [profile 1] [shape=(1, 2, -1, -1)]}
                     -> {Y [profile 1] [shape=(1, 2, -1, -1)]}
                    ---- Attributes ----
                    Origin = IDENTITY
                    Tactic = 0x0
        """,
    ),
]


class TestInspectModel:
    @pytest.mark.parametrize("case", ONNX_CASES, ids=lambda case: f"{case[0]}-{case[1]}")
    def test_onnx(self, case, poly_inspect):
        model, show, expected = case
        status = poly_inspect(["model", ONNX_MODELS[model].path, "--show"] + show)

        expected = dedent(expected).strip()
        actual = "\n".join(status.stdout.splitlines()[1:])  # Ignore loading message

        check_lines_match(actual, expected)

    @pytest.mark.parametrize("model", ["identity", "scan", "tensor_attr"])
    def test_trt_sanity(self, run_inspect_model, model):
        import tensorrt as trt

        if model == "tensor_attr" and mod.version(trt.__version__) < mod.version("7.2"):
            pytest.skip("Models with constant outputs were not supported before 7.2")

        if model == "scan" and mod.version(trt.__version__) < mod.version("7.0"):
            pytest.skip("Scan was not supported until 7.0")

        run_inspect_model([ONNX_MODELS[model].path, "--display-as=trt"])

    def test_trt_network_script(self, poly_inspect):
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
            os.fsync(f.fileno())

            poly_inspect(["model", f.name])

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.2"), reason="Unsupported for TRT 8.0 and older")
    @pytest.mark.parametrize("case", ENGINE_CASES, ids=lambda case: f"{case[0]}")
    def test_trt_engine(self, case, dynamic_identity_engine, poly_inspect):
        show, expected = case
        status = poly_inspect(["model", dynamic_identity_engine, "--show"] + show)

        expected = dedent(expected).strip()
        actual = "\n".join(status.stdout.splitlines()[1:])  # Ignore loading message

        check_lines_match(actual, expected, should_check_line=lambda exline: "Tactic =" not in exline)

    def test_tf_sanity(self, run_inspect_model):
        pytest.importorskip("tensorflow")

        run_inspect_model([TF_MODELS["identity"].path, "--model-type=frozen"])


class TestInspectData:
    @pytest.mark.parametrize("opts", [[], ["--show-values"]])
    def test_outputs(self, opts, poly_run, poly_inspect):
        with util.NamedTemporaryFile() as outpath:
            poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-outputs", outpath.name])
            poly_inspect(["data", outpath.name] + opts)

    @pytest.mark.parametrize("opts", [[], ["--show-values"]])
    def test_inputs(self, opts, poly_run, poly_inspect):
        with util.NamedTemporaryFile() as outpath:
            poly_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-inputs", outpath.name])
            poly_inspect(["data", outpath.name] + opts)


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
class TestInspectTactics:
    @pytest.mark.parametrize("case", TACTIC_REPLAY_CASES, ids=lambda case: case[0])
    def test_show_tactics(self, case, poly_run, poly_inspect):
        with util.NamedTemporaryFile() as replay:
            model_name, expected = case

            poly_run([ONNX_MODELS[model_name].path, "--trt", "--save-tactics", replay.name])
            status = poly_inspect(["tactics", replay.name])

            expected = dedent(expected).strip()
            actual = status.stdout

            check_lines_match(actual, expected, should_check_line=lambda line: "Algorithm: " not in line)


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


class TestCapability:
    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.0"), reason="supports_model API not available before TRT 8.0"
    )
    @pytest.mark.parametrize("case", TEST_CAPABILITY_CASES, ids=lambda case: case[0])
    def test_capability(self, case, poly_inspect):
        model, expected_files, expected_summary = case
        with tempfile.TemporaryDirectory() as outdir:
            status = poly_inspect(
                ["capability", ONNX_MODELS[model].path, "-o", os.path.join(outdir, "subgraphs")],
            )
            assert sorted(map(os.path.basename, glob.glob(os.path.join(outdir, "subgraphs", "**")))) == sorted(
                expected_files
            )
            assert dedent(expected_summary).strip() in status.stdout
