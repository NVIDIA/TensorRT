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
import glob
import os
import tempfile
from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import mod, util, constants
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    network_from_onnx_bytes,
    save_engine,
)
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
    print(f"Expected output:\n{expected}")

    actual = [
        line
        for line in actual.splitlines()
        if "Loading" not in line
        and not line.startswith("[V]")
        and not line.startswith("[W]")
    ]
    expected = expected.splitlines()
    assert len(actual) == len(expected)

    for acline, exline in zip(actual, expected):
        acline = acline.rstrip()
        exline = exline.rstrip()
        print(f"Checking line : {acline}")
        print(f"Expecting line: {exline}")
        if should_check_line(exline):
            assert acline == exline


# Format:
#   model_name
#   show_options
#   expected_output
#   additional_options
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
        [],
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
        [],
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
        [],
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
        [],
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
        [],
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
        [],
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
        [],
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
        [],
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
        [],
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
        [],
    ],
    [
        "ext_weights",
        ["layers", "weights"],
        r"""
        [E] Failed to load weights.
            Note: Error was: Data of TensorProto ( tensor name: a) should be stored in ext_weights.data, but it doesn't exist or is not accessible.
        [E] Failed to load weights.
            Note: Error was: Data of TensorProto ( tensor name: b) should be stored in ext_weights.data, but it doesn't exist or is not accessible.
        [E] Failed to load weights.
            Note: Error was: Data of TensorProto ( tensor name: d) should be stored in ext_weights.data, but it doesn't exist or is not accessible.
        [I] ==== ONNX Model ====
            Name: onnx_graphsurgeon | ONNX Opset: 11

            ---- 1 Graph Input(s) ----
            {input [dtype=float32, shape=(1, 3)]}

            ---- 1 Graph Output(s) ----
            {output [dtype=float32, shape=(1, 3)]}

            ---- 3 Initializer(s) ----
            Initializer | a [dtype=float32, shape=[1, 3]] | Values:
                <error: failed to load weights>

            Initializer | b [dtype=float32, shape=[1, 3]] | Values:
                <error: failed to load weights>

            Initializer | d [dtype=float32, shape=[1, 3]] | Values:
                <error: failed to load weights>

            ---- 3 Node(s) ----
            Node 0    |  [Op: Add]
                {Initializer | a [dtype=float32, shape=(1, 3)],
                 Initializer | b [dtype=float32, shape=(1, 3)]}
                 -> {c}

            Node 1    |  [Op: Add]
                {c,
                 Initializer | d [dtype=float32, shape=(1, 3)]}
                 -> {e}

            Node 2    |  [Op: Add]
                {input [dtype=float32, shape=(1, 3)],
                 e}
                 -> {output [dtype=float32, shape=(1, 3)]}
        """,
        ["--ignore-external-data"],
    ],
    # Ensure shapes without dim_param or dim_value set are treated as dynamic
    [
        "inp_dim_val_not_set",
        [],
        r"""
        [I] ==== ONNX Model ====
            Name:  | ONNX Opset: 13

            ---- 1 Graph Input(s) ----
            {X [dtype=float32, shape=(-1, -1, -1)]}

            ---- 1 Graph Output(s) ----
            {Y [dtype=float32, shape=(-1, -1, -1)]}

            ---- 0 Initializer(s) ----

            ---- 1 Node(s) ----
        """,
        [],
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

            ---- 2 Profile(s) (2 Tensor(s) Each) ----
            - Profile: 0
                Tensor: X          (Input), Index: 0 | Shapes: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
                Tensor: Y         (Output), Index: 1 | Shape: (1, 2, -1, -1)

            - Profile: 1
                Tensor: X          (Input), Index: 0 | Shapes: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
                Tensor: Y         (Output), Index: 1 | Shape: (1, 2, -1, -1)

            ---- 1 Layer(s) Per Profile ----
        """,
    ),
]


class TestInspectModel:
    @pytest.mark.parametrize(
        "case", ONNX_CASES, ids=lambda case: f"{case[0]}-{case[1]}"
    )
    def test_onnx(self, case, poly_inspect):
        model, show, expected, additional_opts = case
        status = poly_inspect(
            ["model", ONNX_MODELS[model].path, "--log-format=no-colors"]
            + (["--show"] + show if show else [])
            + additional_opts
        )

        expected = dedent(expected).strip()
        actual = "\n".join(status.stdout.splitlines()[1:])  # Ignore loading message

        check_lines_match(
            actual,
            expected,
            should_check_line=lambda line: "Note: Error was:" not in line,
        )

    def test_list_unbounded_dds(self, poly_inspect):
        cmd = [
            "model",
            ONNX_MODELS["unbounded_dds"].path,
            "--list-unbounded-dds",
            "--shape-inference",
        ]
        status = poly_inspect(cmd)
        assert "cast_out_6" in status.stdout

    @pytest.mark.parametrize("model", ["identity", "scan", "tensor_attr"])
    def test_trt_sanity(self, run_inspect_model, model):
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

    @pytest.mark.parametrize("case", ENGINE_CASES, ids=lambda case: f"{case[0]}")
    @pytest.mark.flaky(max_runs=3)
    def test_trt_engine(self, case, dynamic_identity_engine, poly_inspect):
        show, expected = case
        status = poly_inspect(
            ["model", dynamic_identity_engine] + (["--show"] + show if show else [])
        )

        expected = dedent(expected).strip()
        actual = "\n".join(status.stdout.splitlines()[1:])  # Ignore loading message

        if mod.version(trt.__version__) >= mod.version("9.0"):
            pytest.skip(
                "Output is different for TRT >=9, this test needs to be updated to account for that. "
            )

        check_lines_match(
            actual,
            expected,
            should_check_line=lambda exline: "Tactic =" not in exline
            and "Device Memory" not in exline
            and "Origin" not in exline
            and "Reformat" not in exline,
        )

    def test_tf_sanity(self, run_inspect_model):
        pytest.importorskip("tensorflow")

        run_inspect_model([TF_MODELS["identity"].path, "--model-type=frozen"])


class TestInspectData:
    @pytest.mark.parametrize("opts", [[], ["--show-values"]])
    def test_outputs(self, opts, poly_run, poly_inspect):
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--save-outputs",
                    outpath.name,
                ]
            )
            poly_inspect(["data", outpath.name] + opts)

    @pytest.mark.parametrize("opts", [[], ["--show-values"]])
    def test_inputs(self, opts, poly_run, poly_inspect):
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [
                    ONNX_MODELS["identity"].path,
                    "--onnxrt",
                    "--save-inputs",
                    outpath.name,
                ]
            )
            poly_inspect(["data", outpath.name] + opts)

    @pytest.mark.parametrize("num_items", [-1, 1, 2, 10, 12])
    def test_num_items(self, poly_run, poly_inspect, num_items):
        with util.NamedTemporaryFile() as outpath:
            poly_run(
                [
                    ONNX_MODELS["dynamic_identity"].path,
                    "--onnxrt",
                    "--save-inputs",
                    outpath.name,
                    "--input-shapes=X:[1,2,24,24]",
                ]
            )
            status = poly_inspect(
                [
                    "data",
                    outpath.name,
                    "--show-values",
                    "--line-width=-1",
                    f"--num-items={num_items}",
                ]
            )

            # Process only lines containing array print outs (which are all indented)
            lines = [
                line.strip()
                for line in status.stdout.splitlines()
                if line.strip()
                and line.startswith(constants.TAB * 2)
                and line.strip() != "..."
            ]
            for line in lines:
                items = [e for e in line.strip("[]").split() if "..." not in e]

                if num_items == -1:
                    assert len(items) == 24
                else:
                    assert len(items) == num_items * 2


TACTIC_REPLAY_CASES = [
    [
        "pow_scalar",
        (
            r"""
        [I] Layer: (Unnamed Layer* 0) [Shuffle]
                Algorithm: (Implementation: 2147483661, Tactic: 0) | Inputs: (TensorInfo(DataType.FLOAT, (), -1, 1),) | Outputs: (TensorInfo(DataType.FLOAT, (1,), -1, 1),)
            Layer: node_of_z
                Algorithm: (Implementation: 2147483651, Tactic: 1) | Inputs: (TensorInfo(DataType.FLOAT, (1,), -1, 1), TensorInfo(DataType.FLOAT, (1,), -1, 1)) | Outputs: (TensorInfo(DataType.FLOAT, (1,), -1, 1),)
        """
            if mod.version(trt.__version__) < mod.version("8.7")
            else r"""
        [I] Layer: ONNXTRT_Broadcast
                Algorithm: (Implementation: 2147483661, Tactic: 0) | Inputs: (TensorInfo(DataType.FLOAT, (), -1, 1),) | Outputs: (TensorInfo(DataType.FLOAT, (1,), -1, 1),)
            Layer: PWN(node_of_z)
                Algorithm: (Implementation: 2147483688, Tactic: 1) | Inputs: (TensorInfo(DataType.FLOAT, (1,), -1, 1), TensorInfo(DataType.FLOAT, (1,), -1, 1)) | Outputs: (TensorInfo(DataType.FLOAT, (1,), -1, 1),)
        """
        ),
    ],
]


class TestInspectTactics:
    @pytest.mark.parametrize("case", TACTIC_REPLAY_CASES, ids=lambda case: case[0])
    def test_show_tactics(self, case, poly_run, poly_inspect):
        with util.NamedTemporaryFile() as replay:
            model_name, expected = case

            poly_run(
                [
                    ONNX_MODELS[model_name].path,
                    "--trt",
                    "--save-tactics",
                    replay.name,
                ]
            )
            status = poly_inspect(["tactics", replay.name])

            expected = dedent(expected).strip()
            actual = status.stdout

            check_lines_match(
                actual,
                expected,
                should_check_line=lambda line: "Algorithm: " not in line,
            )


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
        (
            """
        [I] ===== Summary =====
            Operator | Count   | Reason                                                                                                                                                            | Nodes
            -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            FAKE!    |       2 | In node 0 (importFallbackPluginImporter): UNSUPPORTED_NODE: Assertion failed: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[0, 1], [2, 3]]
            FAKER!   |       1 | In node 0 (importFallbackPluginImporter): UNSUPPORTED_NODE: Assertion failed: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[4, 5]]
        """
            if mod.version(trt.__version__) < mod.version("10.0")
            else """
        [I] ===== Summary =====
            Operator | Count   | Reason                                                                                                                                                                           | Nodes
            --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            FAKE!    |       1 | In node 0 with name: Fake1 and operator: FAKE! (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"  | [[0, 1]]
            FAKE!    |       1 | In node 0 with name: Fake2 and operator: FAKE! (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"  | [[2, 3]]
            FAKER!   |       1 | In node 0 with name: Fake3 and operator: FAKER! (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[4, 5]]
        """
        ),
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
    @pytest.mark.script_launch_mode("subprocess")
    @pytest.mark.parametrize("case", TEST_CAPABILITY_CASES, ids=lambda case: case[0])
    def test_capability(self, case, poly_inspect):
        model, expected_files, expected_summary = case
        with tempfile.TemporaryDirectory() as outdir:
            status = poly_inspect(
                [
                    "capability",
                    "--with-partitioning",
                    ONNX_MODELS[model].path,
                    "-o",
                    os.path.join(outdir, "subgraphs"),
                ],
            )
            assert sorted(
                map(
                    os.path.basename,
                    glob.glob(os.path.join(outdir, "subgraphs", "**")),
                )
            ) == sorted(expected_files)
            assert dedent(expected_summary).strip() in dedent(status.stdout).strip()


class TestDiffTactics:
    def check_output(self, status, expected_output, expected_num=2):
        output = "\n".join(
            line
            for line in status.stdout.strip().splitlines()
            if "Loading tactic replay file from " not in line and "[V]" not in line
        )
        assert output == expected_output.format(num=expected_num).strip()

    def test_dir(self, replay_dir, poly_inspect):
        replay_dir, expected_output = replay_dir
        status = poly_inspect(["diff-tactics", "--dir", replay_dir])
        self.check_output(status, expected_output)

    def test_good_bad(self, replay_dir, poly_inspect):
        replay_dir, expected_output = replay_dir

        good = os.path.join(replay_dir, "good")
        bad = os.path.join(replay_dir, "bad")
        status = poly_inspect(["diff-tactics", "--good", good, "--bad", bad])
        self.check_output(status, expected_output)

    def test_good_bad_file(self, replay_dir, poly_inspect):
        replay_dir, expected_output = replay_dir

        def find_file(dirpath, filename):
            return glob.glob(os.path.join(dirpath, "**", filename), recursive=True)[0]

        good = find_file(os.path.join(replay_dir, "good"), "0.json")
        bad = find_file(os.path.join(replay_dir, "bad"), "1.json")
        status = poly_inspect(["diff-tactics", "--good", good, "--bad", bad])
        self.check_output(status, expected_output, expected_num=1)


class TestInspectSparsity:
    @pytest.mark.parametrize(
        "model_name", ["matmul", "matmul.bf16", "matmul.bf16.i32data", "conv"]
    )
    def test_prune_check(self, poly_inspect, model_name):
        with tempfile.TemporaryDirectory() as outdir:
            ipath = ONNX_MODELS[model_name].path
            status = poly_inspect(["sparsity", ipath])
            assert status

class TestDebugTensors:
    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("10.13"),
        reason="Feature not supported before 10.13",
    )
    def test_unfused_debug_tensors(self, poly_run, poly_inspect):
        with tempfile.TemporaryDirectory() as outdir:
            model = ONNX_MODELS["matmul_2layer"].path

            poly_run(
                [
                    model,
                    "--trt",
                    "--mark-unfused-tensors-as-debug-tensors",
                    "--save-outputs",
                    "output.json",
                    "--save-engine",
                    "debug_unfused.engine"
                ],
                cwd=outdir
            )
            status = poly_inspect(["model", "debug_unfused.engine", "--show", "layers", "--model-type", "engine", "--combine-tensor-info", "output.json"], cwd=outdir)

            assert status.stdout.count("min") == 5
            assert status.stdout.count("max") == 5  
            assert status.stdout.count("avg") == 4

            
