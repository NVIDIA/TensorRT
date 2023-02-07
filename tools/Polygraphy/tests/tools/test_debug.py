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
import io
import os
import sys
import tempfile
from textwrap import dedent

import numpy as np
import onnx_graphsurgeon as gs
import pytest
import tensorrt as trt
from polygraphy import mod
from polygraphy.backend.onnx import onnx_from_path
from polygraphy.json import save_json
from tests.helper import POLYGRAPHY_CMD
from tests.models.meta import ONNX_MODELS


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
class TestBuild:
    def test_good_bad(self, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            # Also includes --show-output sanity test
            status = poly_debug(
                [
                    "build",
                    ONNX_MODELS["identity"].path,
                    "--save-tactics=replay.json",
                    "--show-output",
                    "--artifacts-dir",
                    outdir,
                    "--until=good",
                    "--artifacts",
                    "replay.json",
                    "--check",
                    "true",
                ],
                cwd=outdir,
            )
            assert "Passed: 1/1 | Pass Rate: 100.0%" in status.stdout

            status = poly_debug(
                [
                    "build",
                    ONNX_MODELS["identity"].path,
                    "--save-tactics=replay.json",
                    "--artifacts-dir",
                    outdir,
                    "--until=bad",
                    "--artifacts",
                    "replay.json",
                    "--check",
                    "false",
                ],
                cwd=outdir,
            )
            assert "Passed: 0/1 | Pass Rate: 0.0%" in status.stdout

            def check_outdir(subdir):
                files = glob.glob(os.path.join(outdir, subdir, "*"))
                assert len(files) == 1
                basenames = list(map(os.path.basename, files))
                assert len([f for f in basenames if f.startswith("replay") and f.endswith(".json")]) == 1

            check_outdir("good")
            check_outdir("bad")


class TestPrecision:
    @pytest.mark.parametrize("check_status", ["true", "false"])
    @pytest.mark.parametrize("mode", ["bisect", "linear"])
    @pytest.mark.parametrize("direction", ["forward", "reverse"])
    @pytest.mark.parametrize("model", ["reducable", "const_foldable"])
    def test_sanity(self, mode, direction, check_status, model, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            poly_debug(
                [
                    "precision",
                    "--mode",
                    mode,
                    "--direction",
                    direction,
                    ONNX_MODELS[model].path,
                    "--int8",
                    "--check",
                    check_status,
                ],
                cwd=outdir,
            )


class TestReduce:
    FAKE_REDUCE_CHECKER = os.path.join(os.path.dirname(__file__), "fake_reduce_checker.py")

    # Test left branch, right branch, at the point of branching, and after the branch.
    @pytest.mark.parametrize(
        "fail_node",
        [
            "onnx_graphsurgeon_node_1",
            "onnx_graphsurgeon_node_3",
            "onnx_graphsurgeon_node_5",
            "onnx_graphsurgeon_node_7",
            "onnx_graphsurgeon_node_9",
        ],
    )
    @pytest.mark.parametrize("mode", ["linear", "bisect"])
    def test_can_isolate_node(self, fail_node, mode, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=reduced.onnx",
                    "--mode",
                    mode,
                    "--show-output",
                    "--min-good=good_reduced.onnx",
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    fail_node,
                ],
                cwd=outdir,
            )

            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))

            min_good_path = os.path.join(outdir, "good_reduced.onnx")
            good_model = None
            if os.path.exists(min_good_path):
                good_model = onnx_from_path(min_good_path)

            # The model should only contain one node - the failing one.
            # One exception - since bisect depends on node ordering, it sometimes doesn't
            # reduce branches to the maximum possible extent.
            if mode == "bisect" and fail_node == "onnx_graphsurgeon_node_1":
                assert len(model.graph.node) == 3
            elif mode == "bisect" and fail_node == "onnx_graphsurgeon_node_7":
                assert len(model.graph.node) == 2
            else:
                assert len(model.graph.node) == 1

            node_names = [node.name for node in model.graph.node]
            assert fail_node in node_names

            # For now we're just doing a very basic sanity check for --min-good
            if good_model:
                assert model != good_model

    # Run a test where the last node in the model is failing.
    # If we're not reducing inputs, then only the outputs should change
    def test_no_reduce_inputs(self, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=reduced.onnx",
                    "--show-output",
                    "--no-reduce-inputs",
                    "--mode=linear",
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    "onnx_graphsurgeon_node_7",
                ],
                cwd=outdir,
            )

            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            assert len(model.graph.node) == 4
            assert len(model.graph.input) == 2
            assert len(model.graph.output) == 1
            assert model.graph.output[0].name == "identity_out_6"

            node_names = [node.name for node in model.graph.node]
            assert "onnx_graphsurgeon_node_7" in node_names

    # Run a test where an input node in the model is failing.
    # If we're not reducing outputs, then only the inputs should change
    def test_no_reduce_outputs(self, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=reduced.onnx",
                    "--show-output",
                    "--no-reduce-outputs",
                    "--mode=linear",
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    "onnx_graphsurgeon_node_3",
                ],
                cwd=outdir,
            )

            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            assert len(model.graph.node) == 4
            assert len(model.graph.input) == 1
            assert len(model.graph.output) == 2
            assert model.graph.input[0].name == "Y0"

            node_names = [node.name for node in model.graph.node]
            assert "onnx_graphsurgeon_node_7" in node_names

    # In this test, we set up the checker to return 1 for the bad node, but 2 in other cases.
    # We want to ignore '2's and treat them as successes
    def test_reduce_custom_return_code(self, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=reduced.onnx",
                    "--show-output",
                    "--fail-code=1",  # Only 1s are real failures.
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    "onnx_graphsurgeon_node_5",
                    "--default-return-code=2",
                ],
                cwd=outdir,
            )

            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            assert len(model.graph.node) == 1
            assert model.graph.node[0].name == "onnx_graphsurgeon_node_5"

    # Here we set the failure return code to 0, which would normally mark succeeding cases as failing.
    # However, since we also set the --fail-regex, it will only regard as failures those runs which print the error message.
    @pytest.mark.parametrize(
        "fail_code_arg",
        [
            [],
            ["--fail-code=0"],
        ],
    )
    def test_reduce_custom_fail_message(self, fail_code_arg, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            # fake_reduce_checker will alternate error messages based on whether an arbitrary node is present in the model.
            poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=reduced.onnx",
                    "--show-output",
                    "--fail-regex",
                    "REALLY BAD",
                    "BAD NODE",
                ]
                + fail_code_arg
                + [
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    "onnx_graphsurgeon_node_5",
                    "--fail-return-code=0",
                ],
                cwd=outdir,
            )

            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            assert len(model.graph.node) == 1
            assert model.graph.node[0].name == "onnx_graphsurgeon_node_5"

    # In cases where both sides of a branch are required to reproduce the failure,
    # reduce should not remove the branch.
    @pytest.mark.parametrize(
        "fail_nodes",
        [
            ["onnx_graphsurgeon_node_1", "onnx_graphsurgeon_node_3"],
            ["onnx_graphsurgeon_node_7", "onnx_graphsurgeon_node_9"],
        ],
    )
    def test_no_reduce_required_branches(self, fail_nodes, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=reduced.onnx",
                    "--show-output",
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                ]
                + fail_nodes,
                cwd=outdir,
            )

            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            node_names = [node.name for node in model.graph.node]
            assert all(fail_node in node_names for fail_node in fail_nodes)
            assert len(model.graph.node) <= 3  # The branch on the opposite side of the model should be removed.

    @pytest.mark.parametrize("opts", [[], ["--force-fallback-shape-inference"]])
    def test_reduce_shape_inference(self, opts, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            status = poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["dynamic_identity"].path,
                    "--output=reduced.onnx",
                    "--show-output",
                    "--model-input-shapes=X:[1,2,5,5]",
                ]
                + opts
                + ["--check", "false"],
                cwd=outdir,
            )
            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            graph = gs.import_onnx(model)
            assert tuple(graph.inputs[0].shape) == (1, 2, 5, 5)
            assert tuple(graph.outputs[0].shape) == (1, 2, 5, 5)

    def test_reduce_with_constant(self, poly_debug):
        # Should be no failure when models including Constant nodes use fallback
        # shape inference; Constant nodes will be lowered to constant tensors.
        with tempfile.TemporaryDirectory() as outdir:
            poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable_with_const"].path,
                    "--no-shape-inference",
                    "--mode=linear",
                    "--output=reduced.onnx",
                ]
                + [
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    "onnx_graphsurgeon_node_3",
                ],
                cwd=outdir,
            )
            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            graph = gs.import_onnx(model)
            assert len(graph.nodes) == 1
            assert graph.nodes[0].name == "onnx_graphsurgeon_node_3"
            # Outputs of Constant nodes should not become Variables; thus the model should have no inputs.
            assert not graph.inputs

    # When using custom data, we should be able to provide it to `debug reduce` so that it
    # can use it to fold inputs in branchy models.
    @pytest.mark.parametrize("negative", [True, False])
    @pytest.mark.script_launch_mode("subprocess")
    def test_reduce_custom_data_multibranch_input(self, negative, poly_debug, poly_run):
        with tempfile.TemporaryDirectory() as outdir:
            model = ONNX_MODELS["reducable"].path

            inp_data_path = os.path.join(outdir, "custom_inputs.json")
            inputs = [{"X0": np.array([3.14159265], dtype=np.float32), "Y0": np.array([2.7749389])}]
            save_json(inputs, inp_data_path)

            # Generate golden outputs
            golden_outputs_path = os.path.join(outdir, "layerwise_outputs.json")
            poly_run(
                [
                    model,
                    "--onnxrt",
                    "--load-inputs",
                    inp_data_path,
                    "--onnx-outputs",
                    "mark",
                    "all",
                    "--save-outputs",
                    golden_outputs_path,
                ]
            )

            # Reduce with golden outputs
            out_path = os.path.join(outdir, "reduced.onnx")
            status = poly_debug(
                ["reduce", model, "--no-reduce-outputs", "-o", out_path]
                + (["--load-inputs", inp_data_path] if not negative else [])
                + [
                    "--check",
                    *POLYGRAPHY_CMD,
                    "run",
                    "polygraphy_debug.onnx",
                    "--onnxrt",
                    "--load-inputs",
                    inp_data_path,
                    "--load-outputs",
                    golden_outputs_path,
                ],
                cwd=outdir,
            )

            # In the negative test, this should fail since `reduce` will use a data loader that is
            # distinct from the input data we specified.
            # Otherwise, reduce should use the data loader we provided and hence pass
            assert ("FAILED" in status.stdout + status.stderr) == negative
            assert ("Difference exceeds tolerance" in status.stdout + status.stderr) == negative
            # Reduce should issue a warning when it detects that the default data loader is in use.
            assert (
                "Please ensure that you have provided a data loader argument directly" in status.stdout + status.stderr
            ) == negative

    @pytest.mark.script_launch_mode("subprocess")
    def test_reduce_with_replay(self, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            status = poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=first.onnx",
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    "onnx_graphsurgeon_node_1",
                ],
                cwd=outdir,
            )

            assert "Running fake_reduce_checker.py" in status.stdout
            assert os.path.exists(os.path.join(outdir, "polygraphy_debug_replay.json"))

            # Next reduce using the replay file - the fake reduce checker should *not* be used.
            status = poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=second.onnx",
                    "--load-debug-replay=polygraphy_debug_replay.json",
                    "--check",
                    TestReduce.FAKE_REDUCE_CHECKER,
                    "polygraphy_debug.onnx",
                    "--fail-node",
                    "onnx_graphsurgeon_node_1",
                ],
                cwd=outdir,
            )

            assert "Running fake_reduce_checker.py" not in status.stdout
            assert "Loading iteration information from debug replay" in status.stdout

            # The behavior of both runs should be identical
            first = onnx_from_path(os.path.join(outdir, "first.onnx"))
            second = onnx_from_path(os.path.join(outdir, "second.onnx"))

            assert first == second

    @pytest.mark.script_launch_mode("subprocess")
    @pytest.mark.parametrize(
        "responses",
        [
            ["p", "f", "f", "f", "p"],
            # When an invalid option is provided, it should prompt again.
            ["d", "z", "y", "p", "f", "f", "f", "p"],
        ],
    )
    def test_reduce_interactive(self, poly_debug, responses):
        with tempfile.TemporaryDirectory() as outdir:
            responses = io.StringIO("\n".join(responses))

            status = poly_debug(
                [
                    "reduce",
                    ONNX_MODELS["reducable"].path,
                    "--output=first.onnx",
                ],
                cwd=outdir,
                stdin=responses,
            )
            assert "Did 'polygraphy_debug.onnx' [p]ass or [f]ail?" in status.stdout


class TestRepeat:
    @pytest.mark.parametrize(
        "until, check, expected_iters",
        [
            ("good", "true", 1),
            ("bad", "false", 1),
            ("5", "false", 5),
        ],
    )
    def test_until(self, until, check, expected_iters, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            status = poly_debug(["repeat", "--until", until, "--check", check], cwd=outdir)
            assert f"Finished {expected_iters} iteration(s)" in status.stdout

    def test_iteration_info(self, poly_debug):
        with tempfile.TemporaryDirectory() as outdir:
            iter_info = os.path.join(outdir, "iter_info.json")
            check_script = os.path.join(outdir, "check.py")

            # Hacky Python script to make sure the iteration is actually incremented
            check_num = f"""
            import json
            import os
            iter_info = json.load(open('{iter_info}', "r"))

            file_name = str(iter_info["iteration"]) + ".txt"
            path = os.path.abspath(file_name)
            print(path)

            assert not os.path.exists(path)
            with open(path, "w") as f:
                f.write("File")
            """

            with open(check_script, "w") as f:
                f.write(dedent(check_num))

            status = poly_debug(
                [
                    "repeat",
                    "--until=5",
                    "--iteration-info",
                    iter_info,
                    "--show-output",
                    "--check",
                    sys.executable,
                    check_script,
                ],
                cwd=outdir,
            )
            assert "FAILED" not in status.stdout
            assert "Passed: 5/5 | Pass Rate: 100.0%" in status.stdout

            # Iteration info should be cleaned up afterwards
            assert not os.path.exists(iter_info)

    @pytest.mark.parametrize(
        "opts,expected_output",
        [
            ([], "Passed: 0/5 | Pass Rate: 0.0%"),
            (["--ignore-fail-code=2"], "Passed: 0/5 | Pass Rate: 0.0%"),
            (["--ignore-fail-code=1"], "Passed: 5/5 | Pass Rate: 100.0%"),
        ],
    )
    def test_ignore_fail_code(self, poly_debug, opts, expected_output):
        with tempfile.TemporaryDirectory() as outdir:
            status = poly_debug(["repeat", "--until=5"] + opts + ["--check", "false"], cwd=outdir)
            assert expected_output in status.stdout
