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
import sys
import tempfile
from collections import namedtuple
from textwrap import dedent

import onnx_graphsurgeon as gs
import pytest
import tensorrt as trt
from polygraphy import mod
from polygraphy.backend.onnx import onnx_from_path
from polygraphy.backend.trt import Algorithm, TacticReplayData
from polygraphy.json import save_json
from tests.models.meta import ONNX_MODELS
from tests.tools.common import run_polygraphy_debug

FakeAlgorithmContext = namedtuple("FakeAlgorithmContext", ["name", "num_inputs", "num_outputs"])
FakeAlgorithm = namedtuple("FakeAlgorithm", ["algorithm_variant", "io_info"])
FakeAlgorithm.get_algorithm_io_info = lambda this, index: this.io_info[index]

FakeAlgorithmVariant = namedtuple("FakeAlgorithmVariant", ["implementation", "tactic"])
FakeAlgorithmIOInfo = namedtuple("FakeAlgorithmIOInfo", ["tensor_format", "dtype"])


def fake_context(name, num_inputs=1, num_outputs=1):
    return FakeAlgorithmContext(name=name, num_inputs=num_inputs, num_outputs=num_outputs)


def fake_algo(implementation=6, tactic=0, num_io=2, tensor_format=trt.TensorFormat.LINEAR, dtype=trt.float32):
    io_info = [FakeAlgorithmIOInfo(tensor_format=tensor_format, dtype=dtype)] * num_io
    return FakeAlgorithm(algorithm_variant=FakeAlgorithmVariant(implementation, tactic), io_info=io_info)


@pytest.fixture(scope="session", params=["", "subdir"])
def replay_dir(request):
    def make_replay(tactic):
        return TacticReplayData().add("layer0", Algorithm.from_trt(fake_context("layer0"), fake_algo(0, tactic)))

    with tempfile.TemporaryDirectory() as dir:

        def make_path(prefix, *args):
            path = os.path.join(dir, prefix)
            if request.param:
                path = os.path.join(path, request.param)
            path = os.path.join(path, *args)
            return path

        # Good tactics
        save_json(make_replay(0), make_path("good", "0.json"))
        save_json(make_replay(1), make_path("good", "1.json"))

        # Bad tactics
        save_json(make_replay(1), make_path("bad", "0.json"))
        save_json(make_replay(2), make_path("bad", "1.json"))

        EXPECTED_OUTPUT = dedent(
            """
        [I] Loaded {num} good tactic replays.
        [I] Loaded {num} bad tactic replays.
        [I] Found potentially bad tactics:
        [I] Layer: layer0
                Algorithms: ["(Implementation: 0, Tactic: 2) | Inputs: (('TensorFormat.LINEAR', 'DataType.FLOAT'),) | Outputs: (('TensorFormat.LINEAR', 'DataType.FLOAT'),)"]
        """
        )
        yield dir, EXPECTED_OUTPUT


class TestDiffTactics(object):
    def check_output(self, status, expected_output, expected_num=2):
        output = "\n".join(
            line for line in status.stdout.strip().splitlines() if "Loading tactic replay file from " not in line
        )
        assert output == expected_output.format(num=expected_num).strip()

    def test_dir(self, replay_dir):
        replay_dir, expected_output = replay_dir
        status = run_polygraphy_debug(["diff-tactics", "--dir", replay_dir], disable_verbose=True)
        self.check_output(status, expected_output)

    def test_good_bad(self, replay_dir):
        replay_dir, expected_output = replay_dir

        good = os.path.join(replay_dir, "good")
        bad = os.path.join(replay_dir, "bad")
        status = run_polygraphy_debug(["diff-tactics", "--good", good, "--bad", bad], disable_verbose=True)
        self.check_output(status, expected_output)

    def test_good_bad_file(self, replay_dir):
        replay_dir, expected_output = replay_dir

        def find_file(dirpath, filename):
            return glob.glob(os.path.join(dirpath, "**", filename), recursive=True)[0]

        good = find_file(os.path.join(replay_dir, "good"), "0.json")
        bad = find_file(os.path.join(replay_dir, "bad"), "1.json")
        status = run_polygraphy_debug(["diff-tactics", "--good", good, "--bad", bad], disable_verbose=True)
        self.check_output(status, expected_output, expected_num=1)


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
class TestBuild(object):
    def test_good_bad(self):
        with tempfile.TemporaryDirectory() as outdir:
            # Also includes --show-output sanity test
            status = run_polygraphy_debug(
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

            status = run_polygraphy_debug(
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


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
class TestPrecision(object):
    @pytest.mark.parametrize("check_status", ["true", "false"])
    @pytest.mark.parametrize("mode", ["bisect", "linear"])
    @pytest.mark.parametrize("direction", ["forward", "reverse"])
    @pytest.mark.parametrize("model", ["reducable", "const_foldable"])
    def test_sanity(self, mode, direction, check_status, model):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_debug(
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


class TestReduce(object):
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
    def test_can_isolate_node(self, fail_node, mode):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_debug(
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
                disable_verbose=True,
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
    def test_no_reduce_inputs(self):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_debug(
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
                disable_verbose=True,
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
    def test_no_reduce_outputs(self):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_debug(
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
                disable_verbose=True,
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
    def test_reduce_custom_return_code(self):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_debug(
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
                disable_verbose=True,
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
    def test_reduce_custom_fail_message(self, fail_code_arg):
        with tempfile.TemporaryDirectory() as outdir:
            # fake_reduce_checker will alternate error messages based on whether an arbitrary node is present in the model.
            run_polygraphy_debug(
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
                disable_verbose=True,
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
    def test_no_reduce_required_branches(self, fail_nodes):
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_debug(
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
                disable_verbose=True,
                cwd=outdir,
            )

            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            node_names = [node.name for node in model.graph.node]
            assert all(fail_node in node_names for fail_node in fail_nodes)
            assert len(model.graph.node) <= 3  # The branch on the opposite side of the model should be removed.

    @pytest.mark.parametrize("opts", [[], ["--force-fallback-shape-inference"]])
    def test_reduce_shape_inference(self, opts):
        with tempfile.TemporaryDirectory() as outdir:
            status = run_polygraphy_debug(
                [
                    "reduce",
                    ONNX_MODELS["dynamic_identity"].path,
                    "--output=reduced.onnx",
                    "--show-output",
                    "--model-input-shapes=X:[1,2,5,5]",
                ]
                + opts
                + ["--check", "false"],
                disable_verbose=True,
                cwd=outdir,
            )
            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            graph = gs.import_onnx(model)
            assert tuple(graph.inputs[0].shape) == (1, 2, 5, 5)
            assert tuple(graph.outputs[0].shape) == (1, 2, 5, 5)

    def test_reduce_with_constant(self):
        # Should be no failure when models including Constant nodes use fallback
        # shape inference; Constant nodes will be lowered to constant tensors.
        with tempfile.TemporaryDirectory() as outdir:
            run_polygraphy_debug(
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
                disable_verbose=True,
                cwd=outdir,
            )
            model = onnx_from_path(os.path.join(outdir, "reduced.onnx"))
            graph = gs.import_onnx(model)
            assert len(graph.nodes) == 1
            assert graph.nodes[0].name == "onnx_graphsurgeon_node_3"
            # Outputs of Constant nodes should not become Variables; thus the model should have no inputs.
            assert not graph.inputs


class TestRepeat(object):
    @pytest.mark.parametrize(
        "until, check, expected_iters",
        [
            ("good", "true", 1),
            ("bad", "false", 1),
            ("5", "false", 5),
        ],
    )
    def test_until(self, until, check, expected_iters):
        status = run_polygraphy_debug(["repeat", "--until", until, "--check", check])
        assert "Finished {:} iteration(s)".format(expected_iters) in status.stdout

    def test_iteration_info(self):
        with tempfile.TemporaryDirectory() as outdir:
            iter_info = os.path.join(outdir, "iter_info.json")
            check_script = os.path.join(outdir, "check.py")

            # Hacky Python script to make sure the iteration is actually incremented
            check_num = """
            import json
            import os
            iter_info = json.load(open('{:}', "r"))

            file_name = str(iter_info["iteration"]) + ".txt"
            path = os.path.abspath(file_name)
            print(path)

            assert not os.path.exists(path)
            with open(path, "w") as f:
                f.write("File")
            """.format(
                iter_info
            )

            with open(check_script, "w") as f:
                f.write(dedent(check_num))

            status = run_polygraphy_debug(
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

    def test_ignore_fail_code(self):
        # Sanity check to make sure the command normally fails.
        status = run_polygraphy_debug(["repeat", "--until=5", "--check", "false"])
        assert "Passed: 0/5 | Pass Rate: 0.0%" in status.stdout

        status = run_polygraphy_debug(["repeat", "--until=5", "--ignore-fail-code=2", "--check", "false"])
        assert "Passed: 0/5 | Pass Rate: 0.0%" in status.stdout

        status = run_polygraphy_debug(["repeat", "--until=5", "--ignore-fail-code=1", "--check", "false"])
        assert "Passed: 5/5 | Pass Rate: 100.0%" in status.stdout
