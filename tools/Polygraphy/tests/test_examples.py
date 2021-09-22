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
import copy
import os
import shutil
import subprocess as sp
from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import mod
from polygraphy.logger import G_LOGGER

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))
EXAMPLES_ROOT = os.path.join(ROOT_DIR, "examples")

IGNORE_START_MARKER = "<!-- Polygraphy Test: Ignore Start -->"
IGNORE_STOP_MARKER = "<!-- Polygraphy Test: Ignore End -->"

# Extract any ``` blocks from the README
# Each block is stored as a separate string in the returned list
def load_code_blocks_from_readme(readme):
    with open(readme, "r") as f:
        contents = f.read()
        # Check that the README has all the expected sections.
        assert "## Introduction" in contents, "All example READMEs should have an 'Introduction' section!"
        assert "## Running The Example" in contents, "All example READMEs should have a 'Running The Example' section!"

    commands = []
    with open(readme, "r") as f:
        in_command_block = False
        should_ignore = False
        block = []
        for line in f.readlines():
            if line.strip() == IGNORE_START_MARKER:
                should_ignore = True
            elif line.strip() == IGNORE_STOP_MARKER:
                should_ignore = False

            if should_ignore:
                continue

            if not in_command_block and "```" in line:
                block = [line.rstrip()]
                in_command_block = True
            elif in_command_block:
                if "```" in line:
                    in_command_block = False
                    commands.append(copy.copy(block) + [line.rstrip()])
                else:
                    block.append(line.rstrip())

    # commands is List[List[str]] - flatten and remove start/end markers:
    commands = [dedent("\n".join(block[1:-1])) for block in commands]
    return commands


class Example(object):
    def __init__(self, path_components, artifact_names=[]):
        self.path = os.path.join(EXAMPLES_ROOT, *path_components)
        self.artifacts = [os.path.join(self.path, name) for name in artifact_names]

    def __enter__(self):
        readme = os.path.join(self.path, "README.md")
        return load_code_blocks_from_readme(readme)

    def run(self, command):
        G_LOGGER.info("Running: {:} from cwd: {:}".format(command, self.path))
        env = copy.copy(os.environ)
        env["PYTHONPATH"] = ROOT_DIR
        env["PATH"] = os.path.join(ROOT_DIR, "bin") + os.path.pathsep + env["PATH"]
        # Remove whitespace args and escaped newlines
        command = [arg for arg in command.strip().split(" ") if arg.strip() and arg != "\\\n"]
        print("Running: {:}".format(" ".join(command)))
        status = sp.run(command, cwd=self.path, env=env, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
        print(status.stdout)
        print(status.stderr)
        assert status.returncode == 0, status.stdout + "\n" + status.stderr
        return status

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Checks for and removes artifacts expected by this example
        """
        for artifact in self.artifacts:
            print("Checking for the existence of artifact: {:}".format(artifact))
            assert os.path.exists(artifact), "{:} does not exist!".format(artifact)
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)

    def __str__(self):
        return os.path.relpath(self.path, EXAMPLES_ROOT)


API_EXAMPLES = [
    Example(["api", "00_inference_with_tensorrt"], artifact_names=["identity.engine"]),
    Example(["api", "01_comparing_frameworks"], artifact_names=["inference_results.json"]),
    Example(["api", "02_validating_on_a_dataset"]),
    Example(["api", "03_interoperating_with_tensorrt"]),
    Example(["api", "04_int8_calibration_in_tensorrt"], artifact_names=["identity-calib.cache"]),
    Example(["api", "05_using_tensorrt_network_api"]),
    Example(["api", "06_immediate_eval_api"], artifact_names=["identity.engine"]),
    Example(["api", "07_tensorrt_and_dynamic_shapes"], artifact_names=["dynamic_identity.engine"]),
]


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
@pytest.mark.parametrize("example", API_EXAMPLES, ids=lambda case: str(case))
def test_api_examples(example):
    if mod.version(trt.__version__) < mod.version("8.0") and (example.path.endswith("07_tensorrt_and_dynamic_shapes")):
        pytest.skip("Not intended for older versions of TRT")

    with example as commands:
        for command in commands:
            example.run(command)


CLI_EXAMPLES = [
    # Run
    Example(["cli", "run", "01_comparing_frameworks"]),
    Example(["cli", "run", "02_comparing_across_runs"], artifact_names=["system_a_results.json"]),
    Example(["cli", "run", "03_generating_a_comparison_script"], artifact_names=["compare_trt_onnxrt.py"]),
    Example(
        ["cli", "run", "04_defining_a_tensorrt_network_or_config_manually"],
        artifact_names=["my_define_network.py", "my_create_config.py"],
    ),
    Example(["cli", "run", "05_comparing_with_custom_data"]),
    # Convert
    Example(["cli", "convert", "01_int8_calibration_in_tensorrt"], artifact_names=["identity.engine"]),
    Example(
        ["cli", "convert", "02_deterministic_engine_builds_in_tensorrt"],
        artifact_names=["0.engine", "1.engine", "replay.json"],
    ),
    Example(["cli", "convert", "03_dynamic_shapes_in_tensorrt"], artifact_names=["dynamic_identity.engine"]),
    # Surgeon
    Example(["cli", "surgeon", "01_isolating_subgraphs"], artifact_names=["subgraph.onnx"]),
    Example(["cli", "surgeon", "02_folding_constants"], artifact_names=["folded.onnx"]),
    Example(["cli", "surgeon", "03_modifying_input_shapes"], artifact_names=["dynamic_identity.onnx"]),
    # Debug
    Example(["cli", "debug", "01_debugging_flaky_trt_tactics"], artifact_names=["replays", "golden.json"]),
    Example(
        ["cli", "debug", "02_reducing_failing_onnx_models"],
        artifact_names=[
            "inputs.json",
            "layerwise_golden.json",
            "layerwise_inputs.json",
            "initial_reduced.onnx",
            "final_reduced.onnx",
        ],
    ),
]


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
@pytest.mark.parametrize("example", CLI_EXAMPLES, ids=lambda case: str(case))
def test_cli_examples(example):
    if mod.version(trt.__version__) < mod.version("8.0") and (
        example.path.endswith("01_debugging_flaky_trt_tactics")
        or example.path.endswith("02_deterministic_engine_builds_in_tensorrt")
    ):
        pytest.skip("Tactic replays are not supported on older versions of TRT")

    with example as commands:
        for command in commands:
            example.run(command)


CLI_INSPECT_EXAMPLES = [
    Example(["cli", "inspect", "01_inspecting_a_tensorrt_network"]),
    Example(["cli", "inspect", "02_inspecting_a_tensorrt_engine"], artifact_names=["dynamic_identity.engine"]),
    Example(["cli", "inspect", "03_inspecting_an_onnx_model"]),
    Example(["cli", "inspect", "04_inspecting_a_tensorflow_graph"]),
    Example(["cli", "inspect", "05_inspecting_inference_outputs"], artifact_names=["outputs.json"]),
    Example(["cli", "inspect", "06_inspecting_input_data"], artifact_names=["inputs.json"]),
]

if mod.version(trt.__version__) >= mod.version("8.0"):
    CLI_INSPECT_EXAMPLES += [
        Example(["cli", "inspect", "07_inspecting_tactic_replays"], artifact_names=["replay.json"]),
    ]


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
@pytest.mark.parametrize("example", CLI_INSPECT_EXAMPLES, ids=lambda case: str(case))
def test_cli_inspect_examples(example):
    # Last block should be the expected output, and last command should generate it.
    with example as blocks:
        commands, expected_output = blocks[:-1], blocks[-1]
        for command in commands:
            actual_output = example.run(command).stdout

    print(actual_output)
    # Makes reading the diff way easier
    actual_lines = [line for line in actual_output.splitlines() if "[I] Loading " not in line and "[W] " not in line]

    expected_lines = expected_output.splitlines()
    assert len(actual_lines) == len(expected_lines)

    # Indicates lines that may not match exactly
    NON_EXACT_LINE_MARKERS = ["---- ", "    Layer", "        Algorithm:"]

    for index, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines)):
        # Skip whitespace, and lines that include runner names (since those have timestamps)
        if expected_line.strip() and all([marker not in expected_line for marker in NON_EXACT_LINE_MARKERS]):
            print("Checking line {:}: {:}".format(index, expected_line))
            assert actual_line == expected_line


DEV_EXAMPLES = [
    Example(["dev", "01_writing_cli_tools"], artifact_names=["data.json"]),
]


@pytest.mark.parametrize("example", DEV_EXAMPLES, ids=lambda case: str(case))
def test_dev_examples(example):
    with example as commands:
        for command in commands:
            example.run(command)
