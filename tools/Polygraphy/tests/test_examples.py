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
from polygraphy.logger.logger import G_LOGGER

from tests.common import version

import subprocess as sp
import pytest
import copy
import os

import tensorrt as trt


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))
EXAMPLES_ROOT = os.path.join(ROOT_DIR, "examples")

# Extract any ``` blocks from the README
# Each block is stored as a separate string in the returned list
def load_code_blocks_from_readme(readme):
    def ignore_command(cmd):
        return "pip" in cmd

    commands = []
    with open(readme, 'r') as f:
        in_command_block = False
        for line in f.readlines():
            if not in_command_block and "```" in line:
                commands.append([])
                in_command_block = True
            elif in_command_block:
                if "```" in line:
                    in_command_block = False
                elif not ignore_command(line):
                    commands[-1].append(line.rstrip())
    # commands is List[List[str]] - flatten:
    commands = ["\n".join(command) for command in commands]
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
        status = sp.run(command, cwd=self.path, env=env, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
        assert status.returncode == 0, status.stdout + "\n" + status.stderr
        return status


    def __exit__(self, exc_type, exc_value, traceback):
        """
        Checks for and removes artifacts expected by this example
        """
        for artifact in self.artifacts:
            print("Checking for the existence of artifact: {:}".format(artifact))
            assert os.path.exists(artifact)
            os.remove(artifact)


    def __str__(self):
        return os.path.relpath(self.path, EXAMPLES_ROOT)


API_EXAMPLES = [
    Example(["api", "01_comparing_frameworks"]),
    Example(["api", "02_using_real_data"]),
    Example(["api", "03_interoperating_with_tensorrt"]),
    Example(["api", "04_int8_calibration_in_tensorrt"], artifact_names=["identity-calib.cache"]),
    Example(["api", "05_using_tensorrt_network_api"]),
]

@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
@pytest.mark.parametrize("example", API_EXAMPLES, ids=lambda case: str(case))
def test_api_examples(example):
    with example as commands:
        for command in commands:
            example.run(command)


CLI_EXAMPLES = [
    # Run
    Example(["cli", "run", "01_comparing_frameworks"]),
    Example(["cli", "run", "02_comparing_across_runs"], artifact_names=["system_a_results.pkl"]),
    Example(["cli", "run", "03_generating_a_comparison_script"], artifact_names=["compare_trt_onnxrt.py"]),
    # Surgeon
    Example(["cli", "surgeon", "01_isolating_subgraphs"], artifact_names=["subgraph.onnx"]),
]

@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
@pytest.mark.parametrize("example", CLI_EXAMPLES, ids=lambda case: str(case))
def test_cli_examples(example):
    with example as commands:
        for command in commands:
            example.run(command)


CLI_INSPECT_EXAMPLES = [
    Example(["cli", "inspect", "01_inspecting_a_tensorrt_network"]),
    Example(["cli", "inspect", "02_inspecting_a_tensorrt_engine"], artifact_names=["identity.engine"]),
    Example(["cli", "inspect", "03_inspecting_an_onnx_model"]),
    Example(["cli", "inspect", "04_inspecting_a_tensorflow_graph"]),
    Example(["cli", "inspect", "05_inspecting_run_results"], artifact_names=["onnxrt_results.pkl"]),
]

@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
@pytest.mark.parametrize("example", CLI_INSPECT_EXAMPLES, ids=lambda case: str(case))
def test_cli_inspect_examples(example):
    # Last block should be the expected output, and last command should generate it.
    with example as blocks:
        commands, expected_output = blocks[:-1], blocks[-1]
        for command in commands:
            actual_output = example.run(command).stdout

    print(actual_output)
    # Makes reading the diff way easier
    actual_lines = actual_output.splitlines()
    expected_lines = expected_output.splitlines()
    assert len(actual_lines) == len(expected_lines)

    # Indicates lines that may not match exactly
    NON_EXACT_LINE_MARKERS = ["---- Runner: ", "    Layer"]

    for index, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines)):
        # Skip whitespace, and lines that include runner names (since those have timestamps)
        if expected_line.strip() and all([marker not in expected_line for marker in NON_EXACT_LINE_MARKERS]):
            print("Checking line {:}: {:}".format(index, expected_line))
            assert actual_line == expected_line
