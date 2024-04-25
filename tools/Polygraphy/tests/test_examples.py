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
import copy
import glob
import os
import shutil
from textwrap import dedent
from typing import List

import pytest
import tensorrt as trt
from polygraphy import mod, util

from tests.helper import ROOT_DIR

EXAMPLES_ROOT = os.path.join(ROOT_DIR, "examples")


class Marker:
    def __init__(self, matches_start_func=None, matches_end_func=None):
        self.matches_start = util.default(
            matches_start_func, lambda line: line == self.start
        )
        self.matches_end = util.default(matches_end_func, lambda line: line == self.end)

    @staticmethod
    def from_name(name):
        return Marker(
            matches_start_func=lambda line: line
            == f"<!-- Polygraphy Test: {name} Start -->",
            matches_end_func=lambda line: line
            == f"<!-- Polygraphy Test: {name} End -->",
        )


VALID_MARKERS = {
    # For command markers, the start marker may be annotated with a language tag, e.g. ```py, so an exact match is too strict.
    "command": Marker(
        matches_start_func=lambda line: line.startswith("```"),
        matches_end_func=lambda line: line == "```",
    ),
    # Marks an entire block to be ignored by the tests.
    "ignore": Marker.from_name("Ignore"),
    # Marks an entire block as being expected to fail.
    "xfail": Marker.from_name("XFAIL"),
}


class MarkerTracker:
    """
    Keeps track of active markers in the current README on a line-by-line basis.
    """

    def __init__(self, readme):
        self.readme = readme
        self.active_markers = set()
        self.entering_markers = set()  # The markers that we are currently entering
        self.exiting_markers = set()  # The markers that we are currently exiting

    def __enter__(self):
        self.file = open(self.readme, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def __iter__(self):
        for line in self.file.readlines():
            stripped_line = line.strip()
            self.entering_markers.clear()
            self.exiting_markers.clear()

            for marker in VALID_MARKERS.values():
                if not self.is_in(marker) and marker.matches_start(stripped_line):
                    self.active_markers.add(marker)
                    self.entering_markers.add(marker)
                elif marker.matches_end(stripped_line):
                    self.active_markers.remove(marker)
                    self.exiting_markers.add(marker)

            yield line.rstrip()

    def is_in(self, marker):
        """
        Whether we are currently on a line between the specified start and end marker.
        This will always return False for a line containing the marker itself.
        """
        return marker in self.active_markers and not (
            self.entering(marker) or self.exiting(marker)
        )

    def entering(self, marker):
        return marker in self.entering_markers

    def exiting(self, marker):
        return marker in self.exiting_markers


class CommandBlock:
    def __init__(self, xfail):
        self.xfail = xfail
        self.content = None

    def add(self, line):
        if self.content is None:
            self.content = line
        else:
            self.content += f"\n{line}"

    def __str__(self):
        return dedent(self.content)


# Extract any ``` blocks from the README
# NOTE: This parsing logic is not smart enough to handle multiple separate commands in a single block.
def load_command_blocks_from_readme(readme) -> List[CommandBlock]:
    with open(readme, "r") as f:
        contents = f.read()
        # Check that the README has all the expected sections.
        assert (
            "## Introduction" in contents
        ), "All example READMEs should have an 'Introduction' section!"
        assert (
            "## Running The Example" in contents
        ), "All example READMEs should have a 'Running The Example' section!"

    cmd_blocks = []
    with MarkerTracker(readme) as tracker:
        for line in tracker:
            if tracker.is_in(VALID_MARKERS["ignore"]):
                continue

            if tracker.entering(VALID_MARKERS["command"]):
                current_block = CommandBlock(
                    xfail=tracker.is_in(VALID_MARKERS["xfail"])
                )
            elif tracker.exiting(VALID_MARKERS["command"]):
                cmd_blocks.append(copy.copy(current_block))
            elif tracker.is_in(VALID_MARKERS["command"]):
                current_block.add(line)

    return cmd_blocks


class Example:
    def __init__(self, path_components, artifact_names=[]):
        self.path = os.path.join(EXAMPLES_ROOT, *path_components)
        self.artifacts = [os.path.join(self.path, name) for name in artifact_names]
        # Ensures no files in addition to the specified artifacts were created.
        self.original_files = []

    def _get_file_list(self):
        return [
            path
            for path in glob.iglob(os.path.join(self.path, "*"))
            if "__pycache__" not in path
        ]

    def _remove_artifacts(self, must_exist=True):
        for artifact in self.artifacts:
            if must_exist:
                print(f"Checking for the existence of artifact: {artifact}")
                assert os.path.exists(artifact), f"{artifact} does not exist!"
            elif not os.path.exists(artifact):
                continue

            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)

    def __enter__(self):
        self._remove_artifacts(must_exist=False)

        self.original_files = self._get_file_list()
        readme = os.path.join(self.path, "README.md")
        return load_command_blocks_from_readme(readme)

    def run(self, cmd_block, sandboxed_install_run):
        # Remove whitespace args and escaped newlines
        command = [
            arg
            for arg in str(cmd_block).strip().split(" ")
            if arg.strip() and arg != "\\\n"
        ]
        status = sandboxed_install_run(command, cwd=self.path)

        details = f"Note: Command was: {' '.join(command)}.\n==== STDOUT ====\n{status.stdout}\n==== STDERR ====\n{status.stderr}"
        if cmd_block.xfail:
            assert (
                not status.success
            ), f"Command that was expected to fail did not fail. {details}"
        else:
            assert (
                status.success
            ), f"Command that was expected to succeed did not succeed. {details}"
        return status

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Checks for and removes artifacts expected by this example
        """
        self._remove_artifacts()
        assert (
            self._get_file_list() == self.original_files
        ), "Unexpected files were created. If this is the desired behavior, add the file paths to `artifact_names`"

    def __str__(self):
        return os.path.relpath(self.path, EXAMPLES_ROOT)


API_EXAMPLES = [
    Example(["api", "00_inference_with_tensorrt"], artifact_names=["identity.engine"]),
    Example(
        ["api", "01_comparing_frameworks"], artifact_names=["inference_results.json"]
    ),
    Example(["api", "02_validating_on_a_dataset"]),
    Example(["api", "03_interoperating_with_tensorrt"]),
    Example(
        ["api", "04_int8_calibration_in_tensorrt"],
        artifact_names=["identity-calib.cache"],
    ),
    Example(["api", "05_using_tensorrt_network_api"]),
    Example(["api", "06_immediate_eval_api"], artifact_names=["identity.engine"]),
    Example(
        ["api", "07_tensorrt_and_dynamic_shapes"],
        artifact_names=["dynamic_identity.engine"],
    ),
    Example(
        ["api", "08_working_with_run_results_and_saved_inputs_manually"],
        artifact_names=["inputs.json", "outputs.json"],
    ),
    Example(["api", "09_working_with_pytorch_tensors"]),
]


@pytest.mark.parametrize("example", API_EXAMPLES, ids=lambda case: str(case))
@pytest.mark.script_launch_mode("subprocess")
def test_api_examples(example, sandboxed_install_run):
    if "07_tensorrt_and_dynamic_shapes" in example.path and (
        mod.version(trt.__version__) >= mod.version("8.6")
        and mod.version(trt.__version__) < mod.version("8.7")
    ):
        pytest.skip("Broken on 8.6")

    with example as commands:
        for command in commands:
            example.run(command, sandboxed_install_run)


CLI_EXAMPLES = [
    # Run
    Example(["cli", "run", "01_comparing_frameworks"]),
    Example(
        ["cli", "run", "02_comparing_across_runs"],
        artifact_names=["inputs.json", "run_0_outputs.json", "identity.engine"],
    ),
    Example(
        ["cli", "run", "03_generating_a_comparison_script"],
        artifact_names=["compare_trt_onnxrt.py"],
    ),
    Example(
        ["cli", "run", "04_defining_a_tensorrt_network_or_config_manually"],
        artifact_names=["my_define_network.py", "my_create_config.py"],
    ),
    Example(
        ["cli", "run", "05_comparing_with_custom_input_data"],
        artifact_names=["custom_inputs.json"],
    ),
    Example(
        ["cli", "run", "06_comparing_with_custom_output_data"],
        artifact_names=["custom_inputs.json", "custom_outputs.json"],
    ),
    Example(["cli", "run", "07_checking_nan_inf"]),
    pytest.param(
        Example(
            ["cli", "run", "08_adding_precision_constraints"],
            artifact_names=["inputs.json", "golden_outputs.json"],
        ),
        marks=[pytest.mark.slow],
    ),
    # Convert
    Example(
        ["cli", "convert", "01_int8_calibration_in_tensorrt"],
        artifact_names=["identity.engine", "identity_calib.cache"],
    ),
    pytest.param(
        Example(
            ["cli", "convert", "02_deterministic_engine_builds_in_tensorrt"],
            artifact_names=[
                "0.engine",
                "1.engine",
                "timing.cache",
                "timing.cache.lock",
            ],
        ),
        marks=[pytest.mark.serial, pytest.mark.flaky(max_runs=2)],
    ),
    Example(
        ["cli", "convert", "03_dynamic_shapes_in_tensorrt"],
        artifact_names=["dynamic_identity.engine"],
    ),
    Example(
        ["cli", "convert", "04_converting_models_to_fp16"],
        artifact_names=["identity_fp16.onnx", "inputs.json", "outputs_fp32.json"],
    ),
    # Surgeon
    Example(
        ["cli", "surgeon", "01_isolating_subgraphs"], artifact_names=["subgraph.onnx"]
    ),
    Example(["cli", "surgeon", "02_folding_constants"], artifact_names=["folded.onnx"]),
    Example(
        ["cli", "surgeon", "03_modifying_input_shapes"],
        artifact_names=["dynamic_identity.onnx"],
    ),
    Example(
        ["cli", "surgeon", "04_setting_upper_bounds"],
        artifact_names=["modified.onnx", "folded.onnx"],
    ),
    # Debug
    Example(
        ["cli", "debug", "01_debugging_flaky_trt_tactics"],
        artifact_names=["replays", "golden.json"],
    ),
    pytest.param(
        Example(
            ["cli", "debug", "02_reducing_failing_onnx_models"],
            artifact_names=[
                "polygraphy_debug_replay.json",
                "polygraphy_debug_replay_skip_current.json",
                "folded.onnx",
                "inputs.json",
                "layerwise_golden.json",
                "layerwise_inputs.json",
                "initial_reduced.onnx",
                "final_reduced.onnx",
            ],
        ),
        marks=[pytest.mark.slow],
    ),
    # Plugin
    Example(
        ["cli", "plugin", "01_match_and_replace_plugin"],
        artifact_names=["config.yaml", "replaced.onnx"]
    ),
]


@pytest.mark.parametrize("example", CLI_EXAMPLES, ids=lambda case: str(case))
@pytest.mark.script_launch_mode("subprocess")
def test_cli_examples(example, sandboxed_install_run):
    if "02_deterministic_engine_builds_in_tensorrt" in example.path and mod.version(
        trt.__version__
    ) < mod.version("8.7"):
        pytest.skip("Only supported on TensorRT 8.7 and newer")

    if "03_dynamic_shapes_in_tensorrt" in example.path and (
        mod.version(trt.__version__) >= mod.version("8.6")
        and mod.version(trt.__version__) < mod.version("8.7")
    ):
        pytest.skip("Broken on TensorRT 8.6")

    with example as command_blocks:
        for cmd_block in command_blocks:
            example.run(cmd_block, sandboxed_install_run)


CLI_INSPECT_CHECK_EXAMPLES = [
    Example(["cli", "inspect", "01_inspecting_a_tensorrt_network"]),
    Example(
        ["cli", "inspect", "02_inspecting_a_tensorrt_engine"],
        artifact_names=["dynamic_identity.engine"],
    ),
    Example(["cli", "inspect", "03_inspecting_an_onnx_model"]),
    Example(
        ["cli", "inspect", "05_inspecting_inference_outputs"],
        artifact_names=["outputs.json"],
    ),
    Example(
        ["cli", "inspect", "06_inspecting_input_data"], artifact_names=["inputs.json"]
    ),
    Example(
        ["cli", "inspect", "08_inspecting_tensorrt_onnx_support"],
        artifact_names=[
            "polygraphy_capability_dumps/supported_subgraph-nodes-0-1.onnx",
            "polygraphy_capability_dumps/unsupported_subgraph-nodes-2-2.onnx",
            "polygraphy_capability_dumps/supported_subgraph-nodes-3-3.onnx",
            "polygraphy_capability_dumps/results.txt",
            # Remove directory when done
            "polygraphy_capability_dumps",
        ],
    ),
    Example(
        ["cli", "inspect", "07_inspecting_tactic_replays"],
        artifact_names=["replay.json"],
    ),
    Example(
        ["cli", "check", "01_linting_an_onnx_model"], artifact_names=["report.json"]
    ),
    Example(
        ["cli", "inspect", "09_inspecting_tensorrt_static_onnx_support"],
        artifact_names=[
            "polygraphy_capability_dumps/results.txt",
            # Remove directory when done
            "polygraphy_capability_dumps",
        ],
    ),
]

if mod.has_mod("tensorflow"):
    CLI_INSPECT_CHECK_EXAMPLES.append(
        Example(["cli", "inspect", "04_inspecting_a_tensorflow_graph"])
    )


@pytest.mark.parametrize(
    "example", CLI_INSPECT_CHECK_EXAMPLES, ids=lambda case: str(case)
)
@pytest.mark.script_launch_mode("subprocess")
def test_cli_inspect_check_examples(example, sandboxed_install_run):
    if mod.version(trt.__version__) < mod.version("10.0") and (
        "09_inspecting_tensorrt_static_onnx_support" in example.path
    ):
        pytest.skip("Parser features not supported in TRT <10.0.")

    # Last block should be the expected output, and last command should generate it.
    with example as blocks:
        command_blocks, expected_output = blocks[:-1], str(blocks[-1])
        for cmd_block in command_blocks:
            actual_output = example.run(cmd_block, sandboxed_install_run).stdout

    if mod.version(trt.__version__) >= mod.version("9.0") and (
        "01_inspecting_a_tensorrt_network" in example.path
        or "02_inspecting_a_tensorrt_engine" in example.path
    ):
        pytest.skip(
            "Output is different for TRT >=9, this test needs to be updated to account for that. "
        )

    if mod.version(trt.__version__) < mod.version("10.0") and (
        "08_inspecting_tensorrt_onnx_support" in example.path
    ):
        pytest.skip("Output is different for TRT <10.0.")

    print(actual_output)

    actual_lines = actual_output.splitlines()
    # The output for lint is expected to have errors and warnings, so we can't filter them out.
    # The rest of the examples can be pruned of unnecessary lines.
    if "01_linting_an_onnx_model" not in example.path:
        include_line = (
            lambda line: "[I] Loading" not in line
            and "[I] Saving" not in line
            and "[W]" not in line
            and "[E]" not in line
        )
        actual_lines = [line for line in actual_lines if include_line(line)]

    expected_lines = expected_output.splitlines()
    assert len(actual_lines) == len(expected_lines)

    # Indicates lines that may not match exactly
    NON_EXACT_LINE_MARKERS = [
        "---- ",
        "Layer",
        "        Algorithm:",
        "RUNNING",
        "FAILED",
        "[I] Loading ",
    ]

    for index, (actual_line, expected_line) in enumerate(
        zip(actual_lines, expected_lines)
    ):
        # Skip whitespace, and lines that include runner names (since those have timestamps)
        if expected_line.strip() and all(
            [marker not in expected_line for marker in NON_EXACT_LINE_MARKERS]
        ):
            print(f"Checking line {index}: {expected_line}")
            assert actual_line.strip() == expected_line.strip()


DEV_EXAMPLES = [
    Example(["dev", "01_writing_cli_tools"], artifact_names=["data.json"]),
    Example(
        ["dev", "02_extending_polygraphy_run"],
        artifact_names=[
            os.path.join("extension_module", "build"),
            os.path.join("extension_module", "dist"),
            os.path.join("extension_module", "polygraphy_reshape_destroyer.egg-info"),
        ],
    ),
]


@pytest.mark.parametrize("example", DEV_EXAMPLES, ids=lambda case: str(case))
@pytest.mark.script_launch_mode("subprocess")
def test_dev_examples(example, sandboxed_install_run):
    with example as command_blocks:
        for cmd_block in command_blocks:
            example.run(cmd_block, sandboxed_install_run)
