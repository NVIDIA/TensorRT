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
import os
import numpy as np

import pytest
from polygraphy import util
from polygraphy.json import load_json, save_json
from polygraphy.tools.check.subtool.lint import Lint
from tests.models.meta import ONNX_MODELS, MODELS_DIR

# Dict[str, Tuple[model, expected_valid_nodes, expected_invalid_nodes]]
TEST_LINT_CASES = {
    "test_summary": [
        (
            "identity_identity",
            ["onnx_graphsurgeon_node_1", "onnx_graphsurgeon_node_3"],
            [],
        ),
    ],
    "test_onnxrt_parity": [
        "ext_weights",  # valid
        "capability",  # invalid
    ],
    "test_onnx_spec_check": [
        "bad_graph_with_no_name",
        "bad_graph_with_no_import_domains",
        "bad_graph_with_dup_value_info",
    ],
}

ORT_MATMUL_ERROR_MSG = " Incompatible dimensions for matrix multiplication"


class TestLint:
    def run_lint_get_json(self, poly_check, model_path, *args, expect_error=False):
        """
        Helper function to run `polygraphy check lint`, and load the json output saved by the command.
        the json output is saved in a temporary file and retrieved as a dict.
        The schema for the json output that is used for tests here is:
        {
            'summary': {
                'passing': [list of nodes that passed ORT inference check],
                'failing': [list of nodes that failed ORT inference check],
                },
            'lint_entries': [
                { 'level': Lint.Level, 'source': Lint.Source, 'message': str, 'nodes': [list of node names] },
                ...
            ]
        }
        """
        # Run the command and return the json output
        with util.NamedTemporaryFile(suffix=".json") as outpath:
            status = poly_check(
                ["lint", model_path, "-o", outpath.name, *args],
                expect_error=expect_error,
            )
            # load the json file
            output_json = load_json(outpath.name)
        return output_json, status

    def eval_per_entry(self, lint_entries, lambda_check):
        return list(map(lambda_check, lint_entries))

    @pytest.mark.parametrize(
        "case", TEST_LINT_CASES["test_summary"], ids=lambda case: case[0]
    )
    @pytest.mark.script_launch_mode("subprocess")
    def test_summary(self, case, poly_check):
        """
        Basic test to check that nodes are correctly classified as passing or failing
        """
        model_name, expected_passing, expected_failing = case
        output_json, status = self.run_lint_get_json(
            poly_check, ONNX_MODELS[model_name].path
        )
        passing = sorted(output_json["summary"].get("passing", []))
        assert expected_passing == passing  # check that the valid nodes are as expected
        failing = sorted(output_json["summary"].get("failing", []))
        assert (
            expected_failing == failing
        )  # check that the invalid nodes are as expected

    @pytest.mark.script_launch_mode("subprocess")
    def test_duplicate_node_names_caught(self, poly_check):
        """
        Test that duplicate node names are marked as exception
        """
        output_json, _ = self.run_lint_get_json(
            poly_check,
            ONNX_MODELS["bad_graph_with_duplicate_node_names"].path,
            expect_error=True,
        )

        lint_entry = output_json["lint_entries"][0]
        expected_entry = {
            "level": Lint.Level.EXCEPTION.value,
            "nodes": ["identical"],
            "message": "Duplicate node name: 'identical' for nodes with topological IDs: [0, 1] found.",
            "source": Lint.Source.ONNX_GS.value,
        }
        assert lint_entry == expected_entry
        assert "identical" in output_json["summary"]["failing"]

    @pytest.mark.parametrize(
        "model_name", TEST_LINT_CASES["test_onnx_spec_check"], ids=lambda m: m
    )
    @pytest.mark.script_launch_mode("subprocess")
    def test_onnx_spec_check(self, model_name, poly_check):
        """
        Test that basic onnx specification errors are caught by the lint command from the ONNX Checker
        """
        output_json, _ = self.run_lint_get_json(
            poly_check, ONNX_MODELS[model_name].path, expect_error=True
        )

        assert any(  # Make sure that there is atleast 1 entry with level exception and source onnx_checker
            self.eval_per_entry(
                output_json["lint_entries"],
                lambda entry: entry["level"] == Lint.Level.EXCEPTION.value
                and entry["source"] == Lint.Source.ONNX_CHECKER.value,
            )
        )

    @pytest.mark.parametrize(
        "model_name",
        TEST_LINT_CASES["test_onnxrt_parity"],
        ids=lambda model_name: model_name,
    )
    @pytest.mark.script_launch_mode("subprocess")
    def test_onnxrt_parity(self, model_name, poly_check, poly_run):
        """
        Test that `polygraphy check lint` aligns with `polygraphy run --onnxrt`
        in terms of validating the same models.

        When `polygraphy run --onnxrt` fails,
        `polygraphy check lint` is gauranteed to pick an exception from the model.

        This test also validates `--ext` flag usage in `polygraphy check lint` for loading external weights.
        """
        model_path = ONNX_MODELS[model_name].path

        poly_run_exception = None  # whether `poly_run` picked exception

        extra_args_dict = {
            "ext_weights": [
                "--ext",
                os.path.join(MODELS_DIR, "data"),
            ],
        }

        try:  # try to run the model using onnxrt, may fail.
            status = poly_run(
                [model_path, "--onnxrt", *extra_args_dict.get(model_name, [])]
            )
            poly_run_exception = "FAILED" in status.stdout
        except Exception as e:
            poly_run_exception = True

        # now run the model using polygraphy check and check if its a valid model
        _, _ = self.run_lint_get_json(
            poly_check,
            model_path,
            *extra_args_dict.get(model_name, []),
            expect_error=poly_run_exception,  # if poly_run picked exception, expect poly_check to pick exception
        )

    @pytest.mark.script_launch_mode("subprocess")
    def test_parallel_invalid_nodes_caught(self, poly_check):
        """
        Test that the ort inference check codepath works as expected.
        Check that all independent nodes with exceptions are caught.
        Check correct node is identified as invalid, and that the error message contains expected information.
        """
        model_name = "bad_graph_with_parallel_invalid_nodes"
        # Model's graph is as follows:
        # The graph is invalid due to multiple parallel nodes failing.
        # This is is the graph:
        #    A    B    C    D  E    F    G
        #     \  /      \  /    \  /      \
        #    MatMul_0* Add_0*  MatMul_1 NonZero
        #        \        /        \    /
        #         MatMul_2       MatMul_3*
        #               \       /
        #                \     /
        #                Add_1
        #                  |
        #                output
        # The graph is invalid because MatMul_0, Add_0 and MatMul_3 all will fail.
        # MatMul_0 should fail because A and B are not compatible.
        # Add_0 should fail because C and D are not compatible.
        # MatMul_3 should fail because result of MatMul2 and the Data-dependant shape of output of
        # NonZero are not compatible.

        output_json, _ = self.run_lint_get_json(
            poly_check,
            ONNX_MODELS[model_name].path,
            expect_error=True,
        )

        expected_valid_nodes = ["MatMul_1", "NonZero", "cast_to_int64"]
        expected_invalid_dict = {
            "Add_0": " Incompatible dimensions",
            "MatMul_0": ORT_MATMUL_ERROR_MSG,
            "MatMul_3": ORT_MATMUL_ERROR_MSG,
        }

        # Check each node's entries in json and make sure the required error messages are present.
        for bad_node, msg in expected_invalid_dict.items():
            assert bad_node in output_json["summary"]["failing"]

            expected_entry = {
                "level": Lint.Level.EXCEPTION.value,
                "nodes": [bad_node],
                "message": msg,
                "source": Lint.Source.ONNXRUNTIME.value,
            }
            assert expected_entry in output_json["lint_entries"]

        # Check correct summary
        assert sorted(expected_valid_nodes) == sorted(output_json["summary"]["passing"])
        assert sorted(expected_invalid_dict.keys()) == sorted(
            output_json["summary"]["failing"]
        )

    @pytest.mark.parametrize(
        "input_bool",
        [True, False],
    )
    @pytest.mark.script_launch_mode("subprocess")
    def test_data_dependent_errors_caught(self, poly_check, input_bool):
        """
        Test that data-dependent errors are caught.
        The same test also validates custom data-loading of inputs work.

        Behavior: all the invalid nodes should be caught.
        """
        model_name = "bad_graph_conditionally_invalid"
        #                 cond
        #                  |
        #                 If_Node
        #                  |
        #             z (x or y)
        #              \   |
        #               MatMul
        #                  |
        #               output
        # If `cond` is True, then `x` is used, otherwise `y` is used.
        # `x` is compatible with `z`, while `y` is NOT compatible with `z`.
        # Based on the value of `cond`, the graph may be valid or invalid.
        #
        with util.NamedTemporaryFile() as input_file:
            # Create a custom input file with the value of `cond` as `input_bool`
            json_data = [{"cond": np.array((input_bool,))}]
            save_json(json_data, input_file.name)
            output_json, _ = self.run_lint_get_json(
                poly_check,
                ONNX_MODELS[model_name].path,
                "--load-input-data",
                input_file.name,
                expect_error=not input_bool,
            )
        validation_dict = {  # key: input_bool, value: expected output
            True: {
                "passing": ["If_Node", "MatMul"],
                "failing": [],
            },
            False: {
                "passing": ["If_Node"],
                "failing": ["MatMul"],
            },
        }

        # Check that the output is as expected.
        assert sorted(validation_dict[input_bool]["passing"]) == sorted(
            output_json["summary"]["passing"]
        )
        assert sorted(validation_dict[input_bool]["failing"]) == sorted(
            output_json["summary"].get("failing", [])
        )

        if validation_dict[input_bool]["failing"]:  # when input_bool = False
            expected_entry = {
                "level": Lint.Level.EXCEPTION.value,
                "nodes": ["MatMul"],
                "message": ORT_MATMUL_ERROR_MSG,
                "source": Lint.Source.ONNXRUNTIME.value,
            }
            assert expected_entry in output_json["lint_entries"]

    @pytest.mark.script_launch_mode("subprocess")
    def test_custom_op(self, poly_check):
        """
        Test that a custom-op is handled correctly.
        Behavior: a warning is emitted, and the node is marked as passing.
        """
        model_name = "custom_op_node"

        output_json, _ = self.run_lint_get_json(
            poly_check,
            ONNX_MODELS[model_name].path,
            expect_error=False,
        )
        condition = (
            lambda entry: any(
                [
                    substr in entry["message"]
                    for substr in Lint.CUSTOM_OP_EXCEPTION_SUBSTRS
                ]
            )
            and entry["source"] == Lint.Source.ONNXRUNTIME.value
            and entry["level"] == Lint.Level.WARNING.value
        )
        assert any(self.eval_per_entry(output_json["lint_entries"], condition))

        # node should be present in passing list
        assert "polygraphy_unnamed_node_0" in output_json["summary"]["passing"]

    @pytest.mark.script_launch_mode("subprocess")
    def test_multi_level_errors(self, poly_check):
        """
        Test that multi-level errors are handled correctly.
        The model is invalid because of graph-level error (no name) and node-level error (incompatible inputs).
        Behavior: two lint entries are emitted, one from onnx checker and one from onnxruntime.
        """
        model_name = "bad_graph_with_multi_level_errors"

        output_json, _ = self.run_lint_get_json(
            poly_check,
            ONNX_MODELS[model_name].path,
            expect_error=True,
        )
        lint_entries = output_json["lint_entries"]

        # condition for onnx checker entry
        condition_onnx_checker = (
            lambda entry: "Field 'name' of 'graph' is required to be non-empty."
            in entry["message"]
            and entry["source"] == Lint.Source.ONNX_CHECKER.value
            and entry["level"] == Lint.Level.EXCEPTION.value
        )

        # condition for onnxruntime entry
        condition_onnxruntime = (
            lambda entry: ORT_MATMUL_ERROR_MSG in entry["message"]
            and entry["source"] == Lint.Source.ONNXRUNTIME.value
            and entry["level"] == Lint.Level.EXCEPTION.value
        )

        # checks
        assert len(lint_entries) >= 2  # there should be atleast two lint entries
        assert any(
            self.eval_per_entry(lint_entries, condition_onnx_checker)
        )  # condition for onnx checker entry
        assert any(
            self.eval_per_entry(lint_entries, condition_onnxruntime)
        )  # condition for onnxruntime entry

    @pytest.mark.script_launch_mode("subprocess")
    def test_empty_model_warning(self, poly_check):
        """
        Test that an empty model is handled correctly.
        The empty model should trigger a warning about an empty ONNX model.
        Behavior: a lint entry is emitted from onnx_loader with level warning.
        """
        empty_model_name = "empty"

        # Test with empty model
        output_json, _ = self.run_lint_get_json(
            poly_check, ONNX_MODELS[empty_model_name].path, expect_error=False
        )
        lint_entries = output_json["lint_entries"]

        # condition for onnx_loader entry for empty model
        condition = (
            lambda entry: "ONNX model has no nodes" in entry["message"]
            and entry["source"] == Lint.Source.ONNX_LOADER.value
            and entry["level"] == Lint.Level.WARNING.value
        )

        assert len(lint_entries) == 1  # there should be only one lint entry
        assert condition(lint_entries[0])

    @pytest.mark.script_launch_mode("subprocess")
    def test_cleanable_warning(self, poly_check):
        """
        Test that a cleanable graph with unused nodes/inputs is handled correctly.
        Behavior: The cleanable graph should trigger a warning about unused nodes/inputs.
        """
        cleanable_model_name = "cleanable"

        output_json, _ = self.run_lint_get_json(
            poly_check, ONNX_MODELS[cleanable_model_name].path, expect_error=False
        )
        lint_entries = output_json["lint_entries"]

        # condition for onnx_loader entry for empty model
        node_check = (
            lambda entry: "Does not affect outputs, can be removed" in entry["message"]
            and entry["source"] == Lint.Source.ONNX_GS.value
            and entry["level"] == Lint.Level.WARNING.value
            and entry["nodes"] == ["G"]
        )
        inp_check = (
            lambda entry: "Input: 'e' does not affect outputs, can be removed"
            in entry["message"]
            and entry["source"] == Lint.Source.ONNX_GS.value
            and entry["level"] == Lint.Level.WARNING.value
        )

        assert len(lint_entries) == 2
        assert inp_check(lint_entries[0])
        assert node_check(lint_entries[1])

    @pytest.mark.script_launch_mode("subprocess")
    def test_empty_nodes_renaming(self, poly_check):
        """
        Tests that empty nodes are *gauranteed* a unique name while renaming.
        """
        output_json, _ = self.run_lint_get_json(
            poly_check, ONNX_MODELS["renamable"].path, expect_error=False
        )
        names = output_json["summary"]["passing"]
        expected_names = [
            "polygraphy_unnamed_node_0_0",
            "polygraphy_unnamed_node_3",
            "polygraphy_unnamed_node_0_1",
            "polygraphy_unnamed_node_0",
        ]
        assert sorted(names) == sorted(expected_names)
