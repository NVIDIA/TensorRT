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

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.backend.runner_select import RunnerSelectArgs
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.comparator.compare import (
    CompareFuncCosineSimilarityArgs,
    CompareFuncIndicesArgs,
    CompareFuncL2Args,
    CompareFuncPerceptualMetricsArgs,
    CompareFuncPsnrArgs,
    CompareFuncSimpleArgs,
    CompareFuncSnrArgs,
)
from polygraphy.tools.args.comparator.data_loader import DataLoaderArgs
from polygraphy.tools.args.comparator.postprocess import ComparatorPostprocessArgs
from polygraphy.tools.script import inline, make_invocable, safe


@mod.export()
class ComparatorRunArgs(BaseArgs):
    """
    Comparator Inference: running inference via ``Comparator.run()``.

    Depends on:

        - DataLoaderArgs
        - RunnerSelectArgs
        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--warm-up",
            metavar="NUM",
            help="Number of warm-up runs before timing inference. Requires --sequential-runners "
            "(not supported in the default streaming mode).",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--use-subprocess",
            help="Run runners in isolated subprocesses. Cannot be used with a debugger",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--save-inputs",
            "--save-input-data",
            help="Path to save inference inputs (read back with --load-inputs). A path with a file "
            "extension saves a single JSON List[Dict[str, numpy.ndarray]]; an extensionless path saves "
            "a directory with one JSON file per iteration (the directory must be empty or not yet exist).",
            default=None,
            dest="save_inputs_path",
        )
        self.group.add_argument(
            "--save-input-blob",
            help="Directory in which to save each inference input tensor as its own raw binary file "
            "(via `numpy.ndarray.tofile()`), one per-iteration subdirectory holding "
            "`<input_name>.bin` files. Defaults to the current directory if no path is given. "
            "Existing files/directories are overwritten.",
            nargs="?",
            const=".",
            default=None,
            dest="save_input_blob_path",
        )
        self.group.add_argument(
            "--save-outputs",
            "--save-results",
            help="Path to save results from runners (read back with --load-outputs). A path with a file "
            "extension saves a single JSON RunResults; an extensionless path saves a directory with one "
            "JSON file per iteration (the directory must be empty or not yet exist).",
            default=None,
            dest="save_outputs_path",
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            warm_up (int): The number of warm-up runs to perform.
            use_subprocess (bool): Whether to run each runner in a subprocess.
            save_inputs_path (str): The path at which to save input data.
            save_outputs_path (str): The path at which to save output data.
            save_input_blob_path (str): The directory in which to save raw input tensor data.
        """
        self.warm_up = args_util.get(args, "warm_up")
        self.use_subprocess = args_util.get(args, "use_subprocess")
        self.save_inputs_path = args_util.get(args, "save_inputs_path")
        self.save_outputs_path = args_util.get(args, "save_outputs_path")
        self.save_input_blob_path = args_util.get(args, "save_input_blob_path")

    def add_to_script_impl(self, script):
        script.add_import(imports=["Comparator"], frm="polygraphy.comparator")

        # No runners: the comparison step gets its data entirely from --load-outputs.
        if not self.arg_groups[RunnerSelectArgs].runners:
            return None

        streaming = self.arg_groups[ComparatorCompareArgs].streaming
        if self.save_outputs_path:
            G_LOGGER.verbose(f"Will save runner results to: {self.save_outputs_path}")

        # One run() call for both modes: streaming=True yields a lazy per-iteration stream, else a
        # materialized RunResults. warm_up/use_subprocess are None in streaming mode (rejected by the
        # CLI), so make_invocable omits them.
        DATA_VAR_NAME = inline(safe("run_stream" if streaming else "results"))
        comparator_run = make_invocable(
            "Comparator.run",
            script.get_runners(),
            warm_up=self.warm_up,
            data_loader=self.arg_groups[DataLoaderArgs].add_to_script(script),
            use_subprocess=self.use_subprocess,
            save_inputs_path=self.save_inputs_path,
            save_outputs_path=self.save_outputs_path,
            save_input_blob_path=self.save_input_blob_path,
            streaming=streaming or None,
        )
        script.append_suffix(
            safe(
                "\n# Runner Execution\n{data} = {:}",
                comparator_run,
                data=DATA_VAR_NAME,
            )
        )
        return DATA_VAR_NAME


@mod.export()
class ComparatorCompareArgs(BaseArgs):
    """
    Comparator Comparisons: inference output comparisons.

    Depends on:

        - CompareFuncSimpleArgs
        - CompareFuncIndicesArgs
        - CompareFuncL2Args
        - CompareFuncCosineSimilarityArgs
        - CompareFuncPsnrArgs
        - CompareFuncSnrArgs
        - CompareFuncPerceptualMetricsArgs
        - RunnerSelectArgs
        - DataLoaderArgs
        - ComparatorRunArgs
        - ComparatorPostprocessArgs: if allow_postprocessing == True
    """

    def __init__(
        self,
        allow_postprocessing: bool = None,
        allow_validate: bool = None,
        allow_load_outputs: bool = None,
        allow_sequential_runners: bool = None,
    ):
        """
        Args:
            allow_postprocessing (bool):
                    Whether to post-processing of outputs before comparison.
                    Defaults to True.
            allow_validate (bool):
                    Whether to allow the ``--validate`` option (checking inference outputs for
                    NaNs/Infs). Defaults to True.
            allow_load_outputs (bool):
                    Whether to allow the ``--load-outputs`` option (loading saved runs to compare
                    against). Defaults to True.
            allow_sequential_runners (bool):
                    Whether to allow the ``--sequential-runners`` option (running each runner to
                    completion instead of streaming). Defaults to True.

        The ``allow_*`` options for running/loading runs are disabled by tools that only operate on
        already-computed comparison results (e.g. ``polygraphy check accuracy``).
        """
        super().__init__()
        self._allow_postprocessing = util.default(allow_postprocessing, True)
        self._allow_validate = util.default(allow_validate, True)
        self._allow_load_outputs = util.default(allow_load_outputs, True)
        self._allow_sequential_runners = util.default(allow_sequential_runners, True)

    def add_parser_args_impl(self):
        # Each ``--compare`` choice maps to the argument group(s) whose comparison functions it
        # emits. "distance_metrics"/"quality_metrics" are aliases expanding into multiple functions.
        self._comparison_func_map = {
            "simple": [self.arg_groups[CompareFuncSimpleArgs]],
            "indices": [self.arg_groups[CompareFuncIndicesArgs]],
            "l2": [self.arg_groups[CompareFuncL2Args]],
            "cosine_similarity": [self.arg_groups[CompareFuncCosineSimilarityArgs]],
            "distance_metrics": [
                self.arg_groups[CompareFuncL2Args],
                self.arg_groups[CompareFuncCosineSimilarityArgs],
            ],
            "psnr": [self.arg_groups[CompareFuncPsnrArgs]],
            "snr": [self.arg_groups[CompareFuncSnrArgs]],
            "quality_metrics": [
                self.arg_groups[CompareFuncPsnrArgs],
                self.arg_groups[CompareFuncSnrArgs],
            ],
            "perceptual_metrics": [self.arg_groups[CompareFuncPerceptualMetricsArgs]],
        }

        self.group.add_argument(
            "--no-shape-check",
            help="Disable checking that output shapes match exactly",
            action="store_true",
            default=None,
        )
        if self._allow_validate:
            self.group.add_argument(
                "--validate",
                help="Check outputs for NaNs and Infs",
                action="store_true",
                default=None,
            )
        self.group.add_argument(
            "--fail-fast",
            help="Fail fast (stop comparing after the first failure)",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--check-average",
            help="Check the average of each metric across all iterations against the threshold "
            "instead of each iteration individually. The per-iteration results are still recorded; "
            "this only changes how pass/fail is determined. "
            "Cannot be combined with --fail-fast. "
            "Not supported by 'indices' or 'simple' with --check-error-stat=elemwise (the default).",
            action="store_true",
            default=None,
        )

        if self._allow_sequential_runners:
            self.group.add_argument(
                "--sequential-runners",
                help="Run each runner to completion one at a time instead of streaming all runners "
                "together per iteration (the default). Holds the whole dataset in memory but only one "
                "runner's device memory at a time. Required for --use-subprocess and --warm-up.",
                action="store_true",
                default=None,
            )

        self.group.add_argument(
            "--compare",
            "--compare-func",
            help="Name(s) of the function(s) to use to perform comparison. "
            "Available: 'simple' (absolute/relative tolerances; the default), 'indices' (top-K "
            "index match), 'l2', 'cosine_similarity', 'psnr', 'snr', and 'perceptual_metrics' "
            "(LPIPS). 'distance_metrics' (= l2 + cosine_similarity) and 'quality_metrics' "
            "(= psnr + snr) are aliases. When multiple are specified, they run in sequence and all "
            "must pass. See the `CompareFunc` API documentation for details. Defaults to 'simple'.",
            choices=list(self._comparison_func_map.keys()),
            default=["simple"],
            nargs="+",
        )
        self.group.add_argument(
            "--compare-func-script",
            help="[EXPERIMENTAL] Path to a Python script defining a comparison function or "
            "`BaseCompareFunc` subclass/instance. The function must have the signature: "
            "`(IterationResult, IterationResult) -> OrderedDict[str, bool]`. "
            "Overrides all other comparison function options. "
            "Looks for `compare_outputs` by default; specify a custom name with a colon: "
            "`my_script.py:my_func`",
            default=None,
        )
        if self._allow_load_outputs:
            self.group.add_argument(
                "--load-outputs",
                "--load-results",
                help="Path(s) to load saved runner results to compare against; each path is a separate "
                "run. In the default streaming mode, a path may be a single file or a *directory* of "
                "per-iteration JSON files (as written by `--save-outputs <dir>`); prefer a directory, which "
                "is streamed one iteration at a time (constant memory). With `--sequential-runners`, a path "
                "must be a single file holding a JSON-ified RunResults, materialized in memory.",
                nargs="+",
                default=[],
                dest="load_outputs_paths",
            )
        self.group.add_argument(
            "--save-accuracy-results",
            help="Path at which to save the accuracy comparison results (the computed per-output "
            "metrics and pass/fail verdicts) as a JSON file. Unlike --save-outputs (which saves the "
            "full output tensors), this saves just the comparison results, which can later be "
            "re-checked against different thresholds with `polygraphy check accuracy` without "
            "re-running inference.",
            default=None,
            dest="save_accuracy_results_path",
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            no_shape_check (bool): Whether to skip shape checks.
            validate (bool): Whether to run output validation.
            load_outputs_paths (List[str]): Path(s) of saved runs to load and compare against.
            save_accuracy_results_path (str):
                    Path at which to save the accuracy comparison results.
            fail_fast (bool): Whether to fail fast.
            compare_funcs (List[str]): The names of the comparison functions to use.
            compare_func_script (str): Path to a script defining a custom comparison function.
            compare_func_name (str): The name of the function in the script that runs comparison.
            check_average (bool): Whether to check averaged metrics instead of per-iteration.
            streaming (bool): Whether to stream all runners together one iteration at a time
                    (the default; the inverse of --sequential-runners).
        """
        self.no_shape_check = args_util.get(args, "no_shape_check")
        # The run/load options may be disabled (see __init__); args_util.get returns the default when
        # the argument was not added, so these still work.
        self.validate = args_util.get(args, "validate")
        self.load_outputs_paths = args_util.get(args, "load_outputs_paths", default=[])
        self.save_accuracy_results_path = args_util.get(
            args, "save_accuracy_results_path"
        )
        self.fail_fast = args_util.get(args, "fail_fast")
        self.check_average = args_util.get(args, "check_average")
        # Streaming is the default; --sequential-runners opts out (absent => default True).
        self.streaming = not args_util.get(args, "sequential_runners", default=False)

        self.compare_funcs = args_util.get(args, "compare")

        # Warn about options provided for comparison functions that weren't selected. One group can
        # back multiple choices (e.g. CompareFuncL2Args backs both "l2" and the "distance_metrics"
        # alias), so map each group to every choice activating it and warn once per unselected group.
        selected_groups = set(self._selected_compare_arg_groups())
        group_to_names = {}
        for name, arg_groups in self._comparison_func_map.items():
            for arg_group in arg_groups:
                group_to_names.setdefault(arg_group, []).append(name)
        for arg_group, names in group_to_names.items():
            if arg_group in selected_groups or arg_group.group is None:
                continue
            for action in arg_group.group._group_actions:
                if args_util.get(args, action.dest) is not None:
                    G_LOGGER.warning(
                        f"Option: {'/'.join(action.option_strings)} is only valid for comparison function(s): {names}. "
                        f"The selected comparison functions are: {self.compare_funcs}, so this option will be ignored."
                    )

        self.compare_func_script, self.compare_func_name = (
            args_util.parse_script_and_func_name(
                args_util.get(args, "compare_func_script"),
                default_func_name="compare_outputs",
            )
        )

    def _validate_check_average(self):
        # NOTE: This validation inspects attributes owned by other argument groups (e.g.
        # ``CompareFuncSimpleArgs.check_error_stat``). The order in which argument groups are
        # parsed is not guaranteed, so we cannot rely on those attributes being populated during
        # our own ``parse_impl``. ``add_to_script`` always runs after every group has parsed, so
        # the validation is deferred to there.
        if not self.check_average:
            return

        if self.fail_fast:
            G_LOGGER.critical("--check-average cannot be combined with --fail-fast.")

        # Only check the cheap common case here so it can fail before inference runs: 'simple'
        # defaults to the elemwise stat, which has no scalar to average. Everything else is left to
        # compare_accuracy's generic metric-field-based rejection at runtime.
        if self.compare_func_script is not None:
            return

        if "simple" in self.compare_funcs:
            # The effective default for outputs without an explicit stat is "elemwise",
            # which has no scalar statistic to average.
            stats = self.arg_groups[CompareFuncSimpleArgs].check_error_stat or {}
            default_stat = stats.get("", "elemwise")
            if default_stat == "elemwise" or "elemwise" in stats.values():
                G_LOGGER.critical(
                    "--check-average is not supported with check_error_stat='elemwise' "
                    "(the default). Specify --check-error-stat max, mean, median, or quantile."
                )

    def _validate_run_args(self):
        # Validation that applies in BOTH streaming and --sequential-runners modes. Like
        # _validate_check_average, it inspects attributes owned by other argument groups, so it is
        # deferred to add_to_script (which runs after every group has parsed).

        # Reject load/save paths that overlap: saving would conflict with the data being loaded.
        # Loading and saving to *distinct* paths is fine. Compare normalized absolute paths (not
        # realpath) so this is a pure-string check, keeping --gen-script deterministic.
        save_paths = [
            path
            for path in (
                self.arg_groups[ComparatorRunArgs].save_inputs_path,
                self.arg_groups[ComparatorRunArgs].save_outputs_path,
                self.arg_groups[ComparatorRunArgs].save_input_blob_path,
            )
            if path
        ]
        load_paths = (
            self.arg_groups[DataLoaderArgs].load_inputs_paths + self.load_outputs_paths
        )
        for save_path in save_paths:
            save_abspath = os.path.abspath(save_path)
            for load_path in load_paths:
                load_abspath = os.path.abspath(load_path)
                # commonpath == save_abspath when the load path is the save path or lives under it.
                try:
                    common = os.path.commonpath([load_abspath, save_abspath])
                except ValueError:
                    continue  # Different Windows drives (no shared root) cannot overlap.
                if common == save_abspath:
                    G_LOGGER.critical(
                        f"The load path '{load_path}' overlaps the save path '{save_path}'. Saving "
                        "would overwrite or conflict with the data being loaded. Use distinct paths "
                        "for loading and saving."
                    )

        if (
            self.load_outputs_paths
            and not self.arg_groups[RunnerSelectArgs].runners
            and (
                not self.arg_groups[DataLoaderArgs].is_using_random_data()
                or self.arg_groups[ComparatorRunArgs].save_inputs_path
                or self.arg_groups[ComparatorRunArgs].save_input_blob_path
            )
        ):
            G_LOGGER.warning(
                "No runners were specified, so inference will not be run and input-related options "
                "(e.g. --load-inputs/--save-inputs/--save-input-blob/--data-loader-script) will be "
                "ignored. Comparison will use only the loaded outputs."
            )

    def _validate_streaming(self):
        # Restrictions that apply ONLY to the default streaming mode; --sequential-runners lifts them.
        if not self.streaming:
            return

        if self.arg_groups[ComparatorRunArgs].use_subprocess:
            G_LOGGER.critical("--use-subprocess requires --sequential-runners.")

        if self.arg_groups[ComparatorRunArgs].warm_up is not None:
            G_LOGGER.critical("--warm-up requires --sequential-runners.")

    def _selected_compare_arg_groups(self):
        # Expand each selected --compare choice into its argument group(s), de-duplicating so
        # overlapping choices (e.g. `l2` and `distance_metrics`) don't run the same function twice.
        # Order is preserved; BaseArgs is identity-hashable.
        seen = set()
        arg_groups = []
        for compare_name in self.compare_funcs:
            for arg_group in self._comparison_func_map[compare_name]:
                if arg_group not in seen:
                    seen.add(arg_group)
                    arg_groups.append(arg_group)
        return arg_groups

    def add_to_script_impl(self, script, results_name):
        """
        Args:
            results_name (str):
                    The variable holding the live run from ``Comparator.run`` -- a materialized
                    ``RunResults`` or, in streaming mode, a per-iteration stream -- or ``None`` if
                    there are no runners (data comes entirely from ``--load-outputs``).

        Returns:
            str: The name of the variable holding the overall pass/fail status.
        """
        self._validate_check_average()
        self._validate_run_args()
        self._validate_streaming()

        script.add_import(imports=["Comparator"], frm="polygraphy.comparator")

        # Assemble the runs to compare into a single list variable: the live run from
        # Comparator.run (if there are runners) plus each --load-outputs golden, loaded lazily
        # (streaming, via load_streaming -- a directory or single file) or materialized
        # (--sequential-runners, via RunResults.load -- a single file; a directory is rejected at
        # runtime, deferred so --gen-script does not inspect disk). Threading the whole list through
        # postprocess/validate/compare_accuracy -- all of which accept a list of runs -- applies
        # those steps uniformly to every run (live and loaded) so they are transformed and checked
        # identically before comparison.
        run_exprs = []
        if results_name is not None:
            run_exprs.append(results_name)
        if self.load_outputs_paths:
            script.add_import(imports=["RunResults"], frm="polygraphy.comparator")
        for load_output in self.load_outputs_paths:
            run_exprs.append(
                make_invocable(
                    "RunResults.load_streaming", load_output, allow_dirs=True
                )
                if self.streaming
                else make_invocable("RunResults.load", load_output)
            )

        runs = inline(safe("runs"))
        script.append_suffix(
            safe("\n# Comparison Inputs\n{runs} = {:}", run_exprs, runs=runs)
        )

        # Post-process every run (e.g. Top-K before an 'indices' comparison), if requested.
        if self._allow_postprocessing:
            runs = self.arg_groups[ComparatorPostprocessArgs].add_to_script(
                script, runs
            )

        # Validate every run (if requested). Streamed data is single-use, so streaming wraps each run
        # as a pass-through that fails (via G_LOGGER.critical) on the first invalid value as
        # compare_accuracy consumes it; materialized runs are validated in a separate pass below.
        if self.validate and self.streaming:
            script.append_suffix(
                safe(
                    "\n# Output Validation\n{runs} = Comparator.validate({runs}, check_inf=True, check_nan=True)",
                    runs=runs,
                )
            )

        # compare_accuracy applies every comparison function and returns an AccuracyResults whose
        # bool() is True only if all passed. It also drives consumption of any live/validation
        # streams (so per-iteration validation and saving run), so it is always emitted even with
        # nothing to compare.
        SUCCESS_VAR_NAME = inline(safe("success"))
        compare_accuracy = make_invocable(
            "Comparator.compare_accuracy",
            runs,
            compare_func=self._build_compare_funcs(script),
            fail_fast=self.fail_fast,
            check_average=self.check_average,
        )
        if self.save_accuracy_results_path:
            # Capture the AccuracyResults so they can be saved before reducing to a bool.
            G_LOGGER.verbose(
                f"Will save accuracy results to: {self.save_accuracy_results_path}"
            )
            accuracy_results = inline(safe("accuracy_results"))
            script.append_suffix(
                safe(
                    "\n# Accuracy Comparison\n{results} = {:}",
                    compare_accuracy,
                    results=accuracy_results,
                )
            )
            script.append_suffix(
                safe(
                    "{results}.save({:})\n{success} = bool({results})",
                    self.save_accuracy_results_path,
                    results=accuracy_results,
                    success=SUCCESS_VAR_NAME,
                )
            )
        else:
            script.append_suffix(
                safe(
                    "\n# Accuracy Comparison\n{success} = bool({:})",
                    compare_accuracy,
                    success=SUCCESS_VAR_NAME,
                )
            )

        if self.validate and not self.streaming:
            # Materialized: validate every run in a separate pass.
            script.append_suffix(
                safe(
                    "{success} &= all(Comparator.validate({runs}, check_inf=True, check_nan=True))\n",
                    success=SUCCESS_VAR_NAME,
                    runs=runs,
                )
            )

        return SUCCESS_VAR_NAME

    def _build_compare_funcs(self, script):
        # The comparison functions to apply: a single script-provided function, or one functor per
        # selected --compare arg group. compare_accuracy applies each to every comparison.
        if self.compare_func_script is not None:
            script.add_import(
                imports=["InvokeFromScript"], frm="polygraphy.backend.common"
            )
            return [
                make_invocable(
                    "InvokeFromScript",
                    self.compare_func_script,
                    name=self.compare_func_name,
                )
            ]
        return [
            arg_group.add_to_script(script)
            for arg_group in self._selected_compare_arg_groups()
        ]

    def build_compare_funcs(self):
        """
        Builds comparison function *instances* (rather than script variables) from the parsed args,
        for tools that compare imperatively rather than via a generated script -- e.g.
        ``polygraphy check accuracy``, which re-checks saved accuracy results against new thresholds.

        This runs the same script generation as ``run`` (``_build_compare_funcs``) and returns the
        resulting instances, so the two paths cannot drift.

        Returns:
            List[BaseCompareFunc]
        """

        def add_to_script(script):
            funcs = self._build_compare_funcs(script)
            return script.add_var(
                safe("{:}", funcs), "compare_funcs", category="Comparison Functions"
            )

        return args_util.run_script(add_to_script)
