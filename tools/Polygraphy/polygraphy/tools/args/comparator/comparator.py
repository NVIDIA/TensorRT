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
from polygraphy import constants, mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.backend.runner_select import RunnerSelectArgs
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.comparator.compare import CompareFuncIndicesArgs, CompareFuncSimpleArgs
from polygraphy.tools.args.comparator.data_loader import DataLoaderArgs
from polygraphy.tools.args.comparator.postprocess import ComparatorPostprocessArgs
from polygraphy.tools.script import inline, make_invocable, safe


@mod.export()
class ComparatorRunArgs(BaseArgs):
    """
    Comparator Inference: running inference via ``Comparator.run()``.

    Depends on:

        - DataLoaderArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--warm-up",
            metavar="NUM",
            help="Number of warm-up runs before timing inference",
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
            help="Path to save inference inputs. "
            "The inputs (List[Dict[str, numpy.ndarray]]) will be encoded as JSON and saved",
            default=None,
            dest="save_inputs_path",
        )
        self.group.add_argument(
            "--save-outputs",
            "--save-results",
            help="Path to save results from runners. " "The results (RunResults) will be encoded as JSON and saved",
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
        """
        self.warm_up = args_util.get(args, "warm_up")
        self.use_subprocess = args_util.get(args, "use_subprocess")
        self.save_inputs_path = args_util.get(args, "save_inputs_path")
        self.save_outputs_path = args_util.get(args, "save_outputs_path")

    def add_to_script_impl(self, script):
        script.add_import(imports=["Comparator"], frm="polygraphy.comparator")

        RESULTS_VAR_NAME = inline(safe("results"))

        comparator_run = make_invocable(
            "Comparator.run",
            script.get_runners(),
            warm_up=self.warm_up,
            data_loader=self.arg_groups[DataLoaderArgs].add_to_script(script),
            use_subprocess=self.use_subprocess,
            save_inputs_path=self.save_inputs_path,
        )
        script.append_suffix(safe("\n# Runner Execution\n{results} = {:}", comparator_run, results=RESULTS_VAR_NAME))

        if self.save_outputs_path:
            G_LOGGER.verbose(f"Will save runner results to: {self.save_outputs_path}")
            script.add_import(imports=["util"], frm="polygraphy")
            script.append_suffix(
                safe("\n# Save results\n{results}.save({:})", self.save_outputs_path, results=RESULTS_VAR_NAME)
            )

        return RESULTS_VAR_NAME


@mod.export()
class ComparatorCompareArgs(BaseArgs):
    """
    Comparator Comparisons: inference output comparisons.

    Depends on:

        - CompareFuncSimpleArgs
        - CompareFuncIndicesArgs
        - RunnerSelectArgs
        - ComparatorPostprocessArgs: if allow_postprocessing == True
    """

    def __init__(self, allow_postprocessing: bool = None):
        """
        Args:
            allow_postprocessing (bool):
                    Whether to post-processing of outputs before comparison.
                    Defaults to True.
        """
        super().__init__()
        self._allow_postprocessing = util.default(allow_postprocessing, True)

    def add_parser_args_impl(self):
        self.group.add_argument("--validate", help="Check outputs for NaNs and Infs", action="store_true", default=None)
        self.group.add_argument(
            "--fail-fast", help="Fail fast (stop comparing after the first failure)", action="store_true", default=None
        )

        self.group.add_argument(
            "--compare",
            "--compare-func",
            help="Name of the function to use to perform comparison. See the API documentation for `CompareFunc` for details. "
            "Defaults to 'simple'. ",
            choices=["simple", "indices"],
            default="simple",
            dest="compare",
        )
        self.group.add_argument(
            "--compare-func-script",
            help="[EXPERIMENTAL] Path to a Python script that defines a function that can compare two iteration results.  "
            "This function must have a signature of: `(IterationResult, IterationResult) -> OrderedDict[str, bool]`. "
            "For details, see the API documentation for `Comparator.compare_accuracy()`. "
            "If provided, this will override all other comparison function options. "
            "By default, Polygraphy looks for a function called `compare_outputs`. You can specify a custom function name "
            "by separating it with a colon. For example: `my_custom_script.py:my_func`",
            default=None,
        )
        self.group.add_argument(
            "--load-outputs",
            "--load-results",
            help="Path(s) to load results from runners prior to comparing. "
            "Each file should be a JSON-ified RunResults",
            nargs="+",
            default=[],
            dest="load_outputs_paths",
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            validate (bool): Whether to run output validation.
            load_outputs_paths (List[str]): Path(s) from which to load outputs.
            fail_fast (bool): Whether to fail fast.
            compare_func (str): The name of the comparison function to use.
            compare_func_script (str): Path to a script defining a custom comparison function.
            compare_func_name (str): The name of the function in the script that runs comparison.
        """
        self.validate = args_util.get(args, "validate")
        self.load_outputs_paths = args_util.get(args, "load_outputs_paths")
        self.fail_fast = args_util.get(args, "fail_fast")

        self.compare_func = args_util.get(args, "compare")

        self.compare_func_script, self.compare_func_name = args_util.parse_script_and_func_name(
            args_util.get(args, "compare_func_script"), default_func_name="compare_outputs"
        )

    def add_to_script_impl(self, script, results_name):
        """
        Args:
            results_name (str): The name of the variable containing results from ``Comparator.run()``.

        Returns:
            str: The name of the variable containing the status of ``Comparator.compare_accuracy()``.
        """
        script.add_import(imports=["Comparator"], frm="polygraphy.comparator")

        if self.load_outputs_paths:
            script.add_import(imports=["util"], frm="polygraphy")
            script.add_import(imports=["RunResults"], frm="polygraphy.comparator")
            script.append_suffix(
                safe(
                    "\n# Load results\nfor load_output in {:}:\n{tab}{results}.extend(RunResults.load(load_output))",
                    self.load_outputs_paths,
                    results=results_name,
                    tab=inline(safe(constants.TAB)),
                )
            )

        if self._allow_postprocessing:
            results_name = self.arg_groups[ComparatorPostprocessArgs].add_to_script(script, results_name)

        SUCCESS_VAR_NAME = inline(safe("success"))
        script.append_suffix(safe("\n{success} = True", success=SUCCESS_VAR_NAME))

        if len(self.arg_groups[RunnerSelectArgs].runners) > 1 or self.load_outputs_paths:
            # Only do comparisons if there's actually something to compare.
            script.append_suffix(safe("# Accuracy Comparison"))

            if self.compare_func_script is not None:
                script.add_import(imports=["InvokeFromScript"], frm="polygraphy.backend.common")
                compare_func = make_invocable("InvokeFromScript", self.compare_func_script, name=self.compare_func_name)
            else:
                compare_func = {
                    "simple": self.arg_groups[CompareFuncSimpleArgs],
                    "indices": self.arg_groups[CompareFuncIndicesArgs],
                }[self.compare_func].add_to_script(script)

            compare_accuracy = make_invocable(
                "Comparator.compare_accuracy", results_name, compare_func=compare_func, fail_fast=self.fail_fast
            )
            script.append_suffix(safe("{success} &= bool({:})\n", compare_accuracy, success=SUCCESS_VAR_NAME))
        if self.validate:
            script.append_suffix(
                safe(
                    "# Validation\n{success} &= Comparator.validate({results}, check_inf=True, check_nan=True)\n",
                    success=SUCCESS_VAR_NAME,
                    results=results_name,
                )
            )

        return SUCCESS_VAR_NAME
