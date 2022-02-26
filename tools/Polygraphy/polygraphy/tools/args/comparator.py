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
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import inline, make_invocable, make_invocable_if_nondefault, safe


@mod.export()
class ComparatorRunArgs(BaseArgs):
    def __init__(self, iters=True, write=True):
        super().__init__()
        self._iters = iters
        self._write = write

    def add_to_parser(self, parser):
        comparator_args = parser.add_argument_group(
            "Comparator inference", "Options for running inference via Comparator.run()"
        )
        if self._iters:
            comparator_args.add_argument(
                "--warm-up",
                metavar="NUM",
                help="Number of warm-up runs before timing inference",
                type=int,
                default=None,
            )
            comparator_args.add_argument(
                "--use-subprocess",
                help="Run runners in isolated subprocesses. Cannot be used with a debugger",
                action="store_true",
                default=None,
            )
        if self._write:
            comparator_args.add_argument(
                "--save-inputs",
                "--save-input-data",
                help="[EXPERIMENTAL] Path to save inference inputs. "
                "The inputs (List[Dict[str, numpy.ndarray]]) will be encoded as JSON and saved",
                default=None,
                dest="save_inputs",
            )
            comparator_args.add_argument(
                "--save-outputs",
                "--save-results",
                help="Path to save results from runners. " "The results (RunResults) will be encoded as JSON and saved",
                default=None,
                dest="save_results",
            )

    def register(self, maker):
        from polygraphy.tools.args.data_loader import DataLoaderArgs

        if isinstance(maker, DataLoaderArgs):
            self.data_loader_args = maker

    def check_registered(self):
        assert self.data_loader_args is not None, "DataLoaderArgs is required for comparator!"

    def parse(self, args):
        self.warm_up = args_util.get(args, "warm_up")
        self.use_subprocess = args_util.get(args, "use_subprocess")
        self.save_inputs = args_util.get(args, "save_inputs")
        self.save_results = args_util.get(args, "save_results")

    def add_to_script(self, script):
        script.add_import(imports=["Comparator"], frm="polygraphy.comparator")

        RESULTS_VAR_NAME = inline(safe("results"))

        comparator_run = make_invocable(
            "Comparator.run",
            script.get_runners(),
            warm_up=self.warm_up,
            data_loader=self.data_loader_args.add_data_loader(script),
            use_subprocess=self.use_subprocess,
            save_inputs_path=self.save_inputs,
        )
        script.append_suffix(safe("\n# Runner Execution\n{results} = {:}", comparator_run, results=RESULTS_VAR_NAME))

        if self.save_results:
            G_LOGGER.verbose("Will save runner results to: {:}".format(self.save_results))
            script.add_import(imports=["util"], frm="polygraphy")
            script.append_suffix(
                safe("\n# Save results\n{results}.save({:})", self.save_results, results=RESULTS_VAR_NAME)
            )

        return RESULTS_VAR_NAME


@mod.export()
class ComparatorCompareArgs(BaseArgs):
    def __init__(self, load=True):
        super().__init__()
        self._load = load

    def add_to_parser(self, parser):
        comparator_args = parser.add_argument_group("Comparator comparisons", "Options for comparing inference results")
        comparator_args.add_argument(
            "--no-shape-check",
            help="Disable checking that output shapes match exactly",
            action="store_true",
            default=None,
        )
        comparator_args.add_argument(
            "--rtol",
            "--rel-tol",
            dest="rtol",
            help="Relative tolerance for output comparison. "
            "To specify per-output tolerances, use the format: --rtol [<out_name>:]<rtol>. If no output name is provided, "
            "the tolerance is used for any outputs not explicitly specified. For example: "
            "--rtol 1e-5 out0:1e-4 out1:1e-3",
            nargs="+",
            default=None,
        )
        comparator_args.add_argument(
            "--atol",
            "--abs-tol",
            dest="atol",
            help="Absolute tolerance for output comparison. "
            "To specify per-output tolerances, use the format: --atol [<out_name>:]<atol>. If no output name is provided, "
            "the tolerance is used for any outputs not explicitly specified. For example: "
            "--atol 1e-5 out0:1e-4 out1:1e-3",
            nargs="+",
            default=None,
        )
        comparator_args.add_argument(
            "--validate", help="Check outputs for NaNs and Infs", action="store_true", default=None
        )
        comparator_args.add_argument(
            "--fail-fast", help="Fail fast (stop comparing after the first failure)", action="store_true", default=None
        )
        comparator_args.add_argument(
            "--top-k",
            help="[EXPERIMENTAL] Apply Top-K (i.e. find indices of K largest values) to the outputs before comparing them."
            "To specify per-output top-k, use the format: --top-k [<out_name>:]<k>. If no output name is provided, "
            "top-k is applied to all outputs. For example: "
            "--top-k out:5",
            nargs="+",
            default=None,
        )
        comparator_args.add_argument(
            "--check-error-stat",
            help="The error statistic to check. "
            "For details on possible values, see the documentation for CompareFunc.simple(). "
            "To specify per-output values, use the format: --check-error-stat [<out_name>:]<stat>. If no output name is provided, "
            "the value is used for any outputs not explicitly specified. For example: "
            "--check-error-stat max out0:mean out1:median",
            nargs="+",
            default=None,
        )

        if self._load:
            comparator_args.add_argument(
                "--load-outputs",
                "--load-results",
                help="Path(s) to load results from runners prior to comparing. "
                "Each file should be a JSON-ified RunResults",
                nargs="+",
                default=[],
                dest="load_results",
            )

    def parse(self, args):
        self.no_shape_check = args_util.get(args, "no_shape_check")
        self.rtol = args_util.parse_dict_with_default(args_util.get(args, "rtol"))
        self.atol = args_util.parse_dict_with_default(args_util.get(args, "atol"))
        self.validate = args_util.get(args, "validate")
        self.load_results = args_util.get(args, "load_results")
        self.fail_fast = args_util.get(args, "fail_fast")
        self.top_k = args_util.parse_dict_with_default(args_util.get(args, "top_k"))
        self.check_error_stat = args_util.parse_dict_with_default(args_util.get(args, "check_error_stat"))
        if self.check_error_stat:
            VALID_CHECK_ERROR_STATS = ["max", "mean", "median", "elemwise"]
            for stat in self.check_error_stat.values():
                if stat not in VALID_CHECK_ERROR_STATS:
                    G_LOGGER.critical(
                        "Invalid choice for check_error_stat: {:}.\n"
                        "Note: Valid choices are: {:}".format(stat, VALID_CHECK_ERROR_STATS)
                    )

        # FIXME: This should be a proper dependency from a RunnerArgs
        self.runners = args_util.get(args, "runners", default=[])

    def add_to_script(self, script, results_name):
        script.add_import(imports=["Comparator"], frm="polygraphy.comparator")

        if self.load_results:
            script.add_import(imports=["util"], frm="polygraphy")
            script.add_import(imports=["RunResults"], frm="polygraphy.comparator")
            script.append_suffix(
                safe(
                    "\n# Load results\nfor load_output in {:}:\n\t{results}.extend(RunResults.load(load_output))",
                    self.load_results,
                    results=results_name,
                )
            )

        if self.top_k is not None:
            script.add_import(imports=["PostprocessFunc"], frm="polygraphy.comparator")
            script.append_suffix(
                safe(
                    "\n# Postprocessing - Apply Top-{top_k}\n"
                    "{results} = Comparator.postprocess({results}, PostprocessFunc.topk_func(k={top_k}))",
                    top_k=self.top_k,
                    results=results_name,
                )
            )

        SUCCESS_VAR_NAME = inline(safe("success"))
        script.append_suffix(safe("\n{success} = True", success=SUCCESS_VAR_NAME))

        if len(self.runners) > 1 or self.load_results:  # Only do comparisons if there's actually something to compare.
            script.append_suffix(safe("# Accuracy Comparison"))

            compare_func_str = make_invocable_if_nondefault(
                "CompareFunc.simple",
                rtol=self.rtol,
                atol=self.atol,
                check_shapes=False if self.no_shape_check else None,
                fail_fast=self.fail_fast,
                check_error_stat=self.check_error_stat,
            )
            compare_func = None
            if compare_func_str:
                script.add_import(imports=["CompareFunc"], frm="polygraphy.comparator")
                compare_func = inline(safe("compare_func"))
                script.append_suffix(safe("{:} = {:}", compare_func, compare_func_str))

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
