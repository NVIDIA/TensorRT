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
from polygraphy.common import constants
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Inline, Script


class ComparatorRunArgs(BaseArgs):
    def __init__(self, iters=True, write=True):
        self._iters = iters
        self._write = write


    def add_to_parser(self, parser):
        comparator_args = parser.add_argument_group("Comparator inference", "Options for running inference via Comparator.run()")
        if self._iters:
            comparator_args.add_argument("--warm-up", metavar="NUM", help="Number of warm-up runs before timing inference", type=int, default=None)
            comparator_args.add_argument("--use-subprocess", help="Run runners in isolated subprocesses. Cannot be used with a debugger",
                                         action="store_true", default=None)
        if self._write:
            comparator_args.add_argument("--save-input-data", "--save-inputs", help="[EXPERIMENTAL] Path to save inference inputs. The inputs (List[Dict[str, numpy.ndarray]]) "
                                        "will be pickled and saved", default=None, dest="save_inputs")
            comparator_args.add_argument("--save-results", "--save-outputs", help="Path to save results from runners. "
                                         "The results (RunResults) will be pickled and saved", default=None, dest="save_results")


    def parse(self, args):
        self.warm_up = tools_util.get(args, "warm_up")
        self.use_subprocess = tools_util.get(args, "use_subprocess")
        self.save_inputs = tools_util.get(args, "save_inputs")
        self.save_results = tools_util.get(args, "save_results")


    def add_to_script(self, script, data_loader_name):
        script.add_import(imports=["Comparator"], frm="polygraphy.comparator")
        script.add_import(imports=["sys"])

        RESULTS_VAR_NAME = Inline("results")

        comparator_run = Script.invoke("Comparator.run", script.get_runners(), warm_up=self.warm_up,
                                    data_loader=data_loader_name, use_subprocess=self.use_subprocess,
                                    save_inputs_path=self.save_inputs)
        script.append_suffix(Script.format_str("\n# Runner Execution\n{results} = {:}", Inline(comparator_run), results=RESULTS_VAR_NAME))

        if self.save_results:
            G_LOGGER.verbose("Will save runner results to: {:}".format(self.save_results))
            script.add_import(imports=["misc"], frm="polygraphy.util")
            script.append_suffix(Script.format_str("\n# Save results\nmisc.pickle_save({:}, {results})", self.save_results, results=RESULTS_VAR_NAME))

        return RESULTS_VAR_NAME


class ComparatorCompareArgs(BaseArgs):
    def add_to_parser(self, parser):
        comparator_args = parser.add_argument_group("Comparator comparisons", "Options for comparing inference results")
        comparator_args.add_argument("--no-shape-check", help="Disable checking that output shapes match exactly", action="store_true", default=None)
        comparator_args.add_argument("--rtol", metavar="RTOL", help="Relative tolerance for output comparison. See "
                                     "https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html for details. "
                                     "To specify per-output tolerances, use the format: --rtol [<out_name>,]<rtol>. If no output name is provided, "
                                     "the tolerance is used for any outputs not explicitly specified. For example: "
                                     "--rtol 1e-5 out0,1e-4 out1,1e-3",
                                     nargs="+", default=None)
        comparator_args.add_argument("--atol", metavar="ATOL", help="Absolute tolerance for output comparison. See "
                                     "https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html for details. "
                                     "To specify per-output tolerances, use the format: --atol [<out_name>,]<atol>. If no output name is provided, "
                                     "the tolerance is used for any outputs not explicitly specified. For example: "
                                     "--atol 1e-5 out0,1e-4 out1,1e-3",
                                     nargs="+", default=None)
        comparator_args.add_argument("--validate", help="Check outputs for NaNs", action="store_true", default=None)
        comparator_args.add_argument("--load-results", "--load-outputs", help="Path(s) to load results from runners prior to comparing. "
                                     "Each file should be a pickled RunResults", nargs="+", default=[], dest="load_results")
        comparator_args.add_argument("--fail-fast", help="Fail fast (stop comparing after the first failure)", action="store_true", default=None)
        comparator_args.add_argument("--top-k", help="[EXPERIMENTAL] Apply Top-K (i.e. find indices of K largest values) to the outputs before comparing them.",
                                     type=int, default=None)


    def parse(self, args):
        def parse_tol(tol_arg):
            if tol_arg is None:
                return tol_arg

            tol_map = {}
            for output_tol_arg in tol_arg:
                out_name, _, tol = output_tol_arg.rpartition(",")
                tol_map[out_name] = float(tol)
            return tol_map


        self.no_shape_check = tools_util.get(args, "no_shape_check")
        self.rtol = parse_tol(tools_util.get(args, "rtol"))
        self.atol = parse_tol(tools_util.get(args, "atol"))
        self.validate = tools_util.get(args, "validate")
        self.load_results = tools_util.get(args, "load_results")
        self.fail_fast = tools_util.get(args, "fail_fast")
        self.top_k = tools_util.get(args, "top_k")
        # FIXME: This should be a proper dependency from a RunnerArgs
        self.runners = tools_util.get(args, "runners")


    def add_to_script(self, script, results_name):
        if self.load_results:
            G_LOGGER.verbose("Will load runner results from: {:}".format(self.load_results))
            script.add_import(imports=["misc"], frm="polygraphy.util")
            script.append_suffix(Script.format_str("\n# Load results\nfor load_output in {:}:\n{:}{results}.extend(misc.pickle_load(load_output))",
                                                self.load_results, Inline(constants.TAB), results=results_name))

        if self.top_k is not None:
            script.add_import(imports=["PostprocessFunc"], frm="polygraphy.comparator")
            script.append_suffix(Script.format_str("\n# Postprocessing - Apply Top-{top_k}\n{results} = Comparator.postprocess({results}, PostprocessFunc.topk_func(k={top_k}))",
                                                   top_k=self.top_k, results=results_name))

        SUCCESS_VAR_NAME = Inline("success")
        script.append_suffix("\n{success} = True".format(success=SUCCESS_VAR_NAME))

        if len(self.runners) > 1 or self.load_results: # Only do comparisons if there's actually something to compare.
            script.append_suffix("# Accuracy Comparison")

            compare_func_str = Script.invoke_if_nondefault("CompareFunc.basic_compare_func", rtol=self.rtol, atol=self.atol,
                                                        check_shapes=False if self.no_shape_check else None,
                                                        fail_fast=self.fail_fast)
            compare_func = None
            if compare_func_str:
                script.add_import(imports=["CompareFunc"], frm="polygraphy.comparator")
                compare_func = "compare_func"
                script.append_suffix(Script.format_str("{:} = {:}", Inline(compare_func), Inline(compare_func_str)))

            compare_accuracy = Script.invoke("Comparator.compare_accuracy", results_name, compare_func=Inline(compare_func) if compare_func is not None else None,
                                        fail_fast=self.fail_fast)
            script.append_suffix(Script.format_str("{success} &= bool({:})\n", Inline(compare_accuracy), success=SUCCESS_VAR_NAME))
        if self.validate:
            script.append_suffix("# Validation\n{success} &= Comparator.validate({results})\n".format(success=SUCCESS_VAR_NAME, results=results_name))

        return SUCCESS_VAR_NAME
