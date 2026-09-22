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
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable, safe


@mod.export()
class ComparatorPostprocessArgs(BaseArgs):
    """
    Comparator Postprocessing: applying postprocessing to outputs.
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--postprocess",
            "--postprocess-func",
            help="Apply post-processing on the specified outputs prior to comparison. "
            "Format: --postprocess [<out_name>:]<func>. If no output name is provided, the function is applied to all outputs. "
            "For example: `--postprocess out0:top-5 out1:top-3` or `--postprocess top-5`. "
            "Available post-processing functions are: {{top-<K>[,axis=<axis>]: Takes the indices of the K highest values along "
            "the specified axis (defaulting to the last axis), where K is an integer. "
            "For example: `--postprocess top-5` or `--postprocess top-5,axis=1`}}",
            nargs="+",
            default=None,
            dest="postprocess",
        )
        self.group.add_argument(
            "--postprocess-func-script",
            help="[EXPERIMENTAL] Path to a Python script that defines a function that can post-process a single iteration result. "
            "This function must have a signature of: `(IterationResult) -> IterationResult`. "
            "For details, see the API documentation for `Comparator.postprocess()`. "
            "If provided, this will override all `--postprocess` options. "
            "By default, Polygraphy looks for a function called `postprocess_outputs`. You can specify a custom function name "
            "by separating it with a colon. For example: `my_custom_script.py:my_func`",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            postprocess (Dict[str, Union[int, Tuple[int, int]]]):
                    Maps output names to Top-K parameters. Each value is an integer K (top-K along
                    the last axis) or a ``(K, axis)`` tuple. For example:
                    ::

                        {"output1": 5, "output2": (3, 1)}
            postprocess_func_script (str):
                    Path to a script defining a custom postprocessing function.
            postprocess_func_name (str):
                    The name of the function in the script that performs postprocessing.
        """
        self.postprocess_func_script, self.postprocess_func_name = (
            args_util.parse_script_and_func_name(
                args_util.get(args, "postprocess_func_script"),
                default_func_name="postprocess_outputs",
            )
        )

        self.postprocess = args_util.parse_arglist_to_dict(
            args_util.get(args, "postprocess")
        )

        raw = self.postprocess
        self.postprocess = {}
        if raw is not None:
            for key, val in raw.items():
                if not val.startswith("top-"):
                    G_LOGGER.critical(
                        f"Invalid post-processing function: {val}. Note: Valid choices are: ['top-<K>']."
                    )
                k, _, axis = val.partition(",")
                k = int(k.lstrip("top-"))
                if axis:
                    self.postprocess[key] = (k, int(axis.lstrip("axis=")))
                else:
                    self.postprocess[key] = k

        if self.postprocess_func_script is not None and self.postprocess:
            G_LOGGER.warning(
                "Argument: '--postprocess/--postprocess-func' will be ignored since '--postprocess-func-script' was provided."
            )

    def make_postprocess_func(self, script):
        """
        Builds an invocation of the configured postprocessing function.

        Returns:
            Optional[str]:
                    An invocation of the postprocessing function (a ``Callable(IterationResult)
                    -> IterationResult``), or ``None`` if no postprocessing was configured.
        """
        if self.postprocess_func_script is not None:
            script.add_import(
                imports=["InvokeFromScript"], frm="polygraphy.backend.common"
            )
            return make_invocable(
                "InvokeFromScript",
                self.postprocess_func_script,
                name=self.postprocess_func_name,
            )
        if self.postprocess:
            script.add_import(
                imports=["TopKPostprocessFunc"], frm="polygraphy.comparator"
            )
            return make_invocable("TopKPostprocessFunc", self.postprocess)
        return None

    def add_to_script_impl(self, script, results_name):
        """
        Args:
            results_name (str): The name of the variable containing results from ``Comparator.run()``.

        Returns:
            str:
                    The name of the variable containing the post-processed results.
                    This could be the same as the original name.
        """
        postprocess_func = self.make_postprocess_func(script)
        if postprocess_func is not None:
            script.append_suffix(
                safe(
                    "\n# Postprocessing\n"
                    "{results} = Comparator.postprocess({results}, {:})",
                    postprocess_func,
                    results=results_name,
                )
            )
        return results_name
