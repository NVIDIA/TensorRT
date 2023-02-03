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
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import inline, safe


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

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            postprocess (Dict[str, Dict[str, Any]]):
                    Maps postprocessing function names to dictionaries of output names mapped to parameters.
                    For example, this could be something like:
                    ::

                        {"top_k": {"output1": 5, "output2": 6}}
        """
        self.postprocess = args_util.parse_arglist_to_dict(args_util.get(args, "postprocess"))

        postprocess = {}
        topk_key = inline(safe("top_k"))
        if self.postprocess is not None:
            postprocess[topk_key] = {}
            for key, val in self.postprocess.items():
                if not val.startswith("top-"):
                    G_LOGGER.critical(f"Invalid post-processing function: {val}. Note: Valid choices are: ['top-<K>'].")
                k, _, axis = val.partition(",")
                k = int(k.lstrip("top-"))
                if axis:
                    postprocess[topk_key][key] = (k, int(axis.lstrip("axis=")))
                else:
                    postprocess[topk_key][key] = k
        self.postprocess = postprocess

    def add_to_script_impl(self, script, results_name):
        """
        Args:
            results_name (str): The name of the variable containing results from ``Comparator.run()``.

        Returns:
            str:
                    The name of the variable containing the post-processed results.
                    This could be the same as the original name.
        """
        if self.postprocess:
            script.add_import(imports=["PostprocessFunc"], frm="polygraphy.comparator")
            for func, arg in self.postprocess.items():
                script.append_suffix(
                    safe(
                        "\n# Postprocessing\n"
                        "{results} = Comparator.postprocess({results}, PostprocessFunc.{func}({arg}))",
                        arg=arg,
                        func=func,
                        results=results_name,
                    )
                )
        return results_name
