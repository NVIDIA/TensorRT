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
from polygraphy.tools.script import inline, make_invocable, make_invocable_if_nondefault, safe


@mod.export()
class CompareFuncSimpleArgs(BaseArgs):
    """
    Comparison Function: `simple`: the `CompareFunc.simple` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--no-shape-check",
            help="Disable checking that output shapes match exactly",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--rtol",
            "--rel-tol",
            dest="rtol",
            help="Relative tolerance for output comparison. This is expressed as a percentage of the second set of output values. "
            "For example, a value of 0.01 would check that the first set of outputs is within 1%% of the second. "
            "To specify per-output tolerances, use the format: --rtol [<out_name>:]<rtol>. If no output name is provided, "
            "the tolerance is used for any outputs not explicitly specified. For example: "
            "--rtol 1e-5 out0:1e-4 out1:1e-3. "
            "Note that the default tolerance typically works well for FP32 but may be too strict for lower precisions like FP16 or INT8.",
            nargs="+",
            default=None,
        )
        self.group.add_argument(
            "--atol",
            "--abs-tol",
            dest="atol",
            help="Absolute tolerance for output comparison. "
            "To specify per-output tolerances, use the format: --atol [<out_name>:]<atol>. If no output name is provided, "
            "the tolerance is used for any outputs not explicitly specified. For example: "
            "--atol 1e-5 out0:1e-4 out1:1e-3. "
            "Note that the default tolerance typically works well for FP32 but may be too strict for lower precisions like FP16 or INT8.",
            nargs="+",
            default=None,
        )
        self.group.add_argument(
            "--check-error-stat",
            help="The error statistic to check. "
            "For details on possible values, see the documentation for CompareFunc.simple(). "
            "To specify per-output values, use the format: --check-error-stat [<out_name>:]<stat>. If no output name is provided, "
            "the value is used for any outputs not explicitly specified. For example: "
            "--check-error-stat max out0:mean out1:median",
            nargs="+",
            default=None,
        )
        self.group.add_argument(
            "--infinities-compare-equal",
            help="If set, then any matching +-inf values in outputs will have an absdiff of 0. "
            "Otherwise, by default they will have an absdiff of NaN.",
            action="store_true",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            no_shape_check (bool): Whether to skip shape checks.
            rtol (Dict[str, float]): Per-tensor relative tolerance.
            atol (Dict[str, float]): Per-tensor absolute tolerance.
            check_error_stat (str): The error metric to check.
            infinities_compare_equal (bool): Whether to allow +-inf to compare as equal.
        """
        self.no_shape_check = args_util.get(args, "no_shape_check")
        self.rtol = args_util.parse_arglist_to_dict(args_util.get(args, "rtol"))
        self.atol = args_util.parse_arglist_to_dict(args_util.get(args, "atol"))
        self.check_error_stat = args_util.parse_arglist_to_dict(args_util.get(args, "check_error_stat"))
        self.infinities_compare_equal = args_util.get(args, "infinities_compare_equal")

        # Without this early check, failure would only happen after inference, which is clearly not desirable.
        if self.check_error_stat:
            VALID_CHECK_ERROR_STATS = ["max", "mean", "median", "elemwise"]
            for stat in self.check_error_stat.values():
                if stat not in VALID_CHECK_ERROR_STATS:
                    G_LOGGER.critical(
                        f"Invalid choice for check_error_stat: {stat}.\nNote: Valid choices are: {VALID_CHECK_ERROR_STATS}"
                    )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        compare_func_str = make_invocable_if_nondefault(
            "CompareFunc.simple",
            rtol=self.rtol,
            atol=self.atol,
            check_shapes=False if self.no_shape_check else None,
            fail_fast=self.arg_groups[ComparatorCompareArgs].fail_fast,
            check_error_stat=self.check_error_stat,
            infinities_compare_equal=self.infinities_compare_equal,
        )
        compare_func = None
        if compare_func_str:
            script.add_import(imports=["CompareFunc"], frm="polygraphy.comparator")
            compare_func = inline(safe("compare_func"))
            script.append_suffix(safe("{:} = {:}", compare_func, compare_func_str))

        return compare_func


@mod.export()
class CompareFuncIndicesArgs(BaseArgs):
    """
    Comparison Function: `indices`: the `CompareFunc.indices` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--index-tolerance",
            help="Index tolerance for output comparison. For details on what this means, see the API documentation for `CompareFunc.indices()`. "
            "To specify per-output tolerances, use the format: --index-tolerance [<out_name>:]<index_tol>. If no output name is provided, "
            "the tolerance is used for any outputs not explicitly specified. For example: "
            "--index_tolerance 1 out0:0 out1:3. ",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            index_tolerance (Dict[str, int]): Per-tensor index tolerance.
        """
        self.index_tolerance = args_util.parse_arglist_to_dict(args_util.get(args, "index_tolerance"))

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        compare_func_str = make_invocable(
            "CompareFunc.indices",
            index_tolerance=self.index_tolerance,
            fail_fast=self.arg_groups[ComparatorCompareArgs].fail_fast,
        )
        script.add_import(imports=["CompareFunc"], frm="polygraphy.comparator")
        compare_func = inline(safe("compare_func"))
        script.append_suffix(safe("{:} = {:}", compare_func, compare_func_str))
        return compare_func
