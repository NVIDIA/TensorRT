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
from polygraphy.tools.script import (
    make_invocable,
    make_invocable_if_nondefault,
)

#
# NOTE: The classes here are expected to use `None` as the default value for all arguments.
# This is because `ComparatorCompareArgs` will display warnings for any non-`None` arguments
# present in unselected compare func groups. This requirement is enforced by the test.
#


def _add_threshold_compare_func_to_script(
    script, compare_args, class_name, var_name, **threshold_kwargs
):
    """
    Shared script generation for the single-metric compare functions (l2/cosine_similarity/psnr/
    snr/perceptual_metrics): construct ``class_name`` with its threshold kwarg plus the common
    ``check_shapes``/``fail_fast`` from ``ComparatorCompareArgs``, as a uniquely-named variable.
    """
    compare_func_str = make_invocable(
        class_name,
        **threshold_kwargs,
        check_shapes=(False if compare_args.no_shape_check else None),
        fail_fast=compare_args.fail_fast,
    )
    script.add_import(imports=[class_name], frm="polygraphy.comparator")
    return script.add_var(compare_func_str, var_name, category="Comparison Functions")


@mod.export()
class CompareFuncSimpleArgs(BaseArgs):
    """
    Comparison Function: `simple`: the `SimpleCompareFunc` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
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
            "For details on possible values, see the documentation for `SimpleCompareFunc`. "
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
        self.group.add_argument(
            "--save-heatmaps",
            help="[EXPERIMENTAL] Directory in which to save heatmaps of the absolute and relative error. ",
            default=None,
        )
        self.group.add_argument(
            "--show-heatmaps",
            help="[EXPERIMENTAL] Whether to display heatmaps of the absolute and relative error. Defaults to False. ",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--save-error-metrics-plot",
            help="[EXPERIMENTAL] Path to directory to save error metrics plot(s). If set, generates plot of absolute and relative error against reference output magnitude."
            "This directory is created if it does not already exist."
            "This is useful for finding trends in errors, determining whether accuracy failures are just outliers or deeper problems.",
            default=None,
        )
        self.group.add_argument(
            "--show-error-metrics-plot",
            help="[EXPERIMENTAL] Whether to display the error metrics plots. Defaults to False. ",
            action="store_true",
            default=None,
        )
        self.group.add_argument(
            "--error-quantile",
            help="The error quantile to compare. "
            "Float, valid range [0, 1]"
            "To specify per-output values, use the format: --quantile [<out_name>:]<stat>. If no output name is provided, "
            "the value is used for any outputs not explicitly specified. For example: "
            "--error-quantile 0.95 out0:0.8 out1:0.9",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            rtol (Dict[str, float]): Per-tensor relative tolerance.
            atol (Dict[str, float]): Per-tensor absolute tolerance.
            check_error_stat (str): The error metric to check.
            infinities_compare_equal (bool): Whether to allow +-inf to compare as equal.
            save_heatmaps (str): Directory in which to save heatmaps of error.
            show_heatmaps (bool): Whether to display heatmaps of error.
            save_error_metrics_plot (str): Path to store generated error plots.
            show_error_metrics_plot (bool): Whether to display the error metrics plots.
            error_quantile (Dict[str, float]): Per-tensor quantile of error to compute.
        """
        self.rtol = args_util.parse_arglist_to_dict(args_util.get(args, "rtol"))
        self.atol = args_util.parse_arglist_to_dict(args_util.get(args, "atol"))
        self.check_error_stat = args_util.parse_arglist_to_dict(
            args_util.get(args, "check_error_stat")
        )
        self.infinities_compare_equal = args_util.get(args, "infinities_compare_equal")
        self.save_heatmaps = args_util.get(args, "save_heatmaps")
        self.show_heatmaps = args_util.get(args, "show_heatmaps")
        self.save_error_metrics_plot = args_util.get(args, "save_error_metrics_plot")
        self.show_error_metrics_plot = args_util.get(args, "show_error_metrics_plot")
        self.error_quantile = args_util.parse_arglist_to_dict(
            args_util.get(args, "error_quantile")
        )

        # Without this early check, failure would only happen after inference, which is clearly not desirable.
        if self.check_error_stat:
            VALID_CHECK_ERROR_STATS = ["max", "mean", "median", "elemwise", "quantile"]
            for stat in self.check_error_stat.values():
                if stat not in VALID_CHECK_ERROR_STATS:
                    G_LOGGER.critical(
                        f"Invalid choice for check_error_stat: {stat}.\nNote: Valid choices are: {VALID_CHECK_ERROR_STATS}"
                    )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        compare_func_str = make_invocable_if_nondefault(
            "SimpleCompareFunc",
            rtol=self.rtol,
            atol=self.atol,
            check_shapes=(
                False if self.arg_groups[ComparatorCompareArgs].no_shape_check else None
            ),
            fail_fast=self.arg_groups[ComparatorCompareArgs].fail_fast,
            check_error_stat=self.check_error_stat,
            infinities_compare_equal=self.infinities_compare_equal,
            save_heatmaps=self.save_heatmaps,
            show_heatmaps=self.show_heatmaps,
            save_error_metrics_plot=self.save_error_metrics_plot,
            show_error_metrics_plot=self.show_error_metrics_plot,
            error_quantile=self.error_quantile,
        )
        compare_func = None
        if compare_func_str:
            script.add_import(
                imports=["SimpleCompareFunc"], frm="polygraphy.comparator"
            )
            compare_func = script.add_var(
                compare_func_str, "simple_compare_func", category="Comparison Functions"
            )

        return compare_func


@mod.export()
class CompareFuncIndicesArgs(BaseArgs):
    """
    Comparison Function: `indices`: the `IndicesCompareFunc` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--index-tolerance",
            help="Index tolerance for output comparison. For details on what this means, see the API documentation for `IndicesCompareFunc`. "
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
        self.index_tolerance = args_util.parse_arglist_to_dict(
            args_util.get(args, "index_tolerance")
        )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        compare_func_str = make_invocable(
            "IndicesCompareFunc",
            index_tolerance=self.index_tolerance,
            fail_fast=self.arg_groups[ComparatorCompareArgs].fail_fast,
        )
        script.add_import(imports=["IndicesCompareFunc"], frm="polygraphy.comparator")
        return script.add_var(
            compare_func_str, "indices_compare_func", category="Comparison Functions"
        )


@mod.export()
class CompareFuncL2Args(BaseArgs):
    """
    Comparison Function: `l2`: the `L2CompareFunc` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--l2-threshold",
            "--l2-tolerance",
            dest="l2_threshold",
            help="L2 norm threshold for output comparison. "
            "To specify per-output thresholds, use the format: --l2-threshold [<out_name>:]<threshold>. "
            "If no output name is provided, the threshold is used for any outputs not explicitly specified. "
            "For example: --l2-threshold 1e-5 out0:1e-4 out1:1e-3",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            l2_threshold (Dict[str, float]): Per-tensor L2 norm threshold.
        """
        self.l2_threshold = args_util.parse_arglist_to_dict(
            args_util.get(args, "l2_threshold")
        )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        return _add_threshold_compare_func_to_script(
            script,
            self.arg_groups[ComparatorCompareArgs],
            "L2CompareFunc",
            "l2_compare_func",
            l2_threshold=self.l2_threshold,
        )


@mod.export()
class CompareFuncCosineSimilarityArgs(BaseArgs):
    """
    Comparison Function: `cosine_similarity`: the `CosineSimilarityCompareFunc` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--cosine-similarity-threshold",
            dest="cosine_similarity_threshold",
            help="Minimum cosine similarity required for outputs to match. "
            "To specify per-output values, use the format: --cosine-similarity-threshold [<out_name>:]<threshold>. "
            "If no output name is provided, the value is used for any outputs not explicitly specified. "
            "For example: --cosine-similarity-threshold 0.997 out0:0.999 out1:0.95",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            cosine_similarity_threshold (Dict[str, float]): Per-tensor cosine similarity thresholds.
        """
        self.cosine_similarity_threshold = args_util.parse_arglist_to_dict(
            args_util.get(args, "cosine_similarity_threshold")
        )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        return _add_threshold_compare_func_to_script(
            script,
            self.arg_groups[ComparatorCompareArgs],
            "CosineSimilarityCompareFunc",
            "cosine_similarity_compare_func",
            cosine_similarity_threshold=self.cosine_similarity_threshold,
        )


@mod.export()
class CompareFuncPsnrArgs(BaseArgs):
    """
    Comparison Function: `psnr`: the `PsnrCompareFunc` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--psnr-threshold",
            "--psnr-tolerance",
            dest="psnr_threshold",
            help="Minimum PSNR required for outputs to match. "
            "To specify per-output values, use the format: --psnr-threshold [<out_name>:]<psnr>. "
            "If no output name is provided, the value is used for any outputs not explicitly specified. "
            "For example: --psnr-threshold 30 out0:40 out1:25",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            psnr_threshold (Dict[str, float]): Per-tensor PSNR threshold.
        """
        self.psnr_threshold = args_util.parse_arglist_to_dict(
            args_util.get(args, "psnr_threshold")
        )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        return _add_threshold_compare_func_to_script(
            script,
            self.arg_groups[ComparatorCompareArgs],
            "PsnrCompareFunc",
            "psnr_compare_func",
            psnr_threshold=self.psnr_threshold,
        )


@mod.export()
class CompareFuncSnrArgs(BaseArgs):
    """
    Comparison Function: `snr`: the `SnrCompareFunc` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--snr-threshold",
            "--snr-tolerance",
            dest="snr_threshold",
            help="Minimum SNR required for outputs to match. "
            "To specify per-output values, use the format: --snr-threshold [<out_name>:]<snr>. "
            "If no output name is provided, the value is used for any outputs not explicitly specified. "
            "For example: --snr-threshold 20 out0:30 out1:15",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            snr_threshold (Dict[str, float]): Per-tensor SNR threshold.
        """
        self.snr_threshold = args_util.parse_arglist_to_dict(
            args_util.get(args, "snr_threshold")
        )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        return _add_threshold_compare_func_to_script(
            script,
            self.arg_groups[ComparatorCompareArgs],
            "SnrCompareFunc",
            "snr_compare_func",
            snr_threshold=self.snr_threshold,
        )


@mod.export()
class CompareFuncPerceptualMetricsArgs(BaseArgs):
    """
    Comparison Function: `perceptual_metrics`: the `PerceptualMetricsCompareFunc` comparison function.

    Depends on:

        - ComparatorCompareArgs
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--lpips-threshold",
            "--lpips-tolerance",
            dest="lpips_threshold",
            help="Maximum LPIPS score allowed for outputs to match. "
            "To specify per-output values, use the format: --lpips-threshold [<out_name>:]<threshold>. "
            "If no output name is provided, the value is used for any outputs not explicitly specified. "
            "For example: --lpips-threshold 0.1 out0:0.05 out1:0.2",
            nargs="+",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            lpips_threshold (Dict[str, float]): Per-tensor LPIPS thresholds.
        """
        self.lpips_threshold = args_util.parse_arglist_to_dict(
            args_util.get(args, "lpips_threshold")
        )

    def add_to_script_impl(self, script):
        from polygraphy.tools.args.comparator.comparator import ComparatorCompareArgs

        return _add_threshold_compare_func_to_script(
            script,
            self.arg_groups[ComparatorCompareArgs],
            "PerceptualMetricsCompareFunc",
            "perceptual_metrics_compare_func",
            lpips_threshold=self.lpips_threshold,
        )
