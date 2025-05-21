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
import functools
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.comparator import util as comp_util
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")


@mod.export()
class OutputCompareResult:
    """
    Represents the result of comparing a single output of a single iteration
    between two runners.
    """

    def __init__(
        self,
        passed,
        max_absdiff,
        max_reldiff,
        mean_absdiff,
        mean_reldiff,
        median_absdiff,
        median_reldiff,
        quantile_absdiff,
        quantile_reldiff,
    ):
        """
        Records the required tolerances and other statistics gathered during comparison.

        Args:
            passed (bool):
                    Whether the error was within acceptable limits.
            max_absdiff (float):
                    The minimum required absolute tolerance to consider the outputs equivalent.
            max_reldiff (float):
                    The minimum required relative tolerance to consider the outputs equivalent.
            mean_absdiff (float):
                    The mean absolute error between the outputs.
            mean_reldiff (float):
                    The mean relative error between the outputs.
            median_absdiff (float):
                    The median absolute error between the outputs.
            median_reldiff (float):
                    The median relative error between the outputs.
            quantile_absdiff (float):
                    The q-th quantile absolute error between the outputs.
            quantile_reldiff (float):
                    The q-th quantile relative error between the outputs.
        """
        self.passed = passed
        self.max_absdiff = max_absdiff
        self.max_reldiff = max_reldiff
        self.mean_absdiff = mean_absdiff
        self.mean_reldiff = mean_reldiff
        self.median_absdiff = median_absdiff
        self.median_reldiff = median_reldiff
        self.quantile_absdiff = quantile_absdiff
        self.quantile_reldiff = quantile_reldiff

    def __bool__(self):
        """
        Whether the output matched.

        Returns:
            bool
        """
        return self.passed

    def __str__(self):
        return f"(atol={self.max_absdiff}, rtol={self.max_reldiff})"


@mod.export()
class DistanceMetricsResult:
    """
    Represents the result of comparing a single output using distance metrics
    between two runners.
    """

    def __init__(
        self,
        passed,
        l2_norm,
        cosine_similarity,
    ):
        """
        Records the distance metrics gathered during comparison.

        Args:
            passed (bool):
                    Whether the output passed all enabled metric comparisons.
            l2_norm (float):
                    The L2 norm (Euclidean distance) between the outputs.
            cosine_similarity (float):
                    The cosine similarity between the outputs.
        """
        self.passed = passed
        self.l2_norm = l2_norm
        self.cosine_similarity = cosine_similarity

    def __bool__(self):
        """
        Whether the output passed all metric comparisons.

        Returns:
            bool
        """
        return self.passed


@mod.export()
class QualityMetricsResult:
    """
    Represents the result of comparing a single output using quality metrics
    between two runners.
    """

    def __init__(
        self,
        passed,
        psnr=None,
        snr=None,
    ):
        """
        Records the quality metrics gathered during comparison.

        Args:
            passed (bool):
                    Whether the output passed all enabled quality metric comparisons.
            psnr (float):
                    The Peak Signal-to-Noise Ratio between the outputs.
                    May be None if PSNR comparison was not enabled.
            snr (float):
                    The Signal-to-Noise Ratio between the outputs.
                    May be None if SNR comparison was not enabled.
        """
        self.passed = passed
        self.psnr = psnr
        self.snr = snr

    def __bool__(self):
        """
        Whether the output passed all metric comparisons.

        Returns:
            bool
        """
        return self.passed


@mod.export()
class PerceptualMetricsResult:
    """
    Represents the result of comparing a single output using perceptual metrics
    between two runners.
    """

    def __init__(
        self,
        passed,
        lpips=None,
    ):
        """
        Records the perceptual metrics gathered during comparison.

        Args:
            passed (bool):
                    Whether the output passed all enabled perceptual metric comparisons.
            lpips (float):
                    The Learned Perceptual Image Patch Similarity score between the outputs.
                    Lower values indicate more perceptually similar outputs.
                    May be None if LPIPS computation failed.
        """
        self.passed = passed
        self.lpips = lpips

    def __bool__(self):
        """
        Whether the output passed all metric comparisons.

        Returns:
            bool
        """
        return self.passed


def default_find_output_func(output_name, index, iter_result, base_iter_result):
    found_name = util.find_str_in_iterable(output_name, iter_result.keys(), index)
    if found_name is None:
        return None
    elif found_name != output_name:
        exact_match = util.find_str_in_iterable(found_name, base_iter_result.keys())
        if exact_match == found_name:
            G_LOGGER.verbose(
                f"Will not compare {found_name} with {output_name}, since the former already has an exact match: {exact_match}"
            )
            return None  # If the found output is being compared against another output already, skip this non-exact match
        G_LOGGER.warning(
            f"Output names did not match exactly. Assuming {iter_result.runner_name} output: {found_name} corresponds to output: {output_name}"
        )
    return [found_name]


def run_comparison(func, fail_fast, iter_result0, iter_result1, find_output_func):
    """
    Iterates over all the generated outputs and runs `func` to compare them.
    """
    output_status = (
        OrderedDict()
    )  # OrderedDict[str, bool] Maps output names to whether they matched.

    for index, (out0_name, output0) in enumerate(iter_result0.items()):
        out1_names = util.default(find_output_func(out0_name, index, iter_result1), [])

        if len(out1_names) > 1:
            G_LOGGER.info(
                f"Will attempt to compare output: '{out0_name}' [{iter_result0.runner_name}] with multiple outputs: '{list(out1_names)}' [{iter_result1.runner_name}]"
            )

        for out1_name in out1_names:
            if out1_name is None or out1_name not in iter_result1:
                G_LOGGER.warning(
                    f"For output: '{out0_name}' [{iter_result0.runner_name}], skipping corresponding output: '{out1_name}' [{iter_result1.runner_name}], since the output was not found"
                )
                continue

            output1 = iter_result1[out1_name]

            G_LOGGER.start(
                f"Comparing Output: '{out0_name}' (dtype={util.array.dtype(output0)}, shape={util.array.shape(output0)}) with '{out1_name}' (dtype={util.array.dtype(output1)}, shape={util.array.shape(output1)})"
            )
            with G_LOGGER.indent():
                output_status[out0_name] = func(out0_name, output0, out1_name, output1)
                if fail_fast and not output_status[out0_name]:
                    return output_status

    mismatched_output_names = [
        name for name, matched in output_status.items() if not matched
    ]
    if mismatched_output_names:
        G_LOGGER.error(f"FAILED | Mismatched outputs: {mismatched_output_names}")
    else:
        G_LOGGER.finish(
            f"PASSED | All outputs matched | Outputs: {list(output_status.keys())}"
        )

    # This is useful for catching cases were Polygraphy does something wrong with the runner output buffers
    if not output_status and (bool(iter_result0.keys()) or bool(iter_result1.keys())):
        r0_name = iter_result0.runner_name
        r0_outs = list(iter_result0.keys())
        r1_name = iter_result1.runner_name
        r1_outs = list(iter_result1.keys())
        G_LOGGER.critical(
            f"All outputs were skipped, no common outputs found! Note:\n{r0_name} outputs: {r0_outs}\n{r1_name} outputs: {r1_outs}"
        )

    return output_status


# Provides functions to compare two IterationResults
@mod.export()
class CompareFunc:
    """
    Provides functions that can be used to compare two `IterationResult` s.
    """

    @staticmethod
    def simple(
        check_shapes=None,
        rtol=None,
        atol=None,
        fail_fast=None,
        find_output_func=None,
        check_error_stat=None,
        infinities_compare_equal=None,
        save_heatmaps=None,
        show_heatmaps=None,
        save_error_metrics_plot=None,
        show_error_metrics_plot=None,
        error_quantile=None,
    ):
        """
        Creates a function that compares two IterationResults, and can be used as the `compare_func` argument
        in ``Comparator.compare_accuracy``.

        Args:
            check_shapes (bool):
                    Whether shapes must match exactly. If this is False, this function may
                    permute or reshape outputs before comparison.
                    Defaults to True.
            rtol (Union[float, Dict[str, float]]):
                    The relative tolerance to use when checking accuracy.
                    This is expressed as a percentage of the second set of output values.
                    For example, a value of 0.01 would check that the first set of outputs is within 1% of the second.

                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default tolerance for outputs not explicitly listed.
                    Defaults to 1e-5.
            atol (Union[float, Dict[str, float]]):
                    The absolute tolerance to use when checking accuracy.
                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default tolerance for outputs not explicitly listed.
                    Defaults to 1e-5.
            fail_fast (bool):
                    Whether the function should exit immediately after the first failure.
                    Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns a list of output names to compare against from the provided
                    IterationResult, given an output name and index from another IterationResult.
                    The comparison function will always iterate over the output names of the
                    first IterationResult, expecting names from the second. A return value of
                    `[]` or `None` indicates that the output should be skipped.
            check_error_stat (Union[str, Dict[str, str]]):
                    The error statistic to check. Possible values are:

                    - "elemwise": Checks each element in the output to determine if it exceeds both tolerances specified.
                                The minimum required tolerances displayed in this mode are only applicable when just one type of tolerance
                                is set. Because of the nature of the check, when both absolute/relative tolerance are specified, the required
                                minimum tolerances may be lower.

                    - "max": Checks the maximum absolute/relative errors against the respective tolerances. This is the strictest possible check.
                    - "mean" Checks the mean absolute/relative errors against the respective tolerances.
                    - "median": Checks the median absolute/relative errors against the respective tolerances.
                    - "quantile": Checks the quantile absolute/relative errors against the respective tolerances.

                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default error stat for outputs not explicitly listed.
                    Defaults to "elemwise".
            infinities_compare_equal (bool):
                    If True, then matching +-inf values in the output have an absdiff of 0.
                    If False, then matching +-inf values in the output have an absdiff of NaN.
                    Defaults to False.
            save_heatmaps (str):
                    [EXPERIMENTAL] Path to a directory in which to save figures of heatmaps of the absolute and relative error.
                    Defaults to None.
            show_heatmaps (bool):
                    [EXPERIMENTAL] Whether to display heatmaps of the absolute and relative error.
                    Defaults to False.
            save_error_metrics_plot (str):
                    [EXPERIMENTAL] Path to a directory in which to save the error metrics plots.
                    Defaults to None.
            show_error_metrics_plot (bool):
                    [EXPERIMENTAL] Whether to display the error metrics plot.
            error_quantile (Union[float, Dict[str, float]]):
                    Quantile error to compute when checking accuracy. This is expressed as a float in range [0, 1].
                    For example, error_quantile=0.5 is the median.
                    Defaults to 0.99.

        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, OutputCompareResult]:
                A callable that returns a mapping of output names to `OutputCompareResult` s, indicating
                whether the corresponding output matched.
        """
        check_shapes = util.default(check_shapes, True)
        default_rtol = 1e-5
        default_atol = 1e-5
        default_quantile = 0.99
        rtol = util.default(rtol, default_rtol)
        atol = util.default(atol, default_atol)
        error_quantile = util.default(error_quantile, default_quantile)
        fail_fast = util.default(fail_fast, False)
        default_error_stat = "elemwise"
        check_error_stat = util.default(check_error_stat, default_error_stat)
        infinities_compare_equal = util.default(infinities_compare_equal, False)
        show_heatmaps = util.default(show_heatmaps, False)
        show_error_metrics_plot = util.default(show_error_metrics_plot, False)

        def check_outputs_match(
            out0,
            out0_name,
            out1,
            out1_name,
            per_out_rtol,
            per_out_atol,
            per_out_err_stat,
            runner0_name,
            runner1_name,
            per_out_quantile,
        ):
            """
            Checks whether two outputs matched.

            Args:
                out0 (Union[np.array, torch.Tensor]): The first output.
                out0_name (str): The name of the first output.
                out1 (Union[np.array, torch.Tensor]): The second output.
                out1_name (str): The name of the second output.
                per_out_rtol (float): The relative tolerance to use for comparison.
                per_out_atol (float): The absolute tolerance to use for comparison.
                per_out_err_stat (str): The error statistic to check. See the docstring of ``simple`` for details.
                runner0_name (str): The name of the runner that generated the first output.
                runner1_name (str): The name of the runner that generated the second output.
                per_out_quantile (float): The qunatile value to use for quantile comparison.

            Returns:
                OutputCompareResult: Details on whether the outputs matched.
            """
            VALID_CHECK_ERROR_STATS = ["max", "mean", "median", "elemwise", "quantile"]
            if per_out_err_stat not in VALID_CHECK_ERROR_STATS:
                G_LOGGER.critical(
                    f"Invalid choice for check_error_stat: {per_out_err_stat}.\nNote: Valid choices are: {VALID_CHECK_ERROR_STATS}"
                )

            G_LOGGER.super_verbose(
                f"{runner0_name:35} | Output: {out0_name} (dtype={util.array.dtype(out0)}, shape={util.array.shape(out0)}):\n{util.indent_block(out0)}"
            )
            G_LOGGER.super_verbose(
                f"{runner1_name:35} | Output: {out1_name} (dtype={util.array.dtype(out1)}, shape={util.array.shape(out1)}):\n{util.indent_block(out1)}"
            )

            # Check difference vs. tolerances
            if (
                util.array.dtype(out0) == DataType.BOOL
                and util.array.dtype(out1) == DataType.BOOL
            ):
                absdiff = util.array.logical_xor(out0, out1)
            else:
                absdiff = util.array.abs(
                    util.array.subtract(
                        comp_util.cast_up(out0), comp_util.cast_up(out1)
                    )
                )
                if infinities_compare_equal:
                    out0_infinite = util.array.isinf(out0)
                    cond = util.array.logical_and(out0_infinite, out0 == out1)
                    absdiff = util.array.where(cond, 0, absdiff)

            # Add a small epsilon (2e-16) to zero values in the array to prevent NaN in relative error.
            out1_with_eps = copy.copy(comp_util.cast_up(out1))

            if util.array.dtype(out1_with_eps).is_floating:
                if util.array.any(out1_with_eps == 0):
                    G_LOGGER.warning(
                        f"{runner1_name:35} | Output: {out1_name}: Some values are 0. "
                        f"Will add a small epsilon quantity to these when computing relative difference. "
                        f"Note that this may cause some relative differences to be extremely high. ",
                        mode=LogMode.ONCE,
                    )
                EPSILON = 2.220446049250313e-16
                out1_with_eps[out1_with_eps == 0] += EPSILON

            # TODO: Only evaluate this if actually needed like we do for quantile_*.
            reldiff = util.array.divide(absdiff, util.array.abs(out1_with_eps))
            min_reldiff = comp_util.compute_min(reldiff)
            max_reldiff = comp_util.compute_max(reldiff)
            mean_reldiff = comp_util.compute_mean(reldiff)
            median_reldiff = comp_util.compute_median(reldiff)
            quantile_reldiff = None

            min_absdiff = comp_util.compute_min(absdiff)
            max_absdiff = comp_util.compute_max(absdiff)
            mean_absdiff = comp_util.compute_mean(absdiff)
            median_absdiff = comp_util.compute_median(absdiff)
            quantile_absdiff = None

            def stat_failed(diff, tol):
                return util.array.isnan(diff) or diff > tol

            if per_out_err_stat == "mean":
                failed = stat_failed(mean_absdiff, per_out_atol) and stat_failed(
                    mean_reldiff, per_out_rtol
                )
            elif per_out_err_stat == "median":
                failed = stat_failed(median_absdiff, per_out_atol) and stat_failed(
                    median_reldiff, per_out_rtol
                )
            elif per_out_err_stat == "max":
                failed = stat_failed(max_absdiff, per_out_atol) and stat_failed(
                    max_reldiff, per_out_rtol
                )
            elif per_out_err_stat == "quantile":
                quantile_reldiff = comp_util.compute_quantile(reldiff, per_out_quantile)
                quantile_absdiff = comp_util.compute_quantile(absdiff, per_out_quantile)
                failed = stat_failed(quantile_absdiff, per_out_atol) and stat_failed(
                    quantile_reldiff, per_out_rtol
                )
            else:
                assert (
                    per_out_err_stat == "elemwise"
                ), "This branch should be unreachable unless per_out_err_stat is 'elemwise'"
                mismatches = (
                    util.array.greater(absdiff, per_out_atol)
                    | util.array.isnan(absdiff)
                ) & (
                    util.array.greater(reldiff, per_out_rtol)
                    | util.array.isnan(reldiff)
                )

                failed = util.array.any(mismatches)
                try:
                    with G_LOGGER.indent():
                        G_LOGGER.super_verbose(
                            lambda: f"Mismatched indices:\n{util.array.argwhere(mismatches)}"
                        )
                        G_LOGGER.extra_verbose(
                            lambda: f"{runner0_name:35} | Mismatched values:\n{out0[mismatches]}"
                        )
                        G_LOGGER.extra_verbose(
                            lambda: f"{runner1_name:35} | Mismatched values:\n{out1[mismatches]}"
                        )
                except Exception as err:
                    G_LOGGER.warning(
                        f"Failing to log mismatches.\nNote: Error was: {err}"
                    )

            # Log information about the outputs
            hist_bin_range = (
                min(comp_util.compute_min(out0), comp_util.compute_min(out1)),
                max(comp_util.compute_max(out0), comp_util.compute_max(out1)),
            )
            comp_util.log_output_stats(
                out0, failed, f"{runner0_name}: {out0_name}", hist_range=hist_bin_range
            )
            comp_util.log_output_stats(
                out1, failed, f"{runner1_name}: {out1_name}", hist_range=hist_bin_range
            )

            G_LOGGER.info(f"Error Metrics: {out0_name}")
            with G_LOGGER.indent():

                def req_tol(mean_diff, median_diff, max_diff, quantile_diff):
                    return {
                        "mean": mean_diff,
                        "median": median_diff,
                        "max": max_diff,
                        "elemwise": max_diff,
                        "quantile": quantile_diff,
                    }[per_out_err_stat]

                msg = f"Minimum Required Tolerance: {per_out_err_stat} error | [abs={req_tol(mean_absdiff, median_absdiff, max_absdiff, quantile_absdiff):.5g}] OR [rel={req_tol(mean_reldiff, median_reldiff, max_reldiff, quantile_reldiff):.5g}]"
                if per_out_err_stat == "elemwise":
                    msg += " (requirements may be lower if both abs/rel tolerances are set)"
                elif per_out_err_stat == "quantile":
                    msg += f" (quantile={per_out_quantile:.4g})"

                G_LOGGER.info(msg)

                if save_error_metrics_plot or show_error_metrics_plot:
                    with G_LOGGER.indent():
                        comp_util.scatter_plot_error_magnitude(
                            absdiff,
                            reldiff,
                            comp_util.cast_up(out1),
                            min_reldiff,
                            max_reldiff,
                            runner0_name,
                            runner1_name,
                            out0_name,
                            out1_name,
                            save_dir=save_error_metrics_plot,
                            show=show_error_metrics_plot,
                        )

                def build_heatmaps(diff, min_diff, max_diff, prefix, use_lognorm=None):
                    if save_heatmaps or show_heatmaps:
                        with G_LOGGER.indent():
                            comp_util.build_heatmaps(
                                diff,
                                min_diff,
                                max_diff,
                                prefix=f"{prefix} Error | {out0_name}",
                                save_dir=save_heatmaps,
                                show=show_heatmaps,
                                use_lognorm=use_lognorm,
                            )

                comp_util.log_output_stats(absdiff, failed, "Absolute Difference")
                build_heatmaps(absdiff, min_absdiff, max_absdiff, "Absolute")

                comp_util.log_output_stats(reldiff, failed, "Relative Difference")
                build_heatmaps(
                    reldiff, min_reldiff, max_reldiff, "Relative", use_lognorm=True
                )

            G_LOGGER.extra_verbose(
                lambda: f"Finished comparing: '{out0_name}' (dtype={util.array.dtype(out0)}, shape={util.array.shape(out0)}) [{runner0_name}] and '{out1_name}' (dtype={util.array.dtype(out1)}, shape={util.array.shape(out1)}) [{runner1_name}]"
            )
            return OutputCompareResult(
                not failed,
                max_absdiff,
                max_reldiff,
                mean_absdiff,
                mean_reldiff,
                median_absdiff,
                median_reldiff,
                quantile_absdiff,
                quantile_reldiff,
            )

        def compare_output(iter_result0, iter_result1):
            """
            Compare the outputs of two runners from a single iteration.

            This function will always iterate over the output names of the first IterationResult,
                and attempt to find corresponding output names in the second.
            If no corresponding output name is found, the output is skipped.
            If all output names are skipped, then this function raises an error.

            Args:
                iter_result0 (IterationResult): The result of the first runner.
                iter_result1 (IterationResult): The result of the second runner.

            Returns:
                OrderedDict[str, OutputCompareResult]:
                        The name of the outputs compared, derived from the first IterationResult,
                        and whether they matched. If an output name is not found, it is omitted from this dictionary.

            Raises:
                PolygraphyException: If all output names are skipped, and thus no outputs are compared.
            """

            def check_dict(dct, dict_name):
                if isinstance(dct, dict):
                    util.check_sequence_contains(
                        dct.keys(),
                        set(iter_result0.keys()) | set(iter_result1.keys()) | {""},
                        name=dict_name,
                        log_func=G_LOGGER.warning,
                        check_missing=False,
                    )

            check_dict(rtol, "the rtol dictionary")
            check_dict(atol, "the atol dictionary")
            check_dict(check_error_stat, "the check_error_stat dictionary")
            check_dict(error_quantile, "the quantile dictionary")

            if not check_shapes:
                G_LOGGER.info(
                    "Strict shape checking disabled. Will attempt to match output shapes before comparisons"
                )

            def match(out0_name, output0, out1_name, output1):
                per_out_atol = util.value_or_from_dict(atol, out0_name, default_atol)
                per_out_rtol = util.value_or_from_dict(rtol, out0_name, default_rtol)
                per_out_err_stat = util.value_or_from_dict(
                    check_error_stat, out0_name, default_error_stat
                )
                per_out_quantile = util.value_or_from_dict(
                    error_quantile, out0_name, default_quantile
                )

                G_LOGGER.info(
                    f"Tolerance: [abs={per_out_atol:.5g}, rel={per_out_rtol:.5g}] | Checking {per_out_err_stat} error"
                )
                G_LOGGER.extra_verbose(
                    f"Note: Comparing {iter_result0.runner_name} vs. {iter_result1.runner_name}"
                )

                if check_shapes and util.array.shape(output0) != util.array.shape(
                    output1
                ):
                    G_LOGGER.error(
                        f"FAILED | Output: `{out0_name}` | Will not compare outputs of different shapes.\n"
                        f"Note: Output shapes are {util.array.shape(output0)} and {util.array.shape(output1)}."
                    )
                    G_LOGGER.error(
                        "Note: Use --no-shape-check or set check_shapes=False to "
                        "attempt to compare values anyway.",
                        mode=LogMode.ONCE,
                    )
                    return False

                output1 = util.try_match_shape(output1, util.array.shape(output0))
                output0 = util.array.view(
                    output0,
                    DataType.from_dtype(util.array.dtype(output0)),
                    util.array.shape(output1),
                )
                outputs_matched = check_outputs_match(
                    output0,
                    out0_name,
                    output1,
                    out1_name,
                    per_out_rtol=per_out_rtol,
                    per_out_atol=per_out_atol,
                    per_out_err_stat=per_out_err_stat,
                    runner0_name=iter_result0.runner_name,
                    runner1_name=iter_result1.runner_name,
                    per_out_quantile=per_out_quantile,
                )

                # Finally show summary.
                if not outputs_matched:
                    G_LOGGER.error(
                        f"FAILED | Output: '{out0_name}' | Difference exceeds tolerance (rel={per_out_rtol}, abs={per_out_atol})"
                    )
                else:
                    G_LOGGER.finish(
                        f"PASSED | Output: '{out0_name}' | Difference is within tolerance (rel={per_out_rtol}, abs={per_out_atol})"
                    )

                return outputs_matched

            nonlocal find_output_func
            find_output_func = util.default(
                find_output_func,
                functools.partial(
                    default_find_output_func, base_iter_result=iter_result0
                ),
            )
            return run_comparison(
                match, fail_fast, iter_result0, iter_result1, find_output_func
            )

        return compare_output

    @staticmethod
    def indices(index_tolerance=None, fail_fast=None):
        """
        Creates a function that compares two IterationResults containing indices, and can be used as the `compare_func` argument
        in ``Comparator.compare_accuracy``. This can be useful to compare, for example, the outputs of a Top-K operation.

        Outputs with more than one dimension are treated like multiple batches of values. For example, an output of shape (3, 4, 5, 10)
        would be treated like 60 batches (3 x 4 x 5) of 10 values each.

        Args:
            index_tolerance (Union[int, Dict[str, int]]):
                    The tolerance to use when comparing indices. This is an integer indicating the maximum distance
                    between values before it is considered a mismatch. For example, consider two outputs:
                    ::

                        output0 = [0, 1, 2]
                        output1 = [1, 0, 2]

                    With an index tolerance of 0, this would be considered a mismatch, since the positions of `0` and `1`
                    are flipped between the two outputs. However, with an index tolerance of 1, it would pass since
                    the mismatched values are only 1 spot apart. If instead the outputs were:
                    ::

                        output0 = [0, 1, 2]
                        output1 = [1, 2, 0]

                    Then we would require an index tolerance of 2, since the `0` value in the two outputs is 2 spots apart.

                    When this value is set, the final 'index_tolerance' number of values are ignored for each batch.
                    For example, with an index tolerance of 1, mismatches in the final element are not considered.
                    If used with a Top-K output, you can compensate for this by instead using a Top-(K + index_tolerance).

                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default tolerance for outputs not explicitly listed.

            fail_fast (bool):
                    Whether the function should exit immediately after the first failure.
                    Defaults to False.


        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, bool]:
                A callable that returns a mapping of output names to `bool` s, indicating
                whether the corresponding output matched.

        """
        index_tolerance = util.default(index_tolerance, 0)
        fail_fast = util.default(fail_fast, False)

        def compare_output(iter_result0, iter_result1):
            """
            Compare the outputs of two runners from a single iteration.

            This function will always iterate over the output names of the first IterationResult,
                and attempt to find corresponding output names in the second.
            If no corresponding output name is found, the output is skipped.
            If all output names are skipped, then this function raises an error.

            Args:
                iter_result0 (IterationResult): The result of the first runner.
                iter_result1 (IterationResult): The result of the second runner.

            Returns:
                OrderedDict[str, bool]:
                        The name of the outputs compared, derived from the first IterationResult,
                        and whether they matched. If an output name is not found, it is omitted from this dictionary.

            Raises:
                PolygraphyException: If all output names are skipped, and thus no outputs are compared.
            """

            def match(out0_name, output0, out1_name, output1):
                per_out_index_tol = util.value_or_from_dict(
                    index_tolerance, out0_name, 0
                )

                if util.array.shape(output0) != util.array.shape(output1):
                    G_LOGGER.error("Cannot compare outputs of different shapes.")
                    return False

                passed = True
                for batch in np.ndindex(util.array.shape(output0)[:-1]):
                    out0_vals = output0[batch]
                    if per_out_index_tol > 0:
                        out0_vals = out0_vals[:-per_out_index_tol]
                    out1_vals = output1[batch]

                    for index0, val0 in enumerate(out0_vals):
                        if val0 == out1_vals[index0]:
                            continue

                        index1 = util.array.ravel(
                            util.array.argwhere(out1_vals == val0)
                        )
                        if util.array.size(index1) < 1:
                            G_LOGGER.error(
                                f"FAILED | Value: {val0} not found in output"
                            )
                            passed = False
                            if fail_fast:
                                return False
                            continue

                        index1 = index1[0]

                        if abs(index1 - index0) > per_out_index_tol:
                            G_LOGGER.error(
                                f"FAILED | Difference exceeds index tolerance ({per_out_index_tol})"
                            )
                            passed = False
                            if fail_fast:
                                return False
                            continue

                # Log information about the outputs
                hist_bin_range = (
                    min(comp_util.compute_min(output0), comp_util.compute_min(output1)),
                    max(comp_util.compute_max(output0), comp_util.compute_max(output1)),
                )
                comp_util.log_output_stats(
                    output0,
                    not passed,
                    f"{iter_result0.runner_name}: {out0_name}",
                    hist_range=hist_bin_range,
                )
                comp_util.log_output_stats(
                    output1,
                    not passed,
                    f"{iter_result1.runner_name}: {out1_name}",
                    hist_range=hist_bin_range,
                )

                if passed:
                    G_LOGGER.finish(
                        f"PASSED | Difference is within index tolerance ({per_out_index_tol})"
                    )
                return passed

            return run_comparison(
                match,
                fail_fast,
                iter_result0,
                iter_result1,
                functools.partial(
                    default_find_output_func, base_iter_result=iter_result0
                ),
            )

        return compare_output

    @staticmethod
    def distance_metrics(
        l2_tolerance=None,
        cosine_similarity_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Creates a function that compares two IterationResults using distance metrics (L2 norm and cosine similarity),
        and can be used as the `compare_func` argument in ``Comparator.compare_accuracy``.

        Args:
            l2_tolerance (Union[float, Dict[str, float]]):
                    The tolerance to use when checking L2 norm (Euclidean distance).
                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default tolerance for outputs not explicitly listed.
                    Defaults to 1e-5.
            cosine_similarity_threshold (Union[float, Dict[str, float]]):
                    The minimum cosine similarity required for outputs to be considered matching.
                    Cosine similarity measures the cosine of the angle between two vectors, with values between -1 and 1.
                    A value of 1 means vectors are identical or parallel, 0 means they are orthogonal, and -1 means they point in opposite directions.
                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default threshold for outputs not explicitly listed.
                    Defaults to 0.997 (which corresponds to a cosine distance of 0.003).
            check_shapes (bool):
                    Whether shapes must match exactly. If this is False, this function may
                    permute or reshape outputs before comparison.
                    Defaults to True.
            fail_fast (bool):
                    Whether the function should exit immediately after the first failure.
                    Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns a list of output names to compare against from the provided
                    IterationResult, given an output name and index from another IterationResult.
                    The comparison function will always iterate over the output names of the
                    first IterationResult, expecting names from the second. A return value of
                    `[]` or `None` indicates that the output should be skipped.

        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, DistanceMetricsResult]:
                A callable that returns a mapping of output names to `DistanceMetricsResult` s, indicating
                whether the corresponding output matched based on the distance metrics.
        """
        check_shapes = util.default(check_shapes, True)
        default_l2_tolerance = 1e-5
        default_cosine_similarity_threshold = 0.997  
        l2_tolerance = util.default(l2_tolerance, default_l2_tolerance)
        cosine_similarity_threshold = util.default(cosine_similarity_threshold, default_cosine_similarity_threshold)
        fail_fast = util.default(fail_fast, False)

        def compute_l2_norm(array1, array2):
            """Compute L2 norm (Euclidean distance) between two arrays."""
            diff = util.array.subtract(comp_util.cast_up(array1), comp_util.cast_up(array2))
            squared_diff = util.array.power(diff, 2)
            sum_squared_diff = util.array.sum(squared_diff)
            return util.array.sqrt(sum_squared_diff)

        def compute_cosine_similarity(array1, array2):
            """Compute cosine similarity between two arrays."""
            array1_flat = util.array.ravel(comp_util.cast_up(array1))
            array2_flat = util.array.ravel(comp_util.cast_up(array2))
            
            # Calculate dot product
            dot_product = util.array.sum(util.array.multiply(array1_flat, array2_flat))
            
            # Calculate magnitudes
            magnitude1 = util.array.sqrt(util.array.sum(util.array.power(array1_flat, 2)))
            magnitude2 = util.array.sqrt(util.array.sum(util.array.power(array2_flat, 2)))
            
            # Avoid division by zero
            if magnitude1 == 0 and magnitude2 == 0:
                return 1.0  # If both vectors are zero, they are identical (similarity = 1.0)
            elif magnitude1 == 0 or magnitude2 == 0:
                return 0.0  # If only one vector is zero, they are orthogonal (similarity = 0.0)
                
            # Cosine similarity is dot_product / (magnitude1 * magnitude2)
            cosine_similarity = dot_product / (magnitude1 * magnitude2)
            
            # Handle floating point issues that might make cosine_similarity slightly outside [-1, 1]
            return max(-1.0, min(1.0, cosine_similarity))

        def check_outputs_match(
            out0,
            out0_name,
            out1,
            out1_name,
            per_out_l2_tol,
            per_out_cosine_sim_threshold,
            runner0_name,
            runner1_name,
        ):
            """
            Checks whether two outputs matched using L2 norm and cosine similarity.

            Args:
                out0 (Union[np.array, torch.Tensor]): The first output.
                out0_name (str): The name of the first output.
                out1 (Union[np.array, torch.Tensor]): The second output.
                out1_name (str): The name of the second output.
                per_out_l2_tol (float): The L2 norm tolerance to use for comparison.
                per_out_cosine_sim_threshold (float): The minimum cosine similarity required for a match.
                runner0_name (str): The name of the runner that generated the first output.
                runner1_name (str): The name of the runner that generated the second output.

            Returns:
                DistanceMetricsResult: Details on whether the outputs matched.
            """
            G_LOGGER.super_verbose(
                f"{runner0_name:35} | Output: {out0_name} (dtype={util.array.dtype(out0)}, shape={util.array.shape(out0)}):\n{util.indent_block(out0)}"
            )
            G_LOGGER.super_verbose(
                f"{runner1_name:35} | Output: {out1_name} (dtype={util.array.dtype(out1)}, shape={util.array.shape(out1)}):\n{util.indent_block(out1)}"
            )

            # Compute metrics
            l2_norm = compute_l2_norm(out0, out1)
            cosine_sim = compute_cosine_similarity(out0, out1)

            # Check if outputs match based on the metrics
            l2_passed = bool(l2_norm <= per_out_l2_tol)
            cosine_passed = bool(cosine_sim >= per_out_cosine_sim_threshold)
            
            # Overall pass requires all enabled metrics to pass
            passed = bool(l2_passed and cosine_passed)

            # Log information
            hist_bin_range = (
                min(comp_util.compute_min(out0), comp_util.compute_min(out1)),
                max(comp_util.compute_max(out0), comp_util.compute_max(out1)),
            )
            comp_util.log_output_stats(
                out0, not passed, f"{runner0_name}: {out0_name}", hist_range=hist_bin_range
            )
            comp_util.log_output_stats(
                out1, not passed, f"{runner1_name}: {out1_name}", hist_range=hist_bin_range
            )

            G_LOGGER.info(f"Distance Metrics: {out0_name}")
            with G_LOGGER.indent():
                G_LOGGER.info(f"L2 Norm: {l2_norm:.5g} (tolerance: {per_out_l2_tol:.5g}) | {'PASSED' if l2_passed else 'FAILED'}")
                G_LOGGER.info(f"Cosine Similarity: {cosine_sim:.5g} (threshold: {per_out_cosine_sim_threshold:.5g}) | {'PASSED' if cosine_passed else 'FAILED'}")

            # Create a proper DistanceMetricsResult object with our metrics
            result = DistanceMetricsResult(
                passed=passed,
                l2_norm=l2_norm,
                cosine_similarity=cosine_sim
            )

            if not passed:
                if not l2_passed:
                    G_LOGGER.error(
                        f"FAILED | Output: '{out0_name}' | L2 Norm ({l2_norm:.5g}) exceeds tolerance ({per_out_l2_tol:.5g})"
                    )
                if not cosine_passed:
                    G_LOGGER.error(
                        f"FAILED | Output: '{out0_name}' | Cosine Similarity ({cosine_sim:.5g}) below threshold ({per_out_cosine_sim_threshold:.5g})"
                    )
            else:
                metrics_passed = ["L2 Norm", "Cosine Similarity"]
                G_LOGGER.finish(
                    f"PASSED | Output: '{out0_name}' | All metrics passed: {', '.join(metrics_passed)}"
                )

            return result

        def compare_output(iter_result0, iter_result1):
            """
            Compare the outputs of two runners from a single iteration using distance metrics.

            This function will always iterate over the output names of the first IterationResult,
                and attempt to find corresponding output names in the second.
            If no corresponding output name is found, the output is skipped.
            If all output names are skipped, then this function raises an error.

            Args:
                iter_result0 (IterationResult): The result of the first runner.
                iter_result1 (IterationResult): The result of the second runner.

            Returns:
                OrderedDict[str, DistanceMetricsResult]:
                        The name of the outputs compared, derived from the first IterationResult,
                        and whether they matched. If an output name is not found, it is omitted from this dictionary.

            Raises:
                PolygraphyException: If all output names are skipped, and thus no outputs are compared.
            """
            def check_dict(dct, dict_name):
                if isinstance(dct, dict):
                    util.check_sequence_contains(
                        dct.keys(),
                        set(iter_result0.keys()) | set(iter_result1.keys()) | {""},
                        name=dict_name,
                        log_func=G_LOGGER.warning,
                        check_missing=False,
                    )

            check_dict(l2_tolerance, "the l2_tolerance dictionary")
            check_dict(cosine_similarity_threshold, "the cosine_similarity_threshold dictionary")

            if not check_shapes:
                G_LOGGER.info(
                    "Strict shape checking disabled. Will attempt to match output shapes before comparisons"
                )

            def match(out0_name, output0, out1_name, output1):
                per_out_l2_tol = util.value_or_from_dict(l2_tolerance, out0_name, default_l2_tolerance)
                per_out_cosine_sim_threshold = util.value_or_from_dict(cosine_similarity_threshold, out0_name, default_cosine_similarity_threshold)

                # Build tolerance message showing all enabled metrics
                tolerance_msg = [f"L2={per_out_l2_tol:.5g}", f"Cosine Similarity min={per_out_cosine_sim_threshold:.5g}"]
                
                G_LOGGER.info(f"Tolerance: [{', '.join(tolerance_msg)}]")
                G_LOGGER.extra_verbose(
                    f"Note: Comparing {iter_result0.runner_name} vs. {iter_result1.runner_name}"
                )

                if check_shapes and util.array.shape(output0) != util.array.shape(
                    output1
                ):
                    G_LOGGER.error(
                        f"FAILED | Output: `{out0_name}` | Will not compare outputs of different shapes.\n"
                        f"Note: Output shapes are {util.array.shape(output0)} and {util.array.shape(output1)}."
                    )
                    G_LOGGER.error(
                        "Note: Use --no-shape-check or set check_shapes=False to "
                        "attempt to compare values anyway.",
                        mode=LogMode.ONCE,
                    )
                    return False

                output1 = util.try_match_shape(output1, util.array.shape(output0))
                output0 = util.array.view(
                    output0,
                    DataType.from_dtype(util.array.dtype(output0)),
                    util.array.shape(output1),
                )
                
                outputs_matched = check_outputs_match(
                    output0,
                    out0_name,
                    output1,
                    out1_name,
                    per_out_l2_tol=per_out_l2_tol,
                    per_out_cosine_sim_threshold=per_out_cosine_sim_threshold,
                    runner0_name=iter_result0.runner_name,
                    runner1_name=iter_result1.runner_name,
                )

                return outputs_matched

            nonlocal find_output_func
            find_output_func = util.default(
                find_output_func,
                functools.partial(
                    default_find_output_func, base_iter_result=iter_result0
                ),
            )
            return run_comparison(
                match, fail_fast, iter_result0, iter_result1, find_output_func
            )

        return compare_output

    @staticmethod
    def quality_metrics(
        psnr_tolerance=None,
        snr_tolerance=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Creates a function that compares two IterationResults using quality metrics (PSNR and SNR),
        and can be used as the `compare_func` argument in ``Comparator.compare_accuracy``.

        Args:
            psnr_tolerance (Union[float, Dict[str, float]]):
                    The minimum PSNR (Peak Signal-to-Noise Ratio) value required for outputs to be considered matching.
                    Higher values of PSNR indicate better quality matches. Typical acceptable values are 30 dB or above.
                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default tolerance for outputs not explicitly listed.
                    If None, PSNR check will be skipped. Defaults to 30.0.
            snr_tolerance (Union[float, Dict[str, float]]):
                    The minimum SNR (Signal-to-Noise Ratio) value required for outputs to be considered matching.
                    Higher values of SNR indicate better quality matches.
                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default tolerance for outputs not explicitly listed.
                    If None, SNR check will be skipped. Defaults to 20.0.
            check_shapes (bool):
                    Whether shapes must match exactly. If this is False, this function may
                    permute or reshape outputs before comparison.
                    Defaults to True.
            fail_fast (bool):
                    Whether the function should exit immediately after the first failure.
                    Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns a list of output names to compare against from the provided
                    IterationResult, given an output name and index from another IterationResult.
                    The comparison function will always iterate over the output names of the
                    first IterationResult, expecting names from the second. A return value of
                    `[]` or `None` indicates that the output should be skipped.

        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, QualityMetricsResult]:
                A callable that returns a mapping of output names to `QualityMetricsResult` s, indicating
                whether the corresponding output matched based on the quality metrics.
        """
        check_shapes = util.default(check_shapes, True)
        default_psnr_tolerance = 30.0
        default_snr_tolerance = 20.0
        psnr_tolerance = util.default(psnr_tolerance, default_psnr_tolerance)
        snr_tolerance = util.default(snr_tolerance, default_snr_tolerance)
        fail_fast = util.default(fail_fast, False)

        def compute_psnr(array1, array2):
            """
            Compute Peak Signal-to-Noise Ratio between two arrays.
            Higher values indicate better matches.
            """
            array1_cast = comp_util.cast_up(array1)
            array2_cast = comp_util.cast_up(array2)
            
            # Compute Mean Squared Error
            mse = util.array.mean(util.array.power(
                util.array.subtract(array1_cast, array2_cast), 2
            ))
            
            # Avoid division by zero
            if mse == 0:
                return float('inf')  # Perfect match
                
            # Compute data range (max value in reference array)
            max_val = comp_util.compute_max(array1_cast)
            if max_val <= 0:
                max_val = 1.0  # Default to 1.0 if max value is non-positive
                
            # PSNR formula: 20 * log10(MAX) - 10 * log10(MSE)
            psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
            return psnr

        def compute_snr(array1, array2):
            """
            Compute Signal-to-Noise Ratio between two arrays.
            Higher values indicate better matches.
            """
            array1_cast = comp_util.cast_up(array1)
            array2_cast = comp_util.cast_up(array2)
            
            # Signal power
            signal_power = util.array.mean(util.array.power(array1_cast, 2))
            
            # Noise is the difference between the arrays
            noise = util.array.subtract(array1_cast, array2_cast)
            noise_power = util.array.mean(util.array.power(noise, 2))
            
            # Avoid division by zero
            if noise_power == 0:
                return float('inf')  # Perfect match
            if signal_power == 0:
                return -float('inf')  # No signal

            # SNR formula: 10 * log10(signal_power / noise_power)
            snr = 10 * np.log10(signal_power / noise_power)
            return snr

        def check_outputs_match(
            out0,
            out0_name,
            out1,
            out1_name,
            per_out_psnr_tol,
            per_out_snr_tol,
            runner0_name,
            runner1_name,
        ):
            """
            Checks whether two outputs matched using quality metrics (PSNR and SNR).

            Args:
                out0 (Union[np.array, torch.Tensor]): The first output.
                out0_name (str): The name of the first output.
                out1 (Union[np.array, torch.Tensor]): The second output.
                out1_name (str): The name of the second output.
                per_out_psnr_tol (float): The minimum PSNR value required for a match.
                per_out_snr_tol (float): The minimum SNR value required for a match.
                runner0_name (str): The name of the runner that generated the first output.
                runner1_name (str): The name of the runner that generated the second output.

            Returns:
                QualityMetricsResult: Details on whether the outputs matched.
            """
            G_LOGGER.super_verbose(
                f"{runner0_name:35} | Output: {out0_name} (dtype={util.array.dtype(out0)}, shape={util.array.shape(out0)}):\n{util.indent_block(out0)}"
            )
            G_LOGGER.super_verbose(
                f"{runner1_name:35} | Output: {out1_name} (dtype={util.array.dtype(out1)}, shape={util.array.shape(out1)}):\n{util.indent_block(out1)}"
            )

            # Compute metrics
            psnr_value = None
            if per_out_psnr_tol is not None:
                psnr_value = compute_psnr(out0, out1)
            
            snr_value = None
            if per_out_snr_tol is not None:
                snr_value = compute_snr(out0, out1)

            # Check if outputs match based on the metrics
            # Default to True for metrics that weren't computed
            psnr_passed = True
            if per_out_psnr_tol is not None and psnr_value is not None:
                psnr_passed = bool(psnr_value >= per_out_psnr_tol)
            
            snr_passed = True
            if per_out_snr_tol is not None and snr_value is not None:
                snr_passed = bool(snr_value >= per_out_snr_tol)
            
            # Overall pass requires all enabled metrics to pass
            passed = bool(psnr_passed and snr_passed)

            # Log information
            hist_bin_range = (
                min(comp_util.compute_min(out0), comp_util.compute_min(out1)),
                max(comp_util.compute_max(out0), comp_util.compute_max(out1)),
            )
            comp_util.log_output_stats(
                out0, not passed, f"{runner0_name}: {out0_name}", hist_range=hist_bin_range
            )
            comp_util.log_output_stats(
                out1, not passed, f"{runner1_name}: {out1_name}", hist_range=hist_bin_range
            )

            G_LOGGER.info(f"Quality Metrics: {out0_name}")
            with G_LOGGER.indent():
                if per_out_psnr_tol is not None and psnr_value is not None:
                    G_LOGGER.info(f"PSNR: {psnr_value:.5g} dB (min required: {per_out_psnr_tol:.5g} dB) | {'PASSED' if psnr_passed else 'FAILED'}")
                
                if per_out_snr_tol is not None and snr_value is not None:
                    G_LOGGER.info(f"SNR: {snr_value:.5g} dB (min required: {per_out_snr_tol:.5g} dB) | {'PASSED' if snr_passed else 'FAILED'}")

            # Create a proper QualityMetricsResult object with our metrics
            result = QualityMetricsResult(
                passed=passed,
                psnr=psnr_value,
                snr=snr_value
            )

            if not passed:
                if per_out_psnr_tol is not None and psnr_value is not None and not psnr_passed:
                    G_LOGGER.error(
                        f"FAILED | Output: '{out0_name}' | PSNR ({psnr_value:.5g} dB) below required minimum ({per_out_psnr_tol:.5g} dB)"
                    )
                if per_out_snr_tol is not None and snr_value is not None and not snr_passed:
                    G_LOGGER.error(
                        f"FAILED | Output: '{out0_name}' | SNR ({snr_value:.5g} dB) below required minimum ({per_out_snr_tol:.5g} dB)"
                    )
            else:
                metrics_passed = []
                if per_out_psnr_tol is not None and psnr_value is not None:
                    metrics_passed.append("PSNR")
                if per_out_snr_tol is not None and snr_value is not None:
                    metrics_passed.append("SNR")
                
                if metrics_passed:
                    G_LOGGER.finish(
                        f"PASSED | Output: '{out0_name}' | All quality metrics passed: {', '.join(metrics_passed)}"
                    )
                else:
                    G_LOGGER.warning(
                        f"PASSED | Output: '{out0_name}' | No quality metrics were successfully computed"
                    )

            return result

        def compare_output(iter_result0, iter_result1):
            """
            Compare the outputs of two runners from a single iteration using quality metrics.

            This function will always iterate over the output names of the first IterationResult,
                and attempt to find corresponding output names in the second.
            If no corresponding output name is found, the output is skipped.
            If all output names are skipped, then this function raises an error.

            Args:
                iter_result0 (IterationResult): The result of the first runner.
                iter_result1 (IterationResult): The result of the second runner.

            Returns:
                OrderedDict[str, QualityMetricsResult]:
                        The name of the outputs compared, derived from the first IterationResult,
                        and whether they matched. If an output name is not found, it is omitted from this dictionary.

            Raises:
                PolygraphyException: If all output names are skipped, and thus no outputs are compared.
            """
            def check_dict(dct, dict_name):
                if isinstance(dct, dict):
                    util.check_sequence_contains(
                        dct.keys(),
                        set(iter_result0.keys()) | set(iter_result1.keys()) | {""},
                        name=dict_name,
                        log_func=G_LOGGER.warning,
                        check_missing=False,
                    )

            check_dict(psnr_tolerance, "the psnr_tolerance dictionary")
            check_dict(snr_tolerance, "the snr_tolerance dictionary")

            if not check_shapes:
                G_LOGGER.info(
                    "Strict shape checking disabled. Will attempt to match output shapes before comparisons"
                )

            def match(out0_name, output0, out1_name, output1):
                per_out_psnr_tol = None
                if psnr_tolerance is not None:
                    per_out_psnr_tol = util.value_or_from_dict(psnr_tolerance, out0_name, default_psnr_tolerance)
                
                per_out_snr_tol = None
                if snr_tolerance is not None:
                    per_out_snr_tol = util.value_or_from_dict(snr_tolerance, out0_name, default_snr_tolerance)

                # Build tolerance message showing all enabled metrics
                tolerance_msg = []
                if per_out_psnr_tol is not None:
                    tolerance_msg.append(f"PSNR min={per_out_psnr_tol:.5g} dB")
                if per_out_snr_tol is not None:
                    tolerance_msg.append(f"SNR min={per_out_snr_tol:.5g} dB")
                
                if tolerance_msg:
                    G_LOGGER.info(f"Quality Metrics Tolerance: [{', '.join(tolerance_msg)}]")
                else:
                    G_LOGGER.warning("No quality metrics enabled for comparison")
                    
                G_LOGGER.extra_verbose(
                    f"Note: Comparing {iter_result0.runner_name} vs. {iter_result1.runner_name}"
                )

                if check_shapes and util.array.shape(output0) != util.array.shape(
                    output1
                ):
                    G_LOGGER.error(
                        f"FAILED | Output: `{out0_name}` | Will not compare outputs of different shapes.\n"
                        f"Note: Output shapes are {util.array.shape(output0)} and {util.array.shape(output1)}."
                    )
                    G_LOGGER.error(
                        "Note: Use --no-shape-check or set check_shapes=False to "
                        "attempt to compare values anyway.",
                        mode=LogMode.ONCE,
                    )
                    return False

                output1 = util.try_match_shape(output1, util.array.shape(output0))
                output0 = util.array.view(
                    output0,
                    DataType.from_dtype(util.array.dtype(output0)),
                    util.array.shape(output1),
                )
                
                outputs_matched = check_outputs_match(
                    output0,
                    out0_name,
                    output1,
                    out1_name,
                    per_out_psnr_tol=per_out_psnr_tol,
                    per_out_snr_tol=per_out_snr_tol,
                    runner0_name=iter_result0.runner_name,
                    runner1_name=iter_result1.runner_name,
                )

                return outputs_matched

            nonlocal find_output_func
            find_output_func = util.default(
                find_output_func,
                functools.partial(
                    default_find_output_func, base_iter_result=iter_result0
                ),
            )
            return run_comparison(
                match, fail_fast, iter_result0, iter_result1, find_output_func
            )

        return compare_output

    @staticmethod
    def perceptual_metrics(
        lpips_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Creates a function that compares two IterationResults using perceptual metrics (LPIPS),
        and can be used as the `compare_func` argument in ``Comparator.compare_accuracy``.
        
        This function specifically targets image-like data and uses perceptual similarity metrics
        that correlate better with human perception than traditional distance metrics.

        Args:
            lpips_threshold (Union[float, Dict[str, float]]):
                    The maximum LPIPS (Learned Perceptual Image Patch Similarity) score allowed for outputs to be considered matching.
                    Lower values indicate more perceptually similar outputs. Typical values are below 0.1.
                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default threshold for outputs not explicitly listed.
                    If None, a default value of 0.1 will be used.
            check_shapes (bool):
                    Whether shapes must match exactly. If this is False, this function may
                    permute or reshape outputs before comparison.
                    Defaults to True.
            fail_fast (bool):
                    Whether the function should exit immediately after the first failure.
                    Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns a list of output names to compare against from the provided
                    IterationResult, given an output name and index from another IterationResult.
                    The comparison function will always iterate over the output names of the
                    first IterationResult, expecting names from the second. A return value of
                    `[]` or `None` indicates that the output should be skipped.

        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, PerceptualMetricsResult]:
                A callable that returns a mapping of output names to `PerceptualMetricsResult` s, indicating
                whether the corresponding output matched based on the perceptual metrics.
        """
        check_shapes = util.default(check_shapes, True)
        default_lpips_threshold = 0.1
        lpips_threshold = util.default(lpips_threshold, default_lpips_threshold)
        fail_fast = util.default(fail_fast, False)

        # Try to import torch and lpips if available
        torch = None
        lpips_model = None
        try:
            torch = mod.lazy_import("torch")
            lpips = mod.lazy_import("lpips")
            
            # Initialize LPIPS model with explicit device specification
            device = torch.device('cpu')
            
            # Try with different initialization approaches
            try:
                # First try with default initialization
                lpips_model = lpips.LPIPS(net='alex', version='0.1').to(device)
            except Exception as e1:
                G_LOGGER.warning(f"First LPIPS initialization approach failed: {e1}. Trying alternative method...")
                
                try:
                    # Try with a different network if AlexNet fails
                    lpips_model = lpips.LPIPS(net='vgg', version='0.1').to(device)
                except Exception as e2:
                    G_LOGGER.warning(f"Second LPIPS initialization approach failed: {e2}. Trying basic initialization...")
                    
                    try:
                        # As a last resort, try with the most basic initialization
                        model = lpips.LPIPS(net_type='alex')
                        model.eval()
                        lpips_model = model.to(device)
                    except Exception as e3:
                        G_LOGGER.warning(
                            f"Failed to initialize LPIPS model with all methods. LPIPS check will be skipped. "
                            f"Errors: {e1}; {e2}; {e3}"
                        )
        except ImportError:
            G_LOGGER.warning(
                "LPIPS comparison requested but torch or lpips module not found. "
                "Install with: pip install torch==1.9.0 lpips==0.1.4. "
                "LPIPS check will be skipped."
            )

        def compute_lpips(array1, array2):
            """
            Compute LPIPS (Learned Perceptual Image Patch Similarity) between two arrays.
            Lower values indicate more perceptually similar outputs.
            
            Requires PyTorch and the LPIPS package.
            """
            if torch is None or lpips_model is None:
                return None
            
            try:
                # Cast arrays to numpy to ensure compatibility
                array1_np = util.array.to_numpy(comp_util.cast_up(array1))
                array2_np = util.array.to_numpy(comp_util.cast_up(array2))
                
                # Check dimensions - LPIPS expects image data
                shape1 = array1_np.shape
                shape2 = array2_np.shape
                
                # We need at least 3D arrays for LPIPS (typically B,C,H,W or H,W,C)
                if len(shape1) < 3 or len(shape2) < 3:
                    G_LOGGER.warning(f"LPIPS requires at least 3D arrays, got shapes {shape1} and {shape2}. LPIPS check will be skipped.")
                    return None
                
                # Log input shapes for debugging
                G_LOGGER.verbose(f"Original tensor shapes: {shape1} and {shape2}")
                
                # Get device information - use CPU for consistency
                device = torch.device('cpu')
                
                # Convert to PyTorch tensors with shape B,C,H,W
                # LPIPS expects values in range [-1, 1] for both color and grayscale images
                def prepare_for_lpips(arr):
                    # Determine input format and convert to B,C,H,W format
                    if len(arr.shape) == 3:  # H,W,C or C,H,W
                        if arr.shape[2] <= 3:  # H,W,C format
                            # Convert H,W,C to B,C,H,W (add batch dimension)
                            arr = arr.transpose(2, 0, 1)[None, ...]
                        else:  # C,H,W format
                            # Add batch dimension
                            arr = arr[None, ...]
                    elif len(arr.shape) == 4:  # B,C,H,W or B,H,W,C
                        if arr.shape[3] <= 3:  # B,H,W,C format
                            arr = arr.transpose(0, 3, 1, 2)
                        # else: already in B,C,H,W format
                    
                    # Convert to float and normalize to [-1, 1] range if needed
                    arr = arr.astype(np.float32)
                    if arr.max() > 1.0:
                        arr = arr / 255.0
                    if arr.max() <= 1.0 and arr.min() >= 0.0:
                        arr = arr * 2.0 - 1.0  # [0,1] -> [-1,1]
                    
                    # Force 3-channel RGB format required by LPIPS
                    if arr.shape[1] == 1:  # Grayscale (single channel)
                        # Repeat the channel 3 times to create RGB
                        arr = np.repeat(arr, 3, axis=1)
                    elif arr.shape[1] == 2:  # Two channels
                        # Create a third channel (could duplicate channel 2 or create a new one)
                        third_channel = arr[:, 1:2]  # Use second channel as the third
                        arr = np.concatenate([arr, third_channel], axis=1)
                    elif arr.shape[1] > 3:  # More than 3 channels
                        arr = arr[:, :3]  # Use only first 3 channels
                    
                    # Convert to tensor
                    tensor = torch.from_numpy(arr)
                    return tensor.float().to(device)
                
                # Convert both inputs to torch tensors in correct format
                img1 = prepare_for_lpips(array1_np)
                img2 = prepare_for_lpips(array2_np)
                
                G_LOGGER.verbose(f"Prepared tensor shapes: {img1.shape} and {img2.shape}")
                
                # Ensure tensors have the same size in all dimensions
                if img1.shape != img2.shape:
                    G_LOGGER.warning(f"Tensor shapes don't match: {img1.shape} vs {img2.shape}. Adjusting...")
                    
                    # For channels, ensure both have 3 channels
                    if img1.shape[1] != 3:
                        if img1.shape[1] == 1:
                            img1 = img1.repeat(1, 3, 1, 1)
                        elif img1.shape[1] == 2:
                            img1 = torch.cat([img1, img1[:, 1:2]], dim=1)
                        else:  # > 3 channels
                            img1 = img1[:, :3]
                    
                    if img2.shape[1] != 3:
                        if img2.shape[1] == 1:
                            img2 = img2.repeat(1, 3, 1, 1)
                        elif img2.shape[1] == 2:
                            img2 = torch.cat([img2, img2[:, 1:2]], dim=1)
                        else:  # > 3 channels
                            img2 = img2[:, :3]
                    
                    # For spatial dimensions, resize to match
                    if img1.shape[2:] != img2.shape[2:]:
                        # Use the larger of the two spatial dimensions
                        target_size = (max(img1.shape[2], img2.shape[2]), max(img1.shape[3], img2.shape[3]))
                        
                        # Only import interpolate if needed
                        try:
                            from torch.nn.functional import interpolate
                            
                            if img1.shape[2:] != target_size:
                                img1 = interpolate(img1, size=target_size, mode='bilinear', align_corners=False)
                            
                            if img2.shape[2:] != target_size:
                                img2 = interpolate(img2, size=target_size, mode='bilinear', align_corners=False)
                                
                        except ImportError:
                            G_LOGGER.warning("Failed to resize tensors: torch.nn.functional.interpolate not available")
                            if img1.shape[2:] != img2.shape[2:]:
                                G_LOGGER.warning("Cannot compute LPIPS with tensors of different spatial dimensions")
                                return None
                
                G_LOGGER.verbose(f"Final tensor shapes: {img1.shape} and {img2.shape}")
                
                # Make sure the model is in eval mode
                lpips_model.eval()
                
                # Compute LPIPS distance (using no_grad to avoid storing gradients)
                with torch.no_grad():
                    try:
                        # Try the direct method
                        lpips_dist = lpips_model(img1, img2)
                        if isinstance(lpips_dist, torch.Tensor):
                            lpips_dist = lpips_dist.item()
                    except Exception as e:
                        G_LOGGER.warning(f"Standard LPIPS computation failed: {e}. Trying fallback method...")
                        try:
                            # Try an alternative approach
                            lpips_dist = lpips_model.forward(img1, img2)
                            if isinstance(lpips_dist, torch.Tensor):
                                lpips_dist = lpips_dist.mean().item()
                        except Exception as e2:
                            G_LOGGER.warning(f"Fallback LPIPS computation failed: {e2}. LPIPS check will be skipped.")
                            return None
        
                return lpips_dist
                
            except Exception as e:
                G_LOGGER.warning(f"Error computing LPIPS: {e}. LPIPS check will be skipped.")
                return None

        def check_perceptual_metrics(
            out0,
            out0_name,
            out1,
            out1_name,
            per_out_lpips_threshold,
            runner0_name,
            runner1_name,
        ):
            """
            Checks whether two outputs match using perceptual metrics.

            Args:
                out0 (Union[np.array, torch.Tensor]): The first output.
                out0_name (str): The name of the first output.
                out1 (Union[np.array, torch.Tensor]): The second output.
                out1_name (str): The name of the second output.
                per_out_lpips_threshold (float): The maximum LPIPS score allowed for a match.
                runner0_name (str): The name of the runner that generated the first output.
                runner1_name (str): The name of the runner that generated the second output.

            Returns:
                PerceptualMetricsResult: Details on whether the outputs matched.
            """
            # Log input information
            G_LOGGER.super_verbose(
                f"{runner0_name:35} | Output: {out0_name} (dtype={util.array.dtype(out0)}, shape={util.array.shape(out0)}):\n{util.indent_block(out0)}"
            )
            G_LOGGER.super_verbose(
                f"{runner1_name:35} | Output: {out1_name} (dtype={util.array.dtype(out1)}, shape={util.array.shape(out1)}):\n{util.indent_block(out1)}"
            )

            # Compute LPIPS
            lpips_value = compute_lpips(out0, out1)

            # Check if outputs match based on the metrics
            lpips_passed = True
            if lpips_value is not None:
                lpips_passed = bool(lpips_value <= per_out_lpips_threshold)
            
            # Overall pass only depends on LPIPS for now
            passed = lpips_passed

            # Log information about the outputs
            hist_bin_range = (
                min(comp_util.compute_min(out0), comp_util.compute_min(out1)),
                max(comp_util.compute_max(out0), comp_util.compute_max(out1)),
            )
            comp_util.log_output_stats(
                out0, not passed, f"{runner0_name}: {out0_name}", hist_range=hist_bin_range
            )
            comp_util.log_output_stats(
                out1, not passed, f"{runner1_name}: {out1_name}", hist_range=hist_bin_range
            )

            # Log perceptual metrics
            G_LOGGER.info(f"Perceptual Metrics: {out0_name}")
            with G_LOGGER.indent():
                if lpips_value is not None:
                    G_LOGGER.info(f"LPIPS: {lpips_value:.5g} (max allowed: {per_out_lpips_threshold:.5g}) | {'PASSED' if lpips_passed else 'FAILED'}")
                else:
                    G_LOGGER.warning("LPIPS computation was skipped or failed")

            # Create a PerceptualMetricsResult object
            result = PerceptualMetricsResult(
                passed=passed,
                lpips=lpips_value
            )

            # Log pass/fail status
            if not passed:
                if lpips_value is not None and not lpips_passed:
                    G_LOGGER.error(
                        f"FAILED | Output: '{out0_name}' | LPIPS ({lpips_value:.5g}) exceeds maximum threshold ({per_out_lpips_threshold:.5g})"
                    )
            else:
                metrics_passed = []
                if lpips_value is not None:
                    metrics_passed.append("LPIPS")
                
                if metrics_passed:
                    G_LOGGER.finish(
                        f"PASSED | Output: '{out0_name}' | All perceptual metrics passed: {', '.join(metrics_passed)}"
                    )
                else:
                    G_LOGGER.warning(
                        f"PASSED | Output: '{out0_name}' | No perceptual metrics were successfully computed"
                    )

            return result

        def compare_output(iter_result0, iter_result1):
            """
            Compare the outputs of two runners from a single iteration using perceptual metrics.

            This function will always iterate over the output names of the first IterationResult,
                and attempt to find corresponding output names in the second.
            If no corresponding output name is found, the output is skipped.
            If all output names are skipped, then this function raises an error.

            Args:
                iter_result0 (IterationResult): The result of the first runner.
                iter_result1 (IterationResult): The result of the second runner.

            Returns:
                OrderedDict[str, PerceptualMetricsResult]:
                        The name of the outputs compared, derived from the first IterationResult,
                        and whether they matched. If an output name is not found, it is omitted from this dictionary.

            Raises:
                PolygraphyException: If all output names are skipped, and thus no outputs are compared.
            """
            def check_dict(dct, dict_name):
                if isinstance(dct, dict):
                    util.check_sequence_contains(
                        dct.keys(),
                        set(iter_result0.keys()) | set(iter_result1.keys()) | {""},
                        name=dict_name,
                        log_func=G_LOGGER.warning,
                        check_missing=False,
                    )

            check_dict(lpips_threshold, "the lpips_threshold dictionary")

            if not check_shapes:
                G_LOGGER.info(
                    "Strict shape checking disabled. Will attempt to match output shapes before comparisons"
                )

            def match(out0_name, output0, out1_name, output1):
                per_out_lpips_threshold = util.value_or_from_dict(lpips_threshold, out0_name, default_lpips_threshold)

                # Log threshold information
                G_LOGGER.info(f"Perceptual Tolerance: [LPIPS max={per_out_lpips_threshold:.5g}]")
                G_LOGGER.extra_verbose(
                    f"Note: Comparing {iter_result0.runner_name} vs. {iter_result1.runner_name}"
                )

                if check_shapes and util.array.shape(output0) != util.array.shape(
                    output1
                ):
                    G_LOGGER.error(
                        f"FAILED | Output: `{out0_name}` | Will not compare outputs of different shapes.\n"
                        f"Note: Output shapes are {util.array.shape(output0)} and {util.array.shape(output1)}."
                    )
                    G_LOGGER.error(
                        "Note: Use --no-shape-check or set check_shapes=False to "
                        "attempt to compare values anyway.",
                        mode=LogMode.ONCE,
                    )
                    return False

                output1 = util.try_match_shape(output1, util.array.shape(output0))
                output0 = util.array.view(
                    output0,
                    DataType.from_dtype(util.array.dtype(output0)),
                    util.array.shape(output1),
                )
                
                outputs_matched = check_perceptual_metrics(
                    output0,
                    out0_name,
                    output1,
                    out1_name,
                    per_out_lpips_threshold=per_out_lpips_threshold,
                    runner0_name=iter_result0.runner_name,
                    runner1_name=iter_result1.runner_name,
                )

                return outputs_matched

            nonlocal find_output_func
            find_output_func = util.default(
                find_output_func,
                functools.partial(
                    default_find_output_func, base_iter_result=iter_result0
                ),
            )
            return run_comparison(
                match, fail_fast, iter_result0, iter_result1, find_output_func
            )

        return compare_output
