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
import functools
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.comparator import util as comp_util
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")


@mod.export()
class OutputCompareResult:
    """
    Represents the result of comparing a single output of a single iteration
    between two runners.
    """

    def __init__(self, passed, max_absdiff, max_reldiff, mean_absdiff, mean_reldiff, median_absdiff, median_reldiff):
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
        """
        self.passed = passed
        self.max_absdiff = max_absdiff
        self.max_reldiff = max_reldiff
        self.mean_absdiff = mean_absdiff
        self.mean_reldiff = mean_reldiff
        self.median_absdiff = median_absdiff
        self.median_reldiff = median_reldiff

    def __bool__(self):
        """
        Whether the output matched.

        Returns:
            bool
        """
        return self.passed

    def __str__(self):
        return f"(atol={self.max_absdiff}, rtol={self.max_reldiff})"


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
            return (
                None  # If the found output is being compared against another output already, skip this non-exact match
            )
        G_LOGGER.warning(
            f"Output names did not match exactly. Assuming {iter_result.runner_name} output: {found_name} corresponds to output: {output_name}"
        )
    return [found_name]


def run_comparison(func, fail_fast, iter_result0, iter_result1, find_output_func):
    """
    Iterates over all the generated outputs and runs `func` to compare them.
    """
    output_status = OrderedDict()  # OrderedDict[str, bool] Maps output names to whether they matched.

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
                f"Comparing Output: '{out0_name}' (dtype={output0.dtype}, shape={output0.shape}) with '{out1_name}' (dtype={output1.dtype}, shape={output1.shape})"
            )
            output_status[out0_name] = func(out0_name, output0, out1_name, output1)
            if fail_fast and not output_status[out0_name]:
                return output_status

    mismatched_output_names = [name for name, matched in output_status.items() if not matched]
    if mismatched_output_names:
        G_LOGGER.error(f"FAILED | Mismatched outputs: {mismatched_output_names}")
    else:
        G_LOGGER.finish(f"PASSED | All outputs matched | Outputs: {list(output_status.keys())}")

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

                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default error stat for outputs not explicitly listed.
                    Defaults to "elemwise".
            infinities_compare_equal (bool):
                    If True, then matching +-inf values in the output have an absdiff of 0.
                    If False, then matching +-inf values in the output have an absdiff of NaN.
                    Defaults to False.

        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, OutputCompareResult]:
                A callable that returns a mapping of output names to `OutputCompareResult` s, indicating
                whether the corresponding output matched.
        """
        check_shapes = util.default(check_shapes, True)
        default_rtol = 1e-5
        default_atol = 1e-5
        rtol = util.default(rtol, default_rtol)
        atol = util.default(atol, default_atol)
        fail_fast = util.default(fail_fast, False)
        default_error_stat = "elemwise"
        check_error_stat = util.default(check_error_stat, default_error_stat)
        infinities_compare_equal = util.default(infinities_compare_equal, False)

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
        ):
            """
            Checks whether two outputs matched.

            Args:
                out0 (np.array): The first output.
                out0_name (str): The name of the first output.
                out1 (np.array): The second output.
                out1_name (str): The name of the second output.
                per_out_rtol (float): The relative tolerance to use for comparison.
                per_out_atol (float): The absolute tolerance to use for comparison.
                per_out_err_stat (str): The error statistic to check. See the docstring of ``simple`` for details.
                runner0_name (str): The name of the runner that generated the first output.
                runner1_name (str): The name of the runner that generated the second output.

            Returns:
                OutputCompareResult: Details on whether the outputs matched.
            """
            VALID_CHECK_ERROR_STATS = ["max", "mean", "median", "elemwise"]
            if per_out_err_stat not in VALID_CHECK_ERROR_STATS:
                G_LOGGER.critical(
                    f"Invalid choice for check_error_stat: {per_out_err_stat}.\nNote: Valid choices are: {VALID_CHECK_ERROR_STATS}"
                )

            G_LOGGER.super_verbose(
                f"{runner0_name:35} | Output: {out0_name} (dtype={out0.dtype}, shape={out0.shape}):\n{util.indent_block(out0)}"
            )
            G_LOGGER.super_verbose(
                f"{runner1_name:35} | Output: {out1_name} (dtype={out1.dtype}, shape={out1.shape}):\n{util.indent_block(out1)}"
            )

            # Check difference vs. tolerances
            if np.issubdtype(out0.dtype, np.bool_) and np.issubdtype(out1.dtype, np.bool_):
                absdiff = np.logical_xor(out0, out1)
            else:
                absdiff = np.abs(comp_util.cast_up(out0) - comp_util.cast_up(out1))
                if infinities_compare_equal:
                    out0_infinite = np.isinf(out0)
                    cond = np.logical_and(out0_infinite, out0 == out1)
                    absdiff = np.where(cond, 0, absdiff)

            with np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning)
                reldiff = absdiff / comp_util.cast_up(np.abs(out1))
                max_reldiff = comp_util.compute_max(reldiff)
                mean_reldiff = comp_util.compute_mean(reldiff)
                median_reldiff = comp_util.compute_median(reldiff)

            max_absdiff = comp_util.compute_max(absdiff)
            mean_absdiff = comp_util.compute_mean(absdiff)
            median_absdiff = comp_util.compute_median(absdiff)

            def stat_failed(diff, tol):
                return np.isnan(diff) or diff > tol

            if per_out_err_stat == "mean":
                failed = stat_failed(mean_absdiff, per_out_atol) and stat_failed(mean_reldiff, per_out_rtol)
            elif per_out_err_stat == "median":
                failed = stat_failed(median_absdiff, per_out_atol) and stat_failed(median_reldiff, per_out_rtol)
            elif per_out_err_stat == "max":
                failed = stat_failed(max_absdiff, per_out_atol) and stat_failed(max_reldiff, per_out_rtol)
            else:
                assert (
                    per_out_err_stat == "elemwise"
                ), "This branch should be unreachable unless per_out_err_stat is 'elemwise'"
                with np.testing.suppress_warnings() as sup:
                    sup.filter(RuntimeWarning)
                    mismatches = ((absdiff > per_out_atol) | np.isnan(absdiff)) & (
                        (reldiff > per_out_rtol) | np.isnan(reldiff)
                    )

                failed = np.any(mismatches)
                try:
                    with G_LOGGER.indent():
                        G_LOGGER.super_verbose(f"Mismatched indices:\n{np.argwhere(mismatches)}")
                        G_LOGGER.extra_verbose(f"{runner0_name:35} | Mismatched values:\n{out0[mismatches]}")
                        G_LOGGER.extra_verbose(f"{runner1_name:35} | Mismatched values:\n{out1[mismatches]}")
                except Exception as err:
                    G_LOGGER.warning(f"Failing to log mismatches.\nNote: Error was: {err}")

            # Log information about the outputs
            hist_bin_range = (
                min(comp_util.compute_min(out0), comp_util.compute_min(out1)),
                max(comp_util.compute_max(out0), comp_util.compute_max(out1)),
            )
            comp_util.log_output_stats(out0, failed, f"{runner0_name}: {out0_name}", hist_range=hist_bin_range)
            comp_util.log_output_stats(out1, failed, f"{runner1_name}: {out1_name}", hist_range=hist_bin_range)

            G_LOGGER.info(f"Error Metrics: {out0_name}")
            with G_LOGGER.indent():

                def req_tol(mean_diff, median_diff, max_diff):
                    return {
                        "mean": mean_diff,
                        "median": median_diff,
                        "max": max_diff,
                        "elemwise": max_diff,
                    }[per_out_err_stat]

                msg = f"Minimum Required Tolerance: {per_out_err_stat} error | [abs={req_tol(mean_absdiff, median_absdiff, max_absdiff):.5g}] OR [rel={req_tol(mean_reldiff, median_reldiff, max_reldiff):.5g}]"
                if per_out_err_stat == "elemwise":
                    msg += " (requirements may be lower if both abs/rel tolerances are set)"
                G_LOGGER.info(msg)

                comp_util.log_output_stats(absdiff, failed, "Absolute Difference")
                with np.testing.suppress_warnings() as sup:
                    sup.filter(RuntimeWarning)
                    comp_util.log_output_stats(reldiff, failed, "Relative Difference")

            G_LOGGER.extra_verbose(
                f"Finished comparing: '{out0_name}' (dtype={out0.dtype}, shape={out0.shape}) [{runner0_name}] and '{out1_name}' (dtype={out1.dtype}, shape={out1.shape}) [{runner1_name}]"
            )
            return OutputCompareResult(
                not failed, max_absdiff, max_reldiff, mean_absdiff, mean_reldiff, median_absdiff, median_reldiff
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

            if not check_shapes:
                G_LOGGER.info("Strict shape checking disabled. Will attempt to match output shapes before comparisons")

            def match(out0_name, output0, out1_name, output1):
                per_out_atol = util.value_or_from_dict(atol, out0_name, default_atol)
                per_out_rtol = util.value_or_from_dict(rtol, out0_name, default_rtol)
                per_out_err_stat = util.value_or_from_dict(check_error_stat, out0_name, default_error_stat)

                with G_LOGGER.indent():
                    G_LOGGER.info(
                        f"Tolerance: [abs={per_out_atol:.5g}, rel={per_out_rtol:.5g}] | Checking {per_out_err_stat} error"
                    )
                    G_LOGGER.extra_verbose(f"Note: Comparing {iter_result0.runner_name} vs. {iter_result1.runner_name}")

                    if check_shapes and output0.shape != output1.shape:
                        G_LOGGER.error(
                            f"Will not compare outputs of different shapes. Note: Output shapes are {output0.shape} and {output1.shape}."
                        )
                        G_LOGGER.error(
                            "Note: Use --no-shape-check or set check_shapes=False to "
                            "attempt to compare values anyway.",
                            mode=LogMode.ONCE,
                        )
                        outputs_matched = False
                    else:
                        output1 = util.try_match_shape(output1, output0.shape)
                        output0 = output0.reshape(output1.shape)
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
                find_output_func, functools.partial(default_find_output_func, base_iter_result=iter_result0)
            )
            return run_comparison(match, fail_fast, iter_result0, iter_result1, find_output_func)

        return compare_output

    @staticmethod
    def indices(index_tolerance=None, fail_fast=None):
        """
        Creates a function that compares two IterationResults containing indices, and can be used as the `compare_func` argument
        in ``Comparator.compare_accuracy``.

        Outputs with more than one dimension are treated like multiple batches of values. For example, an output of shape (3, 4, 5, 10)
        would be treated like 60 batches (3 x 4 x 5) of 10 values each.

        Args:
            index_tolerance (Union[int, Dict[str, int]]):
                    The tolerance to use when comparing indices. This is an integer indicating the maximum distance
                    between values before it is considered a mismatch. For example, consider two outputs:
                    ::

                        output0 = [0, 1, 2]
                        output1 = [1, 0, 2]

                    With an index tolerance of 0, this would be considered a mismatch. However, with an index tolerance
                    of 1, it would pass since the mismatched values, 0 and 1, are only one spot apart.

                    When this value is set, the final 'index_tolerance' number of values are ignored for each batch.
                    For example, with an index tolerance of 1, mismatches in the final element are not considered.

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
                per_out_index_tol = util.value_or_from_dict(index_tolerance, out0_name, 0)

                if output0.shape != output1.shape:
                    G_LOGGER.error("Cannot compare outputs of different shapes.")
                    return False

                passed = True
                for batch in np.ndindex(output0.shape[:-1]):
                    out0_vals = output0[batch]
                    if per_out_index_tol > 0:
                        out0_vals = out0_vals[:-per_out_index_tol]
                    out1_vals = output1[batch]

                    for index0, val0 in enumerate(out0_vals):
                        if val0 == out1_vals[index0]:
                            continue

                        index1 = np.argwhere(out1_vals == val0).ravel()
                        if index1.size < 1:
                            G_LOGGER.error(f"FAILED | Value: {val0} not found in output")
                            passed = False
                            continue

                        index1 = index1[0]

                        if abs(index1 - index0) > per_out_index_tol:
                            G_LOGGER.error(f"FAILED | Difference exceeds index tolerance ({per_out_index_tol})")
                            passed = False
                            continue

                # Log information about the outputs
                hist_bin_range = (
                    min(comp_util.compute_min(output0), comp_util.compute_min(output1)),
                    max(comp_util.compute_max(output0), comp_util.compute_max(output1)),
                )
                comp_util.log_output_stats(
                    output0, not passed, f"{iter_result0.runner_name}: {out0_name}", hist_range=hist_bin_range
                )
                comp_util.log_output_stats(
                    output1, not passed, f"{iter_result1.runner_name}: {out1_name}", hist_range=hist_bin_range
                )

                G_LOGGER.finish(f"PASSED | Difference is within index tolerance ({per_out_index_tol})")
                return passed

            return run_comparison(
                match,
                fail_fast,
                iter_result0,
                iter_result1,
                functools.partial(default_find_output_func, base_iter_result=iter_result0),
            )

        return compare_output
