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
import numbers
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.comparator import util as comp_util
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")


@mod.export()
class OutputCompareResult(object):
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
        return "(atol={:}, rtol={:})".format(self.max_absdiff, self.max_reldiff)


def check_outputs_match(
    out0, out0_name, out1, out1_name, per_out_rtol, per_out_atol, per_out_err_stat, runner0_name, runner1_name
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
            "Invalid choice for check_error_stat: {:}.\n"
            "Note: Valid choices are: {:}".format(per_out_err_stat, VALID_CHECK_ERROR_STATS)
        )

    G_LOGGER.super_verbose(
        "{:35} | Output: {:} (dtype={:}, shape={:}):\n{:}".format(
            runner0_name, out0_name, out0.dtype, out0.shape, util.indent_block(out0)
        )
    )
    G_LOGGER.super_verbose(
        "{:35} | Output: {:} (dtype={:}, shape={:}):\n{:}".format(
            runner1_name, out1_name, out1.dtype, out1.shape, util.indent_block(out1)
        )
    )

    # Check difference vs. tolerances
    if np.issubdtype(out0.dtype, np.bool_) and np.issubdtype(out1.dtype, np.bool_):
        absdiff = np.logical_xor(out0, out1)
    else:
        absdiff = np.abs(out0 - out1)

    absout1 = np.abs(out1)
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        reldiff = absdiff / absout1
        max_reldiff = comp_util.compute_max(reldiff)
        mean_reldiff = comp_util.compute_mean(reldiff)
        median_reldiff = comp_util.compute_median(reldiff)

    max_absdiff = comp_util.compute_max(absdiff)
    mean_absdiff = comp_util.compute_mean(absdiff)
    median_absdiff = comp_util.compute_median(absdiff)

    max_elemwiseabs = "Unknown"
    max_elemwiserel = "Unknown"

    if per_out_err_stat == "mean":
        failed = mean_absdiff > per_out_atol and (np.isnan(mean_reldiff) or mean_reldiff > per_out_rtol)
    elif per_out_err_stat == "median":
        failed = median_absdiff > per_out_atol and (np.isnan(median_reldiff) or median_reldiff > per_out_rtol)
    elif per_out_err_stat == "max":
        failed = max_absdiff > per_out_atol and (np.isnan(max_reldiff) or max_reldiff > per_out_rtol)
    else:
        assert per_out_err_stat == "elemwise", "This branch should be unreachable unless per_out_err_stat is 'elemwise'"
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            mismatches = (absdiff > per_out_atol) & (reldiff > per_out_rtol)

        failed = np.any(mismatches)
        try:
            with np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning)
                # Special because we need to account for tolerances too.
                max_elemwiseabs = comp_util.compute_max(absdiff[mismatches])
                max_elemwiserel = comp_util.compute_max(reldiff[mismatches])

            with G_LOGGER.indent():
                G_LOGGER.super_verbose("Mismatched indices:\n{:}".format(np.argwhere(mismatches)))
                G_LOGGER.extra_verbose("{:35} | Mismatched values:\n{:}".format(runner0_name, out0[mismatches]))
                G_LOGGER.extra_verbose("{:35} | Mismatched values:\n{:}".format(runner1_name, out1[mismatches]))
        except Exception as err:
            G_LOGGER.warning("Failing to log mismatches.\nNote: Error was: {:}".format(err))

    # Log information about the outputs
    hist_bin_range = (
        min(comp_util.compute_min(out0), comp_util.compute_min(out1)),
        max(comp_util.compute_max(out0), comp_util.compute_max(out1)),
    )
    comp_util.log_output_stats(out0, failed, runner0_name + ": " + out0_name, hist_range=hist_bin_range)
    comp_util.log_output_stats(out1, failed, runner1_name + ": " + out1_name, hist_range=hist_bin_range)

    G_LOGGER.info("Error Metrics: {:}".format(out0_name))
    with G_LOGGER.indent():

        def req_tol(mean_diff, median_diff, max_diff, elemwise_diff):
            return {
                "mean": mean_diff,
                "median": median_diff,
                "max": max_diff,
                "elemwise": elemwise_diff,
            }[per_out_err_stat]

        G_LOGGER.info(
            "Minimum Required Tolerance: {:} error | [abs={:.5g}] OR [rel={:.5g}]".format(
                per_out_err_stat,
                req_tol(mean_absdiff, median_absdiff, max_absdiff, max_elemwiseabs),
                req_tol(mean_reldiff, median_reldiff, max_reldiff, max_elemwiserel),
            )
        )
        comp_util.log_output_stats(absdiff, failed, "Absolute Difference")
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            comp_util.log_output_stats(reldiff, failed, "Relative Difference")

    # Finally show summary.
    if failed:
        G_LOGGER.error("FAILED | Difference exceeds tolerance (rel={:}, abs={:})".format(per_out_rtol, per_out_atol))
    else:
        G_LOGGER.finish("PASSED | Difference is within tolerance (rel={:}, abs={:})".format(per_out_rtol, per_out_atol))

    G_LOGGER.extra_verbose(
        "Finished comparing: '{:}' (dtype={:}, shape={:}) [{:}] and '{:}' (dtype={:}, shape={:}) [{:}]".format(
            out0_name,
            out0.dtype,
            out0.shape,
            runner0_name,
            out1_name,
            out1.dtype,
            out1.shape,
            runner1_name,
        )
    )
    return OutputCompareResult(
        not failed, max_absdiff, max_reldiff, mean_absdiff, mean_reldiff, median_absdiff, median_reldiff
    )


# Provides functions to compare two IterationResults
@mod.export()
class CompareFunc(object):
    """
    Provides functions that can be used to compare two `IterationResult` s.
    """

    @staticmethod
    def basic_compare_func(*args, **kwargs):
        mod.warn_deprecated("basic_compare_func", remove_in="0.40.0", use_instead="simple")
        return CompareFunc.simple(*args, **kwargs)

    @staticmethod
    def simple(check_shapes=None, rtol=None, atol=None, fail_fast=None, find_output_func=None, check_error_stat=None):
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
                    - "max": Checks the maximum absolute/relative errors against the respective tolerances. This is the strictest possible check.
                    - "mean" Checks the mean absolute/relative errors against the respective tolerances.
                    - "median": Checks the median absolute/relative errors against the respective tolerances.

                    This can be provided on a per-output basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default error stat for outputs not explicitly listed.
                    Defaults to "elemwise".


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
                    util.check_dict_contains(
                        dct,
                        set(iter_result0.keys()) | set(iter_result1.keys()) | {""},
                        check_missing=False,
                        dict_name=dict_name,
                    )

            check_dict(rtol, "the rtol dictionary")
            check_dict(atol, "the atol dictionary")
            check_dict(check_error_stat, "the check_error_stat dictionary")

            output_status = OrderedDict()  # OrderedDict[str, bool] Maps output names to whether they matched.

            if not check_shapes:
                G_LOGGER.info("Strict shape checking disabled. Will attempt to match output shapes before comparisons")

            def default_find_output_func(output_name, index, iter_result):
                found_name = util.find_in_dict(output_name, iter_result, index)
                if found_name is None:
                    return None
                elif found_name != output_name:
                    exact_match = util.find_in_dict(found_name, iter_result0)
                    if exact_match == found_name:
                        G_LOGGER.verbose(
                            "Will not compare {:} with {:}, since the former already has an exact match: {:}".format(
                                found_name, output_name, exact_match
                            )
                        )
                        return None  # If the found output is being compared against another output already, skip this non-exact match
                    G_LOGGER.warning(
                        "Output names did not match exactly. Assuming {:} output: {:} "
                        "corresponds to output: {:}".format(iter_result.runner_name, found_name, output_name)
                    )
                return [found_name]

            nonlocal find_output_func
            find_output_func = util.default(find_output_func, default_find_output_func)

            for index, (out0_name, output0) in enumerate(iter_result0.items()):
                out1_names = util.default(find_output_func(out0_name, index, iter_result1), [])

                if len(out1_names) > 1:
                    G_LOGGER.info(
                        "Will attempt to compare output: '{:}' [{:}] with multiple outputs: '{:}' [{:}]".format(
                            out0_name, iter_result0.runner_name, list(out1_names), iter_result1.runner_name
                        )
                    )

                for out1_name in out1_names:
                    if out1_name is None or out1_name not in iter_result1:
                        G_LOGGER.warning(
                            "For output: '{:}' [{:}], skipping corresponding output: '{:}' [{:}], "
                            "since the output was not found".format(
                                out0_name, iter_result0.runner_name, out1_name, iter_result1.runner_name
                            )
                        )
                        continue

                    per_out_atol = util.value_or_from_dict(atol, out0_name, default_atol)
                    per_out_rtol = util.value_or_from_dict(rtol, out0_name, default_rtol)
                    per_out_err_stat = util.value_or_from_dict(check_error_stat, out0_name, default_error_stat)

                    output1 = iter_result1[out1_name]
                    G_LOGGER.start(
                        "Comparing Output: '{:}' (dtype={:}, shape={:}) with '{:}' (dtype={:}, shape={:}) | "
                        "Tolerance: [abs={:.5g}, rel={:.5g}] | Checking {:} error".format(
                            out0_name,
                            output0.dtype,
                            output0.shape,
                            out1_name,
                            output1.dtype,
                            output1.shape,
                            per_out_atol,
                            per_out_rtol,
                            per_out_err_stat,
                        )
                    )
                    G_LOGGER.extra_verbose(
                        "Note: Comparing {:} vs. {:}".format(iter_result0.runner_name, iter_result1.runner_name)
                    )

                    with G_LOGGER.indent():
                        if check_shapes and output0.shape != output1.shape:
                            G_LOGGER.error(
                                "Will not compare outputs of different shapes. Note: Output shapes are "
                                "{:} and {:}.".format(output0.shape, output1.shape)
                            )
                            G_LOGGER.error(
                                "Note: Use --no-shape-check or set check_shapes=False to "
                                "attempt to compare values anyway.",
                                mode=LogMode.ONCE,
                            )
                            outputs_match = False
                        else:
                            output1 = util.try_match_shape(output1, output0.shape)
                            output0 = output0.reshape(output1.shape)
                            outputs_match = check_outputs_match(
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

                        output_status[out0_name] = outputs_match
                        if fail_fast and not outputs_match:
                            return output_status

            mismatched_output_names = [name for name, matched in output_status.items() if not matched]
            if mismatched_output_names:
                G_LOGGER.error("FAILED | Mismatched outputs: {:}".format(mismatched_output_names))
            else:
                G_LOGGER.finish("PASSED | All outputs matched | Outputs: {:}".format(list(output_status.keys())))

            # This is useful for catching cases were Polygraphy does something wrong with the runner output buffers
            if not output_status and (bool(iter_result0.keys()) or bool(iter_result1.keys())):
                r0_name = iter_result0.runner_name
                r0_outs = list(iter_result0.keys())
                r1_name = iter_result1.runner_name
                r1_outs = list(iter_result1.keys())
                G_LOGGER.critical(
                    "All outputs were skipped, no common outputs found! Note:\n{:} outputs: "
                    "{:}\n{:} outputs: {:}".format(r0_name, r0_outs, r1_name, r1_outs)
                )

            return output_status

        return compare_output
