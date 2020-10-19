#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from collections import OrderedDict

import numpy as np
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc


class OutputCompareResult(object):
    """
    Represents the result of comparing a single output of a single iteration
    between two runners.
    """
    def __init__(self, passed, required_atol, required_rtol):
        """
        Records the required tolerances for the results to be considered equivalent.

        Args:
            passed (bool): Whether the error was within acceptable limits.
            required_atol (float): The minimum required absolute tolerance to consider the outputs equivalent.
            required_rtol (float): The minimum required relative tolerance to consider the outputs equivalent.
        """
        self.passed = passed
        self.required_atol = required_atol
        self.required_rtol = required_rtol


    def __bool__(self):
        """
        Whether the output matched.

        Returns:
            bool
        """
        return self.passed


    def __str__(self):
        return "(atol={:}, rtol={:})".format(self.required_atol, self.required_rtol)


# Provides functions to compare two IterationResults
class CompareFunc(object):
    """
    Provides functions that can be used to compare two `IterationResult` s.
    """

    @staticmethod
    def basic_compare_func(check_shapes=None, rtol=None, atol=None, fail_fast=None, find_output_func=None):
        """
        Creates a function that compares two IterationResults, and can be used as the `compare_func` argument
        in ``Comparator.compare_accuracy``.
        This function uses ``np.isclose`` to determine whether outputs match.
        For details, see https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html.

        Args:
            check_shapes (bool):
                    Whether shapes must match exactly. If this is False, this function may
                    permute or reshape outputs before comparison. Defaults to True.
            rtol (float):
                    The relative tolerance to use when checking accuracy. Defaults to 1e-5.
            atol (float):
                    The absolute tolerance to use when checking accuracy. Defaults to 1e-5.
            fail_fast (bool):
                    Whether the function should exit immediately after the first failure. Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns a list of output names to compare against from the provided
                    IterationResult, given an output name and index from another IterationResult.
                    The comparison function will always iterate over the output names of the
                    first IterationResult, expecting names from the second. A return value of
                    `[]` or `None` indicates that the output should be skipped.

        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, OutputCompareResult]:
                A callable that returns a mapping of output names to `OutputCompareResult` s, indicating
                whether the corresponding output matched.
        """
        check_shapes = misc.default_value(check_shapes, True)
        rtol = misc.default_value(rtol, 1e-5)
        atol = misc.default_value(atol, 1e-5)
        fail_fast = misc.default_value(fail_fast, False)


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
            # Returns whether the outputs match
            def check_outputs_match(out0, out0_name, out1, out1_name):
                def compute_max(buffer):
                    if misc.is_empty_shape(buffer.shape):
                        return 0
                    return np.amax(buffer)

                # Returns index of max value
                def compute_argmax(buffer):
                    if misc.is_empty_shape(buffer.shape):
                        return 0
                    return np.unravel_index(np.argmax(buffer), buffer.shape)

                def compute_min(buffer):
                    if misc.is_empty_shape(buffer.shape):
                        return 0
                    return np.amin(buffer)

                # Returns index of min value
                def compute_argmin(buffer):
                    if misc.is_empty_shape(buffer.shape):
                        return 0
                    return np.unravel_index(np.argmin(buffer), buffer.shape)

                def compute_mean(buffer):
                    if misc.is_empty_shape(buffer.shape):
                        return 0
                    return np.mean(buffer)


                def compute_required():
                    # The purpose of this function is to determine the minimum tolerances such that
                    # the outputs would be considered a match.
                    # The NumPy formula for np.isclose is absolute(out0 - out1) <= (atol + rtol * absolute(out1))
                    # So, for both absolute/relative tolerance, given either one,
                    # we can compute the required value for the other:
                    # atol = absolute(out0 - out1)
                    # atol_if_rtol = absolute(out0 - out1)  - rtol * absolute(out1)
                    # rtol = (absolute(out0 - out1) - atol) / absolute(out1)
                    if np.issubdtype(out0.dtype, np.bool_) and np.issubdtype(out1.dtype, np.bool_):
                        absdiff = np.logical_xor(out0, out1)
                    else:
                        absdiff = np.abs(out0 - out1)
                    absout1 = np.abs(out1)
                    required_atol = max(compute_max(absdiff), 0.0)
                    required_atol_if_rtol = max(compute_max(absdiff - rtol * absout1), 0.0)
                    # Suppress divide by 0 warnings
                    with np.testing.suppress_warnings() as sup:
                        sup.filter(RuntimeWarning)
                        required_rtol = max(compute_max((absdiff - atol) / absout1), 0.0)
                    return required_atol, required_atol_if_rtol, required_rtol


                def log_mismatches(mismatches):
                    try:
                        with G_LOGGER.indent():
                            G_LOGGER.super_verbose("Mismatches at:\n" + str(mismatches))
                            G_LOGGER.extra_verbose("Runner: {:40} | Mismatched values:\n{:}".format(iter_result0.runner_name, out0[mismatches]))
                            G_LOGGER.extra_verbose("Runner: {:40} | Mismatched values:\n{:}".format(iter_result1.runner_name, out1[mismatches]))
                    except:
                        G_LOGGER.warning("Failing to log mismatches - this may be because the outputs are of different shapes")


                try:
                    mismatches = np.logical_not(np.isclose(output0, output1, rtol=rtol, atol=atol))
                except Exception as err:
                    G_LOGGER.warning("Failed to compare outputs with:\n{:}\nSkipping".format(err))
                    return False

                G_LOGGER.super_verbose("Runner: {:40} | Output: {:} (dtype={:}, shape={:}):\n{:}".format(
                                            iter_result0.runner_name, out0_name, out0.dtype, out0.shape, misc.indent_block(out0)))
                G_LOGGER.super_verbose("Runner: {:40} | Output: {:} (dtype={:}, shape={:}):\n{:}".format(
                                            iter_result1.runner_name, out1_name, out1.dtype, out1.shape, misc.indent_block(out1)))

                failed = np.any(mismatches)

                try:
                    required_atol, required_atol_if_rtol, required_rtol = compute_required()
                except Exception as err:
                    required_atol, required_atol_if_rtol, required_rtol = None, None, None
                    G_LOGGER.warning("Could not determine required tolerances due to an error:\n{:}".format(err))
                    log_msg = ""
                else:
                    log_msg = "Required tolerances: [atol={:.5g}] OR [rtol={:.5g}, atol={:.5g}] OR [rtol={:.5g}, atol={:.5g}]\n".format(
                                    required_atol, rtol, required_atol_if_rtol, required_rtol, atol)

                log_msg += "Runner: {:40} | Stats: mean={:.5g}, min={:.5g} at {:}, max={:.5g} at {:}\n".format(
                                iter_result0.runner_name, compute_mean(out0), compute_min(out0), compute_argmin(out0), compute_max(out0), compute_argmax(out0))
                log_msg += "Runner: {:40} | Stats: mean={:.5g}, min={:.5g} at {:}, max={:.5g} at {:}\n".format(
                                iter_result1.runner_name, compute_mean(out1), compute_min(out1), compute_argmin(out1), compute_max(out1), compute_argmax(out1))

                if failed:
                    log_mismatches(mismatches)
                    G_LOGGER.info(log_msg)
                    G_LOGGER.error("FAILED | Difference exceeds tolerance (rtol={:}, atol={:})".format(rtol, atol))
                else:
                    G_LOGGER.verbose(log_msg)
                    G_LOGGER.success("PASSED | Difference is within tolerance (rtol={:}, atol={:})".format(rtol, atol))

                G_LOGGER.extra_verbose("Finished comparing: '{:}' (dtype={:}, shape={:}) [{:}] and '{:}' (dtype={:}, shape={:}) [{:}]"
                                .format(out0_name, out0.dtype, out0.shape, iter_result0.runner_name, out1_name, out1.dtype, out1.shape, iter_result1.runner_name))
                return OutputCompareResult(not failed, required_atol, required_rtol)


            output_status = OrderedDict() # OrderedDict[str, bool] Maps output names to whether they matched.

            if not check_shapes:
                G_LOGGER.info("Strict shape checking disabled. Will attempt to match output shapes before comparisons")


            def default_find_output_func(output_name, index, iter_result):
                found_name = misc.find_in_dict(output_name, iter_result, index)
                if found_name is None:
                    return None
                elif found_name != output_name:
                    exact_match = misc.find_in_dict(found_name, iter_result0)
                    if exact_match == found_name:
                        G_LOGGER.verbose("Will not compare {:} with {:}, since the former already has an exact match: {:}".format(
                                            found_name, output_name, exact_match))
                        return None # If the found output is being compared against another output already, skip this non-exact match
                    G_LOGGER.warning("Output names did not match exactly. Assuming {:} output: {:} "
                                    "corresponds to output: {:}".format(
                                        iter_result.runner_name, found_name, output_name))
                return [found_name]


            nonlocal find_output_func
            find_output_func = misc.default_value(find_output_func, default_find_output_func)

            for index, (out0_name, output0) in enumerate(iter_result0.items()):
                out1_names = misc.default_value(find_output_func(out0_name, index, iter_result1), [])

                if len(out1_names) > 1:
                    G_LOGGER.info("Will attempt to compare output: '{:}' [{:}] with multiple outputs: '{:}' [{:}]".format(
                                    out0_name, iter_result0.runner_name, list(out1_names), iter_result1.runner_name))

                for out1_name in out1_names:
                    if out1_name is None or out1_name not in iter_result1:
                        G_LOGGER.warning("For output: '{:}' [{:}], skipping corresponding output: '{:}' [{:}], "
                                         "since the output was not found".format(out0_name, iter_result0.runner_name,
                                                                                 out1_name, iter_result1.runner_name))
                        continue

                    output1 = iter_result1[out1_name]
                    G_LOGGER.info("Comparing Output: '{:}' (dtype={:}, shape={:}) with '{:}' (dtype={:}, shape={:})".format(
                                        out0_name, output0.dtype, output0.shape, out1_name, output1.dtype, output1.shape))
                    G_LOGGER.extra_verbose("Note: Comparing {:} vs. {:}".format(iter_result0.runner_name, iter_result1.runner_name))

                    with G_LOGGER.indent():
                        if check_shapes and output0.shape != output1.shape:
                            G_LOGGER.error("Will not compare outputs of different shapes. Note: Output shapes are "
                                           "{:} and {:}.".format(output0.shape, output1.shape))
                            G_LOGGER.error("Note: Use --no-strict-shape-checking or set check_shapes=False to "
                                           "attempt to compare values anyway.", mode=LogMode.ONCE)
                            outputs_match = False
                        else:
                            output1 = misc.try_match_shape(output1, output0.shape)
                            output0 = output0.reshape(output1.shape)
                            outputs_match = check_outputs_match(output0, out0_name, output1, out1_name)

                        output_status[out0_name] = outputs_match
                        if fail_fast and not outputs_match:
                            return output_status

            mismatched_output_names = [name for name, matched in output_status.items() if not matched]
            if mismatched_output_names:
                G_LOGGER.error("FAILED | Mismatched outputs: {:}".format(mismatched_output_names))

            # This is useful for catching cases were Polygraphy does something wrong with the runner output buffers
            if not output_status and (bool(iter_result0.keys()) or bool(iter_result1.keys())):
                r0_name = iter_result0.runner_name
                r0_outs = list(iter_result0.keys())
                r1_name = iter_result1.runner_name
                r1_outs = list(iter_result1.keys())
                G_LOGGER.critical("All outputs were skipped, no common outputs found! Note:\n{:} outputs: "
                                  "{:}\n{:} outputs: {:}".format(r0_name, r0_outs, r1_name, r1_outs))

            return output_status

        return compare_output
