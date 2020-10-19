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
import contextlib
import copy
import queue
from multiprocessing import Process, Queue

import numpy as np
from polygraphy.common import TensorMetadata
from polygraphy.comparator.compare import CompareFunc
from polygraphy.comparator.data_loader import DataLoader, DataLoaderCache
from polygraphy.comparator.struct import (AccuracyResult, IterationResult,
                                          RunResults)
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc


class Comparator(object):
    @staticmethod
    def run(runners, data_loader=None, warm_up=None,
            use_subprocess=None, subprocess_timeout=None,
            subprocess_polling_interval=None):
        """
        Runs the supplied runners sequentially.

        Args:
            data_loader (Generator -> OrderedDict[str, np.ndarray]):
                    A generator or iterable that yields a dictionary that maps input names to input numpy buffers.
                    In the simplest case, this can be a `List[Dict[str, np.ndarray]]` .

                    In case you don't know details about the inputs ahead of time, you can access the
                    `input_metadata` property in your data loader, which will be set to an `TensorMetadata`
                    instance by this function.
                    Note that this does not work for generators or lists.

                    The number of iterations run by this function is controlled by the number of items supplied
                    by the data loader.

                    Defaults to an instance of `DataLoader`.
            warm_up (int):
                    The number of warm up runs to perform for each runner before timing.
                    Defaults to 0.
            use_subprocess (bool):
                    Whether each runner should be run in a subprocess. This allows each runner to have exclusive
                    access to the GPU. When using a subprocess, runners and loaders will never be modified.
            subprocess_timeout (int):
                    The timeout before a subprocess is killed automatically. This is useful for handling processes
                    that never terminate. A value of None disables the timeout. Defaults to None.
            subprocess_polling_interval (int):
                    The polling interval, in seconds, for checking whether a subprocess has completed or crashed.
                    In rare cases, omitting this parameter when subprocesses are enabled may cause this function
                    to hang indefinitely if the subprocess crashes.
                    A value of 0 disables polling. Defaults to 30 seconds.

        Returns:
            RunResults: A mapping of runner names to the results of their inference. The ordering of `runners` is preserved in this mapping.
        """
        warm_up = misc.default_value(warm_up, 0)
        data_loader = misc.default_value(data_loader, DataLoader())
        use_subprocess = misc.default_value(use_subprocess, False)
        subprocess_polling_interval = misc.default_value(subprocess_polling_interval, 30)
        loader_cache = DataLoaderCache(data_loader)


        def execute_runner(runner, loader_cache):
            with runner as active_runner:
                input_metadata = active_runner.get_input_metadata()
                G_LOGGER.verbose("Runner: {:40} | Input Metadata:\n{:}".format(active_runner.name, misc.indent_block(input_metadata)))
                loader_cache.set_input_metadata(input_metadata)

                if warm_up:
                    G_LOGGER.info("Runner: {:40} | Running {:} warm-up runs".format(active_runner.name, warm_up))
                    try:
                        feed_dict = loader_cache[0]
                    except IndexError:
                        G_LOGGER.warning("{:} warm-up runs were requested, but data loader did not supply any data. "
                                         "Skipping warm-up runs".format(warm_up))
                    else:
                        G_LOGGER.ultra_verbose("Warm-up Input Buffers:\n{:}".format(misc.indent_block(feed_dict)))
                        # First do a few warm-up runs, and don't time them.
                        for i in range(warm_up):
                            active_runner.infer(feed_dict=feed_dict)

                # Then, actual iterations.
                total_time = 0
                run_results = []
                for feed_dict in loader_cache:
                    G_LOGGER.extra_verbose(lambda: "Runner: {:40} | Feeding inputs:\n{:}".format(active_runner.name, misc.indent_block(feed_dict)))
                    outputs = active_runner.infer(feed_dict=feed_dict)

                    runtime = active_runner.last_inference_time()
                    # Without a deep copy here, outputs will always reference the output of the last run
                    run_results.append(IterationResult(outputs=copy.deepcopy(outputs), runtime=runtime, runner_name=active_runner.name))

                    if len(run_results) == 1:
                        output_metadata = TensorMetadata()
                        for name, out in outputs.items():
                            output_metadata.add(name, out.dtype, out.shape)

                    G_LOGGER.verbose("Runner: {:40} | Output Metadata:\n{:}".format(active_runner.name, misc.indent_block(output_metadata)), mode=LogMode.ONCE)
                    G_LOGGER.extra_verbose(lambda: "Runner: {:40} | Inference Time: {:.3f} ms | Received outputs:\n{:}".format(
                                                        active_runner.name, runtime * 1000.0, misc.indent_block(outputs)))

                G_LOGGER.info("Runner: {:40} | Completed {:} iterations.".format(active_runner.name, len(run_results)))
                return run_results


        # Wraps execute_runner to use a queue.
        def execute_runner_with_queue(runner_queue, runner, loader_cache):
            run_results = None
            try:
                run_results = execute_runner(runner, loader_cache)
            except:
                # Cannot send the exception back, as it is not necessarily pickleable
                import traceback
                G_LOGGER.error(traceback.format_exc())
            misc.try_send_on_queue(runner_queue, run_results)
            # After finishing, send the updated loader_cache back.
            misc.try_send_on_queue(runner_queue, loader_cache)


        # Do all inferences in one loop, then comparisons at a later stage.
        # We run each runner in a separate process so that we can provide exclusive GPU access for each runner.
        runner_queue = Queue()
        run_results = RunResults()
        for index, runner in enumerate(runners):
            G_LOGGER.info("Runner: {:40} | Activating and starting inference".format(runner.name))
            if use_subprocess:
                process = Process(target=execute_runner_with_queue, args=(runner_queue, runner, loader_cache))
                process.start()

                # If a subprocess hangs in a certain way, then process.join could block forever. Hence,
                # we need to keep polling the process to make sure it really is alive.
                run_results[runner.name] = None
                while process.is_alive() and run_results[runner.name] is None:
                    try:
                        run_results[runner.name] = misc.try_receive_on_queue(runner_queue, timeout=subprocess_polling_interval / 2)
                        # Receive updated loader cache, or fall back if it could not be sent.
                        loader_cache = misc.try_receive_on_queue(runner_queue, timeout=subprocess_polling_interval / 2)
                    except queue.Empty:
                        G_LOGGER.extra_verbose("Polled subprocess - still running")

                try:
                    assert run_results[runner.name] is not None
                    process.join(subprocess_timeout)
                except:
                    G_LOGGER.critical("Runner: {:40} | Terminated prematurely. Check the exception logged above. "
                                      "If there is no exception logged above, make sure not to use the --use-subprocess "
                                      "flag or set use_subprocess=False in Comparator.run().".format(runner.name))
                finally:
                    process.terminate()

                if loader_cache is None:
                    G_LOGGER.critical("Could not send data loader cache to runner subprocess. Please try disabling subprocesses "
                                      "by removing the --use-subprocess flag, or setting use_subprocess=False in Comparator.run()")
            else:
                run_results[runner.name] = execute_runner(runner, loader_cache)

        G_LOGGER.verbose("Successfully ran: {:}".format([r.name for r in runners]))
        return run_results


    @staticmethod
    def postprocess(run_results, postprocess_func):
        """
        Applies post processing to all the outputs in the provided run results.
        This is a convenience function to avoid the need for manual iteration over the run_results dictionary.

        Args:
            run_results (RunResults): The result of Comparator.run().
            postprocess_func (Callable(IterationResult) -> IterationResult):
                    The function to apply to each ``IterationResult``.

        Returns:
            RunResults: The updated run results.
        """
        for runner_name, results in run_results.items():
            for index, result in enumerate(results):
                results[index] = postprocess_func(result)
        return run_results


    @staticmethod
    def default_comparisons(run_results):
        # Sets up default comparisons - which is to compare each runner to the subsequent one.
        all_runners = list(run_results.keys())
        return [(r1, r2) for r1, r2 in zip(all_runners[:-1], all_runners[1:])]


    @staticmethod
    def compare_accuracy(run_results, fail_fast=False, comparisons=None, compare_func=None):
        """
        Args:
            run_results (RunResults): The result of Comparator.run()


            fail_fast (bool): Whether to exit after the first failure
            comparisons (List[Tuple[str, str]]):
                    Comparisons to perform, specified by runner names. For example, [(r0, r1), (r1, r2)]
                    would compare the runner named r0 with r1, and r1 with r2.
                    By default, this compares each result to the subsequent one.
            compare_func (Callable(IterationResult, IterationResult) -> OrderedDict[str, bool]):
                    A function that takes in two IterationResults, and returns a dictionary that maps output
                    names to a boolean (or anything convertible to a boolean) indicating whether outputs matched.
                    The order of arguments to this function is guaranteed to be the same as the ordering of the
                    tuples contained in `comparisons`.

        Returns:
            AccuracyResult:
                    A summary of the results of the comparisons. The order of the keys (i.e. runner pairs) is
                    guaranteed to be the same as the order of `comparisons`. For more details, see the AccuracyResult
                    docstring (e.g. help(AccuracyResult)).
        """
        def find_mismatched(match_dict):
            return [name for name, matched in match_dict.items() if not bool(matched)]

        compare_func = misc.default_value(compare_func, CompareFunc.basic_compare_func())
        comparisons = misc.default_value(comparisons, Comparator.default_comparisons(run_results))

        accuracy_result = AccuracyResult()
        for runner0_name, runner1_name in comparisons:
            G_LOGGER.info("Accuracy Comparison | {:} vs. {:}".format(runner0_name, runner1_name))
            with G_LOGGER.indent():
                results0, results1 = run_results[runner0_name], run_results[runner1_name]
                runner_pair = (runner0_name, runner1_name)
                accuracy_result[runner_pair] = []

                num_iters = min(len(results0), len(results1))
                for iteration, (result0, result1) in enumerate(zip(results0, results1)):
                    if num_iters > 1:
                        G_LOGGER.info("Iteration: {:}".format(iteration))
                    with contextlib.ExitStack() as stack:
                        if num_iters > 1:
                            stack.enter_context(G_LOGGER.indent())
                        iteration_match_dict = compare_func(result0, result1)
                        accuracy_result[runner_pair].append(iteration_match_dict)

                    mismatched_outputs = find_mismatched(iteration_match_dict)
                    if fail_fast and mismatched_outputs:
                        return accuracy_result

                G_LOGGER.extra_verbose("Finished comparing {:} with {:}".format(runner0_name, runner1_name,))

                passed, failed, total = accuracy_result.stats(runner_pair)
                pass_rate = accuracy_result.percentage(runner_pair) * 100.0
                if num_iters > 1 or len(comparisons) > 1:
                    msg = "Accuracy Summary | {:} vs. {:} | Passed: {:}/{:} iterations | Pass Rate: {:}%".format(
                            runner0_name, runner1_name, passed, total, pass_rate)
                    if passed == total:
                        G_LOGGER.success(msg)
                    else:
                        G_LOGGER.error(msg)
        return accuracy_result


    @staticmethod
    def validate(run_results, check_finite=None, check_nan=None, fail_fast=None):
        """
        Checks output validity.

        Args:
            run_results (Dict[str, List[IterationResult]]): The result of Comparator.run().
            check_finite (bool): Whether to fail on non-finite values. Defaults to False.
            check_nan (bool): Whether to fail on NaNs. Defaults to True.
            fail_fast (bool): Whether to fail after the first invalid value. Defaults to False.

        Returns:
            bool: True if all outputs were valid, False otherwise.
        """
        check_finite = misc.default_value(check_finite, False)
        check_nan = misc.default_value(check_nan, True)
        fail_fast = misc.default_value(fail_fast, False)


        def is_finite(output):
            non_finite = np.logical_not(np.isfinite(output))
            if np.any(non_finite):
                G_LOGGER.error("Encountered one or more non-finite values")
                G_LOGGER.error("Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display non-finite values", mode=LogMode.ONCE)
                G_LOGGER.extra_verbose("Note: non-finite values at:\n{:}".format(non_finite))
                G_LOGGER.extra_verbose("Note: non-finite values:\n{:}".format(output[non_finite]))
                return False
            return True


        def is_not_nan(output):
            nans = np.isnan(output)
            if np.any(nans):
                G_LOGGER.error("Encountered one or more NaNs")
                G_LOGGER.error("Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display locations of NaNs", mode=LogMode.ONCE)
                G_LOGGER.extra_verbose("Note: NaNs at:\n{:}".format(nans))
                return False
            return True


        all_valid = True
        for runner_name, results in run_results.items():
            for result in results:
                for output_name, output in result.items():
                    G_LOGGER.info("Runner: {:40} | Validating output: {:} (check_finite={:}, check_nan={:})".format(
                                        runner_name, output_name, check_finite, check_nan))

                    output_valid = True
                    with G_LOGGER.indent():
                        if check_nan:
                            output_valid &= is_not_nan(output)
                        if check_finite:
                            output_valid &= is_finite(output)

                        all_valid &= output_valid

                        if output_valid:
                            G_LOGGER.success("Runner: {:40} | Output: {:} is valid".format(runner_name, output_name))
                        else:
                            G_LOGGER.error("Runner: {:40} | Errors detected in output: {:}".format(runner_name, output_name))
                            if fail_fast:
                                return False

        if all_valid:
            G_LOGGER.success("Validation passed")
        else:
            G_LOGGER.error("Validation failed")
        return all_valid
