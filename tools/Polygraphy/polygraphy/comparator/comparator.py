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
import contextlib
import copy
import itertools
import os
import queue
from multiprocessing import Process, Queue

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.comparator import util as comp_util
from polygraphy.comparator.compare import (
    SimpleCompareFunc,
    SimpleThreshold,
    metric_fields_for_func,
)
from polygraphy.comparator.data_loader import (
    DataLoader,
    DataLoaderCache,
    check_feed_dict,
    coerce_feed_dict,
)
from polygraphy.comparator.struct import (
    AccuracyResult,
    AccuracyResults,
    IterationResult,
    RunResults,
)

# Internal helpers (intentionally not part of the public polygraphy.json API).
from polygraphy.json.serde import ITERATION_FILE_INDEX_WIDTH, IterationWriter, is_single_file
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")


def _close_stream(stream):
    # The streaming generators (run_streaming, postprocess_stream, validate_stream) hold their
    # cleanup -- e.g. flushing a single-file save, releasing runner contexts -- in a `finally`. That
    # `finally` only runs when the generator is exhausted OR explicitly closed. A consumer that stops
    # early (fail_fast, an exception, or simply not draining the generator) would otherwise leave the
    # cleanup to run at an arbitrary later time when the garbage collector finalizes the generator.
    # Calling close() (which raises GeneratorExit inside the generator, triggering its `finally`)
    # forces that cleanup to happen deterministically, right when we stop consuming.
    #
    # Streams may be plain generators (which have close()) or our own wrappers; some may legitimately
    # lack a close(). Exceptions from close() (e.g. a failed single-file flush) are left to propagate
    # so the caller learns the save failed.
    close = getattr(stream, "close", None)
    if callable(close):
        close()


class _InputBlobWriter:
    """
    Writes each input tensor in a feed_dict to its own raw binary file via ``numpy.ndarray.tofile()``.
    Each iteration gets its own zero-padded subdirectory (matching ``IterationWriter``'s per-iteration
    file naming) so that multiple iterations do not overwrite one another; within it, each tensor is
    saved as ``<input_name>.bin``. Existing files/directories at the destination are overwritten.
    """

    def __init__(self, path):
        self.path = path
        self._index = 0
        if path is not None:
            G_LOGGER.info(f"Saving raw inference input tensors to: {path}")
            if os.path.exists(path) and not os.path.isdir(path):
                G_LOGGER.critical(
                    f"Cannot save raw inputs to '{path}': a file already exists there, but this "
                    f"path is a directory of per-iteration subdirectories."
                )
            os.makedirs(path, exist_ok=True)

    def append(self, feed_dict):
        if self.path is None:
            return
        iter_dir = os.path.join(self.path, f"{self._index:0{ITERATION_FILE_INDEX_WIDTH}d}")
        os.makedirs(iter_dir, exist_ok=True)
        real_iter_dir = os.path.realpath(iter_dir)
        for name, arr in feed_dict.items():
            dest = os.path.realpath(os.path.join(iter_dir, f"{name}.bin"))
            if os.path.commonpath([dest, real_iter_dir]) != real_iter_dir:
                G_LOGGER.critical(
                    f"Cannot save raw input tensor named '{name}': the name resolves to a path "
                    f"outside of '{iter_dir}'."
                )
            np.asarray(arr).tofile(dest)
        G_LOGGER.verbose(f"Saved raw inference input tensors to: {iter_dir}")
        self._index += 1


@mod.export()
class Comparator:
    """
    Compares inference outputs.
    """

    @staticmethod
    def run(
        runners,
        data_loader=None,
        warm_up=None,
        use_subprocess=None,
        subprocess_timeout=None,
        subprocess_polling_interval=None,
        save_inputs_path=None,
        save_outputs_path=None,
        save_input_blob_path=None,
        streaming=None,
    ):
        """
        Runs the supplied runners.

        By default (``streaming=False``) runs each runner to completion one at a time and returns a
        materialized ``RunResults`` holding every iteration. With ``streaming=True`` returns a
        *generator* that activates all runners simultaneously and yields one single-iteration
        ``RunResults`` per input, keeping memory roughly constant regardless of dataset size. The
        streaming form composes (optionally via ``Comparator.postprocess``/``validate``) with
        ``Comparator.compare_accuracy``.

        .. note::
            In streaming mode, consume each yielded ``RunResults`` (e.g. by comparing or saving it)
            before requesting the next; runner output buffers may be reused between iterations.

        Args:
            runners (List[BaseRunner]):
                    A list of runners to run.
            data_loader (Sequence[OrderedDict[str, numpy.ndarray]]):
                    An iterable that yields ``Dict[str, numpy.ndarray]`` feed dicts. The number of
                    iterations is determined by the number of items yielded.

                    If the data loader exposes a settable ``input_metadata`` attribute (as the
                    built-in ``DataLoader`` does), it is set to the first runner's ``TensorMetadata``
                    before iteration so the loader can size dynamic-shape inputs. This has no effect
                    for plain generators or lists.

                    Defaults to an instance of ``DataLoader``.
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
            save_inputs_path (str):
                    Where to save the inputs used during inference. A path *with a file extension*
                    (e.g. ``inputs.json``) saves all iterations to a single JSON file; an
                    *extensionless* path is treated as a directory holding one ``<index>.json`` file
                    per iteration (the directory must be empty or not yet exist). Defaults to None
                    (inputs are not saved).
            save_outputs_path (str):
                    Where to save the results, with the same file-vs-directory semantics as
                    ``save_inputs_path``. In streaming mode each iteration is written as produced (so
                    an extensionless path keeps memory constant); otherwise the materialized results
                    are saved on completion. Must not resolve to the same path as ``save_inputs_path``.
                    Defaults to None (results are not saved).
            save_input_blob_path (str):
                    A directory in which to save each input tensor as its own raw binary file (via
                    ``numpy.ndarray.tofile()``), one per-iteration subdirectory holding
                    ``<input_name>.bin`` files. Existing files/directories are overwritten. Defaults to
                    None (raw inputs are not saved).
            streaming (bool):
                    Whether to return a lazy per-iteration generator instead of a materialized
                    ``RunResults``. ``warm_up`` and ``use_subprocess`` are not supported in this
                    mode. Defaults to False.

        Returns:
            Union[RunResults, Iterable[RunResults]]:
                    A ``RunResults`` by default, or a generator of single-iteration ``RunResults``
                    with ``streaming=True``.
        """
        # The input, output, and raw input writers each assume they own their destination, so the same
        # path would silently overwrite one with the other. realpath catches symlinks resolving to one
        # location.
        for path_a, path_b in [
            (save_inputs_path, save_outputs_path),
            (save_inputs_path, save_input_blob_path),
            (save_outputs_path, save_input_blob_path),
        ]:
            if (
                path_a is not None
                and path_b is not None
                and os.path.realpath(path_a) == os.path.realpath(path_b)
            ):
                G_LOGGER.critical(
                    f"Two of save_inputs_path, save_outputs_path, and save_input_blob_path resolve to "
                    f"the same path ('{path_a}'). Use distinct paths."
                )

        def run_streaming():
            # streaming=True: infer one input at a time across all active runners, discarding outputs
            # after each is consumed, for constant memory.
            loader = util.default(data_loader, DataLoader())
            if not runners:
                G_LOGGER.warning(
                    "No runners were provided to Comparator.run(streaming=True). Inference will not be run."
                )
                return

            with contextlib.ExitStack() as stack:
                active_runners = []
                for runner in runners:
                    G_LOGGER.start(
                        f"{runner.name:35} | Activating and starting inference"
                    )
                    active_runners.append(stack.enter_context(runner))

                # Inputs are generated once (using the first runner's metadata), then coerced to each
                # runner's own metadata (like DataLoaderCache in the materialized path).
                input_metadatas = [
                    runner.get_input_metadata(use_numpy_dtypes=False)
                    for runner in active_runners
                ]
                with contextlib.suppress(AttributeError):
                    loader.input_metadata = input_metadatas[0]

                # copy_on_accumulate snapshots each iteration for single-file accumulation, since both
                # input feed_dicts and output buffers may be reused between iterations.
                input_writer = IterationWriter(
                    save_inputs_path,
                    copy_on_accumulate=True,
                    description="inference input data",
                )
                output_writer = IterationWriter(
                    save_outputs_path,
                    combine=RunResults.concat,
                    copy_on_accumulate=True,
                    description="inference results",
                )
                input_blob_writer = _InputBlobWriter(save_input_blob_path)
                try:
                    for index, feed_dict in enumerate(loader):
                        # Validate before the log below treats it as a feed_dict.
                        check_feed_dict(feed_dict)
                        G_LOGGER.info(
                            f"Streaming iteration {index}\n---- Inference Input(s) ----\n{TensorMetadata().from_feed_dict(feed_dict)}",
                            mode=LogMode.ONCE,
                        )
                        # A single-file save accumulates everything in memory; warn once at index 1.
                        if (
                            input_writer.single_file or output_writer.single_file
                        ) and index == 1:
                            G_LOGGER.warning(
                                "Saving streamed results to a single file accumulates every "
                                "iteration in memory; pass a directory to --save-inputs/--save-outputs "
                                "(read back with --load-inputs/--load-outputs) to keep memory constant."
                            )

                        input_writer.append(feed_dict)
                        input_blob_writer.append(feed_dict)

                        single = RunResults()
                        for runner, input_metadata in zip(
                            active_runners, input_metadatas
                        ):
                            outputs = runner.infer(
                                feed_dict=coerce_feed_dict(feed_dict, input_metadata)
                            )
                            runtime = runner.last_inference_time()
                            # No deep copy: the consumer compares before the next infer() overwrites
                            # the buffers, and each runner uses its own buffers per iteration.
                            single.append(
                                (
                                    runner.name,
                                    [
                                        IterationResult(
                                            outputs=outputs,
                                            runtime=runtime,
                                            runner_name=runner.name,
                                        )
                                    ],
                                )
                            )
                            G_LOGGER.info(
                                f"{runner.name:35}\n---- Inference Output(s) ----\n{TensorMetadata().from_feed_dict(outputs)}",
                                mode=LogMode.ONCE,
                            )
                            G_LOGGER.extra_verbose(
                                lambda: f"{runner.name:35} | Inference Time: {runtime * 1000.0:.3f} ms | Received outputs:\n{util.indent_block(dict(outputs))}"
                            )

                        output_writer.append(single)
                        yield single
                finally:
                    # Flush accumulated single-file saves once consumed (or closed early).
                    input_writer.flush()
                    output_writer.flush()

        if util.default(streaming, False):
            if warm_up or use_subprocess:
                G_LOGGER.critical(
                    "streaming=True does not support warm_up or use_subprocess. "
                    "Set streaming=False to use them."
                )
            return run_streaming()

        warm_up = util.default(warm_up, 0)
        data_loader = util.default(data_loader, DataLoader())
        use_subprocess = util.default(use_subprocess, False)
        subprocess_polling_interval = util.default(subprocess_polling_interval, 30)
        loader_cache = DataLoaderCache(data_loader)

        def execute_runner(runner, loader_cache):
            with runner as active_runner:
                # DataLoaderCache will ensure that the feed_dict does not contain any extra entries
                # based on the provided input_metadata.
                loader_cache.set_input_metadata(
                    active_runner.get_input_metadata(use_numpy_dtypes=False)
                )

                if warm_up:
                    G_LOGGER.start(
                        f"{active_runner.name:35} | Running {warm_up} warm-up run(s)"
                    )
                    try:
                        feed_dict = loader_cache[0]
                    except IndexError:
                        G_LOGGER.warning(
                            f"{warm_up} warm-up run(s) were requested, but data loader did not supply any data. Skipping warm-up run(s)"
                        )
                    else:
                        G_LOGGER.ultra_verbose(
                            f"Warm-up Input Buffers:\n{util.indent_block(feed_dict)}"
                        )
                        # First do a few warm-up runs, and don't time them.
                        for _ in range(warm_up):
                            active_runner.infer(feed_dict=feed_dict)
                    G_LOGGER.finish(
                        f"{active_runner.name:35} | Finished {warm_up} warm-up run(s)"
                    )

                # Then, actual iterations.
                iteration_results = []
                iterations_num = len(loader_cache)
                total_runtime = 0
                for index, feed_dict in enumerate(loader_cache):
                    G_LOGGER.info(
                        f"{active_runner.name:35}\n---- Inference Input(s) ----\n{TensorMetadata().from_feed_dict(feed_dict)}",
                        mode=LogMode.ONCE,
                    )

                    G_LOGGER.extra_verbose(
                        lambda: f"{active_runner.name:35} | Feeding inputs:\n{util.indent_block(dict(feed_dict))}"
                    )
                    outputs = active_runner.infer(feed_dict=feed_dict)

                    runtime = active_runner.last_inference_time()
                    total_runtime += runtime

                    # Only make a deep copy if we have more than one iteration.
                    # For single iteration case, we can use the outputs directly since they won't be reused.
                    # This allows running with a large number of outputs (e.g. for accuracy debugging) without memory explosion.
                    iteration_results.append(
                        IterationResult(
                            outputs=(
                                copy.deepcopy(outputs)
                                if iterations_num > 1
                                else outputs
                            ),
                            runtime=runtime,
                            runner_name=active_runner.name,
                        )
                    )

                    G_LOGGER.info(
                        f"{active_runner.name:35}\n---- Inference Output(s) ----\n{TensorMetadata().from_feed_dict(outputs)}",
                        mode=LogMode.ONCE,
                    )
                    G_LOGGER.extra_verbose(
                        lambda: f"{active_runner.name:35} | Inference Time: {runtime * 1000.0:.3f} ms | Received outputs:\n{util.indent_block(dict(outputs))}"
                    )

                total_runtime_ms = total_runtime * 1000.0
                G_LOGGER.finish(
                    f"{active_runner.name:35} | Completed {index + 1} iteration(s) in {total_runtime_ms:.4g} ms | Average inference time: {total_runtime_ms / float(index + 1):.4g} ms."
                )
                return iteration_results

        # Wraps execute_runner to use a queue.
        def execute_runner_with_queue(runner_queue, runner, loader_cache):
            iteration_results = None
            try:
                iteration_results = execute_runner(runner, loader_cache)
            except:
                # Cannot necessarily send the exception back over the queue.
                G_LOGGER.backtrace()
            util.try_send_on_queue(runner_queue, iteration_results)
            # After finishing, send the updated loader_cache back.
            util.try_send_on_queue(runner_queue, loader_cache)

        # Do all inferences in one loop, then comparisons at a later stage.
        # We run each runner in a separate process so that we can provide exclusive GPU access for each runner.
        run_results = RunResults()

        if not runners:
            G_LOGGER.warning(
                "No runners were provided to Comparator.run(). Inference will not be run, and run results will be empty."
            )

        for runner in runners:
            G_LOGGER.start(f"{runner.name:35} | Activating and starting inference")
            if use_subprocess:
                runner_queue = Queue()
                process = Process(
                    target=execute_runner_with_queue,
                    args=(runner_queue, runner, loader_cache),
                )
                process.start()

                # If a subprocess hangs in a certain way, then process.join could block forever. Hence,
                # we need to keep polling the process to make sure it really is alive.
                iteration_results = None
                while process.is_alive() and iteration_results is None:
                    try:
                        iteration_results = util.try_receive_on_queue(
                            runner_queue, timeout=subprocess_polling_interval / 2
                        )
                        # Receive updated loader cache, or fall back if it could not be sent.
                        loader_cache = util.try_receive_on_queue(
                            runner_queue, timeout=subprocess_polling_interval / 2
                        )
                    except queue.Empty:
                        G_LOGGER.extra_verbose("Polled subprocess - still running")

                try:
                    assert iteration_results is not None
                    run_results.append((runner.name, iteration_results))
                    process.join(subprocess_timeout)
                except:
                    G_LOGGER.critical(
                        f"{runner.name:35} | Terminated prematurely. Check the exception logged above. If there is no exception logged above, make sure not to use the --use-subprocess flag or set use_subprocess=False in Comparator.run()."
                    )
                finally:
                    process.terminate()

                if loader_cache is None:
                    G_LOGGER.critical(
                        "Could not send data loader cache to runner subprocess. Please try disabling subprocesses "
                        "by removing the --use-subprocess flag, or setting use_subprocess=False in Comparator.run()"
                    )
            else:
                run_results.append((runner.name, execute_runner(runner, loader_cache)))

        G_LOGGER.verbose(f"Successfully ran: {[r.name for r in runners]}")

        # Save inputs and outputs together here (mirroring the streaming path's inline writers). The
        # inputs come from the loader cache, which the runner loop above has now materialized; guard
        # on `runners` so an empty cache from a no-runner call is not written out.
        if save_inputs_path is not None and runners:
            with IterationWriter(
                save_inputs_path, description="inference input data"
            ) as writer:
                for feed_dict in loader_cache.cache:
                    writer.append(feed_dict)

        if save_input_blob_path is not None and runners:
            input_blob_writer = _InputBlobWriter(save_input_blob_path)
            for feed_dict in loader_cache.cache:
                input_blob_writer.append(feed_dict)

        if save_outputs_path is not None:
            if is_single_file(save_outputs_path):
                # The results are already fully materialized, so save them directly. Going through
                # split()+concat() (which matches runners by name) would collapse distinct runners
                # that happen to share a name into one entry.
                run_results.save(save_outputs_path)
            else:
                # A directory holds one single-iteration RunResults per file.
                with IterationWriter(
                    save_outputs_path, description="inference results"
                ) as writer:
                    for single in run_results.split():
                        writer.append(single)

        return run_results

    @staticmethod
    def postprocess(run_results, postprocess_func):
        """
        Applies ``postprocess_func`` to every ``IterationResult`` in ``run_results``.

        Accepts a ``RunResults``, a stream (e.g. from ``Comparator.run(streaming=True)``), or a
        list of either. Returns the same shape: a ``RunResults`` in place, a lazy generator for a
        stream, or a new list.

        Args:
            run_results (Union[RunResults, Iterable[RunResults], List]):
                    The run, stream, or list of runs to process.
            postprocess_func (Callable(IterationResult) -> IterationResult): The function to apply.

        Returns:
            The post-processed run(s) in the same shape as ``run_results``.
        """
        if isinstance(run_results, (list, tuple)):
            return [
                Comparator.postprocess(run, postprocess_func) for run in run_results
            ]

        # Readable log name for both plain functions (__name__) and functor instances like
        # TopKPostprocessFunc (which lack __name__).
        name = getattr(postprocess_func, "__name__", type(postprocess_func).__name__)

        def apply(single):
            for _, iteration_results in single:
                for index, iter_res in enumerate(iteration_results):
                    iteration_results[index] = postprocess_func(iter_res)
            return single

        if isinstance(run_results, RunResults):
            G_LOGGER.start(f"Applying post-processing to outputs: {name}")
            apply(run_results)
            G_LOGGER.finish("Finished applying post-processing")
            return run_results

        def postprocess_stream(stream):
            # Close the inner stream in a finally so an early close() (e.g. fail-fast) flushes any
            # single-file save deterministically rather than waiting for garbage collection.
            G_LOGGER.verbose(f"Will apply post-processing to streamed outputs: {name}")
            try:
                for single in stream:
                    yield apply(single)
            finally:
                _close_stream(stream)

        return postprocess_stream(run_results)

    @staticmethod
    def default_comparisons(run_results):
        # Sets up default comparisons - which is to compare each runner to the subsequent one.
        return [(i, i + 1) for i in range(len(run_results) - 1)]

    @staticmethod
    def compare_accuracy(
        run_results,
        fail_fast=False,
        comparisons=None,
        compare_func=None,
        check_average=False,
    ):
        """
        Compares inference outputs across runners and reports per-comparison accuracy.

        Accepts a ``RunResults``, a stream of single-iteration ``RunResults`` (e.g. from
        ``Comparator.run(streaming=True)``), or a list of either; streaming runs are compared one
        iteration at a time with constant memory. Post-processing and validation are separate steps
        (``Comparator.postprocess`` / ``Comparator.validate``).

        Args:
            run_results (Union[RunResults, Sequence[Union[RunResults, Iterable[RunResults]]]]):
                    The run(s) to compare.
            fail_fast (bool): Whether to stop after the first failure. Defaults to False.
            comparisons (List[Tuple[int, int]]):
                    Runner index pairs to compare. Defaults to comparing each runner to the next.
            compare_func (Union[Callable, Sequence[Callable]]):
                    A comparison functor/callable (or list of them) mapping two ``IterationResult``s
                    to an ``OrderedDict[str, bool]``; each produces one ``AccuracyResult``. Defaults
                    to ``SimpleCompareFunc()``.
            check_average (bool):
                    Check the *average* of each metric across iterations against the threshold rather
                    than each iteration (per-iteration results are still recorded). Requires a
                    ``compare_func`` that produces an averageable metric (e.g. ``SimpleCompareFunc``
                    with a scalar ``check_error_stat``, the single-metric functions, or
                    ``PerceptualMetricsCompareFunc``); ``IndicesCompareFunc`` and ``SimpleCompareFunc``
                    with ``elemwise`` are rejected. Mutually exclusive with ``fail_fast``. Defaults to
                    False.

        Returns:
            AccuracyResults:
                    One ``AccuracyResult`` per comparison function. ``bool(results)`` is True only if
                    *every* result passed, so ``if results:`` is a correct overall pass/fail check.
        """
        funcs = compare_func if util.is_sequence(compare_func) else [compare_func]
        funcs = [util.default(func, SimpleCompareFunc()) for func in funcs]

        # check_average needs every iteration (so it rejects fail_fast) and a compare_func that
        # supports averaging and produces an averageable metric.
        if check_average:
            if fail_fast:
                G_LOGGER.critical(
                    "check_average and fail_fast cannot be used together."
                )
            for func in funcs:
                # A functor with no threshold protocol at all (None, e.g. a plain function) is
                # distinguished from one that has it but produces no averageable metric ([], e.g.
                # 'indices').
                fields = metric_fields_for_func(func)
                if fields:
                    # 'simple' with the elemwise stat carries scalar diff fields but no scalar
                    # verdict to average (it is a per-element check), so reject it here rather than
                    # after inference has already run.
                    threshold = func.thresholds_for("")
                    if (
                        isinstance(threshold, SimpleThreshold)
                        and threshold.check_error_stat == "elemwise"
                    ):
                        G_LOGGER.critical(
                            "check_average is not supported with check_error_stat='elemwise': it is "
                            "a per-element check with no scalar statistic to average. Specify max, "
                            "mean, median, or quantile."
                        )
                    continue
                if fields == []:
                    G_LOGGER.critical(
                        "The selected compare_func does not produce averageable metrics "
                        "and cannot be used with check_average=True."
                    )
                G_LOGGER.critical(
                    "The provided compare_func does not support average comparison. "
                    "Use a compare_func that produces averageable metrics (sets _RESULT_CLASS and "
                    "implements thresholds_for) with check_average=True."
                )

        # Normalize to per-runner streams: split each materialized run into one stream per runner
        # (iterating in list order, so duplicate names stay distinct); use a stream as-is.
        def iterate_runner(name, iteration_results):
            for iter_result in iteration_results:
                single = RunResults()
                single.append((name, [iter_result]))
                yield single

        # Only a list/tuple is treated as *multiple* runs; a single RunResults or a bare stream is
        # wrapped as one. We can't use util.is_sequence here because a bare stream (generator) is
        # also a sequence, so it wouldn't be distinguishable from a list of runs -- and iterating it
        # to normalize would eagerly consume the stream and mis-group its iterations.
        runs = run_results if isinstance(run_results, (list, tuple)) else [run_results]
        streams = []
        for run in runs:
            if isinstance(run, RunResults):
                streams.extend(iterate_runner(name, iters) for name, iters in run)
            else:
                streams.append(run)

        def merged_stream():
            # zip_longest (not zip) so a length mismatch is detected and warned about with the
            # concrete count compared. It pulls one extra item from each longer stream, so a live
            # longer stream runs (and, if saving, persists) one extra uncompared iteration.
            stream_list = streams
            stream_end = object()
            compared = 0
            try:
                for group in itertools.zip_longest(*stream_list, fillvalue=stream_end):
                    exhausted = [
                        i for i, item in enumerate(group) if item is stream_end
                    ]
                    if exhausted:
                        G_LOGGER.warning(
                            f"The provided streams have different numbers of iterations. Only the "
                            f"first {compared} iteration(s) -- present in all {len(stream_list)} "
                            f"streams -- were compared; stream(s) at index {exhausted} ran out first."
                        )
                        break
                    merged = RunResults()
                    for run_results in group:
                        merged.extend(run_results)
                    yield merged
                    compared += 1
            finally:
                # Close each stream (independently, so one failure does not block the rest) to flush
                # any single-file save on an early stop.
                for stream in stream_list:
                    try:
                        _close_stream(stream)
                    except Exception as err:
                        G_LOGGER.warning(f"Error while closing a stream: {err}")

        def make_results(pair_list):
            results = []
            for func in funcs:
                result = AccuracyResult(
                    aggregation="average" if check_average else "per_sample",
                )
                for pair in pair_list:
                    result[pair] = []
                results.append(result)
            return results

        def severity_for(passed):
            return G_LOGGER.FINISH if passed else G_LOGGER.ERROR

        def summarize_pair(results, runner_pair):
            # One combined summary per runner pair: an iteration (or output, in average mode) passes
            # only if every comparison function passes.
            prefix = f"Accuracy Summary | {runner_pair[0]} vs. {runner_pair[1]} | "
            if check_average:
                # Averaged verdicts are computed on demand from each result's per-iteration data.
                per_func = [ar.average_results(runner_pair) for ar in results]
                # Union (order-preserving) across funcs so an output evaluated only by a later func
                # is still counted -- matching the metrics-display block below.
                output_names = list(dict.fromkeys(name for r in per_func for name in r))

                def output_passed(name):
                    # An output passes only if every comparison function passes it; funcs that did
                    # not evaluate it (absent from their results) default to passing.
                    return all(bool(r.get(name, True)) for r in per_func)

                passed = sum(output_passed(name) for name in output_names)
                total = len(output_names)
                summary_line = prefix + f"Passed: {passed}/{total} outputs"
            else:
                num_iters = len(results[0][runner_pair])
                passed = sum(
                    all(
                        bool(match)
                        for ar in results
                        for match in ar[runner_pair][i].values()
                    )
                    for i in range(num_iters)
                )
                total = num_iters
                summary_line = (
                    prefix
                    + f"Passed: {passed}/{total} iterations | Pass Rate: {(float(passed) / float(total) if total else 1.0) * 100:.2f}%"
                )
            G_LOGGER.log(summary_line, severity_for(passed == total))
            if check_average:
                # Mirror the per-iteration metric display: per output, list each metric's averaged
                # value, the threshold it was checked against, and whether it passed.
                merged_descriptions = {}
                for ar in results:
                    for output_name, description in ar.describe_average(
                        runner_pair
                    ).items():
                        merged_descriptions.setdefault(output_name, []).append(
                            description
                        )
                with G_LOGGER.indent():
                    for output_name in output_names:
                        out_passed = output_passed(output_name)
                        G_LOGGER.log(
                            f"{output_name} | {'PASSED' if out_passed else 'FAILED'}:",
                            severity_for(out_passed),
                        )
                        with G_LOGGER.indent():
                            for line, line_passed in merged_descriptions.get(
                                output_name, []
                            ):
                                G_LOGGER.log(line, severity_for(line_passed))

        def compare_pair_iteration(results, runner_pair, result0, result1):
            # Compare one (result0, result1) with each function, recording each match dict and logging
            # the combined per-output stats once. Returns True on a fail_fast mismatch; every function
            # is still evaluated this iteration (keeping results length-consistent), but fail_fast
            # then stops subsequent iterations.
            pair_match_dicts = []
            mismatched = False
            for func_index, func in enumerate(funcs):
                match_dict = func(result0, result1)
                results[func_index][runner_pair].append(match_dict)
                pair_match_dicts.append(match_dict)
                if fail_fast and any(not bool(m) for m in match_dict.values()):
                    mismatched = True
            comp_util.log_compared_output_stats(
                result0, result1, pair_match_dicts, *runner_pair
            )
            return mismatched

        accuracy_results = None
        pairs = None

        def finalize(do_summaries):
            results = (
                accuracy_results if accuracy_results is not None else make_results([])
            )
            # Summaries are skipped on the fail-fast path (a mismatch already failed the run).
            if do_summaries and pairs:
                for pair in pairs:
                    summarize_pair(results, pair)
            return AccuracyResults(results)

        merged_gen = merged_stream()
        try:
            for index, run_results in enumerate(merged_gen):
                iter_runner_names = list(run_results.keys())
                if accuracy_results is None:
                    runner_names = iter_runner_names
                    comps = util.default(
                        comparisons, Comparator.default_comparisons(run_results)
                    )
                    pairs = [(runner_names[i], runner_names[j]) for i, j in comps]
                    accuracy_results = make_results(pairs)
                    for runner0_name, runner1_name in pairs:
                        G_LOGGER.start(
                            f"Accuracy Comparison | {runner0_name} vs. {runner1_name}"
                        )
                elif iter_runner_names != runner_names:
                    # Pairs are resolved by position from the first iteration, so every iteration must
                    # contain the same runners in the same order (a mismatch usually means a stream
                    # with unequal per-runner iteration counts, e.g. a ragged saved run).
                    G_LOGGER.critical(
                        f"Streamed iteration {index} contains a different set of runners than the "
                        f"first ({iter_runner_names} vs. {runner_names}). Every iteration must "
                        f"contain the same runners in the same order."
                    )

                G_LOGGER.info(f"Iteration: {index}")
                # Index by position (not name) so runners sharing a name are still compared correctly.
                for (runner0_index, runner1_index), runner_pair in zip(comps, pairs):
                    result0 = run_results[runner0_index][1][0]
                    result1 = run_results[runner1_index][1][0]
                    with G_LOGGER.indent():
                        mismatched = compare_pair_iteration(
                            accuracy_results, runner_pair, result0, result1
                        )
                    if mismatched:
                        return finalize(do_summaries=False)
            return finalize(do_summaries=True)
        finally:
            # Close the stream chain so a single-file save flushes on an early return.
            merged_gen.close()

    @staticmethod
    def validate(run_results, check_inf=None, check_nan=None, fail_fast=None):
        """
        Checks output validity (for NaNs/Infs).

        Args:
            run_results (Union[RunResults, Iterable[RunResults], List]):
                    A ``RunResults``, a stream of single-iteration ``RunResults`` (e.g. from
                    ``Comparator.run(streaming=True)``), or a list of either. For a stream, returns
                    a lazy pass-through that validates each iteration as it is consumed and aborts
                    via ``G_LOGGER.critical`` on the first invalid value. ``fail_fast`` applies only
                    to the materialized form.
            check_inf (bool): Whether to fail on Infs. Defaults to False.
            check_nan (bool): Whether to fail on NaNs. Defaults to True.
            fail_fast (bool): Whether to fail after the first invalid value. Defaults to False.

        Returns:
            For a ``RunResults``, True if all outputs were valid, False otherwise. For a stream, a
            lazy pass-through that aborts on an invalid value. For a list, a list of the above.
        """
        if isinstance(run_results, (list, tuple)):
            return [
                Comparator.validate(
                    run, check_inf=check_inf, check_nan=check_nan, fail_fast=fail_fast
                )
                for run in run_results
            ]
        if not isinstance(run_results, RunResults):
            check_inf = util.default(check_inf, False)
            check_nan = util.default(check_nan, True)

            def validate_stream(stream):
                # Lazily validate each iteration as it is consumed, aborting on the first invalid
                # value. Like postprocess_stream, the generator's own close() runs the finally,
                # flushing any single-file save deterministically on an early stop.
                try:
                    for index, single in enumerate(stream):
                        valid = Comparator._validate_run_results(
                            single,
                            check_inf=check_inf,
                            check_nan=check_nan,
                            fail_fast=True,
                            iteration=index,
                            emit_summary=False,
                        )
                        if not valid:
                            G_LOGGER.critical(
                                f"Output validation failed: iteration {index} contains invalid "
                                f"values (NaNs/Infs). See the errors above."
                            )
                        yield single
                finally:
                    _close_stream(stream)

            return validate_stream(run_results)
        return Comparator._validate_run_results(
            run_results,
            check_inf=util.default(check_inf, False),
            check_nan=util.default(check_nan, True),
            fail_fast=util.default(fail_fast, False),
            iteration=None,
            emit_summary=True,
        )

    @staticmethod
    def _validate_run_results(
        run_results, check_inf, check_nan, fail_fast, iteration, emit_summary
    ):
        """Validates one materialized ``RunResults``; shared by ``validate`` and its streaming generator."""
        iter_prefix = "" if iteration is None else f"Iteration {iteration} | "

        def is_finite(output):
            non_finite = util.array.logical_not(util.array.isfinite(output))
            if util.array.any(non_finite):
                G_LOGGER.error(
                    "Inf Detected | One or more non-finite values were encountered in this output"
                )
                G_LOGGER.info(
                    "Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display non-finite values",
                    mode=LogMode.ONCE,
                )
                G_LOGGER.extra_verbose(f"Note: non-finite values at:\n{non_finite}")
                G_LOGGER.extra_verbose(
                    f"Note: non-finite values:\n{output[non_finite]}"
                )
                return False
            return True

        def is_not_nan(output):
            nans = util.array.isnan(output)
            if util.array.any(nans):
                G_LOGGER.error(
                    "NaN Detected | One or more NaNs were encountered in this output"
                )
                G_LOGGER.info(
                    "Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display locations of NaNs",
                    mode=LogMode.ONCE,
                )
                G_LOGGER.extra_verbose(f"Note: NaNs at:\n{nans}")
                return False
            return True

        def validate_output(runner_name, output_name, output):
            G_LOGGER.start(
                f"{runner_name:35} | {iter_prefix}Validating output: {output_name} (check_inf={check_inf}, check_nan={check_nan})"
            )
            with G_LOGGER.indent():
                comp_util.log_output_stats(output)

                output_valid = True
                if check_nan:
                    output_valid &= is_not_nan(output)
                if check_inf:
                    output_valid &= is_finite(output)

                if output_valid:
                    G_LOGGER.finish(f"PASSED | Output: {output_name} is valid")
                else:
                    G_LOGGER.error(f"FAILED | Errors detected in output: {output_name}")
                return output_valid

        all_valid = True
        if emit_summary:
            G_LOGGER.start(f"Output Validation | Runners: {list(run_results.keys())}")
        with G_LOGGER.indent() if emit_summary else contextlib.nullcontext():
            for runner_name, results in run_results:
                for result in results:
                    for output_name, output in result.items():
                        all_valid &= validate_output(runner_name, output_name, output)
                        if fail_fast and not all_valid:
                            return False

            if emit_summary:
                if all_valid:
                    G_LOGGER.finish("PASSED | Output Validation")
                else:
                    G_LOGGER.error("FAILED | Output Validation")

        return all_valid
