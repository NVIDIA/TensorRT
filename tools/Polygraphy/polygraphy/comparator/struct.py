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

from collections import OrderedDict

from polygraphy import config, mod, util
from polygraphy.common.interface import TypedDict, TypedList
from polygraphy.json import Decoder, Encoder, add_json_methods, load_json, save_json

# Internal helper (not part of the public polygraphy.json API).
from polygraphy.json.serde import iterate_from_files
from polygraphy.logger import G_LOGGER


class LazyArray:
    """
    Represents a lazily loaded NumPy array or PyTorch Tensor.
    For example, large arrays may be serialized to temporary files on the disk
    to save memory.
    """

    def __init__(self, arr):
        """
        Args:
            arr (Union[np.ndarray, torch.Tensor]): The array.
        """
        self.arr = None
        self.tmpfile = None
        if config.ARRAY_SWAP_THRESHOLD_MB >= 0 and util.array.nbytes(arr) > (
            config.ARRAY_SWAP_THRESHOLD_MB << 20
        ):
            self.tmpfile = util.NamedTemporaryFile(suffix=".json")
            G_LOGGER.extra_verbose(
                f"Evicting large array ({util.array.nbytes(arr) / 1024.0 ** 2:.3f} MiB) from memory and saving to {self.tmpfile.name}"
            )
            save_json(arr, self.tmpfile.name)
        else:
            self.arr = arr

    def load(self):
        """
        Load the array, deserializing from the disk if it was stored earlier.

        Returns:
            Union[np.ndarray, torch.Tensor]: The array
        """
        if self.arr is not None:
            return self.arr

        if self.tmpfile is None:
            G_LOGGER.internal_error(
                f"self.arr is None but self.tmpfile is also None; this should be impossible."
            )
        return load_json(self.tmpfile.name)


@Encoder.register(LazyArray, alias="LazyNumpyArray")
def encode(lazy_arr):
    return {
        "values": lazy_arr.load(),
    }


@Decoder.register(LazyArray, alias="LazyNumpyArray")
def decode(dct):
    return LazyArray(dct["values"])


@mod.export()
class IterationResult(TypedDict(lambda: str, lambda: LazyArray)):
    """
    An ordered dictionary containing the result of a running a single iteration of a runner.

    This maps output names to arrays, and preserves the output ordering from the runner.

    NOTE: The ``POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB`` environment variable can be set to enable
    the arrays to be swapped to the disk.

    Also includes additional fields indicating the name of the runner which produced the
    outputs, and the time required to do so.
    """

    @staticmethod
    def _to_lazy(nparray):
        if isinstance(nparray, LazyArray):
            return nparray
        return LazyArray(nparray)

    @staticmethod
    def _to_lazy_dict(nparray_dict):
        if nparray_dict is None:
            return None

        # Converts a Dict[str, np.ndarray] to a Dict[str, LazyArray]
        lazy = OrderedDict()
        for name, out in nparray_dict.items():
            lazy[name] = IterationResult._to_lazy(out)
        return lazy

    def __init__(self, outputs=None, runtime=None, runner_name=None):
        """
        Args:
            outputs (Dict[str, Union[np.array, torch.Tensor]]): The outputs of this iteration, mapped to their names.

            runtime (float):
                    The time required for this iteration, in seconds.
                    Only used for logging purposes.
            runner_name (str):
                    The name of the runner that produced this output.
                    If this is omitted, a default name is generated.
        """
        if outputs and config.ARRAY_SWAP_THRESHOLD_MB < 0:
            total_size_gb = sum(
                util.array.nbytes(arr)
                for arr in outputs.values()
                if util.array.is_torch(arr) or util.array.is_numpy(arr)
            ) / (1024.0**3)
            if total_size_gb >= 1:
                G_LOGGER.warning(
                    f"It looks like the outputs of this network are very large ({total_size_gb:.3f} GiB).\n"
                    "To reduce memory usage, you may want to allow Polygraphy to swap these arrays to the disk "
                    "using the POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB environment variable."
                )

        super().__init__(IterationResult._to_lazy_dict(outputs))
        self.runtime = runtime
        self.runner_name = util.default(runner_name, "custom_runner")

    # Convenience methods to preserve np.ndarray in the interface.
    def update(self, other):
        return super().update(IterationResult._to_lazy_dict(other))

    def __setitem__(self, name, arr):
        return super().__setitem__(name, IterationResult._to_lazy(arr))

    def values(self):
        for arr in super().values():
            yield arr.load()

    def items(self):
        for name, arr in super().items():
            yield name, arr.load()

    def __getitem__(self, name):
        return super().__getitem__(name).load()

    def __eq__(self, other):
        if self.runtime != other.runtime or self.runner_name != other.runner_name:
            return False

        for key, val in self.items():
            if key not in other:
                return False

            if not util.array.equal(val, other[key]):
                return False

        return True


@Encoder.register(IterationResult)
def encode(iter_result):
    return {
        "outputs": iter_result.dct,
        "runtime": iter_result.runtime,
        "runner_name": iter_result.runner_name,
    }


@Decoder.register(IterationResult)
def decode(dct):
    return IterationResult(
        outputs=dct["outputs"], runtime=dct["runtime"], runner_name=dct["runner_name"]
    )


@mod.export()
@add_json_methods("inference results")
class RunResults(TypedList(lambda: tuple)):
    """
    Maps runners to per-iteration outputs (in the form of a ``List[IterationResult]``).

    For example, if ``results`` is an instance of ``RunResults()``, then
    to access the outputs of the first iteration from a specified runner, do:
    ::

        iteration = 0
        runner_name = "trt-runner"
        outputs = results[runner_name][iteration]

        # `outputs` is a `Dict[str, np.ndarray]`


    Note: Technically, this is a ``List[Tuple[str, List[IterationResult]]]``, but includes
    helpers that make it behave like an OrderedDict that can contain duplicates.
    """

    def items(self):
        """
        Creates a generator that yields ``Tuple[str, List[IterationResult]]`` - runner names
        and corresponding outputs.
        """
        for name, iteration_results in self.lst:
            yield name, iteration_results

    def keys(self):
        """
        Creates a generator that yields runner names (str).
        """
        for name, _ in self.lst:
            yield name

    def values(self):
        """
        Creates a generator that yields runner outputs (List[IterationResult]).
        """
        for _, iteration_results in self.lst:
            yield iteration_results

    def update(self, other):
        """
        Updates the results stored in this instance.

        Args:
            other (Union[Dict[str, List[IterationResult]], RunResults]):
                    A dictionary or RunResults instance from which to update this one.
        """
        for name, iteration_results in other.items():
            self.lst[name] = iteration_results
        return self

    def add(self, out_list, runtime=None, runner_name=None):
        """
        A helper to create a ``List[IterationResult]`` and map it to the specified runner_name.

        This method cannot be used to modify an existing entry.

        Calling this method is equivalent to:
        ::

            results[runner_name] = []
            for out in out_list:
                results[runner_name].append(IterationResult(out, runtime, runner_name))

        Args:
            out_list (List[Dict[str, np.array]]):
                One or more set of outputs where each output is a dictionary
                of output names mapped to NumPy arrays.

            runtime (float):
                    The time required for this iteration, in seconds.
                    Only used for logging purposes.
            runner_name (str):
                    The name of the runner that produced this output.
                    If this is omitted, a default name is generated.
        """
        runner_name = util.default(runner_name, "custom_runner")
        iter_results = [IterationResult(out, runtime, runner_name) for out in out_list]
        self[runner_name] = iter_results

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.lst[key]

        for name, iteration_results in self.lst:
            if name == key:
                return iteration_results

        G_LOGGER.critical(
            f"{key:35} does not exist in this RunResults instance. Note: Available runners: {list(self.keys())}"
        )

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.lst[key] = value
            return

        for index, name in enumerate(self.keys()):
            if name == key:
                self.lst[index] = (key, value)
                break
        else:
            self.append((key, value))

    def __contains__(self, val):
        if isinstance(val, str) or isinstance(val, bytes):
            return val in list(self.keys())
        return val in self.lst

    def __eq__(self, other):
        for (r0, its0), (r1, its1) in zip(self.lst, other.lst):
            if r0 != r1:
                return False

            if its0 != its1:
                return False
        return True

    def split(self):
        """
        A generator that yields one single-iteration ``RunResults`` per iteration of this run
        (each containing a single ``IterationResult`` per runner).

        Yields:
            RunResults: One per iteration.
        """
        num_iterations = max(
            (len(iter_results) for _, iter_results in self.items()),
            default=0,
        )
        for index in range(num_iterations):
            single = RunResults()
            for name, iter_results in self.items():
                if index < len(iter_results):
                    single.append((name, [iter_results[index]]))
            yield single

    @staticmethod
    def concat(runs):
        """
        Concatenates the iterations of several ``RunResults`` into one (the inverse of ``split``).
        Runners are matched by name, so ragged single-iteration runs from ``split`` (which omits a
        runner that has run out of iterations) are inverted correctly.

        Args:
            runs (Iterable[RunResults]): The runs to concatenate, in iteration order.

        Returns:
            RunResults: The combined run.
        """
        combined = RunResults()
        for run in runs:
            for name, iter_results in run.items():
                if name in combined:
                    combined[name].extend(iter_results)
                else:
                    combined.append((name, list(iter_results)))
        return combined

    @staticmethod
    def load_streaming(path, allow_dirs=True):
        """
        A generator that lazily yields single-iteration ``RunResults`` from a saved source, one
        iteration at a time, so arbitrarily large runs can be streamed without loading everything
        into memory. This is the reading inverse of ``Comparator.run(..., save_outputs_path=...)``.

        The ``path`` may be:

        - A directory, treated as a single run whose per-iteration ``*.json`` files (in index
          order) are each a ``RunResults``.
        - A file containing a ``RunResults``, split into one ``RunResults`` per iteration. The file
          is loaded fully into memory; for constant memory over large runs, stream a directory of
          per-iteration files instead.
        - A list of the above, whose iterations are chained into a single run.

        Args:
            path (Union[str, List[str]]): A file or directory path, or a list of such paths.
            allow_dirs (bool):
                    Whether a directory path is permitted. When False, a directory raises an error
                    instead of being treated as a run of per-iteration files. Defaults to True.

        Yields:
            RunResults: One per iteration, each containing a single ``IterationResult`` per runner.
        """
        yield from iterate_from_files(
            path,
            lambda file_path: RunResults.load(file_path).split(),
            allow_dirs=allow_dirs,
        )


@Encoder.register(RunResults)
def encode(results):
    return {"lst": results.lst}


@Decoder.register(RunResults)
def decode(dct):
    return RunResults(list(map(tuple, dct["lst"])))


@mod.export()
class AccuracyResult(TypedDict(lambda: tuple, lambda: list)):
    """
    An ordered dictionary holding the per-iteration comparison results for a single comparison
    function. ``Comparator.compare_accuracy`` returns an ``AccuracyResults`` (a list of these), one
    per comparison function.

    More specifically, it is an ``OrderedDict[Tuple[str, str], List[OrderedDict[str, bool]]]`` which maps a runner
    pair (a tuple containing both runner names) to a list of dictionaries of booleans (or anything that can be
    converted into a boolean, such as an ``OutputCompareResult``), indicating whether there was a match in the outputs of
    the corresponding iteration. The ``List[OrderedDict[str, bool]]`` is constructed from the dictionaries returned
    by ``compare_func`` in ``compare_accuracy``.

    For example, to see if there's a match between ``runner0`` and
    ``runner1`` during the 1st iteration for an output called ``output0``:
    ::

        runner_pair = ("runner0", "runner1")
        iteration = 0
        output_name = "output0"
        match = bool(accuracy_result[runner_pair][iteration][output_name])

    If there's a mismatch, you can inspect the outputs from
    the results of ``Comparator.run()``, assumed here to be called ``run_results``:
    ::

        runner0_output = run_results["runner0"][iteration][output_name]
        runner1_output = run_results["runner1"][iteration][output_name]

    The accessors summarize a runner pair's results in different ways:

    - ``stats``: per-iteration counts ``(matched, mismatched, total)``.
    - ``average_results``: per-output averaged result objects (each metric averaged across
      iterations, checked against its threshold).
    - ``describe_average``: per-output ``(line, passed)`` summaries of the averaged-metric checks.
    """

    def __init__(self, dct=None, aggregation="per_sample"):
        """
        Args:
            dct:
                    Initial contents: a mapping of runner pair to per-iteration results.
            aggregation (str):
                    Either ``"per_sample"`` (the default) or ``"average"``; controls how
                    ``bool(self)`` and the accuracy summary are computed.
        """
        super().__init__(dct)
        self.aggregation = aggregation

    def __bool__(self):
        """
        Whether all outputs matched.
        You can use this function to avoid manually checking each output. For example:
        ::

            if accuracy_result:
                print("All matched!")

        In the default (per-sample) aggregation mode, this is True only if all outputs matched
        for every iteration. In ``"average"`` aggregation mode (set when ``compare_accuracy`` is
        run with ``check_average=True``), this is True only if every output's averaged metrics
        pass.

        Returns:
            bool
        """
        if self.aggregation == "average":
            return all(
                bool(match)
                for runner_pair in self.keys()
                for match in self.average_results(runner_pair).values()
            )
        return all(
            bool(match)
            for outs in self.values()
            for out in outs
            for match in out.values()
        )

    def _get_runner_pair(self, runner_pair):
        return util.default(runner_pair, list(self.keys())[0])

    @mod.deprecate(
        remove_in="0.55.0",
        use_instead="stats",
        name="AccuracyResult.percentage",
    )
    def percentage(self, runner_pair=None):
        """
        Returns the fraction of iterations that matched for the given pair of runners, as a value
        between 0.0 and 1.0. Always returns 1.0 when there are no iterations or no runner
        comparisons.

        Args:
            runner_pair (Tuple[str, str]):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.

        Returns:
            float
        """
        if not list(self.keys()):
            return 1.0  # No data in this result.
        matched, _, total = self.stats(runner_pair)
        if not total:
            return 1.0  # No iterations.
        return float(matched) / float(total)

    def stats(self, runner_pair=None):
        """
        Returns the number of iterations that matched, mismatched, and the total number of iterations.

        Note: This always reflects per-iteration (per-sample) results, regardless of the
        aggregation mode. For per-output averaged pass/fail results, see ``average_results``.

        Args:
            runner_pair (Tuple[str, str]):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.

        Returns:
            Tuple[int, int, int]: Number of iterations that matched, mismatched, and total respectively.
        """
        runner_pair = self._get_runner_pair(runner_pair)
        outs = self[runner_pair]
        matched = sum(all(out.values()) for out in outs)
        total = len(outs)
        return matched, total - matched, total

    @staticmethod
    def _collect_output_names(iterations):
        # Output names in order of first appearance across iterations.
        return util.unique_list(
            name for iteration in iterations for name in iteration.keys()
        )

    @staticmethod
    def _average_output_fields(iterations, output_name, fields):
        """
        Averages each metric field for a single output across iterations, for an averaged pass/fail
        check. ``NaN`` propagates into the average (so the check fails, as it would per-sample), and
        an iteration with no comparable metrics (e.g. the bare ``False`` from a shape mismatch) fails
        the whole averaged comparison, since averaging over only the comparable iterations would be
        misleading.

        Returns:
            OrderedDict[str, float]: The averaged value of each metric field.
        """
        results = [
            iteration[output_name]
            for iteration in iterations
            if output_name in iteration
        ]
        if any(not hasattr(result, field) for result in results for field in fields):
            G_LOGGER.critical(
                f"Output: {output_name} | Cannot compute an averaged comparison because at "
                f"least one iteration produced no comparable metrics (e.g. a shape "
                f"mismatch). Averaging over only the comparable iterations would be "
                f"misleading, so the comparison cannot be evaluated with check_average=True. "
                f"Re-run without check_average to see the per-iteration results."
            )
        averaged = OrderedDict()
        for field in fields:
            values = [
                value
                for value in (getattr(result, field, None) for result in results)
                if value is not None
            ]
            if values:
                averaged[field] = sum(values) / len(values)
        return averaged

    def _result_metric_fields(self):
        # The metric field names this result's result objects carry, sampled from the first non-bool
        # result; empty if there are only bare booleans (e.g. shape mismatches / the 'indices'
        # comparison), which are not re-thresholdable. Sampling keeps matching/field queries working
        # for --compare-func-script functors. Drives matching to a re-supplied comparison function.
        for outs in self.values():
            for out in outs:
                for result in out.values():
                    if not isinstance(result, bool):
                        return result.metric_fields()
        return []

    def output_names(self):
        """
        Returns:
            List[str]:
                    The names of the compared outputs, in order of first appearance, across all
                    runner pairs.
        """
        return util.unique_list(
            name
            for iterations in self.values()
            for name in self._collect_output_names(iterations)
        )

    def describe_average(self, runner_pair=None):
        """
        Produces a per-output, human-readable description of each averaged-metric check, mirroring
        the per-iteration log lines (metric value, threshold, and PASSED/FAILED).

        The descriptions are derived on demand from the averaged result objects (see
        ``average_results``), which carry their own thresholds, so no comparison function is needed.
        Returns an empty mapping for comparisons whose result type has no single-metric description
        line (e.g. ``simple``).

        Args:
            runner_pair (Tuple[str, str]):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.

        Returns:
            OrderedDict[str, Tuple[str, bool]]:
                    Maps each output name to a ``(line, passed)`` tuple describing this result's
                    single averaged metric.
        """
        runner_pair = self._get_runner_pair(runner_pair)
        described = OrderedDict()
        for output_name, result in self.average_results(runner_pair).items():
            description = result.describe()
            if description is not None:
                described[output_name] = description
        return described

    def average_results(self, runner_pair=None):
        """
        Computes, per output, the result of averaging each metric across iterations and checking it
        against the thresholds. Derived on demand from the per-iteration results -- which carry their
        own metric values and thresholds -- so it needs no comparison function.

        ``NaN`` metric values propagate into the average so the check fails (matching the per-sample
        behavior). If any iteration for an output carries no averageable metrics (such as the
        ``False`` from a shape mismatch), that output's averaged comparison fails loudly, since
        averaging over only the comparable iterations would be misleading.

        Args:
            runner_pair (Tuple[str, str]):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.

        Returns:
            OrderedDict[str, <ResultObject>]:
                    Maps each output name to the aggregated result object.
        """
        runner_pair = self._get_runner_pair(runner_pair)
        fields = self._result_metric_fields()
        iterations = self[runner_pair]

        results = OrderedDict()
        for output_name in self._collect_output_names(iterations):
            averaged = self._average_output_fields(iterations, output_name, fields)
            # Every iteration for an output shares the same per-output thresholds, so reuse them for
            # the averaged result; its verdict then derives from the averaged metrics.
            sample = next(it[output_name] for it in iterations if output_name in it)
            results[output_name] = type(sample)(
                **{field: averaged.get(field) for field in fields},
                thresholds=sample.thresholds,
            )
        return results

    @staticmethod
    def _check_rethresholdable(result, threshold, output_name):
        if isinstance(result, bool) or not result.metric_fields():
            G_LOGGER.critical(
                f"Output: '{output_name}' has no re-thresholdable metrics (e.g. it was a shape "
                f"mismatch or an 'indices' comparison), so it cannot be re-evaluated against new "
                f"thresholds.\nNote: Re-run the comparison from saved raw outputs "
                f"(run --save-outputs / --load-outputs) for such comparisons."
            )
        have = result.metric_fields()
        needed = threshold.metric_fields()
        missing = [field for field in needed if field not in have]
        if missing:
            G_LOGGER.critical(
                f"The requested comparison expects metric field(s) {missing} which are not present "
                f"in the saved results for output: '{output_name}' (it has: {have}).\nNote: If "
                f"these results were produced by a custom comparison function, re-supply it with "
                f"--compare-func-script."
            )

    def reevaluate(self, thresholds, aggregation=None):
        """
        Re-checks every stored result against new thresholds in place by updating each result's
        stored ``thresholds`` (the stored metric values are unchanged); the verdict then re-derives
        from the metrics and the new thresholds. Does not build new result objects.

        Args:
            thresholds (Union[Threshold, Dict[str, Threshold]]):
                    The new ``Threshold`` to check against, applied to every output, or per-output
                    thresholds as a dictionary (with ``""`` as the key for a default).
            aggregation (str):
                    If provided, overrides the aggregation mode (``"per_sample"`` or ``"average"``).

        Returns:
            AccuracyResult: self.
        """
        if aggregation is not None:
            self.aggregation = aggregation
        for runner_pair in list(self.keys()):
            for iteration in self[runner_pair]:
                for output_name, result in iteration.items():
                    threshold = util.value_or_from_dict(thresholds, output_name)
                    if threshold is None:
                        G_LOGGER.critical(
                            f"No threshold was provided for output: '{output_name}'."
                        )
                    self._check_rethresholdable(result, threshold, output_name)
                    # set_thresholds re-derives the verdict from the new thresholds (and drops any
                    # stored verdict, e.g. a saved elemwise result). Averaged verdicts are computed
                    # on demand from these per-iteration results, so there is nothing else to update.
                    result.set_thresholds(threshold)
        return self


@mod.export()
@add_json_methods("accuracy results")
class AccuracyResults(TypedList(lambda: AccuracyResult)):
    """
    A list of ``AccuracyResult`` objects -- one per comparison function -- as returned by
    ``Comparator.compare_accuracy``.

    Behaves like a list (iteration, indexing, ``len``), except ``bool(results)`` is True only if
    *every* result passed, so ``if results:`` is a correct overall pass/fail check (mirroring
    ``bool(AccuracyResult)``).

    Can be saved to and loaded from JSON via ``save``/``load``. A loaded ``AccuracyResults`` carries
    the per-iteration metric values and saved verdicts; ``reevaluate`` re-checks them against new
    thresholds (see ``polygraphy check accuracy``).

    Note: This wraps its list in ``.lst`` (rather than subclassing ``list``) so that the JSON encoder
    is actually invoked -- ``json`` serializes ``list`` subclasses directly, bypassing custom
    encoders.
    """

    def __bool__(self):
        return all(bool(result) for result in self)

    def output_names(self):
        """
        Returns:
            List[str]:
                    The union of compared output names across all contained results, in order of
                    first appearance.
        """
        return util.unique_list(
            name for result in self for name in result.output_names()
        )

    @staticmethod
    def _threshold_metric_fields(threshold):
        # The metric fields a threshold checks. A per-output {output_name: Threshold} dict is keyed
        # by output name, but every entry is the same threshold type, so any entry identifies it.
        if isinstance(threshold, dict):
            if not threshold:
                G_LOGGER.critical("A per-output threshold dictionary cannot be empty.")
            threshold = next(iter(threshold.values()))
        return list(threshold.metric_fields())

    def reevaluate(self, thresholds, aggregation=None):
        """
        Re-checks the requested comparisons against new thresholds, in place.

        Each provided threshold selects the saved comparison it applies to -- matched by the metric
        fields it checks -- and that comparison is re-evaluated against it. Saved comparisons that no
        provided threshold selects are dropped, so e.g. re-checking only ``l2`` and
        ``cosine_similarity`` ignores any other metrics in the file. It is an error to provide a
        threshold that matches no saved comparison (you asked to check data that is not present).

        Args:
            thresholds (Union[Threshold, Dict[str, Threshold], Sequence[Union[Threshold, Dict[str, Threshold]]]]):
                    The new threshold(s) to check against. Each is a single ``Threshold`` or a
                    per-output ``{output_name: Threshold}`` dictionary; order does not matter.
            aggregation (str):
                    If provided, overrides the aggregation mode of every checked result.

        Returns:
            AccuracyResults: self.
        """
        threshold_list = (
            list(thresholds) if util.is_sequence(thresholds) else [thresholds]
        )
        remaining = list(self)
        checked = []
        for threshold in threshold_list:
            fields = self._threshold_metric_fields(threshold)
            match = next(
                (r for r in remaining if list(r._result_metric_fields()) == fields),
                None,
            )
            if match is None:
                G_LOGGER.critical(
                    f"No saved comparison results match the requested threshold (metric fields: "
                    f"{fields}).\nNote: the saved results contain these metric fields: "
                    f"{[r._result_metric_fields() for r in self]}. If these results came from a "
                    f"custom comparison function, its threshold type is reachable via "
                    f"--compare-func-script."
                )
            remaining.remove(match)
            match.reevaluate(threshold, aggregation=aggregation)
            checked.append(match)

        # Keep only the comparisons that were re-checked; any others in the file were not requested
        # and should not affect the overall pass/fail or the summary.
        for result in remaining:
            G_LOGGER.info(
                f"Ignoring saved comparison with metric fields {result._result_metric_fields()}: "
                f"no matching threshold was requested."
            )
        self.lst = checked
        return self


@Encoder.register(AccuracyResult)
def encode(result):
    # Tuple runner-pair keys cannot be JSON object keys, so store the mapping as a list of
    # [pair, iterations] (mirroring RunResults). compare_func and the average cache are transient
    # and intentionally not serialized.
    return {
        "lst": [[list(pair), iterations] for pair, iterations in result.dct.items()],
        "aggregation": result.aggregation,
    }


@Decoder.register(AccuracyResult)
def decode(dct):
    data = OrderedDict((tuple(pair), iterations) for pair, iterations in dct["lst"])
    return AccuracyResult(data, aggregation=dct["aggregation"])


@Encoder.register(AccuracyResults)
def encode(results):
    return {"lst": results.lst}


@Decoder.register(AccuracyResults)
def decode(dct):
    return AccuracyResults(dct["lst"])
