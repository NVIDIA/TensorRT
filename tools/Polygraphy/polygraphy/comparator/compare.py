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
import math
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.comparator import util as comp_util
from polygraphy.datatype import DataType
from polygraphy.json import Decoder, Encoder
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")
torch = mod.lazy_import("torch")
lpips = mod.lazy_import("lpips")


@mod.export()
class Threshold:
    """
    Base class for a comparison threshold -- the criterion a comparison's metric(s) are checked
    against. A threshold knows the metric field(s) it applies to (``metric_fields``), how to turn a
    result's metric values into a pass/fail verdict (``passed``), and how to format a human-readable
    summary line (``describe``).

    Comparison functions produce thresholds (see ``BaseCompareFunc.thresholds_for``) and results store
    them, so a saved result can be re-checked against a new threshold -- and described -- without the
    comparison function that produced it.
    """

    # The metric field names this threshold checks; this also identifies the result type it applies
    # to (it matches that result's ``metric_fields``).
    METRIC_FIELDS = []

    @classmethod
    def metric_fields(cls):
        """
        Returns:
            List[str]: The names of the metric fields this threshold checks.
        """
        return list(cls.METRIC_FIELDS)

    def passed(self, metric_values):
        """
        Computes the pass/fail verdict for a set of metric values.

        Args:
            metric_values (Dict[str, float]): A mapping of metric field name to value.

        Returns:
            bool
        """
        raise NotImplementedError()

    def describe(self, metric_values):
        """
        Returns a ``(line, passed)`` human-readable summary of a single-metric check against these
        metric values, or ``None`` if this threshold has no single-metric summary line (e.g.
        ``SimpleThreshold``).
        """
        return None


@mod.export()
class SimpleThreshold(Threshold):
    """
    Absolute/relative tolerances on a chosen error statistic -- the threshold for the ``simple``
    comparison (``SimpleCompareFunc`` / ``OutputCompareResult``).
    """

    METRIC_FIELDS = [
        "max_absdiff",
        "max_reldiff",
        "mean_absdiff",
        "mean_reldiff",
        "median_absdiff",
        "median_reldiff",
        "quantile_absdiff",
        "quantile_reldiff",
    ]

    def __init__(self, check_error_stat, atol, rtol):
        """
        Args:
            check_error_stat (str): The error statistic to check (``max``/``mean``/``median``/``quantile``).
            atol (float): The absolute tolerance.
            rtol (float): The relative tolerance.
        """
        self.check_error_stat = check_error_stat
        self.atol = atol
        self.rtol = rtol

    def passed(self, metric_values):
        """
        Checks the chosen error statistic's stored absolute/relative difference against the tolerances.

        ``"elemwise"`` is not supported here: it is a per-element check with no scalar statistic, so
        it cannot be evaluated from summary statistics. Choose ``max``/``mean``/``median``/``quantile``,
        or re-run the comparison from saved raw outputs.
        """
        if self.check_error_stat == "elemwise":
            G_LOGGER.critical(
                "Cannot re-threshold an 'elemwise' comparison: it is a per-element check with no "
                "scalar statistic to threshold.\nNote: Choose an explicit statistic "
                "(--check-error-stat max, mean, median, or quantile), or re-run the comparison from "
                "saved raw outputs (run --save-outputs / --load-outputs)."
            )
        if self.check_error_stat not in ("max", "mean", "median", "quantile"):
            G_LOGGER.critical(
                f"Invalid choice for check_error_stat: {self.check_error_stat}.\n"
                f"Note: Valid choices are: {['max', 'mean', 'median', 'quantile']}"
            )
        abs_val = metric_values.get(f"{self.check_error_stat}_absdiff")
        rel_val = metric_values.get(f"{self.check_error_stat}_reldiff")
        if abs_val is None or rel_val is None:
            G_LOGGER.critical(
                f"The saved results do not contain a '{self.check_error_stat}' statistic, so they "
                f"cannot be checked against it.\nNote: 'quantile' statistics are only saved when the "
                f"original comparison used --check-error-stat quantile. Choose max, mean, or median "
                f"instead, or re-run the comparison from saved raw outputs."
            )
        return not (
            _simple_stat_failed(abs_val, self.atol)
            and _simple_stat_failed(rel_val, self.rtol)
        )


class _MetricThreshold(Threshold):
    """
    Base class for single-scalar-metric thresholds (L2 norm, cosine similarity, etc.). Subclasses set
    ``_METRIC_FIELD`` (the metric name), ``_METRIC_LABEL``, ``_METRIC_UNIT``, and ``_HIGHER_IS_BETTER``
    (whether a larger value passes).
    """

    _METRIC_FIELD = None
    _METRIC_LABEL = None
    _METRIC_UNIT = ""
    _HIGHER_IS_BETTER = False

    def __init__(self, threshold):
        """
        Args:
            threshold (float): The value the metric is checked against.
        """
        self.threshold = threshold

    @classmethod
    def metric_fields(cls):
        return [cls._METRIC_FIELD]

    def passed(self, metric_values):
        # A None metric (e.g. a computation that could not run) fails the check.
        value = metric_values.get(self._METRIC_FIELD)
        if value is None:
            return False
        return bool(
            value >= self.threshold
            if self._HIGHER_IS_BETTER
            else value <= self.threshold
        )

    @staticmethod
    def _format_pair(value, threshold):
        # Format the value and threshold with enough significant figures to render distinctly when
        # they actually differ, so a near-boundary PASS/FAIL is not shown as value == threshold.
        # Uses the usual 5 significant figures and only adds precision as needed.
        for precision in range(5, 18):
            value_str = f"{value:.{precision}g}"
            threshold_str = f"{threshold:.{precision}g}"
            if value == threshold or value_str != threshold_str:
                return value_str, threshold_str
        return f"{value:.17g}", f"{threshold:.17g}"

    def describe(self, metric_values):
        value = metric_values.get(self._METRIC_FIELD)
        if value is None:
            return None
        passed = self.passed(metric_values)
        unit = self._METRIC_UNIT
        threshold_label = "min required" if self._HIGHER_IS_BETTER else "tolerance"
        value_str, threshold_str = self._format_pair(value, self.threshold)
        line = (
            f"{self._METRIC_LABEL}: {value_str}{unit} "
            f"({threshold_label}: {threshold_str}{unit}) | "
            f"{'PASSED' if passed else 'FAILED'}"
        )
        return line, passed


@mod.export()
class L2Threshold(_MetricThreshold):
    """Maximum allowed L2 norm (``L2CompareFunc`` / ``L2Result``)."""

    _METRIC_FIELD = "l2_norm"
    _METRIC_LABEL = "L2 Norm"
    _HIGHER_IS_BETTER = False


@mod.export()
class CosineSimilarityThreshold(_MetricThreshold):
    """Minimum required cosine similarity (``CosineSimilarityCompareFunc`` / ``CosineSimilarityResult``)."""

    _METRIC_FIELD = "cosine_similarity"
    _METRIC_LABEL = "Cosine Similarity"
    _HIGHER_IS_BETTER = True


@mod.export()
class PsnrThreshold(_MetricThreshold):
    """Minimum required PSNR (``PsnrCompareFunc`` / ``PsnrResult``)."""

    _METRIC_FIELD = "psnr"
    _METRIC_LABEL = "PSNR"
    _METRIC_UNIT = " dB"
    _HIGHER_IS_BETTER = True


@mod.export()
class SnrThreshold(_MetricThreshold):
    """Minimum required SNR (``SnrCompareFunc`` / ``SnrResult``)."""

    _METRIC_FIELD = "snr"
    _METRIC_LABEL = "SNR"
    _METRIC_UNIT = " dB"
    _HIGHER_IS_BETTER = True


@mod.export()
class LpipsThreshold(_MetricThreshold):
    """Maximum allowed LPIPS (``PerceptualMetricsCompareFunc`` / ``PerceptualMetricsResult``)."""

    _METRIC_FIELD = "lpips"
    _METRIC_LABEL = "LPIPS"
    _HIGHER_IS_BETTER = False

    def passed(self, metric_values):
        # A None value (LPIPS computation failed/skipped) is treated as a pass.
        value = metric_values.get("lpips")
        return value is None or bool(value <= self.threshold)


@mod.export()
class CompareResult:
    """
    Base class for the result of comparing a single output between two runners. Subclasses record
    their metric value(s); this base holds the ``thresholds`` (a ``Threshold``) the result was checked
    against and the two compared output names (populated by ``run_comparison``).

    The verdict (``bool(result)``) and the per-metric summary (``describe``) are computed by the
    stored ``Threshold`` from the result's metric values, so a saved result can be re-checked against
    a new threshold without the comparison function that produced it. ``OutputCompareResult`` with the
    ``elemwise`` check is the one exception: its per-element verdict cannot be reconstructed from the
    summary statistics, so it is stored.
    """

    # The Threshold subclass whose criterion applies to this result. It is the single source of truth
    # for this result's metric fields (and constructor order); ``metric_fields``/``metric_values`` and
    # averaged-result construction derive from it. None for the base class.
    _THRESHOLD_CLASS = None

    def __init__(self, thresholds=None):
        # thresholds: the Threshold this result was checked against (see thresholds_for).
        self.thresholds = thresholds

        self.output0_name = None
        self.output1_name = None

    def __bool__(self):
        return self.thresholds.passed(self.metric_values())

    def set_thresholds(self, thresholds):
        """Updates the threshold this result is checked against (used by re-thresholding)."""
        self.thresholds = thresholds

    @classmethod
    def metric_fields(cls):
        """
        Returns:
            List[str]:
                    The names of the scalar metric fields this result type carries, taken from its
                    associated ``Threshold`` class.
        """
        return cls._THRESHOLD_CLASS.metric_fields() if cls._THRESHOLD_CLASS else []

    def metric_values(self):
        """
        Returns:
            OrderedDict[str, float]: The stored value of each metric field (see ``metric_fields``).
        """
        return OrderedDict(
            (field, getattr(self, field)) for field in self.metric_fields()
        )

    def describe(self):
        """
        Returns a ``(line, passed)`` human-readable summary, or ``None`` if this result's threshold
        has no single-metric summary line (e.g. ``simple``).
        """
        return self.thresholds.describe(self.metric_values())


@mod.export()
class OutputCompareResult(CompareResult):
    """
    Represents the result of comparing a single output of a single iteration
    between two runners.
    """

    _THRESHOLD_CLASS = SimpleThreshold

    def __init__(
        self,
        max_absdiff,
        max_reldiff,
        mean_absdiff,
        mean_reldiff,
        median_absdiff,
        median_reldiff,
        quantile_absdiff,
        quantile_reldiff,
        thresholds=None,
        passed=None,
    ):
        """
        Records the statistics gathered during comparison and the threshold checked against.

        Args:
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
            thresholds (SimpleThreshold): The threshold checked against.
            passed (bool):
                    The stored verdict. Only set for the ``elemwise`` check, whose per-element
                    verdict cannot be derived from the summary statistics; otherwise the verdict is
                    computed by the threshold from the metrics.
        """
        super().__init__(thresholds=thresholds)
        # Unlike the other result types, the elemwise verdict cannot be derived from the stored
        # summary statistics, so it is stored here. None means "let the threshold derive it".
        self._stored_passed = passed

        # Coerce metrics to Python floats so results/JSON stay backend-agnostic (the stats are
        # computed as numpy/torch scalars). None (e.g. an uncomputed quantile) is preserved.
        def _to_float(value):
            return None if value is None else float(value)

        self.max_absdiff = _to_float(max_absdiff)
        self.max_reldiff = _to_float(max_reldiff)
        self.mean_absdiff = _to_float(mean_absdiff)
        self.mean_reldiff = _to_float(mean_reldiff)
        self.median_absdiff = _to_float(median_absdiff)
        self.median_reldiff = _to_float(median_reldiff)
        self.quantile_absdiff = _to_float(quantile_absdiff)
        self.quantile_reldiff = _to_float(quantile_reldiff)

    def __str__(self):
        return f"(atol={self.max_absdiff}, rtol={self.max_reldiff})"

    def __bool__(self):
        # The elemwise verdict is stored (not derivable); the scalar statistics derive on demand.
        if self._stored_passed is not None:
            return self._stored_passed
        return super().__bool__()

    def set_thresholds(self, thresholds):
        # Re-thresholding to a scalar statistic: drop any stored elemwise verdict so the verdict
        # derives from the metrics and the new threshold.
        super().set_thresholds(thresholds)
        self._stored_passed = None


@mod.export()
class PerceptualMetricsResult(CompareResult):
    """
    Represents the result of comparing a single output using perceptual metrics
    between two runners.
    """

    _THRESHOLD_CLASS = LpipsThreshold

    def __init__(self, lpips=None, thresholds=None):
        """
        Args:
            lpips (float):
                    The Learned Perceptual Image Patch Similarity score between the outputs. Lower
                    values indicate more perceptually similar outputs. May be None if LPIPS
                    computation failed.
            thresholds (LpipsThreshold): The threshold checked against.
        """
        super().__init__(thresholds=thresholds)
        self.lpips = lpips


@mod.export()
class L2Result(CompareResult):
    """
    Represents the result of comparing a single output using the L2 norm (Euclidean distance)
    between two runners.
    """

    _THRESHOLD_CLASS = L2Threshold

    def __init__(self, l2_norm, thresholds=None):
        """
        Args:
            l2_norm (float): The L2 norm (Euclidean distance) between the outputs.
            thresholds (L2Threshold): The threshold checked against.
        """
        super().__init__(thresholds=thresholds)
        self.l2_norm = l2_norm


@mod.export()
class CosineSimilarityResult(CompareResult):
    """
    Represents the result of comparing a single output using cosine similarity between two
    runners.
    """

    _THRESHOLD_CLASS = CosineSimilarityThreshold

    def __init__(self, cosine_similarity, thresholds=None):
        """
        Args:
            cosine_similarity (float): The cosine similarity between the outputs.
            thresholds (CosineSimilarityThreshold): The threshold checked against.
        """
        super().__init__(thresholds=thresholds)
        self.cosine_similarity = cosine_similarity


@mod.export()
class PsnrResult(CompareResult):
    """
    Represents the result of comparing a single output using PSNR (Peak Signal-to-Noise Ratio)
    between two runners.
    """

    _THRESHOLD_CLASS = PsnrThreshold

    def __init__(self, psnr, thresholds=None):
        """
        Args:
            psnr (float): The Peak Signal-to-Noise Ratio between the outputs.
            thresholds (PsnrThreshold): The threshold checked against.
        """
        super().__init__(thresholds=thresholds)
        self.psnr = psnr


@mod.export()
class SnrResult(CompareResult):
    """
    Represents the result of comparing a single output using SNR (Signal-to-Noise Ratio)
    between two runners.
    """

    _THRESHOLD_CLASS = SnrThreshold

    def __init__(self, snr, thresholds=None):
        """
        Args:
            snr (float): The Signal-to-Noise Ratio between the outputs.
            thresholds (SnrThreshold): The threshold checked against.
        """
        super().__init__(thresholds=thresholds)
        self.snr = snr


def _register_threshold(typ, fields):
    @Encoder.register(typ)
    def encode(threshold):
        return {field: getattr(threshold, field) for field in fields}

    @Decoder.register(typ)
    def decode(dct):
        return typ(*(dct[field] for field in fields))


_register_threshold(SimpleThreshold, ["check_error_stat", "atol", "rtol"])
for _threshold_type in [
    L2Threshold,
    CosineSimilarityThreshold,
    PsnrThreshold,
    SnrThreshold,
    LpipsThreshold,
]:
    _register_threshold(_threshold_type, ["threshold"])


@Encoder.register(OutputCompareResult)
def encode(result):
    dct = {
        field: getattr(result, field) for field in OutputCompareResult.metric_fields()
    }
    dct["thresholds"] = result.thresholds
    # The verdict is computed by the threshold except for the 'elemwise' check, where it is stored;
    # persist whatever was stored (None for the derivable cases).
    dct["passed"] = result._stored_passed
    return dct


@Decoder.register(OutputCompareResult)
def decode(dct):
    return OutputCompareResult(
        *(dct[field] for field in OutputCompareResult.metric_fields()),
        thresholds=dct["thresholds"],
        passed=dct["passed"],
    )


def _register_single_metric_result(typ):
    (field,) = typ.metric_fields()

    @Encoder.register(typ)
    def encode(result):
        # The verdict is computed by the threshold, so only the metric and threshold are stored.
        return {field: getattr(result, field), "thresholds": result.thresholds}

    @Decoder.register(typ)
    def decode(dct):
        return typ(dct[field], thresholds=dct["thresholds"])


for _result_type in [
    PerceptualMetricsResult,
    L2Result,
    CosineSimilarityResult,
    PsnrResult,
    SnrResult,
]:
    _register_single_metric_result(_result_type)


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


def run_comparison(
    func, fail_fast, iter_result0, iter_result1, find_output_func, func_name=None
):
    """
    Iterates over all the generated outputs and runs `func` to compare them.

    Args:
        func_name (str):
                An optional label (e.g. the metric name) included in the final pass/fail summary
                to distinguish it when several comparison functions are run. Defaults to None.
    """
    label = f"{func_name} | " if func_name else ""
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

            # Prefix with the metric/compare-func label so that, when several comparison functions
            # each compare the same output, it is clear which one this line belongs to.
            G_LOGGER.start(
                f"{label}Comparing Output: '{out0_name}' (dtype={util.array.dtype(output0)}, shape={util.array.shape(output0)}) with '{out1_name}' (dtype={util.array.dtype(output1)}, shape={util.array.shape(output1)})"
            )

            with G_LOGGER.indent():
                result = func(out0_name, output0, out1_name, output1)
                # Record matched names on result objects so the Comparator can log raw output stats
                # later. Bare bools (e.g. shape mismatch, indices) can't carry attributes.
                if not isinstance(result, bool):
                    result.output0_name = out0_name
                    result.output1_name = out1_name
                output_status[out0_name] = result
                if fail_fast and not output_status[out0_name]:
                    return output_status

    mismatched_output_names = [
        name for name, matched in output_status.items() if not matched
    ]
    if mismatched_output_names:
        G_LOGGER.error(f"FAILED | {label}Mismatched outputs: {mismatched_output_names}")
    else:
        G_LOGGER.finish(
            f"PASSED | {label}All outputs matched | Outputs: {list(output_status.keys())}"
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


def _simple_stat_failed(diff, tol):
    return util.array.isnan(diff) or diff > tol


@mod.export()
class BaseCompareFunc:
    """
    Base class for comparison functors.

    A comparison functor compares two ``IterationResult`` s and returns an
    ``OrderedDict[str, <ResultObject>]`` mapping output names to result objects (or anything
    convertible to a boolean) indicating whether the corresponding output matched. Instances
    are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.

    Subclass to define custom comparisons. The comparison criterion lives on the ``Threshold`` the
    functor produces (its ``passed``/``describe``); a functor only computes metric values and, via
    ``thresholds_for``, supplies a ``Threshold``. To support average accuracy checks
    (``Comparator.compare_accuracy(..., check_average=True)``) and re-thresholding of saved results,
    set ``_RESULT_CLASS`` (the ``CompareResult`` subclass produced) and implement ``thresholds_for``
    to return a ``Threshold``; this is duck-typed (it does not require subclassing this class), but
    subclassing documents the protocol.
    """

    # The result type this functor produces. None means the functor produces bare booleans (e.g.
    # indices) that cannot be averaged or re-thresholded.
    _RESULT_CLASS = None

    def __call__(self, iter_result0, iter_result1):
        """
        Performs a per-iteration comparison.

        Args:
            iter_result0 (IterationResult): The result of the first runner.
            iter_result1 (IterationResult): The result of the second runner.

        Returns:
            OrderedDict[str, <ResultObject>]:
                    A mapping of output names to result objects indicating whether the
                    corresponding output matched.
        """
        raise NotImplementedError()

    def thresholds_for(self, output_name):
        """
        Returns the (per-output) ``Threshold`` this functor checks against. The threshold carries the
        comparison criterion (``passed``/``describe``) and the metric fields it applies to, and is
        stored on the result. This is reachable through ``InvokeFromScript`` (which only forwards
        public names), so it is how ``--compare-func-script`` functors expose their thresholds.

        Args:
            output_name (str): The name of the output being evaluated.

        Returns:
            Threshold
        """
        raise NotImplementedError()

    _SUMMARY_LABEL = None

    def _summary_label(self):
        return self._SUMMARY_LABEL

    @staticmethod
    def _warn_on_unknown_keys(dct, dict_name, valid_keys):
        # Warn (but do not fail) if a per-output option dictionary references output names that
        # are not present in either iteration result.
        if isinstance(dct, dict):
            util.check_sequence_contains(
                dct.keys(),
                valid_keys,
                name=dict_name,
                log_func=G_LOGGER.warning,
                check_missing=False,
            )

    def _align_shapes(self, out0_name, output0, output1):
        shape0 = util.array.shape(output0)
        shape1 = util.array.shape(output1)
        if self.check_shapes and shape0 != shape1:
            G_LOGGER.error(
                f"FAILED | Output: `{out0_name}` | Will not compare outputs of different shapes.\n"
                f"Note: Output shapes are {shape0} and {shape1}."
            )
            G_LOGGER.error(
                "Note: Use --no-shape-check or set check_shapes=False to "
                "attempt to compare values anyway.",
                mode=LogMode.ONCE,
            )
            return None

        output1 = util.try_match_shape(output1, shape0)
        output0 = util.array.view(
            output0,
            DataType.from_dtype(util.array.dtype(output0)),
            util.array.shape(output1),
        )
        return output0, output1

    def _run_comparison(self, iter_result0, iter_result1, match):
        if not self.check_shapes:
            G_LOGGER.info(
                "Strict shape checking disabled. Will attempt to match output shapes before comparisons",
                mode=LogMode.ONCE,
            )
        find_output_func = util.default(
            self.find_output_func,
            functools.partial(default_find_output_func, base_iter_result=iter_result0),
        )
        return run_comparison(
            match,
            self.fail_fast,
            iter_result0,
            iter_result1,
            find_output_func,
            func_name=self._summary_label(),
        )


def metric_fields_for_func(func):
    """
    Returns the metric field names a comparison function consumes (via the ``Threshold`` it
    produces), or ``None`` if the function exposes no threshold protocol at all (e.g. a plain
    ``--compare-func-script`` function). A function that has the protocol but produces no averageable
    metric (e.g. ``indices``) returns an empty list.

    This distinction drives both the average-comparison support check and the matching of saved
    results to a re-supplied comparison function. ``thresholds_for`` is public so it is reachable
    through ``InvokeFromScript`` (which only forwards public names).
    """
    thresholds_for = getattr(func, "thresholds_for", None)
    if thresholds_for is None:
        return None
    try:
        return list(thresholds_for("").metric_fields())
    except NotImplementedError:
        return []


@mod.export()
class SimpleCompareFunc(BaseCompareFunc):
    """
    Compares two IterationResults using absolute/relative tolerances on a chosen error statistic.

    Instances are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.
    """

    _DEFAULT_RTOL = 1e-5
    _DEFAULT_ATOL = 1e-5
    _DEFAULT_QUANTILE = 0.99
    _DEFAULT_ERROR_STAT = "elemwise"
    _SUMMARY_LABEL = "simple"
    _RESULT_CLASS = OutputCompareResult

    def __init__(
        self,
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
        """
        self.check_shapes = util.default(check_shapes, True)
        self.rtol = util.default(rtol, self._DEFAULT_RTOL)
        self.atol = util.default(atol, self._DEFAULT_ATOL)
        self.error_quantile = util.default(error_quantile, self._DEFAULT_QUANTILE)
        self.fail_fast = util.default(fail_fast, False)
        self.find_output_func = find_output_func
        self.check_error_stat = util.default(check_error_stat, self._DEFAULT_ERROR_STAT)
        self.infinities_compare_equal = util.default(infinities_compare_equal, False)
        self.save_heatmaps = save_heatmaps
        self.show_heatmaps = util.default(show_heatmaps, False)
        self.save_error_metrics_plot = save_error_metrics_plot
        self.show_error_metrics_plot = util.default(show_error_metrics_plot, False)

    def _per_output_options(self, output_name):
        return (
            util.value_or_from_dict(self.atol, output_name, self._DEFAULT_ATOL),
            util.value_or_from_dict(self.rtol, output_name, self._DEFAULT_RTOL),
            util.value_or_from_dict(
                self.check_error_stat, output_name, self._DEFAULT_ERROR_STAT
            ),
            util.value_or_from_dict(
                self.error_quantile, output_name, self._DEFAULT_QUANTILE
            ),
        )

    def thresholds_for(self, output_name):
        per_out_atol, per_out_rtol, per_out_err_stat, _ = self._per_output_options(
            output_name
        )
        return SimpleThreshold(per_out_err_stat, per_out_atol, per_out_rtol)

    def _check_outputs_match(
        self,
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
                util.array.subtract(comp_util.cast_up(out0), comp_util.cast_up(out1))
            )
            if self.infinities_compare_equal:
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

        if per_out_err_stat == "quantile":
            quantile_reldiff = comp_util.compute_quantile(reldiff, per_out_quantile)
            quantile_absdiff = comp_util.compute_quantile(absdiff, per_out_quantile)

        # The result carries the computed statistics and the threshold checked against. For the
        # scalar statistics the verdict is computed by the threshold (SimpleThreshold.passed). The
        # elemwise verdict is a per-element check that needs the arrays, so it is computed here and
        # stored on the result (it cannot be re-derived from the summary statistics).
        result = OutputCompareResult(
            max_absdiff,
            max_reldiff,
            mean_absdiff,
            mean_reldiff,
            median_absdiff,
            median_reldiff,
            quantile_absdiff,
            quantile_reldiff,
            thresholds=SimpleThreshold(per_out_err_stat, per_out_atol, per_out_rtol),
        )
        if per_out_err_stat == "elemwise":
            mismatches = (
                util.array.greater(absdiff, per_out_atol) | util.array.isnan(absdiff)
            ) & (util.array.greater(reldiff, per_out_rtol) | util.array.isnan(reldiff))

            result._stored_passed = not bool(util.array.any(mismatches))
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
                G_LOGGER.warning(f"Failing to log mismatches.\nNote: Error was: {err}")
        failed = not bool(result)

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

            if self.save_error_metrics_plot or self.show_error_metrics_plot:
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
                        save_dir=self.save_error_metrics_plot,
                        show=self.show_error_metrics_plot,
                    )

            def build_heatmaps(diff, min_diff, max_diff, prefix, use_lognorm=None):
                if self.save_heatmaps or self.show_heatmaps:
                    with G_LOGGER.indent():
                        comp_util.build_heatmaps(
                            diff,
                            min_diff,
                            max_diff,
                            prefix=f"{prefix} Error | {out0_name}",
                            save_dir=self.save_heatmaps,
                            show=self.show_heatmaps,
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
        return result

    def __call__(self, iter_result0, iter_result1):
        valid_keys = set(iter_result0.keys()) | set(iter_result1.keys()) | {""}
        self._warn_on_unknown_keys(self.rtol, "the rtol dictionary", valid_keys)
        self._warn_on_unknown_keys(self.atol, "the atol dictionary", valid_keys)
        self._warn_on_unknown_keys(
            self.check_error_stat, "the check_error_stat dictionary", valid_keys
        )
        self._warn_on_unknown_keys(
            self.error_quantile, "the quantile dictionary", valid_keys
        )

        def match(out0_name, output0, out1_name, output1):
            per_out_atol, per_out_rtol, per_out_err_stat, per_out_quantile = (
                self._per_output_options(out0_name)
            )

            G_LOGGER.info(
                f"Tolerance: [abs={per_out_atol:.5g}, rel={per_out_rtol:.5g}] | Checking {per_out_err_stat} error"
            )
            G_LOGGER.extra_verbose(
                f"Note: Comparing {iter_result0.runner_name} vs. {iter_result1.runner_name}"
            )

            aligned = self._align_shapes(out0_name, output0, output1)
            if aligned is None:
                return False
            output0, output1 = aligned

            outputs_matched = self._check_outputs_match(
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

        return self._run_comparison(iter_result0, iter_result1, match)


@mod.export()
class IndicesCompareFunc(BaseCompareFunc):
    """
    Compares two IterationResults containing indices, e.g. the outputs of a Top-K operation.

    Instances are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.
    """

    _SUMMARY_LABEL = "indices"

    def __init__(self, index_tolerance=None, fail_fast=None):
        """
        Compares two IterationResults containing indices, and can be used as the ``compare_func``
        argument in ``Comparator.compare_accuracy``. This can be useful to compare, for example,
        the outputs of a Top-K operation.

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
        """
        self.index_tolerance = util.default(index_tolerance, 0)
        self.fail_fast = util.default(fail_fast, False)

    # No _RESULT_CLASS: the 'indices' comparison produces bare booleans, so it has no averageable
    # metric and cannot be used with average comparison or re-thresholded.

    def __call__(self, iter_result0, iter_result1):
        def match(out0_name, output0, out1_name, output1):
            per_out_index_tol = util.value_or_from_dict(
                self.index_tolerance, out0_name, 0
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

                    index1 = util.array.ravel(util.array.argwhere(out1_vals == val0))
                    if util.array.size(index1) < 1:
                        G_LOGGER.error(f"FAILED | Value: {val0} not found in output")
                        passed = False
                        if self.fail_fast:
                            return False
                        continue

                    index1 = index1[0]

                    if abs(index1 - index0) > per_out_index_tol:
                        G_LOGGER.error(
                            f"FAILED | Difference exceeds index tolerance ({per_out_index_tol})"
                        )
                        passed = False
                        if self.fail_fast:
                            return False
                        continue

            if passed:
                G_LOGGER.finish(
                    f"PASSED | Difference is within index tolerance ({per_out_index_tol})"
                )
            return passed

        return run_comparison(
            match,
            self.fail_fast,
            iter_result0,
            iter_result1,
            functools.partial(default_find_output_func, base_iter_result=iter_result0),
            func_name=self._summary_label(),
        )


class _SingleMetricCompareFunc(BaseCompareFunc):
    """Base class for single-scalar-metric comparison functors.

    The comparison criterion and formatting live on the threshold type (``_THRESHOLD_CLASS``, a
    ``_MetricThreshold``); the functor computes the metric and supplies a threshold, which is stored
    on the result type (``_RESULT_CLASS``).
    """

    # Subclasses override these:
    _RESULT_CLASS = None  # result class to construct, e.g. L2Result
    _THRESHOLD_CLASS = None  # threshold class to construct, e.g. L2Threshold
    _DEFAULT_THRESHOLD = None  # subclasses must set a concrete default threshold
    _THRESHOLD_ARG = "threshold"  # public argument name, used in warnings

    def _summary_label(self):
        return self._THRESHOLD_CLASS._METRIC_LABEL

    def __init__(
        self, threshold=None, check_shapes=None, fail_fast=None, find_output_func=None
    ):
        self.threshold = util.default(threshold, self._DEFAULT_THRESHOLD)
        self.check_shapes = util.default(check_shapes, True)
        self.fail_fast = util.default(fail_fast, False)
        self.find_output_func = find_output_func

    def _compute_metric(self, out0, out1):
        raise NotImplementedError()

    def thresholds_for(self, output_name):
        return self._THRESHOLD_CLASS(
            util.value_or_from_dict(
                self.threshold, output_name, self._DEFAULT_THRESHOLD
            )
        )

    def _check_outputs_match(self, out0, out1, threshold):
        # Coerce to a Python float so results/JSON stay backend-agnostic and a NaN metric fails the
        # threshold check rather than leaking a torch/numpy scalar.
        value = float(self._compute_metric(out0, out1))
        # The threshold owns the criterion and the per-metric log line; the verdict is computed by it
        # from the stored metric.
        result = self._RESULT_CLASS(value, thresholds=threshold)
        msg, passed = result.describe()

        # Emit at FINISH/ERROR (like SimpleCompareFunc) so per-output failures surface at ERROR.
        (G_LOGGER.finish if passed else G_LOGGER.error)(msg)

        return result

    def __call__(self, iter_result0, iter_result1):
        if isinstance(self.threshold, dict):
            valid_keys = set(iter_result0.keys()) | set(iter_result1.keys()) | {""}
            self._warn_on_unknown_keys(
                self.threshold, f"the {self._THRESHOLD_ARG} dictionary", valid_keys
            )

        def match(out0_name, output0, out1_name, output1):
            G_LOGGER.extra_verbose(
                f"Note: Comparing {iter_result0.runner_name} vs. {iter_result1.runner_name}"
            )

            aligned = self._align_shapes(out0_name, output0, output1)
            if aligned is None:
                return False
            output0, output1 = aligned

            return self._check_outputs_match(
                output0, output1, self.thresholds_for(out0_name)
            )

        return self._run_comparison(iter_result0, iter_result1, match)


@mod.export()
class L2CompareFunc(_SingleMetricCompareFunc):
    """
    Compares two IterationResults using the L2 norm (Euclidean distance).

    Instances are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.
    """

    _RESULT_CLASS = L2Result
    _THRESHOLD_CLASS = L2Threshold
    _DEFAULT_THRESHOLD = 1e-5
    _THRESHOLD_ARG = "l2_threshold"

    def __init__(
        self,
        l2_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Args:
            l2_threshold (Union[float, Dict[str, float]]):
                    Maximum allowed L2 norm. Per-output values can be given as a dictionary;
                    use ``""`` as the key for a default. Defaults to 1e-5.
            check_shapes (bool):
                    Whether shapes must match exactly. If False, outputs may be permuted or
                    reshaped before comparison. Defaults to True.
            fail_fast (bool): Whether to exit immediately after the first failure. Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns the names of the output(s) to compare against,
                    given an output name from the first runner, its index, and the second
                    runner's IterationResult.
        """
        super().__init__(
            threshold=l2_threshold,
            check_shapes=check_shapes,
            fail_fast=fail_fast,
            find_output_func=find_output_func,
        )

    def _compute_metric(self, out0, out1):
        diff = util.array.subtract(comp_util.cast_up(out0), comp_util.cast_up(out1))
        squared_diff = util.array.power(diff, 2)
        sum_squared_diff = util.array.sum(squared_diff)
        return util.array.sqrt(sum_squared_diff)


@mod.export()
class CosineSimilarityCompareFunc(_SingleMetricCompareFunc):
    """
    Compares two IterationResults using cosine similarity.

    Instances are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.
    """

    _RESULT_CLASS = CosineSimilarityResult
    _THRESHOLD_CLASS = CosineSimilarityThreshold
    _DEFAULT_THRESHOLD = 0.997
    _THRESHOLD_ARG = "cosine_similarity_threshold"

    def __init__(
        self,
        cosine_similarity_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Args:
            cosine_similarity_threshold (Union[float, Dict[str, float]]):
                    Minimum cosine similarity required (range -1 to 1). Per-output values can be
                    given as a dictionary; use ``""`` as the key for a default. Defaults to 0.997.
            check_shapes (bool):
                    Whether shapes must match exactly. If False, outputs may be permuted or
                    reshaped before comparison. Defaults to True.
            fail_fast (bool): Whether to exit immediately after the first failure. Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns the names of the output(s) to compare against,
                    given an output name from the first runner, its index, and the second
                    runner's IterationResult.
        """
        super().__init__(
            threshold=cosine_similarity_threshold,
            check_shapes=check_shapes,
            fail_fast=fail_fast,
            find_output_func=find_output_func,
        )

    def _compute_metric(self, out0, out1):
        array1_flat = util.array.ravel(comp_util.cast_up(out0))
        array2_flat = util.array.ravel(comp_util.cast_up(out1))

        # Calculate dot product
        dot_product = util.array.sum(util.array.multiply(array1_flat, array2_flat))

        # Calculate magnitudes
        magnitude1 = util.array.sqrt(util.array.sum(util.array.power(array1_flat, 2)))
        magnitude2 = util.array.sqrt(util.array.sum(util.array.power(array2_flat, 2)))

        # Avoid division by zero
        if magnitude1 == 0 and magnitude2 == 0:
            return (
                1.0  # If both vectors are zero, they are identical (similarity = 1.0)
            )
        elif magnitude1 == 0 or magnitude2 == 0:
            return 0.0  # If only one vector is zero, they are orthogonal (similarity = 0.0)

        # Cosine similarity is dot_product / (magnitude1 * magnitude2)
        cosine_similarity = float(dot_product / (magnitude1 * magnitude2))

        # A NaN result (e.g. inf/inf on non-finite inputs) must fail the threshold rather than be
        # clamped to 1.0 (a false PASS).
        if math.isnan(cosine_similarity):
            return cosine_similarity

        # Handle floating point issues that might make cosine_similarity slightly outside [-1, 1]
        return max(-1.0, min(1.0, cosine_similarity))


@mod.export()
class PsnrCompareFunc(_SingleMetricCompareFunc):
    """
    Compares two IterationResults using PSNR (Peak Signal-to-Noise Ratio).

    Instances are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.
    """

    _RESULT_CLASS = PsnrResult
    _THRESHOLD_CLASS = PsnrThreshold
    _DEFAULT_THRESHOLD = 30.0
    _THRESHOLD_ARG = "psnr_threshold"

    def __init__(
        self,
        psnr_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Args:
            psnr_threshold (Union[float, Dict[str, float]]):
                    Minimum PSNR (dB) required. Per-output values can be given as a dictionary;
                    use ``""`` as the key for a default. Defaults to 30.0.
            check_shapes (bool):
                    Whether shapes must match exactly. If False, outputs may be permuted or
                    reshaped before comparison. Defaults to True.
            fail_fast (bool): Whether to exit immediately after the first failure. Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns the names of the output(s) to compare against,
                    given an output name from the first runner, its index, and the second
                    runner's IterationResult.
        """
        super().__init__(
            threshold=psnr_threshold,
            check_shapes=check_shapes,
            fail_fast=fail_fast,
            find_output_func=find_output_func,
        )

    def _compute_metric(self, out0, out1):
        array1_cast = comp_util.cast_up(out0)
        array2_cast = comp_util.cast_up(out1)

        # Compute Mean Squared Error
        mse = util.array.mean(
            util.array.power(util.array.subtract(array1_cast, array2_cast), 2)
        )

        # Avoid division by zero
        if mse == 0:
            return float("inf")  # Perfect match

        # Compute data range (max value in reference array)
        max_val = comp_util.compute_max(array1_cast)
        if max_val <= 0:
            max_val = 1.0  # Default to 1.0 if max value is non-positive

        # PSNR formula: 20 * log10(MAX) - 10 * log10(MSE). Coerce to Python floats so np.log10 does
        # not operate on a (possibly torch) array scalar.
        psnr = 20 * np.log10(float(max_val)) - 10 * np.log10(float(mse))
        return psnr


@mod.export()
class SnrCompareFunc(_SingleMetricCompareFunc):
    """
    Compares two IterationResults using SNR (Signal-to-Noise Ratio).

    Instances are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.
    """

    _RESULT_CLASS = SnrResult
    _THRESHOLD_CLASS = SnrThreshold
    _DEFAULT_THRESHOLD = 20.0
    _THRESHOLD_ARG = "snr_threshold"

    def __init__(
        self,
        snr_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Args:
            snr_threshold (Union[float, Dict[str, float]]):
                    Minimum SNR (dB) required. Per-output values can be given as a dictionary;
                    use ``""`` as the key for a default. Defaults to 20.0.
            check_shapes (bool):
                    Whether shapes must match exactly. If False, outputs may be permuted or
                    reshaped before comparison. Defaults to True.
            fail_fast (bool): Whether to exit immediately after the first failure. Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns the names of the output(s) to compare against,
                    given an output name from the first runner, its index, and the second
                    runner's IterationResult.
        """
        super().__init__(
            threshold=snr_threshold,
            check_shapes=check_shapes,
            fail_fast=fail_fast,
            find_output_func=find_output_func,
        )

    def _compute_metric(self, out0, out1):
        array1_cast = comp_util.cast_up(out0)
        array2_cast = comp_util.cast_up(out1)

        # Signal power
        signal_power = util.array.mean(util.array.power(array1_cast, 2))

        # Noise is the difference between the arrays
        noise = util.array.subtract(array1_cast, array2_cast)
        noise_power = util.array.mean(util.array.power(noise, 2))

        # Avoid division by zero
        if noise_power == 0:
            return float("inf")  # Perfect match
        if signal_power == 0:
            return -float("inf")  # No signal

        # SNR formula: 10 * log10(signal_power / noise_power). Coerce to a Python float so np.log10
        # does not operate on a (possibly torch) array scalar.
        snr = 10 * np.log10(float(signal_power / noise_power))
        return snr


@mod.export()
class PerceptualMetricsCompareFunc(_SingleMetricCompareFunc):
    """
    Compares two IterationResults using perceptual metrics (LPIPS), targeting image-like data.

    Instances are used as the ``compare_func`` argument to ``Comparator.compare_accuracy``.
    """

    _RESULT_CLASS = PerceptualMetricsResult
    _THRESHOLD_CLASS = LpipsThreshold
    _DEFAULT_THRESHOLD = 0.1
    _THRESHOLD_ARG = "lpips_threshold"

    def __init__(
        self,
        lpips_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Compares two IterationResults using perceptual metrics (LPIPS), and can be used as the
        ``compare_func`` argument in ``Comparator.compare_accuracy``.

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
        """
        super().__init__(
            threshold=lpips_threshold,
            check_shapes=check_shapes,
            fail_fast=fail_fast,
            find_output_func=find_output_func,
        )
        # The (torch-based) LPIPS model is loaded lazily on first use so that the functor can
        # be constructed - and the averaging API used - without torch/lpips installed.
        self._lpips_initialized = False
        self._torch = None
        self._lpips_model = None

    def _ensure_lpips_model(self):
        if self._lpips_initialized:
            return
        self._lpips_initialized = True

        lpips_model = None
        try:
            # Initialize LPIPS model with explicit device specification
            device = torch.device("cpu")

            # Try with different initialization approaches
            try:
                # First try with default initialization
                lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device)
            except Exception as e1:
                G_LOGGER.warning(
                    f"First LPIPS initialization approach failed: {e1}. Trying alternative method..."
                )

                try:
                    # Try with a different network if AlexNet fails
                    lpips_model = lpips.LPIPS(net="vgg", version="0.1").to(device)
                except Exception as e2:
                    G_LOGGER.warning(
                        f"Second LPIPS initialization approach failed: {e2}. Trying basic initialization..."
                    )

                    try:
                        # As a last resort, try with the most basic initialization
                        model = lpips.LPIPS(net_type="alex")
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

        self._torch = torch
        self._lpips_model = lpips_model

    def _compute_metric(self, out0, out1):
        self._ensure_lpips_model()
        torch = self._torch
        lpips_model = self._lpips_model
        if torch is None or lpips_model is None:
            return None

        try:
            # Cast arrays to numpy to ensure compatibility
            array1_np = util.array.to_numpy(comp_util.cast_up(out0))
            array2_np = util.array.to_numpy(comp_util.cast_up(out1))

            # Check dimensions - LPIPS expects image data
            shape1 = array1_np.shape
            shape2 = array2_np.shape

            # We need at least 3D arrays for LPIPS (typically B,C,H,W or H,W,C)
            if len(shape1) < 3 or len(shape2) < 3:
                G_LOGGER.warning(
                    f"LPIPS requires at least 3D arrays, got shapes {shape1} and {shape2}. LPIPS check will be skipped."
                )
                return None

            # Log input shapes for debugging
            G_LOGGER.verbose(f"Original tensor shapes: {shape1} and {shape2}")

            # Get device information - use CPU for consistency
            device = torch.device("cpu")

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
                G_LOGGER.warning(
                    f"Tensor shapes don't match: {img1.shape} vs {img2.shape}. Adjusting..."
                )

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
                    target_size = (
                        max(img1.shape[2], img2.shape[2]),
                        max(img1.shape[3], img2.shape[3]),
                    )

                    # Only import interpolate if needed
                    try:
                        from torch.nn.functional import interpolate

                        if img1.shape[2:] != target_size:
                            img1 = interpolate(
                                img1,
                                size=target_size,
                                mode="bilinear",
                                align_corners=False,
                            )

                        if img2.shape[2:] != target_size:
                            img2 = interpolate(
                                img2,
                                size=target_size,
                                mode="bilinear",
                                align_corners=False,
                            )

                    except ImportError:
                        G_LOGGER.warning(
                            "Failed to resize tensors: torch.nn.functional.interpolate not available"
                        )
                        if img1.shape[2:] != img2.shape[2:]:
                            G_LOGGER.warning(
                                "Cannot compute LPIPS with tensors of different spatial dimensions"
                            )
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
                    G_LOGGER.warning(
                        f"Standard LPIPS computation failed: {e}. Trying fallback method..."
                    )
                    try:
                        # Try an alternative approach
                        lpips_dist = lpips_model.forward(img1, img2)
                        if isinstance(lpips_dist, torch.Tensor):
                            lpips_dist = lpips_dist.mean().item()
                    except Exception as e2:
                        G_LOGGER.warning(
                            f"Fallback LPIPS computation failed: {e2}. LPIPS check will be skipped."
                        )
                        return None

            return lpips_dist

        except Exception as e:
            G_LOGGER.warning(
                f"Error computing LPIPS: {e}. LPIPS check will be skipped."
            )
            return None

    def _check_outputs_match(self, out0, out1, threshold):
        # A None value (LPIPS computation failed/skipped) is treated as a pass; the criterion and the
        # per-metric line both live on the threshold, so the per-iteration log line matches the
        # averaged summary. describe() returns None when the metric could not be computed.
        value = self._compute_metric(out0, out1)
        result = PerceptualMetricsResult(value, thresholds=threshold)
        described = result.describe()
        if described is not None:
            msg, passed = described
            (G_LOGGER.finish if passed else G_LOGGER.error)(msg)
        else:
            G_LOGGER.warning(
                "LPIPS could not be computed (see the warning above); "
                "this output is being treated as a PASS, so its accuracy was NOT verified."
            )
        return result


# Provides functions to compare two IterationResults
@mod.export()
class CompareFunc:
    """
    Provides functions that can be used to compare two `IterationResult` s.
    """

    @staticmethod
    @mod.deprecate(
        remove_in="0.60.0",
        use_instead="SimpleCompareFunc",
        name="CompareFunc.simple",
    )
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
        Creates a ``SimpleCompareFunc``. See ``SimpleCompareFunc`` for a description of the
        arguments.
        """
        return SimpleCompareFunc(
            check_shapes=check_shapes,
            rtol=rtol,
            atol=atol,
            fail_fast=fail_fast,
            find_output_func=find_output_func,
            check_error_stat=check_error_stat,
            infinities_compare_equal=infinities_compare_equal,
            save_heatmaps=save_heatmaps,
            show_heatmaps=show_heatmaps,
            save_error_metrics_plot=save_error_metrics_plot,
            show_error_metrics_plot=show_error_metrics_plot,
            error_quantile=error_quantile,
        )

    @staticmethod
    @mod.deprecate(
        remove_in="0.60.0",
        use_instead="IndicesCompareFunc",
        name="CompareFunc.indices",
    )
    def indices(index_tolerance=None, fail_fast=None):
        """
        Creates an ``IndicesCompareFunc``. See ``IndicesCompareFunc`` for a description of the
        arguments.
        """
        return IndicesCompareFunc(index_tolerance=index_tolerance, fail_fast=fail_fast)

    @staticmethod
    @mod.deprecate(
        remove_in="0.60.0",
        use_instead="PerceptualMetricsCompareFunc",
        name="CompareFunc.perceptual_metrics",
    )
    def perceptual_metrics(
        lpips_threshold=None,
        check_shapes=None,
        fail_fast=None,
        find_output_func=None,
    ):
        """
        Creates a ``PerceptualMetricsCompareFunc``. See ``PerceptualMetricsCompareFunc`` for a
        description of the arguments.
        """
        return PerceptualMetricsCompareFunc(
            lpips_threshold=lpips_threshold,
            check_shapes=check_shapes,
            fail_fast=fail_fast,
            find_output_func=find_output_func,
        )
