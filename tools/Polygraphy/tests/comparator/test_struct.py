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

import numpy as np
import pytest
import torch

from polygraphy import config, util
from polygraphy.comparator import IterationResult, RunResults
from polygraphy.comparator.struct import LazyArray
from polygraphy.exception import PolygraphyException
from tests.comparator._helpers import _compare_accuracy, _multi_iteration_results


def make_outputs():
    return {"dummy_out": np.zeros((4, 4))}


def make_iter_results(runner_name):
    return [IterationResult(outputs=make_outputs(), runner_name=runner_name)] * 2


@pytest.fixture()
def run_results():
    results = RunResults()
    results.append(("runner0", make_iter_results("runner0")))
    results.append(("runner1", make_iter_results("runner1")))
    return results


class TestRunResults:
    def test_items(self, run_results):
        for name, iteration_results in run_results.items():
            assert isinstance(name, str)
            assert isinstance(iteration_results, list)
            for iter_res in iteration_results:
                assert isinstance(iter_res, IterationResult)

    def test_keys(self, run_results):
        assert list(run_results.keys()) == ["runner0", "runner1"]

    def test_values(self, run_results):
        for iteration_results in run_results.values():
            for iter_res in iteration_results:
                assert isinstance(iter_res, IterationResult)

    def test_getitem(self, run_results):
        assert isinstance(run_results["runner0"][0], IterationResult)
        assert isinstance(run_results[0][1][0], IterationResult)
        assert run_results[0][1] == run_results["runner0"]
        assert run_results[1][1] == run_results["runner1"]

    def test_getitem_out_of_bounds(self, run_results):
        with pytest.raises(IndexError):
            run_results[2]

        with pytest.raises(PolygraphyException, match="does not exist in this"):
            run_results["runner2"]

    def test_setitem(self, run_results):
        def check_results(results, is_none=False):
            for iter_res in results["runner1"]:
                if is_none:
                    assert not iter_res
                    assert iter_res.runner_name == "custom_runner"
                else:
                    assert iter_res
                    assert iter_res.runner_name

        check_results(run_results)

        iter_results = [IterationResult(outputs=None, runner_name=None)]
        run_results["runner1"] = iter_results

        check_results(run_results, is_none=True)

    def test_setitem_out_of_bounds(self, run_results):
        iter_results = [IterationResult(outputs=None, runner_name="new")]
        run_results["runner2"] = iter_results

        assert len(run_results) == 3
        assert run_results["runner2"][0].runner_name == "new"

    def test_contains(self, run_results):
        assert "runner0" in run_results
        assert "runner1" in run_results
        assert "runner3" not in run_results

    def test_add_new(self):
        results = RunResults()
        results.add([make_outputs()], runner_name="custom")

        iter_results = results["custom"]
        assert len(iter_results) == 1
        assert all(
            isinstance(iter_result, IterationResult) for iter_result in iter_results
        )

    def test_add_new_default_name(self):
        results = RunResults()
        results.add([make_outputs()])

        name = results[0][0]
        iter_results = results[name]
        assert len(iter_results) == 1
        assert all(
            isinstance(iter_result, IterationResult) for iter_result in iter_results
        )


@pytest.mark.parametrize("module", [torch, np])
class TestLazyArray:
    @pytest.mark.parametrize("set_threshold", [True, False])
    def test_unswapped_array(self, set_threshold, module):
        with contextlib.ExitStack() as stack:
            if set_threshold:

                def reset_array_swap():
                    config.ARRAY_SWAP_THRESHOLD_MB = -1

                stack.callback(reset_array_swap)

                config.ARRAY_SWAP_THRESHOLD_MB = 8

            small_shape = (7 * 1024 * 1024,)
            small_array = module.ones(small_shape, dtype=module.uint8)
            lazy = LazyArray(small_array)
            assert util.array.equal(small_array, lazy.arr)
            assert lazy.tmpfile is None

            assert util.array.equal(small_array, lazy.load())

    def test_swapped_array(self, module):
        with contextlib.ExitStack() as stack:

            def reset_array_swap():
                config.ARRAY_SWAP_THRESHOLD_MB = -1

            stack.callback(reset_array_swap)

            config.ARRAY_SWAP_THRESHOLD_MB = 8

            large_shape = (9 * 1024 * 1024,)
            large_array = module.ones(large_shape, dtype=module.uint8)
            lazy = LazyArray(large_array)
            assert lazy.arr is None
            assert lazy.tmpfile is not None

            assert util.array.equal(large_array, lazy.load())


class TestRunResultsStreaming:
    # RunResults.split()/concat()/load_streaming() -- the per-iteration round-trip for saved runs.
    def test_load_streaming_splits_file(self, tmp_path):
        run_results = _multi_iteration_results({"A": [[0.0], [1.0], [2.0]]})
        path = str(tmp_path / "results.json")
        run_results.save(path)
        singles = list(RunResults.load_streaming(path))
        assert len(singles) == 3
        assert all(len(single["A"]) == 1 for single in singles)

    def test_load_streaming_chains_list(self, tmp_path):
        # A list of paths is chained into a single run's iterations, in order.
        path0 = str(tmp_path / "a.json")
        path1 = str(tmp_path / "b.json")
        _multi_iteration_results({"A": [[0.0], [1.0]]}).save(path0)
        _multi_iteration_results({"A": [[2.0]]}).save(path1)
        singles = list(RunResults.load_streaming([path0, path1]))
        assert len(singles) == 3
        assert all(len(single["A"]) == 1 for single in singles)

    def test_load_streaming_directory_multi_iteration(self, tmp_path):
        # A file in a directory may itself hold multiple iterations; each is split out individually
        # and yielded in directory/index order (0.json before 1.json).
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        _multi_iteration_results({"A": [[0.0], [1.0], [2.0]]}).save(
            str(out_dir / "0.json")
        )
        _multi_iteration_results({"A": [[3.0]]}).save(str(out_dir / "1.json"))
        singles = list(RunResults.load_streaming(str(out_dir)))
        assert all(len(single["A"]) == 1 for single in singles)
        # The 4-element ordered comparison also subsumes the count (3 from 0.json + 1 from 1.json).
        assert [np.array(single["A"][0]["out"]).item() for single in singles] == [
            0.0,
            1.0,
            2.0,
            3.0,
        ]

    def test_split_concat_ragged_round_trip(self):
        # split() yields ragged single-iteration runs (a runner that ran out is omitted), so concat()
        # must match runners by name, not position, or B's tail iterations would be mis-assigned to A.
        original = _multi_iteration_results(
            {"A": [[0.0]], "B": [[10.0], [11.0], [12.0]]}
        )
        combined = RunResults.concat(list(original.split()))
        assert [float(np.array(r["out"])[0]) for r in combined["A"]] == [0.0]
        assert [float(np.array(r["out"])[0]) for r in combined["B"]] == [
            10.0,
            11.0,
            12.0,
        ]

    def test_split_empty_yields_nothing(self):
        # split() of a run with no iterations yields nothing (max(..., default=0)).
        assert list(RunResults().split()) == []

    def test_split_single_iteration_all_runners_present(self):
        # split() with exactly one iteration across multiple runners yields one RunResults
        # containing all runners (boundary between empty case and multi-iteration cases).
        original = _multi_iteration_results({"A": [[0.0]], "B": [[1.0]]})
        splits = list(original.split())
        assert len(splits) == 1
        assert list(splits[0].keys()) == ["A", "B"]
        assert len(splits[0]["A"]) == 1
        assert len(splits[0]["B"]) == 1

    def test_concat_overlapping_runners(self):
        # concat() with the same runner in two inputs must extend (not duplicate) the iteration list.
        r0 = _multi_iteration_results({"A": [[0.0]], "B": [[10.0]]})
        r1 = _multi_iteration_results({"A": [[1.0]], "B": [[11.0]]})
        combined = RunResults.concat([r0, r1])
        assert list(combined.keys()) == ["A", "B"]
        assert [float(np.array(it["out"])[0]) for it in combined["A"]] == [0.0, 1.0]
        assert [float(np.array(it["out"])[0]) for it in combined["B"]] == [10.0, 11.0]


class TestAccuracyResultsSerialization:
    # Build a 2-runner, 2-output, 1-iteration comparison and return its AccuracyResults.
    def _compare(self, compare_func, check_average=False):
        return _compare_accuracy(
            {
                "r0": {"out0": [0.0, 0.0], "out1": [1.0]},
                "r1": {"out0": [0.0, 2.0], "out1": [1.0]},
            },
            compare_func,
            check_average=check_average,
        )

    def test_round_trip_simple(self):
        from polygraphy.comparator import AccuracyResults, SimpleCompareFunc

        results = self._compare(SimpleCompareFunc(check_error_stat="max", atol=1.0))
        loaded = AccuracyResults.from_json(results.to_json())
        # Verdicts and metric values survive the round trip.
        assert bool(loaded) == bool(results)
        pair = ("r0", "r1")
        orig = results[0][pair][0]["out0"]
        rt = loaded[0][pair][0]["out0"]
        assert rt.max_absdiff == orig.max_absdiff
        assert bool(rt) == bool(orig)
        assert loaded[0].aggregation == "per_sample"

    @pytest.mark.parametrize(
        "compare_func",
        [
            "L2CompareFunc",
            "CosineSimilarityCompareFunc",
            "PsnrCompareFunc",
            "SnrCompareFunc",
            "PerceptualMetricsCompareFunc",
        ],
    )
    def test_round_trip_single_metric(self, compare_func):
        import polygraphy.comparator as comp

        cf_type = getattr(comp, compare_func)
        results = self._compare(cf_type())
        loaded = comp.AccuracyResults.from_json(results.to_json())
        pair = ("r0", "r1")
        orig = results[0][pair][0]["out0"]
        rt = loaded[0][pair][0]["out0"]
        (field,) = type(orig).metric_fields()
        assert getattr(rt, field) == getattr(orig, field)
        assert bool(rt) == bool(orig)

    def test_round_trip_save_load_file(self, tmp_path):
        from polygraphy.comparator import AccuracyResults, SimpleCompareFunc

        results = self._compare(SimpleCompareFunc(check_error_stat="max", atol=1.0))
        path = str(tmp_path / "acc.json")
        results.save(path)
        loaded = AccuracyResults.load(path)
        assert bool(loaded) == bool(results)

    def test_round_trip_preserves_nan_and_none(self):
        # NaN metric values and a None quantile must round-trip.
        from polygraphy.comparator import AccuracyResults, OutputCompareResult

        result = OutputCompareResult(
            float("nan"),
            float("inf"),
            1.0,
            1.0,
            1.0,
            1.0,
            None,
            None,
            thresholds={"check_error_stat": "max", "atol": 1.0, "rtol": 1.0},
        )
        rt = AccuracyResults.from_json(AccuracyResults([_wrap(result)]).to_json())[0][
            ("r0", "r1")
        ][0]["out"]
        assert util.array.isnan(rt.max_absdiff)
        assert rt.quantile_absdiff is None


def _wrap(output_result):
    # Wraps a single result object in an AccuracyResult with one runner pair / iteration / output.
    from collections import OrderedDict

    from polygraphy.comparator import AccuracyResult

    return AccuracyResult({("r0", "r1"): [OrderedDict([("out", output_result)])]})


class TestAccuracyResultsReevaluate:
    def _compare(self, compare_func, check_average=False):
        return _compare_accuracy(
            {"r0": {"out": [0.0, 0.0]}, "r1": {"out": [0.0, 2.0]}},
            compare_func,
            check_average=check_average,
        )

    def test_reevaluate_flips_verdict(self):
        # max_absdiff is 2.0; loosening/tightening atol flips pass/fail after a save+load round trip.
        from polygraphy.comparator import (
            AccuracyResults,
            SimpleCompareFunc,
            SimpleThreshold,
        )

        results = self._compare(SimpleCompareFunc(check_error_stat="max", atol=1.0))
        loaded = AccuracyResults.from_json(results.to_json())
        loaded.reevaluate(SimpleThreshold("max", 5.0, 1e-5))
        assert bool(loaded)
        loaded.reevaluate(SimpleThreshold("max", 0.5, 1e-5))
        assert not bool(loaded)

    def test_reevaluate_elemwise_save_against_scalar_stat(self):
        # A simple/elemwise (default) save can be re-thresholded against a scalar statistic.
        from polygraphy.comparator import (
            AccuracyResults,
            SimpleCompareFunc,
            SimpleThreshold,
        )

        results = self._compare(SimpleCompareFunc(atol=1.0))  # elemwise
        loaded = AccuracyResults.from_json(results.to_json())
        loaded.reevaluate(SimpleThreshold("max", 5.0, 1e-5))
        assert bool(loaded)

    def test_reevaluate_elemwise_as_new_stat_raises(self):
        from polygraphy.comparator import (
            AccuracyResults,
            SimpleCompareFunc,
            SimpleThreshold,
        )

        results = self._compare(SimpleCompareFunc(check_error_stat="max", atol=1.0))
        loaded = AccuracyResults.from_json(results.to_json())
        loaded.reevaluate(SimpleThreshold("elemwise", 1e-5, 1e-5))
        with pytest.raises(PolygraphyException, match="elemwise"):
            bool(loaded)

    def test_reevaluate_field_mismatch_raises(self):
        # Saved L2 results cannot be re-thresholded with a PSNR threshold (different metric fields).
        from polygraphy.comparator import (
            AccuracyResults,
            L2CompareFunc,
            PsnrThreshold,
        )

        results = self._compare(L2CompareFunc())
        loaded = AccuracyResults.from_json(results.to_json())
        with pytest.raises(
            PolygraphyException, match="No saved comparison results match"
        ):
            loaded.reevaluate(PsnrThreshold(30.0))

    def test_reevaluate_empty_per_output_dict_raises(self):
        # An empty per-output threshold dict is a clean error, not a leaked StopIteration.
        from polygraphy.comparator import AccuracyResults, L2CompareFunc

        results = self._compare(L2CompareFunc())
        loaded = AccuracyResults.from_json(results.to_json())
        with pytest.raises(PolygraphyException, match="cannot be empty"):
            loaded.reevaluate({})

    def test_reevaluate_rejects_threshold_for_absent_comparison(self):
        # Requesting a threshold whose comparison is not in the file is an error: you asked to check
        # data that was never saved.
        from polygraphy.comparator import (
            AccuracyResults,
            L2CompareFunc,
            L2Threshold,
            PsnrThreshold,
        )

        results = self._compare(L2CompareFunc())
        loaded = AccuracyResults.from_json(results.to_json())
        with pytest.raises(
            PolygraphyException, match="No saved comparison results match"
        ):
            loaded.reevaluate([L2Threshold(5.0), PsnrThreshold(30.0)])

    def test_reevaluate_ignores_unrequested_comparisons(self):
        # Saved comparisons that no provided threshold selects are ignored (not an error, and not
        # counted in the overall pass/fail) -- e.g. checking only L2 ignores a saved PSNR comparison.
        from polygraphy.comparator import (
            AccuracyResults,
            L2CompareFunc,
            L2Threshold,
            PsnrCompareFunc,
        )

        # Save both L2 and PSNR comparisons. L2 norm is 2.0; PSNR is poor (large diff).
        results = self._compare([L2CompareFunc(), PsnrCompareFunc()])
        loaded = AccuracyResults.from_json(results.to_json())
        assert len(loaded) == 2

        # Re-check only L2 with a loose threshold; the PSNR comparison is dropped.
        loaded.reevaluate(L2Threshold(5.0))
        assert len(loaded) == 1
        assert loaded[0]._result_metric_fields() == ["l2_norm"]
        assert bool(loaded)  # 2.0 <= 5.0, and PSNR no longer factors in

    def test_reevaluate_matches_by_fields_not_class_identity(self):
        # A saved result is matched to the supplied threshold by metric-field names, NOT threshold
        # class identity -- so a threshold from a --compare-func-script function (which
        # import_from_script re-imports into a fresh class object on every load) still matches its
        # own saved results.
        from polygraphy.comparator import AccuracyResults, CompareResult, Threshold

        def make_threshold_class():
            class _CustomThreshold(Threshold):
                METRIC_FIELDS = ["custom_metric"]

                def __init__(self, threshold):
                    self.threshold = threshold

                def passed(self, metric_values):
                    value = metric_values.get("custom_metric")
                    return bool(value is not None and value <= self.threshold)

            return _CustomThreshold

        # Two distinct threshold classes with identical metric fields, mimicking the saved-vs-
        # reloaded class objects that re-importing a --compare-func-script produces.
        SavedThreshold = make_threshold_class()
        ReloadedThreshold = make_threshold_class()
        assert SavedThreshold is not ReloadedThreshold

        class _CustomResult(CompareResult):
            _THRESHOLD_CLASS = SavedThreshold

            def __init__(self, custom_metric, thresholds=None):
                super().__init__(thresholds=thresholds)
                self.custom_metric = custom_metric

        results = AccuracyResults(
            [_wrap(_CustomResult(2.0, thresholds=SavedThreshold(0.5)))]
        )
        assert not bool(results)  # 2.0 > 0.5
        results.reevaluate(
            ReloadedThreshold(5.0)
        )  # matched by field name despite class mismatch
        assert bool(results)  # 2.0 <= 5.0

    def test_reevaluate_average_aggregation(self):
        from polygraphy.comparator import (
            AccuracyResults,
            SimpleCompareFunc,
            SimpleThreshold,
        )

        results = self._compare(
            SimpleCompareFunc(check_error_stat="mean", atol=1.5), check_average=True
        )
        loaded = AccuracyResults.from_json(results.to_json())
        # Mean abs diff is 1.0; re-checking against a tiny atol fails in average mode.
        loaded.reevaluate(SimpleThreshold("mean", 0.1, 1e-5), aggregation="average")
        assert not bool(loaded)
        loaded.reevaluate(SimpleThreshold("mean", 5.0, 1e-5), aggregation="average")
        assert bool(loaded)

    def test_reevaluate_per_output_threshold_dict(self):
        # A {output_name: Threshold} dict applies per-output thresholds, with "" as the default.
        from polygraphy.comparator import (
            AccuracyResults,
            SimpleCompareFunc,
            SimpleThreshold,
        )

        # out0 differs by 2.0, out1 by 4.0.
        results = _compare_accuracy(
            {
                "r0": {"out0": [0.0, 0.0], "out1": [0.0, 0.0]},
                "r1": {"out0": [0.0, 2.0], "out1": [0.0, 4.0]},
            },
            SimpleCompareFunc(check_error_stat="max", atol=10.0),
        )
        loaded = AccuracyResults.from_json(results.to_json())
        # out0 uses the "" default (2.0 <= 5.0 -> pass); out1 is overridden tighter (4.0 > 1.0 -> fail).
        loaded.reevaluate(
            {
                "": SimpleThreshold("max", 5.0, 1e-5),
                "out1": SimpleThreshold("max", 1.0, 1e-5),
            }
        )
        assert not bool(loaded)
        # Loosen out1 too -> everything passes.
        loaded.reevaluate(
            {
                "": SimpleThreshold("max", 5.0, 1e-5),
                "out1": SimpleThreshold("max", 5.0, 1e-5),
            }
        )
        assert bool(loaded)

    def test_reevaluate_list_matches_each_result_by_type(self):
        # A list supplies one threshold per contained result, matched to each by metric-field type
        # regardless of order.
        from polygraphy.comparator import (
            AccuracyResults,
            L2CompareFunc,
            L2Threshold,
            SimpleCompareFunc,
            SimpleThreshold,
        )

        # l2_norm of [0,0] vs [0,2] is 2.0; max_absdiff is also 2.0.
        results = _compare_accuracy(
            {"r0": {"out": [0.0, 0.0]}, "r1": {"out": [0.0, 2.0]}},
            [
                SimpleCompareFunc(check_error_stat="max", atol=10.0),
                L2CompareFunc(l2_threshold=10.0),
            ],
        )
        loaded = AccuracyResults.from_json(results.to_json())
        # Provide thresholds in the opposite order to prove matching is by type, not position.
        loaded.reevaluate([L2Threshold(1.0), SimpleThreshold("max", 5.0, 1e-5)])
        assert not bool(loaded)  # simple passes (2.0 <= 5.0); l2 fails (2.0 > 1.0)
        loaded.reevaluate([L2Threshold(5.0), SimpleThreshold("max", 5.0, 1e-5)])
        assert bool(loaded)
