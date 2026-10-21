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
import os
import subprocess as sp
import types
from collections import OrderedDict

import numpy as np
import pytest
import tensorrt as trt

from polygraphy import util, mod
from polygraphy.logger import G_LOGGER
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.onnx import GsFromOnnx, OnnxFromBytes
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.pluginref import PluginRefRunner
from polygraphy.common import TensorMetadata
from polygraphy.datatype import DataType
from polygraphy.backend.trt import (
    EngineFromNetwork,
    NetworkFromOnnxBytes,
    TrtRunner,
    network_from_onnx_bytes,
)
from polygraphy.backend.trt.util import get_all_tensors
from polygraphy.comparator import (
    AccuracyResult,
    AccuracyResults,
    BaseCompareFunc,
    Comparator,
    CompareResult,
    DataLoader,
    IndicesCompareFunc,
    IterationResult,
    RunResults,
    SimpleCompareFunc,
    StreamingDataLoader,
    Threshold,
    TopKPostprocessFunc,
)
from polygraphy.exception import PolygraphyException
from tests.comparator._helpers import _multi_iteration_results, _single_iteration
from tests.models.meta import ONNX_MODELS

build_torch = lambda a, **kwargs: util.array.to_torch(np.array(a, **kwargs))


class TestComparator:
    def test_warmup_runs(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnx(onnx_loader))
        run_results = Comparator.run([runner], warm_up=2)
        assert len(run_results[runner.name]) == 1

    @pytest.mark.parametrize("use_generator", [False, True], ids=["list", "generator"])
    def test_data_loader_types(self, use_generator):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnx(onnx_loader), name="onnx_runner")
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}] * 2
        loader = iter(data) if use_generator else data
        run_results = Comparator.run([runner], data_loader=loader)
        iter_results = run_results["onnx_runner"]
        assert len(iter_results) == 2
        for actual, expected in zip(iter_results, data):
            assert np.all(actual["y"] == expected["x"])

    def test_multiple_runners(self):
        onnx_bytes = ONNX_MODELS["identity"].loader()
        build_onnxrt_session = SessionFromOnnx(onnx_bytes)
        load_engine = EngineFromNetwork(NetworkFromOnnxBytes(onnx_bytes))
        gs_graph = GsFromOnnx(OnnxFromBytes(onnx_bytes))

        runners = [
            OnnxrtRunner(build_onnxrt_session),
            PluginRefRunner(gs_graph),
            TrtRunner(load_engine),
        ]

        run_results = Comparator.run(runners)
        compare_func = SimpleCompareFunc(check_shapes=True)
        assert bool(Comparator.compare_accuracy(run_results, compare_func=compare_func))
        assert len(list(run_results.values())[0]) == 1  # Default number of iterations

    def test_postprocess(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        run_results = Comparator.run([OnnxrtRunner(SessionFromOnnx(onnx_loader))])
        # Output shape is (1, 1, 2, 2)
        postprocessed = Comparator.postprocess(
            run_results, postprocess_func=TopKPostprocessFunc(k=(1, -1))
        )
        for _, results in postprocessed.items():
            for result in results:
                for _, output in result.items():
                    assert output.shape == (1, 1, 2, 1)

    def test_errors_do_not_hang(self):
        # Should error because interface is not implemented correctly.
        class FakeRunner:
            def __init__(self):
                self.name = "fake"

        runners = [FakeRunner()]
        with pytest.raises(PolygraphyException):
            Comparator.run(runners, use_subprocess=True, subprocess_polling_interval=1)

    def test_segfault_does_not_hang(self):
        def raise_called_process_error():
            class FakeSegfault(sp.CalledProcessError):
                pass

            raise FakeSegfault(-11, ["simulate", "segfault"])

        runners = [TrtRunner(EngineFromNetwork(raise_called_process_error))]
        with pytest.raises(PolygraphyException):
            Comparator.run(runners, use_subprocess=True, subprocess_polling_interval=1)

    def test_subprocess_infer_error_surfaces_runner_failure(self):
        # A runner failure in a subprocess must surface as a PolygraphyException, not be swallowed.
        runners = [_FailingInferRunner("failing")]
        with pytest.raises(PolygraphyException):
            Comparator.run(runners, use_subprocess=True, subprocess_polling_interval=1)

    def test_multirun_outputs_are_different(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = TrtRunner(EngineFromNetwork(NetworkFromOnnxBytes(onnx_loader)))
        run_results = Comparator.run([runner], data_loader=DataLoader(iterations=2))

        iteration0 = run_results[runner.name][0]
        iteration1 = run_results[runner.name][1]
        for name in iteration0.keys():
            assert util.array.any(iteration0[name] != iteration1[name])

    @pytest.mark.parametrize("array_type", [np.array, build_torch])
    def test_validate_nan(self, array_type):
        run_results = RunResults()
        run_results["fake-runner"] = [
            IterationResult(outputs={"x": array_type(np.nan)})
        ]
        assert not Comparator.validate(run_results)

    @pytest.mark.parametrize("array_type", [np.array, build_torch])
    def test_validate_inf(self, array_type):
        run_results = RunResults()
        run_results["fake-runner"] = [
            IterationResult(outputs={"x": array_type(np.inf)})
        ]
        assert not Comparator.validate(run_results, check_inf=True)

    def test_dim_param_trt_onnxrt(self):
        load_onnx_bytes = ONNX_MODELS["dim_param"].loader
        build_onnxrt_session = SessionFromOnnx(load_onnx_bytes)
        load_engine = EngineFromNetwork(NetworkFromOnnxBytes(load_onnx_bytes))

        runners = [
            OnnxrtRunner(build_onnxrt_session),
            TrtRunner(load_engine),
        ]

        run_results = Comparator.run(runners)
        compare_func = SimpleCompareFunc(check_shapes=True)
        assert bool(Comparator.compare_accuracy(run_results, compare_func=compare_func))
        assert len(list(run_results.values())[0]) == 1  # Default number of iterations

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("10.0"),
        reason="Feature not present before 10.0",
    )
    def test_debug_tensors(self):
        model = ONNX_MODELS["identity"]
        builder, network, parser = network_from_onnx_bytes(model.loader)
        tensor_map = get_all_tensors(network)
        network.mark_debug(tensor_map["x"])
        load_engine = EngineFromNetwork((builder, network, parser))
        runners = [TrtRunner(load_engine)]
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]
        run_results = Comparator.run(runners, data_loader=data)
        for iteration_list in run_results.values():
            # There should be 2 outputs, debug tensor "x" and output "y"
            assert len(list(iteration_list[0].items())) == 2
        run_results["fake-runner"] = [
            IterationResult(
                outputs={
                    "x": np.ones((1, 1, 2, 2), dtype=np.float32),
                    "y": np.ones((1, 1, 2, 2), dtype=np.float32),
                }
            )
        ]
        compare_func = SimpleCompareFunc(check_shapes=True)
        assert bool(Comparator.compare_accuracy(run_results, compare_func=compare_func))

    def test_run_save_directory_round_trip(self, tmp_path):
        # An extensionless save_outputs_path is written as a per-iteration directory.
        data = [{"x": np.full((1, 1, 2, 2), i, dtype=np.float32)} for i in range(3)]
        out_dir = str(tmp_path / "outputs")
        Comparator.run(
            [_identity_runner("A")], data_loader=data, save_outputs_path=out_dir
        )
        assert sorted(os.listdir(out_dir)) == [
            "0000000000.json",
            "0000000001.json",
            "0000000002.json",
        ]

    def test_run_save_single_file_round_trip(self, tmp_path):
        # A save_outputs_path with an extension is written as a single file, readable by RunResults.load.
        data = [{"x": np.full((1, 1, 2, 2), i, dtype=np.float32)} for i in range(3)]
        out_file = str(tmp_path / "outputs.json")
        Comparator.run(
            [_identity_runner("A")], data_loader=data, save_outputs_path=out_file
        )
        assert os.path.isfile(out_file)
        assert len(RunResults.load(out_file)["A"]) == 3

    @pytest.mark.parametrize("extension", ["", ".json"])
    def test_run_save_inputs_round_trip(self, tmp_path, extension):
        # Materialized save_inputs_path writes the inputs fed to the runners (a per-iteration
        # directory for an extensionless path, or a single file otherwise), readable back.
        data = [{"x": np.full((1, 1, 2, 2), i, dtype=np.float32)} for i in range(3)]
        in_path = str(tmp_path / f"inputs{extension}")
        Comparator.run(
            [_identity_runner("A")], data_loader=data, save_inputs_path=in_path
        )
        loaded = list(StreamingDataLoader(in_path, allow_dirs=True))
        assert len(loaded) == 3
        assert all(np.array_equal(loaded[i]["x"], data[i]["x"]) for i in range(3))

    def test_run_save_inputs_skipped_without_runners(self, tmp_path):
        # With no runners the loader cache is never built, so no inputs are written.
        in_dir = tmp_path / "inputs"
        Comparator.run(
            [],
            data_loader=[{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}],
            save_inputs_path=str(in_dir),
        )
        assert not in_dir.exists()

    def test_run_save_input_blob_round_trip(self, tmp_path):
        # save_input_blob_path writes each tensor as a raw .bin file under a per-iteration
        # subdirectory, readable back with numpy.fromfile.
        data = [{"x": np.full((1, 1, 2, 2), i, dtype=np.float32)} for i in range(3)]
        raw_dir = str(tmp_path / "raw_inputs")
        Comparator.run(
            [_identity_runner("A")], data_loader=data, save_input_blob_path=raw_dir
        )
        assert sorted(os.listdir(raw_dir)) == [
            "0000000000",
            "0000000001",
            "0000000002",
        ]
        for i in range(3):
            iter_dir = os.path.join(raw_dir, f"{i:010d}")
            assert os.listdir(iter_dir) == ["x.bin"]
            loaded = np.fromfile(
                os.path.join(iter_dir, "x.bin"), dtype=np.float32
            ).reshape((1, 1, 2, 2))
            assert np.array_equal(loaded, data[i]["x"])

    def test_run_save_input_blob_skipped_without_runners(self, tmp_path):
        # With no runners the loader cache is never built, so no raw inputs are written.
        raw_dir = tmp_path / "raw_inputs"
        Comparator.run(
            [],
            data_loader=[{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}],
            save_input_blob_path=str(raw_dir),
        )
        assert not raw_dir.exists()

    def test_run_save_input_blob_streaming_round_trip(self, tmp_path):
        # The streaming path writes raw inputs inline via the same _InputBlobWriter.
        data = [{"x": np.full((1, 1, 2, 2), i, dtype=np.float32)} for i in range(3)]
        raw_dir = str(tmp_path / "raw_inputs")
        list(
            Comparator.run(
                [_identity_runner("A")],
                data_loader=data,
                save_input_blob_path=raw_dir,
                streaming=True,
            )
        )
        assert sorted(os.listdir(raw_dir)) == [
            "0000000000",
            "0000000001",
            "0000000002",
        ]

    @pytest.mark.parametrize(
        "name",
        ["../../etc/passwd", "/etc/passwd", "../sibling"],
    )
    def test_input_blob_writer_rejects_path_traversal(self, tmp_path, name):
        # A malicious/unexpected tensor name must not be able to escape the per-iteration directory.
        from polygraphy.comparator.comparator import _InputBlobWriter

        writer = _InputBlobWriter(str(tmp_path / "raw_inputs"))
        with pytest.raises(PolygraphyException, match="outside of"):
            writer.append({name: np.ones((1,), dtype=np.float32)})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"save_inputs_path": True, "save_outputs_path": True},
            {"save_inputs_path": True, "save_input_blob_path": True},
            {"save_outputs_path": True, "save_input_blob_path": True},
        ],
    )
    def test_run_rejects_identical_save_paths(self, tmp_path, kwargs):
        # Any two of inputs/outputs/raw-inputs would clobber each other if written to the same
        # path. The guard fires before any inference, so the data loader is never consumed.
        path = str(tmp_path / "same")
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]
        with pytest.raises(PolygraphyException, match="same path"):
            Comparator.run(
                [_identity_runner("A")],
                data_loader=data,
                **{key: path for key in kwargs},
            )


class TestAccuracyResultAverage:
    PAIR = ("runner0", "runner1")

    def _build(self, iter_value_pairs, compare_func, aggregation="average"):
        result = AccuracyResult()
        result[self.PAIR] = [
            compare_func(
                IterationResult(outputs={"out": np.array(v0, dtype=np.float32)}),
                IterationResult(outputs={"out": np.array(v1, dtype=np.float32)}),
            )
            for v0, v1 in iter_value_pairs
        ]
        result.aggregation = aggregation
        return result

    def test_per_sample_default_aggregation(self):
        # Per-sample: iteration 1 fails (|diff|=2 > atol 1.5).
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.5, rtol=0.0)
        result = self._build(
            [([0.0], [0.0]), ([0.0], [2.0])], cf, aggregation="per_sample"
        )
        assert not bool(result)
        assert result.stats(self.PAIR) == (1, 1, 2)

    def test_percentage_is_deprecated(self):
        # percentage() still returns the pass fraction but emits a DeprecationWarning pointing at stats().
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.5, rtol=0.0)
        result = self._build(
            [([0.0], [0.0]), ([0.0], [2.0])], cf, aggregation="per_sample"
        )
        with pytest.warns(DeprecationWarning, match="stats"):
            pct = result.percentage(self.PAIR)
        assert pct == 0.5  # 1 of 2 iterations matched

    def test_average_passes_when_per_sample_fails(self):
        # Average mean abs diff (1.0) is within atol (1.5) even though iteration 1 fails alone.
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.5, rtol=0.0)
        result = self._build([([0.0], [0.0]), ([0.0], [2.0])], cf)
        assert bool(result.average_results(self.PAIR)["out"])
        assert bool(result)
        assert _average_pass_fail_total(result, self.PAIR) == (1, 0, 1)
        # stats() keeps per-iteration semantics regardless of aggregation mode.
        assert result.stats(self.PAIR) == (1, 1, 2)

    def test_average_fails_when_average_exceeds_tol(self):
        # Average mean abs diff (3.0) exceeds atol (1.5).
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.5, rtol=0.0)
        result = self._build([([0.0], [2.0]), ([0.0], [4.0])], cf)
        assert not bool(result.average_results(self.PAIR)["out"])
        assert not bool(result)

    def test_structural_failure_raises(self):
        # A shape mismatch yields a bare False with no metric fields, so averaged pass/fail fails loudly.
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1e9, rtol=1e9)
        result = self._build([([0.0], [0.0]), ([0.0, 0.0], [0.0])], cf)
        with pytest.raises(PolygraphyException, match="check_average"):
            result.average_results(self.PAIR)

    def test_describe_average_mirrors_per_iteration_line(self):
        # The averaged-metric description mirrors the per-iteration log line: metric label,
        # value, threshold (with its tolerance/min-required label), and pass/fail. It is derived on
        # demand from the averaged result object, which carries its own threshold.
        from polygraphy.comparator import L2CompareFunc

        cf = L2CompareFunc(l2_threshold=0.5)
        result = self._build([([0.0], [3.0]), ([0.0], [5.0])], cf)
        described = result.describe_average(self.PAIR)
        assert list(described) == ["out"]
        line, passed = described["out"]
        assert not passed
        assert line == "L2 Norm: 4 (tolerance: 0.5) | FAILED"

    def test_describe_average_empty_without_averaging_support(self):
        # A result type with no single-metric description line (e.g. 'simple') yields an empty mapping.
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.5, rtol=0.0)
        result = self._build([([0.0], [0.0]), ([0.0], [2.0])], cf)
        assert result.describe_average(self.PAIR) == OrderedDict()


class _AbsMeanDiffThreshold(Threshold):
    # Custom threshold carrying the comparison criterion for a single scalar metric. The verdict is
    # derived from the metric + threshold, so the functor only needs to produce the metric and supply
    # a threshold.
    METRIC_FIELDS = ["abs_mean_diff"]

    def __init__(self, threshold):
        self.threshold = threshold

    def passed(self, metric_values):
        value = metric_values.get("abs_mean_diff")
        return bool(value is not None and value <= self.threshold)


class _AbsMeanDiffResult(CompareResult):
    # Custom result carrying a single scalar metric; the criterion and metric fields live on
    # _AbsMeanDiffThreshold (its associated threshold class).
    _THRESHOLD_CLASS = _AbsMeanDiffThreshold

    def __init__(self, abs_mean_diff, thresholds=None):
        super().__init__(thresholds=thresholds)
        self.abs_mean_diff = abs_mean_diff


class _AbsMeanDiffCompareFunc(BaseCompareFunc):
    # Custom functor: metric is |mean(out0) - mean(out1)|; an output matches if it is within `threshold`.
    _RESULT_CLASS = _AbsMeanDiffResult

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, iter_result0, iter_result1):
        result = OrderedDict()
        for name in iter_result0.keys():
            score = float(abs(iter_result0[name].mean() - iter_result1[name].mean()))
            result[name] = _AbsMeanDiffResult(
                score, thresholds=self.thresholds_for(name)
            )
        return result

    def thresholds_for(self, output_name):
        return _AbsMeanDiffThreshold(self.threshold)


class TestCompareAccuracyAverage:
    def test_fail_fast_mutually_exclusive(self):
        run_results = _multi_iteration_results({"r0": [[0.0]], "r1": [[0.0]]})
        cf = SimpleCompareFunc(check_error_stat="mean")
        with pytest.raises(
            PolygraphyException,
            match="check_average and fail_fast cannot be used together",
        ):
            Comparator.compare_accuracy(
                run_results, compare_func=cf, check_average=True, fail_fast=True
            )

    @pytest.mark.parametrize(
        "cf,match",
        [
            (
                lambda ir0, ir1: OrderedDict(out=True),
                "does not support average comparison",
            ),
            (IndicesCompareFunc(), "does not produce averageable metrics"),
        ],
    )
    def test_unsupported_compare_func_rejected(self, cf, match):
        run_results = _multi_iteration_results({"r0": [[0.0]], "r1": [[0.0]]})
        with pytest.raises(PolygraphyException, match=match):
            Comparator.compare_accuracy(
                run_results, compare_func=cf, check_average=True
            )

    def test_elemwise_rejected(self):
        run_results = _multi_iteration_results(
            {"r0": [[0.0], [0.0]], "r1": [[0.0], [2.0]]}
        )
        cf = SimpleCompareFunc(check_error_stat="elemwise")
        with pytest.raises(
            PolygraphyException,
            match=r"check_average is not supported with check_error_stat='elemwise'",
        ):
            Comparator.compare_accuracy(
                run_results, compare_func=cf, check_average=True
            )

    def test_custom_base_compare_func_per_sample(self):
        # A user-defined BaseCompareFunc works as a compare_func. |mean diff| is 0 then 2;
        # with threshold 1.5, the second iteration mismatches.
        run_results = _multi_iteration_results(
            {"r0": [[0.0], [0.0]], "r1": [[0.0], [2.0]]}
        )
        cf = _AbsMeanDiffCompareFunc(threshold=1.5)
        assert not bool(Comparator.compare_accuracy(run_results, compare_func=cf))

    def test_custom_base_compare_func_average(self):
        # A user-defined BaseCompareFunc supports check_average: the averaged metric
        # (|mean diff| = (0 + 2) / 2 = 1.0) is within threshold 1.5 even though one iteration fails.
        run_results = _multi_iteration_results(
            {"r0": [[0.0], [0.0]], "r1": [[0.0], [2.0]]}
        )
        cf = _AbsMeanDiffCompareFunc(threshold=1.5)
        (result,) = Comparator.compare_accuracy(
            run_results, compare_func=cf, check_average=True
        )
        assert bool(result)
        assert bool(result.average_results(("r0", "r1"))["out"])

    def test_average_fails_when_average_exceeds_tol_despite_most_samples_passing(self):
        # A single large outlier pulls the average over the threshold even though most samples pass;
        # the average check fails on the average, not the per-sample majority.
        # |diff| per iteration is [0, 0, 0, 8] -> mean |diff| = 2.0 exceeds atol 1.5.
        run_results = _multi_iteration_results(
            {"r0": [[0.0], [0.0], [0.0], [0.0]], "r1": [[0.0], [0.0], [0.0], [8.0]]}
        )
        cf = SimpleCompareFunc(check_error_stat="mean", atol=1.5, rtol=0.0)
        (result,) = Comparator.compare_accuracy(
            run_results, compare_func=cf, check_average=True
        )
        # 3 of the 4 individual samples meet the threshold ...
        assert result.stats(("r0", "r1")) == (3, 1, 4)
        # ... but the averaged metric (2.0) exceeds it, so the average check fails.
        assert not bool(result)
        assert not bool(result.average_results(("r0", "r1"))["out"])

    def test_plain_function_per_sample_does_not_crash(self):
        # A plain-function compare_func (no averaging API) works in the default per-sample mode,
        # including the informational average-metrics logging.
        run_results = _multi_iteration_results({"r0": [[0.0]], "r1": [[0.0]]})

        def custom(iter_result0, iter_result1):
            return OrderedDict(out=True)

        assert bool(Comparator.compare_accuracy(run_results, compare_func=custom))


def _identity_runner(name):
    return OnnxrtRunner(SessionFromOnnx(ONNX_MODELS["identity"].loader), name=name)


def _average_pass_fail_total(result, runner_pair):
    # Re-derives what the removed AccuracyResult.average_stats() returned: the per-output
    # (passed, failed, total) after averaging metrics across iterations.
    averaged = result.average_results(runner_pair)
    passed = sum(1 for r in averaged.values() if bool(r))
    total = len(averaged)
    return passed, total - passed, total


class _RecordingRunner(BaseRunner):
    # Records the dtype of the input it receives, to verify inputs are coerced per runner.
    def __init__(self, name, dtype, received):
        super().__init__(name=name)
        self.expected_dtype = dtype
        self._received = received

    def get_input_metadata_impl(self):
        return TensorMetadata().add("x", self.expected_dtype, (1, 1, 2, 2))

    def infer_impl(self, feed_dict):
        self.inference_time = 0.0
        self._received.append(util.array.dtype(feed_dict["x"]))
        return {"y": feed_dict["x"]}


class _FailingInferRunner(BaseRunner):
    # Activates cleanly but raises during inference, to exercise the subprocess failure path.
    def get_input_metadata_impl(self):
        return TensorMetadata().add("x", DataType.FLOAT32, (1, 1, 2, 2))

    def infer_impl(self, feed_dict):
        raise RuntimeError("infer failed in subprocess")


class TestRunStreaming:
    def test_yields_one_per_input(self):
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)} for _ in range(3)]
        results = list(
            Comparator.run(
                [_identity_runner("A"), _identity_runner("B")],
                data_loader=data,
                streaming=True,
            )
        )
        assert len(results) == 3
        for run_results in results:
            assert list(run_results.keys()) == ["A", "B"]
            assert len(run_results["A"]) == 1
            assert len(run_results["B"]) == 1

    def test_outputs_match_per_iteration(self):
        # Each iteration's output must correspond to its own input (no buffer aliasing across iterations).
        data = [{"x": np.full((1, 1, 2, 2), i, dtype=np.float32)} for i in range(4)]
        outputs = [
            np.array(run_results["A"][0]["y"])
            for run_results in Comparator.run(
                [_identity_runner("A")], data_loader=data, streaming=True
            )
        ]
        for i, output in enumerate(outputs):
            assert np.all(output == i)

    def test_is_lazy_generator(self):
        # run_streaming must be a generator that pulls inputs one at a time, not materialize them up front.
        pulled = []

        def gen():
            for i in range(3):
                pulled.append(i)
                yield {"x": np.full((1, 1, 2, 2), i, dtype=np.float32)}

        stream = Comparator.run(
            [_identity_runner("A")], data_loader=gen(), streaming=True
        )
        assert isinstance(stream, types.GeneratorType)
        # Nothing is consumed until the first item is requested.
        assert pulled == []
        next(stream)
        assert pulled == [0]
        # Each subsequent request pulls exactly one more input.
        next(stream)
        assert pulled == [0, 1]
        stream.close()

    def test_coerces_input_dtype(self):
        # float64 input should be coerced to the model's float32, not rejected.
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float64)}]
        results = list(
            Comparator.run([_identity_runner("A")], data_loader=data, streaming=True)
        )
        assert len(results) == 1
        assert np.asarray(results[0]["A"][0]["y"]).dtype == np.float32

    def test_no_runners_yields_nothing(self):
        assert list(Comparator.run([], streaming=True)) == []

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"warm_up": 1}, "warm_up"),
            ({"use_subprocess": True}, "use_subprocess"),
        ],
    )
    def test_rejects_warm_up_and_use_subprocess(self, kwargs, match):
        with pytest.raises(PolygraphyException, match=match):
            list(Comparator.run([_identity_runner("A")], streaming=True, **kwargs))

    def test_rejects_non_feed_dict_input(self):
        # A data loader yielding a non-feed_dict must fail with a clear message, not an opaque AttributeError.
        with pytest.raises(
            PolygraphyException, match="cannot be recognized as a feed_dict"
        ):
            list(
                Comparator.run(
                    [_identity_runner("A")], data_loader=[[1, 2, 3]], streaming=True
                )
            )

    @pytest.mark.parametrize("kwarg", ["save_inputs_path", "save_outputs_path"])
    def test_streaming_saves_per_iteration_files(self, tmp_path, kwarg):
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)} for _ in range(3)]
        save_dir = str(tmp_path / kwarg)
        list(
            Comparator.run(
                [_identity_runner("A")],
                data_loader=data,
                **{kwarg: save_dir},
                streaming=True,
            )
        )
        assert sorted(os.listdir(save_dir)) == [
            "0000000000.json",
            "0000000001.json",
            "0000000002.json",
        ]

    def test_coerces_inputs_per_runner(self):
        # Each runner receives inputs coerced to its own input metadata, not just the first runner's.
        received_a, received_b = [], []
        list(
            Comparator.run(
                [
                    _RecordingRunner("A", DataType.FLOAT32, received_a),
                    _RecordingRunner("B", DataType.FLOAT64, received_b),
                ],
                data_loader=[{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}],
                streaming=True,
            )
        )
        assert received_a == [DataType.FLOAT32]
        assert received_b == [DataType.FLOAT64]

    def test_rejects_nonempty_save_dir(self, tmp_path):
        # Saving per-iteration files into an existing, non-empty directory is rejected.
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        (out_dir / "leftover.json").write_text("{}")
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]
        with pytest.raises(PolygraphyException, match="not empty"):
            list(
                Comparator.run(
                    [_identity_runner("A")],
                    data_loader=data,
                    save_outputs_path=str(out_dir),
                    streaming=True,
                )
            )

    def test_run_streaming_saves_single_file(self, tmp_path):
        # A save_outputs_path with a file extension writes one combined file, not a directory.
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)} for _ in range(3)]
        out_file = str(tmp_path / "out.json")
        list(
            Comparator.run(
                [_identity_runner("A")],
                data_loader=data,
                save_outputs_path=out_file,
                streaming=True,
            )
        )
        assert os.path.isfile(out_file)
        assert len(RunResults.load(out_file)["A"]) == 3

    def test_single_file_save_warns_about_memory_once(self, tmp_path, monkeypatch):
        # Saving to a single file forfeits constant memory; the user is warned exactly once.
        warning_msgs = []
        monkeypatch.setattr(
            G_LOGGER, "warning", lambda msg, *a, **k: warning_msgs.append(str(msg))
        )
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)} for _ in range(3)]
        list(
            Comparator.run(
                [_identity_runner("A")],
                data_loader=data,
                save_outputs_path=str(tmp_path / "out.json"),
                streaming=True,
            )
        )
        memory_warnings = [
            w for w in warning_msgs if "accumulates every iteration" in w
        ]
        assert len(memory_warnings) == 1

    def test_single_file_save_flushes_on_early_close(self, tmp_path):
        # The single-file save must flush even if the consumer stops early, so produced iterations aren't lost.
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)} for _ in range(3)]
        out_file = str(tmp_path / "out.json")
        stream = Comparator.run(
            [_identity_runner("A")],
            data_loader=data,
            save_outputs_path=out_file,
            streaming=True,
        )
        next(stream)  # produce only the first iteration
        stream.close()  # stop early -> finally must still flush what was produced
        assert os.path.isfile(out_file)
        assert len(RunResults.load(out_file)["A"]) == 1

    def test_streaming_save_load_round_trip(self, tmp_path):
        # Streaming run with save_outputs_path → load_streaming → compare_accuracy round-trip.
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)} for _ in range(3)]
        out_dir = str(tmp_path / "outputs")
        list(
            Comparator.run(
                [_identity_runner("A"), _identity_runner("B")],
                data_loader=data,
                save_outputs_path=out_dir,
                streaming=True,
            )
        )
        (result,) = Comparator.compare_accuracy([RunResults.load_streaming(out_dir)])
        assert bool(result)
        assert result.stats(("A", "B")) == (3, 0, 3)


class TestCompareAccuracyStreaming:
    PAIR = ("A", "B")

    def test_output_stats_logged_once_per_iteration(self, monkeypatch):
        # Raw per-output stats must be logged once per output per iteration (both runners),
        # regardless of how many comparison functions are applied.
        from polygraphy.comparator import L2CompareFunc, PsnrCompareFunc
        from polygraphy.comparator import util as comp_util

        calls = []
        monkeypatch.setattr(
            comp_util,
            "log_output_stats",
            lambda *args, **kwargs: calls.append(args),
        )

        num_iters = 2
        stream = [
            _single_iteration({"r0": [1.0, 2.0, 3.0], "r1": [1.0, 2.0, 3.0]})
            for _ in range(num_iters)
        ]
        # Neither compare func logs raw output stats itself, so every call comes from the Comparator.
        Comparator.compare_accuracy(
            [stream],
            compare_func=[L2CompareFunc(), PsnrCompareFunc()],
        )

        # 1 output ("out") x 2 runners x num_iters -- NOT multiplied by the 2 compare functions.
        assert len(calls) == 2 * num_iters

    def test_duplicate_runner_names_compared_by_index(self):
        # Two runners share a name but differ; streaming must compare by index, not resolve the
        # name to the first runner (which would silently self-compare and pass). _single_iteration
        # can't be used here: its dict keying would collapse the duplicate "R" name.
        run_results = RunResults()
        for value in (0.0, 100.0):
            run_results.append(
                (
                    "R",
                    [
                        IterationResult(
                            outputs={"out": np.array([value], dtype=np.float32)},
                            runner_name="R",
                        )
                    ],
                )
            )
        (result,) = Comparator.compare_accuracy(
            [iter([run_results])],
            compare_func=SimpleCompareFunc(check_error_stat="mean"),
        )
        assert not bool(result)

    def test_fail_fast_stops_early(self):
        # The first iteration fails (|diff|=5 > 1.5), so comparison should stop immediately.
        stream = [
            _single_iteration({"r0": [0.0], "r1": [5.0]}),
            _single_iteration({"r0": [0.0], "r1": [0.0]}),
        ]
        (result,) = Comparator.compare_accuracy(
            [stream],
            compare_func=_AbsMeanDiffCompareFunc(threshold=1.5),
            fail_fast=True,
        )
        assert not bool(result)
        assert len(result[("r0", "r1")]) == 1

    def test_stream_accuracy_honors_explicit_comparisons(self):
        # An explicit comparisons argument overrides the default [(0, 1), (1, 2)] for 3 runners.
        stream = [_single_iteration({"r0": [0.0], "r1": [0.0], "r2": [0.0]})]
        (result,) = Comparator.compare_accuracy(
            [stream],
            compare_func=SimpleCompareFunc(check_error_stat="mean"),
            comparisons=[(0, 2)],
        )
        assert bool(result)
        assert list(result.keys()) == [("r0", "r2")]

    def test_zip_merge_multiple_streams(self):
        stream0 = (_single_iteration({"r0": [float(i)]}) for i in range(3))
        stream1 = (_single_iteration({"r1": [float(i)]}) for i in range(3))
        (result,) = Comparator.compare_accuracy(
            [stream0, stream1], compare_func=SimpleCompareFunc(check_error_stat="mean")
        )
        assert bool(result)
        assert result.stats(("r0", "r1")) == (3, 0, 3)

    def test_accepts_bare_stream(self):
        # A bare stream (not wrapped in a list) is a single run -- it must compare correctly rather
        # than being iterated during setup, which would eagerly drain it and mis-group its iterations.
        stream = (
            _single_iteration({"r0": [float(i)], "r1": [float(i)]}) for i in range(3)
        )
        (result,) = Comparator.compare_accuracy(
            stream, compare_func=SimpleCompareFunc(check_error_stat="mean")
        )
        assert list(result.keys()) == [("r0", "r1")]
        assert result.stats(("r0", "r1")) == (3, 0, 3)

    def test_empty_stream(self):
        (result,) = Comparator.compare_accuracy([iter([])])
        assert bool(result)

    def test_check_average_streaming_accumulates_all_iterations(self):
        # Average is computed from every iteration, not just the first or last.
        # iter 0: diff=0 (per-sample pass), iter 1: diff=2 (per-sample fail) -> avg=1.0 < 1.5 -> PASS.
        stream = [
            _single_iteration({"r0": [0.0], "r1": [0.0]}),
            _single_iteration({"r0": [0.0], "r1": [2.0]}),
        ]
        cf = _AbsMeanDiffCompareFunc(threshold=1.5)
        (result,) = Comparator.compare_accuracy(
            [stream], compare_func=cf, check_average=True
        )
        assert bool(result)
        assert result.stats(("r0", "r1")) == (1, 1, 2)

    def test_unequal_length_streams_warn(self, monkeypatch):
        # zip-merging stops at the shortest stream and warns with the concrete count actually compared.
        warning_msgs = []
        monkeypatch.setattr(
            G_LOGGER, "warning", lambda msg, *a, **k: warning_msgs.append(msg)
        )
        stream0 = (_single_iteration({"r0": [float(i)]}) for i in range(3))
        stream1 = (_single_iteration({"r1": [float(i)]}) for i in range(2))
        (result,) = Comparator.compare_accuracy(
            [stream0, stream1], compare_func=SimpleCompareFunc(check_error_stat="mean")
        )
        # Only the 2 overlapping iterations were compared.
        assert result.stats(("r0", "r1")) == (2, 0, 2)
        mismatch_warnings = [
            str(w) for w in warning_msgs if "different numbers of iterations" in str(w)
        ]
        assert mismatch_warnings
        # The warning must report the concrete number actually compared (2), not just "they differ".
        assert "first 2 iteration(s)" in mismatch_warnings[0]

    def test_accuracy_results_bool_false_when_any_func_fails(self):
        # AccuracyResults is falsy as soon as any contained AccuracyResult fails.
        stream = [_single_iteration({"r0": [0.0], "r1": [0.0]})]
        results = Comparator.compare_accuracy(
            [stream],
            compare_func=[
                _AbsMeanDiffCompareFunc(threshold=-1.0),  # always fails
                SimpleCompareFunc(check_error_stat="mean"),  # always passes
            ],
        )
        assert isinstance(results, AccuracyResults) and len(results) == 2
        assert not bool(results[0]) and bool(results[1])
        assert not bool(results)

    def test_check_average_shape_mismatch_is_fatal(self):
        # When an output has no comparable metrics in some iteration (e.g. a shape mismatch),
        # the whole averaged comparison must fail loudly rather than averaging only the rest.
        stream = [
            _single_iteration({"r0": [0.0, 1.0], "r1": [0.0, 1.0]}),
            # Second iteration has mismatched shapes -> bare-False result, no metrics to average.
            _single_iteration({"r0": [0.0, 1.0], "r1": [0.0, 1.0, 2.0]}),
        ]
        with pytest.raises(PolygraphyException, match="check_average"):
            Comparator.compare_accuracy(
                [stream],
                compare_func=SimpleCompareFunc(check_error_stat="mean"),
                check_average=True,
            )

    def test_stream_accuracy_runner_set_mismatch_is_fatal(self):
        # Comparison pairs are resolved from the first iteration, so a later iteration with a
        # different set of runners must fail loudly.
        def ragged_stream():
            yield _single_iteration({"r0": [0.0], "r1": [0.0]})
            # Second iteration is missing "r1" (as RunResults.split() omits a runner that ran out).
            yield _single_iteration({"r0": [0.0]})

        with pytest.raises(PolygraphyException, match="different set of runners"):
            Comparator.compare_accuracy(
                [ragged_stream()],
                compare_func=SimpleCompareFunc(check_error_stat="mean"),
            )

    def test_stream_accuracy_over_multi_iteration_file(self, tmp_path):
        # A saved run in a single multi-iteration file is compared iteration-by-iteration when streamed.
        path = str(tmp_path / "golden.json")
        _multi_iteration_results(
            {"A": [[0.0], [1.0], [2.0]], "B": [[0.0], [1.0], [2.0]]}
        ).save(path)
        (result,) = Comparator.compare_accuracy([RunResults.load_streaming(path)])
        assert bool(result)
        assert result.stats(("A", "B")) == (3, 0, 3)


class TestValidateStreaming:
    # Comparator.validate is stream-polymorphic: given a stream it returns a pass-through wrapper
    # that validates each iteration as it is consumed and aborts (G_LOGGER.critical) on an invalid value.
    @pytest.mark.parametrize(
        "value,kwargs",
        [(np.nan, {}), (np.inf, {"check_inf": True, "check_nan": False})],
        ids=["nan", "inf"],
    )
    def test_aborts_on_invalid_value(self, value, kwargs):
        # nan relies on the default check_nan=True; inf is caught only when check_inf is enabled.
        stream = [_single_iteration({"r0": [value], "r1": [value]})]
        with pytest.raises(PolygraphyException):
            list(Comparator.validate(iter(stream), **kwargs))

    def test_inf_valid_when_check_inf_disabled(self):
        stream = [_single_iteration({"r0": [np.inf], "r1": [np.inf]})]
        # No invalid values flagged, so consumption completes without raising.
        list(Comparator.validate(iter(stream), check_inf=False, check_nan=False))

    def test_passes_iterations_through_unchanged(self):
        # The wrapper only observes; it must yield exactly the iterations it was given.
        iterations = [
            _single_iteration({"r0": [0.0]}),
            _single_iteration({"r0": [1.0]}),
        ]
        passed_through = list(Comparator.validate(iter(iterations)))
        assert len(passed_through) == 2

    def test_validates_each_run_in_a_list(self):
        # A list of runs validates each: a NaN in any stream aborts when that stream is consumed.
        good = [_single_iteration({"r0": [1.0]})]
        bad = [_single_iteration({"r1": [np.nan]})]
        validations = Comparator.validate([iter(good), iter(bad)])
        with pytest.raises(PolygraphyException):
            for v in validations:
                list(v)

    def test_composes_with_stream_accuracy(self):
        # Validation wraps the live stream; an invalid value aborts the comparison that consumes it.
        stream = [_single_iteration({"r0": [np.nan]})]
        validation = Comparator.validate(iter(stream))
        with pytest.raises(PolygraphyException):
            Comparator.compare_accuracy([validation])

    def test_empty_stream_yields_nothing(self):
        assert list(Comparator.validate(iter([]))) == []


class TestPostprocessStreaming:
    def test_applies_to_each_iteration_lazily(self):
        # postprocess over a stream returns a generator that processes each iteration as consumed.
        stream = (_single_iteration({"r0": [0.0, 1.0, 5.0, 2.0]}) for _ in range(2))
        postprocessed = Comparator.postprocess(stream, TopKPostprocessFunc(k=(2, -1)))
        assert isinstance(postprocessed, types.GeneratorType)
        out = list(postprocessed)
        assert len(out) == 2
        # top-2 along the last axis -> indices of the two largest values (5.0 @ idx 2, 2.0 @ idx 3).
        result = out[0]["r0"][0]["out"]
        assert sorted(int(i) for i in result) == [2, 3]

    def test_applies_to_each_run_in_a_list(self):
        # A list of runs post-processes each (streams stay lazy), preserving the list shape.
        runs = [
            iter([_single_iteration({"r0": [0.0, 9.0, 1.0]})]),
            iter([_single_iteration({"r1": [7.0, 0.0, 0.0]})]),
        ]
        out = Comparator.postprocess(runs, TopKPostprocessFunc(k=1))
        assert isinstance(out, list) and len(out) == 2
        assert int(list(out[0])[0]["r0"][0]["out"][0]) == 1  # argmax of [0,9,1]
        assert int(list(out[1])[0]["r1"][0]["out"][0]) == 0  # argmax of [7,0,0]

    def test_empty_stream_yields_nothing(self):
        out = list(Comparator.postprocess(iter([]), TopKPostprocessFunc(k=1)))
        assert out == []
