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
import subprocess as sp

import numpy as np
import pytest
import tensorrt as trt
from polygraphy import mod
from polygraphy.backend.onnx import GsFromOnnx, OnnxFromBytes
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.pluginref import PluginRefRunner
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxBytes, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc, DataLoader, IterationResult, PostprocessFunc, RunResults
from polygraphy.exception import PolygraphyException
from tests.models.meta import ONNX_MODELS


class TestComparator:
    def test_warmup_runs(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnx(onnx_loader))
        run_results = Comparator.run([runner], warm_up=2)
        assert len(run_results[runner.name]) == 1

    def test_list_as_data_loader(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnx(onnx_loader), name="onnx_runner")

        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}] * 2
        run_results = Comparator.run([runner], data_loader=data)
        iter_results = run_results["onnx_runner"]
        assert len(iter_results) == 2
        for actual, expected in zip(iter_results, data):
            assert np.all(actual["y"] == expected["x"])

    def test_generator_as_data_loader(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnx(onnx_loader), name="onnx_runner")

        def data():
            for feed_dict in [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}] * 2:
                yield feed_dict

        run_results = Comparator.run([runner], data_loader=data())
        iter_results = run_results["onnx_runner"]
        assert len(iter_results) == 2
        for actual, expected in zip(iter_results, data()):
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
        compare_func = CompareFunc.simple(check_shapes=True)
        assert bool(Comparator.compare_accuracy(run_results, compare_func=compare_func))
        assert len(list(run_results.values())[0]) == 1  # Default number of iterations

    def test_postprocess(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        run_results = Comparator.run([OnnxrtRunner(SessionFromOnnx(onnx_loader))], use_subprocess=True)
        # Output shape is (1, 1, 2, 2)
        postprocessed = Comparator.postprocess(run_results, postprocess_func=PostprocessFunc.top_k(k=(1, -1)))
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

    def test_multirun_outputs_are_different(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = TrtRunner(EngineFromNetwork(NetworkFromOnnxBytes(onnx_loader)))
        run_results = Comparator.run([runner], data_loader=DataLoader(iterations=2))

        iteration0 = run_results[runner.name][0]
        iteration1 = run_results[runner.name][1]
        for name in iteration0.keys():
            assert np.any(iteration0[name] != iteration1[name])

    def test_validate_nan(self):
        run_results = RunResults()
        run_results["fake-runner"] = [IterationResult(outputs={"x": np.array(np.nan)})]
        assert not Comparator.validate(run_results)

    def test_validate_inf(self):
        run_results = RunResults()
        run_results["fake-runner"] = [IterationResult(outputs={"x": np.array(np.inf)})]
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
        compare_func = CompareFunc.simple(check_shapes=True)
        assert bool(Comparator.compare_accuracy(run_results, compare_func=compare_func))
        assert len(list(run_results.values())[0]) == 1  # Default number of iterations
