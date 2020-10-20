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
from polygraphy.backend.onnx import OnnxFromTfGraph, BytesFromOnnx
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnxBytes
from polygraphy.comparator import Comparator, CompareFunc, PostprocessFunc, RunResults, IterationResult, DataLoader
from polygraphy.backend.trt import TrtRunner, EngineFromNetwork, NetworkFromOnnxBytes
from polygraphy.common import PolygraphyException
from polygraphy.backend.tf import TfRunner, SessionFromGraph

from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.common import version

import subprocess as sp
import tensorrt as trt
import numpy as np
import pytest


class TestComparator(object):
    def test_warmup_runs(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnxBytes(onnx_loader))
        run_results = Comparator.run([runner], warm_up=2)
        assert len(run_results[runner.name]) == 1


    def test_list_as_data_loader(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnxBytes(onnx_loader), name="onnx_runner")

        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}] * 2
        run_results = Comparator.run([runner], data_loader=data)
        iter_results = run_results["onnx_runner"]
        assert len(iter_results) == 2
        for actual, expected in zip(iter_results, data):
            assert np.all(actual['y'] == expected['x'])


    def test_generator_as_data_loader(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnxBytes(onnx_loader), name="onnx_runner")

        def data():
            for feed_dict in [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}] * 2:
                yield feed_dict

        run_results = Comparator.run([runner], data_loader=data())
        iter_results = run_results["onnx_runner"]
        assert len(iter_results) == 2
        for actual, expected in zip(iter_results, data()):
            assert np.all(actual['y'] == expected['x'])


    def test_multiple_runners(self):
        load_tf = TF_MODELS["identity"].loader
        build_tf_session = SessionFromGraph(load_tf)
        load_serialized_onnx = BytesFromOnnx(OnnxFromTfGraph(load_tf))
        build_onnxrt_session = SessionFromOnnxBytes(load_serialized_onnx)
        load_engine = EngineFromNetwork(NetworkFromOnnxBytes(load_serialized_onnx))

        runners = [
            TfRunner(build_tf_session),
            OnnxrtRunner(build_onnxrt_session),
            TrtRunner(load_engine),
        ]

        run_results = Comparator.run(runners)
        compare_func = CompareFunc.basic_compare_func(check_shapes=version(trt.__version__) >= version("7.0"))
        assert bool(Comparator.compare_accuracy(run_results, compare_func=compare_func))
        assert len(list(run_results.values())[0]) == 1 # Default number of iterations


    def test_postprocess(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        run_results = Comparator.run([OnnxrtRunner(SessionFromOnnxBytes(onnx_loader))], use_subprocess=True)
        # Output shape is (1, 1, 2, 2)
        postprocessed = Comparator.postprocess(run_results, postprocess_func=PostprocessFunc.topk_func(k=1, axis=-1))
        for name, results in postprocessed.items():
            for result in results:
                for name, output in result.items():
                    assert output.shape == (1, 1, 2, 1)


    # When there is an unpickleable exception in the subprocess, the Comparator should be able to recover and exit gracefully.
    def test_errors_do_not_hang(self):
        # Should error because interface is not implemented correctly.
        class FakeRunner(object):
            def __init__(self):
                self.name = "fake"

        runners = [FakeRunner()]
        with pytest.raises(PolygraphyException):
            Comparator.run(runners, use_subprocess=True, subprocess_polling_interval=1)


    # When there is an unpickleable exception in the subprocess, the Comparator should be able to recover and exit gracefully.
    def test_segfault_does_not_hang(self):
        def raise_called_process_error():
            class UnpickleableException(sp.CalledProcessError):
                pass

            raise UnpickleableException(-11, ["simulate", "segfault"])

        runners = [TrtRunner(EngineFromNetwork(raise_called_process_error))]
        with pytest.raises(PolygraphyException):
            Comparator.run(runners, use_subprocess=True, subprocess_polling_interval=1)


    def test_multirun_outputs_are_different(self):
        onnx_loader = ONNX_MODELS["identity"].loader
        runner = OnnxrtRunner(SessionFromOnnxBytes(onnx_loader))
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
        assert not Comparator.validate(run_results, check_finite=True)
