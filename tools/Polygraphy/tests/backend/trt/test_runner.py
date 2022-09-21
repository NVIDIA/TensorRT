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
import threading

import numpy as np
import pytest
import tensorrt as trt
from polygraphy import cuda, mod
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxBytes,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_bytes,
)
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER
from tests.models.meta import ONNX_MODELS
from tests.helper import time_func


class TestLoggerCallbacks:
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.module_severity = sev


@pytest.fixture(scope="class")
def nonzero_engine():
    model = ONNX_MODELS["nonzero"]
    network_loader = NetworkFromOnnxBytes(model.loader)
    return engine_from_network(network_loader)


class TestTrtRunner:
    def test_can_name_runner(self):
        NAME = "runner"
        runner = TrtRunner(None, name=NAME)
        assert runner.name == NAME

    def test_basic(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            assert runner.optimization_profile is None
            assert runner.is_active
            assert runner.owns_engine
            assert runner.owns_context
            model.check_runner(runner)
            assert runner.last_inference_time() is not None
        assert not runner.is_active

    @pytest.mark.skipif(
        mod.version(trt.__version__) <= mod.version("8.5.0.9"), reason="Unsupported for TRT 8.4 and older"
    )
    @pytest.mark.parametrize(
        "inp, expected",
        [
            ([1, 0, 1, 1], [[0, 2, 3]]),
            ([1, 0, 0, 1], [[0, 3]]),
            ([0, 0, 0, 1], [[3]]),
        ],
    )
    def test_data_dependent_shapes(self, nonzero_engine, inp, expected):
        with TrtRunner(nonzero_engine) as runner:
            outputs = runner.infer({"input": np.array(inp, dtype=np.int32)})
            assert np.array_equal(outputs["nonzero_out_0"], np.array(expected, dtype=np.int32))

    def test_context(self):
        model = ONNX_MODELS["identity"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))
        with engine, TrtRunner(engine.create_execution_context) as runner:
            model.check_runner(runner)
            assert not runner.owns_engine
            assert runner.owns_context

    def test_device_buffer_order_matches_bindings(self):
        model = ONNX_MODELS["reducable"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))
        with engine, TrtRunner(engine) as runner:
            dev_buf_order = list(runner.device_buffers.keys())
            for binding, dev_buf_name in zip(engine, dev_buf_order):
                assert binding == dev_buf_name

    def test_shape_output(self):
        model = ONNX_MODELS["reshape"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))
        with engine, TrtRunner(engine.create_execution_context) as runner:
            model.check_runner(runner)

    def test_multithreaded_runners_from_engine(self):
        model = ONNX_MODELS["identity"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))

        with engine, TrtRunner(engine) as runner0, TrtRunner(engine) as runner1:
            t1 = threading.Thread(target=model.check_runner, args=(runner0,))
            t2 = threading.Thread(target=model.check_runner, args=(runner1,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    @pytest.mark.skipif(mod.version(trt.__version__)[0:2] == mod.version("7.2"), reason="Bugged in TRT 7.2")
    @pytest.mark.parametrize("use_optimization_profile", [True, False])
    def test_multiple_profiles(self, use_optimization_profile):
        model = ONNX_MODELS["dynamic_identity"]
        profile0_shapes = [(1, 2, 1, 1), (1, 2, 1, 1), (1, 2, 1, 1)]  # Use min==opt==max to fix shapes in the engine.
        profile1_shapes = [(1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4)]
        profile2_shapes = [(1, 2, 4, 4), (1, 2, 8, 8), (1, 2, 16, 16)]
        network_loader = NetworkFromOnnxBytes(model.loader)
        profiles = [
            Profile().add("X", *profile0_shapes),
            Profile().add("X", *profile1_shapes),
            Profile().add("X", *profile2_shapes),
        ]
        config_loader = CreateConfig(profiles=profiles)
        engine = engine_from_network(network_loader, config_loader)
        context = engine.create_execution_context()

        for index, shapes in enumerate([profile0_shapes, profile1_shapes, profile2_shapes]):
            with TrtRunner(
                context,
                optimization_profile=index if use_optimization_profile else None,
            ) as runner:
                if not use_optimization_profile:
                    runner.set_profile(index)

                assert runner.context.active_optimization_profile == index
                for shape in shapes:
                    model.check_runner(runner, {"X": shape})

    def test_empty_tensor_with_dynamic_input_shape_tensor(self):
        model = ONNX_MODELS["empty_tensor_expand"]
        shapes = [(1, 2, 0, 3, 0), (2, 2, 0, 3, 0), (4, 2, 0, 3, 0)]
        network_loader = NetworkFromOnnxBytes(model.loader)
        profiles = [Profile().add("new_shape", *shapes)]
        config_loader = CreateConfig(profiles=profiles)

        with TrtRunner(EngineFromNetwork(network_loader, config_loader)) as runner:
            for shape in shapes:
                model.check_runner(runner, {"new_shape": shape})

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Test not compatible with TRT 6")
    @pytest.mark.parametrize(
        "names, err",
        [
            (["fake-input", "x"], "Extra inputs in"),
            (["fake-input"], "The following inputs were not found"),
            ([], "The following inputs were not found"),
        ],
    )
    def test_error_on_wrong_name_feed_dict(self, names, err):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            with pytest.raises(PolygraphyException, match=err):
                runner.infer({name: np.ones(shape=(1, 1, 2, 2), dtype=np.float32) for name in names})

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Test not compatible with TRT 6")
    def test_error_on_wrong_dtype_feed_dict(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            with pytest.raises(PolygraphyException, match="unexpected dtype."):
                runner.infer({"x": np.ones(shape=(1, 1, 2, 2), dtype=np.int32)})

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Test not compatible with TRT 6")
    def test_error_on_wrong_shape_feed_dict(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            with pytest.raises(PolygraphyException, match="incompatible shape."):
                runner.infer({"x": np.ones(shape=(1, 1, 3, 2), dtype=np.float32)})

    @pytest.mark.parametrize("use_view", [True, False])  # We should be able to use DeviceArray in place of DeviceView
    def test_device_views(self, use_view):
        model = ONNX_MODELS["reducable"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner, cuda.DeviceArray((1,), dtype=np.float32) as x:
            x.copy_from(np.ones((1,), dtype=np.float32))
            outputs = runner.infer(
                {
                    "X0": x.view() if use_view else x,
                    "Y0": np.ones((1,), dtype=np.float32),
                }
            )
            assert outputs["identity_out_6"][0] == 2
            assert outputs["identity_out_8"][0] == 2

    def test_no_output_copy(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            inp = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)
            outputs = runner.infer({"x": inp}, copy_outputs_to_host=False)
            assert isinstance(outputs["y"], cuda.DeviceView)
            assert np.array_equal(outputs["y"].numpy(), inp)

    def test_subsequent_infers_with_different_input_types(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            inp = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

            def check(outputs):
                assert np.all(outputs["y"] == inp)

            check(runner.infer({"x": inp}))
            check(runner.infer({"x": cuda.DeviceArray(shape=inp.shape, dtype=inp.dtype).copy_from(inp)}))
            check(runner.infer({"x": inp}))

    @pytest.mark.parametrize("use_view", [True, False])  # We should be able to use DeviceArray in place of DeviceView
    def test_device_view_dynamic_shapes(self, use_view):
        model = ONNX_MODELS["dynamic_identity"]
        profiles = [
            Profile().add("X", (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4)),
        ]
        runner = TrtRunner(EngineFromNetwork(NetworkFromOnnxBytes(model.loader), CreateConfig(profiles=profiles)))
        with runner, cuda.DeviceArray(shape=(1, 2, 3, 3), dtype=np.float32) as arr:
            inp = np.random.random_sample(size=(1, 2, 3, 3)).astype(np.float32)
            arr.copy_from(inp)
            outputs = runner.infer({"X": cuda.DeviceView(arr.ptr, arr.shape, arr.dtype) if use_view else arr})
            assert np.all(outputs["Y"] == inp)
            assert outputs["Y"].shape == (1, 2, 3, 3)

    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported before TRT 8")
    def test_cannot_use_device_view_shape_tensor(self):
        model = ONNX_MODELS["empty_tensor_expand"]
        with TrtRunner(EngineFromNetwork(NetworkFromOnnxBytes(model.loader))) as runner, cuda.DeviceArray(
            shape=(5,), dtype=np.int32
        ) as arr:
            with pytest.raises(PolygraphyException, match="it must reside in host memory"):
                runner.infer({"data": np.ones((2, 0, 3, 0), dtype=np.float32), "new_shape": arr})

    @pytest.mark.flaky
    @pytest.mark.serial
    @pytest.mark.parametrize("copy_outputs", [True, False], ids=["output_dtoh", "no_output_copy"])
    @pytest.mark.parametrize("copy_inputs", [True, False], ids=["input_htod", "no_input_copy"])
    def test_infer_overhead(self, copy_inputs, copy_outputs):
        model = ONNX_MODELS["needs_constraints"]
        inp_name = list(model.input_metadata.keys())[0]
        inp_shape = model.input_metadata[inp_name].shape

        inp = np.ones(shape=inp_shape, dtype=np.float32)
        dev_inp = cuda.DeviceArray(shape=inp.shape, dtype=inp.dtype)
        dev_inp.copy_from(inp)

        out = np.zeros(shape=inp_shape, dtype=np.float32)
        dev_out = cuda.DeviceArray(shape=out.shape, dtype=out.dtype)

        with engine_from_network(
            network_from_onnx_bytes(model.loader)
        ) as engine, engine.create_execution_context() as context, TrtRunner(
            context
        ) as runner, dev_inp, dev_out, cuda.Stream() as stream:
            # Inference outside the TrtRunner
            def infer():
                if copy_inputs:
                    dev_inp.copy_from(inp, stream=stream)
                context.execute_async_v2(bindings=[dev_inp.ptr, dev_out.ptr], stream_handle=stream.ptr)
                if copy_outputs:
                    dev_out.copy_to(out, stream=stream)
                stream.synchronize()

            native_time = time_func(infer)

            feed_dict = {inp_name: (inp if copy_inputs else dev_inp)}

            def runner_infer():
                runner.infer(feed_dict, check_inputs=False, copy_outputs_to_host=copy_outputs)

            runner_time = time_func(runner_infer)

        print(f"Absolute difference: {runner_time - native_time:.5g}")
        print(f"Relative difference: {runner_time / native_time:.5g}")
        assert (runner_time - native_time) < 1e-3 or runner_time <= (native_time * 1.10)
