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
import threading

import numpy as np
import pytest
import torch

from polygraphy import config, cuda, mod
from polygraphy.backend.trt import (
    EngineFromNetwork,
    NetworkFromOnnxBytes,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_bytes,
)
from polygraphy.backend.trt.runner import _get_array_on_cpu
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER
from tests.models.meta import ONNX_MODELS

# Import CreateConfigRTX conditionally for TensorRT-RTX builds
if config.USE_TENSORRT_RTX:
    import tensorrt_rtx as trt
    from polygraphy.backend.tensorrt_rtx import CreateConfigRTX as CreateConfig
else:
    import tensorrt as trt
    from polygraphy.backend.trt import CreateConfig


class TestLoggerCallbacks:
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.module_severity = sev


@pytest.fixture(scope="class")
def nonzero_engine():
    model = ONNX_MODELS["nonzero"]
    network_loader = NetworkFromOnnxBytes(model.loader)
    return engine_from_network(network_loader)


@pytest.fixture()
def identity_engine():
    model = ONNX_MODELS["identity"]
    network_loader = NetworkFromOnnxBytes(model.loader)
    return engine_from_network(network_loader)


@pytest.fixture()
def reducable_engine():
    model = ONNX_MODELS["reducable"]
    network_loader = NetworkFromOnnxBytes(model.loader)
    return engine_from_network(network_loader)


class TestTrtRunner:
    def test_can_name_runner(self):
        NAME = "runner"
        runner = TrtRunner(None, name=NAME)
        assert runner.name == NAME

    def test_basic(self, identity_engine):
        with TrtRunner(identity_engine) as runner:
            assert runner.optimization_profile is None
            assert runner.is_active
            ONNX_MODELS["identity"].check_runner(runner)
            assert runner.last_inference_time() is not None
        assert not runner.is_active

    @pytest.mark.serial
    @pytest.mark.skipif(config.USE_TENSORRT_RTX, reason="TensorRT-RTX has different warning output behavior")
    def test_warn_if_impl_methods_called(self, check_warnings_on_runner_impl_methods, identity_engine):
        runner = TrtRunner(identity_engine)
        check_warnings_on_runner_impl_methods(runner)

    @pytest.mark.parametrize(
        "inp, expected",
        [
            ([1, 0, 1, 1], [[0, 2, 3]]),
            ([1, 0, 0, 1], [[0, 3]]),
            ([0, 0, 0, 1], [[3]]),
        ],
    )
    @pytest.mark.skipif(config.USE_TENSORRT_RTX, reason="TensorRT-RTX does not support data dependent shapes")
    def test_data_dependent_shapes(self, nonzero_engine, inp, expected):
        with TrtRunner(nonzero_engine) as runner:
            outputs = runner.infer(
                {
                    "input": np.array(
                        inp,
                        dtype=(np.int32 if mod.version(trt.__version__) < mod.version("9.0") else np.int64),
                    )
                }
            )
            assert np.array_equal(outputs["nonzero_out_0"], np.array(expected, dtype=np.int32))

    @pytest.mark.parametrize("copy_outputs_to_host", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_torch_tensors(self, copy_outputs_to_host, identity_engine, device):
        with TrtRunner(identity_engine) as runner:
            arr = torch.ones([1, 1, 2, 2], dtype=torch.float32, device=device)
            outputs = runner.infer({"x": arr}, copy_outputs_to_host=copy_outputs_to_host)
            assert all(isinstance(t, torch.Tensor) for t in outputs.values())

            assert torch.equal(outputs["y"].to("cpu"), arr.to("cpu"))

            assert outputs["y"].device.type == ("cpu" if copy_outputs_to_host else "cuda")

    def test_context(self, identity_engine):
        with TrtRunner(identity_engine.create_execution_context) as runner:
            ONNX_MODELS["identity"].check_runner(runner)

    def test_device_buffer_order_matches_bindings(self, reducable_engine):
        with TrtRunner(reducable_engine) as runner:
            dev_buf_order = list(runner.device_input_buffers.keys())
            for binding, dev_buf_name in zip(reducable_engine, dev_buf_order):
                assert binding == dev_buf_name

    def test_shape_output(self):
        model = ONNX_MODELS["reshape"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))
        with engine, TrtRunner(engine.create_execution_context) as runner:
            model.check_runner(runner)

    def test_multithreaded_runners_from_engine(self, identity_engine):
        with TrtRunner(identity_engine) as runner0, TrtRunner(identity_engine) as runner1:
            t1 = threading.Thread(target=ONNX_MODELS["identity"].check_runner, args=(runner0,))
            t2 = threading.Thread(target=ONNX_MODELS["identity"].check_runner, args=(runner1,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    @pytest.mark.parametrize("use_optimization_profile", [True, False])
    @pytest.mark.skipif(
        not config.USE_TENSORRT_RTX
        and mod.version(trt.__version__) >= mod.version("8.6")
        and mod.version(trt.__version__) < mod.version("8.7"),
        reason="Bug in TRT 8.6",
    )
    def test_multiple_profiles(self, use_optimization_profile):
        model = ONNX_MODELS["dynamic_identity"]
        profile0_shapes = [
            (1, 2, 1, 1),
            (1, 2, 1, 1),
            (1, 2, 1, 1),
        ]  # Use min==opt==max to fix shapes in the engine.
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

    @pytest.mark.skipif(
        not config.USE_TENSORRT_RTX and mod.version(trt.__version__) < mod.version("10.0"),
        reason="Feature not present before 10.0",
    )
    @pytest.mark.parametrize("allocation_strategy", [None, "static", "profile", "runtime"])
    def test_allocation_strategies(self, allocation_strategy):
        if config.USE_TENSORRT_RTX and allocation_strategy == "runtime":
            pytest.skip("TensorRT-RTX issues with runtime allocation strategy")

        model = ONNX_MODELS["residual_block"]
        profile0_shapes = [(1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224)]
        profile1_shapes = [(1, 3, 224, 224), (1, 3, 224, 224), (2, 3, 224, 224)]
        profile2_shapes = [(1, 3, 224, 224), (1, 3, 224, 224), (4, 3, 224, 224)]
        network_loader = NetworkFromOnnxBytes(model.loader)
        profiles = [
            Profile().add("gpu_0/data_0", *profile0_shapes),
            Profile().add("gpu_0/data_0", *profile1_shapes),
            Profile().add("gpu_0/data_0", *profile2_shapes),
        ]
        config_loader = CreateConfig(profiles=profiles)
        engine = engine_from_network(network_loader, config_loader)

        for index, shapes in enumerate([profile0_shapes, profile1_shapes, profile2_shapes]):
            with TrtRunner(
                engine,
                optimization_profile=index,
                allocation_strategy=allocation_strategy,
            ) as runner:
                for shape in shapes:
                    model.check_runner(runner, {"gpu_0/data_0": shape})

    def test_empty_tensor_with_dynamic_input_shape_tensor(self):
        model = ONNX_MODELS["empty_tensor_expand"]
        shapes = [(1, 2, 0, 3, 0), (2, 2, 0, 3, 0), (4, 2, 0, 3, 0)]
        network_loader = NetworkFromOnnxBytes(model.loader)
        profiles = [Profile().add("new_shape", *shapes)]
        config_loader = CreateConfig(profiles=profiles)

        with TrtRunner(EngineFromNetwork(network_loader, config_loader)) as runner:
            for shape in shapes:
                model.check_runner(runner, {"new_shape": shape})

    @pytest.mark.parametrize(
        "names, err",
        [
            (["fake-input", "x"], "Extra inputs in"),
            (["fake-input"], "The following inputs were not found"),
            ([], "The following inputs were not found"),
        ],
    )
    @pytest.mark.parametrize("module", [torch, np])
    def test_error_on_wrong_name_feed_dict(self, names, err, identity_engine, module):
        with TrtRunner(identity_engine) as runner:
            with pytest.raises(PolygraphyException, match=err):
                runner.infer({name: module.ones((1, 1, 2, 2), dtype=module.float32) for name in names})

    @pytest.mark.parametrize("module", [torch, np])
    def test_error_on_wrong_dtype_feed_dict(self, identity_engine, module):
        with TrtRunner(identity_engine) as runner:
            with pytest.raises(PolygraphyException, match="unexpected dtype."):
                runner.infer({"x": module.ones((1, 1, 2, 2), dtype=module.int32)})

    @pytest.mark.parametrize("module", [torch, np])
    def test_error_on_wrong_shape_feed_dict(self, identity_engine, module):
        with TrtRunner(identity_engine) as runner:
            with pytest.raises(PolygraphyException, match="incompatible shape."):
                runner.infer({"x": module.ones((1, 1, 3, 2), dtype=module.float32)})

    @pytest.mark.parametrize("use_view", [True, False])  # We should be able to use DeviceArray in place of DeviceView
    def test_device_views(self, use_view, reducable_engine):
        with TrtRunner(reducable_engine) as runner, cuda.DeviceArray((1,), dtype=np.float32) as x:
            x.copy_from(np.ones((1,), dtype=np.float32))
            outputs = runner.infer(
                {
                    "X0": x.view() if use_view else x,
                    "Y0": np.ones((1,), dtype=np.float32),
                }
            )
            assert outputs["identity_out_6"][0] == 2
            assert outputs["identity_out_8"][0] == 2

    def test_no_output_copy(self, identity_engine):
        with TrtRunner(identity_engine) as runner:
            inp = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)
            outputs = runner.infer({"x": inp}, copy_outputs_to_host=False)
            assert isinstance(outputs["y"], cuda.DeviceView)
            assert np.array_equal(outputs["y"].numpy(), inp)

    def test_subsequent_infers_with_different_input_types(self, identity_engine):
        with TrtRunner(identity_engine) as runner:
            inp = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

            def check(outputs):
                assert np.all(outputs["y"] == inp)

            check(runner.infer({"x": inp}))
            check(runner.infer({"x": cuda.DeviceArray(shape=inp.shape, dtype=inp.dtype).copy_from(inp)}))

            torch_outputs = runner.infer({"x": torch.from_numpy(inp)})
            check({name: out.numpy() for name, out in torch_outputs.items()})
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
            outputs = runner.infer({"X": (cuda.DeviceView(arr.ptr, arr.shape, arr.dtype) if use_view else arr)})
            assert np.all(outputs["Y"] == inp)
            assert outputs["Y"].shape == (1, 2, 3, 3)

    def test_cannot_use_device_view_shape_tensor(self):
        model = ONNX_MODELS["empty_tensor_expand"]
        with TrtRunner(EngineFromNetwork(NetworkFromOnnxBytes(model.loader))) as runner, cuda.DeviceArray(
            shape=(5,),
            dtype=(
                np.int32
                if mod.version(trt.__version__) < mod.version("9.0") and not config.USE_TENSORRT_RTX
                else np.int64
            ),
        ) as arr:
            with pytest.raises(PolygraphyException, match="it must reside in host memory"):
                runner.infer({"data": np.ones((2, 0, 3, 0), dtype=np.float32), "new_shape": arr})

    @pytest.mark.parametrize("hwc_input", [True, False], ids=["hwc_input", "chw_input"])
    @pytest.mark.parametrize("hwc_output", [True, False], ids=["hwc_output", "chw_output"])
    @pytest.mark.skipif(config.USE_TENSORRT_RTX, reason="TensorRT-RTX does not support custom I/O format networks")
    def test_infer_chw_format(self, hwc_input, hwc_output):
        model = ONNX_MODELS["identity_multi_ch"]
        inp_shape = model.input_metadata["x"].shape
        builder, network, parser = network_from_onnx_bytes(model.loader)

        formats = 1 << int(trt.TensorFormat.HWC)
        if hwc_input:
            network.get_input(0).allowed_formats = formats
        if hwc_output:
            network.get_output(0).allowed_formats = formats

        engine = engine_from_network((builder, network))

        with TrtRunner(engine) as runner:
            inp = np.random.normal(size=(inp_shape)).astype(np.float32)
            if hwc_input:
                inp = inp.transpose(0, 2, 3, 1)

            outputs = runner.infer({"x": inp})
            if hwc_input == hwc_output:  # output in CHW/HWC format and similarly shaped
                assert np.allclose(outputs["y"], inp)
            elif not hwc_input and hwc_output:  # output in HWC format and shaped (N, H, W, C)
                assert np.allclose(outputs["y"].transpose(0, 3, 1, 2), inp)
            else:  # hwc_input and not hwc_output: output in CHW format and shaped (N, C, H, W)
                assert np.allclose(outputs["y"].transpose(0, 2, 3, 1), inp)

    @pytest.mark.parametrize("use_torch", [True, False])
    def test_get_array_on_cpu(self, use_torch):
        shape = (4,)
        with cuda.DeviceArray.raw(shape) as arr:
            host_buffers = {}
            stream = cuda.Stream()
            host_arr = _get_array_on_cpu(arr, "test", host_buffers, stream, arr.nbytes, use_torch)

            if use_torch:
                assert isinstance(host_arr, torch.Tensor)
            else:
                assert isinstance(host_arr, np.ndarray)

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("10.0") and not config.USE_TENSORRT_RTX,
        reason="Feature not present before 10.0",
    )
    @pytest.mark.parametrize("budget", [None, -2, -1, 0, 0.5, 0.99, 1.0, 1000, np.inf])
    def test_weight_streaming(self, budget):
        model = ONNX_MODELS["matmul_2layer"]
        network_loader = NetworkFromOnnxBytes(model.loader, strongly_typed=True)
        config_loader = CreateConfig(weight_streaming=True)
        engine = engine_from_network(network_loader, config_loader)

        if budget == np.inf:
            # set to max size - 1
            budget = engine.streamable_weights_size - 1

        kwargs = {"weight_streaming_budget": None, "weight_streaming_percent": None}
        if budget is not None:
            if 0 < budget <= 1:
                kwargs["weight_streaming_percent"] = budget * 100
            else:
                kwargs["weight_streaming_budget"] = int(budget)

        with TrtRunner(engine, optimization_profile=0, **kwargs) as runner:
            model.check_runner(runner)

    @pytest.mark.skipif(not config.USE_TENSORRT_RTX, reason="TensorRT-RTX not enabled")
    def test_compute_capabilities_engine_building(self):
        """Test compute capabilities integration with engine building"""
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)

        # Test --use-gpu flag
        config_loader = CreateConfig(use_gpu=True)
        engine = engine_from_network(network_loader, config_loader)
        with TrtRunner(engine) as runner:
            model.check_runner(runner)

        # Test --compute-capabilities flag
        config_loader = CreateConfig(compute_capabilities=[(7, 5), (8, 0), (8, 6)])
        engine = engine_from_network(network_loader, config_loader)
        with TrtRunner(engine) as runner:
            model.check_runner(runner)

    @pytest.mark.skipif(not config.USE_TENSORRT_RTX, reason="TensorRT-RTX not enabled")
    def test_compute_capabilities_mutual_exclusion(self):
        """Test that use_gpu and compute_capabilities are mutually exclusive"""
        # Test mutual exclusion - should raise an exception
        with pytest.raises(PolygraphyException, match="use_gpu and compute_capabilities are mutually exclusive"):
            CreateConfig(use_gpu=True, compute_capabilities=[(7, 5)])
