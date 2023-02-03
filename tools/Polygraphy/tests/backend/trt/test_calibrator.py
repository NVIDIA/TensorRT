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
import numpy as np
import pytest
import tensorrt as trt
from polygraphy import cuda, mod, util
from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    engine_from_network,
    get_trt_logger,
    Profile,
    network_from_onnx_bytes,
)
from polygraphy.common import TensorMetadata
from polygraphy.exception import PolygraphyException
from tests.helper import get_file_size, is_file_non_empty
from tests.models.meta import ONNX_MODELS


@pytest.fixture(scope="session")
def identity_builder_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["identity"].loader)
    with builder, network, parser:
        yield builder, network


@pytest.fixture(scope="session")
def dynamic_identity_builder_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["dynamic_identity"].loader)
    with builder, network, parser:
        yield builder, network


@pytest.fixture(scope="session")
def multi_input_builder_network():
    builder, network, parser = network_from_onnx_bytes(ONNX_MODELS["reducable"].loader)
    with builder, network, parser:
        yield builder, network


def generate_data(num_batches):
    for item in [np.ones((1, 1, 2, 2), dtype=np.float32)] * num_batches:
        yield {"x": item}


class TestCalibrator:
    def check_calibrator_cleanup(self, calibrator):
        # Calibrator buffers should be freed after the build
        assert all([buf.allocated_nbytes == 0 for buf in calibrator.device_buffers.values()])

    @pytest.mark.parametrize(
        "BaseClass",
        [
            trt.IInt8Calibrator,
            trt.IInt8LegacyCalibrator,
            trt.IInt8EntropyCalibrator,
            trt.IInt8EntropyCalibrator2,
            trt.IInt8MinMaxCalibrator,
        ],
    )
    def test_calibrator_basic(self, identity_builder_network, BaseClass):
        if mod.version(trt.__version__) < mod.version("7.0") and BaseClass == trt.IInt8LegacyCalibrator:
            pytest.skip("Bug in TRT 6 causes NaNs with legacy calibrator")

        builder, network = identity_builder_network
        NUM_BATCHES = 2

        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}] * NUM_BATCHES
        calibrator = Calibrator(data, BaseClass=BaseClass)

        create_config = CreateConfig(int8=True, calibrator=calibrator)
        with engine_from_network((builder, network), create_config):
            assert calibrator.num_batches == NUM_BATCHES
        self.check_calibrator_cleanup(calibrator)

    def test_host_data_copied_to_device(self):
        with Calibrator(generate_data(1)) as calibrator:
            [ptr] = calibrator.get_batch(names=["x"])
            v = cuda.DeviceView(ptr, shape=(1, 1, 2, 2), dtype=np.float32)
            arr = v.numpy()
            assert arr.shape == (1, 1, 2, 2)
            assert np.all(arr == 1)
        self.check_calibrator_cleanup(calibrator)

    def test_calibrator_data_and_ordering_correct(self):
        def generate_multidata(num_batches):
            for _ in range(num_batches):
                shape = (4, 5)
                yield {
                    "x0": np.zeros(shape, dtype=np.float32),
                    "x1": cuda.DeviceArray(shape=shape, dtype=np.float32).copy_from(np.ones(shape, dtype=np.float32)),
                    "x2": cuda.DeviceArray(shape=shape, dtype=np.float32)
                    .copy_from(np.ones(shape, dtype=np.float32) * 2)
                    .ptr,
                }

        NUM_BATCHES = 2
        with Calibrator(generate_multidata(NUM_BATCHES)) as calibrator:
            for _ in range(NUM_BATCHES):
                ptrs = calibrator.get_batch(names=["x0", "x1", "x2"])
                for index, ptr in enumerate(ptrs):
                    v = cuda.DeviceView(ptr, shape=(4, 5), dtype=np.float32)
                    assert np.all(v.numpy() == index)
        self.check_calibrator_cleanup(calibrator)

    def test_calibrator_generator_data(self, identity_builder_network):
        builder, network = identity_builder_network
        NUM_BATCHES = 2

        calibrator = Calibrator(generate_data(NUM_BATCHES))

        create_config = CreateConfig(int8=True, calibrator=calibrator)
        with engine_from_network((builder, network), create_config):
            assert calibrator.num_batches == NUM_BATCHES
        self.check_calibrator_cleanup(calibrator)

    # We should be able to mix DeviceView with NumPy arrays.
    @pytest.mark.parametrize(
        "mode", ["array", "view", "pointer"]
    )  # We should be able to use DeviceArray in place of DeviceView
    def test_calibrator_device_buffers_multiinput(self, multi_input_builder_network, mode):
        def generate_dev_data(num_batches):
            with cuda.DeviceArray(shape=(1,), dtype=np.float32) as x:
                for _ in range(num_batches):
                    x.copy_from(np.ones((1,), dtype=np.float32))
                    xdata = {"array": x, "view": cuda.DeviceView(x.ptr, x.shape, x.dtype), "pointer": x.ptr}[mode]
                    yield {"X0": xdata, "Y0": np.zeros((1,), dtype=np.float32)}

        builder, network = multi_input_builder_network
        NUM_BATCHES = 2

        calibrator = Calibrator(generate_dev_data(NUM_BATCHES))

        create_config = CreateConfig(int8=True, calibrator=calibrator)
        with engine_from_network((builder, network), create_config):
            assert calibrator.num_batches == NUM_BATCHES
        self.check_calibrator_cleanup(calibrator)

    # We want the calibrator to inter-op with TRT APIs seamlessly
    def test_calibrator_outside_polygraphy(self, identity_builder_network):
        builder, network = identity_builder_network
        NUM_BATCHES = 2

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = Calibrator(generate_data(NUM_BATCHES))
        config.int8_calibrator = calibrator

        if mod.version(trt.__version__) < mod.version("8.0"):
            engine = builder.build_engine(network, config)
        else:
            with trt.Runtime(get_trt_logger()) as runtime:
                engine = runtime.deserialize_cuda_engine(builder.build_serialized_network(network, config))

        with engine:
            assert engine
        self.check_calibrator_cleanup(calibrator)

    def test_calibrator_with_path_name_cache(self, identity_builder_network):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        with util.NamedTemporaryFile() as cache:
            calibrator = Calibrator(data, cache=cache.name)
            create_config = CreateConfig(int8=True, calibrator=calibrator)
            with engine_from_network((builder, network), create_config):
                assert is_file_non_empty(cache.name)
        self.check_calibrator_cleanup(calibrator)

    @pytest.mark.parametrize("mode", ["wb+", "rb", "wb"])
    def test_calibrator_with_file_object_cache(self, identity_builder_network, mode):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        with util.NamedTemporaryFile(mode=mode) as cache:
            calibrator = Calibrator(data, cache=cache)
            create_config = CreateConfig(int8=True, calibrator=calibrator)
            with engine_from_network((builder, network), create_config):
                if mode != "rb":
                    assert is_file_non_empty(cache.name)
        self.check_calibrator_cleanup(calibrator)

    # read_calibration_cache should work even if an explicit cache is not provided
    # This way, it is possible to calibrate quickly when calibrating multiple times.
    def test_calibrator_caches_without_explicit_cache(self, identity_builder_network):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        calibrator = Calibrator(data)
        # First, populate the cache
        create_config = CreateConfig(int8=True, calibrator=calibrator)
        with engine_from_network((builder, network), create_config):
            pass

        # Check that the internal cache is populated
        assert calibrator.read_calibration_cache()
        self.check_calibrator_cleanup(calibrator)

    def test_calibrator_rechecks_cache_on_reset(self, identity_builder_network):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        with util.NamedTemporaryFile(mode="wb+") as cache:
            calibrator = Calibrator(data, cache=cache.name)
            # First, populate the cache
            create_config = CreateConfig(int8=True, calibrator=calibrator)
            with engine_from_network((builder, network), create_config):
                pass

            # Ensure that now the calibrator will read from the cache when reset
            calibrator.reset()
            assert calibrator.cache_contents is None
            assert len(calibrator.read_calibration_cache()) == get_file_size(cache.name)

        self.check_calibrator_cleanup(calibrator)

    @pytest.mark.parametrize(
        "names",
        [
            (["fake-input", "x"]),
            (["fake-input"]),
        ],
    )
    def test_calibrator_invalid_input_fails(self, identity_builder_network, names):
        builder, network = identity_builder_network

        data = [{name: np.ones((1, 1, 2, 2), dtype=np.float32) for name in names}]
        calibrator = Calibrator(data)

        create_config = CreateConfig(int8=True, calibrator=calibrator)

        with pytest.raises(PolygraphyException):
            with engine_from_network((builder, network), create_config):
                pass
        self.check_calibrator_cleanup(calibrator)

    @pytest.mark.parametrize(
        "expected_meta,meta,should_pass",
        [
            (
                TensorMetadata().add(name="input", dtype=np.float32, shape=(1, 3, 28, 28)),
                TensorMetadata().add(name="input", dtype=np.float32, shape=(1, 3, 28, 28)),
                True,
            ),
            (
                TensorMetadata().add(name="input", dtype=np.float32, shape=(-1, None, 28, 28)),
                TensorMetadata().add(name="input", dtype=np.float32, shape=(1, 3, 28, 28)),
                True,
            ),
            # Wrong data type
            (
                TensorMetadata().add(name="input", dtype=np.float32, shape=(1, 3, 28, 28)),
                TensorMetadata().add(name="input", dtype=np.float64, shape=(1, 3, 28, 28)),
                False,
            ),
            # Wrong shape
            (
                TensorMetadata().add(name="input", dtype=np.float32, shape=(1, 3, 28, 28)),
                TensorMetadata().add(name="input", dtype=np.float32, shape=(1, 2, 28, 28)),
                False,
            ),
        ],
    )
    def test_calibrator_checks_input_metadata(self, expected_meta, meta, should_pass):
        data = [{name: np.ones(shape=shape, dtype=dtype) for name, (dtype, shape) in meta.items()}]
        calibrator = Calibrator(data)
        calibrator.set_input_metadata(expected_meta)

        with calibrator:
            assert (calibrator.get_batch(list(expected_meta.keys())) is not None) == should_pass
        self.check_calibrator_cleanup(calibrator)

    def test_calibrator_dynamic_shapes(self, dynamic_identity_builder_network):
        builder, network = dynamic_identity_builder_network

        SHAPES = [(1, 2, 1, 1), (1, 2, 3, 3)]

        def generate_dynamic_shaped_data():
            for shape in SHAPES:
                yield {"X": np.ones(shape=shape, dtype=np.float32)}

        calibrator = Calibrator(generate_dynamic_shaped_data())

        create_config = CreateConfig(
            int8=True,
            calibrator=calibrator,
            profiles=[Profile().add(name="X", min=(1, 2, 1, 1), opt=(1, 2, 2, 2), max=(1, 2, 4, 4))],
        )
        with engine_from_network((builder, network), create_config) as engine:
            assert calibrator.num_batches == 2
            assert engine
        self.check_calibrator_cleanup(calibrator)
