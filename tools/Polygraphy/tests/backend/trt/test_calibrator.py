#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import os
import tempfile

import numpy as np
import pytest
import tensorrt as trt
from polygraphy.backend.trt import (Calibrator, CreateConfig,
                                    EngineFromNetwork, NetworkFromOnnxBytes)
from polygraphy.backend.trt.loader import EngineFromBytes
from polygraphy.common import func
from polygraphy.util import misc
from tests.common import check_file_non_empty
from tests.models.meta import ONNX_MODELS


@pytest.fixture(scope="session")
def identity_builder_network():
    builder, network, parser = func.invoke(NetworkFromOnnxBytes(ONNX_MODELS["identity"].loader))
    with builder, network, parser:
        yield builder, network


class TestCalibrator(object):
    def test_calibrator_iterable_data(self, identity_builder_network):
        builder, network = identity_builder_network
        NUM_BATCHES = 2

        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}] * NUM_BATCHES
        calibrator = Calibrator(data)

        create_config = CreateConfig(int8=True, calibrator=calibrator)
        loader = EngineFromNetwork((builder, network), create_config)
        with loader():
            assert calibrator.num_batches == NUM_BATCHES


    def test_calibrator_generator_data(self, identity_builder_network):
        builder, network = identity_builder_network
        NUM_BATCHES = 2

        def generate_data():
            for item in [np.ones((1, 1, 2, 2), dtype=np.float32)] * NUM_BATCHES:
                yield {"x": item}
        calibrator = Calibrator(generate_data())

        create_config = CreateConfig(int8=True, calibrator=calibrator)
        loader = EngineFromNetwork((builder, network), create_config)
        with loader():
            assert calibrator.num_batches == NUM_BATCHES


    # We want the calibrator to inter-op with TRT APIs seamlessly
    def test_calibrator_outside_polygraphy(self, identity_builder_network):
        builder, network = identity_builder_network
        NUM_BATCHES = 2

        def generate_data():
            for item in [np.ones((1, 1, 2, 2), dtype=np.float32)] * NUM_BATCHES:
                yield {"x": item}
        calibrator = Calibrator(generate_data())

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

        if misc.version(trt.__version__) < misc.version("7.3"):
            engine = builder.build_engine(network, config)
        else:
            engine = func.invoke(EngineFromBytes(builder.build_serialized_network(network, config)))

        with engine:
            assert engine


    def test_calibrator_with_path_name_cache(self, identity_builder_network):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        with tempfile.NamedTemporaryFile() as cache:
            create_config = CreateConfig(int8=True, calibrator=Calibrator(data, cache=cache.name))
            with func.invoke(EngineFromNetwork((builder, network), create_config)):
                check_file_non_empty(cache.name)


    @pytest.mark.parametrize("mode", ["wb+", "rb", "wb"])
    def test_calibrator_with_file_object_cache(self, identity_builder_network, mode):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        with tempfile.NamedTemporaryFile(mode=mode) as cache:
            create_config = CreateConfig(int8=True, calibrator=Calibrator(data, cache=cache))
            with func.invoke(EngineFromNetwork((builder, network), create_config)):
                if mode != "rb":
                    check_file_non_empty(cache.name)


    # read_calibration_cache should work even if an explicit cache is not provided
    # This way, it is possible to calibrate quickly when calibrating multiple times.
    def test_calibrator_caches_without_explicit_cache(self, identity_builder_network):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        calibrator = Calibrator(data)
        # First, populate the cache
        create_config = CreateConfig(int8=True, calibrator=calibrator)
        with func.invoke(EngineFromNetwork((builder, network), create_config)):
            pass

        # Check that the internal cache is populated
        assert calibrator.read_calibration_cache()


    def test_calibrator_rechecks_cache_on_reset(self, identity_builder_network):
        builder, network = identity_builder_network
        data = [{"x": np.ones((1, 1, 2, 2), dtype=np.float32)}]

        with tempfile.NamedTemporaryFile(mode="wb+") as cache:
            calibrator = Calibrator(data, cache=cache.name)
            # First, populate the cache
            create_config = CreateConfig(int8=True, calibrator=calibrator)
            with func.invoke(EngineFromNetwork((builder, network), create_config)):
                pass

            # Ensure that now the calibrator will read from the cache when reset
            calibrator.reset()
            assert not calibrator.has_cached_scales
            assert len(calibrator.read_calibration_cache()) == os.stat(cache.name).st_size
