#!/usr/bin/env python3
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

"""
This script demonstrates how to use the Calibrator API provided by Polygraphy
to calibrate a TensorRT engine to run in INT8 precision.
"""
from polygraphy.backend.trt import NetworkFromOnnxPath, CreateConfig, EngineFromNetwork, Calibrator, TrtRunner
from polygraphy.logger import G_LOGGER

import numpy as np
import os


MODEL = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "models", "identity.onnx")
INPUT_SHAPE = (1, 1, 2, 2)

# The data loader argument to Calibrator can be any iterable or generator that yields `feed_dict`s.
# A feed_dict is just a mapping of input names to corresponding inputs (as NumPy arrays).
# Calibration will continue until our data loader runs out of data (4 batches in this example).
def calib_data():
    for _ in range(4):
        yield {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)} # Totally real data

# We can provide a path or file-like object if we want to cache calibration data.
# This lets us avoid running calibration the next time we build the engine.
calibrator = Calibrator(data_loader=calib_data(), cache="identity-calib.cache")
build_engine = EngineFromNetwork(NetworkFromOnnxPath(MODEL), config=CreateConfig(int8=True, calibrator=calibrator))

# When we activate our runner, it will calibrate and build the engine. If we want to
# see the logging output from TensorRT, we can temporarily increase logging verbosity:
with G_LOGGER.verbosity(G_LOGGER.VERBOSE):
    with TrtRunner(build_engine) as runner:
        feed_dict = {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}
        outputs = runner.infer(feed_dict=feed_dict)
        assert np.all(outputs["y"] == feed_dict["x"])
