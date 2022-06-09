#!/usr/bin/env python3
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

"""
This script demonstrates how to use the Calibrator API provided by Polygraphy
to calibrate a TensorRT engine to run in INT8 precision.
"""
import numpy as np
from polygraphy.backend.trt import Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.logger import G_LOGGER


# The data loader argument to `Calibrator` can be any iterable or generator that yields `feed_dict`s.
# A `feed_dict` is just a mapping of input names to corresponding inputs.
def calib_data():
    for _ in range(4):
        # TIP: If your calibration data is already on the GPU, you can instead provide GPU pointers
        # (as `int`s) or Polygraphy `DeviceView`s instead of NumPy arrays.
        #
        # For details on `DeviceView`, see `polygraphy/cuda/cuda.py`.
        yield {"x": np.ones(shape=(1, 1, 2, 2), dtype=np.float32)}  # Totally real data


def main():
    # We can provide a path or file-like object if we want to cache calibration data.
    # This lets us avoid running calibration the next time we build the engine.
    #
    # TIP: You can use this calibrator with TensorRT APIs directly (e.g. config.int8_calibrator).
    # You don't have to use it with Polygraphy loaders if you don't want to.
    calibrator = Calibrator(data_loader=calib_data(), cache="identity-calib.cache")

    # We must enable int8 mode in addition to providing the calibrator.
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("identity.onnx"), config=CreateConfig(int8=True, calibrator=calibrator)
    )

    # When we activate our runner, it will calibrate and build the engine. If we want to
    # see the logging output from TensorRT, we can temporarily increase logging verbosity:
    with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
        # Finally, we can test out our int8 TensorRT engine with some dummy input data:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!


if __name__ == "__main__":
    main()
