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
This script uses the Polygraphy Runner API to validate the outputs
of an identity model using a trivial dataset.
"""
import numpy as np
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner

# Pretend that this is a very large dataset.
REAL_DATASET = [
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
]  # Definitely real data

# For an identity network, the golden output values are the same as the input values.
# Though such a network appears useless at first glance, it can be very useful in some cases (like here!).
EXPECTED_OUTPUTS = REAL_DATASET


def main():
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    with TrtRunner(build_engine) as runner:
        for (data, golden) in zip(REAL_DATASET, EXPECTED_OUTPUTS):
            # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
            #   Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
            outputs = runner.infer(feed_dict={"x": data})

            assert np.array_equal(outputs["y"], golden)

        print("Validation succeeded!")


if __name__ == "__main__":
    main()
