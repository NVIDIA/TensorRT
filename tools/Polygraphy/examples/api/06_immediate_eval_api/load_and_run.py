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
This script uses Polygraphy's immediately evaluated functional APIs
to load the TensorRT engine built by `build_and_run.py` and run inference.
"""
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import TrtRunner, engine_from_bytes


def main():
    engine = engine_from_bytes(bytes_from_path("identity.engine"))

    # NOTE: In TensorRT 8.0 and newer, we do *not* need to use a context manager to free `engine`.
    with engine, TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["output"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()
