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
This script runs an identity model with ONNX-Runtime and TensorRT,
then compares outputs.
"""
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc


def main():
    # The OnnxrtRunner requires an ONNX-RT session.
    # We can use the SessionFromOnnx lazy loader to construct one easily:
    build_onnxrt_session = SessionFromOnnx("identity.onnx")

    # The TrtRunner requires a TensorRT engine.
    # To create one from the ONNX model, we can chain a couple lazy loaders together:
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    runners = [
        TrtRunner(build_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    # `Comparator.run()` will run each runner separately using synthetic input data and
    #   return a `RunResults` instance. See `polygraphy/comparator/struct.py` for details.
    #
    # TIP: To use custom input data, you can set the `data_loader` parameter in `Comparator.run()``
    #   to a generator or iterable that yields `Dict[str, np.ndarray]`.
    run_results = Comparator.run(runners)

    # `Comparator.compare_accuracy()` checks that outputs match between runners.
    #
    # TIP: The `compare_func` parameter can be used to control how outputs are compared (see API reference for details).
    #   The default comparison function is created by `CompareFunc.simple()`, but we can construct it
    #   explicitly if we want to change the default parameters, such as tolerance.
    assert bool(Comparator.compare_accuracy(run_results, compare_func=CompareFunc.simple(atol=1e-8)))

    # We can use `RunResults.save()` method to save the inference results to a JSON file.
    # This can be useful if you want to generate and compare results separately.
    run_results.save("inference_results.json")


if __name__ == "__main__":
    main()
