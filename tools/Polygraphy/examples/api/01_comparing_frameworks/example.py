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
This script runs an identity model with ONNX-Runtime and TensorRT,
then compares outputs.
"""
from polygraphy.backend.trt import NetworkFromOnnxBytes, EngineFromNetwork, TrtRunner
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnxBytes
from polygraphy.backend.common import BytesFromPath
from polygraphy.comparator import Comparator

import os

# Create loaders for both ONNX Runtime and TensorRT
MODEL = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "models", "identity.onnx")

load_serialized_onnx = BytesFromPath(MODEL)
build_onnxrt_session = SessionFromOnnxBytes(load_serialized_onnx)
build_engine = EngineFromNetwork(NetworkFromOnnxBytes(load_serialized_onnx))

# Create runners
runners = [
    TrtRunner(build_engine),
    OnnxrtRunner(build_onnxrt_session),
]

# Finally, run and compare the results.
run_results = Comparator.run(runners)
assert bool(Comparator.compare_accuracy(run_results))
