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
This script builds and runs a TensorRT engine with FP16 precision enabled
starting from an ONNX identity model.
"""
import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner


def main():
    # We can compose multiple lazy loaders together to get the desired conversion.
    # In this case, we want ONNX -> TensorRT Network -> TensorRT engine (w/ fp16).
    #
    # NOTE: `build_engine` is a *callable* that returns an engine, not the engine itself.
    #   To get the engine directly, you can use the immediately evaluated functional API.
    #   See examples/api/06_immediate_eval_api for details.
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("identity.onnx"), config=CreateConfig(fp16=True)
    )  # Note that config is an optional argument.

    # To reuse the engine elsewhere, we can serialize and save it to a file.
    # The `SaveEngine` lazy loader will return the TensorRT engine when called,
    # which allows us to chain it together with other loaders.
    build_engine = SaveEngine(build_engine, path="identity.engine")

    # Once our loader is ready, inference is simply a matter of constructing a runner,
    # activating it with a context manager (i.e. `with TrtRunner(...)`) and calling `infer()`.
    #
    # NOTE: You can use the activate() function instead of a context manager, but you will need to make sure to
    # deactivate() to avoid a memory leak. For that reason, a context manager is the safer option.
    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()
