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

# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
from random import randint
from PIL import Image
import numpy as np

import pycuda.driver as cuda

# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME = "input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"


def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.UffParser() as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        plan = builder.build_serialized_network(network, config)
        return runtime.deserialize_cuda_engine(plan)


# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_paths, pagelocked_buffer, case_num=randint(0, 9)):
    [test_case_path] = common.locate_files(
        data_paths,
        [str(case_num) + ".pgm"],
        err_msg="Please follow the README in the mnist data directory (usually in `/usr/src/tensorrt/data/mnist`) to download the MNIST dataset",
    )
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return case_num


def main():
    data_paths, _ = common.find_sample_data(
        description="Runs an MNIST network using a UFF model file", subfolder="mnist"
    )
    model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_path, ModelData.MODEL_FILE)

    with build_engine(model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))


if __name__ == "__main__":
    main()
