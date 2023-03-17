#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
import sys

import numpy as np

import tensorrt as trt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_engine(network, config)


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = (
            np.asarray(image.resize((w, h), Image.ANTIALIAS))
            .transpose([2, 0, 1])
            .astype(trt.nptype(ModelData.DTYPE))
            .ravel()
        )
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image


def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    _, data_files = common.find_sample_data(
        description="Runs a ResNet50 network with a TensorRT inference engine.",
        subfolder="resnet50",
        find_files=[
            "binoculars.jpeg",
            "reflex_camera.jpeg",
            "tabby_tiger_cat.jpg",
            ModelData.MODEL_PATH,
            "class_labels.txt",
        ],
    )
    # Get test images, models and labels.
    test_images = data_files[0:3]
    onnx_model_file, labels_file = data_files[3:]
    labels = open(labels_file, "r").read().split("\n")

    # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_model_file)
    # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()

    # Load a normalized test case into the host input page-locked buffer.
    test_image = random.choice(test_images)
    test_case = load_normalized_test_case(test_image, inputs[0].host)
    # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
    # probability that the image corresponds to that label
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    # We use the highest probability as our prediction. Its index corresponds to the predicted label.
    pred = labels[np.argmax(trt_outputs[0])]
    common.free_buffers(inputs, outputs, stream)
    if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
        print("Correctly recognized " + test_case + " as " + pred)
    else:
        print("Incorrectly recognized " + test_case + " as " + pred)


if __name__ == "__main__":
    main()
