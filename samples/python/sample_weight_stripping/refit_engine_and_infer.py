#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import random
import sys
import time
import datetime
import numpy as np

import tensorrt as trt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

def load_stripped_engine_and_refit(input_file, onnx_model_path):
    runtime = trt.Runtime(TRT_LOGGER)

    with open(input_file, 'rb') as engine_file:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
        refitter = trt.Refitter(engine, TRT_LOGGER)
        parser_refitter = trt.OnnxParserRefitter(refitter, TRT_LOGGER)
        assert parser_refitter.refit_from_file(onnx_model_path)
        assert refitter.refit_cuda_engine()

        return engine

def load_normal_engine(input_file):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(input_file, 'rb') as engine_file:
        engine = runtime.deserialize_cuda_engine(engine_file.read())

        return engine


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = (
            np.asarray(image.resize((w, h), Image.LANCZOS))
            .transpose([2, 0, 1])
            .astype(trt.nptype(ModelData.DTYPE))
            .ravel()
        )
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def main(args):
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

    # Load a TensorRT engine.
    engine = load_normal_engine(args.normal_engine)
    refitted_engine = load_stripped_engine_and_refit(args.stripped_engine, onnx_model_file)

    # Allocate buffers
    inputs, outputs, bindings = common.allocate_buffers(engine)
    inputs_1, outputs_1, bindings_1 = common.allocate_buffers(refitted_engine)

    # Contexts are used to perform inference.
    context = engine.create_execution_context()
    context_1 = refitted_engine.create_execution_context()

    # Load a normalized test case into the host input page-locked buffer.
    test_image = random.choice(test_images)
    test_case = load_normalized_test_case(test_image, inputs[0].host)
    test_case_1 = load_normalized_test_case(test_image, inputs_1[0].host)

    # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
    # probability that the image corresponds to that label
    
    # Use context manager for proper stream lifecycle management - Normal engine
    with common.CudaStreamContext() as stream:
        start_time = time.time()
        for i in range(100): # count time for 100 times of inference
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        total_time = time.time() - start_time
        print("Normal engine inference time on 100 cases: {:.4f} seconds".format(total_time))

    # Use context manager for proper stream lifecycle management - Refitted engine
    with common.CudaStreamContext() as stream_1:
        start_time = time.time()
        for i in range(100):
            trt_outputs_refitted = common.do_inference(context_1, engine=refitted_engine, bindings=bindings_1, inputs=inputs_1, outputs=outputs_1, stream=stream_1)
        total_time = time.time() - start_time
        print("Refitted stripped engine inference time on 100 cases: {:.4f} seconds".format(total_time))

    # We use the highest probability as our prediction. Its index corresponds to the predicted label.
    pred = labels[np.argmax(trt_outputs[0])]
    if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
        print("Normal engine correctly recognized " + test_case + " as " + pred)
    else:
        print("Normal engine incorrectly recognized " + test_case + " as " + pred)
        exit(1)

    pred_refitted = labels[np.argmax(trt_outputs_refitted[0])]
    if "_".join(pred_refitted.split()) in os.path.splitext(os.path.basename(test_case_1))[0]:
        print("Refitted stripped engine correctly recognized " + test_case + " as " + pred_refitted)
    else:
        print("Refitted stripped engine incorrectly recognized " + test_case + " as " + pred_refitted)
        exit(1)

    return trt_outputs, trt_outputs_refitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stripped_engine", default='stripped_engine.trt', type=str,
                        help="The stripped engine file to load.")
    parser.add_argument("--normal_engine", default='normal_engine.trt', type=str,
                        help="The normal engine file to load.")

    args, _ = parser.parse_known_args()
    if not os.path.exists(args.stripped_engine):
        parser.print_help()
        print(f"--stripped_engine {args.stripped_engine} does not exist.")
        sys.exit(1)
    if not os.path.exists(args.normal_engine):
        parser.print_help()
        print(f"--normal_engine {args.normal_engine} does not exist.")
        sys.exit(1)

    trt_outputs, trt_outputs_refitted = main(args)
    print("The MSE of the final layer output is", np.square(np.subtract(trt_outputs, trt_outputs_refitted)).mean())
