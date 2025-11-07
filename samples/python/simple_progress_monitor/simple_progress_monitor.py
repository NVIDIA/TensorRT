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

# This sample demonstrates incremental progress reporting while it uses an ONNX ResNet50 Model to create a TensorRT Inference Engine.
import random
import sys

import numpy as np

import tensorrt as trt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common


class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype().
    DTYPE = trt.float32


# This is a simple ASCII-art progress monitor comparable to the C++ version in sample_progress_monitor.
class SimpleProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            if parent_phase is not None:
                nbIndents = 1 + self._active_phases[parent_phase]["nbIndents"]
            else:
                nbIndents = 0
            self._active_phases[phase_name] = {
                "title": phase_name,
                "steps": 0,
                "num_steps": num_steps,
                "nbIndents": nbIndents,
            }
            self._redraw()
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            del self._active_phases[phase_name]
            self._redraw(blank_lines=1)  # Clear the removed phase.
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            self._active_phases[phase_name]["steps"] = step
            self._redraw()
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False

    def _redraw(self, *, blank_lines=0):
        # The Python curses module is not widely available on Windows platforms.
        # Instead, this function uses raw terminal escape sequences. See the sample documentation for references.
        def clear_line():
            print("\x1B[2K", end="")

        def move_to_start_of_line():
            print("\x1B[0G", end="")

        def move_cursor_up(lines):
            print("\x1B[{}A".format(lines), end="")

        def progress_bar(steps, num_steps):
            INNER_WIDTH = 10
            completed_bar_chars = int(INNER_WIDTH * steps / float(num_steps))
            return "[{}{}]".format(
                "=" * completed_bar_chars, "-" * (INNER_WIDTH - completed_bar_chars)
            )

        # Set max_cols to a default of 200 if not run in interactive mode.
        max_cols = os.get_terminal_size().columns if sys.stdout.isatty() else 200

        move_to_start_of_line()
        for phase in self._active_phases.values():
            phase_prefix = "{indent}{bar} {title}".format(
                indent=" " * phase["nbIndents"],
                bar=progress_bar(phase["steps"], phase["num_steps"]),
                title=phase["title"],
            )
            phase_suffix = "{steps}/{num_steps}".format(**phase)
            allowable_prefix_chars = max_cols - len(phase_suffix) - 2
            if allowable_prefix_chars < len(phase_prefix):
                phase_prefix = phase_prefix[0 : allowable_prefix_chars - 3] + "..."
            clear_line()
            print(phase_prefix, phase_suffix)
        for line in range(blank_lines):
            clear_line()
            print()
        move_cursor_up(len(self._active_phases) + blank_lines)
        sys.stdout.flush()


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    if not sys.stdout.isatty():
        print(
            "Warning: This sample should be run from an interactive terminal in order to showcase the progress monitor correctly."
        )
    config.progress_monitor = SimpleProgressMonitor()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array.
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


def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    _, data_files = common.find_sample_data(
        description="Runs a ResNet50 network with a TensorRT inference engine. Displays intermediate build progress.",
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
    # Allocate buffers
    inputs, outputs, bindings = common.allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()

    # Use context manager for proper stream lifecycle management
    with common.CudaStreamContext() as stream:
        # Load a normalized test case into the host input page-locked buffer.
        test_image = random.choice(test_images)
        test_case = load_normalized_test_case(test_image, inputs[0].host)
        # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
        # probability that the image corresponds to that label
        trt_outputs = common.do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        # We use the highest probability as our prediction. Its index corresponds to the predicted label.
        pred = labels[np.argmax(trt_outputs[0])]
        common.free_buffers(inputs, outputs)
    if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
        print("Correctly recognized " + test_case + " as " + pred)
    else:
        print("Incorrectly recognized " + test_case + " as " + pred)


if __name__ == "__main__":
    main()
