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


import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# If you face the following issue:
#  "pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
#  Add "import pycuda.autoinit", this is needed to initialize cuda!
import pycuda.autoinit
import tensorflow as tf
from examples.data.data_loader import load_data_tfrecord_tf, load_image_np, _SUPPORTED_MODEL_NAMES

TRT_DYNAMIC_DIM = -1


class HostDeviceMem(object):
    """Simple helper data class to store Host and Device memory."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int) -> [list, list, list]:
    """
    Function to allocate buffers and bindings for TensorRT inference.

    Args:
        engine (trt.ICudaEngine):
        batch_size (int): batch size to be used during inference.

    Returns:
        inputs (List): list of input buffers.
        outputs (List): list of output buffers.
        dbindings (List): list of device bindings.
    """
    inputs = []
    outputs = []
    dbindings = []

    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        if binding_shape[0] == TRT_DYNAMIC_DIM:  # dynamic shape
            size = batch_size * abs(trt.volume(binding_shape))
        else:
            size = abs(trt.volume(binding_shape))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        dbindings.append(int(device_mem))

        # Append to the appropriate list (input/output)
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings


def infer(
    engine_path: str,
    val_batches,
    batch_size: int = 8,
    top_k_value: int = 1,
) -> None:
    """
    Performs inference in TensorRT engine.

    Args:
        engine_path (str): path to the TensorRT engine.
        val_batches (tf.data.Dataset): validation dataset (batches).
        batch_size (int): batch size used for inference and dataset batch splitting.
        top_k_value (int): value of `K` for the top K predictions used in the accuracy calculation.

    Raises:
        RuntimeError: raised when loading images in the host fails.
    """

    def override_shape(shape: tuple) -> tuple:
        """Overrides batch dimension if dynamic."""
        if TRT_DYNAMIC_DIM in shape:
            shape = tuple(
                [batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in shape]
            )
        return shape

    # Open engine as runtime
    with open(engine_path, "rb") as f, trt.Runtime(
        trt.Logger(trt.Logger.ERROR)
    ) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings = allocate_buffers(engine, batch_size)

        # Initiate test_accuracy
        test_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=top_k_value, name="top_k_accuracy", dtype=tf.float32
        )
        test_accuracy.reset_states()

        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Resolves dynamic shapes in the context
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                binding_shape = engine.get_binding_shape(binding_idx)
                if engine.binding_is_input(binding_idx):
                    binding_shape = override_shape(binding_shape)
                    context.set_binding_shape(binding_idx, binding_shape)

            if isinstance(val_batches, tf.Tensor):
                # Load images in Host (flatten and copy to page-locked buffer in Host)
                data = val_batches.numpy().astype(np.float32).ravel()
                pagelocked_buffer = inputs[0].host
                np.copyto(pagelocked_buffer, data)
                inp = inputs[0]
                # Transfer input data from Host to Device (GPU)
                cuda.memcpy_htod(inp.device, inp.host)
                # Run inference
                context.execute_v2(dbindings)
                # Transfer predictions back to Host from GPU
                out = outputs[0]
                cuda.memcpy_dtoh(out.host, out.device)

                softmax_output = np.array(out.host)
                top1_idx = np.argmax(softmax_output)
                output_confidence = softmax_output[top1_idx]
                print("Top-1 Index of the image : {} Confidence: {}".format(top1_idx, output_confidence))

            elif isinstance(val_batches, tf.data.Dataset):
                # Loop over number of steps to evaluate entire validation dataset
                for step, example in enumerate(val_batches):
                    images, labels = example
                    if step % 100 == 0 and step != 0:
                        print(
                            "Evaluating batch {}: {:.4f}".format(
                                step, test_accuracy.result()
                            )
                        )
                    try:
                        # Load images in Host (flatten and copy to page-locked buffer in Host)
                        data = images.numpy().astype(np.float32).ravel()
                        pagelocked_buffer = inputs[0].host
                        np.copyto(pagelocked_buffer, data)
                    except RuntimeError:
                        raise RuntimeError(
                            "Failed to load images in Host at step {}".format(step)
                        )

                    inp = inputs[0]
                    # Transfer input data from Host to Device (GPU)
                    cuda.memcpy_htod(inp.device, inp.host)
                    # Run inference
                    context.execute_v2(dbindings)
                    # Transfer predictions back to Host from GPU
                    out = outputs[0]
                    cuda.memcpy_dtoh(out.host, out.device)

                    # Split 1-D output of length N*labels into 2-D array of (N, labels)
                    batch_outs = np.array(np.split(np.array(out.host), batch_size))
                    # Update test accuracy
                    test_accuracy.update_state(labels, batch_outs)

                # Print final accuracy and save to log file
                print("\n======================================\n")
                result_str = "Top-{} accuracy: {:.4f}\n".format(
                    top_k_value, test_accuracy.result()
                )
                print(result_str)
                # Save logs to file
                results_dir = "/".join(args.engine.split("/")[:-1])
                with open(os.path.join(results_dir, args.log_file), "w") as log_file:
                    log_file.write(result_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on TensorRT engines for Imagenet-based Classification models."
    )
    parser.add_argument(
        "-e", "--engine", type=str, required=True, help="Path to TensorRT engine"
    )
    parser.add_argument(
        "--image", type=str, help="Path to an image to perform single image inference"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="resnet_v1",
        help="Name of the model, needed to choose the appropriate input pre-processing."
        "Options include {}".format(_SUPPORTED_MODEL_NAMES),
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        default="/media/Data/ImageNet/train-val-tfrecord",
        type=str,
        help="Path to directory of input images in tfrecord format (val data).",
    )
    parser.add_argument(
        "-k",
        "--top_k_value",
        default=1,
        type=int,
        help="Value of `K` for the top-K predictions used in the accuracy calculation.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="Number of inputs to send in parallel (up to max batch size of engine).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="engine_accuracy.log",
        help="Filename to save logs.",
    )
    args = parser.parse_args()

    if args.model_name not in _SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "Invalid model name ",
            args.model_name,
            " provided. Please select among {}".format(_SUPPORTED_MODEL_NAMES),
        )

    # Load the test data and pre-process input
    val_batches = None
    if args.image:
        val_batches = load_image_np(args.image, args.model_name)
    else:
        data_batches = load_data_tfrecord_tf(
            data_dir=args.data_dir, batch_size=args.batch_size, model_name=args.model_name
        )
        val_batches = data_batches["validation"]

    # Perform inference
    infer(
        args.engine,
        val_batches,
        batch_size=args.batch_size,
        top_k_value=args.top_k_value,
    )
