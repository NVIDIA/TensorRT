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
import sys
import time
import argparse
import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda, runtime as cudart
from PIL import Image
from pathlib import Path
import threading
from visualize import visualize_detections

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common
from common import cuda_call


class AllocatorState:
    """
    Represents the state of an allocator for a tensor.
    """

    def __init__(self, ptr, size, dim=None):
        """
        :param ptr: The pointer to the allocated device memory.
        :param size: The size of the allocated device memory.
        :param dim: The dimensions of the tensor.
        """
        self.ptr = ptr
        self.size = size
        self.dim = dim
        self.lock = threading.Lock()

    def update(self, ptr=None, size=None, dims=None):
        """
        Updates the state of the allocator.

        :param ptr: The new pointer to the allocated device memory. If None, the current pointer is not changed.
        :param size: The new size of the allocated device memory. If None, the current size is not changed.
        :param dims: The new dimensions of the tensor. If None, the current dimensions are not changed.
        """
        with self.lock:
            if ptr is not None:
                self.ptr = ptr
            if size is not None:
                self.size = size
            if dims is not None:
                self.dims = dims


class MyOutputAllocator(trt.IOutputAllocator):
    """
    Custom output allocator class.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If True, enables verbose logging.
        """
        trt.IOutputAllocator.__init__(self)

        self.lock = threading.Lock()
        self.states = {}
        self.verbose = verbose

    def reallocate_output_async(self, tensor_name, current_memory, size, alignment, stream):
        """
        Reallocates output memory for the given tensor.

        :param tensor_name: The name of the tensor.
        :param current_memory: The current device memory pointer.
        :param size: The new size of the device memory block.
        :param alignment: The alignment of the device memory block.
        :param stream: The CUDA stream.
        :return: The new memory pointer.
        """
        size = max(size, 1)
        ptr = current_memory
        with self.lock:
            if tensor_name not in self.states or size > self.states[tensor_name].size:
                ptr = cuda_call(cudart.cudaMalloc(size))
                if tensor_name in self.states:
                    cuda_call(cudart.cudaFree(self.states[tensor_name].ptr))
                    self.states[tensor_name].update(ptr=ptr, size=size)
                else:
                    self.states[tensor_name] = AllocatorState(ptr=ptr, size=size)
                if self.verbose:
                    print(f"Reallocated {size} bytes for tensor '{tensor_name}' to {ptr}")
        return ptr

    def notify_shape(self, tensor_name, dims):
        """
        Notifies the allocator of a change in the shape of the tensor.

        :param tensor_name: The name of the tensor.
        :param dims: The new dimensions of the tensor.
        """
        with self.lock:
            assert tensor_name in self.states, f'Tensor "{tensor_name}" is not in states.'
            self.states[tensor_name].update(dims=dims)
            if self.verbose:
                print(f"Updated shape for tensor '{tensor_name}': {dims}")

    def __del__(self):
        try:
            with self.lock:
                for tensor_name, item in self.states.items():
                    if item.ptr is not None:
                        cuda_call(cudart.cudaFree(item.ptr))
                        if self.verbose:
                            print(f"Freed memory for tensor '{tensor_name}'")
                self.states.clear()
        except Exception:
            # Silently handle cleanup failures to prevent exceptions during object deletion
            pass


class PoolAllocator(trt.IGpuAsyncAllocator):
    """
    A custom GPU async allocator class that manages memory allocation and deallocation.

    It utilizes the CUDA memory pool API to optimize memory allocation and minimize fragmentation.
    """

    def __init__(self):
        """
        Initializes the PoolAllocator instance.

        Creates a CUDA memory pool with the specified properties and sets the release threshold to the maximum possible value.
        """
        trt.IGpuAsyncAllocator.__init__(self)

        pool_props = cudart.cudaMemPoolProps()
        pool_props.allocType = cudart.cudaMemAllocationType.cudaMemAllocationTypePinned
        pool_props.handleTypes = cudart.cudaMemAllocationHandleType.cudaMemHandleTypeNone
        pool_props.location.type = cudart.cudaMemLocationType.cudaMemLocationTypeDevice
        pool_props.location.id = 0

        self.pool = cuda_call(cudart.cudaMemPoolCreate(pool_props))

        max_threshold = np.uint64(np.iinfo(np.uint64).max)
        cuda_call(
            cudart.cudaMemPoolSetAttribute(
                self.pool, cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold, cuda.cuuint64_t(max_threshold)
            )
        )

    def allocate_async(self, size: int, alignment: int, flags: int, stream: cudart.cudaStream_t):
        """
        Allocates memory asynchronously from the CUDA memory pool.

        :param size: The size of the memory block to allocate.
        :param alignment: The alignment of the memory block.
        :param flags: The flags for the allocation.
        :param stream: The CUDA stream for the allocation.
        :return: The pointer to the allocated device memory.
        """
        ptr = cuda_call(cudart.cudaMallocFromPoolAsync(size, self.pool, stream))
        return ptr

    def deallocate_async(self, memory, stream: cudart.cudaStream_t):
        """
        Deallocates memory asynchronously.

        :param memory: The pointer to the memory to deallocate.
        :param stream: The CUDA stream for the deallocation.
        :return: True if the deallocation was successful.
        """
        cuda_call(cudart.cudaFreeAsync(memory, stream))
        return True

    def __del__(self):
        try:
            if self.pool:
                cuda_call(cudart.cudaMemPoolDestroy(self.pool))
        except Exception:
            # Silently handle cleanup failures to prevent exceptions during object deletion
            pass


class TensorRTInfer:
    """
    Implements inference for the FasterRCNN TensorRT engine.
    """

    def __init__(self, engine_path, use_custom_gpu_allocator=False, verbose=False):
        """
        Initializes the TensorRTInfer instance.

        :param engine_path: The path to the serialized engine to load from disk.
        :param use_custom_gpu_allocator: If True, uses a custom GPU allocator.
        :param verbose: If True, enables verbose logging.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        if verbose:
            self.logger.min_severity = trt.Logger.VERBOSE
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            if use_custom_gpu_allocator:
                self.my_pool_allocator = PoolAllocator()
                runtime.gpu_allocator = self.my_pool_allocator
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        self.my_output_allocator = MyOutputAllocator(verbose=True)
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # trt.nptype returns a python 'type'. For here we want a numpy 'dtype' object
            # instead to get more info about the dtype (dtype.itemsize in this case)
            dtype = np.dtype(dtype)
            shape = self.engine.get_tensor_shape(name)

            # Use the max shape in the profile for dynamic shaped inputs
            if is_input and any(value for value in shape if value < 0):
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                shape = profile_shape[2]

            if is_input:
                nbytes = np.prod(shape) * dtype.itemsize
                allocation = cuda_call(cudart.cudaMalloc(nbytes))
            else:
                self.context.set_output_allocator(name, self.my_output_allocator)
                allocation = cuda_call(
                    cudart.cudaMalloc(128 * dtype.itemsize)
                )  # Random number. More will be allocated using our custom allocator

            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print(
                f"{'Input' if is_input else 'Output'} '{binding['name']}' with shape {binding['shape']} and dtype {binding['dtype']}"
            )

        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs

    def preprocess_image(self, image):
        """
        Preprocesses an image for inference. See also
        https://github.com/onnx/models/tree/refs/heads/main/validated/vision/object_detection_segmentation/faster-rcnn#preprocessing-steps

        :param image: The image to preprocess.
        :return: The preprocessed image as a numpy array.
        """
        ratio = 800.0 / min(image.size[0], image.size[1])
        image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

        # RGB -> BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype("float32")

        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])

        # Normalize
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # Pad to be divisible of 32
        padded_h = int(np.ceil(image.shape[1] / 32) * 32)
        padded_w = int(np.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, : image.shape[1], : image.shape[2]] = image
        image = padded_image

        return image

    def infer(self, arr):
        """
        Execute inference on an image.

        :param arr: A numpy array for the input image values.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        common.memcpy_host_to_device(self.inputs[0]["allocation"], arr)
        self.context.execute_v2(self.allocations)

        # copy outputs to host
        return_outputs = []
        for output in self.outputs:
            final_shape = self.my_output_allocator.states[output["name"]].dims
            host_arr = np.random.random(final_shape).astype(output["dtype"])
            device_ptr = self.my_output_allocator.states[output["name"]].ptr

            nbytes = np.prod(final_shape) * output["dtype"].itemsize
            common.memcpy_device_to_host(host_arr, device_ptr)

            return_outputs.append(host_arr)

        return return_outputs

    def process(self, arr):
        """
        Execute inference on an image. The image should already be preprocessed. Memory
        copying to and from the GPU device will be performed here.

        :param arr: A numpy array holding the image values.
        :return: A list of detected object with box, score, class included.
        """
        preprocess_arr = self.preprocess_image(arr.copy())
        self.context.set_input_shape("image", preprocess_arr.shape)

        # Run inference
        outputs = self.infer(preprocess_arr)

        # Post-process the results
        scale = 800.0 / min(arr.size[0], arr.size[1])

        boxes = outputs[0]
        labels = outputs[1]
        scores = outputs[2]
        num = len(labels)

        detections = []
        for i in range(num):
            if scores[i] > 0.9:
                detections.append(
                    {
                        "xmin": boxes[i][0] / scale,
                        "ymin": boxes[i][1] / scale,
                        "xmax": boxes[i][2] / scale,
                        "ymax": boxes[i][3] / scale,
                        "score": scores[i],
                        "class": labels[i] - 1,
                    }
                )
        return detections


def main(args):
    if args.output:
        args.output.resolve().mkdir(exist_ok=True, parents=True)

    labels = []
    if args.labels:
        with open(args.labels) as f:
            for label in f:
                labels.append(label.strip())

    trt_infer = TensorRTInfer(args.engine, args.use_custom_gpu_allocator, args.verbose)
    if args.input:
        print(f"\nInferring data in {args.input}")
        image_paths = []
        if args.input.is_dir():
            for p in args.input.iterdir():
                image_paths.append(p)
        else:
            image_paths.append(args.input)

        for image_path in image_paths:
            image = Image.open(image_path)
            detections = trt_infer.process(image)
            if args.output:
                # Image Visualizations
                output_path = args.output / f"{image_path.stem}.png"
                visualize_detections(image_path, output_path, detections, labels)

                # Text Results
                output_results = ""
                for d in detections:
                    line = [
                        d["xmin"],
                        d["ymin"],
                        d["xmax"],
                        d["ymax"],
                        d["score"],
                    ]
                    output_results += "\t".join([str(f) for f in line]) + "\n"
                with open(args.output / f"{image_path.stem}.txt", "w") as f:
                    f.write(output_results)
    else:
        print("No input provided, running in benchmark mode")
        shape, dtype = trt_infer.input_spec()
        batch = 255 * np.random.rand(*shape).astype(dtype)
        trt_infer.context.set_input_shape("image", (batch.shape))
        iterations = 200
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            trt_infer.infer(batch)
            times.append(time.time() - start)
            print(f"Iteration {i+1} / {iterations}", end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print(f"Average Latency: {1000 * np.average(times):.3f} ms")

    print("\nFinished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        required=True,
        help="The serialized TensorRT engine",
    )
    parser.add_argument("-i", "--input", default=None, type=Path, help="Path to the image or directory to process")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Directory where to save the visualization results",
    )
    parser.add_argument(
        "-l",
        "--labels",
        default="./labels_coco_80.txt",
        help="File to use for reading the class labels from, default: ./labels_coco_80.txt",
    )
    parser.add_argument(
        "-c",
        "--use_custom_gpu_allocator",
        action="store_true",
        default=False,
        help="Use a custom gpu allocator with CUDA memory pools for better performance",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Set to verbose logging")
    args = parser.parse_args()
    main(args)
