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
import sys
import argparse

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit


from image_batcher import ImageBatcher


class TensorRTInfer:
    """
    Implements inference for the EfficientNet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
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
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch, top=1):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """
        # Prepare the output data
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(output, self.outputs[0]["allocation"])

        # Process the results
        classes = np.argmax(output, axis=1)
        scores = np.max(output, axis=1)
        top = min(top, output.shape[1])
        top_classes = np.flip(np.argsort(output, axis=1), axis=1)[:, 0:top]
        top_scores = np.flip(np.sort(output, axis=1), axis=1)[:, 0:top]

        return classes, scores, [top_classes, top_scores]


def main(args):
    trt_infer = TensorRTInfer(args.engine)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec(), preprocessor=args.preprocessor)
    for batch, images in batcher.get_batch():
        classes, scores, top = trt_infer.infer(batch)
        for i in range(len(images)):
            if args.top == 1:
                print(images[i], classes[i], scores[i], sep=args.separator)
            else:
                line = [images[i]]
                assert args.top <= top[0].shape[1]
                for t in range(args.top):
                    line.append(str(top[0][i][t]))
                for t in range(args.top):
                    line.append(str(top[1][i][t]))
                print(args.separator.join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with")
    parser.add_argument(
        "-i", "--input", help="The input to infer, either a single image path, or a directory of images"
    )
    parser.add_argument(
        "-t", "--top", default=1, type=int, help="The amount of top classes and scores to output per image, default: 1"
    )
    parser.add_argument(
        "-s",
        "--separator",
        default="\t",
        help="Separator to use between columns when printing the results, default: \\t",
    )
    parser.add_argument(
        "-p",
        "--preprocessor",
        default="V2",
        choices=["V1", "V1MS", "V2"],
        help="Select the image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2",
    )
    args = parser.parse_args()
    if not all([args.engine, args.input]):
        parser.print_help()
        print("\nThese arguments are required: --engine and --input")
        sys.exit(1)
    main(args)
