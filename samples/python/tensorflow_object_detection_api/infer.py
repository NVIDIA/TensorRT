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
import time
import ctypes
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
from visualize import visualize_detections


class TensorRTInfer:
    """
    Implements inference for the Model TensorRT engine.
    """

    def __init__(self, engine_path, preprocessor, detection_type, iou_threshold):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        self.preprocessor = preprocessor
        self.detection_type = detection_type
        self.iou_threshold = iou_threshold
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
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
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
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
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch, scales=None, nms_threshold=None):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """

        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        # One additional output for segmentation masks
        if len(outputs) == 5:
            masks = outputs[4]
        detections = []
        normalized = (np.max(boxes) < 2.0)
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                # Depending on preprocessor, box scaling will be slightly different.
                if self.preprocessor == "fixed_shape_resizer":
                    scale_x = self.inputs[0]['shape'][1] if normalized else 1.0
                    scale_y = self.inputs[0]['shape'][2] if normalized else 1.0

                    if scales and i < len(scales):
                        scale_x /= scales[i][0]
                        scale_y /= scales[i][1]
                    if nms_threshold and scores[i][n] < nms_threshold:
                        continue
                    # Depending on detection type you need slightly different data.
                    if self.detection_type == 'bbox':
                        mask = None
                    # Segmentation is only supported with Mask R-CNN, which has
                    # fixed_shape_resizer as image_resizer (lookup pipeline.config)
                    elif self.detection_type == 'segmentation':
                        # Select a mask
                        mask = masks[i][n]
                        # Slight scaling, to get binary masks after float32 -> uint8
                        # conversion, if not scaled all pixels are zero.
                        mask = mask > self.iou_threshold
                        # Convert float32 -> uint8.
                        mask = mask.astype(np.uint8)
                elif self.preprocessor == "keep_aspect_ratio_resizer":
                    # No segmentation models with keep_aspect_ratio_resizer
                    mask = None
                    scale = self.inputs[0]['shape'][2] if normalized else 1.0
                    if scales and i < len(scales):
                        scale /= scales[i]
                        scale_y = scale
                        scale_x = scale
                    if nms_threshold and scores[i][n] < nms_threshold:
                        continue
                # Append to detections
                detections[i].append({
                    'ymin': boxes[i][n][0] * scale_y,
                    'xmin': boxes[i][n][1] * scale_x,
                    'ymax': boxes[i][n][2] * scale_y,
                    'xmax': boxes[i][n][3] * scale_x,
                    'score': scores[i][n],
                    'class': int(classes[i][n]),
                    'mask': mask,
                })
        return detections


def main(args):
    output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    labels = []
    if args.labels:
        with open(args.labels) as f:
            for i, label in enumerate(f):
                labels.append(label.strip())

    trt_infer = TensorRTInfer(args.engine, args.preprocessor, args.detection_type, args.iou_threshold)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec(), preprocessor=args.preprocessor)
    for batch, images, scales in batcher.get_batch():
        print("Processing Image {} / {}".format(batcher.image_index, batcher.num_images), end="\r")
        detections = trt_infer.infer(batch, scales, args.nms_threshold)
        for i in range(len(images)):
            basename = os.path.splitext(os.path.basename(images[i]))[0]
            # Image Visualizations
            output_path = os.path.join(output_dir, "{}.png".format(basename))
            visualize_detections(images[i], output_path, detections[i], labels)
            # Text Results
            output_results = ""
            for d in detections[i]:
                line = [d['xmin'], d['ymin'], d['xmax'], d['ymax'], d['score'], d['class']]
                output_results += "\t".join([str(f) for f in line]) + "\n"
            with open(os.path.join(args.output, "{}.txt".format(basename)), "w") as f:
                f.write(output_results)
    print()
    print("Finished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None, help="The serialized TensorRT engine")
    parser.add_argument("-i", "--input", default=None, help="Path to the image or directory to process")
    parser.add_argument("-o", "--output", default=None, help="Directory where to save the visualization results")
    parser.add_argument("-l", "--labels", default="./labels_coco.txt",
                        help="File to use for reading the class labels from, default: ./labels_coco.txt")
    parser.add_argument("-d", "--detection_type", default="bbox", choices=["bbox", "segmentation"],
                        help="Detection type for COCO, either bbox or if you are using Mask R-CNN's instance segmentation - segmentation")
    parser.add_argument("-t", "--nms_threshold", type=float,
                        help="Override the score threshold for the NMS operation, if higher than the threshold in the engine.")
    parser.add_argument("--iou_threshold", default=0.5, type=float,
                        help="Select the IoU threshold for the mask segmentation. Range is 0 to 1. Pixel values more than threshold will become 1, less 0")
    parser.add_argument("--preprocessor", default="fixed_shape_resizer", choices=["fixed_shape_resizer", "keep_aspect_ratio_resizer"],
                        help="Select the image preprocessor to use based on your pipeline.config, either 'fixed_shape_resizer' or 'keep_aspect_ratio_resizer', default: fixed_shape_resizer")
    args = parser.parse_args()
    if not all([args.engine, args.input, args.output, args.preprocessor]):
        parser.print_help()
        print("\nThese arguments are required: --engine --input --output and --preprocessor")
        sys.exit(1)
    main(args)
