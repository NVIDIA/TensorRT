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
import tensorflow as tf

from infer import TensorRTInfer
from image_batcher import ImageBatcher


class TensorFlowInfer:
    """
    Implements TensorFlow inference of a saved model, following the same API as the TensorRTInfer class.
    """

    def __init__(self, saved_model_path):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.model = tf.saved_model.load(saved_model_path)
        self.pred_fn = self.model.signatures["serving_default"]

        # Setup I/O bindings
        self.inputs = []
        fn_inputs = self.pred_fn.structured_input_signature[1]
        for i, input in enumerate(list(fn_inputs.values())):
            self.inputs.append(
                {
                    "index": i,
                    "name": input.name,
                    "dtype": np.dtype(input.dtype.as_numpy_dtype()),
                    "shape": input.shape.as_list(),
                }
            )
        self.outputs = []
        fn_outputs = self.pred_fn.structured_outputs
        for i, output in enumerate(list(fn_outputs.values())):
            self.outputs.append(
                {
                    "index": i,
                    "name": output.name,
                    "dtype": np.dtype(output.dtype.as_numpy_dtype()),
                    "shape": output.shape.as_list(),
                }
            )

    def input_spec(self):
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch, top=1):
        # Process I/O and execute the network
        input = {self.inputs[0]["name"]: tf.convert_to_tensor(batch)}
        output = self.pred_fn(**input)
        output = output[self.outputs[0]["name"]].numpy()

        # Read and process the results
        classes = np.argmax(output, axis=1)
        scores = np.max(output, axis=1)
        top = max(top, output.shape[1])
        top_classes = np.flip(np.argsort(output, axis=1), axis=1)[:, 0:top]
        top_scores = np.flip(np.sort(output, axis=1), axis=1)[:, 0:top]

        return classes, scores, [top_classes, top_scores]


def main(args):
    # Initialize TRT and TF infer objects.
    tf_infer = TensorFlowInfer(args.saved_model)
    trt_infer = TensorRTInfer(args.engine)

    batcher = ImageBatcher(
        args.input, *trt_infer.input_spec(), max_num_images=args.num_images, preprocessor=args.preprocessor
    )

    # Make sure both systems use the same input spec, so we can use the exact same image batches with both
    tf_shape, tf_dtype = tf_infer.input_spec()
    trt_shape, trt_dtype = trt_infer.input_spec()
    if trt_dtype != tf_dtype:
        print("Input datatype does not match")
        print("TRT Engine Input Dtype: {} {}".format(trt_dtype))
        print("TF Saved Model Input Dtype: {} {}".format(tf_dtype))
        print("Please use the same TensorFlow saved model that the TensorRT engine was built with")
        sys.exit(1)

    if (tf_shape[1] and trt_shape[1] != tf_shape[1]) or (tf_shape[2] and trt_shape[2] != tf_shape[2]):
        print("Input shapes do not match")
        print("TRT Engine Input Shape: {} {}".format(trt_shape[1:]))
        print("TF Saved Model Input Shape: {} {}".format(tf_shape[1:]))
        print("Please use the same TensorFlow saved model that the TensorRT engine was built with")
        sys.exit(1)

    match = 0
    error = 0
    for batch, images in batcher.get_batch():
        # Run inference on the same batch with both inference systems
        tf_classes, tf_scores, _ = tf_infer.infer(batch)
        trt_classes, trt_scores, _ = trt_infer.infer(batch)

        # The last batch may not have all image slots filled, so limit the results to only the amount of actual images
        tf_classes = tf_classes[0 : len(images)]
        tf_scores = tf_scores[0 : len(images)]
        trt_classes = trt_classes[0 : len(images)]
        trt_scores = trt_scores[0 : len(images)]

        # Track how many images match on top-1 class id predictions
        match += np.sum(trt_classes == tf_classes)
        # Track the mean square error in confidence score
        error += np.sum((trt_scores - tf_scores) * (trt_scores - tf_scores))

        print(
            "Processing {} / {} images: {:.2f}% match     ".format(
                batcher.image_index, batcher.num_images, (100 * (match / batcher.image_index))
            ),
            end="\r",
        )

    print()
    pc = 100 * (match / batcher.num_images)
    print("Matching Top-1 class predictions for {} out of {} images: {:.2f}%".format(match, batcher.num_images, pc))
    avgerror = np.sqrt(error / batcher.num_images)
    print("RMSE between TensorFlow and TensorRT confidence scores: {:.3f}".format(avgerror))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with")
    parser.add_argument("-m", "--saved_model", help="The TensorFlow saved model path to validate against")
    parser.add_argument(
        "-i", "--input", help="The input to infer, either a single image path, or a directory of images"
    )
    parser.add_argument(
        "-n",
        "--num_images",
        default=5000,
        type=int,
        help="The maximum number of images to use for validation, default: 5000",
    )
    parser.add_argument(
        "-p",
        "--preprocessor",
        default="V2",
        choices=["V1", "V1MS", "V2"],
        help="Select the image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2",
    )
    args = parser.parse_args()
    if not all([args.engine, args.saved_model, args.input]):
        parser.print_help()
        sys.exit(1)
    main(args)
