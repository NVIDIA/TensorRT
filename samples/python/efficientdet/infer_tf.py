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
import argparse

import numpy as np
import tensorflow as tf


class TensorFlowInfer:
    """
    Implements TensorFlow inference of a saved model, following the same API as the TensorRTInfer class.
    """

    def __init__(self, saved_model_path):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.model = tf.saved_model.load(saved_model_path)
        self.pred_fn = self.model.signatures['serving_default']

        # Setup I/O bindings
        self.batch_size = 1
        self.inputs = []
        fn_inputs = self.pred_fn.structured_input_signature[1]
        for i, input in enumerate(list(fn_inputs.values())):
            self.inputs.append({
                'index': i,
                'name': input.name,
                'dtype': np.dtype(input.dtype.as_numpy_dtype()),
                'shape': [1, 512, 512, 3],  # This can be overridden later
            })
        self.outputs = []
        fn_outputs = self.pred_fn.structured_outputs
        for i, output in enumerate(list(fn_outputs.values())):
            self.outputs.append({
                'index': i,
                'name': output.name,
                'dtype': np.dtype(output.dtype.as_numpy_dtype()),
                'shape': output.shape.as_list(),
            })

    def override_input_shape(self, input, shape):
        self.inputs[input]['shape'] = shape
        self.batch_size = shape[0]

    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        return self.outputs[0]['shape'], self.outputs[0]['dtype']

    def infer(self, batch):
        # Process I/O and execute the network
        input = {self.inputs[0]['name']: tf.convert_to_tensor(batch)}
        output = self.pred_fn(**input)
        return output

    def process(self, batch, scales=None, nms_threshold=None):
        # Infer network
        output = self.infer(batch)

        # Extract the results depending on what kind of saved model this is
        boxes = None
        scores = None
        classes = None
        if len(self.outputs) == 1:
            # Detected as AutoML Saved Model
            assert len(self.outputs[0]['shape']) == 3 and self.outputs[0]['shape'][2] == 7
            results = output[self.outputs[0]['name']].numpy()
            boxes = results[:, :, 1:5]
            scores = results[:, :, 5]
            classes = results[:, :, 6].astype(np.int32)
        elif len(self.outputs) >= 4:
            # Detected as TFOD Saved Model
            assert output['num_detections']
            num = int(output['num_detections'].numpy().flatten()[0])
            boxes = output['detection_boxes'].numpy()[:, 0:num, :]
            scores = output['detection_scores'].numpy()[:, 0:num]
            classes = output['detection_classes'].numpy()[:, 0:num]

        # Process the results
        detections = [[]]
        normalized = (np.max(boxes) < 2.0)
        for n in range(scores.shape[1]):
            if scores[0][n] == 0.0:
                break
            scale = self.inputs[0]['shape'][2] if normalized else 1.0
            if scales:
                scale /= scales[0]
            if nms_threshold and scores[0][n] < nms_threshold:
                continue
            detections[0].append({
                'ymin': boxes[0][n][0] * scale,
                'xmin': boxes[0][n][1] * scale,
                'ymax': boxes[0][n][2] * scale,
                'xmax': boxes[0][n][3] * scale,
                'score': scores[0][n],
                'class': int(classes[0][n]) - 1,
            })
        return detections


def main(args):
    print("Running in benchmark mode")
    tf_infer = TensorFlowInfer(args.saved_model)
    input_size = [int(v) for v in args.input_size.split(",")]
    assert len(input_size) == 2
    tf_infer.override_input_shape(0, [args.batch_size, input_size[0], input_size[1], 3])
    spec = tf_infer.input_spec()
    batch = 255 * np.random.rand(*spec[0]).astype(spec[1])
    iterations = 200
    times = []
    for i in range(20):  # Warmup iterations
        tf_infer.infer(batch)
    for i in range(iterations):
        start = time.time()
        tf_infer.infer(batch)
        times.append(time.time() - start)
        print("Iteration {} / {}".format(i + 1, iterations), end="\r")
    print("Benchmark results include TensorFlow host overhead")
    print("Average Latency: {:.3f} ms".format(
        1000 * np.average(times)))
    print("Average Throughput: {:.1f} ips".format(
        tf_infer.batch_size / np.average(times)))

    print()
    print("Finished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--saved_model", required=True,
                        help="The TensorFlow saved model path to validate against")
    parser.add_argument("-i", "--input_size", default="512,512",
                        help="The input size to run the model with, in HEIGHT,WIDTH format")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        help="The batch size to run the model with")
    args = parser.parse_args()
    main(args)
