#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import json
import argparse

import numpy as np
import tensorflow as tf

from infer import TensorRTInfer
from image_batcher import ImageBatcher
from visualize import visualize_detections, concat_visualizations


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

    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        return self.outputs[0]['shape'], self.outputs[0]['dtype']

    def infer(self, batch, scales=None, nms_threshold=None):
        # Process I/O and execute the network
        input = {self.inputs[0]['name']: tf.convert_to_tensor(batch)}
        output = self.pred_fn(**input)

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


def run(batcher, inferer, framework, nms_threshold=None):
    res_images = []
    res_detections = []
    for batch, images, scales in batcher.get_batch():
        res_detections += inferer.infer(batch, scales, nms_threshold)
        res_images += images
        print("Processing {} / {} images ({})".format(batcher.image_index, batcher.num_images, framework), end="\r")
    print()
    return res_images, res_detections


def parse_annotations(annotations_path):
    annotations = {}
    if annotations_path and os.path.exists(annotations_path):
        with open(annotations_path) as f:
            ann_json = json.load(f)
            for ann in ann_json['annotations']:
                img_id = ann['image_id']
                if img_id not in annotations.keys():
                    annotations[img_id] = []
                annotations[img_id].append({
                    'ymin': ann['bbox'][1],
                    'xmin': ann['bbox'][0],
                    'ymax': ann['bbox'][1] + ann['bbox'][3],
                    'xmax': ann['bbox'][0] + ann['bbox'][2],
                    'score': -1,
                    'class': ann['category_id'] - 1,
                })
    return annotations


def compare_images(tf_images, tf_detections, trt_images, trt_detections, output_dir, annotations_path, labels_path):
    labels = []
    if labels_path and os.path.exists(labels_path):
        with open(labels_path) as f:
            for i, label in enumerate(f):
                labels.append(label.strip())

    annotations = parse_annotations(annotations_path)

    count = 1
    for tf_img, tf_det, trt_img, trt_det in zip(tf_images, tf_detections, trt_images, trt_detections):
        vis = []
        names = []
        colors = []

        vis.append(visualize_detections(tf_img, None, tf_det, labels))
        names.append("TensorFlow")
        colors.append("DarkOrange")

        vis.append(visualize_detections(trt_img, None, trt_det, labels))
        names.append("TensorRT")
        colors.append("YellowGreen")

        if annotations:
            img_id = os.path.splitext(os.path.basename(trt_img))[0]
            if img_id.isnumeric():
                img_id = int(img_id)
            if img_id in annotations.keys():
                vis.append(visualize_detections(trt_img, None, annotations[img_id], labels))
                names.append("Ground Truth")
                colors.append("RoyalBlue")
            else:
                print("Image {} does not have a COCO annotation, skipping ground truth visualization".format(trt_img))

        basename = os.path.splitext(os.path.basename(tf_img))[0]
        output_path = os.path.join(output_dir, "{}.compare.png".format(basename))
        os.makedirs(output_dir, exist_ok=True)
        concat_visualizations(vis, names, colors, output_path)

        print("Processing {} / {} images (Visualization)".format(count, len(tf_images)), end="\r")
        count += 1
    print()


def main(args):
    tf_infer = TensorFlowInfer(args.saved_model)
    trt_infer = TensorRTInfer(args.engine)

    trt_batcher = ImageBatcher(args.input, *trt_infer.input_spec(), max_num_images=args.num_images)
    tf_infer.override_input_shape(0, [1, trt_batcher.height, trt_batcher.width, 3])  # Same size input in TF as TRT
    tf_batcher = ImageBatcher(args.input, *tf_infer.input_spec(), max_num_images=args.num_images)

    tf_images, tf_detections = run(tf_batcher, tf_infer, "TensorFlow", args.nms_threshold)
    trt_images, trt_detections = run(trt_batcher, trt_infer, "TensorRT", args.nms_threshold)

    compare_images(tf_images, tf_detections, trt_images, trt_detections, args.output, args.annotations, args.labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with")
    parser.add_argument("-m", "--saved_model", help="The TensorFlow saved model path to validate against")
    parser.add_argument("-i", "--input",
                        help="The input to infer, either a single image path, or a directory of images")
    parser.add_argument("-o", "--output", default=None, help="Directory where to save the visualization results")
    parser.add_argument("-l", "--labels", default="./labels_coco.txt",
                        help="File to use for reading the class labels from, default: ./labels_coco.txt")
    parser.add_argument("-a", "--annotations", default=None,
                        help="Set the path to the 'instances_val2017.json' file to use for COCO annotations, in which "
                             "case --input should point to the COCO val2017 dataset, default: not used")
    parser.add_argument("-n", "--num_images", default=100, type=int,
                        help="The maximum number of images to visualize, default: 100")
    parser.add_argument("-t", "--nms_threshold", type=float, help="Override the score threshold for the NMS operation, "
                                                                  "if higher than the threshold in the model/engine.")
    args = parser.parse_args()
    if not all([args.engine, args.saved_model, args.input, args.output]):
        parser.print_help()
        sys.exit(1)
    main(args)
