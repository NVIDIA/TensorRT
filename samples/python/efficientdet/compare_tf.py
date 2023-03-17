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
import sys
import json
import argparse

import numpy as np
import tensorflow as tf

from infer import TensorRTInfer
from infer_tf import TensorFlowInfer
from image_batcher import ImageBatcher
from visualize import visualize_detections, concat_visualizations


def run(batcher, inferer, framework, nms_threshold=None):
    res_images = []
    res_detections = []
    for batch, images, scales in batcher.get_batch():
        res_detections += inferer.process(batch, scales, nms_threshold)
        res_images += images
        print("Processing {} / {} images ({})".format(batcher.image_index, batcher.num_images, framework), end="\r")
    print()
    return res_images, res_detections


def parse_annotations(annotations_path):
    annotations = {}
    if annotations_path and os.path.exists(annotations_path):
        with open(annotations_path) as f:
            ann_json = json.load(f)
            for ann in ann_json["annotations"]:
                img_id = ann["image_id"]
                if img_id not in annotations.keys():
                    annotations[img_id] = []
                annotations[img_id].append(
                    {
                        "ymin": ann["bbox"][1],
                        "xmin": ann["bbox"][0],
                        "ymax": ann["bbox"][1] + ann["bbox"][3],
                        "xmax": ann["bbox"][0] + ann["bbox"][2],
                        "score": -1,
                        "class": ann["category_id"] - 1,
                    }
                )
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
