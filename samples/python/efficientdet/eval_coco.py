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

from infer import TensorRTInfer
from image_batcher import ImageBatcher


def main(args):
    automl_path = os.path.realpath(args.automl_path)
    sys.path.insert(1, os.path.join(automl_path, "efficientdet"))
    try:
        import coco_metric
    except ImportError:
        print("Could not import the 'coco_metric' module from AutoML. Searching in: {}".format(automl_path))
        print("Please clone the repository https://github.com/google/automl and provide its path with --automl_path.")
        sys.exit(1)

    trt_infer = TensorRTInfer(args.engine)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec())
    evaluator = coco_metric.EvaluationMetric(filename=args.annotations)
    for batch, images, scales in batcher.get_batch():
        print("Processing Image {} / {}".format(batcher.image_index, batcher.num_images), end="\r")
        detections = trt_infer.process(batch, scales, args.nms_threshold)
        coco_det = np.zeros((len(images), max([len(d) for d in detections]), 7))
        coco_det[:, :, -1] = -1
        for i in range(len(images)):
            for n in range(len(detections[i])):
                source_id = int(os.path.splitext(os.path.basename(images[i]))[0])
                det = detections[i][n]
                coco_det[i][n] = [
                    source_id,
                    det["xmin"],
                    det["ymin"],
                    det["xmax"] - det["xmin"],
                    det["ymax"] - det["ymin"],
                    det["score"],
                    det["class"] + 1,  # The COCO evaluator expects class 0 to be background, so offset by 1
                ]
        evaluator.update_state(None, coco_det)
    print()
    evaluator.result(100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with")
    parser.add_argument("-i", "--input",
                        help="The input to infer, either a single image path, or a directory of images")
    parser.add_argument("-a", "--annotations", help="Set the path to the COCO 'instances_val2017.json' file")
    parser.add_argument("-p", "--automl_path", default="./automl",
                        help="Set the path where to find the AutoML repository, from "
                             "https://github.com/google/automl. Default: ./automl")
    parser.add_argument("-t", "--nms_threshold", type=float, help="Override the score threshold for the NMS operation, "
                                                                  "if higher than the threshold in the engine.")
    args = parser.parse_args()
    if not all([args.engine, args.input, args.annotations]):
        parser.print_help()
        print("\nThese arguments are required: --engine  --input and --annotations")
        sys.exit(1)
    main(args)
