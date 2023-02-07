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
    annotations = {}
    for line in open(args.annotations, "r"):
        line = line.strip().split(args.separator)
        if len(line) < 2 or not line[1].isnumeric():
            print("Could not parse the annotations file correctly, make sure the correct separator is used")
            sys.exit(1)
        annotations[os.path.basename(line[0])] = int(line[1])

    trt_infer = TensorRTInfer(args.engine)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec(), preprocessor=args.preprocessor)
    top1 = 0
    top5 = 0
    total = 0
    for batch, images in batcher.get_batch():
        classes, scores, top = trt_infer.infer(batch, top=5)
        for i in range(len(images)):
            image = os.path.basename(images[i])
            if image not in annotations.keys():
                print(
                    "Image '{}' does not appear in the annotations file, please make sure all evaluated "
                    "images have a corresponding ground truth label".format(image)
                )
                sys.exit(1)
            if annotations[image] == classes[i]:
                top1 += 1
            if annotations[image] in top[0][i]:
                top5 += 1
            total += 1
            top1_acc = 100 * (top1 / total)
            top5_acc = 100 * (top5 / total)
            print(
                "Processing {} / {} : Top-1 {:0.1f}% , Top-5: {:0.1f}%     ".format(
                    total, batcher.num_images, top1_acc, top5_acc
                ),
                end="\r",
            )
    print()
    print("Top-1 Accuracy: {:0.3f}%".format(top1_acc))
    print("Top-5 Accuracy: {:0.3f}%".format(top5_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with")
    parser.add_argument(
        "-i", "--input", help="The input to infer, either a single image path, or a directory of images"
    )
    parser.add_argument("-a", "--annotations", help="Set the file to use for classification ground truth annotations")
    parser.add_argument(
        "-s",
        "--separator",
        default=" ",
        help="Separator to use between columns when parsing the annotations file, default: ' ' (space)",
    )
    parser.add_argument(
        "-p",
        "--preprocessor",
        default="V2",
        choices=["V1", "V1MS", "V2"],
        help="Select the image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2",
    )
    args = parser.parse_args()
    if not all([args.engine, args.input, args.annotations]):
        parser.print_help()
        print("\nThese arguments are required: --engine  --input and --annotations")
        sys.exit(1)
    main(args)
