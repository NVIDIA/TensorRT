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
import argparse
import json
import numpy as np
from PIL import Image
from infer import TensorRTInfer
from image_batcher import ImageBatcher

def main(args):
    try:
        import object_detection.metrics.coco_tools as coco_tools
    except ImportError:
        print("Could not import the 'object_detection.metrics.coco_tools' module from TFOD. Maybe you did not install TFOD API")
        print("Please install TensorFlow 2 Object Detection API, check https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html")
        sys.exit(1)

    trt_infer = TensorRTInfer(args.engine, args.preprocessor, args.detection_type, args.iou_threshold)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec(), preprocessor=args.preprocessor)
    # Read annotations json as dictionary.
    with open(args.annotations) as f:
        data = json.load(f)
    groundtruth = coco_tools.COCOWrapper(data, detection_type=args.detection_type)
    detections_list = []
    for batch, images, scales in batcher.get_batch():
        print("Processing Image {} / {}".format(batcher.image_index, batcher.num_images), end="\r")
        detections = trt_infer.infer(batch, scales, args.nms_threshold)
        for i in range(len(images)):
            # Get inference image resolution.
            infer_im = Image.open(images[i])
            im_width, im_height = infer_im.size
            for n in range(len(detections[i])):
                source_id = int(os.path.splitext(os.path.basename(images[i]))[0])
                det = detections[i][n]
                if args.detection_type == 'bbox':
                    coco_det = {
                        'image_id': source_id,
                        'category_id': det['class']+1, # adjust class num
                        'bbox': [det['xmin'], det['ymin'], det['xmax'] - det['xmin'], det['ymax'] - det['ymin']],
                        'score': det['score']
                    }
                    detections_list.append(coco_det)
                elif args.detection_type == 'segmentation':
                    # Get detection bbox resolution.
                    det_width = round(det['xmax'] - det['xmin'])
                    det_height = round(det['ymax'] - det['ymin'])
                    # Create an image out of predicted mask array.
                    small_mask = Image.fromarray(det['mask'])
                    # Upsample mask to detection bbox's size.
                    mask = small_mask.resize((det_width, det_height), resample=Image.BILINEAR)
                    # Create an original image sized template for correct mask placement.
                    pad = Image.new("L", (im_width, im_height))
                    # Place your mask according to detection bbox placement.
                    pad.paste(mask, (round(det['xmin']), (round(det['ymin']))))
                    # Reconvert mask into numpy array for evaluation.
                    padded_mask = np.array(pad)
                    # Add one more dimension of 1, this is required by ExportSingleImageDetectionMasksToCoco.
                    final_mask = padded_mask[np.newaxis, :, :]
                    # Export detection mask to COCO format
                    coco_mask = coco_tools.ExportSingleImageDetectionMasksToCoco(image_id=source_id,
                        category_id_set=set(list(range(1,91))),
                        detection_classes=np.array([det['class']+1]),
                        detection_scores=np.array([det['score']]),
                        detection_masks=final_mask)
                    detections_list.append(coco_mask[0])

    # Finish evalutions.
    detections = groundtruth.LoadAnnotations(detections_list)
    if args.detection_type == 'bbox':
        evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections, iou_type="bbox")
    elif args.detection_type == 'segmentation':
        evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections, iou_type="segm")
    evaluator.ComputeMetrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with.")
    parser.add_argument("-i", "--input",
                        help="The input to infer, either a single image path, or a directory of images.")
    parser.add_argument("-d", "--detection_type", default="bbox", choices=["bbox", "segmentation"],
                        help="Detection type for COCO, either bbox or if you are using Mask R-CNN's instance segmentation - segmentation.")
    parser.add_argument("-a", "--annotations", help="Set the json file to use for COCO instance annotations.")
    parser.add_argument("-t", "--nms_threshold", type=float,
                        help="Override the score threshold for the NMS operation, if higher than the threshold in the engine.")
    parser.add_argument("--iou_threshold", default=0.5, type=float,
                        help="Select the IoU threshold for the mask segmentation. Range is 0 to 1. Pixel values more than threshold will become 1, less 0.")
    parser.add_argument("--preprocessor", default="fixed_shape_resizer", choices=["fixed_shape_resizer", "keep_aspect_ratio_resizer"],
                        help="Select the image preprocessor to use based on your pipeline.config, either 'fixed_shape_resizer' or 'keep_aspect_ratio_resizer', default: fixed_shape_resizer.")
    args = parser.parse_args()
    if not all([args.engine, args.input, args.annotations, args.preprocessor]):
        parser.print_help()
        print("\nThese arguments are required: --engine --input --output and --preprocessor")
        sys.exit(1)
    main(args)
