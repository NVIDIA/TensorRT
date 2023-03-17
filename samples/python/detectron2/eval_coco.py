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
import numpy as np
import torch
from PIL import Image
from infer import TensorRTInfer
from image_batcher import ImageBatcher

try:
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.evaluation import COCOEvaluator
    from detectron2.structures import Instances, Boxes, ROIMasks
except ImportError:
    print("Could not import Detectron 2 modules. Maybe you did not install Detectron 2")
    print("Please install Detectron 2, check https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md")
    sys.exit(1)

def build_evaluator(dataset_name):
    """
    Create evaluator for a COCO dataset.
    Currently only Mask R-CNN is supported, dataset of interest is COCO, so only COCOEvaluator is implemented.
    """
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["coco"]:
        return COCOEvaluator(dataset_name)
    else:
        raise NotImplementedError("Evaluator type is not supported")

def setup(config_file, weights):
    """
    Create config and perform basic setup.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights])
    cfg.freeze()
    return cfg

def main(args):
    # Set up Detectron 2 config and build evaluator.
    cfg = setup(args.det2_config, args.det2_weights)
    dataset_name = cfg.DATASETS.TEST[0]
    evaluator = build_evaluator(dataset_name)
    evaluator.reset()

    trt_infer = TensorRTInfer(args.engine)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec(), config_file=args.det2_config)

    for batch, images, scales in batcher.get_batch():
        print("Processing Image {} / {}".format(batcher.image_index, batcher.num_images), end="\r")
        detections = trt_infer.infer(batch, scales, args.nms_threshold)
        for i in range(len(images)):
            # Get inference image resolution.
            infer_im = Image.open(images[i])
            im_width, im_height = infer_im.size
            pred_boxes = []
            scores = []
            pred_classes = []
            # Number of detections.
            num_instances = len(detections[i])
            # Reserve numpy array to hold all mask predictions per image.
            pred_masks = np.empty((num_instances, 28, 28), dtype=np.float32)
            # Image ID, required for Detectron 2 evaluations.
            source_id = int(os.path.splitext(os.path.basename(images[i]))[0])
            # Loop over every single detection.
            for n in range(num_instances):
                det = detections[i][n]
                # Append box coordinates data.
                pred_boxes.append([det['ymin'], det['xmin'], det['ymax'], det['xmax']])
                # Append score.
                scores.append(det['score'])
                # Append class.
                pred_classes.append(det['class'])
                # Append mask.
                pred_masks[n] = det['mask']
            # Create new Instances object required for Detectron 2 evalutions and add:
            # boxes, scores, pred_classes, pred_masks.
            image_shape = (im_height, im_width)
            instances = Instances(image_shape)
            instances.pred_boxes = Boxes(pred_boxes)
            instances.scores = torch.tensor(scores)
            instances.pred_classes = torch.tensor(pred_classes)
            roi_masks = ROIMasks(torch.tensor(pred_masks))
            instances.pred_masks = roi_masks.to_bitmasks(instances.pred_boxes, im_height, im_width, args.iou_threshold).tensor
            # Process evaluations per image.
            image_dict = [{'instances': instances}]
            input_dict = [{'image_id': source_id}]
            evaluator.process(input_dict, image_dict)

    # Final evaluations, generation of mAP accuracy performance.
    evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with.")
    parser.add_argument("-i", "--input",
                        help="The input to infer, either a single image path, or a directory of images.")
    parser.add_argument("-c", "--det2_config", help="The Detectron 2 config file (.yaml) for the model", type=str)
    parser.add_argument("-w", "--det2_weights", help="The Detectron 2 model weights (.pkl)", type=str)
    parser.add_argument("-t", "--nms_threshold", type=float,
                        help="Override the score threshold for the NMS operation, if higher than the threshold in the engine.")
    parser.add_argument("--iou_threshold", default=0.5, type=float,
                        help="Select the IoU threshold for the mask segmentation. Range is 0 to 1. Pixel values more than threshold will become 1, less 0.")
    args = parser.parse_args()
    if not all([args.engine, args.input, args.det2_config, args.det2_weights]):
        parser.print_help()
        print("\nThese arguments are required: --engine --input --det2_config and --det2_weights")
        sys.exit(1)
    main(args)
