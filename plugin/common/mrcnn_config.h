/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_PLUGIN_MASKRCNN_CONFIG_H
#define TRT_PLUGIN_MASKRCNN_CONFIG_H
#include "NvInfer.h"
#include <string>
#include <vector>

namespace MaskRCNNConfig
{
static const nvinfer1::Dims3 IMAGE_SHAPE{3, 1024, 1024};

// Pooled ROIs
static const int POOL_SIZE = 7;
static const int MASK_POOL_SIZE = 14;

// Threshold to determine the mask area out of final convolution output
static const float MASK_THRESHOLD = 0.5F;

// Bounding box refinement standard deviation for RPN and final detections.
static const float RPN_BBOX_STD_DEV[] = {0.1F, 0.1F, 0.2F, 0.2F};
static const float BBOX_STD_DEV[] = {0.1F, 0.1F, 0.2F, 0.2F};

// Max number of final detections
static const int DETECTION_MAX_INSTANCES = 100;

// Minimum probability value to accept a detected instance
// ROIs below this threshold are skipped
static const float DETECTION_MIN_CONFIDENCE = 0.7F;

// Non-maximum suppression threshold for detection
static const float DETECTION_NMS_THRESHOLD = 0.3F;

// The strides of each layer of the FPN Pyramid. These values
// are based on a Resnet101 backbone.
static const std::vector<float> BACKBONE_STRIDES = {4.F, 8.F, 16.F, 32.F, 64.F};

// Size of the fully-connected layers in the classification graph
static const int FPN_CLASSIF_FC_LAYERS_SIZE = 1024;

// Size of the top-down layers used to build the feature pyramid
static const int TOP_DOWN_PYRAMID_SIZE = 256;

// Number of classification classes (including background)
static const int NUM_CLASSES = 1 + 80; // COCO has 80 classes

// Length of square anchor side in pixels
static const std::vector<float> RPN_ANCHOR_SCALES = {32.F, 64.F, 128.F, 256.F, 512.F};

// Ratios of anchors at each cell (width/height)
// A value of 1 represents a square anchor, and 0.5 is a wide anchor
static const float RPN_ANCHOR_RATIOS[] = {0.5F, 1.F, 2.F};

// Anchor stride
// If 1 then anchors are created for each cell in the backbone feature map.
// If 2, then anchors are created for every other cell, and so on.
static const int RPN_ANCHOR_STRIDE = 1;

// Although Python impementation uses 6000,
//  TRT fails if this number larger than kMAX_TOPK_K defined in engine/checkMacros.h
static const int MAX_PRE_NMS_RESULTS = 1024; // 3840;

// Non-max suppression threshold to filter RPN proposals.
// You can increase this during training to generate more propsals.
static const float RPN_NMS_THRESHOLD = 0.7F;

// ROIs kept after non-maximum suppression (training and inference)
static const int POST_NMS_ROIS_INFERENCE = 1000;

// COCO Class names
static const std::vector<std::string> CLASS_NAMES = {
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
};

static const std::string MODEL_NAME = "mrcnn_nchw.uff";
static const std::string MODEL_INPUT = "input_image";
static const nvinfer1::Dims3 MODEL_INPUT_SHAPE = IMAGE_SHAPE;
static const std::vector<std::string> MODEL_OUTPUTS = {"mrcnn_detection", "mrcnn_mask/Sigmoid"};
static const nvinfer1::Dims2 MODEL_DETECTION_SHAPE{DETECTION_MAX_INSTANCES, 6};
static const nvinfer1::Dims4 MODEL_MASK_SHAPE{DETECTION_MAX_INSTANCES, NUM_CLASSES, 28, 28};
} // namespace MaskRCNNConfig
#endif // TRT_PLUGIN_MASKRCNN_CONFIG_H
