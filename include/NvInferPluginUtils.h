/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_INFER_PLUGIN_UTILS_H
#define NV_INFER_PLUGIN_UTILS_H

#include "NvInferRuntimeCommon.h"

//!
//! \file NvInferPluginUtils.h
//!
//! This is the API for the Nvidia provided TensorRT plugin utilities.
//! It lists all the parameters utilized by the TensorRT plugins.
//!

namespace nvinfer1
{
namespace plugin
{

//!
//! \struct PriorBoxParameters
//!
//! \brief The PriorBox plugin layer generates the prior boxes of designated sizes and aspect ratios across all
//! dimensions (H x W).
//!
//! PriorBoxParameters defines a set of parameters for creating the PriorBox plugin layer.
//!
struct PriorBoxParameters
{
    float *minSize;          //!< Minimum box size in pixels. Can not be nullptr.
    float *maxSize;          //!< Maximum box size in pixels. Can be nullptr.
    float *aspectRatios;     //!< Aspect ratios of the boxes. Can be nullptr.
    int32_t numMinSize;      //!< Number of elements in minSize. Must be larger than 0.
    int32_t numMaxSize;      //!< Number of elements in maxSize. Can be 0 or same as numMinSize.
    int32_t numAspectRatios; //!< Number of elements in aspectRatios. Can be 0.
    bool flip;               //!< If true, will flip each aspect ratio. For example,
                             //!< if there is an aspect ratio "r", the aspect ratio "1.0/r" will be generated as well.
    bool clip;               //!< If true, will clip the prior so that it is within [0,1].
    float variance[4];       //!< Variance for adjusting the prior boxes.
    int32_t imgH;            //!< Image height. If 0, then the H dimension of the data tensor will be used.
    int32_t imgW;            //!< Image width. If 0, then the W dimension of the data tensor will be used.
    float stepH;             //!< Step in H. If 0, then (float)imgH/h will be used where h is the H dimension of the 1st input tensor.
    float stepW;             //!< Step in W. If 0, then (float)imgW/w will be used where w is the W dimension of the 1st input tensor.
    float offset;            //!< Offset to the top left corner of each cell.
};

//!
//! \struct RPROIParams
//!
//! \brief RPROIParams is used to create the RPROIPlugin instance.
//!
struct RPROIParams
{
    int32_t poolingH;          //!< Height of the output in pixels after ROI pooling on feature map.
    int32_t poolingW;          //!< Width of the output in pixels after ROI pooling on feature map.
    int32_t featureStride;     //!< Feature stride; ratio of input image size to feature map size.
                               //!< Assuming that max pooling layers in the neural network use square filters.
    int32_t preNmsTop;         //!< Number of proposals to keep before applying NMS.
    int32_t nmsMaxOut;         //!< Number of remaining proposals after applying NMS.
    int32_t anchorsRatioCount; //!< Number of anchor box ratios.
    int32_t anchorsScaleCount; //!< Number of anchor box scales.
    float iouThreshold;        //!< IoU (Intersection over Union) threshold used for the NMS step.
    float minBoxSize;          //!< Minimum allowed bounding box size before scaling, used for anchor box calculation.
    float spatialScale;        //!< Spatial scale between the input image and the last feature map.
};

//!
//! \struct GridAnchorParameters
//!
//! \brief The Anchor Generator plugin layer generates the prior boxes of designated sizes and aspect ratios across all dimensions (H x W).
//! GridAnchorParameters defines a set of parameters for creating the plugin layer for all feature maps.
//!
struct GridAnchorParameters
{
    float minSize;           //!< Scale of anchors corresponding to finest resolution.
    float maxSize;           //!< Scale of anchors corresponding to coarsest resolution.
    float* aspectRatios;     //!< List of aspect ratios to place on each grid point.
    int32_t numAspectRatios; //!< Number of elements in aspectRatios.
    int32_t H;               //!< Height of feature map to generate anchors for.
    int32_t W;               //!< Width of feature map to generate anchors for.
    float variance[4];       //!< Variance for adjusting the prior boxes.
};

//!
//! \brief When performing yolo9000, softmaxTree is helping to do softmax on confidence scores,
//! for element to get the precise classification through word-tree structured classification definition.
//!
struct softmaxTree
{
    int32_t* leaf;
    int32_t n;
    int32_t* parent;
    int32_t* child;
    int32_t* group;
    char** name;
    int32_t groups;
    int32_t* groupSize;
    int32_t* groupOffset;
};

//!
//! \brief The Region plugin layer performs region proposal calculation.
//!
//! Generate 5 bounding boxes per cell (for yolo9000, generate 3 bounding boxes per cell).
//! For each box, calculating its probabilities of objects detections from 80 pre-defined classifications
//! (yolo9000 has 9418 pre-defined classifications, and these 9418 items are organized as work-tree structure).
//! RegionParameters defines a set of parameters for creating the Region plugin layer.
//!
struct RegionParameters
{
    int32_t num;         //!< Number of predicted bounding box for each grid cell.
    int32_t coords;      //!< Number of coordinates for a bounding box.
    int32_t classes;     //!< Number of classifications to be predicted.
    softmaxTree* smTree; //!< Helping structure to do softmax on confidence scores.
};

} // namespace plugin
} // namespace nvinfer1

#endif // NV_INFER_PLUGIN_UTILS_H
