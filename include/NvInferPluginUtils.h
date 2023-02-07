/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
//! \brief The Permute plugin layer permutes the input tensor by changing the memory order of the data.
//! Quadruple defines a structure that contains an array of 4 integers. They can represent the permute orders or the
//! strides in each dimension.
//!
typedef struct
{
    int32_t data[4];
} Quadruple;

//!
//! \brief The PriorBox plugin layer generates the prior boxes of designated sizes and aspect ratios across all
//! dimensions (H x W). PriorBoxParameters defines a set of parameters for creating the PriorBox plugin layer. It
//! contains:
//! \param minSize Minimum box size in pixels. Can not be nullptr.
//! \param maxSize Maximum box size in pixels. Can be nullptr.
//! \param aspectRatios Aspect ratios of the boxes. Can be nullptr.
//! \param numMinSize Number of elements in minSize. Must be larger than 0.
//! \param numMaxSize Number of elements in maxSize. Can be 0 or same as numMinSize.
//! \param numAspectRatios Number of elements in aspectRatios. Can be 0.
//! \param flip If true, will flip each aspect ratio. For example, if there is an aspect ratio "r", the aspect ratio
//! "1.0/r" will be generated as well.
//! \param clip If true, will clip the prior so that it is within [0,1].
//! \param variance Variance for adjusting the prior boxes.
//! \param imgH Image height. If 0, then the H dimension of the data tensor will be used.
//! \param imgW Image width. If 0, then the W dimension of the data tensor will be used.
//! \param stepH Step in H. If 0, then (float)imgH/h will be used where h is the H dimension of the 1st input tensor.
//! \param stepW Step in W. If 0, then (float)imgW/w will be used where w is the W dimension of the 1st input tensor.
//! \param offset Offset to the top left corner of each cell.
//!
struct PriorBoxParameters
{
    float *minSize, *maxSize, *aspectRatios;
    int32_t numMinSize, numMaxSize, numAspectRatios;
    bool flip;
    bool clip;
    float variance[4];
    int32_t imgH, imgW;
    float stepH, stepW;
    float offset;
};

//!
//! \brief RPROIParams is used to create the RPROIPlugin instance.
//! It contains:
//! \param poolingH Height of the output in pixels after ROI pooling on feature map.
//! \param poolingW Width of the output in pixels after ROI pooling on feature map.
//! \param featureStride Feature stride; ratio of input image size to feature map size. Assuming that max pooling layers
//! in the neural network use square filters.
//! \param preNmsTop Number of proposals to keep before applying NMS.
//! \param nmsMaxOut Number of remaining proposals after applying NMS.
//! \param anchorsRatioCount Number of anchor box ratios.
//! \param anchorsScaleCount Number of anchor box scales.
//! \param iouThreshold IoU (Intersection over Union) threshold used for the NMS step.
//! \param minBoxSize Minimum allowed bounding box size before scaling, used for anchor box calculation.
//! \param spatialScale Spatial scale between the input image and the last feature map.
//!
struct RPROIParams
{
    int32_t poolingH;
    int32_t poolingW;
    int32_t featureStride;
    int32_t preNmsTop;
    int32_t nmsMaxOut;
    int32_t anchorsRatioCount;
    int32_t anchorsScaleCount;
    float iouThreshold;
    float minBoxSize;
    float spatialScale;
};


//!
//! \brief The Anchor Generator plugin layer generates the prior boxes of designated sizes and aspect ratios across all dimensions (H x W).
//! GridAnchorParameters defines a set of parameters for creating the plugin layer for all feature maps.
//! It contains:
//! \param minScale Scale of anchors corresponding to finest resolution.
//! \param maxScale Scale of anchors corresponding to coarsest resolution.
//! \param aspectRatios List of aspect ratios to place on each grid point.
//! \param numAspectRatios Number of elements in aspectRatios.
//! \param H Height of feature map to generate anchors for.
//! \param W Width of feature map to generate anchors for.
//! \param variance Variance for adjusting the prior boxes.
//!
struct GridAnchorParameters
{
    float minSize, maxSize;
    float* aspectRatios;
    int32_t numAspectRatios, H, W;
    float variance[4];
};

//!
//! \enum CodeTypeSSD
//! \brief The type of encoding used for decoding the bounding boxes and loc_data.
//!
enum class CodeTypeSSD : int32_t
{
    CORNER = 0,      //!< Use box corners.
    CENTER_SIZE = 1, //!< Use box centers and size.
    CORNER_SIZE = 2, //!< Use box centers and size.
    TF_CENTER = 3    //!< Use box centers and size but flip x and y coordinates.
};

//!
//! \brief The DetectionOutput plugin layer generates the detection output based on location and confidence predictions by doing non maximum suppression.
//! This plugin first decodes the bounding boxes based on the anchors generated. It then performs non_max_suppression on the decoded bounding boxes.
//! DetectionOutputParameters defines a set of parameters for creating the DetectionOutput plugin layer.
//! It contains:
//! \param shareLocation If true, bounding box are shared among different classes.
//! \param varianceEncodedInTarget If true, variance is encoded in target. Otherwise we need to adjust the predicted offset accordingly.
//! \param backgroundLabelId Background label ID. If there is no background class, set it as -1.
//! \param numClasses Number of classes to be predicted.
//! \param topK Number of boxes per image with top confidence scores that are fed into the NMS algorithm.
//! \param keepTopK Number of total bounding boxes to be kept per image after NMS step.
//! \param confidenceThreshold Only consider detections whose confidences are larger than a threshold.
//! \param nmsThreshold Threshold to be used in NMS.
//! \param codeType Type of coding method for bbox.
//! \param inputOrder Specifies the order of inputs {loc_data, conf_data, priorbox_data}.
//! \param confSigmoid Set to true to calculate sigmoid of confidence scores.
//! \param isNormalized Set to true if bounding box data is normalized by the network.
//! \param isBatchAgnostic Defaults to true. Set to false if prior boxes are unique per batch
//!
struct DetectionOutputParameters
{
    bool shareLocation, varianceEncodedInTarget;
    int32_t backgroundLabelId, numClasses, topK, keepTopK;
    float confidenceThreshold, nmsThreshold;
    CodeTypeSSD codeType;
    int32_t inputOrder[3];
    bool confSigmoid;
    bool isNormalized;
    bool isBatchAgnostic{true};
};

//!
//! \brief When performing yolo9000, softmaxTree is helping to do softmax on confidence scores, for element to get the precise classification through word-tree structured classification definition.
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
//! \brief The Region plugin layer performs region proposal calculation: generate 5 bounding boxes per cell (for yolo9000, generate 3 bounding boxes per cell).
//! For each box, calculating its probablities of objects detections from 80 pre-defined classifications (yolo9000 has 9418 pre-defined classifications,
//! and these 9418 items are organized as work-tree structure).
//! RegionParameters defines a set of parameters for creating the Region plugin layer.
//! \param num Number of predicted bounding box for each grid cell.
//! \param coords Number of coordinates for a bounding box.
//! \param classes Number of classifications to be predicted.
//! \param smTree Helping structure to do softmax on confidence scores.
//!
struct RegionParameters
{
    int32_t num;
    int32_t coords;
    int32_t classes;
    softmaxTree* smTree;
};

//!
//! \brief The NMSParameters are used by the BatchedNMSPlugin for performing
//! the non_max_suppression operation over boxes for object detection networks.
//! \param shareLocation If set to true, the boxes inputs are shared across all
//!        classes. If set to false, the boxes input should account for per class box data.
//! \param backgroundLabelId Label ID for the background class. If there is no background class, set it as -1
//! \param numClasses Number of classes in the network.
//! \param topK Number of bounding boxes to be fed into the NMS step.
//! \param keepTopK Number of total bounding boxes to be kept per image after NMS step.
//!        Should be less than or equal to the topK value.
//! \param scoreThreshold Scalar threshold for score (low scoring boxes are removed).
//! \param iouThreshold scalar threshold for IOU (new boxes that have high IOU overlap
//!        with previously selected boxes are removed).
//! \param isNormalized Set to false, if the box coordinates are not
//!        normalized, i.e. not in the range [0,1]. Defaults to false.
//!

struct NMSParameters
{
    bool shareLocation;
    int32_t backgroundLabelId, numClasses, topK, keepTopK;
    float scoreThreshold, iouThreshold;
    bool isNormalized;
};

} // namespace plugin
} // namespace nvinfer1

#endif // NV_INFER_PLUGIN_UTILS_H
