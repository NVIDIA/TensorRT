/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NV_INFER_PLUGIN_H
#define NV_INFER_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPluginUtils.h"

//!
//! \file NvInferPlugin.h
//!
//! This is the API for the Nvidia provided TensorRT plugins.
//!

namespace nvinfer1
{
namespace plugin
{

//!
//! \brief The PriorBox plugin layer generates the prior boxes of designated sizes and aspect ratios across all dimensions (H x W).
//! PriorBoxParameters defines a set of parameters for creating the PriorBox plugin layer.
//! It contains:
//! \param minSize Minimum box size in pixels. Can not be nullptr.
//! \param maxSize Maximum box size in pixels. Can be nullptr.
//! \param aspectRatios Aspect ratios of the boxes. Can be nullptr.
//! \param numMinSize Number of elements in minSize. Must be larger than 0.
//! \param numMaxSize Number of elements in maxSize. Can be 0 or same as numMinSize.
//! \param numAspectRatios Number of elements in aspectRatios. Can be 0.
//! \param flip If true, will flip each aspect ratio. For example, if there is aspect ratio "r", the aspect ratio "1.0/r" will be generated as well.
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
    int numMinSize, numMaxSize, numAspectRatios;
    bool flip;
    bool clip;
    float variance[4];
    int imgH, imgW;
    float stepH, stepW;
    float offset;
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
    int numAspectRatios, H, W;
    float variance[4];
};

//!
//! \enum CodeTypeSSD
//! \brief The type of encoding used for decoding the bounding boxes and loc_data.
//!
enum class CodeTypeSSD : int
{
    CORNER = 0,      //!< Use box corners.
    CENTER_SIZE = 1, //!< Use box centers and size.
    CORNER_SIZE = 2, //!< Use box centers and size.
    TF_CENTER = 3    //!< Use box centers and size but flip x and y coordinates.
};

//!
//! \brief The DetectionOutput plugin layer generates the detection output based on location and confidence predictions by doing non maximum suppression.
//! This plugin first decodes the bounding boxes based on the anchors generated. It then performs non_max_suppression on the decoded bouding boxes.
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
//!
struct DetectionOutputParameters
{
    bool shareLocation, varianceEncodedInTarget;
    int backgroundLabelId, numClasses, topK, keepTopK;
    float confidenceThreshold, nmsThreshold;
    CodeTypeSSD codeType;
    int inputOrder[3];
    bool confSigmoid;
    bool isNormalized;
};

//!
//! \brief The Region plugin layer performs region proposal calculation: generate 5 bounding boxes per cell (for yolo9000, generate 3 bounding boxes per cell).
//! For each box, calculating its probablities of objects detections from 80 pre-defined classifications (yolo9000 has 9416 pre-defined classifications,
//! and these 9416 items are organized as work-tree structure).
//! RegionParameters defines a set of parameters for creating the Region plugin layer.
//! \param num Number of predicted bounding box for each grid cell.
//! \param coords Number of coordinates for a bounding box.
//! \param classes Number of classfications to be predicted.
//! \param softmaxTree When performing yolo9000, softmaxTree is helping to do softmax on confidence scores, for element to get the precise classfication through word-tree structured classfication definition.
//! \deprecated. This plugin is superseded by createRegionPlugin()
//!
typedef struct
{
    int* leaf;
    int n;
    int* parent;
    int* child;
    int* group;
    char** name;

    int groups;
    int* groupSize;
    int* groupOffset;
} softmaxTree; // softmax tree

struct RegionParameters
{
    int num;
    int coords;
    int classes;
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
//!        normalized, i.e. not in the range [0,1]. Defaults to true.
//!

struct NMSParameters
{
    bool shareLocation;
    int backgroundLabelId, numClasses, topK, keepTopK;
    float scoreThreshold, iouThreshold;
    bool isNormalized;
};

} // end plugin namespace
} // end nvinfer1 namespace

extern "C"
{
//!
//! \brief Create a plugin layer that fuses the RPN and ROI pooling using user-defined parameters.
//! Registered plugin type "RPROI_TRT". Registered plugin version "1".
//! \param featureStride Feature stride.
//! \param preNmsTop Number of proposals to keep before applying NMS.
//! \param nmsMaxOut Number of remaining proposals after applying NMS.
//! \param iouThreshold IoU threshold.
//! \param minBoxSize Minimum allowed bounding box size before scaling.
//! \param spatialScale Spatial scale between the input image and the last feature map.
//! \param pooling Spatial dimensions of pooled ROIs.
//! \param anchorRatios Aspect ratios for generating anchor windows.
//! \param anchorScales Scales for generating anchor windows.
//!
//! \return Returns a FasterRCNN fused RPN+ROI pooling plugin. Returns nullptr on invalid inputs.
//!
TENSORRTAPI nvinfer1::IPluginV2* createRPNROIPlugin(int featureStride, int preNmsTop,
                                                                int nmsMaxOut, float iouThreshold, float minBoxSize,
                                                                float spatialScale, nvinfer1::DimsHW pooling,
                                                                nvinfer1::Weights anchorRatios, nvinfer1::Weights anchorScales);

//!
//! \brief The Normalize plugin layer normalizes the input to have L2 norm of 1 with scale learnable.
//! Registered plugin type "Normalize_TRT". Registered plugin version "1".
//! \param scales Scale weights that are applied to the output tensor.
//! \param acrossSpatial Whether to compute the norm over adjacent channels (acrossSpatial is true) or nearby spatial locations (within channel in which case acrossSpatial is false).
//! \param channelShared Whether the scale weight(s) is shared across channels.
//! \param eps Epsilon for not diviiding by zero.
//!
TENSORRTAPI nvinfer1::IPluginV2* createNormalizePlugin(const nvinfer1::Weights* scales, bool acrossSpatial, bool channelShared, float eps);

//!
//! \brief The PriorBox plugin layer generates the prior boxes of designated sizes and aspect ratios across all dimensions (H x W).
//! PriorBoxParameters defines a set of parameters for creating the PriorBox plugin layer.
//! Registered plugin type "PriorBox_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createPriorBoxPlugin(nvinfer1::plugin::PriorBoxParameters param);

//!
//! \brief The Grid Anchor Generator plugin layer generates the prior boxes of
//! designated sizes and aspect ratios across all dimensions (H x W) for all feature maps.
//! GridAnchorParameters defines a set of parameters for creating the GridAnchorGenerator plugin layer.
//! Registered plugin type "GridAnchor_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createAnchorGeneratorPlugin(nvinfer1::plugin::GridAnchorParameters* param, int numLayers);

//!
//! \brief The DetectionOutput plugin layer generates the detection output based on location and confidence predictions by doing non maximum suppression.
//! DetectionOutputParameters defines a set of parameters for creating the DetectionOutput plugin layer.
//! Registered plugin type "NMS_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createNMSPlugin(nvinfer1::plugin::DetectionOutputParameters param);

//!
//! \brief The Reorg plugin reshapes input of shape CxHxW into a (C*stride*stride)x(H/stride)x(W/stride) shape, used in YOLOv2.
//! It does that by taking 1 x stride x stride slices from tensor and flattening them into (stridexstride) x 1 x 1 shape.
//! Registered plugin type "Reorg_TRT". Registered plugin version "1".
//! \param stride Strides in H and W, it should divide both H and W. Also stride * stride should be less than or equal to C.
//!
TENSORRTAPI nvinfer1::IPluginV2* createReorgPlugin(int stride);

//!
//! \brief The Region plugin layer performs region proposal calculation: generate 5 bounding boxes per cell (for yolo9000, generate 3 bounding boxes per cell).
//! For each box, calculating its probablities of objects detections from 80 pre-defined classifications (yolo9000 has 9416 pre-defined classifications,
//! and these 9416 items are organized as work-tree structure).
//! RegionParameters defines a set of parameters for creating the Region plugin layer.
//! Registered plugin type "Region_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createRegionPlugin(nvinfer1::plugin::RegionParameters params);

//!
//! \brief The BatchedNMS Plugin performs non_max_suppression on the input boxes, per batch, across all classes.
//! It greedily selects a subset of bounding boxes in descending order of
//! score. Prunes away boxes that have a high intersection-over-union (IOU)
//! overlap with previously selected boxes. Bounding boxes are supplied as [y1, x1, y2, x2],
//! where (y1, x1) and (y2, x2) are the coordinates of any
//! diagonal pair of box corners and the coordinates can be provided as normalized
//! (i.e., lying in the interval [0, 1]) or absolute.
//! The plugin expects two inputs.
//! Input0 is expected to be 4-D float boxes tensor of shape [batch_size, num_boxes,
//! q, 4], where q can be either 1 (if shareLocation is true) or num_classes.
//! Input1 is expected to be a 3-D float scores tensor of shape [batch_size, num_boxes, num_classes]
//! representing a single score corresponding to each box.
//! The plugin returns four outputs.
//! num_detections : A [batch_size] int32 tensor indicating the number of valid
//! detections per batch item. Can be less than keepTopK. Only the top num_detections[i] entries in
//! nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid.
//! nmsed_boxes : A [batch_size, max_detections, 4] float32 tensor containing
//! the co-ordinates of non-max suppressed boxes.
//! nmsed_scores : A [batch_size, max_detections] float32 tensor containing the
//! scores for the boxes.
//! nmsed_classes :  A [batch_size, max_detections] float32 tensor containing the
//! classes for the boxes.
//!
//! Registered plugin type "BatchedNMS_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createBatchedNMSPlugin(nvinfer1::plugin::NMSParameters param);

//!
//! \brief Initialize and register all the existing TensorRT plugins to the Plugin Registry with an optional namespace.
//! The plugin library author should ensure that this function name is unique to the library.
//! This function should be called once before accessing the Plugin Registry.
//! \param logger Logger object to print plugin registration information
//! \param libNamespace Namespace used to register all the plugins in this library
//!
TENSORRTAPI bool initLibNvInferPlugins(void* logger, const char* libNamespace);
} // extern "C"

#endif // NV_INFER_PLUGIN_H
