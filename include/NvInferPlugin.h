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
//!
//! \enum PluginType
//!
//! \brief The type values for the various plugins.
//!
//! \see INvPlugin::getPluginType()
//!
enum class PluginType : int
{
    kFASTERRCNN = 0,         //!< FasterRCNN fused plugin (RPN + ROI pooling).
    kNORMALIZE = 1,          //!< Normalize plugin.
    kPERMUTE = 2,            //!< Permute plugin.
    kPRIORBOX = 3,           //!< PriorBox plugin.
    kSSDDETECTIONOUTPUT = 4, //!< SSD DetectionOutput plugin.
    kCONCAT = 5,             //!< Concat plugin.
    kPRELU = 6,              //!< YOLO PReLU Plugin.
    kYOLOREORG = 7,          //!< YOLO Reorg Plugin.
    kYOLOREGION = 8,         //!< YOLO Region Plugin.
    kANCHORGENERATOR = 9,    //!< SSD Grid Anchor Generator.
};

//!< Maximum number of elements in PluginType enum. \see PluginType
template <>
inline int EnumMax<PluginType>()
{
    return 10;
}

namespace plugin
{
//!
//! \class INvPlugin
//!
//! \brief Common interface for the Nvidia created plugins.
//!
//! This class provides a common subset of functionality that is used
//! to provide distinguish the Nvidia created plugins. Each plugin provides a
//! function to validate the parameter options and create the plugin
//! object.
//!
class INvPlugin : public IPlugin
{
public:
    //!
    //! \brief Get the parameter plugin ID.
    //!
    //! \return The ID of the plugin.
    //!
    virtual PluginType getPluginType() const = 0;

    //!
    //! \brief Get the name of the plugin from the ID
    //!
    //! \return The name of the plugin specified by \p id. Return nullptr if invalid ID is specified.
    //!
    //! The valid \p id values are ranged [0, numPlugins()).
    //!
    virtual const char* getName() const = 0;

    //!
    //! \brief Destroy the plugin.
    //!
    //! The valid \p id values are ranged [0, numPlugins()).
    //!
    virtual void destroy() = 0;

protected:
    ~INvPlugin() {}
}; // INvPlugin

//!
//! \param featureStride Feature stride.
//! \param preNmsTop Number of proposals to keep before applying NMS.
//! \param nmsMaxOut Number of remaining proposals after applying NMS.
//! \param iouThreshold IoU threshold.
//! \param minBoxSize Minimum allowed bounding box size before scaling.
//! \param spatialScale Spatial scale between the input image and the last feature map.
//! \param pooling Spatial dimensions of pooled ROIs.
//! \param anchorRatios Aspect ratios for generating anchor windows.
//! \param anchorScales Scales for generating anchor windows.
//! \brief Create a plugin layer that fuses the RPN and ROI pooling using user-defined parameters.
//!
//! \return Returns a FasterRCNN fused RPN+ROI pooling plugin. Returns nullptr on invalid inputs.
//!
//! \see INvPlugin
//! \deprecated. This plugin is superseded by createRPNROIPlugin()
//!
TENSORRTAPI INvPlugin* createFasterRCNNPlugin(int featureStride, int preNmsTop,
                                              int nmsMaxOut, float iouThreshold, float minBoxSize,
                                              float spatialScale, DimsHW pooling,
                                              Weights anchorRatios, Weights anchorScales);
TENSORRTAPI INvPlugin* createFasterRCNNPlugin(const void* data, size_t length);

//!
//! \brief The Normalize plugin layer normalizes the input to have L2 norm of 1 with scale learnable.
//! \param scales Scale weights that are applied to the output tensor.
//! \param acrossSpatial Whether to compute the norm over adjacent channels (acrossSpatial is true) or nearby spatial locations (within channel in which case acrossSpatial is false).
//! \param channelShared Whether the scale weight(s) is shared across channels.
//! \param eps Epsilon for not diviiding by zero.
//! \deprecated. This plugin is superseded by createNormalizePlugin()
//!
TENSORRTAPI INvPlugin* createSSDNormalizePlugin(const Weights* scales, bool acrossSpatial, bool channelShared, float eps);
TENSORRTAPI INvPlugin* createSSDNormalizePlugin(const void* data, size_t length);

//!
//! \brief The Permute plugin layer permutes the input tensor by changing the memory order of the data.
//! Quadruple defines a structure that contains an array of 4 integers. They can represent the permute orders or the strides in each dimension.
//!
typedef struct
{
    int data[4];
} Quadruple;

//!
//! \param permuteOrder The new orders that are used to permute the data.
//! \deprecated. Please use the TensorRT Shuffle layer for Permute operation
//!
TENSORRTAPI INvPlugin* createSSDPermutePlugin(Quadruple permuteOrder);
TENSORRTAPI INvPlugin* createSSDPermutePlugin(const void* data, size_t length);

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
//! \param param Set of parameters for creating the PriorBox plugin layer.
//! \deprecated. This plugin is superseded by createPriorBoxPlugin()
//!
TENSORRTAPI INvPlugin* createSSDPriorBoxPlugin(PriorBoxParameters param);
TENSORRTAPI INvPlugin* createSSDPriorBoxPlugin(const void* data, size_t length);

//!
//! \brief The Grid Anchor Generator plugin layer generates the prior boxes of
//! designated sizes and aspect ratios across all dimensions (H x W) for all feature maps.
//! GridAnchorParameters defines a set of parameters for creating the GridAnchorGenerator plugin layer.
//! \deprecated. This plugin is superseded by createAnchorGeneratorPlugin()
//!
TENSORRTAPI INvPlugin* createSSDAnchorGeneratorPlugin(GridAnchorParameters* param, int numLayers);
TENSORRTAPI INvPlugin* createSSDAnchorGeneratorPlugin(const void* data, size_t length);

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
//! \param param Set of parameters for creating the DetectionOutput plugin layer.
//! \deprecated. This plugin is superseded by createNMSPlugin()
//!
TENSORRTAPI INvPlugin* createSSDDetectionOutputPlugin(DetectionOutputParameters param);
TENSORRTAPI INvPlugin* createSSDDetectionOutputPlugin(const void* data, size_t length);

//!
//! \brief The Concat plugin layer basically performs the concatention for 4D tensors. Unlike the Concatenation layer in early version of TensorRT,
//! it allows the user to specify the axis along which to concatenate. The axis can be 1 (across channel), 2 (across H), or 3 (across W).
//! More particularly, this Concat plugin layer also implements the "ignoring the batch dimension" switch. If turned on, all the input tensors will be treated as if their batch sizes were 1.
//! \param concatAxis Axis along which to concatenate. Can't be the "N" dimension.
//! \param ignoreBatch If true, all the input tensors will be treated as if their batch sizes were 1.
//! \deprecated. This plugin is superseded by native TensorRT concatenation layer
//!
TENSORRTAPI INvPlugin* createConcatPlugin(int concatAxis, bool ignoreBatch);
TENSORRTAPI INvPlugin* createConcatPlugin(const void* data, size_t length);

//!
//! \brief The PReLu plugin layer performs leaky ReLU for 4D tensors. Give an input value x, the PReLU layer computes the output as x if x > 0
//!  and negative_slope //! x if x <= 0.
//! \param negSlope Negative_slope value.
//! \deprecated. This plugin is superseded by createLReLUPlugin()
//!
TENSORRTAPI INvPlugin* createPReLUPlugin(float negSlope);
TENSORRTAPI INvPlugin* createPReLUPlugin(const void* data, size_t length);

//!
//! \brief The Reorg plugin layer maps the 512x26x26 feature map onto a 2048x13x13 feature map, so that it can be concatenated with the feature maps at 13x13 resolution.
//! \param stride Strides in H and W.
//! \deprecated. This plugin is superseded by createReorgPlugin()
//!
TENSORRTAPI INvPlugin* createYOLOReorgPlugin(int stride);
TENSORRTAPI INvPlugin* createYOLOReorgPlugin(const void* data, size_t length);

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

TENSORRTAPI INvPlugin* createYOLORegionPlugin(RegionParameters params);
TENSORRTAPI INvPlugin* createYOLORegionPlugin(const void* data, size_t length);

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
//! \brief The LReLu plugin layer performs leaky ReLU for 4D tensors. Give an input value x, the PReLU layer computes the output as x if x > 0 and negative_slope //! x if x <= 0.
//! Registered plugin type "LReLU_TRT". Registered plugin version "1".
//! \param negSlope Negative_slope value.
//!
TENSORRTAPI nvinfer1::IPluginV2* createLReLUPlugin(float negSlope);

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
//! \brief The Clip Plugin performs a clip operation on the input tensor. It
//! clips the tensor values to a specified min and max. Any value less than clipMin are set to clipMin.
//! Any values greater than clipMax are set to clipMax. For example, this plugin can be used
//! to perform a Relu6 operation by specifying clipMin=0.0 and clipMax=6.0
//! Registered plugin type "Clip_TRT". Registered plugin version "1".
//! \param layerName The name of the TensorRT layer.
//! \param clipMin The minimum value to clip to.
//! \param clipMax The maximum value to clip to.
//!
TENSORRTAPI nvinfer1::IPluginV2* createClipPlugin(const char* layerName, float clipMin, float clipMax);

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
