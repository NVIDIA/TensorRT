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

#ifndef TRT_KERNEL_H
#define TRT_KERNEL_H

#include "common/plugin.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>

#define DEBUG_ENABLE 0

typedef enum
{
    NCHW = 0,
    NC4HW = 1,
    NC32HW = 2
} DLayout_t;
#ifndef TRT_RPNLAYER_H

pluginStatus_t allClassNMS(cudaStream_t stream, int32_t num, int32_t num_classes, int32_t num_preds_per_class,
    int32_t top_k, float nms_threshold, bool share_location, bool isNormalized, nvinfer1::DataType DT_SCORE,
    nvinfer1::DataType DT_BBOX, void* bbox_data, void* beforeNMS_scores, void* beforeNMS_index_array,
    void* afterNMS_scores, void* afterNMS_index_array, bool flipXY, const float score_shift, bool caffeSemantics);

pluginStatus_t detectionInference(cudaStream_t stream, int32_t N, int32_t C1, int32_t C2, bool shareLocation,
    bool varianceEncodedInTarget, int32_t backgroundLabelId, int32_t numPredsPerClass, int32_t numClasses, int32_t topK,
    int32_t keepTopK, float confidenceThreshold, float nmsThreshold, nvinfer1::plugin::CodeTypeSSD codeType,
    nvinfer1::DataType DT_BBOX, const void* locData, const void* priorData, nvinfer1::DataType DT_SCORE,
    const void* confData, void* keepCount, void* topDetections, void* workspace, bool isNormalized = true,
    bool confSigmoid = false, int32_t scoreBits = 16, const bool isBatchAgnostic = true);

pluginStatus_t nmsInference(cudaStream_t stream, int32_t N, int32_t boxesSize, int32_t scoresSize, bool shareLocation,
    int32_t backgroundLabelId, int32_t numPredsPerClass, int32_t numClasses, int32_t topK, int32_t keepTopK,
    float scoreThreshold, float iouThreshold, nvinfer1::DataType DT_BBOX, const void* locData,
    nvinfer1::DataType DT_SCORE, const void* confData, void* keepCount, void* nmsedBoxes, void* nmsedScores,
    void* nmsedClasses, void* workspace, bool isNormalized = true, bool confSigmoid = false, bool clipBoxes = true,
    int32_t scoreBits = 16, bool caffeSemantics = true);

pluginStatus_t gatherTopDetections(cudaStream_t stream, bool shareLocation, int32_t numImages, int32_t numPredsPerClass,
    int32_t numClasses, int32_t topK, int32_t keepTopK, nvinfer1::DataType DT_BBOX, nvinfer1::DataType DT_SCORE,
    const void* indices, const void* scores, const void* bboxData, void* keepCount, void* topDetections,
    const float scoreShift);

size_t detectionForwardBBoxDataSize(int32_t N, int32_t C1, nvinfer1::DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int32_t N, int32_t C1, nvinfer1::DataType DT_BBOX);

size_t sortScoresPerClassWorkspaceSize(
    int32_t num, int32_t num_classes, int32_t num_preds_per_class, nvinfer1::DataType DT_CONF);

size_t sortScoresPerImageWorkspaceSize(int32_t num_images, int32_t num_items_per_image, nvinfer1::DataType DT_SCORE);

pluginStatus_t sortScoresPerImage(cudaStream_t stream, int32_t num_images, int32_t num_items_per_image,
    nvinfer1::DataType DT_SCORE, void* unsorted_scores, void* unsorted_bbox_indices, void* sorted_scores,
    void* sorted_bbox_indices, void* workspace, int32_t score_bits);

pluginStatus_t sortScoresPerClass(cudaStream_t stream, int32_t num, int32_t num_classes, int32_t num_preds_per_class,
    int32_t background_label_id, float confidence_threshold, nvinfer1::DataType DT_SCORE, void* conf_scores_gpu,
    void* index_array_gpu, void* workspace, const int32_t score_bits, const float score_shift);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int32_t count);

const char* cublasGetErrorString(cublasStatus_t error);

pluginStatus_t permuteData(cudaStream_t stream, int32_t nthreads, int32_t num_classes, int32_t num_data,
    int32_t num_dim, nvinfer1::DataType DT_DATA, bool confSigmoid, const void* data, void* new_data);

size_t detectionForwardPreNMSSize(int32_t N, int32_t C2);

size_t detectionForwardPostNMSSize(int32_t N, int32_t numClasses, int32_t topK);

pluginStatus_t decodeBBoxes(cudaStream_t stream, int32_t nthreads, nvinfer1::plugin::CodeTypeSSD code_type,
    bool variance_encoded_in_target, int32_t num_priors, bool share_location, int32_t num_loc_classes,
    int32_t background_label_id, bool clip_bbox, nvinfer1::DataType DT_BBOX, const void* loc_data,
    const void* prior_data, void* bbox_data, const bool batch_agnostic);

size_t normalizePluginWorkspaceSize(bool acrossSpatial, int32_t C, int32_t H, int32_t W);

pluginStatus_t normalizeInference(cudaStream_t stream, cublasHandle_t handle, bool acrossSpatial, bool channelShared,
    int32_t N, int32_t C, int32_t H, int32_t W, float eps, const void* scale, const void* inputData, void* outputData,
    void* workspace);

pluginStatus_t scatterNDInference(cudaStream_t stream, int32_t* outputDims, int32_t nOutputDims, int32_t sliceRank,
    int32_t nRows, int32_t rowSize, int32_t CopySize, int32_t sizeOfElementInBytes, const void* index,
    const void* updates, const void* data, void* output, void* workspace);

pluginStatus_t priorBoxInference(cudaStream_t stream, nvinfer1::plugin::PriorBoxParameters param, int32_t H, int32_t W,
    int32_t numPriors, int32_t numAspectRatios, const void* minSize, const void* maxSize, const void* aspectRatios,
    void* outputData);

pluginStatus_t lReLUInference(cudaStream_t stream, int32_t n, float negativeSlope, const void* input, void* output);

pluginStatus_t reorgInference(cudaStream_t stream, int32_t batch, int32_t C, int32_t H, int32_t W, int32_t stride,
    const void* input, void* output);

pluginStatus_t anchorGridInference(cudaStream_t stream, nvinfer1::plugin::GridAnchorParameters param,
    int32_t numAspectRatios, const void* aspectRatios, const void* scales, void* outputData);

pluginStatus_t regionInference(cudaStream_t stream, int32_t batch, int32_t C, int32_t H, int32_t W, int32_t num,
    int32_t coords, int32_t classes, bool hasSoftmaxTree, const nvinfer1::plugin::softmaxTree* smTree,
    const void* input, void* output);

// GENERATE ANCHORS
// For now it takes host pointers - ratios and scales but
// in GPU MODE anchors should be device pointer
pluginStatus_t generateAnchors(cudaStream_t stream,
    int32_t numRatios, // number of ratios
    float* ratios,     // ratio array
    int32_t numScales, // number of scales
    float* scales,     // scale array
    int32_t baseSize,  // size of the base anchor (baseSize x baseSize)
    float* anchors);   // output anchors (numRatios x numScales)

// BBD2P
pluginStatus_t bboxDeltas2Proposals(cudaStream_t stream,
    int32_t N,                     // batch size
    int32_t A,                     // number of anchors
    int32_t H,                     // last feature map H
    int32_t W,                     // last feature map W
    int32_t featureStride,         // feature stride
    float minBoxSize,              // minimum allowed box size before scaling
    const float* imInfo,           // image info (nrows, ncols, image scale)
    const float* anchors,          // input anchors
    nvinfer1::DataType tDeltas,    // type of input deltas
    DLayout_t lDeltas,             // layout of input deltas
    const void* deltas,            // input deltas
    nvinfer1::DataType tProposals, // type of output proposals
    DLayout_t lProposals,          // layout of output proposals
    void* proposals,               // output proposals
    nvinfer1::DataType tScores,    // type of output scores
    DLayout_t lScores,             // layout of output scores
    void* scores);                 // output scores (the score associated with too small box will be set to -inf)

// NMS
pluginStatus_t nms(cudaStream_t stream,
    int32_t N,                     // batch size
    int32_t R,                     // number of ROIs (region of interest) per image
    int32_t preNmsTop,             // number of proposals before applying NMS
    int32_t nmsMaxOut,             // number of remaining proposals after applying NMS
    float iouThreshold,            // IoU threshold
    nvinfer1::DataType tFgScores,  // type of foreground scores
    DLayout_t lFgScores,           // layout of foreground scores
    void* fgScores,                // foreground scores
    nvinfer1::DataType tProposals, // type of proposals
    DLayout_t lProposals,          // layout of proposals
    const void* proposals,         // proposals
    void* workspace,               // workspace
    nvinfer1::DataType tRois,      // type of ROIs
    void* rois);                   // ROIs

// WORKSPACE SIZES
size_t proposalsForwardNMSWorkspaceSize(int32_t N, int32_t A, int32_t H, int32_t W, int32_t nmsMaxOut);

size_t proposalsForwardBboxWorkspaceSize(int32_t N, int32_t A, int32_t H, int32_t W);

size_t proposalForwardFgScoresWorkspaceSize(int32_t N, int32_t A, int32_t H, int32_t W);

size_t proposalsInferenceWorkspaceSize(int32_t N, int32_t A, int32_t H, int32_t W, int32_t nmsMaxOut);

size_t RPROIInferenceFusedWorkspaceSize(int32_t N, int32_t A, int32_t H, int32_t W, int32_t nmsMaxOut);

// PROPOSALS INFERENCE
pluginStatus_t proposalsInference(cudaStream_t stream, int32_t N, int32_t A, int32_t H, int32_t W,
    int32_t featureStride, int32_t preNmsTop, int32_t nmsMaxOut, float iouThreshold, float minBoxSize,
    const float* imInfo, const float* anchors, nvinfer1::DataType tScores, DLayout_t lScores, const void* scores,
    nvinfer1::DataType tDeltas, DLayout_t lDeltas, const void* deltas, void* workspace, nvinfer1::DataType tRois,
    void* rois);

// EXTRACT FG SCORES
pluginStatus_t extractFgScores(cudaStream_t stream, int32_t N, int32_t A, int32_t H, int32_t W,
    nvinfer1::DataType tScores, DLayout_t lScores, const void* scores, nvinfer1::DataType tFgScores,
    DLayout_t lFgScores, void* fgScores);

// ROI INFERENCE
pluginStatus_t roiInference(cudaStream_t stream,
    const int32_t R,        // TOTAL number of rois -> ~nmsMaxOut * N
    const int32_t N,        // Batch size
    const int32_t C,        // Channels
    const int32_t H,        // Input feature map H
    const int32_t W,        // Input feature map W
    const int32_t poolingH, // Output feature map H
    const int32_t poolingW, // Output feature map W
    const float spatialScale, const nvinfer1::DataType tRois, const void* rois, const nvinfer1::DataType tFeatureMap,
    const DLayout_t lFeatureMap, const void* featureMap, const nvinfer1::DataType tTop, const DLayout_t lTop, void* top,
    size_t deviceSmemSize);

// ROI FORWARD
pluginStatus_t roiForward(cudaStream_t stream,
    int32_t R,        // TOTAL number of rois -> ~nmsMaxOut * N
    int32_t N,        // Batch size
    int32_t C,        // Channels
    int32_t H,        // Input feature map H
    int32_t W,        // Input feature map W
    int32_t poolingH, // Output feature map H
    int32_t poolingW, // Output feature map W
    float spatialScale, nvinfer1::DataType tRois, const void* rois, nvinfer1::DataType tFeatureMap,
    DLayout_t lFeatureMap, const void* featureMap, nvinfer1::DataType tTop, DLayout_t lTop, void* top, int32_t* maxIds);

// RP ROI Fused INFERENCE
pluginStatus_t RPROIInferenceFused(cudaStream_t stream, int32_t N, int32_t A, int32_t C, int32_t H, int32_t W,
    int32_t poolingH, int32_t poolingW, int32_t featureStride, int32_t preNmsTop, int32_t nmsMaxOut, float iouThreshold,
    float minBoxSize, float spatialScale, const float* imInfo, const float* anchors, nvinfer1::DataType tScores,
    DLayout_t lScores, const void* scores, nvinfer1::DataType tDeltas, DLayout_t lDeltas, const void* deltas,
    nvinfer1::DataType tFeatureMap, DLayout_t lFeatureMap, const void* featureMap, void* workspace,
    nvinfer1::DataType tRois, void* rois, nvinfer1::DataType tTop, DLayout_t lTop, void* top, size_t deviceSmemSize);

// GENERATE ANCHORS CPU
pluginStatus_t generateAnchors_cpu(
    int32_t numRatios, float* ratios, int32_t numScales, float* scales, int32_t baseSize, float* anchors);

int32_t cropAndResizeInference(cudaStream_t stream, int32_t n, const void* image, const void* rois, int32_t batch_size,
    int32_t input_height, int32_t input_width, int32_t num_boxes, int32_t crop_height, int32_t crop_width, int32_t depth, void* output);

int32_t proposalInference_gpu(cudaStream_t stream, const void* rpn_prob, const void* rpn_regr, int32_t batch_size,
    int32_t input_height, int32_t input_width, int32_t rpn_height, int32_t rpn_width, int32_t MAX_BOX_NUM, int32_t RPN_PRE_NMS_TOP_N,
    float* ANCHOR_SIZES, int32_t anc_size_num, float* ANCHOR_RATIOS, int32_t anc_ratio_num, float rpn_std_scaling,
    int32_t rpn_stride, float bbox_min_size, float nms_iou_threshold, void* workspace, void* output);

size_t _get_workspace_size(int32_t N, int32_t anc_size_num, int32_t anc_ratio_num, int32_t H, int32_t W, int32_t nmsMaxOut);

void  decodeBbox3DLaunch(
    const int32_t batch_size,
    const float *cls_input,
    float *box_input,
    const float *dir_cls_input,
    float *anchors,
    float *anchors_bottom_height,
    float *bndbox_output,
    int32_t *object_counter,
    const float min_x_range,
    const float max_x_range,
    const float min_y_range,
    const float max_y_range,
    const int32_t feature_x_size,
    const int32_t feature_y_size,
    const int32_t num_anchors,
    const int32_t num_classes,
    const int32_t num_box_values,
    const float score_thresh,
    const float dir_offset,
    const float dir_limit_offset,
    const int32_t num_dir_bins,
    cudaStream_t stream = 0);

template <typename Element>
int32_t pillarScatterKernelLaunch(
  int32_t batch_size,
  int32_t max_pillar_num,
  int32_t num_features,
  const Element *pillar_features_data,
  const uint32_t *coords_data,
  const uint32_t *params_data,
  uint32_t featureX, uint32_t featureY,
  Element *spatial_feature_data,
  cudaStream_t stream);

void generateVoxels_launch(
        int32_t batch_size, int32_t max_num_points,
        float *points, uint32_t* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int32_t grid_y_size, int32_t grid_x_size, int32_t num_point_values,
        int32_t max_points_per_voxel,
        uint32_t *mask, float *voxels,
        cudaStream_t stream);

void generateBaseFeatures_launch(
        int32_t batch_size,
        uint32_t *mask, float *voxels,
        int32_t grid_y_size, int32_t grid_x_size,
        uint32_t *pillar_num,
        int32_t max_pillar_num,
        int32_t max_points_per_voxel,
        int32_t num_point_values,
        float *voxel_features,
        uint32_t *voxel_num_points,
        uint32_t *coords,
        cudaStream_t stream);

int32_t generateFeatures_launch(
    int32_t batch_size,
    int32_t dense_pillar_num,
    float* voxel_features,
    uint32_t* voxel_num_points,
    uint32_t* coords,
    uint32_t *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    uint32_t voxel_features_size, uint32_t max_points,
    uint32_t max_voxels, uint32_t num_point_values,
    float* features,
    cudaStream_t stream);


#endif // TRT_RPNLAYER_H
#endif
