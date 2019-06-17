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
#ifndef TRT_KERNEL_H
#define TRT_KERNEL_H

#include "cublas_v2.h"
#include "plugin.h"
#include <cassert>
#include <cstdio>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
#define DEBUG_ENABLE 0

typedef enum
{
    NCHW = 0,
    NC4HW = 1
} DLayout_t;

pluginStatus_t allClassNMS(cudaStream_t stream, int num, int num_classes, int num_preds_per_class, int top_k,
    float nms_threshold, bool share_location, bool isNormalized, DataType DT_SCORE, DataType DT_BBOX, void* bbox_data,
    void* beforeNMS_scores, void* beforeNMS_index_array, void* afterNMS_scores, void* afterNMS_index_array,
    bool flipXY = false);

pluginStatus_t detectionInference(cudaStream_t stream, int N, int C1, int C2, bool shareLocation,
    bool varianceEncodedInTarget, int backgroundLabelId, int numPredsPerClass, int numClasses, int topK, int keepTopK,
    float confidenceThreshold, float nmsThreshold, CodeTypeSSD codeType, DataType DT_BBOX, const void* locData,
    const void* priorData, DataType DT_SCORE, const void* confData, void* keepCount, void* topDetections,
    void* workspace, bool isNormalized = true, bool confSigmoid = false

);

pluginStatus_t gatherTopDetections(cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass,
    int numClasses, int topK, int keepTopK, DataType DT_BBOX, DataType DT_SCORE, const void* indices,
    const void* scores, const void* bboxData, void* keepCount, void* topDetections);

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX);

size_t sortScoresPerClassWorkspaceSize(int num, int num_classes, int num_preds_per_class, DataType DT_CONF);

size_t sortScoresPerImageWorkspaceSize(int num_images, int num_items_per_image, DataType DT_SCORE);

pluginStatus_t sortScoresPerImage(cudaStream_t stream, int num_images, int num_items_per_image, DataType DT_SCORE,
    void* unsorted_scores, void* unsorted_bbox_indices, void* sorted_scores, void* sorted_bbox_indices,
    void* workspace);

pluginStatus_t sortScoresPerClass(cudaStream_t stream, int num, int num_classes, int num_preds_per_class,
    int background_label_id, float confidence_threshold, DataType DT_SCORE, void* conf_scores_gpu,
    void* index_array_gpu, void* workspace);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

const char* cublasGetErrorString(cublasStatus_t error);

pluginStatus_t permuteData(cudaStream_t stream, int nthreads, int num_classes, int num_data, int num_dim,
    DataType DT_DATA, bool confSigmoid, const void* data, void* new_data);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

pluginStatus_t decodeBBoxes(cudaStream_t stream, int nthreads, CodeTypeSSD code_type, bool variance_encoded_in_target,
    int num_priors, bool share_location, int num_loc_classes, int background_label_id, bool clip_bbox, DataType DT_BBOX,
    const void* loc_data, const void* prior_data, void* bbox_data);

size_t normalizePluginWorkspaceSize(bool acrossSpatial, int C, int H, int W);

pluginStatus_t normalizeInference(cudaStream_t stream, cublasHandle_t handle, bool acrossSpatial, bool channelShared,
    int N, int C, int H, int W, float eps, const void* scale, const void* inputData, void* outputData, void* workspace);

pluginStatus_t priorBoxInference(cudaStream_t stream, PriorBoxParameters param, int H, int W, int numPriors,
    int numAspectRatios, const void* minSize, const void* maxSize, const void* aspectRatios, void* outputData);

pluginStatus_t reorgInference(
    cudaStream_t stream, int batch, int C, int H, int W, int stride, const void* input, void* output);

pluginStatus_t anchorGridInference(cudaStream_t stream, GridAnchorParameters param, int numAspectRatios,
    const void* aspectRatios, const void* scales, void* outputData);

pluginStatus_t regionInference(cudaStream_t stream, int batch, int C, int H, int W, int num, int coords, int classes,
    bool hasSoftmaxTree, const nvinfer1::plugin::softmaxTree* smTree, const void* input, void* output);

// GENERATE ANCHORS
// For now it takes host pointers - ratios and scales but
// in GPU MODE anchors should be device pointer
pluginStatus_t generateAnchors(cudaStream_t stream,
    int numRatios,   // number of ratios
    float* ratios,   // ratio array
    int numScales,   // number of scales
    float* scales,   // scale array
    int baseSize,    // size of the base anchor (baseSize x baseSize)
    float* anchors); // output anchors (numRatios x numScales)

// BBD2P
pluginStatus_t bboxDeltas2Proposals(cudaStream_t stream,
    int N,                // batch size
    int A,                // number of anchors
    int H,                // last feature map H
    int W,                // last feature map W
    int featureStride,    // feature stride
    float minBoxSize,     // minimum allowed box size before scaling
    const float* imInfo,  // image info (nrows, ncols, image scale)
    const float* anchors, // input anchors
    DataType tDeltas,     // type of input deltas
    DLayout_t lDeltas,    // layout of input deltas
    const void* deltas,   // input deltas
    DataType tProposals,  // type of output proposals
    DLayout_t lProposals, // layout of output proposals
    void* proposals,      // output proposals
    DataType tScores,     // type of output scores
    DLayout_t lScores,    // layout of output scores
    void* scores);        // output scores (the score associated with too small box will be set to -inf)

// NMS
pluginStatus_t nms(cudaStream_t stream,
    int N,                 // batch size
    int R,                 // number of ROIs (region of interest) per image
    int preNmsTop,         // number of proposals before applying NMS
    int nmsMaxOut,         // number of remaining proposals after applying NMS
    float iouThreshold,    // IoU threshold
    DataType tFgScores,    // type of foreground scores
    DLayout_t lFgScores,   // layout of foreground scores
    void* fgScores,        // foreground scores
    DataType tProposals,   // type of proposals
    DLayout_t lProposals,  // layout of proposals
    const void* proposals, // proposals
    void* workspace,       // workspace
    DataType tRois,        // type of ROIs
    void* rois);           // ROIs

// WORKSPACE SIZES
size_t proposalsForwardNMSWorkspaceSize(int N, int A, int H, int W, int nmsMaxOut);

size_t proposalsForwardBboxWorkspaceSize(int N, int A, int H, int W);

size_t proposalForwardFgScoresWorkspaceSize(int N, int A, int H, int W);

size_t proposalsInferenceWorkspaceSize(int N, int A, int H, int W, int nmsMaxOut);

size_t RPROIInferenceFusedWorkspaceSize(int N, int A, int H, int W, int nmsMaxOut);

// PROPOSALS INFERENCE
pluginStatus_t proposalsInference(cudaStream_t stream, int N, int A, int H, int W, int featureStride, int preNmsTop,
    int nmsMaxOut, float iouThreshold, float minBoxSize, const float* imInfo, const float* anchors, DataType tScores,
    DLayout_t lScores, const void* scores, DataType tDeltas, DLayout_t lDeltas, const void* deltas, void* workspace,
    DataType tRois, void* rois);

// EXTRACT FG SCORES
pluginStatus_t extractFgScores(cudaStream_t stream, int N, int A, int H, int W, DataType tScores, DLayout_t lScores,
    const void* scores, DataType tFgScores, DLayout_t lFgScores, void* fgScores);

// ROI INFERENCE
pluginStatus_t roiInference(cudaStream_t stream,
    const int R,        // TOTAL number of rois -> ~nmsMaxOut * N
    const int N,        // Batch size
    const int C,        // Channels
    const int H,        // Input feature map H
    const int W,        // Input feature map W
    const int poolingH, // Output feature map H
    const int poolingW, // Output feature map W
    const float spatialScale, const DataType tRois, const void* rois, const DataType tFeatureMap,
    const DLayout_t lFeatureMap, const void* featureMap, const DataType tTop, const DLayout_t lTop, void* top);

// ROI FORWARD
pluginStatus_t roiForward(cudaStream_t stream,
    int R,        // TOTAL number of rois -> ~nmsMaxOut * N
    int N,        // Batch size
    int C,        // Channels
    int H,        // Input feature map H
    int W,        // Input feature map W
    int poolingH, // Output feature map H
    int poolingW, // Output feature map W
    float spatialScale, DataType tRois, const void* rois, DataType tFeatureMap, DLayout_t lFeatureMap,
    const void* featureMap, DataType tTop, DLayout_t lTop, void* top, int* maxIds);

// RP ROI Fused INFERENCE
pluginStatus_t RPROIInferenceFused(cudaStream_t stream, int N, int A, int C, int H, int W, int poolingH, int poolingW,
    int featureStride, int preNmsTop, int nmsMaxOut, float iouThreshold, float minBoxSize, float spatialScale,
    const float* imInfo, const float* anchors, DataType tScores, DLayout_t lScores, const void* scores,
    DataType tDeltas, DLayout_t lDeltas, const void* deltas, DataType tFeatureMap, DLayout_t lFeatureMap,
    const void* featureMap, void* workspace, DataType tRois, void* rois, DataType tTop, DLayout_t lTop, void* top);

// GENERATE ANCHORS CPU
pluginStatus_t generateAnchors_cpu(
    int numRatios, float* ratios, int numScales, float* scales, int baseSize, float* anchors);

int cropAndResizeInference(cudaStream_t stream, int n, const void* image, const void* rois, int batch_size,
    int input_height, int input_width, int num_boxes, int crop_height, int crop_width, int depth, void* output);

int proposalInference_gpu(cudaStream_t stream, const void* rpn_prob, const void* rpn_regr, int batch_size,
    int input_height, int input_width, int rpn_height, int rpn_width, int MAX_BOX_NUM, int RPN_PRE_NMS_TOP_N,
    float* ANCHOR_SIZES, int anc_size_num, float* ANCHOR_RATIOS, int anc_ratio_num, float rpn_std_scaling,
    int rpn_stride, float bbox_min_size, float nms_iou_threshold, void* workspace, void* output);

size_t _get_workspace_size(int N, int anc_size_num, int anc_ratio_num, int H, int W, int nmsMaxOut);

#endif
