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
#include "common/bboxUtils.h"
#include "common/kernel.h"

using namespace nvinfer1;
// PROPOSALS INFERENCE
pluginStatus_t proposalsInference(cudaStream_t stream, const int N, const int A, const int H, const int W,
    const int featureStride, const int preNmsTop, const int nmsMaxOut, const float iouThreshold, const float minBoxSize,
    const float* imInfo, const float* anchors, const DataType t_scores, const DLayout_t l_scores, const void* scores,
    const DataType t_deltas, const DLayout_t l_deltas, const void* deltas, void* workspace, const DataType t_rois,
    void* rois)
{
    /*
     * N: batch size
     * A: number of anchor boxes per grid cell on feature map
     * H: height of feature map
     * W: width of feature map
     */
    if (imInfo == NULL || anchors == NULL || scores == NULL || deltas == NULL || workspace == NULL || rois == NULL)
    {
        return STATUS_BAD_PARAM;
    }

    DEBUG_PRINTF("&&&& IM INFO %u\n", hash(imInfo, N * 3 * sizeof(float)));
    // anchors: anchor boxes
    /*
     * The following line of code looks somewhat incorrect because it sounds like we always have 9 fixed anchor boxes.
     * The "corrected" implementation should be
     * DEBUG_PRINTF("&&&& ANCHORS %u\n", A * 4 * sizeof(float)));
     */
    DEBUG_PRINTF("&&&& ANCHORS %u\n", hash(anchors, 9 * 4 * sizeof(float)));
    // scores: objectness of each predicted bounding boxes
    // 2: softmax, instead of sigmoid, was used for binary objectness classifcation in Faster R-CNN
    DEBUG_PRINTF("&&&& SCORES  %u\n", hash(scores, N * A * 2 * H * W * sizeof(float)));
    // deltas: predicted bounding box offsets
    DEBUG_PRINTF("&&&& DELTAS  %u\n", hash(deltas, N * A * 4 * H * W * sizeof(float)));

    size_t nmsWorkspaceSize = proposalsForwardNMSWorkspaceSize(N, A, H, W, nmsMaxOut);

    void* nmsWorkspace = workspace;

    size_t proposalsSize = proposalsForwardBboxWorkspaceSize(N, A, H, W);
    const DataType t_proposals = nvinfer1::DataType::kFLOAT;
    const DLayout_t l_proposals = NC4HW;
    void* proposals = nextWorkspacePtr((int8_t*) nmsWorkspace, nmsWorkspaceSize);

    const DataType t_fgScores = t_scores;
    const DLayout_t l_fgScores = NCHW;
    void* fgScores = nextWorkspacePtr((int8_t*) proposals, proposalsSize);

    pluginStatus_t status;

    /*
     * Only the second probability value of the objectness (probability of being a object) from the scores will be extracted.
     * Because the first probability (probability of not being a object) value is redundant.
     */
    status = extractFgScores(stream,
                             N, A, H, W,
                             t_scores, l_scores, scores,
                             t_fgScores, l_fgScores, fgScores);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    DEBUG_PRINTF("&&&& FG SCORES %u\n", hash((void*) fgScores, N * A * H * W * sizeof(float)));
    DEBUG_PRINTF("&&&& DELTAS %u\n", hash((void*) proposals, N * A * H * W * 4 * sizeof(float)));

    /*
     * Decode predicted bounding boxes.
     * Decoded predicted bounding boxes were at the raw input image scale.
     */
    status = bboxDeltas2Proposals(stream,
                                  N, A, H, W,
                                  featureStride,
                                  minBoxSize,
                                  imInfo,
                                  anchors,
                                  t_deltas, l_deltas, deltas,
                                  t_proposals, l_proposals, proposals,
                                  t_fgScores, l_fgScores, fgScores);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    DEBUG_PRINTF("&&&& PROPOSALS %u\n", hash((void*) proposals, N * A * H * W * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& FG SCORES %u\n", hash((void*) fgScores, N * A * H * W * sizeof(float)));

    /*
     * Non maximum suppression using objectness scores to get the most representative bounding boxes (ROIs).
     * The rois were at the feature map scale.
     */
    status = nms(stream,
                 N,
                 A * H * W,
                 preNmsTop,
                 nmsMaxOut,
                 iouThreshold,
                 t_fgScores, l_fgScores, fgScores,
                 t_proposals, l_proposals, proposals,
                 nmsWorkspace,
                 t_rois, rois);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    DEBUG_PRINTF("&&&& ROIS %u\n", hash((void*) rois, N * nmsMaxOut * 4 * sizeof(float)));
    return STATUS_SUCCESS;
}

// WORKSPACE SIZES
size_t proposalsForwardNMSWorkspaceSize(int N,
                                        int A,
                                        int H,
                                        int W,
                                        int nmsMaxOut)
{
    return N * A * H * W * 5 * 5 * sizeof(float) + (1 << 22);
}

size_t proposalsForwardBboxWorkspaceSize(int N,
                                         int A,
                                         int H,
                                         int W)
{
    return N * A * H * W * 4 * sizeof(float);
}
size_t proposalForwardFgScoresWorkspaceSize(int N,
                                            int A,
                                            int H,
                                            int W)
{
    return N * A * H * W * sizeof(float);
}

size_t proposalsInferenceWorkspaceSize(int N,
                                       int A,
                                       int H,
                                       int W,
                                       int nmsMaxOut)
{
    size_t wss[3];
    wss[0] = proposalsForwardNMSWorkspaceSize(N, A, H, W, nmsMaxOut);
    wss[1] = proposalsForwardBboxWorkspaceSize(N, A, H, W);
    wss[2] = proposalForwardFgScoresWorkspaceSize(N, A, H, W);
    return calculateTotalWorkspaceSize(wss, 3);
}
