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
#include "common/kernel.h"
using namespace nvinfer1;
using namespace nvinfer1::plugin;
pluginStatus_t RPROIInferenceFused(cudaStream_t stream, const int N, const int A, const int C, const int H, const int W,
    const int poolingH, const int poolingW, const int featureStride, const int preNmsTop, const int nmsMaxOut,
    const float iouThreshold, const float minBoxSize, const float spatialScale, const float* imInfo,
    const float* anchors, const DataType t_scores, const DLayout_t l_scores, const void* scores,
    const DataType t_deltas, const DLayout_t l_deltas, const void* deltas, const DataType t_featureMap,
    const DLayout_t l_featureMap, const void* featureMap, void* workspaces, const DataType t_rois, void* rois,
    const DataType t_top, const DLayout_t l_top, void* top, size_t deviceSmemSize)
{
    if (imInfo == NULL || anchors == NULL || scores == NULL || deltas == NULL || featureMap == NULL
        || workspaces == NULL || rois == NULL || top == NULL)
    {
        return STATUS_BAD_PARAM;
    }

    pluginStatus_t status;
    // Region proposal inference
    // Getting the region of interests (ROIs) bounding box coordinates from region proposals using non maximum suppression (NMS)
    status = proposalsInference(stream,
                                N,
                                A,
                                H,
                                W,
                                featureStride,
                                preNmsTop,
                                nmsMaxOut,
                                iouThreshold,
                                minBoxSize,
                                imInfo,
                                anchors,
                                t_scores,
                                l_scores,
                                scores,
                                t_deltas,
                                l_deltas,
                                deltas,
                                workspaces,
                                t_rois,
                                rois);
    ASSERT_FAILURE(status == STATUS_SUCCESS);
    // ROI inference
    // ROI pooling for ROIs
    status = roiInference(stream,
                          N * nmsMaxOut, // TOTAL number of rois -> ~nmsMaxOut * N
                          N,             // Batch size
                          C,             // Channels
                          H,             // Input feature map H
                          W,             // Input feature map W
                          poolingH,      // Output feature map H
                          poolingW,      // Output feature map W
                          spatialScale,
                          t_rois,
                          rois,
                          t_featureMap,
                          l_featureMap,
                          featureMap,
                          t_top,
                          l_top,
                          top,
                          deviceSmemSize);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

size_t RPROIInferenceFusedWorkspaceSize(int N,
                                        int A,
                                        int H,
                                        int W,
                                        int nmsMaxOut)
{
    return proposalsInferenceWorkspaceSize(N, A, H, W, nmsMaxOut);
}
