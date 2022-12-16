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
#include <algorithm>
#include <array>
#include <math.h>
#include <stdio.h>

using namespace nvinfer1;
using std::max;
using std::min;

// BBD2P KERNEL
template <typename T_DELTAS,
          DLayout_t L_DELTAS,
          typename TV_PROPOSALS,
          DLayout_t L_PROPOSALS,
          typename T_FGSCORES,
          DLayout_t L_FGSCORES>
__global__ void bboxDeltas2Proposals_kernel(
    int N,
    int A,
    int H,
    int W,
    const float* __restrict__ anchors,
    const float* __restrict__ imInfo,
    int featureStride,
    float minSize,
    const T_DELTAS* __restrict__ deltas,
    TV_PROPOSALS* __restrict__ proposals,
    T_FGSCORES* __restrict__ scores)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N * A * H * W)
    { // TODO this can be a loop.
        // Find out the index of bounding box for the output
        int cnt = tid;
        // width index
        int w = cnt % W;
        cnt = cnt / W;
        // height index
        int h = cnt % H;
        cnt = cnt / H;
        // anchor box index
        int a = cnt % A;
        cnt = cnt / A;
        // batch index
        int n = cnt;
        int hw = h * W + w;

        // Get the height and width of the original input image
        float imHeight = imInfo[3 * n];
        float imWidth = imInfo[3 * n + 1];

        // Point to the right anchor box
        float4 anchor = ((float4*) anchors)[a];
        // Get anchor box coordinates
        float a_ctr_x = anchor.x;
        float a_ctr_y = anchor.y;
        float a_w = anchor.z;
        float a_h = anchor.w;

        // NCHW format
        // Find out the starting position of the bounding box in the input (predicted bounding box offsets)
        int id = ((tid - hw) * 4) + hw;
        T_DELTAS dx;
        T_DELTAS dy;
        T_DELTAS dw;
        T_DELTAS dh;
        if (L_DELTAS == NCHW)
        {
            // The offsets between adjacent coordinates on linear memory is H * W
            dx = deltas[id];
            dy = deltas[id + 1 * H * W];
            dw = deltas[id + 2 * H * W];
            dh = deltas[id + 3 * H * W];
        }
        // NC4HW format
        else if (L_DELTAS == NC4HW)
        {
            dx = deltas[tid * 4 + 0];
            dy = deltas[tid * 4 + 1];
            dw = deltas[tid * 4 + 2];
            dh = deltas[tid * 4 + 3];
        }
        /*
         * Calculate the coordinates of decoded bounding box on the original input image scale
         * Only works if param.minBoxSize == param.featureStride
         */
        float ctr_x = a_ctr_x + w * featureStride;
        float ctr_y = a_ctr_y + h * featureStride;
        // float ctr_x = (w + 0.5) * featureStride;
        // float ctr_y = (h + 0.5) * featureStride;

        /*
         * Decode the predicted bounding box
         * The decoded bounding boxes has coordinates of [x_topleft, y_topleft, x_bottomright, y_bottomright]
         * The units are in pixels
         */
        ctr_x = ctr_x + dx * a_w;
        ctr_y = ctr_y + dy * a_h;
        float b_w = __expf(dw) * a_w;
        float b_h = __expf(dh) * a_h;
        float bx = ctr_x - (b_w / 2);
        float by = ctr_y - (b_h / 2);
        float bz = ctr_x + (b_w / 2);
        float bw = ctr_y + (b_h / 2);

        TV_PROPOSALS bbox;
        // Make sure that the decoded bouding box go outside of the original input image
        bbox.x = fminf(fmaxf(bx, 0.0f), imWidth - 1.0f);
        bbox.y = fminf(fmaxf(by, 0.0f), imHeight - 1.0f);
        bbox.z = fminf(fmaxf(bz, 0.0f), imWidth - 1.0f);
        bbox.w = fminf(fmaxf(bw, 0.0f), imHeight - 1.0f);

        // Put the decoded bounding box information to the outputs
        if (L_PROPOSALS == NC4HW)
        {
            proposals[tid] = bbox;
        }

        int ininf = 0xff800000;
        float ninf = *(float*) &ininf;
        // minBoxSize at the original input image scale
        float scaledMinSize = minSize * imInfo[3 * n + 2];
        // Set the objectness score to -inf if the predicted bounding box has edgth length less than the minimal box size expected.
        if (bbox.z - bbox.x + 1 < scaledMinSize || bbox.w - bbox.y + 1 < scaledMinSize)
        {
            if (L_FGSCORES == NCHW)
                scores[tid] = ninf;
        }
    }
}

// BBD2P KERNEL LAUNCHER
template <typename T_DELTAS,
          DLayout_t L_DELTAS,
          typename TV_PROPOSALS,
          DLayout_t L_PROPOSALS,
          typename T_FGSCORES,
          DLayout_t L_FGSCORES>
pluginStatus_t bboxDeltas2Proposals_gpu(cudaStream_t stream,
                                       int N,
                                       int A,
                                       int H,
                                       int W,
                                       const float* imInfo,
                                       int featureStride,
                                       float minBoxSize,
                                       const float* anchors,
                                       const void* deltas,
                                       void* propos,
                                       void* scores)
{
    const int BS = 32;
    const int GS = ((N * A * H * W) + BS - 1) / BS;

    bboxDeltas2Proposals_kernel<T_DELTAS, L_DELTAS, TV_PROPOSALS, L_PROPOSALS, T_FGSCORES, L_FGSCORES><<<GS, BS, 0, stream>>>(N, A, H, W,
                                                                                                                              anchors,
                                                                                                                              imInfo,
                                                                                                                              featureStride,
                                                                                                                              minBoxSize,
                                                                                                                              (T_DELTAS*) deltas,
                                                                                                                              (TV_PROPOSALS*) propos,
                                                                                                                              (T_FGSCORES*) scores);

    DEBUG_PRINTF("&&&& [bboxD2P] POST LAUNCH\n");
    DEBUG_PRINTF("&&&& [bboxD2P] PROPOS %u\n", hash(propos, N * A * H * W * 4 * sizeof(float)));

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// BBD2P LAUNCH CONFIG {{{
typedef pluginStatus_t (*bd2pFun)(cudaStream_t,
                                 int,
                                 int,
                                 int,
                                 int,
                                 const float*,
                                 int,
                                 float,
                                 const float*,
                                 const void*,
                                 void*,
                                 void*);

struct bd2pLaunchConfig
{
    DataType t_deltas;
    DLayout_t l_deltas;
    DataType t_proposals;
    DLayout_t l_proposals;
    DataType t_scores;
    DLayout_t l_scores;
    bd2pFun function;

    bd2pLaunchConfig(DataType t_deltas, DLayout_t l_deltas, DataType t_proposals, DLayout_t l_proposals,
        DataType t_scores, DLayout_t l_scores)
        : t_deltas(t_deltas)
        , l_deltas(l_deltas)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_scores(t_scores)
        , l_scores(l_scores)
        , function(nullptr)
    {
    }

    bd2pLaunchConfig(DataType t_deltas, DLayout_t l_deltas, DataType t_proposals, DLayout_t l_proposals, DataType t_scores, DLayout_t l_scores, bd2pFun function)
        : t_deltas(t_deltas)
        , l_deltas(l_deltas)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_scores(t_scores)
        , l_scores(l_scores)
        , function(function)
    {
    }

    bool operator==(const bd2pLaunchConfig& other)
    {
        return t_deltas == other.t_deltas && l_deltas == other.l_deltas && t_proposals == other.t_proposals && l_proposals == other.l_proposals && t_scores == other.t_scores && l_scores == other.l_scores;
    }
};

#define FLOAT32 nvinfer1::DataType::kFLOAT
static std::array<bd2pLaunchConfig, 2> bd2pLCOptions = {
    bd2pLaunchConfig(FLOAT32, NC4HW, FLOAT32, NC4HW, FLOAT32, NCHW, bboxDeltas2Proposals_gpu<float, NC4HW, float4, NC4HW, float, NCHW>),
    bd2pLaunchConfig(FLOAT32, NCHW, FLOAT32, NC4HW, FLOAT32, NCHW, bboxDeltas2Proposals_gpu<float, NCHW, float4, NC4HW, float, NCHW>)};

// BBD2P
pluginStatus_t bboxDeltas2Proposals(cudaStream_t stream,
                                   const int N,
                                   const int A,
                                   const int H,
                                   const int W,
                                   const int featureStride,
                                   const float minBoxSize,
                                   const float* imInfo,
                                   const float* anchors,
                                   const DataType t_deltas,
                                   const DLayout_t l_deltas,
                                   const void* deltas,
                                   const DataType t_proposals,
                                   const DLayout_t l_proposals,
                                   void* proposals,
                                   const DataType t_scores,
                                   const DLayout_t l_scores,
                                   void* scores)
{
    bd2pLaunchConfig lc = bd2pLaunchConfig(t_deltas, l_deltas, t_proposals, l_proposals, t_scores, l_scores);
    for (unsigned i = 0; i < bd2pLCOptions.size(); i++)
    {
        if (lc == bd2pLCOptions[i])
        {
            DEBUG_PRINTF("BBD2P kernel %d\n", i);
            return bd2pLCOptions[i].function(stream,
                                          N, A, H, W,
                                          imInfo,
                                          featureStride,
                                          minBoxSize,
                                          anchors,
                                          deltas,
                                          proposals,
                                          scores);
        }
    }
    return STATUS_BAD_PARAM;
}
