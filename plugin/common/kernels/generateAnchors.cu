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
#include <cstdio>

pluginStatus_t generateAnchors_cpu(
    int numRatios, float* ratios, int numScales, float* scales, int baseSize, float* anchors)
{
#ifdef DEBUG
    DEBUG_PRINTF("Generating Anchors with:\n");
    DEBUG_PRINTF("Scales:");
    for (int s = 0; s < numScales; ++s)
    {
        DEBUG_PRINTF("%f\t", scales[s]);
    }
    DEBUG_PRINTF("\n");
    DEBUG_PRINTF("Ratios:");
    for (int r = 0; r < numRatios; ++r)
    {
        DEBUG_PRINTF("%f\t", ratios[r]);
    }
    DEBUG_PRINTF("\n");
#endif

    if ((numScales <= 0) || (numRatios <= 0) || (baseSize <= 0))
    {
        return STATUS_BAD_PARAM;
    }

    // Generate parameters for numRatios * numScales general anchor boxes
    for (int r = 0; r < numRatios; ++r)
    {
        for (int s = 0; s < numScales; ++s)
        {
            int id = r * numScales + s;
            float scale = scales[s];
            float ratio = ratios[r];
            float bs = baseSize;
            float ws = round(sqrt((float) (bs * bs) / ratio));
            float hs = round(ws * ratio);
            // Width: bs / sqrt(ratio) * scale
            // Height: bs * sqrt(ratio) * scale
            ws *= scale;
            hs *= scale;

            // x_anchor_ctr
            /*
             * This value should not useful in this implementation of generating numRatios * numScales general anchor boxes.
             * Because the center of anchor box in the original input raw image scale will not be dependent on this.
             */
            anchors[id * 4] = (bs - 1) / 2;
            // y_anchor_ctr
            /*
             * This value should not useful in this implementation of generating numRatios * numScales general anchor boxes.
             * Because the center of anchor box in the original input raw image scale will not be dependent on this.
             */
            anchors[id * 4 + 1] = (bs - 1) / 2;
            // w_anchor
            anchors[id * 4 + 2] = ws;
            // h_anchor
            anchors[id * 4 + 3] = hs;
        }
    }
    return STATUS_SUCCESS;
}

pluginStatus_t generateAnchors(cudaStream_t stream,
                              int numRatios,
                              float* ratios,
                              int numScales,
                              float* scales,
                              int baseSize,
                              float* anchors)
{
    // Each anchor box has 4 parameters
    int ac = numRatios * numScales * 4;
    float* anchors_cpu;
    CSC(cudaMallocHost((void**) &anchors_cpu, sizeof(float) * ac), STATUS_FAILURE);
    pluginStatus_t status = generateAnchors_cpu(numRatios, ratios, numScales, scales, baseSize, anchors_cpu);
    CSC(cudaMemcpyAsync(anchors, anchors_cpu, sizeof(float) * ac, cudaMemcpyHostToDevice, stream), STATUS_FAILURE);
    CSC(cudaFreeHost(anchors_cpu), STATUS_FAILURE);
    return status;
}
