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
#include "kernel.h"
#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <cstdio>
#include <math.h>
#include <stdio.h>
#include <vector>

// ROI POOLING FORWARD KERNEL 
template <typename DATA_T, typename ROI_T, bool INFER_ONLY>
__global__ void ROIPoolingForwardKernelAligned(int ROICount,
                                               const ROI_T* rois,
                                               int N, //feature map size
                                               int C, //feature map size
                                               int H, //feature map size
                                               int W, //feature map size
                                               const DATA_T* featureMap,
                                               const int poolingH,
                                               const int poolingW,
                                               const float spatialScale,
                                               DATA_T* top,
                                               int* maxIds)
{
    extern __shared__ float smem[];
    int* rois_shr = (int*) &smem[H * W];
    ROICount = ROICount / N;

    const int batch = blockIdx.x / C;
    const int channel = blockIdx.x % C;

    // load ROIs to shared memory
    for (int j = threadIdx.x; j < ROICount; j += blockDim.x)
    {
        int offset = j << 2;
        float4 roi = reinterpret_cast<float4*>(const_cast<float*>(rois))[batch * ROICount + j];
        // spatialScale = 1.0 / featureStride
        // Convert the coordinates to feature map scale
        rois_shr[offset] = round(roi.x * spatialScale);     //roi_start_w
        rois_shr[offset + 1] = round(roi.y * spatialScale); //roi_start_h
        rois_shr[offset + 2] = round(roi.z * spatialScale); //roi_end_w
        rois_shr[offset + 3] = round(roi.w * spatialScale); //roi_end_h
    }

    // Assumes #CTAs is just enough to cover all channels of all blocks
    const DATA_T* bottom_data_offset = featureMap + blockIdx.x * H * W;

    // load the current channel to the shared memory
    for (int j = threadIdx.x; j < H * W; j += blockDim.x)
    {
        smem[j] = bottom_data_offset[j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < ROICount; j += blockDim.x)
    {
        const int offset = j << 2;
        // Force malformed ROIs to be 1x1
        int roi_width = max(rois_shr[offset + 2] - rois_shr[offset + 0] + 1, 1);
        int roi_height = max(rois_shr[offset + 3] - rois_shr[offset + 1] + 1, 1);
        float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(poolingH);
        float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(poolingW);

        for (int ph = 0; ph < poolingH; ++ph)
        {
            for (int pw = 0; pw < poolingW; ++pw)
            {
                int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
                int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
                int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
                int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

                // Add roi offsets and clip to input boundaries
                // In fact, clipping should be done in the RPN, but just in case...
                hstart = min(max(hstart + rois_shr[offset + 1], 0), H);
                hend = min(max(hend + rois_shr[offset + 1], 0), H);
                wstart = min(max(wstart + rois_shr[offset], 0), W);
                wend = min(max(wend + rois_shr[offset], 0), W);
                bool is_empty = (hend <= hstart) || (wend <= wstart);

                // Define an empty pooling region to be zero
                float maxval = is_empty ? 0 : -FLT_MAX;
                int maxId = -1;
                for (int h = hstart; h < hend; ++h)
                {
                    for (int w = wstart; w < wend; ++w)
                    {
                        int index = h * W + w;
                        if (smem[index] > maxval)
                        {
                            maxval = smem[index];
                            maxId = index;
                        }
                    }
                }
                top[(((batch * ROICount + j) * C + channel) * poolingH + ph) * poolingW + pw] = maxval;
                if (!INFER_ONLY)
                {
                    maxIds[(((batch * ROICount + j) * C + channel) * poolingH + ph) * poolingW + pw] = maxId;
                }
            } //for:pw
        }     //for:ph
    }         //for:j
}

template <typename DATA_T, typename ROI_T, bool INFER_ONLY>
pluginStatus_t ROIPoolingForwardKernelAlignedLauncher(cudaStream_t stream,
                                                     const int R,        // TOTAL number of rois -> ~nmsMaxOut * N
                                                     const int N,        // Batch size
                                                     const int C,        // Channels
                                                     const int H,        // Input feature map H
                                                     const int W,        // Input feature map W
                                                     const int poolingH, // Output feature map H
                                                     const int poolingW, // Output feature map W
                                                     const float spatialScale,
                                                     const void* rois,
                                                     const void* featureMap,
                                                     void* top,
                                                     int* maxIds)
{
    size_t shmemSize = H * W * sizeof(DATA_T) + (R / N) * 4 * sizeof(ROI_T);

    if (shmemSize > 48 * 1024)
    {
        return STATUS_BAD_PARAM;
    }

    // in the aligned version of ROI Pooling R should always be a multiple of N
    assert(R % N == 0);

    ROIPoolingForwardKernelAligned<DATA_T, ROI_T, INFER_ONLY><<<N * C, 256, shmemSize, stream>>>(R,
                                                                                                 (const ROI_T*) rois,
                                                                                                 N, //feature map size
                                                                                                 C, //feature map size
                                                                                                 H, //feature map size
                                                                                                 W, //feature map size
                                                                                                 (const DATA_T*) featureMap,
                                                                                                 poolingH,
                                                                                                 poolingW,
                                                                                                 spatialScale,
                                                                                                 (DATA_T*) top,
                                                                                                 maxIds);
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// ROI POOLING LAUNCH CONFIG 

typedef pluginStatus_t (*roiFwd)(cudaStream_t,
                                const int,   //R, // TOTAL number of rois -> ~nmsMaxOut * N
                                const int,   //N, // Batch size
                                const int,   //C, // Channels
                                const int,   //H, // Input feature map H
                                const int,   //W, // Input feature map W
                                const int,   //poolingH, // Output feature map H
                                const int,   //poolingW, // Output feature map W
                                const float, //spatialScale,
                                const void*, //rois,
                                const void*, //featureMap,
                                void*,       //top
                                int*);       //maxIds);

// struct 
struct roiFwdLaunchConfig
{
    DataType t_rois;
    DataType t_featureMap;
    DLayout_t l_featureMap;
    DataType t_top;
    DLayout_t l_top;
    bool inferOnly;
    roiFwd function;

    roiFwdLaunchConfig(DataType t_rois,
                       DataType t_featureMap,
                       DLayout_t l_featureMap,
                       DataType t_top,
                       DLayout_t l_top,
                       bool inferOnly)
        : t_rois(t_rois)
        , t_featureMap(t_featureMap)
        , l_featureMap(l_featureMap)
        , t_top(t_top)
        , l_top(l_top)
        , inferOnly(inferOnly)
    {
    }

    roiFwdLaunchConfig(DataType t_rois,
                       DataType t_featureMap,
                       DLayout_t l_featureMap,
                       DataType t_top,
                       DLayout_t l_top,
                       bool inferOnly,
                       roiFwd function)
        : t_rois(t_rois)
        , t_featureMap(t_featureMap)
        , l_featureMap(l_featureMap)
        , t_top(t_top)
        , l_top(l_top)
        , inferOnly(inferOnly)
        , function(function)
    {
    }

    bool operator==(const roiFwdLaunchConfig& other)
    {
        return (t_rois == other.t_rois)
            && (t_featureMap == other.t_featureMap)
            && (l_featureMap == other.l_featureMap)
            && (t_top == other.t_top)
            && (l_top == other.l_top)
            && (inferOnly == other.inferOnly);
    }
};

#define FLOAT32 nvinfer1::DataType::kFLOAT
bool roiFwdLCVecInit(std::vector<roiFwdLaunchConfig>& roiFwdLCVec)
{
    roiFwdLCVec.push_back(roiFwdLaunchConfig(FLOAT32,
                                             FLOAT32,
                                             NCHW,
                                             FLOAT32,
                                             NCHW,
                                             true,
                                             ROIPoolingForwardKernelAlignedLauncher<float, float, true>));
    roiFwdLCVec.push_back(roiFwdLaunchConfig(FLOAT32,
                                             FLOAT32,
                                             NCHW,
                                             FLOAT32,
                                             NCHW,
                                             false,
                                             ROIPoolingForwardKernelAlignedLauncher<float, float, false>));
    return true;
}

const std::vector<roiFwdLaunchConfig>& getRoiFwdLCVec()
{
    static std::vector<roiFwdLaunchConfig> roiFwdLCVec;
    static bool roiFwdLCVecI = roiFwdLCVecInit(roiFwdLCVec);

    return roiFwdLCVec;
}


// ROI INFERENCE 
pluginStatus_t roiInference(cudaStream_t stream,
                           const int R,        // TOTAL number of rois -> ~nmsMaxOut * N
                           const int N,        // Batch size
                           const int C,        // Channels
                           const int H,        // Input feature map H
                           const int W,        // Input feature map W
                           const int poolingH, // Output feature map H
                           const int poolingW, // Output feature map W
                           const float spatialScale,
                           const nvinfer1::DataType t_rois,
                           const void* rois,
                           const nvinfer1::DataType t_featureMap,
                           const DLayout_t l_featureMap,
                           const void* featureMap,
                           const nvinfer1::DataType t_top,
                           const DLayout_t l_top,
                           void* top)
{
    if (featureMap == NULL || rois == NULL || top == NULL)
    {
        return STATUS_BAD_PARAM;
    }

    DEBUG_PRINTF("&&&& ROIS %u\n", hash(rois, R * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& FMAP %u\n", hash(featureMap, N * C * H * W * sizeof(float)));

    roiFwdLaunchConfig rflc = roiFwdLaunchConfig(t_rois, t_featureMap, l_featureMap, t_top, l_top, true);
    ASSERT_PARAM(R > 0);

    for (unsigned i = 0; i < getRoiFwdLCVec().size(); i++)
    {
        if (rflc == getRoiFwdLCVec()[i])
        {
            DEBUG_PRINTF("$$$$ ROI KERNEL %d\n", i);
            return getRoiFwdLCVec()[i].function(stream,
                                                R, N, C, H, W, poolingH, poolingW,
                                                spatialScale, rois, featureMap, top, NULL);
        }
    }
    return STATUS_BAD_PARAM;
}
