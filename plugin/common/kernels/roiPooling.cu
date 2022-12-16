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
#include <assert.h>
#include <cfloat>
#include <cstdio>
#include <math.h>
#include <stdio.h>

using namespace nvinfer1;
// This macro is to control shared memory usage. If set to 1, kernel loads the whole feature map
// into shared memory for reuse; If set to 0, kernel loads data from global memory directly.
// Roi pooling performance is data dependent. You can test which value is better to your data.
// If all bboxes are very small, 0 is recommended, otherwise, shared memory will load many unused
// data; If bboxes have many overlaps, 1 is recommended to avoid duplicate loads.
// 1 requires larger shared memory size. It may fail if it is larger than CUDA allowed per-block
// shared memory size upper bound. Then you have to use 0.
#define ROIPOOLING_FEATURE_MAP_USE_SHMEM 1

template <typename T>
__device__ T getMax();

template <>
__device__ __forceinline__ int8_t getMax<int8_t>()
{
    return INT8_MAX;
}

template <>
__device__ __forceinline__ float getMax<float>()
{
    return FLT_MAX;
}

// ROI POOLING FORWARD KERNEL
template <typename DATA_T, typename ROI_T, bool INFER_ONLY, bool FM_IN_SMEM>
__global__ void ROIPoolingForwardKernelAligned(int32_t ROICount, const ROI_T* rois,
    int32_t N, // feature map size
    int32_t C, // feature map size
    int32_t H, // feature map size
    int32_t W, // feature map size
    const DATA_T* featureMap, const int32_t poolingH, const int32_t poolingW, const float spatialScale, DATA_T* top,
    int32_t* maxIds, int32_t fmapStep)
{
    extern __shared__ float smem[];
    DATA_T* feature_shr = (DATA_T*) &smem[0];
    int* rois_shr = nullptr;
    if (FM_IN_SMEM)
    {
        rois_shr = (int*) &feature_shr[H * W];
    }
    else
    {
        rois_shr = (int*) &feature_shr[0];
        feature_shr = nullptr;
    }

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
        rois_shr[offset + 2] = round(roi.z * spatialScale) - round(roi.x * spatialScale); //roi_length_w
        rois_shr[offset + 3] = round(roi.w * spatialScale) - round(roi.y * spatialScale); // roi_length_h
    }

    // NC/xHW
    int fmapOffset = blockIdx.x / fmapStep * H * W * fmapStep + blockIdx.x % fmapStep;

    // Assumes #CTAs is just enough to cover all channels of all blocks
    const DATA_T* bottom_data_offset = featureMap + fmapOffset;
    if (FM_IN_SMEM)
    {
        // load the current channel to the shared memory
        for (int j = threadIdx.x; j < H * W; j += blockDim.x)
        {
            feature_shr[j] = bottom_data_offset[j * fmapStep];
        }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < ROICount; j += blockDim.x)
    {
        const int offset = j << 2;
        // Force malformed ROIs to be 1x1
        int roi_start_w = rois_shr[offset];
        int roi_start_h = rois_shr[offset + 1];
        int roi_width = max(rois_shr[offset + 2] + 1, 1);
        int roi_height = max(rois_shr[offset + 3] + 1, 1);
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
                hstart = min(max(hstart + roi_start_h, 0), H);
                hend = min(max(hend + roi_start_h, 0), H);
                wstart = min(max(wstart + roi_start_w, 0), W);
                wend = min(max(wend + roi_start_w, 0), W);
                bool is_empty = (hend <= hstart) || (wend <= wstart);

                // Define an empty pooling region to be zero
                DATA_T maxval = is_empty ? 0 : -getMax<DATA_T>();
                int maxId = -1;
                DATA_T data = 0;
                for (int h = hstart; h < hend; ++h)
                {
                    for (int w = wstart; w < wend; ++w)
                    {
                        int index = h * W + w;
                        if (FM_IN_SMEM)
                        {
                            data = feature_shr[index];
                        }
                        else
                        {
                            data = bottom_data_offset[index * fmapStep];
                        }
                        if (data > maxval)
                        {
                            maxval = data;
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
    }         // for:j
}

template <typename DATA_T, DLayout_t DATA_L, typename ROI_T, bool INFER_ONLY>
pluginStatus_t ROIPoolingForwardKernelAlignedLauncher(cudaStream_t stream,
    const int R,        // TOTAL number of rois -> ~nmsMaxOut * N
    const int N,        // Batch size
    const int C,        // Channels
    const int H,        // Input feature map H
    const int W,        // Input feature map W
    const int poolingH, // Output feature map H
    const int poolingW, // Output feature map W
    const float spatialScale, const void* rois, const void* featureMap, void* top, int* maxIds, size_t deviceSmemSize)
{
    size_t roiShmemSize = (R / N) * 4 * sizeof(ROI_T);

#if ROIPOOLING_FEATURE_MAP_USE_SHMEM
    size_t shmemSize = H * W * sizeof(DATA_T) + roiShmemSize;
    const bool fmap_in_shmem = true;
#else
    size_t shmemSize = roiShmemSize;
    const bool fmap_in_shmem = false;
#endif

    if (shmemSize > deviceSmemSize)
    {
        return STATUS_BAD_PARAM;
    }

    // in the aligned version of ROI Pooling R should always be a multiple of N
    PLUGIN_ASSERT(R % N == 0);

    // NC/xHW
    int32_t fmapStep = 1;
    switch(DATA_L)
    {
    case NCHW: fmapStep = 1; break;
    case NC4HW:
        fmapStep = 4;
        PLUGIN_ASSERT((N * C) % 4 == 0);
        break;
    case NC32HW:
        fmapStep = 32;
        PLUGIN_ASSERT((N * C) % 32 == 0);
        break;
    default: PLUGIN_ASSERT(false);
    }

    if (shmemSize > 48 * 1024)
    {
        PLUGIN_CHECK(cudaFuncSetAttribute(&ROIPoolingForwardKernelAligned<DATA_T, ROI_T, INFER_ONLY, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmemSize));
    }
    ROIPoolingForwardKernelAligned<DATA_T, ROI_T, INFER_ONLY, fmap_in_shmem><<<N * C, 256, shmemSize, stream>>>(R / N,
        (const ROI_T*) rois,
        N, // feature map size
        C, // feature map size
        H, // feature map size
        W, // feature map size
        (const DATA_T*) featureMap, poolingH, poolingW, spatialScale, (DATA_T*) top, maxIds, fmapStep);

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
                                int*,        //maxIds
                                size_t);     //device shmem size

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

    roiFwdLaunchConfig(
        DataType t_rois, DataType t_featureMap, DLayout_t l_featureMap, DataType t_top, DLayout_t l_top, bool inferOnly)
        : t_rois(t_rois)
        , t_featureMap(t_featureMap)
        , l_featureMap(l_featureMap)
        , t_top(t_top)
        , l_top(l_top)
        , inferOnly(inferOnly)
        , function(nullptr)
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
#define INT8 nvinfer1::DataType::kINT8
static std::array<roiFwdLaunchConfig, 6> roiFwdLCOptions = {
    roiFwdLaunchConfig(FLOAT32, FLOAT32, NCHW, FLOAT32, NCHW, true, ROIPoolingForwardKernelAlignedLauncher<float, NCHW, float, true>),
    roiFwdLaunchConfig(FLOAT32, FLOAT32, NCHW, FLOAT32, NCHW, false, ROIPoolingForwardKernelAlignedLauncher<float, NCHW, float, false>),
    roiFwdLaunchConfig(FLOAT32, INT8, NCHW, INT8, NCHW, true, ROIPoolingForwardKernelAlignedLauncher<int8_t, NCHW, float, true>),
    roiFwdLaunchConfig(FLOAT32, INT8, NC4HW, INT8, NCHW, true, ROIPoolingForwardKernelAlignedLauncher<int8_t, NC4HW, float, true>),
    roiFwdLaunchConfig(FLOAT32, INT8, NC32HW, INT8, NCHW, true, ROIPoolingForwardKernelAlignedLauncher<int8_t, NC32HW, float, true>),
    roiFwdLaunchConfig(FLOAT32, FLOAT32, NC4HW, FLOAT32, NCHW, true, ROIPoolingForwardKernelAlignedLauncher<float, NC4HW, float, true>)};

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
                           void* top,
                           size_t deviceSmemSize)
{
    if (featureMap == NULL || rois == NULL || top == NULL)
    {
        return STATUS_BAD_PARAM;
    }

    DEBUG_PRINTF("&&&& ROIS %u\n", hash(rois, R * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& FMAP %u\n", hash(featureMap, N * C * H * W * sizeof(float)));

    roiFwdLaunchConfig rflc = roiFwdLaunchConfig(t_rois, t_featureMap, l_featureMap, t_top, l_top, true);
    ASSERT_PARAM(R > 0);

    for (unsigned i = 0; i < roiFwdLCOptions.size(); i++)
    {
        if (rflc == roiFwdLCOptions[i])
        {
            DEBUG_PRINTF("$$$$ ROI KERNEL %d\n", i);
            return roiFwdLCOptions[i].function(stream,
                                                R, N, C, H, W, poolingH, poolingW,
                                                spatialScale, rois, featureMap, top, NULL, deviceSmemSize);
        }
    }
    return STATUS_BAD_PARAM;
}
