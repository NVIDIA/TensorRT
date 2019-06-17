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
#include <algorithm>
#include "cuda_runtime_api.h"
#include <cub/cub.cuh>
#include <functional>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "kernel.h"
#include "bboxUtils.h"

// CUB's bug workaround:
// To work properly for large batch size CUB segmented sort needs ridiculous
// workspace alignment.
const uintptr_t ALIGNMENT = 1 << 20;

// IOU 
template <typename TFloat>
__device__ __host__ inline float IoU(const Bbox<TFloat>& a, const Bbox<TFloat>& b)
{
    TFloat left = max(a.xmin, b.xmin), right = min(a.xmax, b.xmax);
    TFloat top = max(a.ymin, b.ymin), bottom = min(a.ymax, b.ymax);
    TFloat width = max((TFloat)(right - left + (TFloat) 1.0), (TFloat) 0.0);
    TFloat height = max((TFloat)(bottom - top + (TFloat) 1.0), (TFloat) 0.0);
    TFloat interS = width * height;
    TFloat Sa = (a.xmax - a.xmin + (TFloat) 1) * (a.ymax - a.ymin + (TFloat) 1);
    TFloat Sb = (b.xmax - b.xmin + (TFloat) 1) * (b.ymax - b.ymin + (TFloat) 1);
    return (float) interS / (float) (Sa + Sb - interS);
}

// NMS KERNEL FOR SMALL BATCH SIZE 
template <typename T_PROPOSALS, typename T_ROIS, int DIM, int TSIZE>
__global__ __launch_bounds__(DIM) void nmsKernel1(const int propSize,
                                                  Bbox<T_PROPOSALS> const* __restrict__ preNmsProposals,
                                                  T_ROIS* __restrict__ afterNmsProposals,
                                                  const int preNmsTopN,
                                                  const float nmsThres,
                                                  const int afterNmsTopN)
{
    __shared__ bool kept_boxes[TSIZE * DIM];
    int kept = 0;
    int batch_offset = blockIdx.x * propSize;
    int max_box_idx = batch_offset + preNmsTopN;
    int batch_offset_out = blockIdx.x * afterNmsTopN;

    int flag_idx[TSIZE];
    int boxes_idx[TSIZE];
    Bbox<T_PROPOSALS> cur_boxes[TSIZE];

// initialize kept_boxes
#pragma unroll
    for (int i = 0; i < TSIZE; i++)
    {
        boxes_idx[i] = threadIdx.x + batch_offset + DIM * i;
        flag_idx[i] = threadIdx.x + DIM * i;

        if (boxes_idx[i] < max_box_idx)
        {
            cur_boxes[i] = preNmsProposals[boxes_idx[i]];
            kept_boxes[flag_idx[i]] = true;
        }
        else
        {
            kept_boxes[flag_idx[i]] = false;
            boxes_idx[i] = -1.0f;
            flag_idx[i] = -1.0f;
        }
    }

    int ref_box_idx = 0 + batch_offset;

    // remove the overlapped boxes
    while ((kept < afterNmsTopN) && (ref_box_idx < max_box_idx))
    {
        Bbox<T_PROPOSALS> ref_box;
        ref_box = preNmsProposals[ref_box_idx];

#pragma unroll
        for (int i = 0; i < TSIZE; i++)
        {
            if (boxes_idx[i] > ref_box_idx)
            {
                if (IoU(ref_box, cur_boxes[i]) > nmsThres)
                {
                    kept_boxes[flag_idx[i]] = false;
                }
            }
            else if (boxes_idx[i] == ref_box_idx)
            {
                afterNmsProposals[(batch_offset_out + kept) * 4 + 0] = ref_box.xmin;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 1] = ref_box.ymin;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 2] = ref_box.xmax;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 3] = ref_box.ymax;
            }
        }
        __syncthreads();

        do
        {
            ref_box_idx++;
        } while (!kept_boxes[ref_box_idx - batch_offset] && ref_box_idx < max_box_idx);

        kept++;
    }
}

// NMS KERNEL FOR LARGE BATCH SIZE 
template <typename T_PROPOSALS, typename T_ROIS, int DIM, int TSIZE>
__global__ __launch_bounds__(DIM) void nmsKernel2(const int propSize,
                                                  Bbox<T_PROPOSALS> const* __restrict__ proposals,
                                                  T_ROIS* __restrict__ filtered,
                                                  const int preNmsTopN,
                                                  const float nmsThres,
                                                  const int afterNmsTopN)
{
    Bbox<T_PROPOSALS> const* cProposals = proposals + blockIdx.x * propSize;

    Bbox<T_PROPOSALS> t[TSIZE];
    uint64_t del = 0;

    for (int i = 0; i < TSIZE; i++)
    {
        if (i < TSIZE - 1 || i * DIM + threadIdx.x < preNmsTopN)
        {
            t[i] = cProposals[i * DIM + threadIdx.x];
        }
    }

    __shared__ Bbox<T_PROPOSALS> last;
    __shared__ bool kept;
    __shared__ int foundBatch;
    if (threadIdx.x == 0)
        foundBatch = 0;

    for (int i = 0; i < TSIZE; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            int offset = i * DIM;
            int index = offset + j;
            if (index >= preNmsTopN)
                break;

            __syncthreads();

            if (threadIdx.x == j)
            {
                kept = 0 == (del & ((uint64_t) 1 << i));
                last = t[i];

                if (kept)
                {
                    int cnt = blockIdx.x * afterNmsTopN + foundBatch;
                    filtered[cnt * 4 + 0] = t[i].xmin;
                    filtered[cnt * 4 + 1] = t[i].ymin;
                    filtered[cnt * 4 + 2] = t[i].xmax;
                    filtered[cnt * 4 + 3] = t[i].ymax;
                    foundBatch++;
                }
            }

            __syncthreads();

            if (foundBatch == afterNmsTopN)
            {
                return;
            }

            if (kept)
            {
                Bbox<T_PROPOSALS> test = last;

                for (int k = 0; k < TSIZE; k++)
                {
                    if (index < k * DIM + threadIdx.x
                        && IoU<T_PROPOSALS>(test, t[k]) > nmsThres)
                    {
                        del |= (uint64_t) 1 << k;
                    }
                }
            }
        }
    }
}

// NMS LAUNCH 
template <typename T_PROPOSALS, DLayout_t L_PROPOSALS, typename T_ROIS>
pluginStatus_t nmsLaunch(cudaStream_t stream,
                        const int batch,
                        const int propSize,
                        void* proposals,
                        void* filtered,
                        const int preNmsTopN,
                        const float nmsThres,
                        const int afterNmsTopN)
{
    const int blockSize = 1024;

#define P1(tsize) nmsKernel1<T_PROPOSALS, T_ROIS, blockSize, (tsize)>
#define P2(tsize) nmsKernel2<T_PROPOSALS, T_ROIS, blockSize, (tsize)>

    void (*kernel[64])(int, Bbox<T_PROPOSALS> const*, T_ROIS*, int, float, int) = {
        P1(1), P1(2), P1(3), P1(4), P1(5), P1(6), P1(7), P1(8), P1(9), P1(10), P1(11), P1(12), P2(13), P2(14), P2(15), P2(16),
        P2(17), P2(18), P2(19), P2(20), P2(21), P2(22), P2(23), P2(24), P2(25), P2(26), P2(27), P2(28), P2(29), P2(30), P2(31), P2(32),
        P2(33), P2(34), P2(35), P2(36), P2(37), P2(38), P2(39), P2(40), P2(41), P2(42), P2(43), P2(44), P2(45), P2(46), P2(47), P2(48),
        P2(49), P2(50), P2(51), P2(52), P2(53), P2(54), P2(55), P2(56), P2(57), P2(58), P2(59), P2(60), P2(61), P2(62), P2(63), P2(64)};

    ASSERT_PARAM(preNmsTopN < 64 * blockSize);

    CSC(cudaMemsetAsync(filtered, 0, batch * afterNmsTopN * 4 * sizeof(T_ROIS), stream), STATUS_FAILURE);

    kernel[(preNmsTopN + blockSize - 1) / blockSize - 1]<<<batch, blockSize, 0, stream>>>(propSize,
                                                                                          (Bbox<T_PROPOSALS>*) proposals,
                                                                                          (T_ROIS*) filtered,
                                                                                          preNmsTopN,
                                                                                          nmsThres,
                                                                                          afterNmsTopN);

    CSC(cudaGetLastError(), STATUS_FAILURE);

    return STATUS_SUCCESS;
}

// SET OFFSET 
// Works for up to 2Gi elements (cub's limitation)!
__global__ void setOffset(int stride, int size, int* output)
{
    // One block, because batch size shouldn't be too large.
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        output[i] = i * stride;
    }
}

// NMS GPU 
template <typename T_SCORES, typename T_ROIS>
pluginStatus_t nmsGpu(cudaStream_t stream,
                     const int N,
                     const int R,
                     const int preNmsTop,
                     const int nmsMaxOut,
                     const float iouThreshold,
                     //const float       minBoxSize,
                     //const float *     imInfo,
                     void* fgScores,
                     const void* proposals,
                     void* workspace,
                     void* rois)
{
    int8_t* vworkspace = alignPtr((int8_t*) workspace, ALIGNMENT);

    DEBUG_PRINTF("&&&& [NMS] PROPOSALS %u\n", hash(proposals, N * R * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& [NMS] SCORES %u\n", hash(fgScores, N * R * sizeof(float)));

    pluginStatus_t error;

    DEBUG_PRINTF("&&&& [NMS] DISCARD\n");
    DEBUG_PRINTF("&&&& [NMS] PROPOSALS %u\n", hash(proposals, N * R * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& [NMS] SCORES %u\n", hash(fgScores, N * R * sizeof(float)));

    // Generate offsets
    int* offsets = (int*) vworkspace;
    setOffset<<<1, 1024, 0, stream>>>(R, N + 1, offsets);
    CSC(cudaGetLastError(), STATUS_FAILURE);

    vworkspace = vworkspace + N + 1;
    vworkspace = alignPtr(vworkspace, ALIGNMENT);

    // Sort (batched)
    std::size_t tempStorageBytes = 0;

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        NULL, tempStorageBytes,
        (T_SCORES*) fgScores, (T_SCORES*) fgScores,
        (Bbox<T_ROIS>*) proposals, (Bbox<T_ROIS>*) proposals,
        N * R, N,
        offsets, offsets + 1, 0, 8 * sizeof(T_SCORES), stream);

    CSC(cudaGetLastError(), STATUS_FAILURE);

    T_SCORES* scoresOut = (T_SCORES*) vworkspace;
    vworkspace = (int8_t*) (scoresOut + N * R);
    vworkspace = alignPtr(vworkspace, ALIGNMENT);
    Bbox<T_ROIS>* proposalsOut = (Bbox<T_ROIS>*) vworkspace;
    vworkspace = (int8_t*) (proposalsOut + N * R);
    vworkspace = alignPtr(vworkspace, ALIGNMENT);

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        vworkspace, tempStorageBytes,
        (T_SCORES*) fgScores, (T_SCORES*) scoresOut,
        (Bbox<T_ROIS>*) proposals, (Bbox<T_ROIS>*) proposalsOut,
        N * R, N,
        offsets, offsets + 1,
        0, 8 * sizeof(T_SCORES), stream);

    CSC(cudaGetLastError(), STATUS_FAILURE);

    DEBUG_PRINTF("&&&& [NMS] POST CUB\n");
    DEBUG_PRINTF("&&&& [NMS] PROPOSALS %u\n", hash(proposalsOut, N * R * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& [NMS] SCORES %u\n", hash(scoresOut, N * R * sizeof(float)));

    error = nmsLaunch<T_ROIS, NC4HW, T_ROIS>(stream,
                                             N,
                                             R,
                                             proposalsOut,
                                             rois,
                                             preNmsTop,
                                             iouThreshold,
                                             nmsMaxOut);

    DEBUG_PRINTF("&&&& [NMS] POST LAUNCH\n");
    DEBUG_PRINTF("&&&& [NMS] SCORES %u\n", hash(rois, N * nmsMaxOut * 4 * sizeof(float)));

    if (error != STATUS_SUCCESS)
    {
        return error;
    }

    return STATUS_SUCCESS;
}

// NMS LAUNCH CONFIG 
typedef pluginStatus_t (*nmsFun)(cudaStream_t,
                                const int,   // N
                                const int,   // R
                                const int,   // preNmsTop
                                const int,   // nmsMaxOut
                                const float, // iouThreshold
                                //const float,       // minBoxSize
                                //const float *,     // imInfo
                                void*,       // fgScores
                                const void*, // proposals,
                                void*,       // workspace,
                                void*);      // rois

struct nmsLaunchConfig
{
    DataType t_fgScores;
    DLayout_t l_fgScores;
    DataType t_proposals;
    DLayout_t l_proposals;
    DataType t_rois;
    nmsFun function;

    nmsLaunchConfig(DataType t_fgScores,
                    DLayout_t l_fgScores,
                    DataType t_proposals,
                    DLayout_t l_proposals,
                    DataType t_rois,
                    nmsFun function)
        : t_fgScores(t_fgScores)
        , l_fgScores(l_fgScores)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_rois(t_rois)
        , function(function)
    {
    }

    nmsLaunchConfig(DataType t_fgScores,
                    DLayout_t l_fgScores,
                    DataType t_proposals,
                    DLayout_t l_proposals,
                    DataType t_rois)
        : t_fgScores(t_fgScores)
        , l_fgScores(l_fgScores)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_rois(t_rois)
    {
    }

    bool operator==(const nmsLaunchConfig& other)
    {
        return (t_fgScores == other.t_fgScores) && (l_fgScores == other.l_fgScores) && (t_proposals == other.t_proposals) && (l_proposals == other.l_proposals) && (t_rois == other.t_rois);
    }
};

static std::vector<nmsLaunchConfig> nmsLCVec;
#define FLOAT32 nvinfer1::DataType::kFLOAT
bool initNmsLC()
{
    nmsLCVec.reserve(1);
    nmsLCVec.push_back(nmsLaunchConfig(FLOAT32, NCHW,
                                       FLOAT32, NC4HW,
                                       FLOAT32,
                                       nmsGpu<float, float>));
    return true;
}

static bool initializedNmsLC = initNmsLC();


// NMS 
pluginStatus_t nms(cudaStream_t stream,
                  const int N,
                  const int R,
                  const int preNmsTop,
                  const int nmsMaxOut,
                  const float iouThreshold,
                  const DataType t_fgScores,
                  const DLayout_t l_fgScores,
                  void* fgScores,
                  const DataType t_proposals,
                  const DLayout_t l_proposals,
                  const void* proposals,
                  void* workspace,
                  const DataType t_rois,
                  void* rois)
{
    if (!initializedNmsLC)
        return STATUS_NOT_INITIALIZED;
    nmsLaunchConfig lc(t_fgScores, l_fgScores, t_proposals, l_proposals, t_rois);
    for (unsigned i = 0; i < nmsLCVec.size(); i++)
    {
        if (nmsLCVec[i] == lc)
        {
            DEBUG_PRINTF("NMS KERNEL %d\n", i);
            return nmsLCVec[i].function(stream,
                                        N, R,
                                        preNmsTop,
                                        nmsMaxOut,
                                        iouThreshold,
                                        fgScores,
                                        proposals,
                                        workspace,
                                        rois);
        }
    }
    return STATUS_BAD_PARAM;
}
