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


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <cub/cub.cuh>
#include <functional>
#include <stdint.h>
#include "NvInfer.h"
#include "plugin.h"

// CUB's bug workaround:
// To work properly for large batch size CUB segmented sort needs ridiculous
// workspace alignment.
const uintptr_t ALIGNMENT = 1 << 20;
template <typename TFloat>
struct Bbox
{
    TFloat x1, y1, x2, y2;
};

typedef nvinfer1::DataType DType_t;

typedef enum
{
    NCHW = 0,
    NC4HW = 1
} DLayout_t;

typedef pluginStatus_t frcnnStatus_t;

#define DEBUG_RPN_ENABLE 0

#define FRCNN_ASSERT_PARAM(exp)                                                         \
    do                                                                                  \
    {                                                                                   \
        if (!(exp))                                                                     \
        {                                                                               \
            DEBUG_FPRINTF(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__); \
            return STATUS_BAD_PARAM;                                                    \
        }                                                                               \
    } while (0)


#define DEBUG_FPRINTF(...)        \
    do                            \
    {                             \
        if (DEBUG_RPN_ENABLE)     \
        {                         \
            fprintf(__VA_ARGS__); \
        }                         \
    } while (0)

#define CUDA_MEM_ALIGN 256

unsigned int hash(const void* array_, size_t size);
int8_t* alignPtr(int8_t* ptr, uintptr_t to);
__global__ void setOffset(int stride, int size, int* output);
bool initNmsLC();
frcnnStatus_t nms(cudaStream_t stream,
    const int N,
    const int R,
    const int preNmsTop,
    const int nmsMaxOut,
    const float iouThreshold,
    const DType_t t_fgScores,
    const DLayout_t l_fgScores,
    void* fgScores,
    const DType_t t_proposals,
    const DLayout_t l_proposals,
    const void* proposals,
    void* workspace,
    const DType_t t_rois,
    void* rois);
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);


template <typename TFloat>
__device__ __host__ inline float IoU(const Bbox<TFloat>& a, const Bbox<TFloat>& b)
{
    TFloat left = max(a.x1, b.x1), right = min(a.x2, b.x2);
    TFloat top = max(a.y1, b.y1), bottom = min(a.y2, b.y2);
    TFloat width = max((TFloat)(right - left + (TFloat) 1.0), (TFloat) 0.0);
    TFloat height = max((TFloat)(bottom - top + (TFloat) 1.0), (TFloat) 0.0);
    TFloat interS = width * height;
    TFloat Sa = (a.x2 - a.x1 + (TFloat) 1) * (a.y2 - a.y1 + (TFloat) 1);
    TFloat Sb = (b.x2 - b.x1 + (TFloat) 1) * (b.y2 - b.y1 + (TFloat) 1);
    return (float) interS / (float) (Sa + Sb - interS);
}


// NMS KERNEL FOR SMALL BATCH SIZE {{{
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
                afterNmsProposals[(batch_offset_out + kept) * 4 + 0] = ref_box.x1;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 1] = ref_box.y1;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 2] = ref_box.x2;
                afterNmsProposals[(batch_offset_out + kept) * 4 + 3] = ref_box.y2;
            }
        }

        __syncthreads();

        do
        {
            ref_box_idx++;
        }
        while (!kept_boxes[ref_box_idx - batch_offset] && ref_box_idx < max_box_idx);

        kept++;
    }
}
// }}}

// NMS KERNEL FOR LARGE BATCH SIZE {{{
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
    {
        foundBatch = 0;
    }

    for (int i = 0; i < TSIZE; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            int offset = i * DIM;
            int index = offset + j;

            if (index >= preNmsTopN)
            {
                break;
            }
            __syncthreads();

            if (threadIdx.x == j)
            {
                kept = 0 == (del & ((uint64_t) 1 << i));
                last = t[i];

                if (kept)
                {
                    int cnt = blockIdx.x * afterNmsTopN + foundBatch;
                    filtered[cnt * 4 + 0] = t[i].x1;
                    filtered[cnt * 4 + 1] = t[i].y1;
                    filtered[cnt * 4 + 2] = t[i].x2;
                    filtered[cnt * 4 + 3] = t[i].y2;
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
// }}}

// NMS LAUNCH {{{
template <typename T_PROPOSALS, DLayout_t L_PROPOSALS, typename T_ROIS>
frcnnStatus_t nmsLaunch(cudaStream_t stream,
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
    void (*kernel[64])(int, Bbox<T_PROPOSALS> const*, T_ROIS*, int, float, int) =
    {
        P1(1), P1(2), P1(3), P1(4), P1(5), P1(6), P1(7), P1(8), P1(9), P1(10), P1(11), P1(12), P2(13), P2(14), P2(15), P2(16),
        P2(17), P2(18), P2(19), P2(20), P2(21), P2(22), P2(23), P2(24), P2(25), P2(26), P2(27), P2(28), P2(29), P2(30), P2(31), P2(32),
        P2(33), P2(34), P2(35), P2(36), P2(37), P2(38), P2(39), P2(40), P2(41), P2(42), P2(43), P2(44), P2(45), P2(46), P2(47), P2(48),
        P2(49), P2(50), P2(51), P2(52), P2(53), P2(54), P2(55), P2(56), P2(57), P2(58), P2(59), P2(60), P2(61), P2(62), P2(63), P2(64)
    };
    FRCNN_ASSERT_PARAM(preNmsTopN < 64 * blockSize);
    CSC(cudaMemsetAsync(filtered, 0, batch * afterNmsTopN * 4 * sizeof(T_ROIS), stream),
        STATUS_FAILURE);
    kernel[(preNmsTopN + blockSize - 1) / blockSize - 1] <<< batch, blockSize, 0, stream>>>(propSize,
            (Bbox<T_PROPOSALS>*) proposals,
            (T_ROIS*) filtered,
            preNmsTopN,
            nmsThres,
            afterNmsTopN);
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}
// }}}


// NMS GPU {{{
template <typename T_SCORES, typename T_ROIS>
frcnnStatus_t nmsGpu(cudaStream_t stream,
                     const int N,
                     const int R,
                     const int preNmsTop,
                     const int nmsMaxOut,
                     const float iouThreshold,
                     void* fgScores,
                     const void* proposals,
                     void* workspace,
                     void* rois)
{
    int8_t* vworkspace = alignPtr((int8_t*) workspace, ALIGNMENT);
    DEBUG_PRINTF("&&&& [NMS] PROPOSALS %u\n", hash(proposals, N * R * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& [NMS] SCORES %u\n", hash(fgScores, N * R * sizeof(float)));
    frcnnStatus_t error;
    DEBUG_PRINTF("&&&& [NMS] DISCARD\n");
    DEBUG_PRINTF("&&&& [NMS] PROPOSALS %u\n", hash(proposals, N * R * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& [NMS] SCORES %u\n", hash(fgScores, N * R * sizeof(float)));
    // Generate offsets
    int* offsets = (int*) vworkspace;
    setOffset <<< 1, 1024, 0, stream>>>(R, N + 1, offsets);
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
// }}}

// NMS LAUNCH CONFIG {{{
typedef frcnnStatus_t (*nmsFun)(cudaStream_t,
                                const int,   // N
                                const int,   // R
                                const int,   // preNmsTop
                                const int,   // nmsMaxOut
                                const float, // iouThreshold
                                void*,       // fgScores
                                const void*, // proposals,
                                void*,       // workspace,
                                void*);      // rois

struct nmsLaunchConfig
{
    DType_t t_fgScores;
    DLayout_t l_fgScores;
    DType_t t_proposals;
    DLayout_t l_proposals;
    DType_t t_rois;
    nmsFun function;

    nmsLaunchConfig(DType_t t_fgScores,
                    DLayout_t l_fgScores,
                    DType_t t_proposals,
                    DLayout_t l_proposals,
                    DType_t t_rois,
                    nmsFun function)
        : t_fgScores(t_fgScores)
        , l_fgScores(l_fgScores)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_rois(t_rois)
        , function(function)
    {
    }

    nmsLaunchConfig(DType_t t_fgScores,
                    DLayout_t l_fgScores,
                    DType_t t_proposals,
                    DLayout_t l_proposals,
                    DType_t t_rois)
        : t_fgScores(t_fgScores)
        , l_fgScores(l_fgScores)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_rois(t_rois)
    {
    }

    bool operator==(const nmsLaunchConfig& other)
    {
        return (t_fgScores == other.t_fgScores) && (l_fgScores == other.l_fgScores)
               && (t_proposals == other.t_proposals) && (l_proposals == other.l_proposals)
               && (t_rois == other.t_rois);
    }
};

static std::vector<nmsLaunchConfig> nmsLCVec;
#define FLOAT32 nvinfer1::DataType::kFLOAT


static bool initializedNmsLC = initNmsLC();

// }}}


__global__ void _inverse_transform_gpu(const float* RPN_prob, const float* RPN_regr, int N,
                                       int INPUT_H, int INPUT_W, int RPN_H, int RPN_W, float RPN_STD_SCALING, int RPN_STRIDE,
                                       float* ANCHOR_SIZES, int anc_size_num, float* ANCHOR_RATIOS, int anc_ratio_num, float bbox_min_size,
                                       float* fg_scores, float* proposal_out)
{
    int nthreads = N * RPN_H * RPN_W * anc_size_num * anc_ratio_num;
    int num_ancs = anc_size_num * anc_ratio_num;

    for (int out_idx = threadIdx.x + blockDim.x * blockIdx.x; out_idx < nthreads;
            out_idx += blockDim.x * gridDim.x)
    {
        //input RPN_regr: (N, A4, H, W), thread: (N, A, H, W)
        int idx = out_idx;
        int w = idx % RPN_W;
        idx /= RPN_W;
        int h = idx % RPN_H;
        idx /= RPN_H;
        int a = idx % num_ancs;
        int n = idx / num_ancs;
        // normalize by RPN_STD_SCALING
        int ptr_1 = ((((n * num_ancs) + a) * 4) * RPN_H + h) * RPN_W + w;
        int ptr_2 = ((((n * num_ancs) + a) * 4 + 1) * RPN_H + h) * RPN_W + w;
        int ptr_3 = ((((n * num_ancs) + a) * 4 + 2) * RPN_H + h) * RPN_W + w;
        int ptr_4 = ((((n * num_ancs) + a) * 4 + 3) * RPN_H + h) * RPN_W + w;
        float tx = RPN_regr[ptr_1] / RPN_STD_SCALING;
        float ty = RPN_regr[ptr_2] / RPN_STD_SCALING;
        float tw = RPN_regr[ptr_3] / RPN_STD_SCALING;
        float th = RPN_regr[ptr_4] / RPN_STD_SCALING;
        // do inverse transform
        int ar = a % anc_ratio_num;
        int as = a / anc_ratio_num;
        float anchor_w = ANCHOR_SIZES[as] * ANCHOR_RATIOS[ar];
        float anchor_h = ANCHOR_SIZES[as] / ANCHOR_RATIOS[ar];
        float anchor_cx = (w + 0.5f) * RPN_STRIDE; 
        float anchor_cy = (h + 0.5f) * RPN_STRIDE; 
        float cx1 = anchor_cx + anchor_w * tx; 
        float cy1 = anchor_cy + anchor_h * ty; 
        float w1 = __expf(tw) * anchor_w; 
        float h1 = __expf(th) * anchor_h; 
        tx = cx1 - w1 / 2.0f; 
        ty = cy1 - h1 / 2.0f; 
        tw = w1; 
        th = h1; 
        tw += tx; 
        th += ty; 
        // clip to min
        tx = (tx >= 0.0f) ? tx : 0.0f;
        ty = (ty >= 0.0f) ? ty : 0.0f;
        tw = (tw >= 0.0f) ? tw : 0.0f;
        th = (th >= 0.0f) ? th : 0.0f;
        //clip to max
        tx = (tx <= INPUT_W - 1.0f) ? tx : (INPUT_W - 1.0f);
        ty = (ty <= INPUT_H - 1.0f) ? ty : (INPUT_H - 1.0f);
        tw = (tw <= INPUT_W - 1.0f) ? tw : (INPUT_W - 1.0f);
        th = (th <= INPUT_H - 1.0f) ? th : (INPUT_H - 1.0f);
        // filter out small boxes by setting the confidence to -inf
        int ininf = 0xff800000;
        float ninf = *(float*) &ininf;

        if (tw - tx <= bbox_min_size || th - ty <= bbox_min_size)
        {
            fg_scores[out_idx] = ninf;
        }

        // copy to proposal_out, output shape: (N, A, H, W, 4)
        proposal_out[out_idx * 4] = tx;
        proposal_out[out_idx * 4 + 1] = ty;
        proposal_out[out_idx * 4 + 2] = tw;
        proposal_out[out_idx * 4 + 3] = th;
    }
}



void _inverse_transform_wrapper(const float* RPN_prob, const float* RPN_regr, int N, int INPUT_H,
                                int INPUT_W, int RPN_H, int RPN_W, float RPN_STD_SCALING, int RPN_STRIDE,  float* ANCHOR_SIZES,
                                int anc_size_num, float* ANCHOR_RATIOS, int anc_ratio_num, float bbox_min_size, float* fg_scores,
                                float* proposal_out, cudaStream_t stream)
{
    const int block_size = 1024;
    const int grid_size = (N * anc_size_num * anc_ratio_num * RPN_H * RPN_W + block_size - 1) /
                          (block_size);
    _inverse_transform_gpu <<< grid_size, block_size, 0, stream>>> (RPN_prob, RPN_regr, N, INPUT_H,
            INPUT_W, RPN_H, RPN_W, RPN_STD_SCALING, RPN_STRIDE, ANCHOR_SIZES, anc_size_num, ANCHOR_RATIOS,
            anc_ratio_num, bbox_min_size, fg_scores, proposal_out);
}

size_t _proposalsForwardNMSWorkspaceSize(int N,
                                        int A,
                                        int H,
                                        int W,
                                        int nmsMaxOut)
{
    return N * A * H * W * 5 * 5 * sizeof(float) + (1 << 22);
}

size_t _proposalsForwardBboxWorkspaceSize(int N, int A, int H, int W)
{
    return N * A * H * W * 4 * sizeof(float);
}


size_t _proposalForwardFgScoresWorkspaceSize(int N, int A, int H, int W)
{
    return N * A * H * W * sizeof(float);
}


size_t anchors_buf_size(int anc_size_num, int anc_ratio_num)
{
    return (anc_size_num + anc_ratio_num) * sizeof(float);
}

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

size_t _get_workspace_size(int N,
                           int anc_size_num,
                           int anc_ratio_num,
                           int H,
                           int W,
                           int nmsMaxOut)
{
    size_t wss[4];
    int A = anc_size_num * anc_ratio_num;
    wss[0] = _proposalsForwardNMSWorkspaceSize(N, A, H, W, nmsMaxOut);
    wss[1] = _proposalsForwardBboxWorkspaceSize(N, A, H, W);
    wss[2] = _proposalForwardFgScoresWorkspaceSize(N, A, H, W);
    wss[3] = anchors_buf_size(anc_size_num, anc_ratio_num);
    return calculateTotalWorkspaceSize(wss, 4);
}



template <typename T>
frcnnStatus_t extractFgScores_gpu(cudaStream_t stream,
                                  int N,
                                  int A,
                                  int H,
                                  int W,
                                  const void* scores,
                                  void* fgScores)
{
    //TODO custom kernel for this
    size_t size = A * H * W * sizeof(T);

    for (int n = 0; n < N; n++)
    {
        size_t offset_ld = n * A * H * W;
        size_t offset_st = n * A * H * W;
        CSC(cudaMemcpyAsync(((T*) fgScores) + offset_st, ((T*) scores) + offset_ld, size,
                            cudaMemcpyDeviceToDevice, stream), STATUS_FAILURE);
    }

    return STATUS_SUCCESS;
}




void _copy_anchors_to_gpu(cudaStream_t stream, float* ANCHOR_SIZES, int anc_size_num,
                          float* ANCHOR_RATIOS, int anc_ratio_num, void* anchor_size_buf)
{
    cudaMemcpyAsync(anchor_size_buf, static_cast<void*>(ANCHOR_SIZES), sizeof(float) * anc_size_num,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(static_cast<void*>(static_cast<float*>(anchor_size_buf) + anc_size_num), static_cast<void*>(ANCHOR_RATIOS), sizeof(float) * anc_ratio_num,
                                           cudaMemcpyHostToDevice, stream);
}


__global__ void _normalize_rois_kernel(float* roi_after_nms, int nthreads, int width, int height)
{
    for(int i = threadIdx.x + blockDim.x * blockIdx.x; i < nthreads; i += blockDim.x * gridDim.x)
    {
        float x1 = roi_after_nms[i * 4];
        float y1 = roi_after_nms[i * 4 + 1];
        float x2 = roi_after_nms[i * 4 + 2];
        float y2 = roi_after_nms[i * 4 + 3];
        roi_after_nms[i * 4] = y1 / (height - 1.0f);
        roi_after_nms[i * 4 + 1] = x1 / (width - 1.0f);
        roi_after_nms[i * 4 + 2] = y2 / (height - 1.0f);
        roi_after_nms[i * 4 + 3] = x2 / (width - 1.0f);
    }
}



void _normalize_rois(float* roi_after_nms, int n, int max_box_num, int input_width,
                     int input_height, cudaStream_t stream)
{
    const int block_size = 1024;
    const int grid_size = (n * max_box_num + block_size - 1) / block_size;
    _normalize_rois_kernel <<< grid_size, block_size, 0, stream>>>(roi_after_nms, n * max_box_num,
            input_width, input_height);
}


int proposalInference_gpu(
    cudaStream_t stream,
    const void* rpn_prob,
    const void* rpn_regr,
    int batch_size,
    int input_height,
    int input_width,
    int rpn_height,
    int rpn_width,
    int MAX_BOX_NUM,
    int RPN_PRE_NMS_TOP_N,
    float* ANCHOR_SIZES,
    int anc_size_num,
    float* ANCHOR_RATIOS,
    int anc_ratio_num,
    float rpn_std_scaling,
    int rpn_stride,
    float bbox_min_size,
    float nms_iou_threshold,
    void * workspace,
    void* output)
{
    size_t nmsWorkspaceSize = _proposalsForwardNMSWorkspaceSize(batch_size, anc_size_num * anc_ratio_num,
                              rpn_height, rpn_width, MAX_BOX_NUM);
    void* nmsWorkspace = workspace;
    size_t proposalsSize = _proposalsForwardBboxWorkspaceSize(batch_size, anc_size_num * anc_ratio_num,
                           rpn_height, rpn_width);
    const DType_t t_proposals = nvinfer1::DataType::kFLOAT;
    const DLayout_t l_proposals = NC4HW;
    void* proposals = nextWorkspacePtr((int8_t*) nmsWorkspace, nmsWorkspaceSize);
    void* fg_scores = nextWorkspacePtr((int8_t*) proposals, proposalsSize);
    size_t fg_scores_size = _proposalForwardFgScoresWorkspaceSize(batch_size,
                            anc_size_num * anc_ratio_num, rpn_height, rpn_width);
    void* anchor_size_buf = nextWorkspacePtr((int8_t*) fg_scores, fg_scores_size);
    void* anchor_ratio_buf = static_cast<void*>(static_cast<float*>(anchor_size_buf) + anc_size_num);
    frcnnStatus_t status;
    _copy_anchors_to_gpu(stream, ANCHOR_SIZES, anc_size_num, ANCHOR_RATIOS, anc_ratio_num,
                         anchor_size_buf);
    status = extractFgScores_gpu<float>(stream,
                                        batch_size,
                                        anc_size_num * anc_ratio_num,
                                        rpn_height,
                                        rpn_width,
                                        rpn_prob,
                                        fg_scores);
    ASSERT(status == 0);
    _inverse_transform_wrapper(static_cast<const float*>(rpn_prob), static_cast<const float*>(rpn_regr),
                               batch_size, input_height, input_width, rpn_height, rpn_width, rpn_std_scaling, rpn_stride,
                               static_cast<float*>(anchor_size_buf), anc_size_num, static_cast<float*>(anchor_ratio_buf),
                               anc_ratio_num, bbox_min_size, static_cast<float*>(fg_scores), static_cast<float*>(proposals),
                               stream);
    status = nms(stream,
                 batch_size,
                 anc_size_num * anc_ratio_num * rpn_height * rpn_width,
                 RPN_PRE_NMS_TOP_N,
                 MAX_BOX_NUM,
                 nms_iou_threshold,
                 nvinfer1::DataType::kFLOAT,
                 NCHW,
                 fg_scores,
                 t_proposals,
                 l_proposals,
                 proposals,
                 workspace,
                 nvinfer1::DataType::kFLOAT,
                 output);
    ASSERT(status == 0);
    _normalize_rois(static_cast<float*>(output), batch_size, MAX_BOX_NUM, input_width, input_height,
                    stream);
    return 0;
}

