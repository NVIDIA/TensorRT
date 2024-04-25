/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/plugin.h"
#include "maskRCNNKernels.h"
#include <NvInfer.h>
#include <assert.h>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define DUBUG_KERNEL 0
#define DUBUG_BATCH 0
#define DEBUG_T 1

#define dMIN(a, b) ((a) < (b) ? (a) : (b))
#define dMAX(a, b) ((a) > (b) ? (a) : (b))
#define dCLAMP(x, xMin, xMax) ((x) > (xMin) ? ((x) < (xMax) ? (x) : (xMax)) : (xMin))

template <typename BoxType>
struct BBoxT
{
    BoxType y1, x1, y2, x2;
};


inline __device__ __half mul_fb(const __half & a, const __half & b) {
    #if __CUDA_ARCH__ >= 530
        return a * b;
    #else
        return __float2half(__half2float(a) * __half2float(b));
    #endif
}

inline __device__ __half add_fb(const __half & a, const half & b) {
    #if __CUDA_ARCH__ >= 530
        return a + b;
    #else
        return __float2half(__half2float(a) + __half2float(b));
    #endif
}

template <typename DType>
__global__ void argMaxReset_kernel(
    int samples, int NClass, const DType* in_scores, const int* maxIdx, DType* out_scores)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int max_idx = samples * NClass;
    if (idx >= max_idx)
        return;

    int sampleIdx = idx / NClass;
    int classIdx = idx % NClass;
    if (classIdx != maxIdx[sampleIdx])
        out_scores[idx] = 0;
    else
        out_scores[idx] = in_scores[idx];
}

template <typename DType>
struct ScanItem
{
    DType data;
    int idx;
};

template <typename DType>
struct GreaterItem
{
    __host__ __device__ __forceinline__ ScanItem<DType> operator()(
        const ScanItem<DType>& a, const ScanItem<DType>& b) const
    {
        return (a.data > b.data ? a : b);
    }
};

template <typename DType>
__global__ void resetMemValue_kernel(void* outPtr, int samples, float val)
{
    DType* out = static_cast<DType*>(outPtr);
    int loop = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += loop)
    {
        out[idx] = (DType) val;
    }
}

template <>
__global__ void resetMemValue_kernel<half>(void* outPtr, int samples, float val)
{
    __half* out = static_cast<__half*>(outPtr);
    int loop = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += loop)
    {
        out[idx] = __float2half(val);
    }
}
// blockDim.x : NClass
// GroupDim.x : sample count
// GroupDim.y : batch N
// outScore : DType[ N * sample * 1 ]
// outLabel : int[ N * sample * 1 ]
// outBbox : int[ N * sample * 4 ]
template <typename DType, typename BoxType, int Threads = 32>
__global__ void argMaxGroup_kernel(int samples, int start_class_id, int NClass, const void* inScorePtr,
    const void* inBboxPtr, const void* validSampleCountPtr, void* outScorePtr, void* outLabelPtr, void* outBboxPtr)
{
    const DType* inScore = static_cast<const DType*>(inScorePtr);
    const BoxType* inBbox = static_cast<const BoxType*>(inBboxPtr);
    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    DType* outScore = static_cast<DType*>(outScorePtr);
    BoxType* outLabel = static_cast<BoxType*>(outLabelPtr);
    BoxType* outBbox = static_cast<BoxType*>(outBboxPtr);

    const int N = blockIdx.y;
    const int validSamples = validSampleCount[N];

    typedef ScanItem<DType> ScanItemD;
    typedef cub::BlockReduce<ScanItemD, Threads> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    for (int iSample = blockIdx.x; iSample < validSamples; iSample += gridDim.x)
    {
        int classOffset = (N * samples + iSample) * NClass; // start from [batch, count, class0]
        // total IPerThread * blockDim
        ScanItemD maxItem = {0.0f, -1};
        for (int i = start_class_id; i < NClass; i += Threads)
        {
            int curIdx = i + threadIdx.x;
            ScanItemD item = {0.0f, -1};
            if (curIdx < NClass)
            {
                item.data = inScore[classOffset + curIdx];
                item.idx = curIdx;
            }
            const int validNum = (NClass - i > Threads ? Threads : NClass - i);
            ScanItemD aggregate = BlockReduce(temp_storage).Reduce(item, GreaterItem<DType>(), validNum);
            __syncthreads();
            if (aggregate.data > maxItem.data)
            {
                maxItem = aggregate;
            }
#if DUBUG_KERNEL
            if (N == DUBUG_BATCH && threadIdx.x == 0 && iSample < 15 /*&& maxItem.idx >= 32*/)
            {
                printf("argMaxGroup N:%d, iSample:%d, maxItem(score:%.3f, idx:%d)validReduceNum:%d\n", N, iSample,
                    (float) maxItem.data, maxItem.idx, validNum);
            }
#endif
        }

        const int dstOffset = N * samples + iSample;
        if (threadIdx.x == 0)
        {
            outScore[dstOffset] = maxItem.data;
            outLabel[dstOffset] = (BoxType) maxItem.idx;
            outBbox[dstOffset * 4] = inBbox[(classOffset + maxItem.idx) * 4];
            outBbox[dstOffset * 4 + 1] = inBbox[(classOffset + maxItem.idx) * 4 + 1];
            outBbox[dstOffset * 4 + 2] = inBbox[(classOffset + maxItem.idx) * 4 + 2];
            outBbox[dstOffset * 4 + 3] = inBbox[(classOffset + maxItem.idx) * 4 + 3];
        }
    }
}

struct BlockClassSumPrefix
{
    int total;
    // Constructor
    __device__ BlockClassSumPrefix()
        : total(0)
    {
    }
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int aggregate)
    {
        int old = total;
        total += aggregate;
        return old;
    }
};

#define LabelShift (2.5f)
#define MinValidScore (0.01f)
#define ScoreShift (1.0f)

template <typename DType>
__device__ __forceinline__ DType getKey(DType score, int lable, int NClass)
{
    return (lable < 0 ? (DType) 0 : ((DType)(NClass - lable - 1) * LabelShift + score + ScoreShift));
}

template <typename DType, typename BoxType>
__device__ __forceinline__ void getScoreLable(DType key, int NClass, DType& score, BoxType& lable)
{
    int i = key / LabelShift;
    score = (key <= ScoreShift ? (DType) 0 : key - (DType) i * LabelShift - ScoreShift);
    score = dCLAMP(score, (DType) 0, (DType) 1.0);
    lable = (BoxType)(key <= ScoreShift ? -1 : (NClass - i - 1));
}

// blockDim.x : threads
// gridDim.x : batch N
// validSampleCount INPUT : int [N]
// classStartPos OUTPUT: int [N * (Class + 1)], need memset to zero before this kernel
// outScore OUTPUT : DType [N * samples]
// outLabel OUTPUT : int [N * samples]
// outSampleIdx OUTPUT : int [N * samples]
// outValidSampleCount : int [N]
// IPerThread * Threads >= sample-count
#define MaxClassNum 255
template <typename DType, typename BoxType, int Threads = 256, int IPerThread = 4>
__global__ void sortPerClass_kernel(
    // int N,
    int samples, int NClass, int background, float scoreThreshold, const void* validSampleCountPtr,
    const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, void* classStartPosPtr, void* outScorePtr,
    void* outLabelPtr, void* outSampleIdxPtr, void* outValidSampleCountPtr)
{
    typedef cub::BlockExchange<DType, Threads, IPerThread> BlockExchangeKey;
    typedef cub::BlockExchange<int, Threads, IPerThread> BlockExchangeI;
    typedef cub::BlockRadixSort<DType, Threads, IPerThread, int> BlockRadixSort;
    typedef cub::BlockScan<int, Threads> BlockScanClass;
    __shared__ union
    {
        typename BlockExchangeKey::TempStorage storageKey;
        typename BlockExchangeI::TempStorage storageI;
        typename BlockRadixSort::TempStorage storageSort;
        typename BlockScanClass::TempStorage storageScan;
    } temp_storage;
    __shared__ int smemClassCount[MaxClassNum];
    assert(NClass < MaxClassNum);
    assert(IPerThread * Threads >= samples);

    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    const DType* inScore = static_cast<const DType*>(inScorePtr);
    const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr);
    int* classStartPos = static_cast<int*>(classStartPosPtr);
    DType* outScore = static_cast<DType*>(outScorePtr);
    BoxType* outLabel = static_cast<BoxType*>(outLabelPtr);
    int* outSampleIdx = static_cast<int*>(outSampleIdxPtr);
    int* outValidSampleCount = static_cast<int*>(outValidSampleCountPtr);

    for (int s = threadIdx.x; s < NClass + 1; s += blockDim.x)
    {
        smemClassCount[s] = 0;
    }

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int validSamples = validSampleCount[N];
    DType key[IPerThread];
    int iSample[IPerThread];
    for (int i = 0; i < IPerThread; ++i)
    {
        iSample[i] = -1;
        key[i] = -1.0f;
        int curIdx = i * Threads + threadIdx.x;
        if (curIdx < validSamples)
        {
            int label = (int) (inLabel[blockOffset + curIdx]);
            DType score = inScore[blockOffset + curIdx];
            if (label != background && label != -1 && score >= scoreThreshold)
            {
                key[i] = getKey(score, label, NClass);
                iSample[i] = curIdx;
            }
        }
    }

    BlockExchangeKey(temp_storage.storageKey).StripedToBlocked(key);
    __syncthreads();
    BlockExchangeI(temp_storage.storageI).StripedToBlocked(iSample);
    __syncthreads();
    BlockRadixSort(temp_storage.storageSort).SortDescendingBlockedToStriped(key, iSample);
    __syncthreads();

    // store Idx
    cub::StoreDirectStriped<Threads>(threadIdx.x, outSampleIdx + blockOffset, iSample, validSamples);
    BoxType lable[IPerThread];
    DType score[IPerThread];

#pragma unroll
    for (int i = 0; i < IPerThread; ++i)
    {
        getScoreLable(key[i], NClass, score[i], lable[i]);
    }
    cub::StoreDirectStriped<Threads>(threadIdx.x, outScore + blockOffset, score, validSamples);
    cub::StoreDirectStriped<Threads>(threadIdx.x, outLabel + blockOffset, lable, validSamples);

    // final
    for (int i = 0; i < IPerThread; ++i)
    {
        if (lable[i] >= (BoxType) 0)
        {
            atomicAdd(&smemClassCount[(int) lable[i]], 1);
        }
    }
    __syncthreads();

    int classBlockOffset = N * (NClass + 1); // Exclusive-sum, 1st is 0, last is final sum

#if DUBUG_KERNEL
    if (N == DUBUG_BATCH && threadIdx.x == 0)
    {
        printf("sortPerClass(N:%d) final count of each label, valid samples:%d\n", N, validSamples);
        for (int k = 0; k < NClass; ++k)
        {
            if (smemClassCount[k] > 0)
                printf("Batch:%d, L:%d, count:%d, \n", N, k, smemClassCount[k]);
        }
    }
    __syncthreads();
#endif

    BlockClassSumPrefix sumPrefix;
    for (int s = 0; s < NClass; s += blockDim.x)
    { // s start from block
        int iClassSamples = 0;
        int iClass = s + threadIdx.x;
        if (iClass < NClass)
        {
            iClassSamples = smemClassCount[iClass];
        }
        BlockScanClass(temp_storage.storageScan).ExclusiveSum(iClassSamples, iClassSamples, sumPrefix);
        __syncthreads();
        if (iClass < NClass)
        {
            classStartPos[classBlockOffset + iClass] = iClassSamples;
        }
    }
    if (threadIdx.x == 0)
    {
        classStartPos[classBlockOffset + NClass] = sumPrefix.total;
        assert(sumPrefix.total <= validSamples); // background data removed.
        outValidSampleCount[N] = sumPrefix.total;
#if DUBUG_KERNEL
        if (N == DUBUG_BATCH)
            printf("After sortPerClass, batch:%d valid samples total:%d\n", N, sumPrefix.total);
#endif
    }
}

template <int Threads = 256, int IPerThread = 4>
__global__ void sortPerClass_kernel_half(
    // int N,
    int samples, int NClass, int background, float scoreThreshold, const void* validSampleCountPtr,
    const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, void* classStartPosPtr, void* outScorePtr,
    void* outLabelPtr, void* outSampleIdxPtr, void* outValidSampleCountPtr)
{
    typedef cub::BlockExchange<float, Threads, IPerThread> BlockExchangeKey;
    typedef cub::BlockExchange<int, Threads, IPerThread> BlockExchangeI;
    typedef cub::BlockRadixSort<float, Threads, IPerThread, int> BlockRadixSort;
    typedef cub::BlockScan<int, Threads> BlockScanClass;
    __shared__ union
    {
        typename BlockExchangeKey::TempStorage storageKey;
        typename BlockExchangeI::TempStorage storageI;
        typename BlockRadixSort::TempStorage storageSort;
        typename BlockScanClass::TempStorage storageScan;
    } temp_storage;
    __shared__ int smemClassCount[MaxClassNum];
    assert(NClass < MaxClassNum);
    assert(IPerThread * Threads >= samples);

    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    const __half* inScore = static_cast<const __half*>(inScorePtr);
    const __half* inLabel = static_cast<const __half*>(inLabelPtr);
    int* classStartPos = static_cast<int*>(classStartPosPtr);
    __half* outScore = static_cast<__half*>(outScorePtr);
    __half* outLabel = static_cast<__half*>(outLabelPtr);
    int* outSampleIdx = static_cast<int*>(outSampleIdxPtr);
    int* outValidSampleCount = static_cast<int*>(outValidSampleCountPtr);

    for (int s = threadIdx.x; s < NClass + 1; s += blockDim.x)
    {
        smemClassCount[s] = 0;
    }

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int validSamples = validSampleCount[N];
    float key[IPerThread];
    int iSample[IPerThread];
    for (int i = 0; i < IPerThread; ++i)
    {
        iSample[i] = -1;
        key[i] = -1.0f;
        int curIdx = i * Threads + threadIdx.x;
        if (curIdx < validSamples)
        {
            int label = __half2int_rd(inLabel[blockOffset + curIdx]);
            float score = __half2float(inScore[blockOffset + curIdx]);
            if (label != background && label != -1 && score >= scoreThreshold)
            {
                key[i] = getKey<float>(score, label, NClass);
                iSample[i] = curIdx;
            }
        }
    }

    BlockExchangeKey(temp_storage.storageKey).StripedToBlocked(key);
    __syncthreads();
    BlockExchangeI(temp_storage.storageI).StripedToBlocked(iSample);
    __syncthreads();
    BlockRadixSort(temp_storage.storageSort).SortDescendingBlockedToStriped(key, iSample);
    __syncthreads();

    // store Idx
    cub::StoreDirectStriped<Threads>(threadIdx.x, outSampleIdx + blockOffset, iSample, validSamples);
    __half lable[IPerThread];
    __half score[IPerThread];

    for (int i = 0; i < IPerThread; ++i)
    {
        float label_float;
        float score_float;
        getScoreLable<float>(key[i], NClass, score_float, label_float);
        lable[i] = __float2half(label_float);
        score[i] = __float2half(score_float);
    }
    cub::StoreDirectStriped<Threads>(threadIdx.x, outScore + blockOffset, score, validSamples);
    cub::StoreDirectStriped<Threads>(threadIdx.x, outLabel + blockOffset, lable, validSamples);

    // final
    for (int i = 0; i < IPerThread; ++i)
    {
        if (__half2float(lable[i]) >= 0)
        {
            atomicAdd(&smemClassCount[__half2int_rd(lable[i])], 1);
        }
    }
    __syncthreads();

    int classBlockOffset = N * (NClass + 1); // Exclusive-sum, 1st is 0, last is final sum

#if DUBUG_KERNEL
    if (N == DUBUG_BATCH && threadIdx.x == 0)
    {
        printf("sortPerClass(N:%d) final count of each label, valid samples:%d\n", N, validSamples);
        for (int k = 0; k < NClass; ++k)
        {
            if (smemClassCount[k] > 0)
                printf("Batch:%d, L:%d, count:%d, \n", N, k, smemClassCount[k]);
        }
    }
    __syncthreads();
#endif

    BlockClassSumPrefix sumPrefix;
    for (int s = 0; s < NClass; s += blockDim.x)
    { // s start from block
        int iClassSamples = 0;
        int iClass = s + threadIdx.x;
        if (iClass < NClass)
        {
            iClassSamples = smemClassCount[iClass];
        }
        BlockScanClass(temp_storage.storageScan).ExclusiveSum(iClassSamples, iClassSamples, sumPrefix);
        __syncthreads();
        if (iClass < NClass)
        {
            classStartPos[classBlockOffset + iClass] = iClassSamples;
        }
    }
    if (threadIdx.x == 0)
    {
        classStartPos[classBlockOffset + NClass] = sumPrefix.total;
        assert(sumPrefix.total <= validSamples); // background data removed.
        outValidSampleCount[N] = sumPrefix.total;
#if DUBUG_KERNEL
        if (N == DUBUG_BATCH)
            printf("After sortPerClass, batch:%d valid samples total:%d\n", N, sumPrefix.total);
#endif
    }
}

template <typename DType>
__device__ __forceinline__ BBoxT<DType> readBbox(const BBoxT<DType>* inBbox, int idx)
{
    BBoxT<DType> ret = ((BBoxT<DType>*) (inBbox))[idx];
    return ret;
}

template <typename DType>
__device__ __forceinline__ DType boxIoU(const BBoxT<DType>& a, const BBoxT<DType>& b)
{
    BBoxT<DType> overlap = {
        dMAX(a.y1, b.y1), dMAX(a.x1, b.x1), dMIN(a.y2, b.y2), dMIN(a.x2, b.x2),
    };
    DType oW = overlap.x2 - overlap.x1;
    DType oH = overlap.y2 - overlap.y1;
    if (oW < (DType) 0 || oH < (DType) 0)
        return (DType) 0;
    DType oA = oW * oH;
    return (oA / ((a.y2 - a.y1) * (a.x2 - a.x1) + (b.y2 - b.y1) * (b.x2 - b.x1) - oA));
}

// PerClassNMS
// gridDim.x : batch-N
// blockDim.x : Threads
// ItemsPerThreads : = divUp(samples, Threads)
// outFlagSamples OUT: int [N * samples]
template <typename DType, typename BoxType, int Threads = 256, int ItemsPerThreads = 4>
__global__ void PerClassNMS_kernel(
    // int N,
    int samples, int NClass, const float nmsThreshold, const void* validSampleCountPtr,
    // const void *inScorePtr,
    const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr, const void* classStartsPtr,
    void* outFlagSamplesPtr)
{
    typedef BBoxT<BoxType> BBox;
    __shared__ struct
    {
        BBox refBox[MaxClassNum];
        int endIdx[MaxClassNum];
        int refIdx[MaxClassNum + 1];
        bool markSamples[Threads * ItemsPerThreads];
        int done;
    } smemClasses;
    assert(NClass + 1 < MaxClassNum);
    assert(samples <= Threads * ItemsPerThreads);

    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    // const DType *inScore = static_cast<const DType *>(inScorePtr);
    const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr);
    const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
    const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
    const int* classStarts = static_cast<const int*>(classStartsPtr);
    int* outFlagSamples = static_cast<int*>(outFlagSamplesPtr);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int validSamples = validSampleCount[N];

    if (threadIdx.x == 0)
    {
        smemClasses.done = 0;
    }

    BBox curBox[ItemsPerThreads];
    int label[ItemsPerThreads];
#pragma unroll
    for (int ite = 0; ite * blockDim.x < validSamples; ++ite)
    {
        int curIdx = ite * blockDim.x + threadIdx.x;
        if (curIdx < validSamples)
        {
            label[ite] = (int) inLabel[blockOffset + curIdx];
            curBox[ite] = readBbox(inBbox, blockOffset + inBboxRefIdx[blockOffset + curIdx]);
        }
        else
        {
            label[ite] = -1;
        }
        smemClasses.markSamples[curIdx] = (label[ite] < 0 ? false : true);
    }

    int classBlockOffset = N * (NClass + 1);
    for (int i = threadIdx.x; i < NClass + 1; i += blockDim.x)
    {
        int refIdx = classStarts[classBlockOffset + i];
        smemClasses.refIdx[i] = refIdx;
        smemClasses.refBox[i] = readBbox(inBbox, blockOffset + inBboxRefIdx[blockOffset + refIdx]);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NClass; i += blockDim.x)
    {
        int endIdx = smemClasses.refIdx[i + 1];
        smemClasses.endIdx[i] = endIdx;
        if (endIdx == smemClasses.refIdx[i])
        {
            atomicAdd(&smemClasses.done, 1);
        }
    }
    __syncthreads();

#if DUBUG_KERNEL
    // print info
    if (N == DUBUG_BATCH && threadIdx.x == 0)
    {
        printf("batch:%d, before starting NMS, done count:%d\n", N, smemClasses.done);
        printf("batch:%d, Total num:%d, startPos:\n", N, validSamples);
        for (int k = 0; k < NClass; ++k)
        {
            if (smemClasses.refIdx[k] != smemClasses.endIdx[k])
            {
                printf("Batch:%d, label:%d [%d : %d], check ref-label:%d\n", N, k, smemClasses.refIdx[k],
                    smemClasses.endIdx[k], (int) inLabel[blockOffset + smemClasses.refIdx[k]]);
            }
        }
        printf("\n");
    }
    __syncthreads();
#endif

    // class done to check stop point
    while (smemClasses.done < NClass)
    {

        for (int ite = 0; ite * blockDim.x < validSamples; ++ite)
        {
            int curIdx = ite * blockDim.x + threadIdx.x;
            int refIdx = -1;
            int endIdx = -1;
            if (curIdx < validSamples && smemClasses.markSamples[curIdx])
            {
                if (label[ite] >= 0)
                {
                    refIdx = smemClasses.refIdx[label[ite]];
                    endIdx = smemClasses.endIdx[label[ite]];
                    if (curIdx > refIdx && curIdx < endIdx)
                    {
                        BBox refBox = smemClasses.refBox[label[ite]];
                        if (boxIoU(refBox, curBox[ite]) > (DType) nmsThreshold)
                        {
                            smemClasses.markSamples[curIdx] = false;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // push refIdx/refBox forward to next mark
        // only the refIdx thread to push itself. other threads idle
        for (int i = threadIdx.x; i < NClass; i += blockDim.x)
        {
            int refIdx = smemClasses.refIdx[i];
            int endIdx = smemClasses.endIdx[i];
            if (refIdx < endIdx)
            {
                do
                {
                    ++refIdx;
                } while (refIdx < endIdx && smemClasses.markSamples[refIdx] == false);
                smemClasses.refIdx[i] = refIdx;
                if (refIdx < endIdx)
                {
                    smemClasses.refBox[i] = readBbox(inBbox, blockOffset + inBboxRefIdx[blockOffset + refIdx]);
                }
                else
                {
                    atomicAdd(&smemClasses.done, 1);
                }
            }
        }
        __syncthreads();
    }

    // no need to write all data out
    for (int segment = 0; segment < validSamples; segment += blockDim.x)
    {
        int curIdx = segment + threadIdx.x;
        if (curIdx < validSamples)
        {
            outFlagSamples[blockOffset + curIdx] = (smemClasses.markSamples[curIdx] ? 1 : 0);
        }
    }
}

template <int Threads = 256, int ItemsPerThreads = 4>
__global__ void PerClassNMS_half_kernel(
    // int N,
    int samples, int NClass, const float nmsThreshold, const void* validSampleCountPtr,
    // const void *inScorePtr,
    const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr, const void* classStartsPtr,
    void* outFlagSamplesPtr)
{
    typedef BBoxT<__half> BBox;
    __shared__ struct
    {
        BBox refBox[MaxClassNum];
        int endIdx[MaxClassNum];
        int refIdx[MaxClassNum + 1];
        bool markSamples[Threads * ItemsPerThreads];
        int done;
    } smemClasses;
    assert(NClass + 1 < MaxClassNum);
    assert(samples <= Threads * ItemsPerThreads);

    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    // const DType *inScore = static_cast<const DType *>(inScorePtr);
    const __half* inLabel = static_cast<const __half*>(inLabelPtr);
    const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
    const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
    const int* classStarts = static_cast<const int*>(classStartsPtr);
    int* outFlagSamples = static_cast<int*>(outFlagSamplesPtr);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int validSamples = validSampleCount[N];

    if (threadIdx.x == 0)
    {
        smemClasses.done = 0;
    }

    BBox curBox[ItemsPerThreads];
    int label[ItemsPerThreads];
#pragma unroll
    for (int ite = 0; ite * blockDim.x < validSamples; ++ite)
    {
        int curIdx = ite * blockDim.x + threadIdx.x;
        if (curIdx < validSamples)
        {
            label[ite] = __half2int_rd(inLabel[blockOffset + curIdx]);
            curBox[ite] = readBbox<__half>(inBbox, blockOffset + inBboxRefIdx[blockOffset + curIdx]);
        }
        else
        {
            label[ite] = -1;
        }
        smemClasses.markSamples[curIdx] = (label[ite] < 0 ? false : true);
    }

    int classBlockOffset = N * (NClass + 1);
    for (int i = threadIdx.x; i < NClass + 1; i += blockDim.x)
    {
        int refIdx = classStarts[classBlockOffset + i];
        smemClasses.refIdx[i] = refIdx;
        smemClasses.refBox[i] = readBbox<__half>(inBbox, blockOffset + inBboxRefIdx[blockOffset + refIdx]);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NClass; i += blockDim.x)
    {
        int endIdx = smemClasses.refIdx[i + 1];
        smemClasses.endIdx[i] = endIdx;
        if (endIdx == smemClasses.refIdx[i])
        {
            atomicAdd(&smemClasses.done, 1);
        }
    }
    __syncthreads();

#if DUBUG_KERNEL
    // print info
    if (N == DUBUG_BATCH && threadIdx.x == 0)
    {
        printf("batch:%d, before starting NMS, done count:%d\n", N, smemClasses.done);
        printf("batch:%d, Total num:%d, startPos:\n", N, validSamples);
        for (int k = 0; k < NClass; ++k)
        {
            if (smemClasses.refIdx[k] != smemClasses.endIdx[k])
            {
                printf("Batch:%d, label:%d [%d : %d], check ref-label:%d\n", N, k, smemClasses.refIdx[k],
                    smemClasses.endIdx[k], (int) inLabel[blockOffset + smemClasses.refIdx[k]]);
            }
        }
        printf("\n");
    }
    __syncthreads();
#endif

    // class done to check stop point
    while (smemClasses.done < NClass)
    {

        for (int ite = 0; ite * blockDim.x < validSamples; ++ite)
        {
            int curIdx = ite * blockDim.x + threadIdx.x;
            int refIdx = -1;
            int endIdx = -1;
            if (curIdx < validSamples && smemClasses.markSamples[curIdx])
            {
                if (label[ite] >= 0)
                {
                    refIdx = smemClasses.refIdx[label[ite]];
                    endIdx = smemClasses.endIdx[label[ite]];
                    if (curIdx > refIdx && curIdx < endIdx)
                    {
                        BBox refBox_half = smemClasses.refBox[label[ite]];
                        BBox curBox_half = curBox[ite];
                        BBoxT<float> refBox;
                        BBoxT<float> curBox_float;
                        refBox.y1 = __half2float(refBox_half.y1);
                        refBox.x1 = __half2float(refBox_half.x1);
                        refBox.y2 = __half2float(refBox_half.y2);
                        refBox.x2 = __half2float(refBox_half.x2);
                        curBox_float.y1 = __half2float(curBox_half.y1);
                        curBox_float.x1 = __half2float(curBox_half.x1);
                        curBox_float.y2 = __half2float(curBox_half.y2);
                        curBox_float.x2 = __half2float(curBox_half.x2);
                        if (boxIoU<float>(refBox, curBox_float) > nmsThreshold)
                        {
                            smemClasses.markSamples[curIdx] = false;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // push refIdx/refBox forward to next mark
        // only the refIdx thread to push itself. other threads idle
        for (int i = threadIdx.x; i < NClass; i += blockDim.x)
        {
            int refIdx = smemClasses.refIdx[i];
            int endIdx = smemClasses.endIdx[i];
            if (refIdx < endIdx)
            {
                do
                {
                    ++refIdx;
                } while (refIdx < endIdx && smemClasses.markSamples[refIdx] == false);
                smemClasses.refIdx[i] = refIdx;
                if (refIdx < endIdx)
                {
                    smemClasses.refBox[i] = readBbox<__half>(inBbox, blockOffset + inBboxRefIdx[blockOffset + refIdx]);
                }
                else
                {
                    atomicAdd(&smemClasses.done, 1);
                }
            }
        }
        __syncthreads();
    }

    // no need to write all data out
    for (int segment = 0; segment < validSamples; segment += blockDim.x)
    {
        int curIdx = segment + threadIdx.x;
        if (curIdx < validSamples)
        {
            outFlagSamples[blockOffset + curIdx] = (smemClasses.markSamples[curIdx] ? 1 : 0);
        }
    }
}

// TopKGather
// gridDim.x : batch-N
// blockDim.x : Threads
// ItemsPerThreads : = divUp(samples, Threads)
// outDetectionCount : int [N], must be set 0 before kernel
#define MaxItemsPerThreads 8
template <typename DType, typename BoxType, int Threads = 256>
__global__ void TopKGatherProposal_kernel(int samples, int keepTopK, const void* validSampleCountPtr,
    const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr,
    const void* inFlagSamplesPtr, void* outBboxPtr)
{
    typedef BBoxT<BoxType> BBox;
    typedef cub::BlockRadixSort<DType, Threads, 1, int> BlockRadixSort1;
    typedef cub::BlockRadixSort<DType, Threads, 2, int> BlockRadixSort2;
    typedef cub::BlockRadixSort<DType, Threads, 3, int> BlockRadixSort3;
    typedef cub::BlockRadixSort<DType, Threads, 4, int> BlockRadixSort4;
    typedef cub::BlockRadixSort<DType, Threads, 5, int> BlockRadixSort5;
    typedef cub::BlockRadixSort<DType, Threads, 6, int> BlockRadixSort6;
    typedef cub::BlockRadixSort<DType, Threads, 7, int> BlockRadixSort7;
    typedef cub::BlockRadixSort<DType, Threads, 8, int> BlockRadixSort8;
    __shared__ union
    {
        typename BlockRadixSort8::TempStorage sort8;
        typename BlockRadixSort7::TempStorage sort7;
        typename BlockRadixSort6::TempStorage sort6;
        typename BlockRadixSort5::TempStorage sort5;
        typename BlockRadixSort4::TempStorage sort4;
        typename BlockRadixSort3::TempStorage sort3;
        typename BlockRadixSort2::TempStorage sort2;
        typename BlockRadixSort1::TempStorage sort1;
    } temp_storage;
    assert(MaxItemsPerThreads * Threads >= samples);

    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    const DType* inScore = static_cast<const DType*>(inScorePtr);
    const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
    const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
    const int* inFlagSamples = static_cast<const int*>(inFlagSamplesPtr);
    BBox* outBbox = static_cast<BBox*>(outBboxPtr);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int validSamples = validSampleCount[N];
    int finalTopK = dMIN(keepTopK, validSamples);

    int idx[MaxItemsPerThreads];
    DType score[MaxItemsPerThreads];
    int totalItems = (validSamples + (blockDim.x - 1)) / blockDim.x;

    for (int ite = 0; ite < totalItems; ++ite)
    {
        int curIdx = ite * blockDim.x + threadIdx.x;
        if (curIdx < validSamples && inFlagSamples[blockOffset + curIdx])
        {
            idx[ite] = curIdx;
            score[ite] = inScore[blockOffset + curIdx];
        }
        else
        {
            idx[ite] = -1;
            score[ite] = 0.0f;
        }
    }

    switch (totalItems)
    {
    case 0: break;
    case 1:
        BlockRadixSort1(temp_storage.sort1).SortDescendingBlockedToStriped((DType(&)[1]) score, (int(&)[1]) idx);
        break;
    case 2:
        BlockRadixSort2(temp_storage.sort2).SortDescendingBlockedToStriped((DType(&)[2]) score, (int(&)[2]) idx);
        break;
    case 3:
        BlockRadixSort3(temp_storage.sort3).SortDescendingBlockedToStriped((DType(&)[3]) score, (int(&)[3]) idx);
        break;
    case 4:
        BlockRadixSort4(temp_storage.sort4).SortDescendingBlockedToStriped((DType(&)[4]) score, (int(&)[4]) idx);
        break;
    case 5:
        BlockRadixSort5(temp_storage.sort5).SortDescendingBlockedToStriped((DType(&)[5]) score, (int(&)[5]) idx);
        break;
    case 6:
        BlockRadixSort6(temp_storage.sort6).SortDescendingBlockedToStriped((DType(&)[6]) score, (int(&)[6]) idx);
        break;
    case 7:
        BlockRadixSort7(temp_storage.sort7).SortDescendingBlockedToStriped((DType(&)[7]) score, (int(&)[7]) idx);
        break;
    case 8:
        BlockRadixSort8(temp_storage.sort8).SortDescendingBlockedToStriped((DType(&)[8]) score, (int(&)[8]) idx);
        break;
    default: assert(false);
    }
    __syncthreads();

    int outBlockOffset = N * keepTopK;
    int topkItems = (keepTopK + (Threads - 1)) / Threads;
    for (int i = 0; i < topkItems; ++i)
    {
        int curI = i * blockDim.x + threadIdx.x;
        if (curI < keepTopK)
        {
            BBox oB = {(BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f};
            if (curI < finalTopK && idx[i] >= 0 && float(score[i]) > MinValidScore)
            {
                oB = ((BBox*) inBbox)[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
            }
            ((BBox*) outBbox)[outBlockOffset + curI] = oB;
        }
    }
}

#define MaxItemsPerThreads 8
template <typename DType, typename BoxType, int Threads = 256>
__global__ void TopKGather_kernel(int samples, int keepTopK, const void* validSampleCountPtr, const void* inScorePtr,
    const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr, const void* inFlagSamplesPtr,
    void* outDetectionPtr)
{
    typedef BBoxT<BoxType> BBox;
    typedef cub::BlockRadixSort<DType, Threads, 1, int> BlockRadixSort1;
    typedef cub::BlockRadixSort<DType, Threads, 2, int> BlockRadixSort2;
    typedef cub::BlockRadixSort<DType, Threads, 3, int> BlockRadixSort3;
    typedef cub::BlockRadixSort<DType, Threads, 4, int> BlockRadixSort4;
    typedef cub::BlockRadixSort<DType, Threads, 5, int> BlockRadixSort5;
    typedef cub::BlockRadixSort<DType, Threads, 6, int> BlockRadixSort6;
    typedef cub::BlockRadixSort<DType, Threads, 7, int> BlockRadixSort7;
    typedef cub::BlockRadixSort<DType, Threads, 8, int> BlockRadixSort8;
    __shared__ union
    {
        typename BlockRadixSort8::TempStorage sort8;
        typename BlockRadixSort7::TempStorage sort7;
        typename BlockRadixSort6::TempStorage sort6;
        typename BlockRadixSort5::TempStorage sort5;
        typename BlockRadixSort4::TempStorage sort4;
        typename BlockRadixSort3::TempStorage sort3;
        typename BlockRadixSort2::TempStorage sort2;
        typename BlockRadixSort1::TempStorage sort1;
    } temp_storage;
    assert(MaxItemsPerThreads * Threads >= samples);

    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    const DType* inScore = static_cast<const DType*>(inScorePtr);
    const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr); // InLabel keeps INT32
    const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
    const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
    const int* inFlagSamples = static_cast<const int*>(inFlagSamplesPtr);
    DType* outDetections = static_cast<DType*>(outDetectionPtr);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int validSamples = validSampleCount[N];
    int finalTopK = dMIN(keepTopK, validSamples);

    int idx[MaxItemsPerThreads];
    DType score[MaxItemsPerThreads];
    int totalItems = (validSamples + (blockDim.x - 1)) / blockDim.x;

    for (int ite = 0; ite < totalItems; ++ite)
    {
        int curIdx = ite * blockDim.x + threadIdx.x;
        if (curIdx < validSamples && inFlagSamples[blockOffset + curIdx])
        {
            idx[ite] = curIdx;
            score[ite] = inScore[blockOffset + curIdx];
        }
        else
        {
            idx[ite] = -1;
            score[ite] = 0.0f;
        }
    }

    switch (totalItems)
    {
    case 0: break;
    case 1:
        BlockRadixSort1(temp_storage.sort1).SortDescendingBlockedToStriped((DType(&)[1]) score, (int(&)[1]) idx);
        break;
    case 2:
        BlockRadixSort2(temp_storage.sort2).SortDescendingBlockedToStriped((DType(&)[2]) score, (int(&)[2]) idx);
        break;
    case 3:
        BlockRadixSort3(temp_storage.sort3).SortDescendingBlockedToStriped((DType(&)[3]) score, (int(&)[3]) idx);
        break;
    case 4:
        BlockRadixSort4(temp_storage.sort4).SortDescendingBlockedToStriped((DType(&)[4]) score, (int(&)[4]) idx);
        break;
    case 5:
        BlockRadixSort5(temp_storage.sort5).SortDescendingBlockedToStriped((DType(&)[5]) score, (int(&)[5]) idx);
        break;
    case 6:
        BlockRadixSort6(temp_storage.sort6).SortDescendingBlockedToStriped((DType(&)[6]) score, (int(&)[6]) idx);
        break;
    case 7:
        BlockRadixSort7(temp_storage.sort7).SortDescendingBlockedToStriped((DType(&)[7]) score, (int(&)[7]) idx);
        break;
    case 8:
        BlockRadixSort8(temp_storage.sort8).SortDescendingBlockedToStriped((DType(&)[8]) score, (int(&)[8]) idx);
        break;
    default: assert(false);
    }
    __syncthreads();

    int outBlockOffset = N * keepTopK;
    int topkItems = (keepTopK + (Threads - 1)) / Threads;
    for (int i = 0; i < topkItems; ++i)
    {
        int curI = i * blockDim.x + threadIdx.x;
        if (curI < keepTopK)
        {
            BBox oB = {(BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f};
            DType oS = 0.0f;
            BoxType oL = -1;
            if (curI < finalTopK && idx[i] >= 0 && float(score[i]) > MinValidScore)
            {
                oB = ((BBox*) inBbox)[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
                oS = score[i];
                oL = (BoxType) inLabel[blockOffset + idx[i]];
            }
            outDetections[(outBlockOffset + curI) * 6] = oB.y1;
            outDetections[(outBlockOffset + curI) * 6 + 1] = oB.x1;
            outDetections[(outBlockOffset + curI) * 6 + 2] = oB.y2;
            outDetections[(outBlockOffset + curI) * 6 + 3] = oB.x2;
            outDetections[(outBlockOffset + curI) * 6 + 4] = oL;
            outDetections[(outBlockOffset + curI) * 6 + 5] = oS;
        }
    }
}

RefineDetectionWorkSpace::RefineDetectionWorkSpace(
    const int batchSize, const int sampleCount, const RefineNMSParameters& param, const nvinfer1::DataType inType)
    : argMaxScoreDims(sampleCount, 1)
    , argMaxBboxDims(sampleCount, 4)
    , argMaxLabelDims(sampleCount, 1)
    , sortClassScoreDims(sampleCount, 1)
    , sortClassLabelDims(sampleCount, 1)
    , sortClassSampleIdxDims(sampleCount + 1, 1)
    , sortClassPosDims(param.numClasses + 1, 1)
    , sortNMSMarkDims(sampleCount, 1)
{
    size_t sumSize = 0;

    const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;

    // resource
    // arMaxScore : [N, samples] : m_Type
    argMaxScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(argMaxScoreDims) * typeSize(type) * batchSize);

    argMaxBboxOffset = sumSize;
    // argMaxBbox : [N, samples, 4] : m_Type
    sumSize += AlignMem(dimVolume(argMaxBboxDims) * typeSize(type) * batchSize);

    argMaxLabelOffset = sumSize;
    // argMaxLabel : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(argMaxLabelDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassScoreOffset = sumSize;
    // sortClassScore : [N, samples] : m_Type
    sumSize += AlignMem(dimVolume(sortClassScoreDims) * typeSize(type) * batchSize);

    sortClassLabelOffset = sumSize;
    // sortClassLabel : [N, samples] : m_Type
    sumSize += AlignMem(dimVolume(sortClassLabelDims) * typeSize(type) * batchSize);

    sortClassSampleIdxOffset = sumSize;
    // sortClassSampleIdx : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(sortClassSampleIdxDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassValidCountOffset = sumSize;
    // sortClassValidCount : [N, 1] : kINT32
    sumSize += AlignMem(dimVolume(sortClassValidCountDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassPosOffset = sumSize;
    // sortClassPos : [N, numClasses+1] : kINT32
    sumSize += AlignMem(dimVolume(sortClassPosDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortNMSMarkOffset = sumSize;
    // sortNMSMark : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(sortNMSMarkDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    totalSize = sumSize;
}

ProposalWorkSpace::ProposalWorkSpace(const int batchSize, const int inputCnt, const int sampleCount,
    const RefineNMSParameters& param, const nvinfer1::DataType inType)
    : preRefineScoreDims(inputCnt, 1)
    , preRefineSortedScoreDims(inputCnt, 1)
    , preRefineBboxDims(inputCnt, 4)
    , argMaxScoreDims(sampleCount, 1)
    , argMaxBboxDims(sampleCount, 4)
    , argMaxLabelDims(sampleCount, 1)
    , sortClassScoreDims(sampleCount, 1)
    , sortClassLabelDims(sampleCount, 1)
    , sortClassSampleIdxDims(sampleCount, 1)
    , sortClassPosDims(param.numClasses + 1, 1)
    , sortNMSMarkDims(sampleCount, 1)
{
    size_t sumSize = 0;

    const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;

    // resource
    // temp storage size for sorting scores
    tempStorageOffset = sumSize;
    sumSize += (1 << 23) * batchSize;

    // preRefineScore : [N, inputcnt, 1] // extracted foreground score from inputs[0]
    preRefineScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(preRefineScoreDims) * typeSize(type) * batchSize);

    // preRefineSortedScore: [N, inputcnt, 1]
    preRefineSortedScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(preRefineSortedScoreDims) * typeSize(type) * batchSize);

    // preRefineBbox: [N, inputcnt, 4] // sorted bbox
    preRefineBboxOffset = sumSize;
    sumSize += AlignMem(dimVolume(preRefineBboxDims) * typeSize(type) * batchSize);

    // arMaxScore : [N, samples] : m_Type
    argMaxScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(argMaxScoreDims) * typeSize(type) * batchSize);

    argMaxBboxOffset = sumSize;
    // argMaxBbox : [N, samples, 4] : m_Type
    sumSize += AlignMem(dimVolume(argMaxBboxDims) * typeSize(type) * batchSize);

    argMaxLabelOffset = sumSize;
    // argMaxLabel : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(argMaxLabelDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassScoreOffset = sumSize;
    // sortClassScore : [N, samples] : m_Type
    sumSize += AlignMem(dimVolume(sortClassScoreDims) * typeSize(type) * batchSize);

    sortClassLabelOffset = sumSize;
    // sortClassLabel : [N, samples] : m_Type
    sumSize += AlignMem(dimVolume(sortClassLabelDims) * typeSize(type) * batchSize);

    sortClassSampleIdxOffset = sumSize;
    // sortClassSampleIdx : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(sortClassSampleIdxDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassValidCountOffset = sumSize;
    // sortClassValidCount : [N, 1] : kINT32
    sumSize += AlignMem(dimVolume(sortClassValidCountDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassPosOffset = sumSize;
    // sortClassPos : [N, numClasses+1] : kINT32
    sumSize += AlignMem(dimVolume(sortClassPosDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortNMSMarkOffset = sumSize;
    // sortNMSMark : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(sortNMSMarkDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    totalSize = sumSize;
}

MultilevelProposeROIWorkSpace::MultilevelProposeROIWorkSpace(const int batchSize, const int inputCnt,
    const int sampleCount, const RefineNMSParameters& param, const nvinfer1::DataType inType)
    : preRefineSortedScoreDims(inputCnt, 1)
    , preRefineBboxDims(inputCnt, 4)
    , argMaxScoreDims(sampleCount, 1)
    , argMaxBboxDims(sampleCount, 4)
    , argMaxLabelDims(sampleCount, 1)
    , sortClassScoreDims(sampleCount, 1)
    , sortClassLabelDims(sampleCount, 1)
    , sortClassSampleIdxDims(sampleCount + 1, 1)
    , sortClassPosDims(param.numClasses + 1, 1)
    , sortNMSMarkDims(sampleCount, 1)
{
    size_t sumSize = 0;

    const nvinfer1::DataType type = inType;

    // resource
    // temp storage size for sorting scores
    tempStorageOffset = sumSize;
    sumSize += (1 << 23) * batchSize;

    // preRefineSortedScore: [N, inputcnt, 1]
    preRefineSortedScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(preRefineSortedScoreDims) * typeSize(type) * batchSize);

    // preRefineBbox: [N, inputcnt, 4] // sorted bbox
    preRefineBboxOffset = sumSize;
    sumSize += AlignMem(dimVolume(preRefineBboxDims) * typeSize(type) * batchSize);

    // argMaxScore : [N, samples] : m_Type
    argMaxScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(argMaxScoreDims) * typeSize(type) * batchSize);

    argMaxBboxOffset = sumSize;
    // argMaxBbox : [N, samples, 4] : m_Type
    sumSize += AlignMem(dimVolume(argMaxBboxDims) * typeSize(type) * batchSize);

    argMaxLabelOffset = sumSize;
    // argMaxLabel : [N, samples] : m_Type
    sumSize += AlignMem(dimVolume(argMaxLabelDims) * typeSize(type) * batchSize);

    sortClassScoreOffset = sumSize;
    // sortClassScore : [N, samples] : m_Type
    sumSize += AlignMem(dimVolume(sortClassScoreDims) * typeSize(type) * batchSize);

    sortClassLabelOffset = sumSize;
    // sortClassLabel : [N, samples] : m_Type
    sumSize += AlignMem(dimVolume(sortClassLabelDims) * typeSize(type) * batchSize);

    sortClassSampleIdxOffset = sumSize;
    // sortClassSampleIdx : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(sortClassSampleIdxDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassValidCountOffset = sumSize;
    // sortClassValidCount : [N, 1] : kINT32
    sumSize += AlignMem(dimVolume(sortClassValidCountDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortClassPosOffset = sumSize;
    // sortClassPos : [N, numClasses+1] : kINT32
    sumSize += AlignMem(dimVolume(sortClassPosDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    sortNMSMarkOffset = sumSize;
    // sortNMSMark : [N, samples] : kINT32
    sumSize += AlignMem(dimVolume(sortNMSMarkDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

    totalSize = sumSize;
}

ConcatTopKWorkSpace::ConcatTopKWorkSpace(
    const int batchSize, const int concatCnt, const int topK, const nvinfer1::DataType inType)
    : concatedScoreDims(concatCnt * topK, 1)
    , concatedBBoxDims(concatCnt * topK, 4)
    , sortedScoreDims(concatCnt * topK, 1)
    , sortedBBoxDims(concatCnt * topK, 4)
{
    size_t sumSize = 0;

    // const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
    const nvinfer1::DataType type = inType;

    // resource
    // temp storage size for sorting scores
    tempStorageOffset = sumSize;
    sumSize += (1 << 23) * batchSize;

    // concatedScoreOffset: [N, concatCnt*topK, 1]
    concatedScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(concatedScoreDims) * typeSize(type) * batchSize);

    // concatedBBoxOffset: [N, concatCnt*topK, 4]
    concatedBBoxOffset = sumSize;
    sumSize += AlignMem(dimVolume(concatedBBoxDims) * typeSize(type) * batchSize);

    // sortedScoreOffset: [N, concatCnt * topK, 1]
    sortedScoreOffset = sumSize;
    sumSize += AlignMem(dimVolume(sortedScoreDims) * typeSize(type) * batchSize);

    // sortedBBoxOffset: [N, concatCnt * topK, 4]
    sortedBBoxOffset = sumSize;
    sumSize += AlignMem(dimVolume(sortedBBoxDims) * typeSize(type) * batchSize);

    totalSize = sumSize;
}

template <int Threads>
cudaError_t argMaxGroup(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
    const void* inScore, const void* inBbox, const void* validSamples, void* outScore, void* outLabel, void* outBbox)
{
    int gridX = nAlignDown(dMIN(samples, 512 / N), 32);
    gridX = dMAX(gridX, 1);

    dim3 gridDim = {static_cast<unsigned int>(gridX), static_cast<unsigned int>(N), 1};
    dim3 threads = {Threads, 1, 1};
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        argMaxGroup_kernel<float, float, Threads><<<gridDim, threads, 0, stream>>>(
            samples, 0, NClass, inScore, inBbox, validSamples, outScore, outLabel, outBbox);
        break;
    case nvinfer1::DataType::kHALF: break;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kINT64: PLUGIN_FAIL("Unsupported data type");

    default: PLUGIN_ASSERT(false);
    }

    return cudaGetLastError();
}

template <int Threads>
cudaError_t argMaxWOBackground(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
    const void* inScore, const void* inBbox, const void* validSamples, void* outScore, void* outLabel, void* outBbox)
{
    int gridX = nAlignDown(dMIN(samples, 512 / N), 32);
    gridX = dMAX(gridX, 1);

    dim3 gridDim = {static_cast<unsigned int>(gridX), static_cast<unsigned int>(N), 1};
    dim3 threads = {Threads, 1, 1};
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        argMaxGroup_kernel<float, float, Threads><<<gridDim, threads, 0, stream>>>(
            samples, 1, NClass, inScore, inBbox, validSamples, outScore, outLabel, outBbox);
        break;
    case nvinfer1::DataType::kHALF: break;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kINT64: PLUGIN_FAIL("Unsupported data type");

    default: PLUGIN_ASSERT(false);
    }

    return cudaGetLastError();
}

template <int Threads, int ItermPerThreads>
cudaError_t sortPerClass(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass, int background,
    float scoreThreshold, const void* inSampleValidCount, const void* inScorePtr, const void* inLabelPtr,
    const void* inBboxPtr, void* outclassStartPosPtr, void* outScorePtr, void* outLabelPtr, void* outSampleIdxPtr,
    void* outValidSampleCountPtr)
{
    int blocks = N;
    int threads = Threads;

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        sortPerClass_kernel<float, float, Threads, ItermPerThreads><<<blocks, threads, 0, stream>>>(samples, NClass,
            background, scoreThreshold, inSampleValidCount, inScorePtr, inLabelPtr, inBboxPtr, outclassStartPosPtr,
            outScorePtr, outLabelPtr, outSampleIdxPtr, outValidSampleCountPtr);
        break;
    case nvinfer1::DataType::kHALF:
        sortPerClass_kernel_half<Threads, ItermPerThreads><<<blocks, threads, 0, stream>>>(samples, NClass,
            background, scoreThreshold, inSampleValidCount, inScorePtr, inLabelPtr, inBboxPtr, outclassStartPosPtr,
            outScorePtr, outLabelPtr, outSampleIdxPtr, outValidSampleCountPtr);
        break;
    default: PLUGIN_ASSERT(false);
    }

    return cudaGetLastError();
};

template <int Threads>
cudaError_t PerClassNMS(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
    const float nmsThreshold, const void* validSampleCount,
    // const void *inScore,
    const void* inLabel, const void* inBbox, const void* inBboxRefIdx, const void* classStarts, void* outFlagSamples)
{
    int blocks = N;
    int threads = Threads;

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        PerClassNMS_kernel<float, float, Threads><<<blocks, threads, 0, stream>>>(samples, NClass, nmsThreshold,
            validSampleCount, inLabel, inBbox, inBboxRefIdx, classStarts, outFlagSamples);
        break;
    case nvinfer1::DataType::kHALF:
        PerClassNMS_half_kernel<Threads><<<blocks, threads, 0, stream>>>(samples, NClass, nmsThreshold,
            validSampleCount, inLabel, inBbox, inBboxRefIdx, classStarts, outFlagSamples);
        break;
    default: PLUGIN_ASSERT(false);
    }

    return cudaGetLastError();
}

template <int Threads>
cudaError_t KeepTopKGather(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int keepTopK,
    const void* validSampleCountPtr, const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr,
    const void* inBboxRefIdxPtr, const void* inFlagSamplesPtr, void* outDetections, int proposal)
{
    int blocks = N;
    int threads = Threads;

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        if (proposal)
        {
            TopKGatherProposal_kernel<float, float, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
                validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
                outDetections);
        }
        else
        {
            TopKGather_kernel<float, float, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
                validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
                outDetections);
        }
        break;
    case nvinfer1::DataType::kHALF: break;
    default: PLUGIN_ASSERT(false);
    }

    return cudaGetLastError();
}

// TopKGather For TLT RPN Proposal
// gridDim.x : batch-N
// blockDim.x : Threads
// ItemsPerThreads : = divUp(samples, Threads)
// outDetectionCount : int [N], must be set 0 before kernel
#define MaxItemsPerThreads 8
template <typename DType, typename BoxType, int Threads = 256>
__global__ void TopKGatherBoxScore_kernel(int samples, int keepTopK, const void* validSampleCountPtr,
    const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr,
    const void* inFlagSamplesPtr, void* outScorePtr, void* outBboxPtr)
{
    typedef cub::BlockRadixSort<DType, Threads, 1, int> BlockRadixSort1;
    typedef cub::BlockRadixSort<DType, Threads, 2, int> BlockRadixSort2;
    typedef cub::BlockRadixSort<DType, Threads, 3, int> BlockRadixSort3;
    typedef cub::BlockRadixSort<DType, Threads, 4, int> BlockRadixSort4;
    typedef cub::BlockRadixSort<DType, Threads, 5, int> BlockRadixSort5;
    typedef cub::BlockRadixSort<DType, Threads, 6, int> BlockRadixSort6;
    typedef cub::BlockRadixSort<DType, Threads, 7, int> BlockRadixSort7;
    typedef cub::BlockRadixSort<DType, Threads, 8, int> BlockRadixSort8;
    __shared__ union {
        typename BlockRadixSort8::TempStorage sort8;
        typename BlockRadixSort7::TempStorage sort7;
        typename BlockRadixSort6::TempStorage sort6;
        typename BlockRadixSort5::TempStorage sort5;
        typename BlockRadixSort4::TempStorage sort4;
        typename BlockRadixSort3::TempStorage sort3;
        typename BlockRadixSort2::TempStorage sort2;
        typename BlockRadixSort1::TempStorage sort1;
    } temp_storage;
    assert(MaxItemsPerThreads * Threads >= samples);

    typedef BBoxT<BoxType> BBox;
    const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
    const DType* inScore = static_cast<const DType*>(inScorePtr);
    const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
    const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
    const int* inFlagSamples = static_cast<const int*>(inFlagSamplesPtr);
    BBox* outBbox = static_cast<BBox*>(outBboxPtr);
    DType* outScore = static_cast<DType*>(outScorePtr);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int validSamples = validSampleCount[N];
    int finalTopK = dMIN(keepTopK, validSamples);

    int idx[MaxItemsPerThreads];
    DType score[MaxItemsPerThreads];
    int totalItems = (validSamples + (blockDim.x - 1)) / blockDim.x;

    for (int ite = 0; ite < totalItems; ++ite)
    {
        int curIdx = ite * blockDim.x + threadIdx.x;
        if (curIdx < validSamples && inFlagSamples[blockOffset + curIdx])
        {
            idx[ite] = curIdx;
            score[ite] = inScore[blockOffset + curIdx];
        }
        else
        {
            idx[ite] = -1;
            score[ite] = 0.0f;
        }
    }

    switch (totalItems)
    {
    case 0: break;
    case 1:
        BlockRadixSort1(temp_storage.sort1).SortDescendingBlockedToStriped((DType(&)[1]) score, (int(&)[1]) idx);
        break;
    case 2:
        BlockRadixSort2(temp_storage.sort2).SortDescendingBlockedToStriped((DType(&)[2]) score, (int(&)[2]) idx);
        break;
    case 3:
        BlockRadixSort3(temp_storage.sort3).SortDescendingBlockedToStriped((DType(&)[3]) score, (int(&)[3]) idx);
        break;
    case 4:
        BlockRadixSort4(temp_storage.sort4).SortDescendingBlockedToStriped((DType(&)[4]) score, (int(&)[4]) idx);
        break;
    case 5:
        BlockRadixSort5(temp_storage.sort5).SortDescendingBlockedToStriped((DType(&)[5]) score, (int(&)[5]) idx);
        break;
    case 6:
        BlockRadixSort6(temp_storage.sort6).SortDescendingBlockedToStriped((DType(&)[6]) score, (int(&)[6]) idx);
        break;
    case 7:
        BlockRadixSort7(temp_storage.sort7).SortDescendingBlockedToStriped((DType(&)[7]) score, (int(&)[7]) idx);
        break;
    case 8:
        BlockRadixSort8(temp_storage.sort8).SortDescendingBlockedToStriped((DType(&)[8]) score, (int(&)[8]) idx);
        break;
    default: assert(false);
    }
    __syncthreads();

    int outBlockOffset = N * keepTopK;
    int topkItems = (keepTopK + (Threads - 1)) / Threads;
    for (int i = 0; i < topkItems; ++i)
    {
        int curI = i * blockDim.x + threadIdx.x;
        if (curI < keepTopK)
        {
            BBox oB = {(BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f};
            DType oS = 0.0f;
            if (curI < finalTopK && idx[i] >= 0)
            {
                oB = ((BBox*) inBbox)[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
                oS = score[i];
            }
            ((BBox*) outBbox)[outBlockOffset + curI] = oB;
            outScore[outBlockOffset + curI] = oS;
        }
    }
}

template <int Threads>
cudaError_t KeepTopKGatherBoxScore(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int keepTopK,
    const void* validSampleCountPtr, const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr,
    const void* inBboxRefIdxPtr, const void* inFlagSamplesPtr, void* outScores, void* outDetections, int proposal)
{
    int blocks = N;
    int threads = Threads;

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        if (proposal)
        {
            TopKGatherBoxScore_kernel<float, float, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
                validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr, outScores,
                outDetections);
        }
        else
        {
            TopKGather_kernel<float, float, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
                validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
                outDetections);
        }
        break;
    case nvinfer1::DataType::kHALF:
        if (proposal)
        {
            TopKGatherBoxScore_kernel<__half, __half, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
                validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr, outScores,
                outDetections);
        }
        else
        {
            TopKGather_kernel<__half, __half, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
                validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
                outDetections);
        }
        break;
    default: PLUGIN_ASSERT(false);
    }

    return cudaGetLastError();
}

cudaError_t RefineBatchClassNMS(cudaStream_t stream, int N, int samples, nvinfer1::DataType dtype,
    const RefineNMSParameters& param, const RefineDetectionWorkSpace& refineOffset, void* workspace,
    const void* inScores, const void* inDelta, const void* inCountValid, const void* inROI, void* outDetections)
{
    int NClass = param.numClasses;
    int8_t* wsPtr = static_cast<int8_t*>(workspace);
    void* argMaxScorePtr = wsPtr + refineOffset.argMaxScoreOffset;
    void* argMaxLabelPtr = wsPtr + refineOffset.argMaxLabelOffset;
    void* argMaxBBoxPtr = wsPtr + refineOffset.argMaxBboxOffset;

    void* sortClassScorePtr = wsPtr + refineOffset.sortClassScoreOffset;
    void* sortClassLabelPtr = wsPtr + refineOffset.sortClassLabelOffset;
    void* sortClassSampleIdxPtr = wsPtr + refineOffset.sortClassSampleIdxOffset;
    void* sortClassValidCountPtr = wsPtr + refineOffset.sortClassValidCountOffset;
    void* sortClassPosPtr = wsPtr + refineOffset.sortClassPosOffset;
    void* sortNMSMarkPtr = wsPtr + refineOffset.sortNMSMarkOffset;

    cudaError_t status = cudaSuccess;
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));

    if (NClass > 1)
    { // multiple classes
        status = argMaxGroup<32>(stream, N, dtype, samples, NClass, inScores, inDelta, inCountValid, argMaxScorePtr,
            argMaxLabelPtr, argMaxBBoxPtr); // argMaxBBoxPtr means delta of bboxes
        PLUGIN_ASSERT(status == cudaSuccess);
        PLUGIN_CUASSERT(status);
    }
    else
    { // Only one class
        argMaxScorePtr = const_cast<void*>(inScores);
        argMaxBBoxPtr = const_cast<void*>(inDelta);
        int threads = 512;
        int blocks = (N * samples + threads - 1) / threads;
        blocks = dMIN(blocks, 8);
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
        {
            resetMemValue_kernel<float><<<blocks, threads, 0, stream>>>(argMaxLabelPtr, N * samples, 0);
            break;
        }
        case nvinfer1::DataType::kHALF: { break;
        }
        default: PLUGIN_ASSERT(false);
        }
    }

    status = ApplyDelta2Bboxes(stream, N, samples, inROI, argMaxBBoxPtr, argMaxBBoxPtr);
    PLUGIN_ASSERT(status == cudaSuccess);

    if (samples <= 1024)
    {
        status = sortPerClass<256, 4>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 2048)
    {
        status = sortPerClass<256, 8>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 4096)
    {
        status = sortPerClass<256, 16>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else
    {
        PLUGIN_ASSERT(false && "unsupported sortPerClass");
        return cudaErrorLaunchFailure;
    }
    PLUGIN_ASSERT(status == cudaSuccess);
    PLUGIN_CUASSERT(status);

    status = PerClassNMS<256>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
        // sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    PLUGIN_ASSERT(status == cudaSuccess);
    PLUGIN_CUASSERT(status);

    status = KeepTopKGather<256>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outDetections, 0);
    PLUGIN_ASSERT(status == cudaSuccess);
    PLUGIN_CUASSERT(status);
    return status;
}

cudaError_t DetectionPostProcess(cudaStream_t stream, int N, int samples, const float* regWeight,
    const float inputHeight, const float inputWidth, nvinfer1::DataType dtype, const RefineNMSParameters& param,
    const RefineDetectionWorkSpace& refineOffset, void* workspace, const void* inScores, const void* inDelta,
    const void* inCountValid, const void* inROI, void* outDetections)
{
    int NClass = param.numClasses;
    int8_t* wsPtr = static_cast<int8_t*>(workspace);
    void* argMaxScorePtr = wsPtr + refineOffset.argMaxScoreOffset;
    void* argMaxLabelPtr = wsPtr + refineOffset.argMaxLabelOffset;
    void* argMaxBBoxPtr = wsPtr + refineOffset.argMaxBboxOffset;

    void* sortClassScorePtr = wsPtr + refineOffset.sortClassScoreOffset;
    void* sortClassLabelPtr = wsPtr + refineOffset.sortClassLabelOffset;
    void* sortClassSampleIdxPtr = wsPtr + refineOffset.sortClassSampleIdxOffset;
    void* sortClassValidCountPtr = wsPtr + refineOffset.sortClassValidCountOffset;
    void* sortClassPosPtr = wsPtr + refineOffset.sortClassPosOffset;
    void* sortNMSMarkPtr = wsPtr + refineOffset.sortNMSMarkOffset;

    cudaError_t status = cudaSuccess;
    PLUGIN_CUASSERT(cudaMemsetAsync(argMaxScorePtr, 0, N * samples * sizeof(float), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(argMaxBBoxPtr, 0, N * samples * 4 * sizeof(float), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassPosPtr, 0, N * (NClass + 1) * sizeof(int), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassSampleIdxPtr, 0, N * (samples + 1) * sizeof(int), stream));

    if (NClass > 1)
    { // multiple classes
        status = argMaxWOBackground<32>(stream, N, dtype, samples, NClass, inScores, inDelta, inCountValid,
            argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr); // argMaxBBoxPtr means delta of bboxes
        PLUGIN_ASSERT(status == cudaSuccess);
        PLUGIN_CUASSERT(status);
    }
    else
    { // Only one class
        argMaxScorePtr = const_cast<void*>(inScores);
        argMaxBBoxPtr = const_cast<void*>(inDelta);
        int threads = 512;
        int blocks = (N * samples + threads - 1) / threads;
        blocks = dMIN(blocks, 8);
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
        {
            resetMemValue_kernel<float><<<blocks, threads, 0, stream>>>(argMaxLabelPtr, N * samples, 0);
            break;
        }
        case nvinfer1::DataType::kHALF: { break;
        }
        default: PLUGIN_ASSERT(false);
        }
    }

    status = DecodeBBoxes(stream, N, samples, regWeight, inputHeight, inputWidth, inROI, argMaxBBoxPtr, argMaxBBoxPtr, dtype);
    PLUGIN_ASSERT(status == cudaSuccess);

    if (samples <= 1024)
    {
        status = sortPerClass<256, 4>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 2048)
    {
        status = sortPerClass<256, 8>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 4096)
    {
        status = sortPerClass<256, 16>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else
    {
        PLUGIN_ASSERT(false && "unsupported sortPerClass");
        return cudaErrorLaunchFailure;
    }
    PLUGIN_ASSERT(status == cudaSuccess);
    PLUGIN_CUASSERT(status);

    status = PerClassNMS<256>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
        // sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    PLUGIN_CUASSERT(status);

    status = KeepTopKGather<256>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outDetections, 0);
    PLUGIN_CUASSERT(status);

    return status;
}

struct BF_SCORE
{
    float bg, fg;
};
// in_scores : [N, samples, 2]
// output_score : [N, samples, 1]
__global__ void extract_fg_kernel(int samples, const void* in_scores, void* output_score)
{
    const BF_SCORE* in = static_cast<const BF_SCORE*>(in_scores);
    float* out = static_cast<float*>(output_score);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;
        if (cur_id < samples)
        {
            out[blockOffset + cur_id] = in[blockOffset + cur_id].fg;
        }
    }
}
__global__ void set_offset_kernel(int stride, int size, int* output)
{
    // One block, because batch size shouldn't be too large.
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        output[i] = i * stride;
    }
}

template <typename Dtype>
__global__ void resample_kernel(int orig_size, int sample_size, const void* orig_score_ptr, const void* orig_bbox_ptr,
    void* sampled_score_ptr, void* sampled_bbox_ptr)
{
    const Dtype* in_score = static_cast<const Dtype*>(orig_score_ptr);
    const BBoxT<Dtype>* in_bbox = static_cast<const BBoxT<Dtype>*>(orig_bbox_ptr);
    Dtype* out_score = static_cast<Dtype*>(sampled_score_ptr);
    BBoxT<Dtype>* out_bbox = static_cast<BBoxT<Dtype>*>(sampled_bbox_ptr);

    int N = blockIdx.x;
    int blockOffset_in = N * orig_size;
    int blockOffset_out = N * sample_size;
    int realSampleCnt = dMIN(sample_size, orig_size);
    int totalItems = (realSampleCnt + (blockDim.x - 1)) / blockDim.x;

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;
        if (cur_id < realSampleCnt)
        {
            out_score[blockOffset_out + cur_id] = in_score[blockOffset_in + cur_id];
            out_bbox[blockOffset_out + cur_id] = in_bbox[blockOffset_in + cur_id];
        }
    }
}

cudaError_t proposalRefineBatchClassNMS(cudaStream_t stream, int N, int inputCnt, int samples, nvinfer1::DataType dtype,
    const RefineNMSParameters& param, const ProposalWorkSpace& proposalOffset, void* workspace,
    const void* inScores, //[N, inputcnt, 2]
    const void* inDelta,  //[N, inputcnt, 4]
    const void* inCountValid,
    const void* inAnchors, //[N, inputcnt, 4]
    void* outProposals)
{
    int8_t* wsPtr = static_cast<int8_t*>(workspace);
    void* tempStoragePtr = wsPtr + proposalOffset.tempStorageOffset;
    void* preRefineScorePtr = wsPtr + proposalOffset.preRefineScoreOffset;
    void* preRefineSortedScorePtr = wsPtr + proposalOffset.preRefineSortedScoreOffset;
    void* preRefineBboxPtr = wsPtr + proposalOffset.preRefineBboxOffset;

    void* argMaxScorePtr = wsPtr + proposalOffset.argMaxScoreOffset;
    void* argMaxLabelPtr = wsPtr + proposalOffset.argMaxLabelOffset;
    void* argMaxBBoxPtr = wsPtr + proposalOffset.argMaxBboxOffset;

    void* sortClassScorePtr = wsPtr + proposalOffset.sortClassScoreOffset;
    void* sortClassLabelPtr = wsPtr + proposalOffset.sortClassLabelOffset;
    void* sortClassSampleIdxPtr = wsPtr + proposalOffset.sortClassSampleIdxOffset;
    void* sortClassValidCountPtr = wsPtr + proposalOffset.sortClassValidCountOffset;
    void* sortClassPosPtr = wsPtr + proposalOffset.sortClassPosOffset;
    void* sortNMSMarkPtr = wsPtr + proposalOffset.sortNMSMarkOffset;

    cudaError_t status = cudaSuccess;
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));

    // extract foreground score
    extract_fg_kernel<<<N, dMIN(inputCnt, 1024), 0, stream>>>(inputCnt, inScores, preRefineScorePtr);
    PLUGIN_CUASSERT(cudaGetLastError());

    // Here, inDelta are converted to normalize coordinates based on anchors
    status = ApplyDelta2Bboxes(stream, N, inputCnt, inAnchors, inDelta, const_cast<void*>(inDelta));
    PLUGIN_CUASSERT(status);

    // sort the score
    // d_key_in: preRefineScorePtr [N, inputCnt, 1]
    // d_key_out: preRefineSortedScorePtr
    // d_values_in: inDelta [N, inputCnt, 4]
    // d_values_out: preRefineBboxPtr
    // num_items: inputCnt*N
    // num_segments: N
    // offsets: [0, inputCnt, inputCnt*2, ..., ]
    int* offsets = static_cast<int*>(tempStoragePtr);
    set_offset_kernel<<<1, 1024, 0, stream>>>(inputCnt, N + 1, offsets);
    PLUGIN_ASSERT(cudaGetLastError() == cudaSuccess);
    tempStoragePtr = static_cast<void*>(static_cast<int*>(tempStoragePtr) + (N + 1));

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(NULL, temp_storage_bytes, (float*) preRefineScorePtr,
        (float*) preRefineSortedScorePtr, (BBoxT<float>*) inDelta, (BBoxT<float>*) preRefineBboxPtr, N * inputCnt, N,
        offsets, offsets + 1, 0, 8 * sizeof(float), stream);

    PLUGIN_ASSERT((1 << 23) * (size_t) N > temp_storage_bytes);

    cub::DeviceSegmentedRadixSort::SortPairsDescending(tempStoragePtr, temp_storage_bytes, (float*) preRefineScorePtr,
        (float*) preRefineSortedScorePtr, (BBoxT<float>*) inDelta, (BBoxT<float>*) preRefineBboxPtr, N * inputCnt, N,
        offsets, offsets + 1, 0, 8 * sizeof(float), stream);

    int NClass = param.numClasses;
    PLUGIN_ASSERT(NClass == 1);
    if (NClass == 1)
    { // Only one class
        resample_kernel<float><<<N, dMIN(samples, 1024), 0, stream>>>(
            inputCnt, samples, preRefineSortedScorePtr, preRefineBboxPtr, argMaxScorePtr, argMaxBBoxPtr);

        int threads = 512;
        int blocks = (N * samples + threads - 1) / threads;
        blocks = dMIN(blocks, 8);
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
        {
            resetMemValue_kernel<float><<<blocks, threads, 0, stream>>>(argMaxLabelPtr, N * samples, 0);
            break;
        }
        case nvinfer1::DataType::kHALF: { break;
        }
        default: PLUGIN_ASSERT(false);
        }
    }

    if (samples <= 1024)
    {
        status = sortPerClass<256, 4>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 2048)
    {
        status = sortPerClass<256, 8>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 4096)
    {
        status = sortPerClass<256, 16>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else
    {
        PLUGIN_ASSERT(false && "unsupported sortPerClass");
        return cudaErrorLaunchFailure;
    }
    PLUGIN_CUASSERT(status);

    status = PerClassNMS<256>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
        // sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    PLUGIN_CUASSERT(status);

    status = KeepTopKGather<256>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outProposals, 1);
    PLUGIN_CUASSERT(status);

    return status;
}

template<typename Dtype>
void score_bbox_cub_sort(void* tempStorage,
                         const void* inScore,
                         void* sortedScore,
                         const void* inBBox,
                         void* sortedBBox,
                         int totalCnt,
                         int segCnt,
                         int* offsets,
                         cudaStream_t stream
                         )
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(NULL, temp_storage_bytes, (Dtype*) inScore, (Dtype*) sortedScore,
        (BBoxT<Dtype>*) inBBox, (BBoxT<Dtype>*) sortedBBox, totalCnt, segCnt, offsets, offsets + 1, 0,
        8 * sizeof(Dtype), stream);
    PLUGIN_CUASSERT(cudaGetLastError());

    cub::DeviceSegmentedRadixSort::SortPairsDescending(tempStorage, temp_storage_bytes, (Dtype*) inScore,
        (Dtype*) sortedScore, (BBoxT<Dtype>*) inBBox, (BBoxT<Dtype>*) sortedBBox, totalCnt, segCnt, offsets,
        offsets + 1, 0, 8 * sizeof(Dtype), stream);
    PLUGIN_CUASSERT(cudaGetLastError());
}

cudaError_t MultilevelPropose(cudaStream_t stream, int N, int inputCnt, int samples, const float* regWeight,
    const float inputHeight, const float inputWidth, nvinfer1::DataType dtype, const RefineNMSParameters& param,
    const MultilevelProposeROIWorkSpace& proposalOffset, void* workspace,
    const void* inScore, //[N, inputcnt, 1]
    const void* inDelta, //[N, inputcnt, 4]
    void* inCountValid,
    const void* inAnchors, //[N, inputcnt, 4]
    void* outScore, void* outBbox)
{
    int8_t* wsPtr = static_cast<int8_t*>(workspace);
    void* tempStoragePtr = wsPtr + proposalOffset.tempStorageOffset;
    void* preRefineSortedScorePtr = wsPtr + proposalOffset.preRefineSortedScoreOffset;
    void* preRefineBboxPtr = wsPtr + proposalOffset.preRefineBboxOffset;

    void* argMaxScorePtr = wsPtr + proposalOffset.argMaxScoreOffset;
    void* argMaxLabelPtr = wsPtr + proposalOffset.argMaxLabelOffset;
    void* argMaxBBoxPtr = wsPtr + proposalOffset.argMaxBboxOffset;

    void* sortClassScorePtr = wsPtr + proposalOffset.sortClassScoreOffset;
    void* sortClassLabelPtr = wsPtr + proposalOffset.sortClassLabelOffset;
    void* sortClassSampleIdxPtr = wsPtr + proposalOffset.sortClassSampleIdxOffset;
    void* sortClassValidCountPtr = wsPtr + proposalOffset.sortClassValidCountOffset;
    void* sortClassPosPtr = wsPtr + proposalOffset.sortClassPosOffset;
    void* sortNMSMarkPtr = wsPtr + proposalOffset.sortNMSMarkOffset;

    cudaError_t status = cudaSuccess;
    int NClass = param.numClasses;
    PLUGIN_ASSERT(NClass == 1);
    PLUGIN_CUASSERT(cudaMemsetAsync(argMaxScorePtr, 0, N * samples * sizeof(dtype), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(argMaxBBoxPtr, 0, N * samples * 4 * sizeof(dtype), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassPosPtr, 0, N * (NClass + 1) * sizeof(int), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(sortClassSampleIdxPtr, 0, N * (samples + 1) * sizeof(int), stream));

    PLUGIN_CUASSERT(cudaGetLastError());

    // Here, inDelta are converted to normalize coordinates based on anchors
    status = DecodeBBoxes(
        stream, N, inputCnt, regWeight, inputHeight, inputWidth, inAnchors, inDelta, const_cast<void*>(inDelta), dtype);
    PLUGIN_CUASSERT(cudaGetLastError());

    // sort the score
    // d_key_in: preRefineScorePtr [N, inputCnt, 1]
    // d_key_out: preRefineSortedScorePtr
    // d_values_in: inDelta [N, inputCnt, 4]
    // d_values_out: preRefineBboxPtr
    // num_items: inputCnt*N
    // num_segments: N
    // offsets: [0, inputCnt, inputCnt*2, ..., ]

    int* offsets = static_cast<int*>(tempStoragePtr);
    set_offset_kernel<<<1, 1024, 0, stream>>>(inputCnt, N + 1, offsets);
    PLUGIN_CUASSERT(cudaGetLastError());
    tempStoragePtr = static_cast<void*>(static_cast<int*>(tempStoragePtr) + (N + 1));

    switch (dtype)
    {
        case nvinfer1::DataType::kFLOAT:
        {
            score_bbox_cub_sort<float>(tempStoragePtr, inScore, preRefineSortedScorePtr,
                                       inDelta, preRefineBboxPtr, N * inputCnt, N,
                                       offsets, stream);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            score_bbox_cub_sort<__half>(tempStoragePtr, inScore, preRefineSortedScorePtr,
                                        inDelta, preRefineBboxPtr, N * inputCnt, N,
                                        offsets, stream);
            break;
        }
        default: PLUGIN_ASSERT(false);
    }

    if (NClass == 1)
    { // Only one class
        switch (dtype)
        {
            case nvinfer1::DataType::kFLOAT:
            {
                resample_kernel<float><<<N, dMIN(samples, 1024), 0, stream>>>(
                    inputCnt, samples, preRefineSortedScorePtr, preRefineBboxPtr, argMaxScorePtr, argMaxBBoxPtr);
                PLUGIN_CUASSERT(cudaGetLastError());
                break;
            }
            case nvinfer1::DataType::kHALF:
            {
                resample_kernel<__half><<<N, dMIN(samples, 1024), 0, stream>>>(
                    inputCnt, samples, preRefineSortedScorePtr, preRefineBboxPtr, argMaxScorePtr, argMaxBBoxPtr);
                PLUGIN_CUASSERT(cudaGetLastError());
                break;
            }
            default: PLUGIN_ASSERT(false);
        }

        int threads = 512;
        int blocks = (N * samples + threads - 1) / threads;
        blocks = dMIN(blocks, 8);

        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
        {
            resetMemValue_kernel<float><<<blocks, threads, 0, stream>>>(argMaxLabelPtr, N * samples, 0);
            PLUGIN_CUASSERT(cudaGetLastError());
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            resetMemValue_kernel<__half><<<blocks, threads, 0, stream>>>(argMaxLabelPtr, N * samples, 0);
            PLUGIN_CUASSERT(cudaGetLastError());
            break;
        }
        default: PLUGIN_ASSERT(false);
        }
    }

    if (samples <= 1024)
    {
        status = sortPerClass<256, 4>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 2048)
    {
        status = sortPerClass<256, 8>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else if (samples <= 4096)
    {
        status = sortPerClass<256, 16>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
            inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
            sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    }
    else
    {
        PLUGIN_FAIL("Unsupported sortPerClass");
        return cudaErrorLaunchFailure;
    }
    PLUGIN_CUASSERT(cudaGetLastError());

    status = PerClassNMS<1024>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
        // sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);

    PLUGIN_CUASSERT(cudaGetLastError());

    status = KeepTopKGatherBoxScore<512>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr,
        sortClassScorePtr, sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outScore, outBbox,
        1);

    PLUGIN_CUASSERT(cudaGetLastError());

    return status;
}

struct BBOX
{
    float y1, x1, y2, x2;
};

struct DELTA
{
    float dy, dx, logdh, logdw;
};

struct DELTA_HALF
{
    __half dy, dx, logdh, logdw;
};

__global__ void decode_bboxes_kernel(int samples, const void* anchors, const void* delta, const float* regWeight,
    const float inputHeight, const float inputWidth, void* outputBbox, float bboxClipThresh)
{

    const BBOX* anchors_in = static_cast<const BBOX*>(anchors);
    const DELTA* delta_in = static_cast<const DELTA*>(delta);
    BBOX* bbox_out = static_cast<BBOX*>(outputBbox);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;

        if (cur_id < samples)
        {
            BBOX cur_anchor_yxyx = anchors_in[blockOffset + cur_id];
            // convert yxyx -> cyxhw
            // cy, cx, h, w
            /*BBOX cur_anchor_cyxhw;*/

            float cur_anchor_h = (cur_anchor_yxyx.y2 - cur_anchor_yxyx.y1 + 1.0);
            float cur_anchor_w = (cur_anchor_yxyx.x2 - cur_anchor_yxyx.x1 + 1.0); // w
            float cur_anchor_yc = cur_anchor_yxyx.y1 + cur_anchor_h * 0.5;        // cy
            float cur_anchor_xc = cur_anchor_yxyx.x1 + cur_anchor_w * 0.5;        // cx

            DELTA cur_delta = delta_in[blockOffset + cur_id];

            // divided by regWeight
            cur_delta.dy /= regWeight[0];
            cur_delta.dx /= regWeight[1];
            cur_delta.logdh /= regWeight[2];
            cur_delta.logdw /= regWeight[3];

            cur_delta.logdh = dMIN(cur_delta.logdh, bboxClipThresh);
            cur_delta.logdw = dMIN(cur_delta.logdw, bboxClipThresh);

            // apply delta
            float decoded_box_yc = cur_anchor_yc + cur_delta.dy * cur_anchor_h;
            float decoded_box_xc = cur_anchor_xc + cur_delta.dx * cur_anchor_w;
            float decoded_box_h = expf(cur_delta.logdh) * cur_anchor_h;
            float decoded_box_w = expf(cur_delta.logdw) * cur_anchor_w;

            float decoded_box_ymin = decoded_box_yc - 0.5 * decoded_box_h;
            float decoded_box_xmin = decoded_box_xc - 0.5 * decoded_box_w;
            float decoded_box_ymax = decoded_box_ymin + decoded_box_h - 1.0;
            float decoded_box_xmax = decoded_box_xmin + decoded_box_w - 1.0;

            // clip bbox: a more precision clip method based on real window could be implemented
            decoded_box_ymin = dMAX(dMIN(decoded_box_ymin, inputHeight - 1.0), 0.0);
            decoded_box_xmin = dMAX(dMIN(decoded_box_xmin, inputWidth - 1.0), 0.0);
            decoded_box_ymax = dMAX(dMIN(decoded_box_ymax, inputHeight - 1.0), 0.0);
            decoded_box_xmax = dMAX(dMIN(decoded_box_xmax, inputWidth - 1.0), 0.0);

            bbox_out[blockOffset + cur_id].y1 = decoded_box_ymin;
            bbox_out[blockOffset + cur_id].x1 = decoded_box_xmin;
            bbox_out[blockOffset + cur_id].y2 = decoded_box_ymax;
            bbox_out[blockOffset + cur_id].x2 = decoded_box_xmax;
        }
    }
}

__global__ void decode_bboxes_kernel_half(int samples, const void* anchors, const void* delta, const float* regWeight,
    const float inputHeight, const float inputWidth, void* outputBbox, float bboxClipThresh)
{

    const BBoxT<float>* anchors_in = static_cast<const BBoxT<float>*>(anchors);
    const DELTA_HALF* delta_in = static_cast<const DELTA_HALF*>(delta);
    BBoxT<__half>* bbox_out = static_cast<BBoxT<__half>*>(outputBbox);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;

        if (cur_id < samples)
        {
            BBoxT<float> cur_anchor_yxyx = anchors_in[blockOffset + cur_id];
            // convert yxyx -> cyxhw
            // cy, cx, h, w

            float cur_anchor_h = (cur_anchor_yxyx.y2 - cur_anchor_yxyx.y1 + 1.0);
            float cur_anchor_w = (cur_anchor_yxyx.x2 - cur_anchor_yxyx.x1 + 1.0); // w
            float cur_anchor_yc = cur_anchor_yxyx.y1 + cur_anchor_h * 0.5;        // cy
            float cur_anchor_xc = cur_anchor_yxyx.x1 + cur_anchor_w * 0.5;        // cx

            DELTA_HALF cur_delta_half = delta_in[blockOffset + cur_id];
            DELTA cur_delta;
            cur_delta.dy = __half2float(cur_delta_half.dy);
            cur_delta.dx = __half2float(cur_delta_half.dx);
            cur_delta.logdh = __half2float(cur_delta_half.logdh);
            cur_delta.logdw = __half2float(cur_delta_half.logdw);

            // divided by regWeight
            cur_delta.dy /= regWeight[0];
            cur_delta.dx /= regWeight[1];
            cur_delta.logdh /= regWeight[2];
            cur_delta.logdw /= regWeight[3];

            cur_delta.logdh = dMIN(cur_delta.logdh, bboxClipThresh);
            cur_delta.logdw = dMIN(cur_delta.logdw, bboxClipThresh);

            // apply delta
            float decoded_box_yc = cur_anchor_yc + cur_delta.dy * cur_anchor_h;
            float decoded_box_xc = cur_anchor_xc + cur_delta.dx * cur_anchor_w;
            float decoded_box_h = expf(cur_delta.logdh) * cur_anchor_h;
            float decoded_box_w = expf(cur_delta.logdw) * cur_anchor_w;

            float decoded_box_ymin = decoded_box_yc - 0.5 * decoded_box_h;
            float decoded_box_xmin = decoded_box_xc - 0.5 * decoded_box_w;
            float decoded_box_ymax = decoded_box_ymin + decoded_box_h - 1.0;
            float decoded_box_xmax = decoded_box_xmin + decoded_box_w - 1.0;

            // clip bbox: a more precision clip method based on real window could be implemented
            decoded_box_ymin = dMAX(dMIN(decoded_box_ymin, inputHeight - 1.0), 0.0);
            decoded_box_xmin = dMAX(dMIN(decoded_box_xmin, inputWidth - 1.0), 0.0);
            decoded_box_ymax = dMAX(dMIN(decoded_box_ymax, inputHeight - 1.0), 0.0);
            decoded_box_xmax = dMAX(dMIN(decoded_box_xmax, inputWidth - 1.0), 0.0);

            bbox_out[blockOffset + cur_id].y1 = __float2half(decoded_box_ymin);
            bbox_out[blockOffset + cur_id].x1 = __float2half(decoded_box_xmin);
            bbox_out[blockOffset + cur_id].y2 = __float2half(decoded_box_ymax);
            bbox_out[blockOffset + cur_id].x2 = __float2half(decoded_box_xmax);
        }
    }
}

cudaError_t DecodeBBoxes(cudaStream_t stream, int N,
    int samples, // number of anchors per image
    const float* regWeight, const float inputHeight, const float inputWidth,
    const void* anchors, // [N, anchors, (y1, x1, y2, x2)]
    const void* delta,   //[N, anchors, (dy, dx, log(dh), log(dw)])
    void* outputBbox,     //[N, anchors, (y1, x1, y2, x2)]
    nvinfer1::DataType dtype
)
{

    int blocks = N;
    int threads = dMIN(samples, 1024);

    // delta multiply bbox_std
    // apply delta steps:
    //  cy = anchor_cy + dy*height
    //  cx = anchor_cx + dx*weight
    //  h = exp(dh)*anchor_h
    //  w = exp(dw)*anchor_w
    // clip the bbox in absolute coordinates
    float bboxClipThresh = log(1000.0f / 16.0f);

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        decode_bboxes_kernel<<<blocks, threads, 0, stream>>>(
            samples, anchors, delta, regWeight, inputHeight, inputWidth, outputBbox, bboxClipThresh);
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        decode_bboxes_kernel_half<<<blocks, threads, 0, stream>>>(
            samples, anchors, delta, regWeight, inputHeight, inputWidth, outputBbox, bboxClipThresh);
        break;
    }
    default: PLUGIN_ASSERT(false);
    }

    return cudaGetLastError();
}

__global__ void apply_delta_kernel(int samples, const void* anchors, const void* delta, void* outputBbox)
{

    const BBOX* anchors_in = static_cast<const BBOX*>(anchors);
    const DELTA* delta_in = static_cast<const DELTA*>(delta);
    BBOX* bbox_out = static_cast<BBOX*>(outputBbox);

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;
        if (cur_id < samples)
        {
            BBOX cur_anchor_yxyx = anchors_in[blockOffset + cur_id];
            // convert yxyx -> cyxhw
            // cy, cx, h, w
            BBOX cur_anchor_cyxhw;

            cur_anchor_cyxhw.y1 = (cur_anchor_yxyx.y1 + cur_anchor_yxyx.y2) / 2.f;
            cur_anchor_cyxhw.x1 = (cur_anchor_yxyx.x1 + cur_anchor_yxyx.x2) / 2.f;
            cur_anchor_cyxhw.y2 = (cur_anchor_yxyx.y2 - cur_anchor_yxyx.y1);
            cur_anchor_cyxhw.x2 = (cur_anchor_yxyx.x2 - cur_anchor_yxyx.x1);

            DELTA cur_delta = delta_in[blockOffset + cur_id];

            // multiply std_dev
            cur_delta.dy *= 0.1f;
            cur_delta.dx *= 0.1f;
            cur_delta.logdh *= 0.2f;
            cur_delta.logdw *= 0.2f;

            // apply delta
            cur_anchor_cyxhw.y1 += cur_delta.dy * cur_anchor_cyxhw.y2;
            cur_anchor_cyxhw.x1 += cur_delta.dx * cur_anchor_cyxhw.x2;
            cur_anchor_cyxhw.y2 *= expf(cur_delta.logdh);
            cur_anchor_cyxhw.x2 *= expf(cur_delta.logdw);

            cur_anchor_yxyx.y1 = cur_anchor_cyxhw.y1 - 0.5f * cur_anchor_cyxhw.y2;
            cur_anchor_yxyx.x1 = cur_anchor_cyxhw.x1 - 0.5f * cur_anchor_cyxhw.x2;
            cur_anchor_yxyx.y2 = cur_anchor_yxyx.y1 + cur_anchor_cyxhw.y2;
            cur_anchor_yxyx.x2 = cur_anchor_yxyx.x1 + cur_anchor_cyxhw.x2;

            // clip bbox: a more precision clip method based on real window could be implemented
            cur_anchor_yxyx.y1 = dMAX(dMIN(cur_anchor_yxyx.y1, 1.f), 0.f);
            cur_anchor_yxyx.x1 = dMAX(dMIN(cur_anchor_yxyx.x1, 1.f), 0.f);
            cur_anchor_yxyx.y2 = dMAX(dMIN(cur_anchor_yxyx.y2, 1.f), 0.f);
            cur_anchor_yxyx.x2 = dMAX(dMIN(cur_anchor_yxyx.x2, 1.f), 0.f);

            bbox_out[blockOffset + cur_id].y1 = cur_anchor_yxyx.y1;
            bbox_out[blockOffset + cur_id].x1 = cur_anchor_yxyx.x1;
            bbox_out[blockOffset + cur_id].y2 = cur_anchor_yxyx.y2;
            bbox_out[blockOffset + cur_id].x2 = cur_anchor_yxyx.x2;
        }
    }
}

cudaError_t ApplyDelta2Bboxes(cudaStream_t stream, int N,
    int samples,         // number of anchors per image
    const void* anchors, // [N, anchors, (y1, x1, y2, x2)]
    const void* delta,   //[N, anchors, (dy, dx, log(dh), log(dw)])
    void* outputBbox     //[N, anchors, (y1, x1, y2, x2)]
    )
{

    int blocks = N;
    int threads = dMIN(samples, 1024);

    // delta multiply bbox_std
    // apply delta steps:
    //  cy = anchor_cy + dy*height
    //  cx = anchor_cx + dx*weight
    //  h = exp(dh)*anchor_h
    //  w = exp(dw)*anchor_w
    // clip the bbox

    apply_delta_kernel<<<blocks, threads, 0, stream>>>(samples, anchors, delta, outputBbox);

    return cudaGetLastError();
}

template <typename Tfeat>
__device__ inline Tfeat interpolateBilinear(const Tfeat* src, xy_t srcDims, float y, float x)
{
    const int y0 = static_cast<int>(y);
    const float yAlpha = y - static_cast<float>(y0);
    const int x0 = static_cast<int>(x);
    const float xAlpha = x - static_cast<float>(x0);

    assert(y0 < srcDims.y);
    assert(x0 < srcDims.x);

    const int y1 = (yAlpha == 0) ? y0 : y0 + 1; // ceil
    const int x1 = (xAlpha == 0) ? x0 : x0 + 1; // ceil

    assert(y1 < srcDims.y);
    assert(x1 < srcDims.x);

    const Tfeat src00 = src[(y0) *srcDims.x + (x0)];
    const Tfeat src01 = src[(y0) *srcDims.x + (x1)];
    const Tfeat src10 = src[(y1) *srcDims.x + (x0)];
    const Tfeat src11 = src[(y1) *srcDims.x + (x1)];

    const Tfeat src0 = src00 * (1.0F - xAlpha) + src01 * xAlpha;
    const Tfeat src1 = src10 * (1.0F - xAlpha) + src11 * xAlpha;

    return src0 * (1.0F - yAlpha) + src1 * yAlpha;
}

template <>
__device__ inline __half interpolateBilinear(const __half* src, xy_t srcDims, float y, float x)
{
    const int y0 = static_cast<int>(y);
    const float yAlpha = y - static_cast<float>(y0);
    const int x0 = static_cast<int>(x);
    const float xAlpha = x - static_cast<float>(x0);

    assert(y0 < srcDims.y);
    assert(x0 < srcDims.x);

    const int y1 = (yAlpha == 0) ? y0 : y0 + 1; // ceil
    const int x1 = (xAlpha == 0) ? x0 : x0 + 1; // ceil

    assert(y1 < srcDims.y);
    assert(x1 < srcDims.x);

    const __half src00 = src[(y0) *srcDims.x + (x0)];
    const __half src01 = src[(y0) *srcDims.x + (x1)];
    const __half src10 = src[(y1) *srcDims.x + (x0)];
    const __half src11 = src[(y1) *srcDims.x + (x1)];

    const __half src0 = add_fb(mul_fb(src00, (1.0F - xAlpha)), mul_fb(src01, xAlpha));
    const __half src1 = add_fb(mul_fb(src10, (1.0F - xAlpha)), mul_fb(src11, xAlpha));

    return add_fb(mul_fb(src0, (1.0F - yAlpha)), mul_fb(src1, yAlpha));
}

template <typename Trois, typename Tfeat>
__global__ void roiAlign_kernel(xy_t const imageSize, int32_t const featureCount, int32_t const roiCount,
    float const firstThreshold, int32_t const transformCoords, bool const absCoords, bool const swapCoords,
    bool const plusOneCoords, int32_t const samplingRatio, Trois const* rois, Tfeat const* P2, xy_t const P2dims,
    Tfeat const* P3, xy_t const P3dims, Tfeat const* P4, xy_t const P4dims, Tfeat const* P5, xy_t const P5dims,
    Tfeat* pooled, xy_t const poolDims)
{
    int32_t const batch = blockIdx.x;
    int32_t const feature = blockIdx.y;
    int32_t const roiIdx = blockIdx.z;

    Trois const* roi = rois + 4 * (batch * roiCount + roiIdx);
    float y1, x1, y2, x2, hw;
    if (swapCoords)
    {
        y1 = min(roi[0], roi[2]);
        x1 = min(roi[1], roi[3]);
        y2 = max(roi[0], roi[2]);
        x2 = max(roi[1], roi[3]);
    }
    else
    {
        x1 = min(roi[0], roi[2]);
        y1 = min(roi[1], roi[3]);
        x2 = max(roi[0], roi[2]);
        y2 = max(roi[1], roi[3]);
    }
    if (absCoords)
    {
        y1 = max(0.F, min(static_cast<float>(imageSize.y), y1)) / imageSize.y;
        x1 = max(0.F, min(static_cast<float>(imageSize.x), x1)) / imageSize.x;
        y2 = max(0.F, min(static_cast<float>(imageSize.y), y2)) / imageSize.y;
        x2 = max(0.F, min(static_cast<float>(imageSize.x), x2)) / imageSize.x;
    }
    else
    {
        y1 = max(0.F, min(1.F, y1));
        x1 = max(0.F, min(1.F, x1));
        y2 = max(0.F, min(1.F, y2));
        x2 = max(0.F, min(1.F, x2));
    }
    if (plusOneCoords)
    {
        hw = (y2 - y1 + 1.F / imageSize.y) * (x2 - x1 + 1.F / imageSize.x);
    }
    else
    {
        hw = (y2 - y1) * (x2 - x1);
    }

    Tfeat const* src = P2;
    xy_t srcDims = P2dims;
    int32_t iP = 2;
    float threshold = firstThreshold;

    if (hw > threshold)
    {
        src = P3;
        srcDims = P3dims;
        ++iP;
    }
    threshold *= 4;

    if (hw > threshold)
    {
        src = P4;
        srcDims = P4dims;
        ++iP;
    }
    threshold *= 4;

    if (hw > threshold)
    {
        src = P5;
        srcDims = P5dims;
        ++iP;
    }

    src += srcDims.x * srcDims.y * (batch * featureCount + feature);

    Tfeat* dst = pooled + poolDims.x * poolDims.y * (batch * roiCount * featureCount + roiIdx * featureCount + feature);

    float yStart, xStart, yEnd, xEnd, yDelta, xDelta;
    float samplingOffset;
    if (transformCoords == -1)
    {
        // Back-Compatibility with old PyramidROIAlign implementation.
        samplingOffset = 0.F;

        yStart = y1 * (srcDims.y - 1);
        xStart = x1 * (srcDims.x - 1);

        yEnd = y2 * (srcDims.y - 1);
        xEnd = x2 * (srcDims.x - 1);

        yDelta = (yEnd - yStart) / (poolDims.y - 1);
        xDelta = (xEnd - xStart) / (poolDims.x - 1);
    }
    else
    {
        float inputOffset;
        if (transformCoords == 0) // No Half Pixel
        {
            inputOffset = 0.F;
            samplingOffset = 0.F;
        }
        if (transformCoords == 1) // Output Half Pixel
        {
            inputOffset = 0.F;
            samplingOffset = 0.5F;
        }
        if (transformCoords == 2) // Half Pixel
        {
            inputOffset = 0.5F;
            samplingOffset = 0.5F;
        }

        yStart = y1 * srcDims.y - inputOffset;
        xStart = x1 * srcDims.x - inputOffset;

        yEnd = y2 * srcDims.y - inputOffset;
        xEnd = x2 * srcDims.x - inputOffset;

        yDelta = (yEnd - yStart) / poolDims.y;
        xDelta = (xEnd - xStart) / poolDims.x;
    }

    int32_t const samplingRatioX
        = samplingRatio > 0 ? samplingRatio : max(1, static_cast<int32_t>(ceilf((xEnd - xStart) / poolDims.x)));
    int32_t const samplingRatioY
        = samplingRatio > 0 ? samplingRatio : max(1, static_cast<int32_t>(ceilf((yEnd - yStart) / poolDims.y)));
    int32_t const samplingCount = samplingRatioX * samplingRatioY;

    for (int32_t outIdx = threadIdx.x; outIdx < poolDims.x * poolDims.y; outIdx += blockDim.x)
    {
        int32_t xx = outIdx % poolDims.x;
        int32_t yy = outIdx / poolDims.x;
        Tfeat* out = dst + poolDims.x * yy + xx;
        Tfeat result = 0;
        for (int32_t iy = 0; iy < samplingRatioY; iy++)
        {
            float ySample = yStart + yDelta * yy;
            ySample += yDelta * (iy + samplingOffset) / samplingRatioY;
            ySample = min(max(ySample, 0.F), srcDims.y - 1.F);

            for (int32_t ix = 0; ix < samplingRatioX; ix++)
            {
                float xSample = xStart + xDelta * xx;
                xSample += xDelta * (ix + samplingOffset) / samplingRatioX;
                xSample = min(max(xSample, 0.F), srcDims.x - 1.F);

                result += interpolateBilinear(src, srcDims, ySample, xSample);
            }
        }
        *out = result / samplingCount;
    }
}

cudaError_t roiAlign(cudaStream_t const stream, int32_t const batchSize, xy_t const imageSize,
    int32_t const featureCount, int32_t const roiCount, float const firstThreshold, int32_t const transformCoords,
    bool const absCoords, bool const swapCoords, bool const plusOneCoords, int32_t const samplingRatio,
    void const* rois, void const* const layers[], xy_t const* layerDims, void* const pooled, xy_t const poolDims)
{
    dim3 const blocks(batchSize, featureCount, roiCount);
    int32_t const threads(min(256, poolDims.x * poolDims.y));

    roiAlign_kernel<<<blocks, threads, 0, stream>>>(imageSize, featureCount, roiCount, firstThreshold, transformCoords,
        absCoords, swapCoords, plusOneCoords, samplingRatio, static_cast<float const*>(rois),
        static_cast<float const*>(layers[0]), layerDims[0], static_cast<float const*>(layers[1]), layerDims[1],
        static_cast<float const*>(layers[2]), layerDims[2], static_cast<float const*>(layers[3]), layerDims[3],
        static_cast<float*>(pooled), poolDims);
    return cudaGetLastError();
}

template <typename Trois, typename Tfeat>
__global__ void roiAlignHalfCenter_kernel(int featureCount, int roiCount,

    float threshold, int inputHeight, int inputWidth, const void* rois_,

    const void* const P2_, const xy_t P2dims, const void* const P3_, const xy_t P3dims, const void* const P4_, const xy_t P4dims,
    const void* const P5_, const xy_t P5dims, const void* const P6_, const xy_t P6dims,

    void* pooled_, const xy_t poolDims)
{
    const Trois* rois = static_cast<const Trois*>(rois_);
    const Tfeat* P2 = static_cast<const Tfeat*>(P2_);
    const Tfeat* P3 = static_cast<const Tfeat*>(P3_);
    const Tfeat* P4 = static_cast<const Tfeat*>(P4_);
    const Tfeat* P5 = static_cast<const Tfeat*>(P5_);
    const Tfeat* P6 = static_cast<const Tfeat*>(P6_);
    Tfeat* pooled = static_cast<Tfeat* >(pooled_);
    const int batch = blockIdx.x;
    const int feature = blockIdx.y;
    const int roiIdx = blockIdx.z;
    const int total_item_cnt = poolDims.x * poolDims.y;

    for (int itemIdx = threadIdx.x; itemIdx < total_item_cnt; itemIdx += blockDim.x)
    {
        const Trois* roi = rois + 4 * (batch * roiCount + roiIdx);

        const float y1 = roi[0];
        const float x1 = roi[1];
        const float y2 = roi[2];
        const float x2 = roi[3];

        if (!(0 <= y1 && y1 <= inputHeight && 0 <= x1 && x1 <= inputWidth && 0 <= y2 && y2 <= inputHeight && 0 <= x2
                && x2 <= inputWidth && y1 < y2 && x1 < x2))
        {

            continue;
        }
        else
        {
        }

        const float hw = (y2 - y1) * (x2 - x1);

        const Tfeat* src = P2;
        xy_t srcDims = P2dims;
        int iP = 2;

        float threshold_per_item = threshold;

        if (hw > threshold_per_item)
        {
            src = P3;
            srcDims = P3dims;
            ++iP;
        }
        threshold_per_item *= 4;

        if (hw > threshold_per_item)
        {
            src = P4;
            srcDims = P4dims;
            ++iP;
        }
        threshold_per_item *= 4;

        if (hw > threshold_per_item)
        {
            src = P5;
            srcDims = P5dims;
            ++iP;
        }
        threshold_per_item *= 4;

        if (hw > threshold_per_item)
        {
            src = P6;
            srcDims = P6dims;
            ++iP;
        }

        src += srcDims.x * srcDims.y * (batch * featureCount + feature);

        Tfeat* dst
            = pooled + poolDims.x * poolDims.y * (batch * roiCount * featureCount + roiIdx * featureCount + feature) + itemIdx;

        float scale_to_level = 1.0f;

        for (int i = 0; i < iP; i++)
        {
            scale_to_level *= 2.0f;
        }

        const float yStart = y1 / scale_to_level;
        const float xStart = x1 / scale_to_level;

        const float yEnd = y2 / scale_to_level;
        const float xEnd = x2 / scale_to_level;

        const float yDelta = (yEnd - yStart) / (poolDims.y);
        const float xDelta = (xEnd - xStart) / (poolDims.x);

        const int yy = itemIdx / poolDims.y;
        const int xx = itemIdx % poolDims.x;

        const float ySample = dMIN(dMAX(yStart + yDelta * (yy + 0.5), 0.0f), srcDims.y - 1.0f);
        const float xSample = dMIN(dMAX(xStart + xDelta * (xx + 0.5), 0.0f), srcDims.x - 1.0f);

        Tfeat result = interpolateBilinear<Tfeat>(src, srcDims, ySample, xSample);

        *dst = result;

    }
}

template <>
__global__ void roiAlignHalfCenter_kernel<__half, __half>(int featureCount, int roiCount,

    float threshold, int inputHeight, int inputWidth, const void* rois_,

    const void* const P2_, const xy_t P2dims, const void* const P3_, const xy_t P3dims, const void* const P4_, const xy_t P4dims,
    const void* const P5_, const xy_t P5dims, const void* const P6_, const xy_t P6dims,

    void* pooled_, const xy_t poolDims)
{
    const __half* rois = static_cast<const __half*>(rois_);
    const __half* P2 = static_cast<const __half*>(P2_);
    const __half* P3 = static_cast<const __half*>(P3_);
    const __half* P4 = static_cast<const __half*>(P4_);
    const __half* P5 = static_cast<const __half*>(P5_);
    const __half* P6 = static_cast<const __half*>(P6_);
    __half* pooled = static_cast<__half* >(pooled_);
    const int batch = blockIdx.x;
    const int feature = blockIdx.y;
    const int roiIdx = blockIdx.z;
    const int total_item_cnt = poolDims.x * poolDims.y;

    for (int itemIdx = threadIdx.x; itemIdx < total_item_cnt; itemIdx += blockDim.x)
    {
        const __half* roi = rois + 4 * (batch * roiCount + roiIdx);

        const float y1 = __half2float(roi[0]);
        const float x1 = __half2float(roi[1]);
        const float y2 = __half2float(roi[2]);
        const float x2 = __half2float(roi[3]);

        if (!(0 <= y1 && y1 <= inputHeight && 0 <= x1 && x1 <= inputWidth && 0 <= y2 && y2 <= inputHeight && 0 <= x2
                && x2 <= inputWidth && y1 < y2 && x1 < x2))
        {

            continue;
        }
        else
        {
        }

        const float hw = (y2 - y1) * (x2 - x1);

        const __half* src = P2;
        xy_t srcDims = P2dims;
        int iP = 2;

        float threshold_per_item = threshold;

        if (hw > threshold_per_item)
        {
            src = P3;
            srcDims = P3dims;
            ++iP;
        }
        threshold_per_item *= 4;

        if (hw > threshold_per_item)
        {
            src = P4;
            srcDims = P4dims;
            ++iP;
        }
        threshold_per_item *= 4;

        if (hw > threshold_per_item)
        {
            src = P5;
            srcDims = P5dims;
            ++iP;
        }
        threshold_per_item *= 4;

        if (hw > threshold_per_item)
        {
            src = P6;
            srcDims = P6dims;
            ++iP;
        }

        src += srcDims.x * srcDims.y * (batch * featureCount + feature);

        __half* dst
            = pooled + poolDims.x * poolDims.y * (batch * roiCount * featureCount + roiIdx * featureCount + feature) + itemIdx;

        float scale_to_level = 1.0f;

        for (int i = 0; i < iP; i++)
        {
            scale_to_level *= 2.0f;
        }

        const float yStart = y1 / scale_to_level;
        const float xStart = x1 / scale_to_level;

        const float yEnd = y2 / scale_to_level;
        const float xEnd = x2 / scale_to_level;

        const float yDelta = (yEnd - yStart) / (poolDims.y);
        const float xDelta = (xEnd - xStart) / (poolDims.x);

        const int yy = itemIdx / poolDims.y;
        const int xx = itemIdx % poolDims.x;

        const float ySample = dMIN(dMAX(yStart + yDelta * (yy + 0.5), 0.0f), srcDims.y - 1.0f);
        const float xSample = dMIN(dMAX(xStart + xDelta * (xx + 0.5), 0.0f), srcDims.x - 1.0f);

        __half result = interpolateBilinear<__half>(src, srcDims, ySample, xSample);

        *dst = result;
    }
}

cudaError_t roiAlignHalfCenter(cudaStream_t stream, int batchSize, int featureCount, int roiCount, float firstThreshold,

    int inputHeight, int inputWidth, const void* rois, const void* const layers[], const xy_t* layerDims,

    void* pooled, const xy_t poolDims, const nvinfer1::DataType dtype)
{
    const dim3 blocks(batchSize, featureCount, roiCount);
    const int threads(64);
    switch (dtype){
        case nvinfer1::DataType::kFLOAT:
        {
            roiAlignHalfCenter_kernel<float, float><<<blocks, threads, 0, stream>>>(featureCount, roiCount,
                firstThreshold, inputHeight, inputWidth, rois, layers[0], layerDims[0], layers[1], layerDims[1],
                layers[2], layerDims[2], layers[3], layerDims[3], layers[4], layerDims[4], pooled, poolDims);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            roiAlignHalfCenter_kernel<__half, __half><<<blocks, threads, 0, stream>>>(featureCount, roiCount,
                firstThreshold, inputHeight, inputWidth, rois, layers[0], layerDims[0], layers[1], layerDims[1],
                layers[2], layerDims[2], layers[3], layerDims[3], layers[4], layerDims[4], pooled, poolDims);
            break;
        }
        default: PLUGIN_ASSERT(false);
    }



    return cudaGetLastError();
}

__global__ void resize_nearest_kernel_2d(int nbatch, float scale, int2 osize, float const* idata, int istride,
    int ibatchstride, float* odata, int ostride, int obatchstride)
{

    int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    int z0 = blockIdx.z;
    for (int batch = z0; batch < nbatch; batch += gridDim.z)
    {
        for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y)
        {
            for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x)
            {
                int ix = int(ox / scale);
                int iy = int(oy / scale);
                odata[batch * obatchstride + oy * ostride + ox] = idata[batch * ibatchstride + iy * istride + ix];
            }
        }
    }
}

void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
    int istride, int ibatchstride, float* odata, int ostride, int obatchstride)
{

    resize_nearest_kernel_2d<<<grid, block, 0, stream>>>(
        nbatch, scale, osize, idata, istride, ibatchstride, odata, ostride, obatchstride);
}

struct BOX
{
    float y1, x1, y2, x2;
};

struct DETECTION
{
    float y1, x1, y2, x2, class_id, score;
};

__global__ void specialslice_kernel(int samples, const void* idata, void* odata)
{

    int N = blockIdx.x;
    int blockOffset = N * samples;
    int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;
    const DETECTION* in_detections = static_cast<const DETECTION*>(idata);
    BOX* out_bboxes = static_cast<BOX*>(odata);

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;
        if (cur_id < samples)
        {
            out_bboxes[blockOffset + cur_id].y1 = in_detections[blockOffset + cur_id].y1;
            out_bboxes[blockOffset + cur_id].x1 = in_detections[blockOffset + cur_id].x1;
            out_bboxes[blockOffset + cur_id].y2 = in_detections[blockOffset + cur_id].y2;
            out_bboxes[blockOffset + cur_id].x2 = in_detections[blockOffset + cur_id].x2;
        }
    }
}

void specialSlice(cudaStream_t stream, int batch_size, int boxes_cnt, const void* idata, void* odata)
{
    int blocks = batch_size;
    int threads = dMIN(boxes_cnt, 2048);

    specialslice_kernel<<<blocks, threads, 0, stream>>>(boxes_cnt, idata, odata);
}

template <typename Dtype>
__global__ void concatenate(int featureCnt, int sampleCnt, const void* const* inScores, const void* const* inBBox,
    void* outScore, void* outBBox)
{
    int N = blockIdx.x;
    int outBlockOffset = N * sampleCnt * featureCnt;
    int inBlockOffset = N * sampleCnt;
    int itemsPerThread = (sampleCnt + blockDim.x - 1) / blockDim.x;
    Dtype* outScorePtr = static_cast<Dtype*>(outScore);
    BBoxT<Dtype>* outBBoxPtr = static_cast<BBoxT<Dtype>*>(outBBox);

    for (int fId = 0; fId < featureCnt; fId++)
    {
        const Dtype* fInScorePtr = static_cast<const Dtype*>(inScores[fId]);
        const BBoxT<Dtype>* fInBBoxPtr = static_cast<const BBoxT<Dtype>*>(inBBox[fId]);
        int featureOffset = fId * sampleCnt;
        for (int i = 0; i < itemsPerThread; i++)
        {
            int curId = i * blockDim.x + threadIdx.x;
            if (curId < sampleCnt)
            {
                outScorePtr[outBlockOffset + featureOffset + curId] = fInScorePtr[inBlockOffset + curId];
                outBBoxPtr[outBlockOffset + featureOffset + curId] = fInBBoxPtr[inBlockOffset + curId];
            }
        }
    }
}

template <typename Dtype>
__global__ void resampleBBox_kernel(int orig_size, int sample_size, const void* orig_bbox_ptr, void* sampled_bbox_ptr)
{
    const BBoxT<Dtype>* in_bbox = static_cast<const BBoxT<Dtype>*>(orig_bbox_ptr);
    BBoxT<Dtype>* out_bbox = static_cast<BBoxT<Dtype>*>(sampled_bbox_ptr);

    int N = blockIdx.x;
    int blockOffset_in = N * orig_size;
    int blockOffset_out = N * sample_size;
    int totalItems = (sample_size + (blockDim.x - 1)) / blockDim.x;

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;
        if (cur_id < sample_size)
        {
            out_bbox[blockOffset_out + cur_id] = in_bbox[blockOffset_in + cur_id];
        }
    }
}

cudaError_t ConcatTopK(cudaStream_t stream, int N, int featureCnt, int topK, nvinfer1::DataType dtype, void* workspace,
    const ConcatTopKWorkSpace& spaceOffset, void** inScores, void** inBBox, void* outProposals)
{
    // Prepare Offset
    int8_t* wsPtr = static_cast<int8_t*>(workspace);
    void* tempStoragePtr = wsPtr + spaceOffset.tempStorageOffset;
    void* concatedScorePtr = wsPtr + spaceOffset.concatedScoreOffset;
    void* concatedBBoxPtr = wsPtr + spaceOffset.concatedBBoxOffset;
    void* sortedScorePtr = wsPtr + spaceOffset.sortedScoreOffset;
    void* sortedBBoxPtr = wsPtr + spaceOffset.sortedBBoxOffset;

    int blocks = N; // batch_size
    int threads = dMIN(topK, 2048);
    // Concat Scores and inBBox
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        concatenate<float>
            <<<blocks, threads, 0, stream>>>(featureCnt, topK, inScores, inBBox, concatedScorePtr, concatedBBoxPtr);

        PLUGIN_CUASSERT(cudaGetLastError());
        break;
    case nvinfer1::DataType::kHALF:
        concatenate<__half>
            <<<blocks, threads, 0, stream>>>(featureCnt, topK, inScores, inBBox, concatedScorePtr, concatedBBoxPtr);
        PLUGIN_CUASSERT(cudaGetLastError());
        break;
    default: PLUGIN_ASSERT(false);
    }

    // Sort and sample topK
    int itemCnt = topK * featureCnt;
    int* offsets = static_cast<int*>(tempStoragePtr);
    set_offset_kernel<<<1, 1024, 0, stream>>>(itemCnt, N + 1, offsets);
    PLUGIN_ASSERT(cudaGetLastError() == cudaSuccess);
    tempStoragePtr = static_cast<void*>(static_cast<int*>(tempStoragePtr) + (N + 1));

    switch (dtype)
    {
        case nvinfer1::DataType::kFLOAT:
        {
            score_bbox_cub_sort<float>(tempStoragePtr, concatedScorePtr, sortedScorePtr,
                                       concatedBBoxPtr, sortedBBoxPtr, N * itemCnt, N,
                                       offsets, stream);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            score_bbox_cub_sort<__half>(tempStoragePtr, concatedScorePtr, sortedScorePtr,
                                       concatedBBoxPtr, sortedBBoxPtr, N * itemCnt, N,
                                       offsets, stream);
            break;
        }
        default: PLUGIN_ASSERT(false);
    }

    // Sample
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        resampleBBox_kernel<float><<<N, dMIN(topK, 1024), 0, stream>>>(itemCnt, topK, sortedBBoxPtr, outProposals);
        PLUGIN_CUASSERT(cudaGetLastError());
        break;
    case nvinfer1::DataType::kHALF:
        resampleBBox_kernel<__half><<<N, dMIN(topK, 1024), 0, stream>>>(itemCnt, topK, sortedBBoxPtr, outProposals);
        PLUGIN_CUASSERT(cudaGetLastError());
        break;
    default: PLUGIN_ASSERT(false);
    }

    PLUGIN_ASSERT(cudaGetLastError() == cudaSuccess);
    return cudaGetLastError();
}
