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

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "maskRCNNKernels.h"
#include "plugin.h"
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

// blockDim.x : NClass
// GroupDim.x : sample count
// GroupDim.y : batch N
// outScore : DType[ N * sample * 1 ]
// outLabel : int[ N * sample * 1 ]
// outBbox : int[ N * sample * 4 ]
template <typename DType, typename BoxType, int Threads = 32>
__global__ void argMaxGroup_kernel(int samples, int NClass, const void* inScorePtr, const void* inBboxPtr,
    const void* validSampleCountPtr, void* outScorePtr, void* outLabelPtr, void* outBboxPtr)
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
        for (int i = 0; i < NClass; i += Threads)
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

#define LabelShift (DType)(2.5f)
#define MinValidScore (DType)(0.01f)

template <typename DType>
__device__ __forceinline__ DType getKey(DType score, int lable, int NClass)
{
    return (lable < 0 ? (DType) 0 : ((DType)(NClass - lable - 1) * LabelShift + score + MinValidScore));
}

template <typename DType, typename BoxType>
__device__ __forceinline__ void getScoreLable(DType key, int NClass, DType& score, BoxType& lable)
{
    int i = key / LabelShift;
    score = (key <= MinValidScore ? (DType) 0 : key - (DType) i * LabelShift - MinValidScore);
    score = dCLAMP(score, (DType) 0, (DType) 1.0);
    lable = (BoxType)(key <= MinValidScore ? -1 : (NClass - i - 1));
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
            if (label != background && label != -1 && score >= (DType) scoreThreshold)
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
                        if (boxIoU(refBox, curBox[ite]) >= (DType) nmsThreshold)
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
            if (curI < finalTopK && idx[i] >= 0 && score[i] > MinValidScore)
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
            if (curI < finalTopK && idx[i] >= 0 && score[i] > MinValidScore)
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
    , sortClassSampleIdxDims(sampleCount, 1)
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
    // sortClassLabel : [N, samples] : kINT32
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
    // sortClassLabel : [N, samples] : kINT32
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

template <int Threads>
cudaError_t argMaxGroup(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
    const void* inScore, const void* inBbox, const void* validSamples, void* outScore, void* outLabel, void* outBbox)
{
    int maxGridX = dMIN(samples, 512 / N);
    dim3 gridDim = {(unsigned int) nAlignDown(maxGridX, 32), (unsigned int) N, 1};
    dim3 threads = {Threads, 1, 1};
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        argMaxGroup_kernel<float, float, Threads><<<gridDim, threads, 0, stream>>>(
            samples, NClass, inScore, inBbox, validSamples, outScore, outLabel, outBbox);
        break;
    case nvinfer1::DataType::kHALF: break;
    default: assert(false);
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
    case nvinfer1::DataType::kHALF: break;
    default: assert(false);
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
    case nvinfer1::DataType::kHALF: break;
    default: assert(false);
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
    default: assert(false);
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
    CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));

    if (NClass > 1)
    { // multiple classes
        status = argMaxGroup<32>(stream, N, dtype, samples, NClass, inScores, inDelta, inCountValid, argMaxScorePtr,
            argMaxLabelPtr, argMaxBBoxPtr); // argMaxBBoxPtr means delta of bboxes
        assert(status == cudaSuccess);
        CUASSERT(status);
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
        default: assert(false);
        }
    }

    status = ApplyDelta2Bboxes(stream, N, samples, inROI, argMaxBBoxPtr, argMaxBBoxPtr);
    assert(status == cudaSuccess);

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
        assert(false && "unsupported sortPerClass");
        return cudaErrorLaunchFailure;
    }
    assert(status == cudaSuccess);
    CUASSERT(status);

    status = PerClassNMS<256>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
        // sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    assert(status == cudaSuccess);
    CUASSERT(status);

    status = KeepTopKGather<256>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outDetections, 0);
    assert(status == cudaSuccess);
    CUASSERT(status);
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
        out[blockOffset + cur_id] = in[blockOffset + cur_id].fg;
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

__global__ void resample_kernel(int orig_size, int sample_size, const void* orig_score_ptr, const void* orig_bbox_ptr,
    void* sampled_score_ptr, void* sampled_bbox_ptr)
{
    const float* in_score = static_cast<const float*>(orig_score_ptr);
    const BBoxT<float>* in_bbox = static_cast<const BBoxT<float>*>(orig_bbox_ptr);
    float* out_score = static_cast<float*>(sampled_score_ptr);
    BBoxT<float>* out_bbox = static_cast<BBoxT<float>*>(sampled_bbox_ptr);

    int N = blockIdx.x;
    int blockOffset_in = N * orig_size;
    int blockOffset_out = N * sample_size;
    int totalItems = (sample_size + (blockDim.x - 1)) / blockDim.x;

    for (int i = 0; i < totalItems; i++)
    {
        int cur_id = i * blockDim.x + threadIdx.x;
        out_score[blockOffset_out + cur_id] = in_score[blockOffset_in + cur_id];
        out_bbox[blockOffset_out + cur_id] = in_bbox[blockOffset_in + cur_id];
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
    CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));

    // extract foreground score
    extract_fg_kernel<<<N, dMIN(inputCnt, 1024), 0, stream>>>(inputCnt, inScores, preRefineScorePtr);
    CUASSERT(cudaGetLastError());

    // Here, inDelta are converted to normalize coordinates based on anchors
    status = ApplyDelta2Bboxes(stream, N, inputCnt, inAnchors, inDelta, const_cast<void*>(inDelta));
    CUASSERT(status);

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
    assert(cudaGetLastError() == cudaSuccess);
    tempStoragePtr = static_cast<void*>(static_cast<int*>(tempStoragePtr) + (N + 1));

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(NULL, temp_storage_bytes, (float*) preRefineScorePtr,
        (float*) preRefineSortedScorePtr, (BBoxT<float>*) inDelta, (BBoxT<float>*) preRefineBboxPtr, N * inputCnt, N,
        offsets, offsets + 1, 0, 8 * sizeof(float), stream);

    assert((1 << 23) * N > temp_storage_bytes);

    cub::DeviceSegmentedRadixSort::SortPairsDescending(tempStoragePtr, temp_storage_bytes, (float*) preRefineScorePtr,
        (float*) preRefineSortedScorePtr, (BBoxT<float>*) inDelta, (BBoxT<float>*) preRefineBboxPtr, N * inputCnt, N,
        offsets, offsets + 1, 0, 8 * sizeof(float), stream);

    int NClass = param.numClasses;
    assert(NClass == 1);
    if (NClass > 1)
    { // multiple classes
    }
    else
    { // Only one class
        resample_kernel<<<N, dMIN(samples, 1024), 0, stream>>>(
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
        default: assert(false);
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
        assert(false && "unsupported sortPerClass");
        return cudaErrorLaunchFailure;
    }
    assert(status == cudaSuccess);
    CUASSERT(status);

    status = PerClassNMS<256>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
        // sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    assert(status == cudaSuccess);
    CUASSERT(status);

    status = KeepTopKGather<256>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
        sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outProposals, 1);
    assert(status == cudaSuccess);
    CUASSERT(status);

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

        BBOX cur_anchor_yxyx = anchors_in[blockOffset + cur_id];
        // convert yxyx -> cyxhw
        // cy, cx, h, w
        BBOX cur_anchor_cyxhw;

        cur_anchor_cyxhw.y1 = (cur_anchor_yxyx.y1 + cur_anchor_yxyx.y2) / 2;
        cur_anchor_cyxhw.x1 = (cur_anchor_yxyx.x1 + cur_anchor_yxyx.x2) / 2;
        cur_anchor_cyxhw.y2 = (cur_anchor_yxyx.y2 - cur_anchor_yxyx.y1);
        cur_anchor_cyxhw.x2 = (cur_anchor_yxyx.x2 - cur_anchor_yxyx.x1);

        DELTA cur_delta = delta_in[blockOffset + cur_id];

        // multiply std_dev
        cur_delta.dy *= 0.1;
        cur_delta.dx *= 0.1;
        cur_delta.logdh *= 0.2;
        cur_delta.logdw *= 0.2;

        // apply delta
        cur_anchor_cyxhw.y1 += cur_delta.dy * cur_anchor_cyxhw.y2;
        cur_anchor_cyxhw.x1 += cur_delta.dx * cur_anchor_cyxhw.x2;
        cur_anchor_cyxhw.y2 *= expf(cur_delta.logdh);
        cur_anchor_cyxhw.x2 *= expf(cur_delta.logdw);

        cur_anchor_yxyx.y1 = cur_anchor_cyxhw.y1 - 0.5 * cur_anchor_cyxhw.y2;
        cur_anchor_yxyx.x1 = cur_anchor_cyxhw.x1 - 0.5 * cur_anchor_cyxhw.x2;
        cur_anchor_yxyx.y2 = cur_anchor_yxyx.y1 + cur_anchor_cyxhw.y2;
        cur_anchor_yxyx.x2 = cur_anchor_yxyx.x1 + cur_anchor_cyxhw.x2;

        // clip bbox: a more precision clip method based on real window could be implemented
        cur_anchor_yxyx.y1 = dMAX(dMIN(cur_anchor_yxyx.y1, 1.0), 0.0);
        cur_anchor_yxyx.x1 = dMAX(dMIN(cur_anchor_yxyx.x1, 1.0), 0.0);
        cur_anchor_yxyx.y2 = dMAX(dMIN(cur_anchor_yxyx.y2, 1.0), 0.0);
        cur_anchor_yxyx.x2 = dMAX(dMIN(cur_anchor_yxyx.x2, 1.0), 0.0);

        bbox_out[blockOffset + cur_id].y1 = cur_anchor_yxyx.y1;
        bbox_out[blockOffset + cur_id].x1 = cur_anchor_yxyx.x1;
        bbox_out[blockOffset + cur_id].y2 = cur_anchor_yxyx.y2;
        bbox_out[blockOffset + cur_id].x2 = cur_anchor_yxyx.x2;
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

    const float src00 = src[(y0) *srcDims.x + (x0)];
    const float src01 = src[(y0) *srcDims.x + (x1)];
    const float src10 = src[(y1) *srcDims.x + (x0)];
    const float src11 = src[(y1) *srcDims.x + (x1)];

    const float src0 = src00 * (1 - xAlpha) + src01 * xAlpha;
    const float src1 = src10 * (1 - xAlpha) + src11 * xAlpha;

    return src0 * (1 - yAlpha) + src1 * yAlpha;
}

template <typename Trois, typename Tfeat>
__global__ void roiAlign_kernel(int featureCount, int roiCount,

    float threshold, const Trois* rois,

    const Tfeat* P2, const xy_t P2dims, const Tfeat* P3, const xy_t P3dims, const Tfeat* P4, const xy_t P4dims,
    const Tfeat* P5, const xy_t P5dims,

    Tfeat* pooled, const xy_t poolDims)
{
    const int batch = blockIdx.x;
    const int feature = blockIdx.y;

    // int prev_invalid = -1;
    for (int roiIdx = threadIdx.x; roiIdx < roiCount; roiIdx += blockDim.x)
    {
        const Trois* roi = rois + 4 * (batch * roiCount + roiIdx);

        const float y1 = roi[0];
        const float x1 = roi[1];
        const float y2 = roi[2];
        const float x2 = roi[3];

        if (!(0 <= y1 && y1 <= 1 && 0 <= x1 && x1 <= 1 && 0 <= y2 && y2 <= 1 && 0 <= x2 && x2 <= 1 && y1 < y2
                && x1 < x2))
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

        Tfeat* dst
            = pooled + poolDims.x * poolDims.y * (batch * roiCount * featureCount + roiIdx * featureCount + feature);

        const float yStart = y1 * (srcDims.y - 1);
        const float xStart = x1 * (srcDims.x - 1);

        const float yEnd = y2 * (srcDims.y - 1);
        const float xEnd = x2 * (srcDims.x - 1);

        const float yDelta = (yEnd - yStart) / (poolDims.y - 1);
        const float xDelta = (xEnd - xStart) / (poolDims.x - 1);

        for (int yy = 0; yy < poolDims.y; ++yy)
        {
            const float ySample = min(yStart + yDelta * yy, yEnd);

            for (int xx = 0; xx < poolDims.x; ++xx)
            {
                const float xSample = min(xStart + xDelta * xx, xEnd);

                float result = interpolateBilinear(src, srcDims, ySample, xSample);

                *dst = result;
                dst++;
            }
        }
    }
}

cudaError_t roiAlign(cudaStream_t stream, int batchSize, int featureCount, int roiCount, float firstThreshold,

    const void* rois, const void* const layers[], const xy_t* layerDims,

    void* pooled, const xy_t poolDims)
{
    const dim3 blocks(batchSize, featureCount);
    const int threads(256);

    roiAlign_kernel<<<blocks, threads, 0, stream>>>(featureCount, roiCount, firstThreshold,
        static_cast<const float*>(rois),

        static_cast<const float*>(layers[0]), layerDims[0], static_cast<const float*>(layers[1]), layerDims[1],
        static_cast<const float*>(layers[2]), layerDims[2], static_cast<const float*>(layers[3]), layerDims[3],

        static_cast<float*>(pooled), poolDims);
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

        out_bboxes[blockOffset + cur_id].y1 = in_detections[blockOffset + cur_id].y1;
        out_bboxes[blockOffset + cur_id].x1 = in_detections[blockOffset + cur_id].x1;
        out_bboxes[blockOffset + cur_id].y2 = in_detections[blockOffset + cur_id].y2;
        out_bboxes[blockOffset + cur_id].x2 = in_detections[blockOffset + cur_id].x2;
    }
}

void specialSlice(cudaStream_t stream, int batch_size, int boxes_cnt, const void* idata, void* odata)
{
    int blocks = batch_size;
    int threads = dMIN(boxes_cnt, 2048);

    specialslice_kernel<<<blocks, threads, 0, stream>>>(boxes_cnt, idata, odata);
}
