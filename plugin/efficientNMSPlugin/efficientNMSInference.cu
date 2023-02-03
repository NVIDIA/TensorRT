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
#include "cub/cub.cuh"
#include "cuda_runtime_api.h"

#include "efficientNMSInference.cuh"
#include "efficientNMSInference.h"

#define NMS_TILES 5

using namespace nvinfer1;
using namespace nvinfer1::plugin;

template <typename T>
__device__ float IOU(EfficientNMSParameters param, BoxCorner<T> box1, BoxCorner<T> box2)
{
    // Regardless of the selected box coding, IOU is always performed in BoxCorner coding.
    // The boxes are copied so that they can be reordered without affecting the originals.
    BoxCorner<T> b1 = box1;
    BoxCorner<T> b2 = box2;
    b1.reorder();
    b2.reorder();
    float intersectArea = BoxCorner<T>::intersect(b1, b2).area();
    if (intersectArea <= 0.f)
    {
        return 0.f;
    }
    float unionArea = b1.area() + b2.area() - intersectArea;
    if (unionArea <= 0.f)
    {
        return 0.f;
    }
    return intersectArea / unionArea;
}

template <typename T, typename Tb>
__device__ BoxCorner<T> DecodeBoxes(EfficientNMSParameters param, int boxIdx, int anchorIdx,
    const Tb* __restrict__ boxesInput, const Tb* __restrict__ anchorsInput)
{
    // The inputs will be in the selected coding format, as well as the decoding function. But the decoded box
    // will always be returned as BoxCorner.
    Tb box = boxesInput[boxIdx];
    if (!param.boxDecoder)
    {
        return BoxCorner<T>(box);
    }
    Tb anchor = anchorsInput[anchorIdx];
    box.reorder();
    anchor.reorder();
    return BoxCorner<T>(box.decode(anchor));
}

template <typename T, typename Tb>
__device__ void MapNMSData(EfficientNMSParameters param, int idx, int imageIdx, const Tb* __restrict__ boxesInput,
    const Tb* __restrict__ anchorsInput, const int* __restrict__ topClassData, const int* __restrict__ topAnchorsData,
    const int* __restrict__ topNumData, const T* __restrict__ sortedScoresData, const int* __restrict__ sortedIndexData,
    T& scoreMap, int& classMap, BoxCorner<T>& boxMap, int& boxIdxMap)
{
    // idx: Holds the NMS box index, within the current batch.
    // idxSort: Holds the batched NMS box index, which indexes the (filtered, but sorted) score buffer.
    // scoreMap: Holds the score that corresponds to the indexed box being processed by NMS.
    if (idx >= topNumData[imageIdx])
    {
        return;
    }
    int idxSort = imageIdx * param.numScoreElements + idx;
    scoreMap = sortedScoresData[idxSort];

    // idxMap: Holds the re-mapped index, which indexes the (filtered, but unsorted) buffers.
    // classMap: Holds the class that corresponds to the idx'th sorted score being processed by NMS.
    // anchorMap: Holds the anchor that corresponds to the idx'th sorted score being processed by NMS.
    int idxMap = imageIdx * param.numScoreElements + sortedIndexData[idxSort];
    classMap = topClassData[idxMap];
    int anchorMap = topAnchorsData[idxMap];

    // boxIdxMap: Holds the re-re-mapped index, which indexes the (unfiltered, and unsorted) boxes input buffer.
    boxIdxMap = -1;
    if (param.shareLocation) // Shape of boxesInput: [batchSize, numAnchors, 1, 4]
    {
        boxIdxMap = imageIdx * param.numAnchors + anchorMap;
    }
    else // Shape of boxesInput: [batchSize, numAnchors, numClasses, 4]
    {
        int batchOffset = imageIdx * param.numAnchors * param.numClasses;
        int anchorOffset = anchorMap * param.numClasses;
        boxIdxMap = batchOffset + anchorOffset + classMap;
    }
    // anchorIdxMap: Holds the re-re-mapped index, which indexes the (unfiltered, and unsorted) anchors input buffer.
    int anchorIdxMap = -1;
    if (param.shareAnchors) // Shape of anchorsInput: [1, numAnchors, 4]
    {
        anchorIdxMap = anchorMap;
    }
    else // Shape of anchorsInput: [batchSize, numAnchors, 4]
    {
        anchorIdxMap = imageIdx * param.numAnchors + anchorMap;
    }
    // boxMap: Holds the box that corresponds to the idx'th sorted score being processed by NMS.
    boxMap = DecodeBoxes<T, Tb>(param, boxIdxMap, anchorIdxMap, boxesInput, anchorsInput);
}

template <typename T>
__device__ void WriteNMSResult(EfficientNMSParameters param, int* __restrict__ numDetectionsOutput,
    T* __restrict__ nmsScoresOutput, int* __restrict__ nmsClassesOutput, BoxCorner<T>* __restrict__ nmsBoxesOutput,
    T threadScore, int threadClass, BoxCorner<T> threadBox, int imageIdx, unsigned int resultsCounter)
{
    int outputIdx = imageIdx * param.numOutputBoxes + resultsCounter - 1;
    if (param.scoreSigmoid)
    {
        nmsScoresOutput[outputIdx] = sigmoid_mp(threadScore);
    }
    else if (param.scoreBits > 0)
    {
        nmsScoresOutput[outputIdx] = add_mp(threadScore, (T) -1);
    }
    else
    {
        nmsScoresOutput[outputIdx] = threadScore;
    }
    nmsClassesOutput[outputIdx] = threadClass;
    if (param.clipBoxes)
    {
        nmsBoxesOutput[outputIdx] = threadBox.clip((T) 0, (T) 1);
    }
    else
    {
        nmsBoxesOutput[outputIdx] = threadBox;
    }
    numDetectionsOutput[imageIdx] = resultsCounter;
}

__device__ void WriteONNXResult(EfficientNMSParameters param, int* outputIndexData, int* __restrict__ nmsIndicesOutput,
    int imageIdx, int threadClass, int boxIdxMap)
{
    int index = boxIdxMap % param.numAnchors;
    int idx = atomicAdd((unsigned int*) &outputIndexData[0], 1);
    nmsIndicesOutput[idx * 3 + 0] = imageIdx;
    nmsIndicesOutput[idx * 3 + 1] = threadClass;
    nmsIndicesOutput[idx * 3 + 2] = index;
}

__global__ void PadONNXResult(EfficientNMSParameters param, int* outputIndexData, int* __restrict__ nmsIndicesOutput)
{
    if (threadIdx.x > 0)
    {
        return;
    }
    int pidx = outputIndexData[0] - 1;
    if (pidx < 0)
    {
        return;
    }
    for (int idx = pidx + 1; idx < param.batchSize * param.numOutputBoxes; idx++)
    {
        nmsIndicesOutput[idx * 3 + 0] = nmsIndicesOutput[pidx * 3 + 0];
        nmsIndicesOutput[idx * 3 + 1] = nmsIndicesOutput[pidx * 3 + 1];
        nmsIndicesOutput[idx * 3 + 2] = nmsIndicesOutput[pidx * 3 + 2];
    }
}

template <typename T, typename Tb>
__global__ void EfficientNMS(EfficientNMSParameters param, const int* topNumData, int* outputIndexData,
    int* outputClassData, const int* sortedIndexData, const T* __restrict__ sortedScoresData,
    const int* __restrict__ topClassData, const int* __restrict__ topAnchorsData, const Tb* __restrict__ boxesInput,
    const Tb* __restrict__ anchorsInput, int* __restrict__ numDetectionsOutput, T* __restrict__ nmsScoresOutput,
    int* __restrict__ nmsClassesOutput, int* __restrict__ nmsIndicesOutput, BoxCorner<T>* __restrict__ nmsBoxesOutput)
{
    unsigned int thread = threadIdx.x;
    unsigned int imageIdx = blockIdx.y;
    unsigned int tileSize = blockDim.x;
    if (imageIdx >= param.batchSize)
    {
        return;
    }

    int numSelectedBoxes = min(topNumData[imageIdx], param.numSelectedBoxes);
    int numTiles = (numSelectedBoxes + tileSize - 1) / tileSize;
    if (thread >= numSelectedBoxes)
    {
        return;
    }

    __shared__ int blockState;
    __shared__ unsigned int resultsCounter;
    if (thread == 0)
    {
        blockState = 0;
        resultsCounter = 0;
    }

    int threadState[NMS_TILES];
    unsigned int boxIdx[NMS_TILES];
    T threadScore[NMS_TILES];
    int threadClass[NMS_TILES];
    BoxCorner<T> threadBox[NMS_TILES];
    int boxIdxMap[NMS_TILES];
    for (int tile = 0; tile < numTiles; tile++)
    {
        threadState[tile] = 0;
        boxIdx[tile] = thread + tile * blockDim.x;
        MapNMSData<T, Tb>(param, boxIdx[tile], imageIdx, boxesInput, anchorsInput, topClassData, topAnchorsData,
            topNumData, sortedScoresData, sortedIndexData, threadScore[tile], threadClass[tile], threadBox[tile],
            boxIdxMap[tile]);
    }

    // Iterate through all boxes to NMS against.
    for (int i = 0; i < numSelectedBoxes; i++)
    {
        int tile = i / tileSize;

        if (boxIdx[tile] == i)
        {
            // Iteration lead thread, figure out what the other threads should do,
            // this will be signaled via the blockState shared variable.
            if (threadState[tile] == -1)
            {
                // Thread already dead, this box was already dropped in a previous iteration,
                // because it had a large IOU overlap with another lead thread previously, so
                // it would never be kept anyway, therefore it can safely be skip all IOU operations
                // in this iteration.
                blockState = -1; // -1 => Signal all threads to skip iteration
            }
            else if (threadState[tile] == 0)
            {
                // As this box will be kept, this is a good place to find what index in the results buffer it
                // should have, as this allows to perform an early loop exit if there are enough results.
                if (resultsCounter >= param.numOutputBoxes)
                {
                    blockState = -2; // -2 => Signal all threads to do an early loop exit.
                }
                else
                {
                    // Thread is still alive, because it has not had a large enough IOU overlap with
                    // any other kept box previously. Therefore, this box will be kept for sure. However,
                    // we need to check against all other subsequent boxes from this position onward,
                    // to see how those other boxes will behave in future iterations.
                    blockState = 1;        // +1 => Signal all (higher index) threads to calculate IOU against this box
                    threadState[tile] = 1; // +1 => Mark this box's thread to be kept and written out to results

                    // If the numOutputBoxesPerClass check is enabled, write the result only if the limit for this
                    // class on this image has not been reached yet. Other than (possibly) skipping the write, this
                    // won't affect anything else in the NMS threading.
                    bool write = true;
                    if (param.numOutputBoxesPerClass >= 0)
                    {
                        int classCounterIdx = imageIdx * param.numClasses + threadClass[tile];
                        write = (outputClassData[classCounterIdx] < param.numOutputBoxesPerClass);
                        outputClassData[classCounterIdx]++;
                    }
                    if (write)
                    {
                        // This branch is visited by one thread per iteration, so it's safe to do non-atomic increments.
                        resultsCounter++;
                        if (param.outputONNXIndices)
                        {
                            WriteONNXResult(
                                param, outputIndexData, nmsIndicesOutput, imageIdx, threadClass[tile], boxIdxMap[tile]);
                        }
                        else
                        {
                            WriteNMSResult<T>(param, numDetectionsOutput, nmsScoresOutput, nmsClassesOutput,
                                nmsBoxesOutput, threadScore[tile], threadClass[tile], threadBox[tile], imageIdx,
                                resultsCounter);
                        }
                    }
                }
            }
            else
            {
                // This state should never be reached, but just in case...
                blockState = 0; // 0 => Signal all threads to not do any updates, nothing happens.
            }
        }

        __syncthreads();

        if (blockState == -2)
        {
            // This is the signal to exit from the loop.
            return;
        }

        if (blockState == -1)
        {
            // This is the signal for all threads to just skip this iteration, as no IOU's need to be checked.
            continue;
        }

        // Grab a box and class to test the current box against. The test box corresponds to iteration i,
        // therefore it will have a lower index than the current thread box, and will therefore have a higher score
        // than the current box because it's located "before" in the sorted score list.
        T testScore;
        int testClass;
        BoxCorner<T> testBox;
        int testBoxIdxMap;
        MapNMSData<T, Tb>(param, i, imageIdx, boxesInput, anchorsInput, topClassData, topAnchorsData, topNumData,
            sortedScoresData, sortedIndexData, testScore, testClass, testBox, testBoxIdxMap);

        for (int tile = 0; tile < numTiles; tile++)
        {
            // IOU
            if (boxIdx[tile] > i && // Make sure two different boxes are being tested, and that it's a higher index;
                boxIdx[tile] < numSelectedBoxes && // Make sure the box is within numSelectedBoxes;
                blockState == 1 &&                 // Signal that allows IOU checks to be performed;
                threadState[tile] == 0 &&          // Make sure this box hasn't been either dropped or kept already;
                threadClass[tile] == testClass &&  // Compare only boxes of matching classes;
                lte_mp(threadScore[tile], testScore) && // Make sure the sorting order of scores is as expected;
                IOU<T>(param, threadBox[tile], testBox) >= param.iouThreshold) // And... IOU overlap.
            {
                // Current box overlaps with the box tested in this iteration, this box will be skipped.
                threadState[tile] = -1; // -1 => Mark this box's thread to be dropped.
            }
        }
    }
}

template <typename T>
cudaError_t EfficientNMSLauncher(EfficientNMSParameters& param, int* topNumData, int* outputIndexData,
    int* outputClassData, int* sortedIndexData, T* sortedScoresData, int* topClassData, int* topAnchorsData,
    const void* boxesInput, const void* anchorsInput, int* numDetectionsOutput, T* nmsScoresOutput,
    int* nmsClassesOutput, int* nmsIndicesOutput, void* nmsBoxesOutput, cudaStream_t stream)
{
    unsigned int tileSize = param.numSelectedBoxes / NMS_TILES;
    if (param.numSelectedBoxes <= 512)
    {
        tileSize = 512;
    }
    if (param.numSelectedBoxes <= 256)
    {
        tileSize = 256;
    }

    const dim3 blockSize = {tileSize, 1, 1};
    const dim3 gridSize = {1, (unsigned int) param.batchSize, 1};

    if (param.boxCoding == 0)
    {
        EfficientNMS<T, BoxCorner<T>><<<gridSize, blockSize, 0, stream>>>(param, topNumData, outputIndexData,
            outputClassData, sortedIndexData, sortedScoresData, topClassData, topAnchorsData,
            (BoxCorner<T>*) boxesInput, (BoxCorner<T>*) anchorsInput, numDetectionsOutput, nmsScoresOutput,
            nmsClassesOutput, nmsIndicesOutput, (BoxCorner<T>*) nmsBoxesOutput);
    }
    else if (param.boxCoding == 1)
    {
        // Note that nmsBoxesOutput is always coded as BoxCorner<T>, regardless of the input coding type.
        EfficientNMS<T, BoxCenterSize<T>><<<gridSize, blockSize, 0, stream>>>(param, topNumData, outputIndexData,
            outputClassData, sortedIndexData, sortedScoresData, topClassData, topAnchorsData,
            (BoxCenterSize<T>*) boxesInput, (BoxCenterSize<T>*) anchorsInput, numDetectionsOutput, nmsScoresOutput,
            nmsClassesOutput, nmsIndicesOutput, (BoxCorner<T>*) nmsBoxesOutput);
    }

    if (param.outputONNXIndices)
    {
        PadONNXResult<<<1, 1, 0, stream>>>(param, outputIndexData, nmsIndicesOutput);
    }

    return cudaGetLastError();
}

__global__ void EfficientNMSFilterSegments(EfficientNMSParameters param, const int* __restrict__ topNumData,
    int* __restrict__ topOffsetsStartData, int* __restrict__ topOffsetsEndData)
{
    int imageIdx = threadIdx.x;
    if (imageIdx > param.batchSize)
    {
        return;
    }
    topOffsetsStartData[imageIdx] = imageIdx * param.numScoreElements;
    topOffsetsEndData[imageIdx] = imageIdx * param.numScoreElements + topNumData[imageIdx];
}

template <typename T>
__global__ void EfficientNMSFilter(EfficientNMSParameters param, const T* __restrict__ scoresInput,
    int* __restrict__ topNumData, int* __restrict__ topIndexData, int* __restrict__ topAnchorsData,
    T* __restrict__ topScoresData, int* __restrict__ topClassData)
{
    int elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int imageIdx = blockDim.y * blockIdx.y + threadIdx.y;

    // Boundary Conditions
    if (elementIdx >= param.numScoreElements || imageIdx >= param.batchSize)
    {
        return;
    }

    // Shape of scoresInput: [batchSize, numAnchors, numClasses]
    int scoresInputIdx = imageIdx * param.numScoreElements + elementIdx;

    // For each class, check its corresponding score if it crosses the threshold, and if so select this anchor,
    // and keep track of the maximum score and the corresponding (argmax) class id
    T score = scoresInput[scoresInputIdx];
    if (gte_mp(score, (T) param.scoreThreshold))
    {
        // Unpack the class and anchor index from the element index
        int classIdx = elementIdx % param.numClasses;
        int anchorIdx = elementIdx / param.numClasses;

        // If this is a background class, ignore it.
        if (classIdx == param.backgroundClass)
        {
            return;
        }

        // Use an atomic to find an open slot where to write the selected anchor data.
        if (topNumData[imageIdx] >= param.numScoreElements)
        {
            return;
        }
        int selectedIdx = atomicAdd((unsigned int*) &topNumData[imageIdx], 1);
        if (selectedIdx >= param.numScoreElements)
        {
            topNumData[imageIdx] = param.numScoreElements;
            return;
        }

        // Shape of topScoresData / topClassData: [batchSize, numScoreElements]
        int topIdx = imageIdx * param.numScoreElements + selectedIdx;

        if (param.scoreBits > 0)
        {
            score = add_mp(score, (T) 1);
            if (gt_mp(score, (T) (2.f - 1.f / 1024.f)))
            {
                // Ensure the incremented score fits in the mantissa without changing the exponent
                score = (2.f - 1.f / 1024.f);
            }
        }

        topIndexData[topIdx] = selectedIdx;
        topAnchorsData[topIdx] = anchorIdx;
        topScoresData[topIdx] = score;
        topClassData[topIdx] = classIdx;
    }
}

template <typename T>
__global__ void EfficientNMSDenseIndex(EfficientNMSParameters param, int* __restrict__ topNumData,
    int* __restrict__ topIndexData, int* __restrict__ topAnchorsData, int* __restrict__ topOffsetsStartData,
    int* __restrict__ topOffsetsEndData, T* __restrict__ topScoresData, int* __restrict__ topClassData)
{
    int elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int imageIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if (elementIdx >= param.numScoreElements || imageIdx >= param.batchSize)
    {
        return;
    }

    int dataIdx = imageIdx * param.numScoreElements + elementIdx;
    int anchorIdx = elementIdx / param.numClasses;
    int classIdx = elementIdx % param.numClasses;
    if (param.scoreBits > 0)
    {
        T score = topScoresData[dataIdx];
        if (lt_mp(score, (T) param.scoreThreshold))
        {
            score = (T) 1;
        }
        else if (classIdx == param.backgroundClass)
        {
            score = (T) 1;
        }
        else
        {
            score = add_mp(score, (T) 1);
            if (gt_mp(score, (T) (2.f - 1.f / 1024.f)))
            {
                // Ensure the incremented score fits in the mantissa without changing the exponent
                score = (2.f - 1.f / 1024.f);
            }
        }
        topScoresData[dataIdx] = score;
    }
    else
    {
        T score = topScoresData[dataIdx];
        if (lt_mp(score, (T) param.scoreThreshold))
        {
            topScoresData[dataIdx] = -(1 << 15);
        }
        else if (classIdx == param.backgroundClass)
        {
            topScoresData[dataIdx] = -(1 << 15);
        }
    }

    topIndexData[dataIdx] = elementIdx;
    topAnchorsData[dataIdx] = anchorIdx;
    topClassData[dataIdx] = classIdx;

    if (elementIdx == 0)
    {
        // Saturate counters
        topNumData[imageIdx] = param.numScoreElements;
        topOffsetsStartData[imageIdx] = imageIdx * param.numScoreElements;
        topOffsetsEndData[imageIdx] = (imageIdx + 1) * param.numScoreElements;
    }
}

template <typename T>
cudaError_t EfficientNMSFilterLauncher(EfficientNMSParameters& param, const T* scoresInput, int* topNumData,
    int* topIndexData, int* topAnchorsData, int* topOffsetsStartData, int* topOffsetsEndData, T* topScoresData,
    int* topClassData, cudaStream_t stream)
{
    const unsigned int elementsPerBlock = 512;
    const unsigned int imagesPerBlock = 1;
    const unsigned int elementBlocks = (param.numScoreElements + elementsPerBlock - 1) / elementsPerBlock;
    const unsigned int imageBlocks = (param.batchSize + imagesPerBlock - 1) / imagesPerBlock;
    const dim3 blockSize = {elementsPerBlock, imagesPerBlock, 1};
    const dim3 gridSize = {elementBlocks, imageBlocks, 1};

    float kernelSelectThreshold = 0.007f;
    if (param.scoreSigmoid)
    {
        // Inverse Sigmoid
        if (param.scoreThreshold <= 0.f)
        {
            param.scoreThreshold = -(1 << 15);
        }
        else
        {
            param.scoreThreshold = logf(param.scoreThreshold / (1.f - param.scoreThreshold));
        }
        kernelSelectThreshold = logf(kernelSelectThreshold / (1.f - kernelSelectThreshold));
        // Disable Score Bits Optimization
        param.scoreBits = -1;
    }

    if (param.scoreThreshold < kernelSelectThreshold)
    {
        // A full copy of the buffer is necessary because sorting will scramble the input data otherwise.
        PLUGIN_CHECK_CUDA(cudaMemcpyAsync(topScoresData, scoresInput,
            param.batchSize * param.numScoreElements * sizeof(T), cudaMemcpyDeviceToDevice, stream));

        EfficientNMSDenseIndex<T><<<gridSize, blockSize, 0, stream>>>(param, topNumData, topIndexData, topAnchorsData,
            topOffsetsStartData, topOffsetsEndData, topScoresData, topClassData);
    }
    else
    {
        EfficientNMSFilter<T><<<gridSize, blockSize, 0, stream>>>(
            param, scoresInput, topNumData, topIndexData, topAnchorsData, topScoresData, topClassData);

        EfficientNMSFilterSegments<<<1, param.batchSize, 0, stream>>>(
            param, topNumData, topOffsetsStartData, topOffsetsEndData);
    }

    return cudaGetLastError();
}

template <typename T>
size_t EfficientNMSSortWorkspaceSize(int batchSize, int numScoreElements)
{
    size_t sortedWorkspaceSize = 0;
    cub::DoubleBuffer<T> keysDB(nullptr, nullptr);
    cub::DoubleBuffer<int> valuesDB(nullptr, nullptr);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, sortedWorkspaceSize, keysDB, valuesDB,
        numScoreElements, batchSize, (const int*) nullptr, (const int*) nullptr);
    return sortedWorkspaceSize;
}

size_t EfficientNMSWorkspaceSize(int batchSize, int numScoreElements, int numClasses, DataType datatype)
{
    size_t total = 0;
    const size_t align = 256;
    // Counters
    // 3 for Filtering
    // 1 for Output Indexing
    // C for Max per Class Limiting
    size_t size = (3 + 1 + numClasses) * batchSize * sizeof(int);
    total += size + (size % align ? align - (size % align) : 0);
    // Int Buffers
    for (int i = 0; i < 4; i++)
    {
        size = batchSize * numScoreElements * sizeof(int);
        total += size + (size % align ? align - (size % align) : 0);
    }
    // Float Buffers
    for (int i = 0; i < 2; i++)
    {
        size = batchSize * numScoreElements * dataTypeSize(datatype);
        total += size + (size % align ? align - (size % align) : 0);
    }
    // Sort Workspace
    if (datatype == DataType::kHALF)
    {
        size = EfficientNMSSortWorkspaceSize<__half>(batchSize, numScoreElements);
        total += size + (size % align ? align - (size % align) : 0);
    }
    else if (datatype == DataType::kFLOAT)
    {
        size = EfficientNMSSortWorkspaceSize<float>(batchSize, numScoreElements);
        total += size + (size % align ? align - (size % align) : 0);
    }

    return total;
}

template <typename T>
T* EfficientNMSWorkspace(void* workspace, size_t& offset, size_t elements)
{
    T* buffer = (T*) ((size_t) workspace + offset);
    size_t align = 256;
    size_t size = elements * sizeof(T);
    size_t sizeAligned = size + (size % align ? align - (size % align) : 0);
    offset += sizeAligned;
    return buffer;
}

template <typename T>
pluginStatus_t EfficientNMSDispatch(EfficientNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* nmsIndicesOutput, void* workspace, cudaStream_t stream)
{
    // Clear Outputs (not all elements will get overwritten by the kernels, so safer to clear everything out)
    if (param.outputONNXIndices)
    {
        CSC(cudaMemsetAsync(nmsIndicesOutput, 0xFF, param.batchSize * param.numOutputBoxes * 3 * sizeof(int), stream), STATUS_FAILURE);
    }
    else
    {
        CSC(cudaMemsetAsync(numDetectionsOutput, 0x00, param.batchSize * sizeof(int), stream), STATUS_FAILURE);
        CSC(cudaMemsetAsync(nmsScoresOutput, 0x00, param.batchSize * param.numOutputBoxes * sizeof(T), stream), STATUS_FAILURE);
        CSC(cudaMemsetAsync(nmsBoxesOutput, 0x00, param.batchSize * param.numOutputBoxes * 4 * sizeof(T), stream), STATUS_FAILURE);
        CSC(cudaMemsetAsync(nmsClassesOutput, 0x00, param.batchSize * param.numOutputBoxes * sizeof(int), stream), STATUS_FAILURE);
    }

    // Empty Inputs
    if (param.numScoreElements < 1)
    {
        return STATUS_SUCCESS;
    }

    // Counters Workspace
    size_t workspaceOffset = 0;
    int countersTotalSize = (3 + 1 + param.numClasses) * param.batchSize;
    int* topNumData = EfficientNMSWorkspace<int>(workspace, workspaceOffset, countersTotalSize);
    int* topOffsetsStartData = topNumData + param.batchSize;
    int* topOffsetsEndData = topNumData + 2 * param.batchSize;
    int* outputIndexData = topNumData + 3 * param.batchSize;
    int* outputClassData = topNumData + 4 * param.batchSize;
    CSC(cudaMemsetAsync(topNumData, 0x00, countersTotalSize * sizeof(int), stream), STATUS_FAILURE);
    cudaError_t status = cudaGetLastError();
    CSC(status, STATUS_FAILURE);

    // Other Buffers Workspace
    int* topIndexData
        = EfficientNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* topClassData
        = EfficientNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* topAnchorsData
        = EfficientNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* sortedIndexData
        = EfficientNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    T* topScoresData = EfficientNMSWorkspace<T>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    T* sortedScoresData
        = EfficientNMSWorkspace<T>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    size_t sortedWorkspaceSize = EfficientNMSSortWorkspaceSize<T>(param.batchSize, param.numScoreElements);
    char* sortedWorkspaceData = EfficientNMSWorkspace<char>(workspace, workspaceOffset, sortedWorkspaceSize);
    cub::DoubleBuffer<T> scoresDB(topScoresData, sortedScoresData);
    cub::DoubleBuffer<int> indexDB(topIndexData, sortedIndexData);

    // Kernels
    status = EfficientNMSFilterLauncher<T>(param, (T*) scoresInput, topNumData, topIndexData, topAnchorsData,
        topOffsetsStartData, topOffsetsEndData, topScoresData, topClassData, stream);
    CSC(status, STATUS_FAILURE);

    status = cub::DeviceSegmentedRadixSort::SortPairsDescending(sortedWorkspaceData, sortedWorkspaceSize, scoresDB,
        indexDB, param.batchSize * param.numScoreElements, param.batchSize, topOffsetsStartData, topOffsetsEndData,
        param.scoreBits > 0 ? (10 - param.scoreBits) : 0, param.scoreBits > 0 ? 10 : sizeof(T) * 8, stream, false);
    CSC(status, STATUS_FAILURE);

    status = EfficientNMSLauncher<T>(param, topNumData, outputIndexData, outputClassData, indexDB.Current(),
        scoresDB.Current(), topClassData, topAnchorsData, boxesInput, anchorsInput, (int*) numDetectionsOutput,
        (T*) nmsScoresOutput, (int*) nmsClassesOutput, (int*) nmsIndicesOutput, nmsBoxesOutput, stream);
    CSC(status, STATUS_FAILURE);

    return STATUS_SUCCESS;
}

pluginStatus_t EfficientNMSInference(EfficientNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* nmsIndicesOutput, void* workspace, cudaStream_t stream)
{
    if (param.datatype == DataType::kFLOAT)
    {
        param.scoreBits = -1;
        return EfficientNMSDispatch<float>(param, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
            nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, nmsIndicesOutput, workspace, stream);
    }
    else if (param.datatype == DataType::kHALF)
    {
        if (param.scoreBits <= 0 || param.scoreBits > 10)
        {
            param.scoreBits = -1;
        }
        return EfficientNMSDispatch<__half>(param, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
            nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, nmsIndicesOutput, workspace, stream);
    }
    else
    {
        return STATUS_NOT_SUPPORTED;
    }
}
