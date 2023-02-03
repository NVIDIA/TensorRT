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
#include "common/kernel.h"
#include "common/nmsUtils.h"
#include "cuda_runtime_api.h"
#include "gatherNMSOutputs.h"
using namespace nvinfer1;
pluginStatus_t nmsInference(cudaStream_t stream, const int N, const int perBatchBoxesSize, const int perBatchScoresSize,
    const bool shareLocation, const int backgroundLabelId, const int numPredsPerClass, const int numClasses,
    const int topK, const int keepTopK, const float scoreThreshold, const float iouThreshold, const DataType DT_BBOX,
    const void* locData, const DataType DT_SCORE, const void* confData, void* keepCount, void* nmsedBoxes,
    void* nmsedScores, void* nmsedClasses, void* workspace, bool isNormalized, bool confSigmoid, bool clipBoxes,
    int scoreBits, bool caffeSemantics)
{
    // locCount = batch_size * number_boxes_per_sample * 4
    const int locCount = N * perBatchBoxesSize;
    /*
     * shareLocation
     * Bounding box are shared among all classes, i.e., a bounding box could be classified as any candidate class.
     * Otherwise
     * Bounding box are designed for specific classes, i.e., a bounding box could be classified as one certain class or
     * not (binary classification).
     */
    const int numLocClasses = shareLocation ? 1 : numClasses;

    size_t bboxDataSize = detectionForwardBBoxDataSize(N, perBatchBoxesSize, DT_BBOX);
    void* bboxDataRaw = workspace;
    CSC(cudaMemcpyAsync(bboxDataRaw, locData, bboxDataSize, cudaMemcpyDeviceToDevice, stream), STATUS_FAILURE);
    pluginStatus_t status;

    /*
     * bboxDataRaw format:
     * [batch size, numPriors (per sample), numLocClasses, 4]
     */
    // float for now
    void* bboxData;
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, perBatchBoxesSize, DT_BBOX);
    void* bboxPermute = nextWorkspacePtr((int8_t*) bboxDataRaw, bboxDataSize);

    /*
     * After permutation, bboxData format:
     * [batch_size, numLocClasses, numPriors (per sample) (numPredsPerClass), 4]
     * This is equivalent to swapping axis
     */
    if (!shareLocation)
    {
        status = permuteData(
            stream, locCount, numLocClasses, numPredsPerClass, 4, DT_BBOX, false, bboxDataRaw, bboxPermute);
        ASSERT_FAILURE(status == STATUS_SUCCESS);
        bboxData = bboxPermute;
    }
    /*
     * If shareLocation, numLocClasses = 1
     * No need to permute data on linear memory
     */
    else
    {
        bboxData = bboxDataRaw;
    }

    /*
     * Conf data format
     * [batch size, numPriors * param.numClasses, 1, 1]
     */
    const int numScores = N * perBatchScoresSize;
    size_t totalScoresSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
    if(DT_SCORE == DataType::kHALF) totalScoresSize /= 2; // detectionForwardPreNMSSize is implemented in terms of kFLOAT
    void* scores = nextWorkspacePtr((int8_t*) bboxPermute, bboxPermuteSize);

    // need a conf_scores
    /*
     * After permutation, bboxData format:
     * [batch_size, numClasses, numPredsPerClass, 1]
     */
    status = permuteData(
        stream, numScores, numClasses, numPredsPerClass, 1, DT_SCORE, confSigmoid, confData, scores);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    size_t indicesSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
    void* indices = nextWorkspacePtr((int8_t*) scores, totalScoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    if(DT_SCORE == DataType::kHALF) postNMSScoresSize /= 2; // detectionForwardPostNMSSize is implemented in terms of kFLOAT
    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK); // indices are full int32
    void* postNMSScores = nextWorkspacePtr((int8_t*) indices, indicesSize);
    void* postNMSIndices = nextWorkspacePtr((int8_t*) postNMSScores, postNMSScoresSize);

    void* sortingWorkspace = nextWorkspacePtr((int8_t*) postNMSIndices, postNMSIndicesSize);
    // Sort the scores so that the following NMS could be applied.
    float scoreShift = 0.f;
    if(DT_SCORE == DataType::kHALF && scoreBits > 0 && scoreBits <= 10)
        scoreShift = 1.f;
    status = sortScoresPerClass(stream, N, numClasses, numPredsPerClass, backgroundLabelId, scoreThreshold,
        DT_SCORE, scores, indices, sortingWorkspace, scoreBits, scoreShift);

    ASSERT_FAILURE(status == STATUS_SUCCESS);

    // This is set to true as the input bounding boxes are of the format [ymin,
    // xmin, ymax, xmax]. The default implementation assumes [xmin, ymin, xmax, ymax]
    bool flipXY = true;
    // NMS
    status = allClassNMS(stream, N, numClasses, numPredsPerClass, topK, iouThreshold, shareLocation, isNormalized,
        DT_SCORE, DT_BBOX, bboxData, scores, indices, postNMSScores, postNMSIndices, flipXY, scoreShift,
        caffeSemantics);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    // Sort the bounding boxes after NMS using scores
    status = sortScoresPerImage(stream, N, numClasses * topK, DT_SCORE, postNMSScores, postNMSIndices, scores,
        indices, sortingWorkspace, scoreBits);

    ASSERT_FAILURE(status == STATUS_SUCCESS);

    // Gather data from the sorted bounding boxes after NMS
    status = gatherNMSOutputs(stream, shareLocation, N, numPredsPerClass, numClasses, topK, keepTopK, DT_BBOX,
        DT_SCORE, indices, scores, bboxData, keepCount, nmsedBoxes, nmsedScores, nmsedClasses, clipBoxes, scoreShift);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    return STATUS_SUCCESS;
}
