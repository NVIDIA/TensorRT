/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "common.h"
#include "lstmEncoder.h"
#include "trtUtil.h"
#include <sstream>

namespace nmtSample
{

LSTMEncoder::LSTMEncoder(ComponentWeights::ptr& weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    ASSERT(mWeights->mMetaData.size() >= 4);
    const nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    ASSERT(dataType == nvinfer1::DataType::kFLOAT);
    mRNNKind = mWeights->mMetaData[1];
    mNumLayers = mWeights->mMetaData[2];
    mNumUnits = mWeights->mMetaData[3];

    size_t elementSize = inferTypeToBytes(dataType);
    // compute weights offsets
    size_t kernelOffset = 0;
    size_t biasStartOffset = ((4 * mNumUnits + 4 * mNumUnits) * mNumUnits * mNumLayers) * elementSize;
    size_t biasOffset = biasStartOffset;
    int32_t numGates = 8;
    for (int32_t layerIndex = 0; layerIndex < mNumLayers; layerIndex++)
    {
        for (int32_t gateIndex = 0; gateIndex < numGates; gateIndex++)
        {
            // encoder input size == mNumUnits
            int64_t inputSize = mNumUnits;
            nvinfer1::Weights gateKernelWeights{dataType, &mWeights->mWeights[0] + kernelOffset, inputSize * mNumUnits};
            nvinfer1::Weights gateBiasWeights{dataType, &mWeights->mWeights[0] + biasOffset, mNumUnits};
            mGateKernelWeights.push_back(std::move(gateKernelWeights));
            mGateBiasWeights.push_back(std::move(gateBiasWeights));
            kernelOffset = kernelOffset + inputSize * mNumUnits * elementSize;
            biasOffset = biasOffset + mNumUnits * elementSize;
        }
    }
    ASSERT(kernelOffset + biasOffset - biasStartOffset == mWeights->mWeights.size());
}

void LSTMEncoder::addToModel(nvinfer1::INetworkDefinition* network, int32_t maxInputSequenceLength,
    nvinfer1::ITensor* inputEmbeddedData, nvinfer1::ITensor* actualInputSequenceLengths,
    nvinfer1::ITensor** inputStates, nvinfer1::ITensor** memoryStates, nvinfer1::ITensor** lastTimestepStates)
{
    auto encoderLayer = network->addRNNv2(
        *inputEmbeddedData, mNumLayers, mNumUnits, maxInputSequenceLength, nvinfer1::RNNOperation::kLSTM);
    ASSERT(encoderLayer != nullptr);
    encoderLayer->setName("LSTM encoder");

    encoderLayer->setSequenceLengths(*actualInputSequenceLengths);
    encoderLayer->setInputMode(nvinfer1::RNNInputMode::kLINEAR);
    encoderLayer->setDirection(nvinfer1::RNNDirection::kUNIDIRECTION);

    std::vector<nvinfer1::RNNGateType> gateOrder({nvinfer1::RNNGateType::kFORGET, nvinfer1::RNNGateType::kINPUT,
        nvinfer1::RNNGateType::kCELL, nvinfer1::RNNGateType::kOUTPUT});
    for (size_t i = 0; i < mGateKernelWeights.size(); i++)
    {
        // we have 4 + 4 gates
        bool isW = ((i % 8) < 4);
        encoderLayer->setWeightsForGate(i / 8, gateOrder[i % 4], isW, mGateKernelWeights[i]);
        encoderLayer->setBiasForGate(i / 8, gateOrder[i % 4], isW, mGateBiasWeights[i]);
    }

    encoderLayer->setHiddenState(*inputStates[0]);
    encoderLayer->setCellState(*inputStates[1]);
    *memoryStates = encoderLayer->getOutput(0);
    ASSERT(*memoryStates != nullptr);

    if (lastTimestepStates)
    {
        // Per layer hidden output
        lastTimestepStates[0] = encoderLayer->getOutput(1);
        ASSERT(lastTimestepStates[0] != nullptr);

        // Per layer cell output
        lastTimestepStates[1] = encoderLayer->getOutput(2);
        ASSERT(lastTimestepStates[1] != nullptr);
    }
}

int32_t LSTMEncoder::getMemoryStatesSize()
{
    return mNumUnits;
}

std::vector<nvinfer1::Dims> LSTMEncoder::getStateSizes()
{
    nvinfer1::Dims hiddenStateDims{2, {mNumLayers, mNumUnits}};
    nvinfer1::Dims cellStateDims{2, {mNumLayers, mNumUnits}};
    return std::vector<nvinfer1::Dims>({hiddenStateDims, cellStateDims});
}

std::string LSTMEncoder::getInfo()
{
    std::stringstream ss;
    ss << "LSTM Encoder, num layers = " << mNumLayers << ", num units = " << mNumUnits;
    return ss.str();
}
} // namespace nmtSample
