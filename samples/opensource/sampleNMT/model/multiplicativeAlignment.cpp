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

#include "multiplicativeAlignment.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
MultiplicativeAlignment::MultiplicativeAlignment(ComponentWeights::ptr weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    assert(mWeights->mMetaData.size() >= 3);
    mKernelWeights.type = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    assert(mKernelWeights.type == nvinfer1::DataType::kFLOAT);
    mInputChannelCount = mWeights->mMetaData[1];
    mOutputChannelCount = mWeights->mMetaData[2];

    mKernelWeights.values = (void*) (&mWeights->mWeights[0]);
    mKernelWeights.count = mInputChannelCount * mOutputChannelCount;
}

void MultiplicativeAlignment::addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* attentionKeys,
    nvinfer1::ITensor* queryStates, nvinfer1::ITensor** alignmentScores)
{
    auto mmLayer = network->addMatrixMultiply(*queryStates, false, *attentionKeys, true);
    assert(mmLayer != nullptr);
    mmLayer->setName("Raw Alignment Scores MM (Queries x Keys) in multiplicative attention");
    *alignmentScores = mmLayer->getOutput(0);
    assert(*alignmentScores != nullptr);
}

void MultiplicativeAlignment::addAttentionKeys(
    nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* memoryStates, nvinfer1::ITensor** attentionKeys)
{
    nvinfer1::Dims weightDims{2, {mInputChannelCount, mOutputChannelCount},
        {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kCHANNEL}};
    auto constLayer = network->addConstant(weightDims, mKernelWeights);
    assert(constLayer != nullptr);
    constLayer->setName("Matrix in multiplicative attention");
    auto weights = constLayer->getOutput(0);
    assert(weights != nullptr);

    auto mmLayer = network->addMatrixMultiply(*memoryStates, false, *weights, false);
    assert(mmLayer != nullptr);
    mmLayer->setName("Attention Keys MM in multiplicative attention");
    *attentionKeys = mmLayer->getOutput(0);
    assert(*attentionKeys != nullptr);
}

int MultiplicativeAlignment::getSourceStatesSize()
{
    return mInputChannelCount;
}

int MultiplicativeAlignment::getAttentionKeySize()
{
    return mOutputChannelCount;
}

std::string MultiplicativeAlignment::getInfo()
{
    std::stringstream ss;
    ss << "Multiplicative Alignment, source states size = " << mInputChannelCount
       << ", attention keys size = " << mOutputChannelCount;
    return ss.str();
}
} // namespace nmtSample
