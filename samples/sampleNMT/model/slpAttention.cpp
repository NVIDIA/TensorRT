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
#include "slpAttention.h"
#include <sstream>

namespace nmtSample
{
SLPAttention::SLPAttention(ComponentWeights::ptr& weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    ASSERT(mWeights->mMetaData.size() >= 3);
    mKernelWeights.type = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    ASSERT(mKernelWeights.type == nvinfer1::DataType::kFLOAT);
    mInputChannelCount = mWeights->mMetaData[1];
    mOutputChannelCount = mWeights->mMetaData[2];

    mKernelWeights.values = (void*) (&mWeights->mWeights[0]);
    mKernelWeights.count = mInputChannelCount * mOutputChannelCount;
}

void SLPAttention::addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* inputFromDecoder,
    nvinfer1::ITensor* context, nvinfer1::ITensor** attentionOutput)
{
    nvinfer1::ITensor* inputTensors[] = {inputFromDecoder, context};
    auto concatLayer = network->addConcatenation(inputTensors, 2);
    ASSERT(concatLayer != nullptr);
    concatLayer->setName("Concatinate decoder output and context");
    concatLayer->setAxis(1);
    auto concatinatedTensor = concatLayer->getOutput(0);
    ASSERT(concatinatedTensor != nullptr);

    nvinfer1::Dims weightDims{2, {mInputChannelCount, mOutputChannelCount}};
    auto constLayer = network->addConstant(weightDims, mKernelWeights);
    ASSERT(constLayer != nullptr);
    constLayer->setName("Attention Matrix");
    auto weights = constLayer->getOutput(0);
    ASSERT(weights != nullptr);

    auto mmLayer
        = network->addMatrixMultiply(*concatinatedTensor, MatrixOperation::kNONE, *weights, MatrixOperation::kNONE);
    ASSERT(mmLayer != nullptr);
    mmLayer->setName("Attention Matrix Multiply");

    auto actLayer = network->addActivation(*mmLayer->getOutput(0), nvinfer1::ActivationType::kTANH);
    ASSERT(actLayer != nullptr);

    *attentionOutput = actLayer->getOutput(0);
    ASSERT(*attentionOutput != nullptr);
}

int32_t SLPAttention::getAttentionSize()
{
    return mOutputChannelCount;
}

std::string SLPAttention::getInfo()
{
    std::stringstream ss;
    ss << "SLP Attention, num inputs = " << mInputChannelCount << ", num outputs = " << mOutputChannelCount;
    return ss.str();
}
} // namespace nmtSample
