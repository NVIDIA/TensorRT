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

#include "slpProjection.h"
#include "common.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
SLPProjection::SLPProjection(ComponentWeights::ptr weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    assert(mWeights->mMetaData.size() >= 3);
    mKernelWeights.type = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    assert(mKernelWeights.type == nvinfer1::DataType::kFLOAT);
    // Resize dimensions to be multiples of gPadMultiple for performance
    mInputChannelCount = samplesCommon::roundUp(mWeights->mMetaData[1], gPadMultiple);  // matches embedder outputs
    mOutputChannelCount = samplesCommon::roundUp(mWeights->mMetaData[2], gPadMultiple); // matches embedder inputs
    mResizedKernelWeights = resizeWeights(mWeights->mMetaData[1], mWeights->mMetaData[2], mInputChannelCount,
        mOutputChannelCount, (const float*) &mWeights->mWeights[0]);
    mKernelWeights.values = mResizedKernelWeights.data();
    mKernelWeights.count = mInputChannelCount * mOutputChannelCount;
}

void SLPProjection::addToModel(
    nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input, nvinfer1::ITensor** outputLogits)
{
    nvinfer1::Dims weightDims{2, {mInputChannelCount, mOutputChannelCount},
        {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kCHANNEL}};
    auto constLayer = network->addConstant(weightDims, mKernelWeights);
    assert(constLayer != nullptr);
    constLayer->setName("Projection matrix");
    auto weights = constLayer->getOutput(0);
    assert(weights != nullptr);

    auto mmLayer = network->addMatrixMultiply(*input, false, *weights, false);
    assert(mmLayer != nullptr);
    mmLayer->setName("Projection Matrix Multiply");
    *outputLogits = mmLayer->getOutput(0);
    assert(*outputLogits != nullptr);
}

int SLPProjection::getOutputSize()
{
    return mOutputChannelCount;
}

std::string SLPProjection::getInfo()
{
    std::stringstream ss;
    ss << "SLP Projection, num inputs = " << mInputChannelCount << ", num outputs = " << mOutputChannelCount;
    return ss.str();
}
} // namespace nmtSample
