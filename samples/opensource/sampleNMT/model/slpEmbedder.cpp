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

#include "slpEmbedder.h"
#include "common.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
SLPEmbedder::SLPEmbedder(ComponentWeights::ptr weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    assert(mWeights->mMetaData.size() >= 3);
    mKernelWeights.type = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    assert(mKernelWeights.type == nvinfer1::DataType::kFLOAT);
    // Resize dimensions to be multiples of gPadMultiple for performance
    mNumInputs = samplesCommon::roundUp(mWeights->mMetaData[1], gPadMultiple);  // matches projection output channels
    mNumOutputs = samplesCommon::roundUp(mWeights->mMetaData[2], gPadMultiple); // matches projection input channels
    mResizedKernelWeights = resizeWeights(
        mWeights->mMetaData[1], mWeights->mMetaData[2], mNumInputs, mNumOutputs, (const float*) &mWeights->mWeights[0]);
    mKernelWeights.values = mResizedKernelWeights.data();
    mKernelWeights.count = mNumInputs * mNumOutputs;
}

void SLPEmbedder::addToModel(
    nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input, nvinfer1::ITensor** output)
{
    nvinfer1::Dims weightDims{
        2, {mNumInputs, mNumOutputs}, {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kCHANNEL}};
    auto constLayer = network->addConstant(weightDims, mKernelWeights);
    assert(constLayer != nullptr);
    constLayer->setName("Embedding matrix");
    auto weights = constLayer->getOutput(0);
    assert(weights != nullptr);

    auto gatherLayer = network->addGather(*weights, *input, 0);
    assert(gatherLayer != nullptr);
    gatherLayer->setName("Gather in embedding");
    *output = gatherLayer->getOutput(0);
    assert(*output != nullptr);
}

int SLPEmbedder::getInputDimensionSize()
{
    return mNumInputs;
}

std::string SLPEmbedder::getInfo()
{
    std::stringstream ss;
    ss << "SLP Embedder, num inputs = " << mNumInputs << ", num outputs = " << mNumOutputs;
    return ss.str();
}
} // namespace nmtSample
