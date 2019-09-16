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

#ifndef SAMPLE_NMT_SLP_ATTENTION_
#define SAMPLE_NMT_SLP_ATTENTION_

#include "attention.h"

#include "componentWeights.h"

namespace nmtSample
{
/** \class SLPAttention
 *
 * \brief Linear attention calculation
 *
 * Calculates attention vector by concatinating input from the decoder with context vector
 * and projecting the result into attention space by multiplying with weight matrix
 *
 */
class SLPAttention : public Attention
{
public:
    SLPAttention(ComponentWeights::ptr weights);

    void addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* inputFromDecoder,
        nvinfer1::ITensor* context, nvinfer1::ITensor** attentionOutput) override;

    int getAttentionSize() override;

    std::string getInfo() override;

protected:
    ComponentWeights::ptr mWeights;
    nvinfer1::Weights mKernelWeights;
    int mInputChannelCount;
    int mOutputChannelCount;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_SLP_ATTENTION_
