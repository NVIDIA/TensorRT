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

#ifndef SAMPLE_NMT_MULTIPLICATIVE_ALIGNMENT_
#define SAMPLE_NMT_MULTIPLICATIVE_ALIGNMENT_

#include "alignment.h"

#include "componentWeights.h"

namespace nmtSample
{
/** \class MultiplicativeAlignment
 *
 * \brief alignment scores from Luong attention mechanism
 *
 */
class MultiplicativeAlignment : public Alignment
{
public:
    MultiplicativeAlignment(ComponentWeights::ptr weights);

    void addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* attentionKeys,
        nvinfer1::ITensor* queryStates, nvinfer1::ITensor** alignmentScores) override;

    void addAttentionKeys(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* memoryStates,
        nvinfer1::ITensor** attentionKeys) override;

    int getSourceStatesSize() override;

    int getAttentionKeySize() override;

    std::string getInfo() override;

    ~MultiplicativeAlignment() override = default;

protected:
    ComponentWeights::ptr mWeights;
    nvinfer1::Weights mKernelWeights;
    int mInputChannelCount;
    int mOutputChannelCount;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_MULTIPLICATIVE_ALIGNMENT_
