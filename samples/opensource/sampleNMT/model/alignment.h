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

#ifndef SAMPLE_NMT_ALIGNMENT_
#define SAMPLE_NMT_ALIGNMENT_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Alignment
 *
 * \brief represents the core of attention mechanism
 *
 */
class Alignment : public Component
{
public:
    typedef std::shared_ptr<Alignment> ptr;

    Alignment() = default;

    /**
     * \brief add the alignment scores calculation to the network
     */
    virtual void addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* attentionKeys,
        nvinfer1::ITensor* queryStates, nvinfer1::ITensor** alignmentScores)
        = 0;

    /**
     * \brief add attention keys calculation (from source memory states) to the network
     *
     * The funtion is called if getAttentionKeySize returns positive value
     */
    virtual void addAttentionKeys(
        nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* memoryStates, nvinfer1::ITensor** attentionKeys)
        = 0;

    /**
     * \brief get the size of the source states
     */
    virtual int getSourceStatesSize() = 0;

    /**
     * \brief get the size of the attention keys
     */
    virtual int getAttentionKeySize() = 0;

    ~Alignment() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_ALIGNMENT_
