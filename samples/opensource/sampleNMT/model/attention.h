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

#ifndef SAMPLE_NMT_ATTENTION_
#define SAMPLE_NMT_ATTENTION_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Attention
 *
 * \brief calculates attention vector from context and decoder output vectors
 *
 */
class Attention : public Component
{
public:
    typedef std::shared_ptr<Attention> ptr;

    Attention() = default;

    /**
     * \brief add the attention vector calculation to the network
     */
    virtual void addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* inputFromDecoder,
        nvinfer1::ITensor* context, nvinfer1::ITensor** attentionOutput)
        = 0;

    /**
     * \brief get the size of the attention vector
     */
    virtual int getAttentionSize() = 0;

    ~Attention() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_ATTENTION_
