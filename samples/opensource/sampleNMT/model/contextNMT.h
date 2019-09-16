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

#ifndef SAMPLE_NMT_CONTEXT_
#define SAMPLE_NMT_CONTEXT_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Context
 *
 * \brief calculates context vector from raw alignment scores and memory states
 *
 */
class Context : public Component
{
public:
    typedef std::shared_ptr<Context> ptr;

    Context() = default;

    /**
     * \brief add the context vector calculation to the network
     */
    void addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* actualInputSequenceLengths,
        nvinfer1::ITensor* memoryStates, nvinfer1::ITensor* alignmentScores, nvinfer1::ITensor** contextOutput);

    std::string getInfo() override;

    ~Context() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_CONTEXT_
