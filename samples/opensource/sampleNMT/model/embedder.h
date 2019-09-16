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

#ifndef SAMPLE_NMT_EMBEDDER_
#define SAMPLE_NMT_EMBEDDER_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Embedder
 *
 * \brief projects 1-hot vectors (represented as a vector with indices) into dense embedding space
 *
 */
class Embedder : public Component
{
public:
    typedef std::shared_ptr<Embedder> ptr;

    Embedder() = default;

    /**
     * \brief add the embedding vector calculation to the network
     */
    virtual void addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input, nvinfer1::ITensor** output)
        = 0;

    /**
     * \brief get the upper bound for the possible values of indices
     */
    virtual int getInputDimensionSize() = 0;

    ~Embedder() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_EMBEDDER_
