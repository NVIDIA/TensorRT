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

#ifndef SAMPLE_NMT_DECODER_
#define SAMPLE_NMT_DECODER_

#include <memory>
#include <vector>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Decoder
 *
 * \brief encodes single input into output states
 *
 */
class Decoder : public Component
{
public:
    typedef std::shared_ptr<Decoder> ptr;

    Decoder() = default;

    /**
     * \brief add the memory, cell, and hidden states to the network
     */
    virtual void addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* inputData,
        nvinfer1::ITensor** inputStates, nvinfer1::ITensor** outputData, nvinfer1::ITensor** outputStates)
        = 0;

    /**
     * \brief get the sizes (vector of them) of the hidden state vectors
     */
    virtual std::vector<nvinfer1::Dims> getStateSizes() = 0;

    ~Decoder() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_DECODER_
