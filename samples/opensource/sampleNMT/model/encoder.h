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

#ifndef SAMPLE_NMT_ENCODER_
#define SAMPLE_NMT_ENCODER_

#include <memory>
#include <vector>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Encoder
 *
 * \brief encodes input sentences into output states
 *
 */
class Encoder : public Component
{
public:
    typedef std::shared_ptr<Encoder> ptr;

    Encoder() = default;

    /**
     * \brief add the memory and last timestep states to the network
     * lastTimestepHiddenStates is the pointer to the tensor where the encoder stores all layer hidden states for the
     * last timestep (which is dependent on the sample), the function should define the tensor, it could be nullptr
     * indicating these data are not needed
     */
    virtual void addToModel(nvinfer1::INetworkDefinition* network, int maxInputSequenceLength,
        nvinfer1::ITensor* inputEmbeddedData, nvinfer1::ITensor* actualInputSequenceLengths,
        nvinfer1::ITensor** inputStates, nvinfer1::ITensor** memoryStates, nvinfer1::ITensor** lastTimestepStates)
        = 0;

    /**
     * \brief get the size of the memory state vector
     */
    virtual int getMemoryStatesSize() = 0;

    /**
     * \brief get the sizes (vector of them) of the hidden state vectors
     */
    virtual std::vector<nvinfer1::Dims> getStateSizes() = 0;

    ~Encoder() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_ENCODER_
