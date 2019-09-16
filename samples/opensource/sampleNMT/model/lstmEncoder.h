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

#ifndef SAMPLE_NMT_LSTM_ENCODER_
#define SAMPLE_NMT_LSTM_ENCODER_

#include "encoder.h"

#include "componentWeights.h"

namespace nmtSample
{
/** \class LSTMEncoder
 *
 * \brief encodes input sentences into output states using LSTM
 *
 */
class LSTMEncoder : public Encoder
{
public:
    LSTMEncoder(ComponentWeights::ptr weights);

    void addToModel(nvinfer1::INetworkDefinition* network, int maxInputSequenceLength,
        nvinfer1::ITensor* inputEmbeddedData, nvinfer1::ITensor* actualInputSequenceLengths,
        nvinfer1::ITensor** inputStates, nvinfer1::ITensor** memoryStates,
        nvinfer1::ITensor** lastTimestepStates) override;

    int getMemoryStatesSize() override;

    std::vector<nvinfer1::Dims> getStateSizes() override;

    std::string getInfo() override;

    ~LSTMEncoder() override = default;

protected:
    ComponentWeights::ptr mWeights;
    std::vector<nvinfer1::Weights> mGateKernelWeights;
    std::vector<nvinfer1::Weights> mGateBiasWeights;
    bool mRNNKind;
    int mNumLayers;
    int mNumUnits;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_LSTM_ENCODER_
