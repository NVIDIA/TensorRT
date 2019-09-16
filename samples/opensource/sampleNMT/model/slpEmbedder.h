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

#ifndef SAMPLE_NMT_SLP_EMBEDDER_
#define SAMPLE_NMT_SLP_EMBEDDER_

#include "embedder.h"
#include "trtUtil.h"

#include "componentWeights.h"

#include "NvInfer.h"

extern int gPadMultiple;

namespace nmtSample
{
/** \class SLPEmbedder
 *
 * \brief selects the embedding vector from the weight matrix using index provided in the input
 *
 */
class SLPEmbedder : public Embedder
{
public:
    SLPEmbedder(ComponentWeights::ptr weights);

    void addToModel(
        nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input, nvinfer1::ITensor** output) override;

    int getInputDimensionSize() override;

    std::string getInfo() override;

    ~SLPEmbedder() override = default;

protected:
    ComponentWeights::ptr mWeights;
    nvinfer1::Weights mKernelWeights;
    int mNumInputs;
    int mNumOutputs;
    std::vector<float> mResizedKernelWeights;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_SLP_EMBEDDER_
