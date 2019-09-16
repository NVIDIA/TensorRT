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

#ifndef SAMPLE_NMT_SOFTMAX_LIKELIHOOD_
#define SAMPLE_NMT_SOFTMAX_LIKELIHOOD_

#include "NvInfer.h"
#include "likelihood.h"

namespace nmtSample
{
/** \class SoftmaxLikelihood
 *
 * \brief calculates softmax likelihood and TopK indices for the raw input logits
 *
 */
class SoftmaxLikelihood : public Likelihood
{
private:
    class SoftmaxLikelihoodCombinationOperator : public LikelihoodCombinationOperator
    {
    public:
        SoftmaxLikelihoodCombinationOperator() = default;

        float combine(float rayLikelihood, float optionLikelihood) const override;

        float init() const override;

        float smallerThanMinimalLikelihood() const override;

        ~SoftmaxLikelihoodCombinationOperator() override = default;
    };

public:
    SoftmaxLikelihood() = default;

    LikelihoodCombinationOperator::ptr getLikelihoodCombinationOperator() const override;

    void addToModel(nvinfer1::INetworkDefinition* network, int beamWidth, nvinfer1::ITensor* inputLogits,
        nvinfer1::ITensor* inputLikelihoods, nvinfer1::ITensor** newCombinedLikelihoods,
        nvinfer1::ITensor** newRayOptionIndices, nvinfer1::ITensor** newVocabularyIndices) override;

    std::string getInfo() override;

    ~SoftmaxLikelihood() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_SOFTMAX_LIKELIHOOD_
