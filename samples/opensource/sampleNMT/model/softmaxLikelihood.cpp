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

#include "softmaxLikelihood.h"

#include <cassert>

#include <math.h>

namespace nmtSample
{
void SoftmaxLikelihood::addToModel(nvinfer1::INetworkDefinition* network, int beamWidth, nvinfer1::ITensor* inputLogits,
    nvinfer1::ITensor* inputLikelihoods, nvinfer1::ITensor** newCombinedLikelihoods,
    nvinfer1::ITensor** newRayOptionIndices, nvinfer1::ITensor** newVocabularyIndices)
{
    auto softmaxLayer = network->addSoftMax(*inputLogits);
    assert(softmaxLayer != nullptr);
    softmaxLayer->setName("Softmax in likelihood calculation");
    softmaxLayer->setAxes(2);
    auto softmaxTensor = softmaxLayer->getOutput(0);
    assert(softmaxTensor != nullptr);

    auto topKLayer = network->addTopK(*softmaxTensor, nvinfer1::TopKOperation::kMAX, beamWidth, 2);
    assert(topKLayer != nullptr);
    topKLayer->setName("TopK 1st in likelihood calculation");
    auto newLikelihoods = topKLayer->getOutput(0);
    assert(newLikelihoods != nullptr);
    auto vocabularyIndices = topKLayer->getOutput(1);
    assert(vocabularyIndices != nullptr);

    auto eltWiseLayer
        = network->addElementWise(*newLikelihoods, *inputLikelihoods, nvinfer1::ElementWiseOperation::kPROD);
    assert(eltWiseLayer != nullptr);
    eltWiseLayer->setName("EltWise multiplication in likelihood calculation");
    auto combinedLikelihoods = eltWiseLayer->getOutput(0);
    assert(combinedLikelihoods != nullptr);

    auto shuffleLayer = network->addShuffle(*combinedLikelihoods);
    assert(shuffleLayer != nullptr);
    shuffleLayer->setName("Reshape combined likelihoods");
    nvinfer1::Dims shuffleDims{1, {beamWidth * beamWidth}, {nvinfer1::DimensionType::kCHANNEL}};
    shuffleLayer->setReshapeDimensions(shuffleDims);
    auto reshapedCombinedLikelihoods = shuffleLayer->getOutput(0);
    assert(reshapedCombinedLikelihoods != nullptr);

    auto topKLayer2 = network->addTopK(*reshapedCombinedLikelihoods, nvinfer1::TopKOperation::kMAX, beamWidth, 1);
    assert(topKLayer2 != nullptr);
    topKLayer2->setName("TopK 2nd in likelihood calculation");
    *newCombinedLikelihoods = topKLayer2->getOutput(0);
    assert(*newCombinedLikelihoods != nullptr);
    *newRayOptionIndices = topKLayer2->getOutput(1);
    assert(*newRayOptionIndices != nullptr);

    auto shuffleLayer2 = network->addShuffle(*vocabularyIndices);
    assert(shuffleLayer2 != nullptr);
    shuffleLayer2->setName("Reshape vocabulary indices");
    nvinfer1::Dims shuffleDims2{1, {beamWidth * beamWidth}, {nvinfer1::DimensionType::kCHANNEL}};
    shuffleLayer2->setReshapeDimensions(shuffleDims2);
    auto reshapedVocabularyIndices = shuffleLayer2->getOutput(0);
    assert(reshapedVocabularyIndices != nullptr);

    auto gatherLayer = network->addGather(*reshapedVocabularyIndices, **newRayOptionIndices, 0);
    assert(gatherLayer != nullptr);
    gatherLayer->setName("Shuffle vocabulary indices");
    *newVocabularyIndices = gatherLayer->getOutput(0);
    assert(*newVocabularyIndices != nullptr);
}

float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::combine(
    float rayLikelihood, float optionLikelihood) const
{
    return rayLikelihood * optionLikelihood;
}

float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::init() const
{
    return 1.0F;
}

float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::smallerThanMinimalLikelihood() const
{
    return -1.0F;
}

LikelihoodCombinationOperator::ptr SoftmaxLikelihood::getLikelihoodCombinationOperator() const
{
    return std::make_shared<SoftmaxLikelihoodCombinationOperator>();
}

std::string SoftmaxLikelihood::getInfo()
{
    return "Softmax Likelihood";
}
} // namespace nmtSample
