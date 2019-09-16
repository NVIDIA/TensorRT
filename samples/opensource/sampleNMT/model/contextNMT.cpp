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

#include "contextNMT.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
void Context::addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* actualInputSequenceLengths,
    nvinfer1::ITensor* memoryStates, nvinfer1::ITensor* alignmentScores, nvinfer1::ITensor** contextOutput)
{
    auto raggedSoftmaxLayer = network->addRaggedSoftMax(*alignmentScores, *actualInputSequenceLengths);
    assert(raggedSoftmaxLayer != nullptr);
    raggedSoftmaxLayer->setName("Context Ragged Softmax");
    auto softmaxTensor = raggedSoftmaxLayer->getOutput(0);
    assert(softmaxTensor != nullptr);

    auto mmLayer = network->addMatrixMultiply(*softmaxTensor, false, *memoryStates, false);
    assert(mmLayer != nullptr);
    mmLayer->setName("Context Matrix Multiply");
    *contextOutput = mmLayer->getOutput(0);
    assert(*contextOutput != nullptr);
}

std::string Context::getInfo()
{
    return "Ragged softmax + Batch GEMM";
}
} // namespace nmtSample
