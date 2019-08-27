/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TRT_ATTENTION_H
#define TRT_ATTENTION_H

#include "attentionKeys.h"
#include "qkvToContextPlugin.h"

namespace bert
{

ILayer* attention(const std::string& prefix, const BertConfig& config, WeightMap& weightMap,
    INetworkDefinition* network, ITensor* inputTensor, ITensor* inputMask = nullptr)
{
    assert(inputTensor);
    assert(network);

    const Dims idims = inputTensor->getDimensions();
    assert(idims.nbDims == 4);

    const int S = idims.d[0];
    const int hiddenSize = idims.d[1];
    const int numHeads = config.numAttentionHeads;
    const int headSize = hiddenSize / numHeads;

    assert(hiddenSize % numHeads == 0);

    const Weights Wall = weightMap.at(prefix + WQKV);
    const Weights Ball = weightMap.at(prefix + BQKV);

    IFullyConnectedLayer* multAllLayer = network->addFullyConnected(*inputTensor, 3 * hiddenSize, Wall, Ball);
    setOutputName(multAllLayer, prefix, "qkv_mult");

    ITensor* shuffleOut = multAllLayer->getOutput(0);

    const bool hasMask = inputMask != nullptr;
    QKVToContextPlugin qkvPlugin("qkv2ctx", hiddenSize, numHeads, S, hasMask);
    ITensor* qkvIn[2] = {shuffleOut, inputMask};
    IPluginV2Layer* qkv2ctxLayer = network->addPluginV2(qkvIn, 1 + hasMask, qkvPlugin);
    setOutputName(qkv2ctxLayer, prefix, "context_layer");
    return qkv2ctxLayer;
}
}
#endif // TRT_ATTENTION_H
