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

#ifndef TRT_BERT_ENCODER_H
#define TRT_BERT_ENCODER_H

#include "transformer.h"
#include <sstream>

namespace bert
{

inline ILayer* bertModelDynamic(const BertConfig& config, WeightMap& weightMap, INetworkDefinition* network, ITensor* inputTensor,
    ITensor* input_mask = nullptr)
{

    ITensor* prevInput = inputTensor;
    ILayer* prevLayer = nullptr;

    for (int layer = 0; layer < config.numHiddenLayers; layer++)
    {
        std::stringstream ss;
        ss << "l" << layer << "_";

        prevLayer = transformerDynamic(ss.str(), config, weightMap, network, prevInput, input_mask);
        prevInput = prevLayer->getOutput(0);
    }
    assert(prevLayer);

    return prevLayer;
}
}

#endif // TRT_BERT_ENCODER_H
