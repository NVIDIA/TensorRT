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

#ifndef TRT_SQUAD_H
#define TRT_SQUAD_H

namespace bert
{

const std::string SQD_W = "squad_output_weights";
const std::string SQD_B = "squad_output_bias";

inline ILayer* squadDynamic(const std::string& prefix, const BertConfig& config, WeightMap& weightMap, INetworkDefinition* network,
    ITensor* inputTensor)
{

    assert(inputTensor);
    assert(network);

    //const Dims idims = inputTensor->getDimensions();
    //assert(idims.nbDims == 5);

    //const int S = idims.d[1];
    const int hiddenSize = config.hiddenSize; //idims.d[2];

    const Weights W_out = weightMap.at(prefix + SQD_W);
    const Weights B_out = weightMap.at(prefix + SQD_B);

    IFullyConnectedLayer* logitsLayer = network->addFullyConnected(*inputTensor, 2, W_out, B_out);
    assert(logitsLayer);

    setOutputName(logitsLayer, prefix, "squad_logits");

    return logitsLayer;
}
}
#endif // TRT_SQUAD_H
