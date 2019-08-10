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

const std::string P_W = "bert_pooler_dense_kernel";
const std::string P_B = "bert_pooler_dense_bias";
const std::string O_W = "output_weights_multi";
const std::string O_B = "output_bias_multi";

ILayer* clf(const std::string& prefix, const BertConfig& config, WeightMap& weightMap, INetworkDefinition* network,
    ITensor* inputTensor)
{

    assert(inputTensor);
    assert(network);

    const Dims idims = inputTensor->getDimensions();
    assert(idims.nbDims == 4);

    const int S = idims.d[0];
    const int hiddenSize = idims.d[1];
    const Weights PW_out = weightMap.at(P_W);
    const Weights PB_out = weightMap.at(P_B);
    const Weights W_out = weightMap.at(O_W);
    const Weights B_out = weightMap.at(O_B);

    IFullyConnectedLayer* poolerLayer = network->addFullyConnected(*inputTensor, 768, PW_out, PB_out);
    ITensor* poolerTensor = poolerLayer->getOutput(0);
    IActivationLayer* poolerActLayer = network->addActivation(*poolerTensor, nvinfer1::ActivationType(2)); //TANH
    ITensor* poolerActTensor = poolerActLayer->getOutput(0);
    IFullyConnectedLayer* logitsLayer = network->addFullyConnected(*poolerActTensor, 1, W_out, B_out);
    assert(logitsLayer);

    setOutputName(logitsLayer, prefix, "clf_logits");

    return logitsLayer;
}
}
#endif // TRT_SQUAD_H
