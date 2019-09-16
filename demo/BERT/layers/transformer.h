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

#ifndef TRT_TRANSFORMER_H
#define TRT_TRANSFORMER_H

#include "attention.h"
#include "geluPlugin.h"
#include "skipLayerNormPlugin.h"
#include "transformerKeys.h"

namespace bert
{
inline ILayer* skiplnDynamic(
    const std::string& prefix,const BertConfig & config,  WeightMap& weightMap, INetworkDefinition* network, ITensor* inputTensor, ITensor* skip)
{

    //const Dims idims = inputTensor->getDimensions();
    //assert(idims.nbDims == 5);
    //const int hiddenSize = idims.d[2];
    const int hiddenSize = config.hiddenSize;

    const Weights& wbeta = weightMap.at(prefix + "beta");
    const Weights& wgamma = weightMap.at(prefix + "gamma");
    test::SkipLayerNormPluginDynamic skipln_plug("skipln", hiddenSize, wbeta, wgamma);
    ITensor* skiplnInputs[2] = {inputTensor, skip};

    IPluginV2Layer* skiplnLayer = network->addPluginV2(skiplnInputs, 2, skipln_plug);
    return skiplnLayer;
}

inline ILayer* transformerDynamic(const std::string& prefix, const BertConfig& config, WeightMap& weightMap,
    INetworkDefinition* network, ITensor* inputTensor, ITensor* imask = nullptr)
{

    assert(inputTensor);
    assert(network);

    //const Dims idims = inputTensor->getDimensions();
    //assert(idims.nbDims == 5);
    const int hiddenSize = config.hiddenSize;

    ILayer* attentionHeads = attentionDynamic(prefix + "attention_self_", config, weightMap, network, inputTensor, imask);

    const Weights wA = weightMap.at(prefix + W_AOUT);
    const Weights bA = weightMap.at(prefix + B_AOUT);

    IFullyConnectedLayer* attOutFCLayer = network->addFullyConnected(*attentionHeads->getOutput(0), hiddenSize, wA, bA);

    ILayer* attLNLayer
        = skiplnDynamic(prefix + "attention_output_layernorm_",config, weightMap, network, attOutFCLayer->getOutput(0), inputTensor);

    const Weights wMid = weightMap.at(prefix + W_MID);
    const Weights bMid = weightMap.at(prefix + B_MID);

    IFullyConnectedLayer* midDenseLayer
        = network->addFullyConnected(*attLNLayer->getOutput(0), config.intermediateSize, wMid, bMid);

    // gelu
    auto geluPlugin = test::GeluPluginDynamic("gelu");
    ITensor* midDenseOut = midDenseLayer->getOutput(0);
    IPluginV2Layer* geluLayer = network->addPluginV2(&midDenseOut, 1, geluPlugin);
    ITensor* midAct = geluLayer->getOutput(0);
    assert(midAct);
    setTensorName(midAct, prefix, "gelu");

    // dense to hidden size
    const Weights wL = weightMap.at(prefix + W_LOUT);
    const Weights bL = weightMap.at(prefix + B_LOUT);

    IFullyConnectedLayer* outDenseLayer = network->addFullyConnected(*midAct, hiddenSize, wL, bL);
    setOutputName(outDenseLayer, prefix + "output_", "dense");

    ILayer* outLNLayer = skiplnDynamic(
        prefix + "output_layernorm_", config, weightMap, network, outDenseLayer->getOutput(0), attLNLayer->getOutput(0));

    assert(outLNLayer);

    setOutputName(outLNLayer, prefix + "output_", "reshape");

    return outLNLayer;
}
}
#endif // TRT_TRANSFORMER_H
