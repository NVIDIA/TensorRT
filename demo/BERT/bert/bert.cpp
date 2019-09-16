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

#include "bert.h"

#include "bertEncoder.h"
#include "bertUtils.h"
#include "dataUtils.h"
#include "embLayerNormPlugin.h"
#include "squad.h"
#include <common.h>
#include <iostream>
#include <numeric>

using namespace nvinfer1;

namespace bert
{

BERTDriver::BERTDriver(const int numHeads, const bool useFp16,
    const size_t maxWorkspaceSize, const OptProfiles& optProfiles)
    : DynamicDriver(useFp16, maxWorkspaceSize, optProfiles)
    , mNumHeads(numHeads)
{
}

void BERTDriver::buildNetwork(nvinfer1::INetworkDefinition* network, const HostTensorMap& params)
{
    WeightMap weightMap;
    for (auto kv : params)
    {
        const HostTensor& t = *kv.second;
        weightMap[kv.first] = Weights{t.mType, t.mData, (int64_t) t.mSize};
    }
    int intermediateSize = 0;
    int numHiddenLayers = 0;
    int hiddenSize = 0;

    gLogVerbose << "Inferring Network size" << endl;
    inferNetworkSizes(weightMap, hiddenSize, intermediateSize, numHiddenLayers);

    assert(intermediateSize);
    assert(hiddenSize);
    assert(numHiddenLayers);
    gLogVerbose << intermediateSize << endl;
    gLogVerbose << hiddenSize << endl;
    gLogVerbose << numHiddenLayers << endl;

    // create the model to populate the network, then set the outputs and create an engine
    ITensor* inputIds = network->addInput(kMODEL_INPUT0_NAME, DataType::kINT32, Dims{2, -1, -1});
    ITensor* segmentIds = network->addInput(kMODEL_INPUT1_NAME, DataType::kINT32, Dims{2, -1, -1});
    ITensor* inputMask = network->addInput(kMODEL_INPUT2_NAME, DataType::kINT32, Dims{2, -1, -1});

    const Weights& wBeta = weightMap.at("bert_embeddings_layernorm_beta");
    const Weights& wGamma = weightMap.at("bert_embeddings_layernorm_gamma");
    const Weights& wWordEmb = weightMap.at("bert_embeddings_word_embeddings");
    const Weights& wTokEmb = weightMap.at("bert_embeddings_token_type_embeddings");
    const Weights& wPosEmb = weightMap.at("bert_embeddings_position_embeddings");

    gLogVerbose << "embeddings params read" << endl;

    ITensor* inputs[3] = {inputIds, segmentIds, inputMask};

    auto embPlugin = test::EmbLayerNormPluginDynamic("embeddings", mUseFp16, wBeta, wGamma, wWordEmb, wPosEmb, wTokEmb);
    IPluginV2Layer* embLayer = network->addPluginV2(inputs, 3, embPlugin);
    embLayer->setName("EmbeddingsLayer");
    setOutputName(embLayer, "embeddings_", "output");

    ITensor* embeddings = embLayer->getOutput(0);
    ITensor* maskIdx = embLayer->getOutput(1);
    auto dims = embeddings->getDimensions();
    gLogVerbose << "emb out dims: " << dims << endl;

    /// BERT Encoder

    const BertConfig config(mNumHeads, hiddenSize, intermediateSize, numHiddenLayers, mUseFp16);

    ILayer* bertLayer = bertModelDynamic(config, weightMap, network, embeddings, maskIdx);

    /// SQuAD Output Layer

    ILayer* squadLayer = squadDynamic("cls_", config, weightMap, network, bertLayer->getOutput(0));

    network->markOutput(*squadLayer->getOutput(0));
}
}
