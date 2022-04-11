/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parsePReLU(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory,
                   BlobNameToTensor& tensors)
{
    // Caffe stores the slopes as weights rather than as a tensor, and only supports different slopes
    // per channel
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::PReLUParameter& p = msg.prelu_param();
    bool channelShared = p.has_channel_shared() ? p.channel_shared() : false;
    auto inputDims = tensors[msg.bottom(0)]->getDimensions();
    if (inputDims.nbDims < 2)
    {
        return nullptr;
    }

    int nWeights = channelShared ? 1 : inputDims.d[0]; // Caffe treats second input dimension as channels
    Dims slopesDims{inputDims.nbDims, {}};
    std::fill(slopesDims.d, slopesDims.d + slopesDims.nbDims, 1);
    slopesDims.d[0] = nWeights;

    Weights w = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) :
                weightFactory.allocateWeights(nWeights, std::uniform_real_distribution<float>(0.F, 1.F));
    auto constLayer = network.addConstant(slopesDims, w);
    return network.addParametricReLU(*tensors[msg.bottom(0)], *constLayer->getOutput(0));
}
} //namespace nvcaffeparser1
