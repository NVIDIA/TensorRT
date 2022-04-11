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
ILayer* parseScale(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::ScaleParameter& p = msg.scale_param();
    int C = parserutils::getC(tensors[msg.bottom(0)]->getDimensions());

    Weights scale = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
    Weights shift = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(C)) : weightFactory.getNullWeights();
    Weights power = weightFactory.getNullWeights();
    weightFactory.convert(shift);
    weightFactory.convert(scale);
    weightFactory.convert(power);
    return network.addScale(*tensors[msg.bottom(0)], ScaleMode::kCHANNEL, shift, scale, power);
}
} //namespace nvcaffeparser1
