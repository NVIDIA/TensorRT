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
ILayer* parseInnerProduct(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    const trtcaffe::InnerProductParameter& p = msg.inner_product_param();

    int64_t nbInputs = parserutils::volume(parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions()));
    int64_t nbOutputs = p.num_output();

    float std_dev = 1.0F / sqrtf(nbInputs * nbOutputs);
    Weights kernelWeights = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(nbInputs * nbOutputs, std::normal_distribution<float>(0.0F, std_dev));
    Weights biasWeights = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(nbOutputs)) : weightFactory.getNullWeights();

    weightFactory.convert(kernelWeights);
    weightFactory.convert(biasWeights);
    return network.addFullyConnected(*tensors[msg.bottom(0)], p.num_output(), kernelWeights, biasWeights);
}
} //namespace nvcaffeparser1
