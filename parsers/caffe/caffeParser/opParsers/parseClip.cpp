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
ILayer* parseClip(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }
    const trtcaffe::ClipParameter& p = msg.clip_param();
    float alpha = std::numeric_limits<float>::lowest(); // lower bound
    float beta = std::numeric_limits<float>::max();     // upper bound
    if(p.has_min())
    {
        alpha = p.min();
    }
    if(p.has_max())
    {
        beta = p.max();
    }
    auto layer = network.addActivation(*tensors[msg.bottom(0)], ActivationType::kCLIP);
    layer->setAlpha(alpha);
    layer->setBeta(beta);
    return layer;
}
} //namespace nvcaffeparser1
