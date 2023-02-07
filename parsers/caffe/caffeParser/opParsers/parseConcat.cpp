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
ILayer* parseConcat(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    const trtcaffe::ConcatParameter& p = msg.concat_param();
    bool hasAxis = p.has_axis(); // optional parameter

    if (hasAxis && p.axis() < 0)
    {
        std::cout << "Caffe parser: Concat negative axis is not supported." << std::endl;
        return nullptr;
    }
    if (network.hasImplicitBatchDimension() && p.axis() == 0)
    {
        std::cout << "Caffe parser: Concat across batch axis with implicit batch dimensions is not supported."
                  << std::endl;
        return nullptr;
    }

    std::vector<ITensor*> ptrs;
    for (unsigned int i = 0, n = msg.bottom_size(); i < n; i++)
    {
        ptrs.push_back(tensors[msg.bottom().Get(i)]);
    }

    auto concat = network.addConcatenation(&ptrs[0], msg.bottom_size());

    // If no axis is explicitly provided, do not call setAxis.
    // Rely on the default axis setting inside TRT which takes into account NPCHW and higher dimensional input.
    if (hasAxis)
    {
        concat->setAxis(p.axis() - static_cast<int>(network.hasImplicitBatchDimension()));
    }

    return concat;
}
} //namespace nvcaffeparser1
