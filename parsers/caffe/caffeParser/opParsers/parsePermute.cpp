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
ILayer* parsePermute(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::PermuteParameter& p = msg.permute_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    Dims topDims = tensors[msg.bottom(0)]->getDimensions();
    int nbDims = bottomDims.nbDims;

    std::vector<int> orders;
    std::vector<bool> knownOrders(nbDims + 1, false);
    bool orderAbort = (p.order(0) != 0); // First order must be 0 (batch dimension)
    for (int i = 0; i < p.order_size(); i++)
    {
        int order = p.order(i);
        orderAbort |= (order > nbDims) || (std::find(orders.begin(), orders.end(), order) != orders.end());
        orders.push_back(order);
        knownOrders[order] = true;
    }

    if (orderAbort)
    {
        std::cout << "Caffe Parser: Invalid permute param. TensorRT does not support permute in N (batch) dimension, and order index must be within the tensor dimensions. no duplicate order allowed." << std::endl;
        return nullptr;
    }

    // Keep the rest of the order
    for (int i = 0; i < nbDims; i++)
    {
        if (!knownOrders[i])
        {
            orders.push_back(i);
        }
    }

    // Remove the first order (batch)
    orders.erase(orders.begin());

    for (int i = 0; i < nbDims; i++)
    {
        topDims.d[i] = bottomDims.d[orders[i] - 1];
    }
    assert(parserutils::volume(topDims) == parserutils::volume(bottomDims));

    nvinfer1::Permutation permuteOrder;
    for (int i = 0; i < nbDims; i++)
    {
        permuteOrder.order[i] = orders[i] - 1;
    }

    auto permute = network.addShuffle(*tensors[msg.bottom(0)]);
    permute->setReshapeDimensions(topDims);
    permute->setFirstTranspose(permuteOrder);
    return permute;
}
} //namespace nvcaffeparser1
