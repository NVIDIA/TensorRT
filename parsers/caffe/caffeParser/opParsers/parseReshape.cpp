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
ILayer* parseReshape(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::ReshapeParameter& p = msg.reshape_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    int axis = p.has_axis() ? p.axis() : 0;

    const ::trtcaffe::BlobShape& shape = p.shape();
    // Check that N (batch dim) is 0. TensorRT does not support reshape in batch dimension
    if (network.hasImplicitBatchDimension() && (axis == 0) && (shape.dim(0) != 0))
    {
        std::cout << "Caffe Parser: Invalid reshape param. TensorRT does not support reshape in N (batch) dimension"
                  << std::endl;
        return nullptr;
    }

    // Handle axis and dims parameters
    int axStart = std::max(0, axis - 1);
    int axEnd = p.has_num_axes()
        ? std::max(0, axis - static_cast<int>(network.hasImplicitBatchDimension()) + p.num_axes())
        : bottomDims.nbDims;
    std::vector<int> reshapeDims;

    reshapeDims.reserve(axStart);
    for (int i = 0; i < axStart; i++)
    {
        reshapeDims.push_back(bottomDims.d[i]);
    }

    for (int i = 0; i < shape.dim_size(); i++)
    {
        // skip first 0 (batch)
        if (network.hasImplicitBatchDimension() && axis == 0 && i == 0)
        {
            continue;
        }
        if (shape.dim(i) == 0)
        {
            // If there is no bottom dimension corresponding to the current axis, then the params are invalid
            assert(static_cast<int>(reshapeDims.size()) <= bottomDims.nbDims);
            reshapeDims.push_back(bottomDims.d[reshapeDims.size()]);
        }
        else
        {
            reshapeDims.push_back(shape.dim(i));
        }
    }

    for (int i = axEnd; i < bottomDims.nbDims; i++)
    {
        reshapeDims.push_back(bottomDims.d[i]);
    }

    Dims topDims{};
    topDims.nbDims = static_cast<int>(reshapeDims.size());
    for (int i = 0; i < topDims.nbDims; i++)
    {
        topDims.d[i] = reshapeDims[i];
    }

    // Check there is at most one -1, and handle such case
    int countMinusOne = 0;
    for (int i = 0; i < topDims.nbDims; i++)
    {
        if (topDims.d[i] == -1)
        {
            countMinusOne += 1;
            // Inferred dimension
            int64_t newDim = parserutils::volume(bottomDims) / -parserutils::volume(topDims);
            topDims.d[i] = newDim;
        }
    }

    if (countMinusOne > 1)
    {
        std::cout << "Caffe Parser: Invalid reshape param. At most one axis can be inferred from other dimensions" << std::endl;
        return nullptr;
    }

    auto layer = network.addShuffle(*tensors[msg.bottom(0)]);
    layer->setReshapeDimensions(topDims);
    return layer;
}
} //namespace nvcaffeparser1
