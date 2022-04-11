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
ILayer* parsePooling(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::PoolingParameter& p = msg.pooling_param();
    if (p.pool() != trtcaffe::PoolingParameter::MAX && p.pool() != trtcaffe::PoolingParameter::AVE)
    {
        std::cout << "Caffe Parser: only AVE and MAX pool operations are supported" << std::endl;
        return nullptr;
    }

    int kernelH, kernelW;
    if (p.has_global_pooling() && p.global_pooling())
    {
        Dims3 dims = parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());
        kernelH = dims.d[1];
        kernelW = dims.d[2];
    }
    else
    {
        // mandatory
        kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size();
        kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size();
    }

    PoolingType type = p.has_pool() && p.pool() == trtcaffe::PoolingParameter::AVE ? PoolingType::kAVERAGE : PoolingType::kMAX;
    auto layer = network.addPooling(*tensors[msg.bottom(0)], type, DimsHW{kernelH, kernelW});

    if (layer)
    {
        int stride = p.has_stride() ? p.stride() : 1;
        layer->setStride(DimsHW{p.has_stride_h() ? int(p.stride_h()) : stride, p.has_stride_w() ? int(p.stride_w()) : stride});

        int pad = p.has_pad() ? p.pad() : 0;
        layer->setPadding(DimsHW{p.has_pad_h() ? int(p.pad_h()) : pad, p.has_pad_w() ? int(p.pad_w()) : pad});

        layer->setName(msg.name().c_str());
		layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_UP); // caffe pool use ceil mode by default
        // FB pooling parameters
        // Use floor((height + 2 * padding - kernel) / stride) + 1
        // instead of ceil((height + 2 * padding - kernel) / stride) + 1
        if (p.has_torch_pooling() ? p.torch_pooling() : false)
        {
		    layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN); // facebook torch pool use floor mode
		}

        tensors[msg.top(0)] = layer->getOutput(0);

        layer->setAverageCountExcludesPadding(false); // unlike other frameworks, caffe use inclusive counting for padded averaging
    }
    return layer;
}
} //namespace nvcaffeparser1
