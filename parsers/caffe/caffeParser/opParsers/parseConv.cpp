/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parseConvolution(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::ConvolutionParameter& p = msg.convolution_param();
    int nbOutputs = p.num_output();

    int kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size(0);
    int kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);
    int C = parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions()).c();
    int G = p.has_group() ? p.group() : 1;

    auto CbyG = float(C / G * nbOutputs);
    float std_dev = 1.0F / sqrtf((kernelW * kernelH * sqrtf(CbyG)));
    Weights kernelWeights = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(kernelW * kernelH * CbyG, std::normal_distribution<float>(0.0F, std_dev));
    Weights biasWeights = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(nbOutputs)) : weightFactory.getNullWeights();

    weightFactory.convert(kernelWeights);
    weightFactory.convert(biasWeights);
    auto inTensor = tensors[msg.bottom(0)];
    auto layer = network.addConvolution(*inTensor, nbOutputs, DimsHW{kernelH, kernelW}, kernelWeights, biasWeights);

    if (layer)
    {
        int strideH = p.has_stride_h() ? p.stride_h() : p.stride_size() > 0 ? p.stride(0) : 1;
        int strideW = p.has_stride_w() ? p.stride_w() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;

        int padH = p.has_pad_h() ? p.pad_h() : p.pad_size() > 0 ? p.pad(0) : 0;
        int padW = p.has_pad_w() ? p.pad_w() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;

        int dilationH = p.dilation_size() > 0 ? p.dilation(0) : 1;
        int dilationW = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;

        layer->setStride(DimsHW{strideH, strideW});
        layer->setPadding(DimsHW{padH, padW});
		layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
        layer->setDilation(DimsHW{dilationH, dilationW});

        layer->setNbGroups(G);
    }
    return layer;
}
} //namespace nvcaffeparser1

