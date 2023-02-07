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

#ifndef TRT_CAFFE_PARSER_OP_PARSERS_H
#define TRT_CAFFE_PARSER_OP_PARSERS_H
#include <unordered_map>
#include <iostream>

#include "caffeMacros.h"
#include "NvInfer.h"
#include "trtcaffe.pb.h"
#include "parserUtils.h"
#include "half.h"
#include "blobNameToTensor.h"
#include "caffeWeightFactory.h"

namespace nvcaffeparser1
{
inline bool checkBlobs(const trtcaffe::LayerParameter& msg, int bottoms, int tops)
{
    if (msg.bottom_size() != bottoms)
    {
        std::cout << msg.name() << ": expected " << bottoms << " bottom blobs, found " << msg.bottom_size() << std::endl;
        return false;
    }

    if (msg.top_size() != tops)
    {
        std::cout << msg.name() << ": expected " << tops << " tops blobs, found " << msg.top_size() << std::endl;
        return false;
    }
    return true;
}

typedef nvinfer1::ILayer* (*LayerParseFn)(nvinfer1::INetworkDefinition&, const trtcaffe::LayerParameter&, CaffeWeightFactory&, BlobNameToTensor&);

nvinfer1::ILayer* parseAbsVal(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseBatchNormalization(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseBNLL(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseClip(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseConcat(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseConvolution(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseCrop(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseDeconvolution(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseEltwise(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseELU(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseInnerProduct(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseLRN(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parsePermute(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parsePooling(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parsePower(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parsePReLU(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseReduction(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseReLU(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseReshape(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseScale(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseSigmoid(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseSoftMax(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);
nvinfer1::ILayer* parseTanH(nvinfer1::INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors);

static std::unordered_map<std::string, LayerParseFn> gParseTable
{
    {"Convolution", parseConvolution},
    {"Pooling", parsePooling},
    {"InnerProduct", parseInnerProduct},
    {"ReLU", parseReLU},
    {"Softmax", parseSoftMax},
    {"SoftmaxWithLoss", parseSoftMax},
    {"LRN", parseLRN},
    {"Power", parsePower},
    {"Eltwise", parseEltwise},
    {"Concat", parseConcat},
    {"Deconvolution", parseDeconvolution},
    {"Sigmoid", parseSigmoid},
    {"TanH", parseTanH},
    {"BatchNorm", parseBatchNormalization},
    {"Scale", parseScale},
    {"Crop", parseCrop},
    {"Reduction", parseReduction},
    {"Reshape", parseReshape},
    {"Permute", parsePermute},
    {"ELU", parseELU},
    {"BNLL", parseBNLL},
    {"Clip", parseClip},
    {"AbsVal", parseAbsVal},
    {"PReLU", parsePReLU}
};
} // namespace nvcaffeparser1
#endif //TRT_CAFFE_PARSER_OP_PARSERS_H
