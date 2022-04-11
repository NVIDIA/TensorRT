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

#ifndef TRT_CAFFE_PARSER_CAFFE_PARSER_H
#define TRT_CAFFE_PARSER_CAFFE_PARSER_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

#include "NvCaffeParser.h"
#include "caffeWeightFactory.h"
#include "blobNameToTensor.h"
#include "trtcaffe.pb.h"

namespace nvcaffeparser1
{
class CaffeParser : public ICaffeParser
{
public:
    const IBlobNameToTensor* parse(const char* deploy,
                                   const char* model,
                                   nvinfer1::INetworkDefinition& network,
                                   nvinfer1::DataType weightType) noexcept override;

    const IBlobNameToTensor* parseBuffers(const uint8_t* deployBuffer,
                                          size_t deployLength,
                                          const uint8_t* modelBuffer,
                                          size_t modelLength,
                                          nvinfer1::INetworkDefinition& network,
                                          nvinfer1::DataType weightType) noexcept override;

    void setProtobufBufferSize(size_t size) noexcept override { mProtobufBufferSize = size; }
    void setPluginFactoryV2(nvcaffeparser1::IPluginFactoryV2* factory) noexcept override { mPluginFactoryV2 = factory; }
    void setPluginNamespace(const char* libNamespace) noexcept override { mPluginNamespace = libNamespace; }
    IBinaryProtoBlob* parseBinaryProto(const char* fileName) noexcept override;
    void destroy() noexcept override { delete this; }
    void setErrorRecorder(nvinfer1::IErrorRecorder* recorder) noexcept override { (void)recorder; assert(!"TRT- Not implemented."); }
    nvinfer1::IErrorRecorder* getErrorRecorder() const noexcept override { assert(!"TRT- Not implemented."); return nullptr; }

private:
    ~CaffeParser() noexcept override;
    std::vector<nvinfer1::PluginField> parseNormalizeParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<nvinfer1::PluginField> parsePriorBoxParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<nvinfer1::PluginField> parseDetectionOutputParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<nvinfer1::PluginField> parseLReLUParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<nvinfer1::PluginField> parseRPROIParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    template <typename T>
    T* allocMemory(int size = 1)
    {
        T* tmpMem = static_cast<T*>(malloc(sizeof(T) * size));
        mTmpAllocs.push_back(tmpMem);
        return tmpMem;
    }

    const IBlobNameToTensor* parse(nvinfer1::INetworkDefinition& network,
                                   nvinfer1::DataType weightType,
                                   bool hasModel);

private:
    std::shared_ptr<trtcaffe::NetParameter> mDeploy;
    std::shared_ptr<trtcaffe::NetParameter> mModel;
    std::vector<void*> mTmpAllocs;
    BlobNameToTensor* mBlobNameToTensor{nullptr};
    size_t mProtobufBufferSize{INT_MAX};
    nvcaffeparser1::IPluginFactoryV2* mPluginFactoryV2{nullptr};
    bool mPluginFactoryIsExt{false};
    std::vector<nvinfer1::IPluginV2*> mNewPlugins;
    std::unordered_map<std::string, nvinfer1::IPluginCreator*> mPluginRegistry;
    std::string mPluginNamespace = "";
};
} //namespace nvcaffeparser1
#endif //TRT_CAFFE_PARSER_CAFFE_PARSER_H
