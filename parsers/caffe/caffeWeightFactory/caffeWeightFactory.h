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

#ifndef TRT_CAFFE_PARSER_CAFFE_WEIGHT_FACTORY_H
#define TRT_CAFFE_PARSER_CAFFE_WEIGHT_FACTORY_H

#include <vector>
#include <string>
#include <random>
#include <memory>
#include "NvInfer.h"
#include "weightType.h"
#include "trtcaffe.pb.h"

namespace nvcaffeparser1
{
class CaffeWeightFactory
{
public:
    CaffeWeightFactory(const trtcaffe::NetParameter& msg, nvinfer1::DataType dataType, std::vector<void*>& tmpAllocs, bool isInitialized);
    nvinfer1::DataType getDataType() const;
    size_t getDataTypeSize() const;
    std::vector<void*>& getTmpAllocs();
    int getBlobsSize(const std::string& layerName);
    const trtcaffe::BlobProto* getBlob(const std::string& layerName, int index);
    std::vector<nvinfer1::Weights> getAllWeights(const std::string& layerName);
    virtual nvinfer1::Weights operator()(const std::string& layerName, WeightType weightType);
    void convert(nvinfer1::Weights& weights, nvinfer1::DataType targetType);
    void convert(nvinfer1::Weights& weights);
    bool isOK();
    bool isInitialized();
    nvinfer1::Weights getNullWeights();
    nvinfer1::Weights allocateWeights(int64_t elems, std::uniform_real_distribution<float> distribution = std::uniform_real_distribution<float>(-0.01f, 0.01F));
    nvinfer1::Weights allocateWeights(int64_t elems, std::normal_distribution<float> distribution);
    static trtcaffe::Type getBlobProtoDataType(const trtcaffe::BlobProto& blobMsg);
    static size_t sizeOfCaffeType(trtcaffe::Type type);
    // The size returned here is the number of array entries, not bytes
    static std::pair<const void*, size_t> getBlobProtoData(const  trtcaffe::BlobProto& blobMsg, trtcaffe::Type type, std::vector<void*>& tmpAllocs);

private:
    template <typename T>
    bool checkForNans(const void* values, int count, const std::string& layerName);
    nvinfer1::Weights getWeights(const trtcaffe::BlobProto& blobMsg, const std::string& layerName);

    const trtcaffe::NetParameter& mMsg;
    std::unique_ptr<trtcaffe::NetParameter> mRef;
    std::vector<void*>& mTmpAllocs;
    nvinfer1::DataType mDataType;
    // bool mQuantize;
    bool mInitialized;
    std::default_random_engine generator;
    bool mOK{true};
};
} //namespace nvcaffeparser1
#endif //TRT_CAFFE_PARSER_CAFFE_WEIGHT_FACTORY_H
