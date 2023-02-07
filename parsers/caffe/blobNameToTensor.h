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

#ifndef TRT_CAFFE_PARSER_BLOB_NAME_TO_TENSOR_H
#define TRT_CAFFE_PARSER_BLOB_NAME_TO_TENSOR_H

#include <map>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"

namespace nvcaffeparser1
{
class BlobNameToTensor : public IBlobNameToTensor
{
public:
    void add(const std::string& name, nvinfer1::ITensor* tensor)
    {
        mMap[name] = tensor;
    }

    nvinfer1::ITensor* find(const char* name) const noexcept override
    {
        auto p = mMap.find(name);
        if (p == mMap.end())
        {
            return nullptr;
        }
        return p->second;
    }

    nvinfer1::ITensor*& operator[](const std::string& name)
    {
        return mMap[name];
    }

    void setTensorNames()
    {
        for (auto& p : mMap)
        {
            p.second->setName(p.first.c_str());
        }
    }

    ~BlobNameToTensor() override = default;

    bool isOK()
    {
        return !mError;
    }

private:
    std::map<std::string, nvinfer1::ITensor*> mMap;
    bool mError{false};
};
} // namespace nvcaffeparser1
#endif // TRT_CAFFE_PARSER_BLOB_NAME_TO_TENSOR_H
