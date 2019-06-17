/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRT_CAFFE_PARSER_BINARY_PROTO_BLOB_H
#define TRT_CAFFE_PARSER_BINARY_PROTO_BLOB_H
#include <stdlib.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"

namespace nvcaffeparser1
{
class BinaryProtoBlob : public IBinaryProtoBlob
{
public:
    BinaryProtoBlob(void* memory, nvinfer1::DataType type, nvinfer1::DimsNCHW dimensions)
        : mMemory(memory)
        , mDataType(type)
        , mDimensions(dimensions)
    {
    }

    nvinfer1::DimsNCHW getDimensions() override
    {
        return mDimensions;
    }

    nvinfer1::DataType getDataType() override
    {
        return mDataType;
    }

    const void* getData() override
    {
        return mMemory;
    }

    void destroy() override
    {
        delete this;
    }

    ~BinaryProtoBlob() override
    {
        free(mMemory);
    }

    void* mMemory;
    nvinfer1::DataType mDataType;
    nvinfer1::DimsNCHW mDimensions;
};
} // namespace nvcaffeparser1
#endif // TRT_CAFFE_PARSER_BINARY_PROTO_BLOB_H