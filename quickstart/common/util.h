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

#ifndef TENSORRT_SAMPLE_COMMON_UTILS_H
#define TENSORRT_SAMPLE_COMMON_UTILS_H
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"

namespace util
{

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, util::InferDeleter>;

size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size);

struct PPM
{
    std::string filename;
    std::string magic;
    int c;
    int h;
    int w;
    int max;
    std::vector<uint8_t> buffer;
};

class ImageBase
{
public:
    ImageBase(const std::string& filename, const nvinfer1::Dims& dims);
    virtual ~ImageBase() {}
    virtual size_t volume() const;
    void read();
    void write();
protected:
    nvinfer1::Dims mDims;
    PPM mPPM;
};

class RGBImageReader : public ImageBase
{
public:
    RGBImageReader(const std::string& filename, const nvinfer1::Dims& dims, const std::vector<float>& mean, const std::vector<float>& std);
    std::unique_ptr<float> process() const;
private:
    std::vector<float> mMean;
    std::vector<float> mStd;
};

class ArgmaxImageWriter : public ImageBase
{
public:
    ArgmaxImageWriter(const std::string& filename, const nvinfer1::Dims& dims, const std::vector<int>& palette, const int num_classes);
    void process(const int* buffer);
private:
    int mNumClasses;
    std::vector<int> mPalette;
};

}; // namespace util

#endif // TENSORRT_SAMPLE_COMMON_UTILS_H
