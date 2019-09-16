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

#ifndef TRT_SAMPLE_UTILS_H
#define TRT_SAMPLE_UTILS_H

#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"

#include "sampleDevice.h"

namespace sample
{

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t) { t->destroy(); }
};

template <typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T> >;

inline int volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}

inline int dataTypeSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

template <typename T>
inline T roundUp(T m, T n) { return ((m + n - 1) / n) * n; }

class BindingBuffers
{
public:

    void allocate(size_t size)
    {
        mSize = size;
        mHostBuffer.allocate(size);
        mDeviceBuffer.allocate(size);
    }

    void* getDeviceBuffer() const { return mDeviceBuffer.get(); }

    void* getHostBuffer() const { return mHostBuffer.get(); }

private:

    int mSize{0};
    TrtHostBuffer mHostBuffer;
    TrtDeviceBuffer mDeviceBuffer;
};

class Bindings
{
public:

    void addBinding(int b, const std::string& name, size_t size)
    {
        while (mBuffers.size() <= static_cast<size_t>(b))
        {
             mBuffers.emplace_back();
             mDevicePointers.emplace_back();
        }
        mBindings[name] = b;
        mBuffers[b].allocate(size);
        mDevicePointers[b] = mBuffers[b].getDeviceBuffer();
    }

    void** getDeviceBuffers() { return mDevicePointers.data(); }

private:

    std::unordered_map<std::string, int> mBindings;
    std::vector<BindingBuffers> mBuffers;
    std::vector<void*> mDevicePointers; 
};

} // namespace sample

#endif // TRT_SAMPLE_UTILS_H
