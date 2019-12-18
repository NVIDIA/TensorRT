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
#include <fstream>
#include <random>
#include <numeric>
#include <unordered_map>
#include <vector>
#if CUDA_VERSION < 10000
#include <half.h>
#else
#include <cuda_fp16.h>
#endif

#include "NvInfer.h"

#include "sampleDevice.h"

namespace sample
{

template <typename T>
inline T roundUp(T m, T n) { return ((m + n - 1) / n) * n; }

inline int volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}

inline int volume(nvinfer1::Dims dims, int vecDim, int comps, int batch)
{
    if (vecDim != -1)
    {
        dims.d[vecDim] = roundUp(dims.d[vecDim], comps);
    }
    return volume(dims) * std::max(batch, 1);
}

inline
std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? "x" : "") << dims.d[i];
    }
    return os;
}

inline int dataTypeSize(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

template <typename T>
inline void fillBuffer(void* buffer, int volume, T min, T max)
{
    T* typedBuffer = static_cast<T*>(buffer);
    std::default_random_engine engine;
    if (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<int> distribution(min, max);
        auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
        std::generate(typedBuffer, typedBuffer + volume, generator);
    }
    else
    {
        std::uniform_real_distribution<float> distribution(min, max);
        auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
        std::generate(typedBuffer, typedBuffer + volume, generator);
    }
}

// Specialization needed for custom type __half
template <typename H>
inline void fillBufferHalf(void* buffer, int volume, H min, H max)
{
    H* typedBuffer = static_cast<H*>(buffer);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<H>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
}
template <>
#if CUDA_VERSION < 10000
inline void fillBuffer<half_float::half>(void* buffer, int volume, half_float::half min, half_float::half max)
#else
inline void fillBuffer<__half>(void* buffer, int volume, __half min, __half max)
#endif
{
    fillBufferHalf(buffer, volume, min, max);
}

template <typename T>
inline void dumpBuffer(const void* buffer, int volume, const std::string& separator, std::ostream& os)
{
    const T* typedBuffer = static_cast<const T*>(buffer);
    std::string sep;
    for (int v = 0; v < volume; ++v)
    {
        os << sep << typedBuffer[v];
        sep = separator;
    }
}

struct Binding
{
    bool isInput{false};
    MirroredBuffer buffer;
    int volume{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};

    void fill(const std::string& fileName)
    {
        std::ifstream file(fileName, std::ios::in|std::ios::binary);
        if (file.is_open())
        {
            file.read(static_cast<char*>(buffer.getHostBuffer()), buffer.getSize());
            file.close();
        }
    }

    void fill()
    {
        switch (dataType)
        {
        case nvinfer1::DataType::kBOOL:
        {
            fillBuffer<bool>(buffer.getHostBuffer(), volume, 0, 1);
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            fillBuffer<int32_t>(buffer.getHostBuffer(), volume, -128, 127);
            break;
        }
        case nvinfer1::DataType::kINT8:
        {
            fillBuffer<int8_t>(buffer.getHostBuffer(), volume, -128, 127);
            break;
        }
        case nvinfer1::DataType::kFLOAT:
        {
            fillBuffer<float>(buffer.getHostBuffer(), volume, -1.0, 1.0);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
#if CUDA_VERSION < 10000
            fillBuffer<half_float::half>(buffer.getHostBuffer(), volume, static_cast<half_float::half>(-1.0), static_cast<half_float::half>(-1.0));
#else
            fillBuffer<__half>(buffer.getHostBuffer(), volume, -1.0, 1.0);
#endif
            break;
        }
        }
    }

    void dump(std::ostream& os, const std::string separator = " ") const
    {
        switch (dataType)
        {
        case nvinfer1::DataType::kBOOL:
        {
            dumpBuffer<bool>(buffer.getHostBuffer(), volume, separator, os);
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            dumpBuffer<int32_t>(buffer.getHostBuffer(), volume, separator, os);
            break;
        }
        case nvinfer1::DataType::kINT8:
        {
            dumpBuffer<int8_t>(buffer.getHostBuffer(), volume, separator, os);
            break;
        }
        case nvinfer1::DataType::kFLOAT:
        {
            dumpBuffer<float>(buffer.getHostBuffer(), volume, separator, os);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
#if CUDA_VERSION < 10000
            dumpBuffer<half_float::half>(buffer.getHostBuffer(), volume, separator, os);
#else
            dumpBuffer<__half>(buffer.getHostBuffer(), volume, separator, os);
#endif
            break;
        }
        }
    }

};

class Bindings
{
public:

    void addBinding(int b, const std::string& name, bool isInput, int volume, nvinfer1::DataType dataType, const std::string& fileName = "")
    {
        while (mBindings.size() <= static_cast<size_t>(b))
        {
             mBindings.emplace_back();
             mDevicePointers.emplace_back();
        }
        mNames[name] = b;
        mBindings[b].isInput = isInput;
        mBindings[b].buffer.allocate(volume * dataTypeSize(dataType));
        mBindings[b].volume = volume;
        mBindings[b].dataType = dataType;
        mDevicePointers[b] = mBindings[b].buffer.getDeviceBuffer();
        if (isInput)
        {
            if (fileName.empty())
            {
                fill(b);
            }
            else
            {
                fill(b, fileName);
            }
        }
    }

    void** getDeviceBuffers() { return mDevicePointers.data(); }

    void transferInputToDevice(TrtCudaStream& stream)
    {
        for (auto& b : mNames)
        {
            if (mBindings[b.second].isInput)
            {
                mBindings[b.second].buffer.hostToDevice(stream);
            }
        }
    }

    void transferOutputToHost(TrtCudaStream& stream)
    {
        for (auto& b : mNames)
        {
            if (!mBindings[b.second].isInput)
            {
                mBindings[b.second].buffer.deviceToHost(stream);
            }
        }
    }

    void fill(int binding, const std::string& fileName)
    {
        mBindings[binding].fill(fileName);
    }

    void fill(int binding)
    {
        mBindings[binding].fill();
    }

    void dumpBindingDimensions(int binding, const nvinfer1::IExecutionContext& context, std::ostream& os) const
    {
        const auto dims = context.getBindingDimensions(binding);
        os << dims;
    }

    void dumpBindingValues(int binding, std::ostream& os, const std::string& separator = " ") const
    {
        mBindings[binding].dump(os, separator);
    }

    void dumpInputs(const nvinfer1::IExecutionContext& context, std::ostream& os) const
    {
        auto isInput = [](const Binding& b) { return b.isInput; };
        dumpBindings(context, isInput, os);
    }

    void dumpOutputs(const nvinfer1::IExecutionContext& context, std::ostream& os) const
    {
        auto isOutput = [](const Binding& b) { return !b.isInput; };
        dumpBindings(context, isOutput, os);
    }

    void dumpBindings(const nvinfer1::IExecutionContext& context, std::ostream& os) const
    {
        auto all = [](const Binding& b) { return true; };
        dumpBindings(context, all, os);
    }

    void dumpBindings(const nvinfer1::IExecutionContext& context, bool (*predicate)(const Binding& b), std::ostream& os) const
    {
        for (const auto& n : mNames)
        {
            const auto binding = n.second;
            if (predicate(mBindings[binding]))
            {
                os << n.first << ": (";
                dumpBindingDimensions(binding, context, os);
                os << ")" << std::endl;
                dumpBindingValues(binding, os);
                os << std::endl;
            }
        }
    }

    std::unordered_map<std::string, int> getInputBindings() const
    {
        auto isInput = [](const Binding& b) { return b.isInput; };
        return getBindings(isInput);
    }

    std::unordered_map<std::string, int> getOutputBindings() const
    {
        auto isOutput = [](const Binding& b) { return !b.isInput; };
        return getBindings(isOutput);
    }

    std::unordered_map<std::string, int> getBindings() const
    {
        auto all = [](const Binding& b) { return true; };
        return getBindings(all);
    }

    std::unordered_map<std::string, int> getBindings(bool (*predicate)(const Binding& b)) const
    {
        std::unordered_map<std::string, int> bindings;
        for (const auto& n : mNames)
        {
            const auto binding = n.second;
            if (predicate(mBindings[binding]))
            {
                bindings.insert(n);
            }
        }
        return bindings;
    }

private:

    std::unordered_map<std::string, int> mNames;
    std::vector<Binding> mBindings;
    std::vector<void*> mDevicePointers;
};

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t) { t->destroy(); }
};

template <typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T> >;

} // namespace sample

#endif // TRT_SAMPLE_UTILS_H
