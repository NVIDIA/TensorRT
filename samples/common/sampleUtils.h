/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

#include "NvInfer.h"

#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleOptions.h"

namespace sample
{

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
inline T roundUp(T m, T n)
{
    return ((m + n - 1) / n) * n;
}

inline int volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}

//! comps is the number of components in a vector. Ignored if vecDim < 0.
inline int64_t volume(const nvinfer1::Dims& dims, const nvinfer1::Dims& strides, int vecDim, int comps, int batch)
{
    int maxNbElems = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        // Get effective length of axis.
        int d = dims.d[i];
        // Any dimension is 0, it is an empty tensor.
        if (d == 0)
        {
            return 0;
        }
        if (i == vecDim)
        {
            d = samplesCommon::divUp(d, comps);
        }
        maxNbElems = std::max(maxNbElems, d * strides.d[i]);
    }
    return static_cast<int64_t>(maxNbElems) * batch * (vecDim < 0 ? 1 : comps);
}

inline int64_t volume(nvinfer1::Dims dims, int vecDim, int comps, int batch)
{
    if (vecDim != -1)
    {
        dims.d[vecDim] = roundUp(dims.d[vecDim], comps);
    }
    return volume(dims) * std::max(batch, 1);
}

inline nvinfer1::Dims toDims(const std::vector<int>& vec)
{
    int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
    if (static_cast<int>(vec.size()) > limit)
    {
        sample::gLogWarning << "Vector too long, only first 8 elements are used in dimension." << std::endl;
    }
    // Pick first nvinfer1::Dims::MAX_DIMS elements
    nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
    std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
    return dims;
}

template <typename T>
inline void fillBuffer(void* buffer, int64_t volume, T min, T max)
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
inline void fillBufferHalf(void* buffer, int64_t volume, H min, H max)
{
    H* typedBuffer = static_cast<H*>(buffer);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<H>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
}
template <>
inline void fillBuffer<__half>(void* buffer, int64_t volume, __half min, __half max)
{
    fillBufferHalf(buffer, volume, min, max);
}

template <typename T>
inline void dumpBuffer(const void* buffer, const std::string& separator, std::ostream& os, const Dims& dims,
    const Dims& strides, int32_t vectorDim, int32_t spv)
{
    const int64_t volume = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
    const T* typedBuffer = static_cast<const T*>(buffer);
    std::string sep;
    for (int64_t v = 0; v < volume; ++v)
    {
        int64_t curV = v;
        int32_t dataOffset = 0;
        for (int32_t dimIndex = dims.nbDims - 1; dimIndex >= 0; --dimIndex)
        {
            int32_t dimVal = curV % dims.d[dimIndex];
            if (dimIndex == vectorDim)
            {
                dataOffset += (dimVal / spv) * strides.d[dimIndex] * spv + dimVal % spv;
            }
            else
            {
                dataOffset += dimVal * strides.d[dimIndex] * (vectorDim == -1 ? 1 : spv);
            }
            curV /= dims.d[dimIndex];
            ASSERT(curV >= 0);
        }

        os << sep << typedBuffer[dataOffset];
        sep = separator;
    }
}

struct Binding
{
    bool isInput{false};
    std::unique_ptr<IMirroredBuffer> buffer;
    int64_t volume{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};

    void fill(const std::string& fileName)
    {
        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        if (file.is_open())
        {
            file.read(static_cast<char*>(buffer->getHostBuffer()), buffer->getSize());
            file.close();
        }
        else
        {
            std::stringstream msg;
            msg << "Cannot open file " << fileName << "!";
            throw std::invalid_argument(msg.str());
        }
    }

    void fill()
    {
        switch (dataType)
        {
        case nvinfer1::DataType::kBOOL:
        {
            fillBuffer<bool>(buffer->getHostBuffer(), volume, 0, 1);
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            fillBuffer<int32_t>(buffer->getHostBuffer(), volume, -128, 127);
            break;
        }
        case nvinfer1::DataType::kINT8:
        {
            fillBuffer<int8_t>(buffer->getHostBuffer(), volume, -128, 127);
            break;
        }
        case nvinfer1::DataType::kFLOAT:
        {
            fillBuffer<float>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            fillBuffer<__half>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
            break;
        }
        }
    }

    void dump(std::ostream& os, Dims dims, Dims strides, int32_t vectorDim, int32_t spv,
        const std::string separator = " ") const
    {
        switch (dataType)
        {
        case nvinfer1::DataType::kBOOL:
        {
            dumpBuffer<bool>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            dumpBuffer<int32_t>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kINT8:
        {
            dumpBuffer<int8_t>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kFLOAT:
        {
            dumpBuffer<float>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            dumpBuffer<__half>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        }
    }
};

class Bindings
{
public:
    Bindings() = delete;
    explicit Bindings(bool useManaged)
        : mUseManaged(useManaged)
    {
    }

    void addBinding(int b, const std::string& name, bool isInput, int64_t volume, nvinfer1::DataType dataType,
        const std::string& fileName = "")
    {
        while (mBindings.size() <= static_cast<size_t>(b))
        {
            mBindings.emplace_back();
            mDevicePointers.emplace_back();
        }
        mNames[name] = b;
        if (mBindings[b].buffer == nullptr)
        {
            if (mUseManaged)
            {
                mBindings[b].buffer.reset(new UnifiedMirroredBuffer);
            }
            else
            {
                mBindings[b].buffer.reset(new DiscreteMirroredBuffer);
            }
        }
        mBindings[b].isInput = isInput;
        // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
        // even for empty tensors, so allocate a dummy byte.
        if (volume == 0)
        {
            mBindings[b].buffer->allocate(1);
        }
        else
        {
            mBindings[b].buffer->allocate(static_cast<size_t>(volume) * static_cast<size_t>(dataTypeSize(dataType)));
        }
        mBindings[b].volume = volume;
        mBindings[b].dataType = dataType;
        mDevicePointers[b] = mBindings[b].buffer->getDeviceBuffer();
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

    void** getDeviceBuffers()
    {
        return mDevicePointers.data();
    }

    void transferInputToDevice(TrtCudaStream& stream)
    {
        for (auto& b : mNames)
        {
            if (mBindings[b.second].isInput)
            {
                mBindings[b.second].buffer->hostToDevice(stream);
            }
        }
    }

    void transferOutputToHost(TrtCudaStream& stream)
    {
        for (auto& b : mNames)
        {
            if (!mBindings[b.second].isInput)
            {
                mBindings[b.second].buffer->deviceToHost(stream);
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
        // Do not add a newline terminator, because the caller may be outputting a JSON string.
        os << dims;
    }

    void dumpBindingValues(const nvinfer1::IExecutionContext& context, int binding, std::ostream& os,
        const std::string& separator = " ", int32_t batch = 1) const
    {
        Dims dims = context.getBindingDimensions(binding);
        Dims strides = context.getStrides(binding);
        int32_t vectorDim = context.getEngine().getBindingVectorizedDim(binding);
        const int32_t spv = context.getEngine().getBindingComponentsPerElement(binding);

        if (context.getEngine().hasImplicitBatchDimension())
        {
            auto insertN = [](Dims& d, int32_t bs) {
                const int32_t nbDims = d.nbDims;
                ASSERT(nbDims < Dims::MAX_DIMS);
                std::copy_backward(&d.d[0], &d.d[nbDims], &d.d[nbDims + 1]);
                d.d[0] = bs;
                d.nbDims = nbDims + 1;
            };
            int32_t batchStride = 0;
            for (int32_t i = 0; i < strides.nbDims; ++i)
            {
                if (strides.d[i] * dims.d[i] > batchStride)
                {
                    batchStride = strides.d[i] * dims.d[i];
                }
            }
            insertN(dims, batch);
            insertN(strides, batchStride);
            vectorDim = (vectorDim == -1) ? -1 : vectorDim + 1;
        }

        mBindings[binding].dump(os, dims, strides, vectorDim, spv, separator);
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

    void dumpBindings(
        const nvinfer1::IExecutionContext& context, bool (*predicate)(const Binding& b), std::ostream& os) const
    {
        for (const auto& n : mNames)
        {
            const auto binding = n.second;
            if (predicate(mBindings[binding]))
            {
                os << n.first << ": (";
                dumpBindingDimensions(binding, context, os);
                os << ")" << std::endl;

                dumpBindingValues(context, binding, os);
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
    std::unordered_map<std::string, int32_t> mNames;
    std::vector<Binding> mBindings;
    std::vector<void*> mDevicePointers;
    bool mUseManaged{false};
};

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t)
    {
        t->destroy();
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

inline bool broadcastIOFormats(const std::vector<IOFormat>& formats, size_t nbBindings, bool isInput = true)
{
    bool broadcast = formats.size() == 1;
    bool validFormatsCount = broadcast || (formats.size() == nbBindings);
    if (!formats.empty() && !validFormatsCount)
    {
        if (isInput)
        {
            throw std::invalid_argument(
                "The number of inputIOFormats must match network's inputs or be one for broadcasting.");
        }
        else
        {
            throw std::invalid_argument(
                "The number of outputIOFormats must match network's outputs or be one for broadcasting.");
        }
    }
    return broadcast;
}

inline std::vector<char> loadTimingCacheFile(const std::string inFileName)
{
    std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);
    if (!iFile)
    {
        sample::gLogWarning << "Could not read timing cache from: " << inFileName
                            << ". A new timing cache will be generated and written." << std::endl;
        return std::vector<char>();
    }
    iFile.seekg(0, std::ifstream::end);
    size_t fsize = iFile.tellg();
    iFile.seekg(0, std::ifstream::beg);
    std::vector<char> content(fsize);
    iFile.read(content.data(), fsize);
    iFile.close();
    sample::gLogInfo << "Loaded " << fsize << " bytes of timing cache from " << inFileName << std::endl;
    return content;
}

inline void saveTimingCacheFile(const std::string outFileName, const IHostMemory* blob)
{
    std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);
    if (!oFile)
    {
        sample::gLogWarning << "Could not write timing cache to: " << outFileName << std::endl;
        return;
    }
    oFile.write((char*) blob->data(), blob->size());
    oFile.close();
    sample::gLogInfo << "Saved " << blob->size() << " bytes of timing cache to " << outFileName << std::endl;
}

inline int32_t getCudaDriverVersion()
{
    int32_t version{-1};
    cudaCheck(cudaDriverGetVersion(&version));
    return version;
}

inline int32_t getCudaRuntimeVersion()
{
    int32_t version{-1};
    cudaCheck(cudaRuntimeGetVersion(&version));
    return version;
}

} // namespace sample

#endif // TRT_SAMPLE_UTILS_H
