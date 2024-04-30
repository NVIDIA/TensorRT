/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvInfer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

using namespace nvinfer1;

static void caughtError(std::exception const& e)
{
    std::cout << e.what() << std::endl;
}

// Write values into buffer
template <typename T>
void write(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(char const*& buffer)
{
    T val{};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

#define ASSERT(condition)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cout << "Assertion failure: " << #condition << std::endl;                                             \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

template <typename Dtype>
struct CudaBind
{
    size_t mSize;
    Dtype* mPtr;

    CudaBind(size_t size)
    {
        mSize = size;
        ASSERT(!cudaMalloc((void**) &mPtr, sizeof(Dtype) * mSize));
    }

    ~CudaBind()
    {
        if (mPtr != nullptr)
        {
            ASSERT(!cudaFree(mPtr));
            mPtr = nullptr;
        }
    }
};

static int64_t volume(Dims const& dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

template <typename T>
__global__ void circPadKernel(
    T const* x, int32_t const* allPads, int32_t const* origDims, T* y, int32_t const* yShape, int32_t yLen)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = index; i < yLen; i += stride)
    {
        int32_t i3 = i % yShape[3];
        int32_t i2 = (i / yShape[3]) % yShape[2];
        int32_t i1 = (i / yShape[3] / yShape[2]) % yShape[1];
        int32_t i0 = i / yShape[3] / yShape[2] / yShape[1];

        int32_t j0 = (i0 - allPads[0] + origDims[0]) % origDims[0];
        int32_t j1 = (i1 - allPads[2] + origDims[1]) % origDims[1];
        int32_t j2 = (i2 - allPads[4] + origDims[2]) % origDims[2];
        int32_t j3 = (i3 - allPads[6] + origDims[3]) % origDims[3];

        y[i] = x[origDims[3] * origDims[2] * origDims[1] * j0 + origDims[3] * origDims[2] * j1 + origDims[3] * j2 + j3];
    }
}

class CircPadPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    CircPadPlugin() = default;

    CircPadPlugin(std::vector<int32_t> pads)
        : mPads(pads)
    {
    }

    CircPadPlugin(CircPadPlugin const& p) = default;

    CircPadPlugin(void const* serialData, size_t length)
    {
        ASSERT(serialData != nullptr);

        char const* d = static_cast<char const*>(serialData);
        char const* a = d;

        int32_t padsSize = read<int32_t>(d);
        mPads.resize(padsSize);
        for (int i = 0; i < padsSize; ++i)
        {
            mPads[i] = read<int32_t>(d);
        }

        ASSERT(d == a + length);
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
    {
        PluginTensorDesc const& desc = inOut[pos];
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }

        // first input should be float16 or float32
        if (pos == 0)
        {
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
        }

        // output should have the same type as the input
        if (pos == 1)
        {
            return (inOut[pos].type == inOut[0].type);
        }

        return false;
    }

    void configureWithFormat(nvinfer1::Dims const*, int32_t, nvinfer1::Dims const*, int32_t, nvinfer1::DataType type,
        nvinfer1::PluginFormat floatFormat, int32_t) noexcept override
    {
    }

    int32_t initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override
    {
        mAllPadsPtr.reset();
        mOrigDimsPtr.reset();
        mOutDimsPtr.reset();
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        auto inpDType = inputDesc[0].type;

        int32_t const blockSize = 256;
        int32_t const numBlocks = (volume(outputDesc[0].dims) + blockSize - 1) / blockSize;

        ASSERT(inpDType == DataType::kFLOAT || inpDType == DataType::kHALF);

        if (inpDType == DataType::kFLOAT)
        {
            circPadKernel<float><<<numBlocks, blockSize, 0, stream>>>(static_cast<float const*>(inputs[0]),
                mAllPadsPtr->mPtr, mOrigDimsPtr->mPtr, static_cast<float*>(outputs[0]), mOutDimsPtr->mPtr,
                volume(outputDesc[0].dims));
        }
        else if (inpDType == DataType::kHALF)
        {
            circPadKernel<half><<<numBlocks, blockSize, 0, stream>>>(static_cast<half const*>(inputs[0]),
                mAllPadsPtr->mPtr, mOrigDimsPtr->mPtr, static_cast<half*>(outputs[0]), mOutDimsPtr->mPtr,
                volume(outputDesc[0].dims));
        }
        return 0;
    }

    size_t getSerializationSize() const noexcept override
    {
        return (mPads.size() + 1) * sizeof(int32_t);
    }

    void serialize(void* buffer) const noexcept override
    {
        ASSERT(buffer != nullptr);
        char* d = static_cast<char*>(buffer);
        char* a = d;
        write(d, static_cast<int32_t>(mPads.size()));
        for (int i = 0; i < mPads.size(); ++i)
        {
            write(d, mPads[i]);
        }
        ASSERT(d == a + getSerializationSize());
    }

    char const* getPluginType() const noexcept override
    {
        return "CircPadPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        return new CircPadPlugin(*this);
    }

    void destroy() noexcept override
    {
        delete this;
    }

    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    DataType getOutputDataType(int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
    {
        return inputTypes[0];
    }

    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
    {
        nvinfer1::DimsExprs outDims{inputs[0]};
        int32_t nbOutDims = inputs[0].nbDims;

        for (int32_t i = 0; i < mPads.size() / 2; ++i)
        {
            outDims.d[nbOutDims - i - 1] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                *inputs[0].d[nbOutDims - i - 1], *exprBuilder.constant(mPads[i * 2] + mPads[i * 2 + 1]));
        }

        return outDims;
    }

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept
    {
        mN = in[0].desc.dims.nbDims;

        std::vector<int32_t> allPads(mN * 2);
        std::vector<int32_t> origDims(mN);
        std::vector<int32_t> outDims(mN);

        for (int32_t i = 0; i < mN; ++i)
        {
            origDims[i] = in[0].desc.dims.d[i];
            outDims[i] = in[0].desc.dims.d[i];
        }

        for (int32_t i = 0; i < mPads.size() / 2; ++i)
        {
            outDims[mN - i - 1] += mPads[i * 2] + mPads[i * 2 + 1];
            allPads[mN * 2 - 2 * i - 2] = mPads[i * 2];
            allPads[mN * 2 - 2 * i - 1] = mPads[i * 2 + 1];
        }

        mAllPadsPtr = std::make_shared<CudaBind<int32_t>>(mN * 2);
        mOrigDimsPtr = std::make_shared<CudaBind<int32_t>>(mN);
        mOutDimsPtr = std::make_shared<CudaBind<int32_t>>(mN);

        ASSERT(
            !cudaMemcpy(mAllPadsPtr->mPtr, &allPads.front(), allPads.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        ASSERT(!cudaMemcpy(
            mOrigDimsPtr->mPtr, &origDims.front(), origDims.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        ASSERT(
            !cudaMemcpy(mOutDimsPtr->mPtr, &outDims.front(), outDims.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept
    {
        return 0;
    }

private:
    std::vector<int32_t> mPads{};
    int32_t mN{};
    std::shared_ptr<CudaBind<int32_t>> mAllPadsPtr{};
    std::shared_ptr<CudaBind<int32_t>> mOrigDimsPtr{};
    std::shared_ptr<CudaBind<int32_t>> mOutDimsPtr{};
    std::string mNamespace;
};

class CircPadPluginCreator : public nvinfer1::IPluginCreator
{
public:
    CircPadPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("pads", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const* getPluginName() const noexcept
    {
        return "CircPadPlugin";
    }

    char const* getPluginVersion() const noexcept
    {
        return "1";
    }

    PluginFieldCollection const* getFieldNames() noexcept
    {
        return &mFC;
    }

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
    {
        try
        {
            std::vector<int32_t> pads;

            for (int32_t i = 0; i < fc->nbFields; i++)
            {
                std::string field_name(fc->fields[i].name);
                if (field_name.compare("pads") == 0)
                {
                    pads.resize(fc->fields[i].length);
                    auto const* padsPtr = static_cast<int32_t const*>(fc->fields[i].data);
                    std::copy_n(padsPtr, fc->fields[i].length, pads.data());
                }
            }

            return new CircPadPlugin(pads);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
    {
        try
        {
            return new CircPadPlugin(serialData, serialLength);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    void setPluginNamespace(char const* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
    }

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(CircPadPluginCreator);
