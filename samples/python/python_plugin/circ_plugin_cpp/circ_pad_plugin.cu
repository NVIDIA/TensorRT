/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string_view>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

using namespace nvinfer1;
using namespace std::string_view_literals;

static void caughtError(std::exception const& e)
{
    std::cout << e.what() << std::endl;
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

class CircPadPlugin : public IPluginV3,
                      public IPluginV3OneCore,
                      public IPluginV3OneBuild,
                      public IPluginV3OneRuntime
{
public:
    CircPadPlugin() = default;

    CircPadPlugin(std::vector<int32_t> pads)
        : mPads(std::move(pads))
    {
    }

    CircPadPlugin(CircPadPlugin const& p) = default;

    ~CircPadPlugin() override = default;

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        PluginTensorDesc const& desc = inOut[pos].desc;
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }

        // first input should be float16 or float32
        if (pos == 0)
        {
            return (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF);
        }

        // output should have the same type as the input
        if (pos == 1)
        {
            return (desc.type == inOut[0].desc.type);
        }

        return false;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
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

    char const* getPluginName() const noexcept override
    {
        return "CircPadPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }

    IPluginV3* clone() noexcept override
    {
        try
        {
            auto plugin = std::make_unique<CircPadPlugin>(*this);
            // Build-time clones do not need GPU memory. Clear shared_ptrs so the
            // clone does not share GPU allocations with the source.
            plugin->mAllPadsPtr.reset();
            plugin->mOrigDimsPtr.reset();
            plugin->mOutDimsPtr.reset();
            return plugin.release();
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

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        outputs[0] = inputs[0];
        int32_t nbOutDims = inputs[0].nbDims;

        for (int32_t i = 0; i < static_cast<int32_t>(mPads.size()) / 2; ++i)
        {
            outputs[0].d[nbOutDims - i - 1] = exprBuilder.operation(DimensionOperation::kSUM,
                *inputs[0].d[nbOutDims - i - 1], *exprBuilder.constant(mPads[i * 2] + mPads[i * 2 + 1]));
        }

        return 0;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        return 0;
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        mN = in[0].dims.nbDims;

        std::vector<int32_t> allPads(mN * 2);
        std::vector<int32_t> origDims(mN);
        std::vector<int32_t> outDims(mN);

        for (int32_t i = 0; i < mN; ++i)
        {
            origDims[i] = in[0].dims.d[i];
            outDims[i] = in[0].dims.d[i];
        }

        for (int32_t i = 0; i < static_cast<int32_t>(mPads.size()) / 2; ++i)
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

        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("pads", mPads.data(), PluginFieldType::kINT32, mPads.size());
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }

private:
    std::vector<int32_t> mPads{};
    int32_t mN{};
    std::shared_ptr<CudaBind<int32_t>> mAllPadsPtr{};
    std::shared_ptr<CudaBind<int32_t>> mOrigDimsPtr{};
    std::shared_ptr<CudaBind<int32_t>> mOutDimsPtr{};
    std::string mNamespace;

    std::vector<PluginField> mDataToSerialize;
    PluginFieldCollection mFCToSerialize;
};

class CircPadPluginCreator : public IPluginCreatorV3One
{
public:
    CircPadPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("pads", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const* getPluginName() const noexcept override
    {
        return "CircPadPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override
    {
        try
        {
            std::vector<int32_t> pads;

            for (int32_t i = 0; i < fc->nbFields; i++)
            {
                if (fc->fields[i].name == "pads"sv)
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

    void setPluginNamespace(char const* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(CircPadPluginCreator);
