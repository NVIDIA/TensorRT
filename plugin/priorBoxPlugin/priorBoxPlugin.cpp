/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "priorBoxPlugin.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::PriorBox;
using nvinfer1::plugin::PriorBoxPluginCreator;

namespace
{
char const* const kPRIOR_BOX_PLUGIN_VERSION{"1"};
char const* const kPRIOR_BOX_PLUGIN_NAME{"PriorBox_TRT"};
} // namespace

// Constructor
PriorBox::PriorBox(PriorBoxParameters param, int32_t H, int32_t W)
    : mParam(param)
    , mH(H)
    , mW(W)
{
    // Each object should manage its copy of param.
    auto copyParamData = [](float*& dstPtr, std::vector<float>& dstVec, float const* src, int32_t size) {
        PLUGIN_VALIDATE(size >= 0);
        PLUGIN_VALIDATE(src != nullptr);

        dstVec.resize(size);
        dstPtr = dstVec.data();
        std::copy_n(src, size, dstPtr);
    };
    copyParamData(mParam.minSize, mMinSizeCPU, param.minSize, param.numMinSize);
    copyParamData(mParam.maxSize, mMaxSizeCPU, param.maxSize, param.numMaxSize);
    copyParamData(mParam.aspectRatios, mAspectRatiosCPU, param.aspectRatios, param.numAspectRatios);

    setupDeviceMemory();
}

void PriorBox::setupDeviceMemory() noexcept
{
    auto copyToDevice = [](void const* hostData, int32_t count) -> Weights {
        PLUGIN_VALIDATE(count >= 0);
        void* deviceData = nullptr;
        PLUGIN_CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
        PLUGIN_CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
        return Weights{DataType::kFLOAT, deviceData, static_cast<int64_t>(count)};
    };

    // minSize is required and needs to be positive.
    PLUGIN_VALIDATE(mParam.numMinSize > 0);
    PLUGIN_VALIDATE(mParam.minSize != nullptr);
    for (int32_t i = 0; i < mParam.numMinSize; ++i)
    {
        PLUGIN_VALIDATE(mParam.minSize[i] > 0.F, "minSize must be positive");
    }
    mMinSizeGPU = copyToDevice(mParam.minSize, mParam.numMinSize);

    PLUGIN_VALIDATE(mParam.numAspectRatios >= 0);
    PLUGIN_VALIDATE(mParam.aspectRatios != nullptr);
    // Aspect ratio of 1.0 is built in.
    std::vector<float> tmpAR(1, 1);
    for (int32_t i = 0; i < mParam.numAspectRatios; ++i)
    {
        float aspectRatio = mParam.aspectRatios[i];
        bool alreadyExist = false;
        // Prevent duplicated aspect ratios from input
        for (size_t j = 0; j < tmpAR.size(); ++j)
        {
            if (std::fabs(aspectRatio - tmpAR[j]) < 1e-6)
            {
                alreadyExist = true;
                break;
            }
        }
        if (!alreadyExist)
        {
            PLUGIN_VALIDATE(aspectRatio > 0.F);
            tmpAR.push_back(aspectRatio);
            if (mParam.flip)
            {
                tmpAR.push_back(1.0F / aspectRatio);
            }
        }
    }
    //
    // mAspectRatiosGPU is of type nvinfer1::Weights.
    // https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_weights.html
    // mAspectRatiosGPU.count is different to mParam.numAspectRatios.
    //
    mAspectRatiosGPU = copyToDevice(&tmpAR[0], tmpAR.size());

    // Number of prior boxes per grid cell on the feature map
    // tmpAR already included an aspect ratio of 1.0
    mNumPriors = tmpAR.size() * mParam.numMinSize;

    //
    // If we have maxSizes, as long as all the maxSizes meets assertion requirement, we add one bounding box per maxSize
    // The final number of prior boxes per grid cell on feature map
    // mNumPriors =
    // tmpAR.size() * mParam.numMinSize If numMaxSize == 0
    // (tmpAR.size() + 1) * mParam.numMinSize If mParam.numMinSize == mParam.numMaxSize
    //
    if (mParam.numMaxSize > 0)
    {
        PLUGIN_VALIDATE(mParam.numMinSize == mParam.numMaxSize);
        PLUGIN_VALIDATE(mParam.maxSize != nullptr);
        PLUGIN_VALIDATE(mParam.minSize != nullptr);
        for (int32_t i = 0; i < mParam.numMaxSize; ++i)
        {
            // maxSize should be greater than minSize
            // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
            PLUGIN_VALIDATE(mParam.maxSize[i] > mParam.minSize[i], "maxSize must be greater than minSize");
            mNumPriors++;
        }
        mMaxSizeGPU = copyToDevice(mParam.maxSize, mParam.numMaxSize);
    }
}

PriorBox::PriorBox(void const* data, size_t length)
{
    deserialize(static_cast<uint8_t const*>(data), length);
}

void PriorBox::deserialize(uint8_t const* data, size_t length)
{
    auto const* d{data};
    mParam = read<PriorBoxParameters>(d);

    auto readArray = [&d](int32_t size, std::vector<float>& dstVec, float*& dstPtr) {
        PLUGIN_VALIDATE(size >= 0);
        dstVec.resize(size);
        for (int32_t i = 0; i < size; i++)
        {
            dstVec[i] = read<float>(d);
        }
        dstPtr = dstVec.data();
    };
    readArray(mParam.numMinSize, mMinSizeCPU, mParam.minSize);
    readArray(mParam.numMaxSize, mMaxSizeCPU, mParam.maxSize);
    readArray(mParam.numAspectRatios, mAspectRatiosCPU, mParam.aspectRatios);

    mH = read<int32_t>(d);
    mW = read<int32_t>(d);

    PLUGIN_VALIDATE(d == data + length);

    setupDeviceMemory();
}

// Returns the number of output from the plugin layer
int32_t PriorBox::getNbOutputs() const noexcept
{
    // Number of outputs from the plugin layer is 1
    return 1;
}

// Computes and returns the output dimensions
Dims PriorBox::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_VALIDATE(nbInputDims == 2);
    // Only one output from the plugin layer
    PLUGIN_VALIDATE(index == 0);
    // Particularity of the PriorBox layer: no batchSize dimension needed
    mH = inputs[0].d[1];
    mW = inputs[0].d[2];
    // workaround for TRT
    // The first channel is for prior box coordinates.
    // The second channel is for prior box scaling factors, which is simply a copy of the variance provided.
    return Dims3(2, mH * mW * mNumPriors * 4, 1);
}

int32_t PriorBox::initialize() noexcept
{
    return STATUS_SUCCESS;
}

size_t PriorBox::getWorkspaceSize(int32_t /*maxBatchSize*/) const noexcept
{
    return 0;
}

int32_t PriorBox::enqueue(int32_t /*batchSize*/, void const* const* /*inputs*/, void* const* outputs,
    void* /*workspace*/, cudaStream_t stream) noexcept
{
    void* outputData = outputs[0];
    pluginStatus_t status = priorBoxInference(stream, mParam, mH, mW, mNumPriors, mAspectRatiosGPU.count,
        mMinSizeGPU.values, mMaxSizeGPU.values, mAspectRatiosGPU.values, outputData);

    return status;
}

// Returns the size of serialized parameters
size_t PriorBox::getSerializationSize() const noexcept
{
    // PriorBoxParameters, minSize, maxSize, aspectRatios, mH, mW - the construct parameters
    return sizeof(PriorBoxParameters) + sizeof(float) * (mParam.numMinSize + mParam.numMaxSize + mParam.numAspectRatios)
        + sizeof(int32_t) * 2;
}

void PriorBox::serialize(void* buffer) const noexcept
{
    uint8_t* d = static_cast<uint8_t*>(buffer);
    uint8_t* a = d;
    write(d, mParam);

    auto writeArray = [&d](int32_t const size, float const* srcPtr, std::vector<float> const& srcVec) {
        // srcVec is only used here to check that the size and srcPtr are correct.
        PLUGIN_VALIDATE(srcVec.data() == srcPtr);
        PLUGIN_VALIDATE(srcVec.size() == static_cast<size_t>(size));
        for (int32_t i = 0; i < size; i++)
        {
            write(d, srcPtr[i]);
        }
    };
    writeArray(mParam.numMinSize, mParam.minSize, mMinSizeCPU);
    writeArray(mParam.numMaxSize, mParam.maxSize, mMaxSizeCPU);
    writeArray(mParam.numAspectRatios, mParam.aspectRatios, mAspectRatiosCPU);

    write(d, mH);
    write(d, mW);

    PLUGIN_VALIDATE(d == a + getSerializationSize());
}

bool PriorBox::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

char const* PriorBox::getPluginType() const noexcept
{
    return kPRIOR_BOX_PLUGIN_NAME;
}

char const* PriorBox::getPluginVersion() const noexcept
{
    return kPRIOR_BOX_PLUGIN_VERSION;
}

void PriorBox::destroy() noexcept
{
    PLUGIN_CUASSERT(cudaFree(const_cast<void*>(mMinSizeGPU.values)));
    if (mParam.numMaxSize > 0)
    {
        PLUGIN_CUASSERT(cudaFree(const_cast<void*>(mMaxSizeGPU.values)));
    }
    if (mParam.numAspectRatios > 0)
    {
        PLUGIN_CUASSERT(cudaFree(const_cast<void*>(mAspectRatiosGPU.values)));
    }

    delete this;
}

IPluginV2Ext* PriorBox::clone() const noexcept
{
    try
    {
        PriorBox* obj = new PriorBox(mParam, mH, mW);
        obj->setPluginNamespace(mPluginNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// Set plugin namespace
void PriorBox::setPluginNamespace(char const* pluginNamespace) noexcept
{
    PLUGIN_VALIDATE(pluginNamespace != nullptr);
    mPluginNamespace = pluginNamespace;
}

char const* PriorBox::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType PriorBox::getOutputDataType(
    int32_t index, nvinfer1::DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept
{
    // Two outputs
    PLUGIN_VALIDATE(index == 0 || index == 1);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool PriorBox::isOutputBroadcastAcrossBatch(
    int32_t /*outputIndex*/, bool const* /*inputIsBroadcasted*/, int32_t /*nbInputs*/) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool PriorBox::canBroadcastInputAcrossBatch(int32_t /*inputIndex*/) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void PriorBox::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* /*outputTypes*/, bool const* /*inputIsBroadcast*/,
    bool const* /*outputIsBroadcast*/, PluginFormat floatFormat, int32_t /*maxBatchSize*/) noexcept
{
    PLUGIN_VALIDATE(nbInputs == 2);
    PLUGIN_VALIDATE(nbOutputs == 1);
    PLUGIN_VALIDATE(inputDims && outputDims && inputTypes);
    PLUGIN_VALIDATE(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
    PLUGIN_VALIDATE(inputDims[0].nbDims == 3);
    PLUGIN_VALIDATE(inputDims[1].nbDims == 3);
    PLUGIN_VALIDATE(outputDims[0].nbDims == 3);
    mH = inputDims[0].d[1];
    mW = inputDims[0].d[2];
    // Prepare for the inference function.
    if (mParam.imgH == 0 || mParam.imgW == 0)
    {
        mParam.imgH = inputDims[1].d[1];
        mParam.imgW = inputDims[1].d[2];
    }
    if (mParam.stepH == 0 || mParam.stepW == 0)
    {
        mParam.stepH = static_cast<float>(mParam.imgH) / mH;
        mParam.stepW = static_cast<float>(mParam.imgW) / mW;
    }
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void PriorBox::attachToContext(
    cudnnContext* /*cudnnContext*/, cublasContext* /*cublasContext*/, IGpuAllocator* /*gpuAllocator*/) noexcept
{
}

// Detach the plugin object from its execution context.
void PriorBox::detachFromContext() noexcept {}

PriorBoxPluginCreator::PriorBoxPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("minSize", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("maxSize", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("aspectRatios", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("flip", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clip", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("variance", nullptr, PluginFieldType::kFLOAT32, 4));
    mPluginAttributes.emplace_back(PluginField("imgH", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("imgW", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stepH", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("stepW", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("offset", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

PriorBoxPluginCreator::~PriorBoxPluginCreator()
{
    // Free allocated memory (if any) here
}

char const* PriorBoxPluginCreator::getPluginName() const noexcept
{
    return kPRIOR_BOX_PLUGIN_NAME;
}

char const* PriorBoxPluginCreator::getPluginVersion() const noexcept
{
    return kPRIOR_BOX_PLUGIN_VERSION;
}

PluginFieldCollection const* PriorBoxPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* PriorBoxPluginCreator::createPlugin(char const* /*name*/, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PluginField const* fields = fc->fields;

        PriorBoxParameters params;
        std::vector<float> minSize;
        std::vector<float> maxSize;
        std::vector<float> aspectRatios;
        for (auto i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "minSize"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                int32_t const size = fields[i].length;
                params.numMinSize = size;
                if (size > 0)
                {
                    minSize.resize(size);
                    params.minSize = minSize.data();
                    auto const* minS = static_cast<float const*>(fields[i].data);
                    std::copy_n(minS, size, params.minSize);
                }
                else
                {
                    params.minSize = nullptr;
                }
            }
            else if (!strcmp(attrName, "maxSize"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                int32_t const size = fields[i].length;
                params.numMaxSize = size;
                if (size > 0)
                {
                    maxSize.resize(size);
                    params.maxSize = maxSize.data();
                    auto const* maxS = static_cast<float const*>(fields[i].data);
                    std::copy_n(maxS, size, params.maxSize);
                }
                else
                {
                    params.maxSize = nullptr;
                }
            }
            else if (!strcmp(attrName, "aspectRatios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                int32_t const size = fields[i].length;
                params.numAspectRatios = size;
                if (size > 0)
                {
                    aspectRatios.resize(size);
                    params.aspectRatios = aspectRatios.data();
                    auto const* aR = static_cast<float const*>(fields[i].data);
                    std::copy_n(aR, size, params.aspectRatios);
                }
                else
                {
                    params.aspectRatios = nullptr;
                }
            }
            else if (!strcmp(attrName, "variance"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                int32_t const size = fields[i].length;
                PLUGIN_VALIDATE(size == 4);
                auto const* lVar = static_cast<float const*>(fields[i].data);
                for (auto j = 0; j < size; j++)
                {
                    params.variance[j] = (*lVar);
                    lVar++;
                }
            }
            else if (!strcmp(attrName, "flip"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.flip = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "clip"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.clip = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "imgH"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.imgH = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "imgW"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.imgW = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "stepH"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.stepH = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "stepW"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.stepW = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "offset"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.offset = *(static_cast<float const*>(fields[i].data));
            }
        }
        PriorBox* obj = new PriorBox(params);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* PriorBoxPluginCreator::deserializePlugin(
    char const* /*name*/, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call PriorBox::destroy()
        PriorBox* obj = new PriorBox(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
