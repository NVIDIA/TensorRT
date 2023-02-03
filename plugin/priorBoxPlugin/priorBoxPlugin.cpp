/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::PriorBox;
using nvinfer1::plugin::PriorBoxPluginCreator;

namespace
{
const char* PRIOR_BOX_PLUGIN_VERSION{"1"};
const char* PRIOR_BOX_PLUGIN_NAME{"PriorBox_TRT"};
} // namespace

PluginFieldCollection PriorBoxPluginCreator::mFC{};
std::vector<PluginField> PriorBoxPluginCreator::mPluginAttributes;

// Constructor
PriorBox::PriorBox(PriorBoxParameters param, int32_t H, int32_t W)
    : mParam(param)
    , mH(H)
    , mW(W)
{
    // each obj should manage its copy of param
    auto copyParamData = [](float*& dest, const float* src, const size_t size)
    {
        if (size > 0)
        {
            dest = new float[size];
            std::copy_n(src, size, dest);
        }
        else
        {
            PLUGIN_VALIDATE(dest == nullptr);
        }
    };
    copyParamData(mParam.minSize, param.minSize, param.numMinSize);
    copyParamData(mParam.maxSize, param.maxSize, param.numMaxSize);
    copyParamData(mParam.aspectRatios, param.aspectRatios, param.numAspectRatios);

    setupDeviceMemory();
}

void PriorBox::setupDeviceMemory() noexcept
{
    auto copyToDevice = [](const void* hostData, size_t count) -> Weights {
        void* deviceData = nullptr;
        PLUGIN_CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
        PLUGIN_CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
        return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
    };

    // minSize is required and needs to be positive.
    PLUGIN_ASSERT(mParam.numMinSize > 0 && mParam.minSize != nullptr);
    for (auto i = 0; i < mParam.numMinSize; ++i)
    {
        PLUGIN_ASSERT(mParam.minSize[i] > 0 && "minSize must be positive");
    }
    minSize = copyToDevice(mParam.minSize, mParam.numMinSize);

    PLUGIN_ASSERT(mParam.numAspectRatios >= 0 && mParam.aspectRatios != nullptr);
    // Aspect ratio of 1.0 is built in.
    std::vector<float> tmpAR(1, 1);
    for (auto i = 0; i < mParam.numAspectRatios; ++i)
    {
        float ar = mParam.aspectRatios[i];
        bool alreadyExist = false;
        // Prevent duplicated aspect ratios from input
        for (unsigned j = 0; j < tmpAR.size(); ++j)
        {
            if (std::fabs(ar - tmpAR[j]) < 1e-6)
            {
                alreadyExist = true;
                break;
            }
        }
        if (!alreadyExist)
        {
            tmpAR.push_back(ar);
            if (mParam.flip)
            {
                tmpAR.push_back(1.0F / ar);
            }
        }
    }
    /*
     * aspectRatios is of type nvinfer1::Weights
     * https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_weights.html
     * aspectRatios.count is different to mParam.numAspectRatios
     */
    aspectRatios = copyToDevice(&tmpAR[0], tmpAR.size());

    // Number of prior boxes per grid cell on the feature map
    // tmpAR already included an aspect ratio of 1.0
    mNumPriors = tmpAR.size() * mParam.numMinSize;
    /*
     * If we have maxSizes, as long as all the maxSizes meets assertion requirement, we add one bounding box per maxSize
     * The final number of prior boxes per grid cell on feature map
     * mNumPriors =
     * tmpAR.size() * mParam.numMinSize If numMaxSize == 0
     * (tmpAR.size() + 1) * mParam.numMinSize If mParam.numMinSize == mParam.numMaxSize
     */
    if (mParam.numMaxSize > 0)
    {
        PLUGIN_ASSERT(mParam.numMinSize == mParam.numMaxSize && mParam.maxSize != nullptr && mParam.minSize != nullptr);
        for (auto i = 0; i < mParam.numMaxSize; ++i)
        {
            // maxSize should be greater than minSize
            // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
            PLUGIN_ASSERT(mParam.maxSize[i] > mParam.minSize[i] && "maxSize must be greater than minSize");
            mNumPriors++;
        }
        maxSize = copyToDevice(mParam.maxSize, mParam.numMaxSize);
    }
}

PriorBox::PriorBox(const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data), *a = d;
    mParam = read<PriorBoxParameters>(d);

    auto readArray = [&d](const int32_t size, float*& array)
    {
        if (size > 0)
        {
            array = new float[size];
            for (auto i = 0; i < size; i++)
            {
                array[i] = read<float>(d);
            }
        }
        else
        {
            array = nullptr;
        }
    };
    readArray(mParam.numMinSize, mParam.minSize);
    readArray(mParam.numMaxSize, mParam.maxSize);
    readArray(mParam.numAspectRatios, mParam.aspectRatios);

    mH = read<int>(d);
    mW = read<int>(d);

    PLUGIN_VALIDATE(d == a + length);

    setupDeviceMemory();
}

// Returns the number of output from the plugin layer
int32_t PriorBox::getNbOutputs() const noexcept
{
    // Number of outputs from the plugin layer is 1
    return 1;
}

// Computes and returns the output dimensions
Dims PriorBox::getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 2);
    // Only one output from the plugin layer
    PLUGIN_ASSERT(index == 0);
    // Particularity of the PriorBox layer: no batchSize dimension needed
    mH = inputs[0].d[1], mW = inputs[0].d[2];
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

int32_t PriorBox::enqueue(int32_t /*batchSize*/, const void* const* /*inputs*/, void* const* outputs, void* /*workspace*/,
    cudaStream_t stream) noexcept
{
    void* outputData = outputs[0];
    pluginStatus_t status = priorBoxInference(stream, mParam, mH, mW, mNumPriors, aspectRatios.count, minSize.values,
        maxSize.values, aspectRatios.values, outputData);

    return status;
}

// Returns the size of serialized parameters
size_t PriorBox::getSerializationSize() const noexcept
{
    // PriorBoxParameters, minSize, maxSize, aspectRatios, mH, mW - the construct parameters
    return sizeof(PriorBoxParameters) + sizeof(float) * (mParam.numMinSize + mParam.numMaxSize + mParam.numAspectRatios)
        + sizeof(int) * 2;
}

void PriorBox::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mParam);

    auto writeArray = [&d](const int32_t size, const float* array) {
        for (auto i = 0; i < size; i++)
        {
            write(d, array[i]);
        }
    };
    writeArray(mParam.numMinSize, mParam.minSize);
    writeArray(mParam.numMaxSize, mParam.maxSize);
    writeArray(mParam.numAspectRatios, mParam.aspectRatios);

    write(d, mH);
    write(d, mW);

    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool PriorBox::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char* PriorBox::getPluginType() const noexcept
{
    return PRIOR_BOX_PLUGIN_NAME;
}

const char* PriorBox::getPluginVersion() const noexcept
{
    return PRIOR_BOX_PLUGIN_VERSION;
}

void PriorBox::destroy() noexcept
{
    PLUGIN_CUASSERT(cudaFree(const_cast<void*>(minSize.values)));
    if (mParam.numMaxSize > 0)
    {
        PLUGIN_CUASSERT(cudaFree(const_cast<void*>(maxSize.values)));
    }
    if (mParam.numAspectRatios > 0)
    {
        PLUGIN_CUASSERT(cudaFree(const_cast<void*>(aspectRatios.values)));
    }
    delete[] mParam.minSize;
    delete[] mParam.maxSize;
    delete[] mParam.aspectRatios;

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
void PriorBox::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* PriorBox::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType PriorBox::getOutputDataType(
    int32_t index, const nvinfer1::DataType* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept
{
    // Two outputs
    PLUGIN_ASSERT(index == 0 || index == 1);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool PriorBox::isOutputBroadcastAcrossBatch(int32_t /*outputIndex*/, const bool* /*inputIsBroadcasted*/, int32_t /*nbInputs*/) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool PriorBox::canBroadcastInputAcrossBatch(int32_t /*inputIndex*/) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void PriorBox::configurePlugin(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
    const DataType* inputTypes, const DataType* /*outputTypes*/, const bool* /*inputIsBroadcast*/,
    const bool* /*outputIsBroadcast*/, PluginFormat floatFormat, int32_t /*maxBatchSize*/) noexcept
{
    PLUGIN_ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inputDims[0].nbDims == 3);
    PLUGIN_ASSERT(inputDims[1].nbDims == 3);
    PLUGIN_ASSERT(outputDims[0].nbDims == 3);
    mH = inputDims[0].d[1];
    mW = inputDims[0].d[2];
    // prepare for the inference function
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
void PriorBox::attachToContext(cudnnContext* /*cudnnContext*/, cublasContext* /*cublasContext*/, IGpuAllocator* /*gpuAllocator*/) noexcept {}

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

const char* PriorBoxPluginCreator::getPluginName() const noexcept
{
    return PRIOR_BOX_PLUGIN_NAME;
}

const char* PriorBoxPluginCreator::getPluginVersion() const noexcept
{
    return PRIOR_BOX_PLUGIN_VERSION;
}

const PluginFieldCollection* PriorBoxPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* PriorBoxPluginCreator::createPlugin(const char* /*name*/, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;

        PriorBoxParameters params;
        std::unique_ptr<float[]> minSize;
        std::unique_ptr<float[]> maxSize;
        std::unique_ptr<float[]> aspectRatios;
        for (auto i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "minSize"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const int32_t size = fields[i].length;
                params.numMinSize = size;
                if (size > 0)
                {
                    minSize.reset(new float[size]);
                    params.minSize = minSize.get();
                    const auto* minS = static_cast<const float*>(fields[i].data);
                    for (auto j = 0; j < size; j++)
                    {
                        params.minSize[j] = *minS;
                        minS++;
                    }
                }
                else
                {
                    params.minSize = nullptr;
                }
            }
            else if (!strcmp(attrName, "maxSize"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const int32_t size = fields[i].length;
                params.numMaxSize = size;
                if (size > 0)
                {
                    maxSize.reset(new float[size]);
                    params.maxSize = maxSize.get();
                    const auto* maxS = static_cast<const float*>(fields[i].data);
                    for (auto j = 0; j < size; j++)
                    {
                        params.maxSize[j] = *maxS;
                        maxS++;
                    }
                }
                else
                {
                    params.maxSize = nullptr;
                }
            }
            else if (!strcmp(attrName, "aspectRatios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const int32_t size = fields[i].length;
                params.numAspectRatios = size;
                if (size > 0)
                {
                    aspectRatios.reset(new float[size]);
                    params.aspectRatios = aspectRatios.get();
                    const auto* aR = static_cast<const float*>(fields[i].data);
                    for (auto j = 0; j < size; j++)
                    {
                        params.aspectRatios[j] = *aR;
                        aR++;
                    }
                }
                else
                {
                    params.aspectRatios = nullptr;
                }
            }
            else if (!strcmp(attrName, "variance"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const int32_t size = fields[i].length;
                const auto* lVar = static_cast<const float*>(fields[i].data);
                for (auto j = 0; j < size; j++)
                {
                    params.variance[j] = (*lVar);
                    lVar++;
                }
            }
            else if (!strcmp(attrName, "flip"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.flip = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "clip"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.clip = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "imgH"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.imgH = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "imgW"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.imgW = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "stepH"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.stepH = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "stepW"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.stepW = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "offset"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.offset = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
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
    const char* /*name*/, const void* serialData, size_t serialLength) noexcept
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
