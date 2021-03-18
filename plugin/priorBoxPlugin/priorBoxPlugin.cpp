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

#include "priorBoxPlugin.h"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
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
PriorBox::PriorBox(PriorBoxParameters param, int H, int W)
    : mParam(param)
    , mH(H)
    , mW(W)
{
    setupDeviceMemory();
}

void PriorBox::setupDeviceMemory()
{
    auto copyToDevice = [](const void* hostData, size_t count) -> Weights {
        void* deviceData = nullptr;
        CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
        CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
        return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
    };

    // minSize is required and needs to be non-negative
    ASSERT(mParam.numMinSize > 0 && mParam.minSize != nullptr);
    for (int i = 0; i < mParam.numMinSize; ++i)
    {
        ASSERT(mParam.minSize[i] > 0 && "minSize must be positive");
    }
    minSize = copyToDevice(mParam.minSize, mParam.numMinSize);

    ASSERT(mParam.numAspectRatios >= 0 && mParam.aspectRatios != nullptr);
    // Aspect ratio of 1.0 is built in.
    std::vector<float> tmpAR(1, 1);
    for (int i = 0; i < mParam.numAspectRatios; ++i)
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
        ASSERT(mParam.numMinSize == mParam.numMaxSize && mParam.maxSize != nullptr);
        for (int i = 0; i < mParam.numMaxSize; ++i)
        {
            // maxSize should be greater than minSize
            ASSERT(mParam.maxSize[i] > mParam.minSize[i] && "maxSize must be greater than minSize");
            mNumPriors++;
        }
        maxSize = copyToDevice(mParam.maxSize, mParam.numMaxSize);
    }
}

PriorBox::PriorBox(const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data), *a = d;
    mParam = read<PriorBoxParameters>(d);

    auto readArray = [&d](const int size, float*& array) {
        if (size > 0)
        {
            array = new float[size];
            for (int i = 0; i < size; i++)
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

    ASSERT(d == a + length);

    setupDeviceMemory();
}

// Returns the number of output from the plugin layer
int PriorBox::getNbOutputs() const
{
    // Number of outputs from the plugin layer is 1
    return 1;
}

// Computes and returns the output dimensions
Dims PriorBox::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 2);
    // Only one output from the plugin layer
    ASSERT(index == 0);
    // Particularity of the PriorBox layer: no batchSize dimension needed
    mH = inputs[0].d[1], mW = inputs[0].d[2];
    // workaround for TRT
    // The first channel is for prior box coordinates.
    // The second channel is for prior box scaling factors, which is simply a copy of the variance provided.
    return DimsCHW(2, mH * mW * mNumPriors * 4, 1);
}

int PriorBox::initialize()
{
    return STATUS_SUCCESS;
}

size_t PriorBox::getWorkspaceSize(int /*maxBatchSize*/) const
{
    return 0;
}

int PriorBox::enqueue(
    int /*batchSize*/, const void* const* /*inputs*/, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    void* outputData = outputs[0];
    pluginStatus_t status = priorBoxInference(stream, mParam, mH, mW, mNumPriors, aspectRatios.count, minSize.values,
        maxSize.values, aspectRatios.values, outputData);
    ASSERT(status == STATUS_SUCCESS);

    return 0;
}

// Returns the size of serialized parameters
size_t PriorBox::getSerializationSize() const
{
    // PriorBoxParameters, minSize, maxSize, aspectRatios, mH, mW - the construct parameters
    return sizeof(PriorBoxParameters) + sizeof(float) * (mParam.numMinSize + mParam.numMaxSize + mParam.numAspectRatios)
        + sizeof(int) * 2;
}

void PriorBox::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mParam);

    auto writeArray = [&d](const int size, const float* array) {
        for (int i = 0; i < size; i++)
        {
            write(d, array[i]);
        }
    };
    writeArray(mParam.numMinSize, mParam.minSize);
    writeArray(mParam.numMaxSize, mParam.maxSize);
    writeArray(mParam.numAspectRatios, mParam.aspectRatios);

    write(d, mH);
    write(d, mW);

    ASSERT(d == a + getSerializationSize());
}

bool PriorBox::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* PriorBox::getPluginType() const
{
    return PRIOR_BOX_PLUGIN_NAME;
}

const char* PriorBox::getPluginVersion() const
{
    return PRIOR_BOX_PLUGIN_VERSION;
}

void PriorBox::destroy()
{
    CUASSERT(cudaFree(const_cast<void*>(minSize.values)));
    if (mParam.numMaxSize > 0)
    {
        CUASSERT(cudaFree(const_cast<void*>(maxSize.values)));
    }
    if (mParam.numAspectRatios > 0)
    {
        CUASSERT(cudaFree(const_cast<void*>(aspectRatios.values)));
    }
    delete[] mParam.minSize;
    delete[] mParam.maxSize;
    delete[] mParam.aspectRatios;

    delete this;
}

IPluginV2Ext* PriorBox::clone() const
{
    // each obj should manage its copy of param
    PriorBoxParameters params = mParam;
    auto copyParamData = [](float*& dest, const float* src, const size_t size) {
        if (size > 0)
        {
            dest = new float[size];
            std::copy_n(src, size, dest);
        }
        else
        {
            ASSERT(dest == nullptr);
        }
    };
    copyParamData(params.minSize, mParam.minSize, mParam.numMinSize);
    copyParamData(params.maxSize, mParam.maxSize, mParam.numMaxSize);
    copyParamData(params.aspectRatios, mParam.aspectRatios, mParam.numAspectRatios);

    PriorBox* obj = new PriorBox(params, mH, mW);
    obj->setPluginNamespace(mPluginNamespace.c_str());
    return obj;
}

// Set plugin namespace
void PriorBox::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* PriorBox::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType PriorBox::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Two outputs
    ASSERT(index == 0 || index == 1);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool PriorBox::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool PriorBox::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void PriorBox::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
    ASSERT(inputDims[0].nbDims == 3);
    ASSERT(inputDims[1].nbDims == 3);
    ASSERT(outputDims[0].nbDims == 3);
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
void PriorBox::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) {}

// Detach the plugin object from its execution context.
void PriorBox::detachFromContext() {}

PriorBoxPluginCreator::PriorBoxPluginCreator()
{
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

const char* PriorBoxPluginCreator::getPluginName() const
{
    return PRIOR_BOX_PLUGIN_NAME;
}

const char* PriorBoxPluginCreator::getPluginVersion() const
{
    return PRIOR_BOX_PLUGIN_VERSION;
}

const PluginFieldCollection* PriorBoxPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* PriorBoxPluginCreator::createPlugin(const char* /*name*/, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;

    PriorBoxParameters params;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "minSize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            params.minSize = new float[size];
            const auto* minS = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.minSize[j] = *minS;
                minS++;
            }
            params.numMinSize = size;
        }
        else if (!strcmp(attrName, "maxSize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            params.numMaxSize = size;
            params.maxSize = nullptr;
            if (size > 0)
            {
                params.maxSize = new float[size];
                const auto* maxS = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    params.maxSize[j] = *maxS;
                    maxS++;
                }
            }
        }
        else if (!strcmp(attrName, "aspectRatios"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            params.numAspectRatios = size;
            params.aspectRatios = nullptr;
            if (size > 0)
            {
                params.aspectRatios = new float[size];
                const auto* aR = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    params.aspectRatios[j] = *aR;
                    aR++;
                }
            }
        }
        else if (!strcmp(attrName, "variance"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            const auto* lVar = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.variance[j] = (*lVar);
                lVar++;
            }
        }
        else if (!strcmp(attrName, "flip"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.flip = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "clip"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.clip = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "imgH"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.imgH = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "imgW"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.imgW = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "stepH"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.stepH = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "stepW"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.stepW = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "offset"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.offset = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
    }
    PriorBox* obj = new PriorBox(params);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* PriorBoxPluginCreator::deserializePlugin(
    const char* /*name*/, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call PriorBox::destroy()
    PriorBox* obj = new PriorBox(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
