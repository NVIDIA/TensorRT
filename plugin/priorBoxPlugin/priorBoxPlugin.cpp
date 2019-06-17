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
PriorBox::PriorBox(PriorBoxParameters param)
    : mParam(param)
{
    // minSize is required and needs to be non-negative
    ASSERT(param.numMinSize > 0 && param.minSize != nullptr);
    for (int i = 0; i < param.numMinSize; ++i)
    {
        ASSERT(param.minSize[i] > 0 && "minSize must be positive");
    }
    minSize = copyToDevice(param.minSize, param.numMinSize);
    ASSERT(param.numAspectRatios >= 0 && param.aspectRatios != nullptr);
    // Aspect ratio of 1.0 is built in.
    std::vector<float> tmpAR(1, 1);
    for (int i = 0; i < param.numAspectRatios; ++i)
    {
        float ar = param.aspectRatios[i];
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
            if (param.flip)
            {
                tmpAR.push_back(1.0F / ar);
            }
        }
    }
    /*
     * aspectRatios is of type nvinfer1::Weights
     * https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_weights.html
     * aspectRatios.count is different to param.numAspectRatios
     */
    aspectRatios = copyToDevice(&tmpAR[0], tmpAR.size());
    // Number of prior boxes per grid cell on the feature map
    // tmpAR already included an aspect ratio of 1.0
    numPriors = tmpAR.size() * param.numMinSize;
    /*
     * If we have maxSizes, as long as all the maxSizes meets assertion requirement, we add one bounding box per maxSize
     * The final number of prior boxes per grid cell on feature map
     * numPriors =
     * tmpAR.size() * param.numMinSize If numMaxSize == 0
     * (tmpAR.size() + 1) * param.numMinSize If param.numMinSize == param.numMaxSize
     */
    if (param.numMaxSize > 0)
    {
        ASSERT(param.numMinSize == param.numMaxSize && param.maxSize != nullptr);
        for (int i = 0; i < param.numMaxSize; ++i)
        {
            // maxSize should be greater than minSize
            ASSERT(param.maxSize[i] > param.minSize[i] && "maxSize must be greater than minSize");
            numPriors++;
        }
        maxSize = copyToDevice(param.maxSize, param.numMaxSize);
    }
}

// Constructor
PriorBox::PriorBox(PriorBoxParameters param, int H, int W)
    : mParam(param)
    , H(H)
    , W(W)
{
    // minSize is required and needs to be non-negative
    ASSERT(param.numMinSize > 0 && param.minSize != nullptr);
    for (int i = 0; i < param.numMinSize; ++i)
    {
        ASSERT(param.minSize[i] > 0 && "minSize must be positive");
    }
    minSize = copyToDevice(param.minSize, param.numMinSize);
    ASSERT(param.numAspectRatios >= 0 && param.aspectRatios != nullptr);
    std::vector<float> tmpAR(1, 1);
    for (int i = 0; i < param.numAspectRatios; ++i)
    {
        float ar = param.aspectRatios[i];
        bool alreadyExist = false;
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
            if (param.flip)
            {
                tmpAR.push_back(1.0F / ar);
            }
        }
    }
    aspectRatios = copyToDevice(&tmpAR[0], tmpAR.size());
    numPriors = tmpAR.size() * param.numMinSize;
    if (param.numMaxSize > 0)
    {
        ASSERT(param.numMinSize == param.numMaxSize && param.maxSize != nullptr);
        for (int i = 0; i < param.numMaxSize; ++i)
        {
            // maxSize should be greater than minSize
            ASSERT(param.maxSize[i] > param.minSize[i] && "maxSize must be greater than minSize");
            numPriors++;
        }
        maxSize = copyToDevice(param.maxSize, param.numMaxSize);
    }
}

PriorBox::PriorBox(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mParam = read<PriorBoxParameters>(d);
    numPriors = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    minSize = deserializeToDevice(d, mParam.numMinSize);
    if (mParam.numMaxSize > 0)
    {
        maxSize = deserializeToDevice(d, mParam.numMaxSize);
    }
    int numAspectRatios = read<int>(d);
    aspectRatios = deserializeToDevice(d, numAspectRatios);
    ASSERT(d == a + length);
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
    H = inputs[0].d[1], W = inputs[0].d[2];
    // workaround for TRT
    // The first channel is for prior box coordinates.
    // The second channel is for prior box scaling factors, which is simply a copy of the variance provided.
    return DimsCHW(2, H * W * numPriors * 4, 1);
}

int PriorBox::initialize()
{
    return STATUS_SUCCESS;
}

void PriorBox::terminate()
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
}

size_t PriorBox::getWorkspaceSize(int /*maxBatchSize*/) const
{
    return 0;
}

int PriorBox::enqueue(
    int /*batchSize*/, const void* const* /*inputs*/, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    void* outputData = outputs[0];
    pluginStatus_t status = priorBoxInference(stream, mParam, H, W, numPriors, aspectRatios.count, minSize.values,
        maxSize.values, aspectRatios.values, outputData);
    ASSERT(status == STATUS_SUCCESS);

    return 0;
}

// Returns the size of serialized parameters
size_t PriorBox::getSerializationSize() const
{
    // PriorBoxParameters, numPriors,H,W, minSize, maxSize, numAspectRatios, aspectRatios
    return sizeof(PriorBoxParameters) + sizeof(int) * 3 + sizeof(float) * (mParam.numMinSize + mParam.numMaxSize)
        + sizeof(int) + sizeof(float) * aspectRatios.count;
}

void PriorBox::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mParam);
    write(d, numPriors);
    write(d, H);
    write(d, W);
    serializeFromDevice(d, minSize);
    if (mParam.numMaxSize > 0)
    {
        serializeFromDevice(d, maxSize);
    }
    write(d, (int) aspectRatios.count);
    serializeFromDevice(d, aspectRatios);
    ASSERT(d == a + getSerializationSize());
}

bool PriorBox::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

Weights PriorBox::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData = nullptr;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void PriorBox::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
    cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights PriorBox::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
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
    delete this;
}

IPluginV2Ext* PriorBox::clone() const
{
    IPluginV2Ext* plugin = new PriorBox(mParam, H, W);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// Set plugin namespace
void PriorBox::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* PriorBox::getPluginNamespace() const
{
    return mPluginNamespace;
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
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    // prepare for the inference function
    if (mParam.imgH == 0 || mParam.imgW == 0)
    {
        mParam.imgH = inputDims[1].d[1];
        mParam.imgW = inputDims[1].d[2];
    }
    if (mParam.stepH == 0 || mParam.stepW == 0)
    {
        mParam.stepH = static_cast<float>(mParam.imgH) / H;
        mParam.stepW = static_cast<float>(mParam.imgW) / W;
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
    for (auto v : mTmpAllocs)
    {
        free(v);
    }
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
            params.minSize = allocMemory<float>(size);
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
            params.maxSize = allocMemory<float>(size);
            const auto* maxS = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.maxSize[j] = *maxS;
                maxS++;
            }
            params.numMaxSize = size;
        }
        else if (!strcmp(attrName, "aspectRatios"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            params.aspectRatios = allocMemory<float>(size);
            const auto* aR = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.aspectRatios[j] = *aR;
                aR++;
            }
            params.numAspectRatios = size;
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
