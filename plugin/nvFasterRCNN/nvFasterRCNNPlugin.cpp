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
#include "nvFasterRCNNPlugin.h"
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <iostream>

using namespace nvinfer1;
using nvinfer1::Dims;
using nvinfer1::PluginType;
using nvinfer1::plugin::RPROIPlugin;
using nvinfer1::plugin::RPROIPluginCreator;

namespace
{
const char* RPROI_PLUGIN_VERSION{"1"};
const char* RPROI_PLUGIN_NAME{"RPROI_TRT"};
} // namespace

PluginFieldCollection RPROIPluginCreator::mFC{};
std::vector<PluginField> RPROIPluginCreator::mPluginAttributes;

RPROIPlugin::RPROIPlugin(RPROIParams params, const float* anchorsRatios, const float* anchorsScales)
    : params(params)
{
    /*
     * It only supports the scenario where params.featureStride == params.minBoxSize
     * assert(params.featureStride == params.minBoxSize);
     */
    ASSERT(params.anchorsRatioCount > 0 && params.anchorsScaleCount > 0);
    anchorsRatiosHost = copyToHost(anchorsRatios, params.anchorsRatioCount);
    anchorsScalesHost = copyToHost(anchorsScales, params.anchorsScaleCount);

    CHECK(cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    pluginStatus_t status = generateAnchors(0, params.anchorsRatioCount, anchorsRatiosHost, params.anchorsScaleCount,
        anchorsScalesHost, params.featureStride, anchorsDev);
    ASSERT(status == STATUS_SUCCESS);
}

// Constructor for cloning one plugin instance to another
RPROIPlugin::RPROIPlugin(RPROIParams params, const float* anchorsRatios, const float* anchorsScales, int A, int C,
    int H, int W, const float* _anchorsDev)
    : params(params)
    , A(A)
    , C(C)
    , H(H)
    , W(W)
{
    ASSERT(params.anchorsRatioCount > 0 && params.anchorsScaleCount > 0);
    anchorsRatiosHost = copyToHost(anchorsRatios, params.anchorsRatioCount);
    anchorsScalesHost = copyToHost(anchorsScales, params.anchorsScaleCount);

    CHECK(cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    // Perform deep copy
    if (_anchorsDev != nullptr)
    {
        CHECK(cudaMemcpy(anchorsDev, _anchorsDev,
            4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

RPROIPlugin::RPROIPlugin(const void* data, size_t length)
    : anchorsDev(nullptr)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    params = *reinterpret_cast<const RPROIParams*>(d);
    d += sizeof(RPROIParams);
    A = read<int>(d);
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    anchorsRatiosHost = copyToHost(d, params.anchorsRatioCount);
    d += params.anchorsRatioCount * sizeof(float);
    anchorsScalesHost = copyToHost(d, params.anchorsScaleCount);
    d += params.anchorsScaleCount * sizeof(float);
    ASSERT(d == a + length);

    CHECK(cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    pluginStatus_t status = generateAnchors(0, params.anchorsRatioCount, anchorsRatiosHost, params.anchorsScaleCount,
        anchorsScalesHost, params.featureStride, anchorsDev);
    ASSERT(status == STATUS_SUCCESS);
}

RPROIPlugin::~RPROIPlugin()
{
    if (anchorsDev != nullptr)
    {
        CHECK(cudaFree(anchorsDev));
        anchorsDev = nullptr;
    }
    if (anchorsRatiosHost != nullptr)
    {
        CHECK(cudaFreeHost(anchorsRatiosHost));
        anchorsRatiosHost = nullptr;
    }
    if (anchorsScalesHost != nullptr)
    {
        CHECK(cudaFreeHost(anchorsScalesHost));
        anchorsScalesHost = nullptr;
    }
}

int RPROIPlugin::initialize()
{
    return STATUS_SUCCESS;
}

int RPROIPlugin::getNbOutputs() const
{
    return 2;
}

Dims RPROIPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(index >= 0 && index < 2);
    ASSERT(nbInputDims == 4);
    ASSERT(inputs[0].nbDims == 3 && inputs[1].nbDims == 3 && inputs[2].nbDims == 3 && inputs[3].nbDims == 3);
    if (index == 0) // rois
    {
        return DimsCHW(1, params.nmsMaxOut, 4);
    }
    // Feature map of each ROI after ROI Pooling
    else // pool5
    {
        return DimsNCHW(params.nmsMaxOut, inputs[2].d[0], params.poolingH, params.poolingW);
    }
}

size_t RPROIPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return RPROIInferenceFusedWorkspaceSize(maxBatchSize, A, H, W, params.nmsMaxOut);
}

int RPROIPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // Bounding box (region proposal) objectness scores.
    const void* const scores = inputs[0];
    // Predicted bounding box offsets.
    const void* const deltas = inputs[1];
    // Feature map using for bounding box regression and classification.
    const void* const fmap = inputs[2];
    // Original image input information.
    const void* const iinfo = inputs[3];

    // Coordinates of region of interest (ROI) bounding boxes on the original input image.
    void* rois = outputs[0];
    // ROI pooled feature map corresponding to the region of interest (ROI).
    void* pfmap = outputs[1];

    pluginStatus_t status = RPROIInferenceFused(stream, batchSize, A, C, H, W, params.poolingH, params.poolingW,
        params.featureStride, params.preNmsTop, params.nmsMaxOut, params.iouThreshold, params.minBoxSize,
        params.spatialScale, (const float*) iinfo, this->anchorsDev, nvinfer1::DataType::kFLOAT, NCHW, scores,
        nvinfer1::DataType::kFLOAT, NCHW, deltas, nvinfer1::DataType::kFLOAT, NCHW, fmap, workspace,
        nvinfer1::DataType::kFLOAT, rois, nvinfer1::DataType::kFLOAT, NCHW, pfmap);
    ASSERT(status == STATUS_SUCCESS);
    return 0;
}

size_t RPROIPlugin::getSerializationSize() const
{
    size_t paramSize = sizeof(RPROIParams);
    size_t intSize = sizeof(int) * 4;
    size_t ratiosSize = sizeof(float) * params.anchorsRatioCount;
    size_t scalesSize = sizeof(float) * params.anchorsScaleCount;
    return paramSize + intSize + ratiosSize + scalesSize;
}

void RPROIPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    *reinterpret_cast<RPROIParams*>(d) = params;
    d += sizeof(RPROIParams);
    *reinterpret_cast<int*>(d) = A;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = C;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = H;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = W;
    d += sizeof(int);
    d += copyFromHost(d, anchorsRatiosHost, params.anchorsRatioCount);
    d += copyFromHost(d, anchorsScalesHost, params.anchorsScaleCount);
    ASSERT(d == a + getSerializationSize());
}

float* RPROIPlugin::copyToHost(const void* srcHostData, int count)
{
    float* dstHostPtr = nullptr;
    CHECK(cudaMallocHost(&dstHostPtr, count * sizeof(float)));
    CHECK(cudaMemcpy(dstHostPtr, srcHostData, count * sizeof(float), cudaMemcpyHostToHost));
    return dstHostPtr;
}

int RPROIPlugin::copyFromHost(char* dstHostBuffer, const void* source, int count) const
{
    cudaMemcpy(dstHostBuffer, source, count * sizeof(float), cudaMemcpyHostToHost);
    return count * sizeof(float);
}

bool RPROIPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* RPROIPlugin::getPluginType() const
{
    return RPROI_PLUGIN_NAME;
}

const char* RPROIPlugin::getPluginVersion() const
{
    return RPROI_PLUGIN_VERSION;
}

void RPROIPlugin::terminate() {}

void RPROIPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* RPROIPlugin::clone() const
{
    IPluginV2Ext* plugin = new RPROIPlugin(params, anchorsRatiosHost, anchorsScalesHost, A, C, H, W, anchorsDev);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// Set plugin namespace
void RPROIPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* RPROIPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index.
DataType RPROIPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Two outputs
    ASSERT(index == 0 || index == 1);
    return DataType::kFLOAT;
}
// Return true if output tensor is broadcast across a batch.
bool RPROIPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool RPROIPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
// inutDims: input Dimensions for the plugin layer
// nInputs : Number of inputs to the plugin layer
// outputDims: output Dimensions from the plugin layer
// nOutputs: number of outputs from the plugin layer
// type: DataType configuration for the plugin layer
// format: format NCHW, NHWC etc
// maxbatchSize: maximum batch size for the plugin layer
void RPROIPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);

    A = params.anchorsRatioCount * params.anchorsScaleCount;
    C = inputDims[2].d[0];
    H = inputDims[2].d[1];
    W = inputDims[2].d[2];

    ASSERT(nbInputs == 4);
    ASSERT(inputDims[0].d[0] == (2 * A) && inputDims[1].d[0] == (4 * A));
    ASSERT(inputDims[0].d[1] == inputDims[1].d[1] && inputDims[0].d[1] == inputDims[2].d[1]);
    ASSERT(inputDims[0].d[2] == inputDims[1].d[2] && inputDims[0].d[2] == inputDims[2].d[2]);
    ASSERT(nbOutputs == 2 && outputDims[0].nbDims == 3 // rois
        && outputDims[1].nbDims == 4);                 // pooled feature map
    ASSERT(outputDims[0].d[0] == 1 && outputDims[0].d[1] == params.nmsMaxOut && outputDims[0].d[2] == 4);
    ASSERT(outputDims[1].d[0] == params.nmsMaxOut && outputDims[1].d[1] == C && outputDims[1].d[2] == params.poolingH
        && outputDims[1].d[3] == params.poolingW);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void RPROIPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void RPROIPlugin::detachFromContext() {}

RPROIPluginCreator::RPROIPluginCreator()
{

    mPluginAttributes.emplace_back(PluginField("poolingH", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("poolingW", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("featureStride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("preNmsTop", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nmsMaxOut", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchorsRatioCount", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchorsScaleCount", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("minBoxSize", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("spatialScale", nullptr, PluginFieldType::kFLOAT32, 1));

    // TODO Do we need to pass the size attribute here for float arrarys, we
    // dont have that information at this point.
    mPluginAttributes.emplace_back(PluginField("anchorsRatios", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchorsScales", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

RPROIPluginCreator::~RPROIPluginCreator()
{
    // Free allocated memory (if any) here
}

const char* RPROIPluginCreator::getPluginName() const
{
    return RPROI_PLUGIN_NAME;
}

const char* RPROIPluginCreator::getPluginVersion() const
{
    return RPROI_PLUGIN_VERSION;
}

const PluginFieldCollection* RPROIPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* RPROIPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "poolingH"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.poolingH = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "poolingW"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.poolingW = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "featureStride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.featureStride = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "preNmsTop"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.preNmsTop = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "nmsMaxOut"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.nmsMaxOut = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "anchorsRatioCount"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.anchorsRatioCount = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "anchorsScaleCount"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.anchorsScaleCount = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "iouThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.iouThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        if (!strcmp(attrName, "minBoxSize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.minBoxSize = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        if (!strcmp(attrName, "spatialScale"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.spatialScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        if (!strcmp(attrName, "anchorsRatios"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            anchorsRatios.reserve(params.anchorsRatioCount);
            const float* ratios = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < params.anchorsRatioCount; ++j)
            {
                anchorsRatios.push_back(*ratios);
                ratios++;
            }
        }
        if (!strcmp(attrName, "anchorsScales"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            anchorsScales.reserve(params.anchorsScaleCount);
            const float* scales = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < params.anchorsScaleCount; ++j)
            {
                anchorsScales.push_back(*scales);
                scales++;
            }
        }
    }

    // This object will be deleted when the network is destroyed, which will
    // call RPROIPlugin::terminate()
    RPROIPlugin* plugin = new RPROIPlugin(params, anchorsRatios.data(), anchorsScales.data());
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* RPROIPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call RPROIPlugin::terminate()
    RPROIPlugin* plugin = new RPROIPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
