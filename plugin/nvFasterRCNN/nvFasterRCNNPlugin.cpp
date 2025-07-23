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
#include "nvFasterRCNNPlugin.h"
#include <cstdio>
#include <cstring>
#include <iostream>

namespace nvinfer1::plugin
{
namespace
{
char const* const kRPROI_PLUGIN_VERSION{"1"};
char const* const kRPROI_PLUGIN_NAME{"RPROI_TRT"};
} // namespace

RPROIPlugin::RPROIPlugin(RPROIParams params, float const* anchorsRatios, float const* anchorsScales)
    : params(params)
{
    /*
     * It only supports the scenario where params.featureStride == params.minBoxSize
     * assert(params.featureStride == params.minBoxSize);
     */
    PLUGIN_VALIDATE(params.anchorsRatioCount > 0 && params.anchorsScaleCount > 0);
    anchorsRatiosHost = copyToHost(anchorsRatios, params.anchorsRatioCount);
    anchorsScalesHost = copyToHost(anchorsScales, params.anchorsScaleCount);

    PLUGIN_CHECK(
        cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    pluginStatus_t status = generateAnchors(0, params.anchorsRatioCount, anchorsRatiosHost, params.anchorsScaleCount,
        anchorsScalesHost, params.featureStride, anchorsDev);
    PLUGIN_VALIDATE(status == STATUS_SUCCESS);

    deviceSmemSize = getSmemSize();
}

// Constructor for cloning one plugin instance to another
RPROIPlugin::RPROIPlugin(RPROIParams params, float const* anchorsRatios, float const* anchorsScales, int32_t A,
    int32_t C, int32_t H, int32_t W, float const* _anchorsDev, size_t deviceSmemSize, DataType inFeatureType,
    DataType outFeatureType, DLayout_t inFeatureLayout)
    : deviceSmemSize(deviceSmemSize)
    , params(params)
    , A(A)
    , C(C)
    , H(H)
    , W(W)
    , inFeatureType(inFeatureType)
    , outFeatureType(outFeatureType)
    , inFeatureLayout(inFeatureLayout)
{
    PLUGIN_VALIDATE(params.anchorsRatioCount > 0 && params.anchorsScaleCount > 0);
    anchorsRatiosHost = copyToHost(anchorsRatios, params.anchorsRatioCount);
    anchorsScalesHost = copyToHost(anchorsScales, params.anchorsScaleCount);

    PLUGIN_CHECK(
        cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    // Perform deep copy
    if (_anchorsDev != nullptr)
    {
        PLUGIN_CHECK(cudaMemcpy(anchorsDev, _anchorsDev,
            4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

RPROIPlugin::RPROIPlugin(void const* data, size_t length)
{
    deserialize(static_cast<int8_t const*>(data), length);
}

void RPROIPlugin::deserialize(int8_t const* data, size_t length)
{
    auto const* d{data};
    params = *reinterpret_cast<RPROIParams const*>(d);
    d += sizeof(RPROIParams);
    A = read<int32_t>(d);
    C = read<int32_t>(d);
    H = read<int32_t>(d);
    W = read<int32_t>(d);
    inFeatureType = read<DataType>(d);
    outFeatureType = read<DataType>(d);
    inFeatureLayout = read<DLayout_t>(d);
    anchorsRatiosHost = copyToHost(d, params.anchorsRatioCount);
    d += params.anchorsRatioCount * sizeof(float);
    anchorsScalesHost = copyToHost(d, params.anchorsScaleCount);
    d += params.anchorsScaleCount * sizeof(float);
    PLUGIN_VALIDATE(d == data + length);

    PLUGIN_CHECK(
        cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    pluginStatus_t status = generateAnchors(0, params.anchorsRatioCount, anchorsRatiosHost, params.anchorsScaleCount,
        anchorsScalesHost, params.featureStride, anchorsDev);
    PLUGIN_VALIDATE(status == STATUS_SUCCESS);

    deviceSmemSize = getSmemSize();
}

RPROIPlugin::~RPROIPlugin()
{
    if (anchorsDev != nullptr)
    {
        PLUGIN_CHECK(cudaFree(anchorsDev));
        anchorsDev = nullptr;
    }
    if (anchorsRatiosHost != nullptr)
    {
        PLUGIN_CHECK(cudaFreeHost(anchorsRatiosHost));
        anchorsRatiosHost = nullptr;
    }
    if (anchorsScalesHost != nullptr)
    {
        PLUGIN_CHECK(cudaFreeHost(anchorsScalesHost));
        anchorsScalesHost = nullptr;
    }
}

int32_t RPROIPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

size_t RPROIPlugin::getSmemSize() const noexcept
{
    int32_t devId{-1};
    PLUGIN_CHECK(cudaGetDevice(&devId));
    cudaDeviceProp prop{};
    PLUGIN_CHECK(cudaGetDeviceProperties(&prop, devId));
    return prop.sharedMemPerBlockOptin;
}

int32_t RPROIPlugin::getNbOutputs() const noexcept
{
    return 2;
}

Dims RPROIPlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_ASSERT(index >= 0 && index < 2);
    PLUGIN_ASSERT(nbInputDims == 4);
    PLUGIN_ASSERT(inputs[0].nbDims == 3 && inputs[1].nbDims == 3 && inputs[2].nbDims == 3 && inputs[3].nbDims == 3);
    if (index == 0) // rois
    {
        return Dims3(1, params.nmsMaxOut, 4);
    }
    // Feature map of each ROI after ROI Pooling
    // pool5
    return Dims4(params.nmsMaxOut, inputs[2].d[0], params.poolingH, params.poolingW);
}

size_t RPROIPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return RPROIInferenceFusedWorkspaceSize(maxBatchSize, A, H, W, params.nmsMaxOut);
}

int32_t RPROIPlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // Bounding box (region proposal) objectness scores.
    void const* const scores = inputs[0];
    // Predicted bounding box offsets.
    void const* const deltas = inputs[1];
    // Feature map using for bounding box regression and classification.
    void const* const fmap = inputs[2];
    // Original image input information.
    void const* const iinfo = inputs[3];

    // Coordinates of region of interest (ROI) bounding boxes on the original input image.
    void* rois = outputs[0];
    // ROI pooled feature map corresponding to the region of interest (ROI).
    void* pfmap = outputs[1];

    pluginStatus_t status = RPROIInferenceFused(stream, batchSize, A, C, H, W, params.poolingH, params.poolingW,
        params.featureStride, params.preNmsTop, params.nmsMaxOut, params.iouThreshold, params.minBoxSize,
        params.spatialScale, (float const*) iinfo, this->anchorsDev, nvinfer1::DataType::kFLOAT, NCHW, scores,
        nvinfer1::DataType::kFLOAT, NCHW, deltas, inFeatureType, inFeatureLayout, fmap, workspace,
        nvinfer1::DataType::kFLOAT, rois, outFeatureType, NCHW, pfmap, deviceSmemSize);
    return status;
}

size_t RPROIPlugin::getSerializationSize() const noexcept
{
    size_t paramSize = sizeof(RPROIParams);
    size_t intSize = sizeof(int32_t) * 4;
    size_t ratiosSize = sizeof(float) * params.anchorsRatioCount;
    size_t scalesSize = sizeof(float) * params.anchorsScaleCount;
    size_t typeSize = sizeof(DataType) * 2;
    size_t layoutSize = sizeof(DLayout_t);
    return paramSize + intSize + ratiosSize + scalesSize + typeSize + layoutSize;
}

void RPROIPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    *reinterpret_cast<RPROIParams*>(d) = params;
    d += sizeof(RPROIParams);
    *reinterpret_cast<int32_t*>(d) = A;
    d += sizeof(int32_t);
    *reinterpret_cast<int32_t*>(d) = C;
    d += sizeof(int32_t);
    *reinterpret_cast<int32_t*>(d) = H;
    d += sizeof(int32_t);
    *reinterpret_cast<int32_t*>(d) = W;
    d += sizeof(int32_t);
    *reinterpret_cast<DataType*>(d) = inFeatureType;
    d += sizeof(DataType);
    *reinterpret_cast<DataType*>(d) = outFeatureType;
    d += sizeof(DataType);
    *reinterpret_cast<DLayout_t*>(d) = inFeatureLayout;
    d += sizeof(DLayout_t);
    d += copyFromHost(d, anchorsRatiosHost, params.anchorsRatioCount);
    d += copyFromHost(d, anchorsScalesHost, params.anchorsScaleCount);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

float* RPROIPlugin::copyToHost(void const* srcHostData, int32_t count) noexcept
{
    float* dstHostPtr = nullptr;
    PLUGIN_CHECK(cudaMallocHost(&dstHostPtr, count * sizeof(float)));
    PLUGIN_CHECK(cudaMemcpy(dstHostPtr, srcHostData, count * sizeof(float), cudaMemcpyHostToHost));
    return dstHostPtr;
}

int32_t RPROIPlugin::copyFromHost(char* dstHostBuffer, void const* source, int32_t count) const noexcept
{
    PLUGIN_CHECK(cudaMemcpy(dstHostBuffer, source, count * sizeof(float), cudaMemcpyHostToHost));
    return count * sizeof(float);
}

bool RPROIPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept
{
    PLUGIN_ASSERT(nbInputs == PluginNbInputs && nbOutputs == PluginNbOutputs && pos < nbInputs + nbOutputs);
    bool isValidCombination = false;

    // input:  bbox confindence, bbox offset, image info and output: rois
    if (pos == 0 || pos == 1 || pos == 3 || pos == 4)
    {
        isValidCombination |= (inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT);
    }
    // input:  feature map
    else if (pos == 2)
    {
        isValidCombination |= (inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT8);
        isValidCombination |= (inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT);
        isValidCombination |= (inOut[pos].format == TensorFormat::kCHW4 && inOut[pos].type == DataType::kINT8);
        isValidCombination |= (inOut[pos].format == TensorFormat::kCHW32 && inOut[pos].type == DataType::kINT8);
    }
    // output: pooled feature map (data type should be the same with input feature map)
    else if (pos == 5)
    {
        isValidCombination |= (inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT8);
        isValidCombination |= (inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT);
        isValidCombination &= inOut[pos].type == inOut[2].type;
    }
    return isValidCombination;
}

char const* RPROIPlugin::getPluginType() const noexcept
{
    return kRPROI_PLUGIN_NAME;
}

char const* RPROIPlugin::getPluginVersion() const noexcept
{
    return kRPROI_PLUGIN_VERSION;
}

void RPROIPlugin::terminate() noexcept {}

void RPROIPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* RPROIPlugin::clone() const noexcept
{
    try
    {
        IPluginV2Ext* plugin = new RPROIPlugin(params, anchorsRatiosHost, anchorsScalesHost, A, C, H, W, anchorsDev,
            deviceSmemSize, inFeatureType, outFeatureType, inFeatureLayout);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// Set plugin namespace
void RPROIPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

char const* RPROIPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType RPROIPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Two outputs
    PLUGIN_ASSERT(index == 0 || index == 1);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool RPROIPlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool RPROIPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

DLayout_t RPROIPlugin::convertTensorFormat(TensorFormat const& srcFormat) const noexcept
{
    PLUGIN_ASSERT(
        srcFormat == TensorFormat::kLINEAR || srcFormat == TensorFormat::kCHW4 || srcFormat == TensorFormat::kCHW32);
    switch (srcFormat)
    {
    case nvinfer1::TensorFormat::kLINEAR: return DLayout_t::NCHW;
    case nvinfer1::TensorFormat::kCHW4: return DLayout_t::NC4HW;
    case nvinfer1::TensorFormat::kCHW32: return DLayout_t::NC32HW;
    default: return DLayout_t::NCHW;
    }
}

void RPROIPlugin::configurePlugin(
    PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept
{
    PLUGIN_ASSERT(nbInput == PluginNbInputs);
    PLUGIN_ASSERT(nbOutput == PluginNbOutputs);

    A = params.anchorsRatioCount * params.anchorsScaleCount;
    C = in[2].dims.d[0];
    H = in[2].dims.d[1];
    W = in[2].dims.d[2];
    inFeatureType = in[2].type;
    outFeatureType = out[1].type;
    inFeatureLayout = convertTensorFormat(in[2].format);

    PLUGIN_ASSERT(in[0].dims.d[0] == (2 * A) && in[1].dims.d[0] == (4 * A));
    PLUGIN_ASSERT(in[0].dims.d[1] == in[1].dims.d[1] && in[0].dims.d[1] == in[2].dims.d[1]);
    PLUGIN_ASSERT(in[0].dims.d[2] == in[1].dims.d[2] && in[0].dims.d[2] == in[2].dims.d[2]);
    PLUGIN_ASSERT(out[0].dims.nbDims == 3 // rois
        && out[1].dims.nbDims == 4);      // pooled feature map
    PLUGIN_ASSERT(out[0].dims.d[0] == 1 && out[0].dims.d[1] == params.nmsMaxOut && out[0].dims.d[2] == 4);
    PLUGIN_ASSERT(out[1].dims.d[0] == params.nmsMaxOut && out[1].dims.d[1] == C && out[1].dims.d[2] == params.poolingH
        && out[1].dims.d[3] == params.poolingW);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void RPROIPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void RPROIPlugin::detachFromContext() noexcept {}

RPROIPluginCreator::RPROIPluginCreator()
{
    mPluginAttributes.clear();
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

char const* RPROIPluginCreator::getPluginName() const noexcept
{
    return kRPROI_PLUGIN_NAME;
}

char const* RPROIPluginCreator::getPluginVersion() const noexcept
{
    return kRPROI_PLUGIN_VERSION;
}

PluginFieldCollection const* RPROIPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* RPROIPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;

        for (int32_t i = 0; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "poolingH"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.poolingH = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "poolingW"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.poolingW = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "featureStride"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.featureStride = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "preNmsTop"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.preNmsTop = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "nmsMaxOut"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.nmsMaxOut = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "anchorsRatioCount"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.anchorsRatioCount = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "anchorsScaleCount"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.anchorsScaleCount = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "iouThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.iouThreshold = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
            }
            if (!strcmp(attrName, "minBoxSize"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.minBoxSize = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
            }
            if (!strcmp(attrName, "spatialScale"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.spatialScale = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
            }
            if (!strcmp(attrName, "anchorsRatios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                anchorsRatios.reserve(params.anchorsRatioCount);
                float const* ratios = static_cast<float const*>(fields[i].data);
                for (int32_t j = 0; j < params.anchorsRatioCount; ++j)
                {
                    anchorsRatios.push_back(*ratios);
                    ratios++;
                }
            }
            if (!strcmp(attrName, "anchorsScales"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                anchorsScales.reserve(params.anchorsScaleCount);
                float const* scales = static_cast<float const*>(fields[i].data);
                for (int32_t j = 0; j < params.anchorsScaleCount; ++j)
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
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* RPROIPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call RPROIPlugin::terminate()
        RPROIPlugin* plugin = new RPROIPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
} // namespace nvinfer1::plugin
