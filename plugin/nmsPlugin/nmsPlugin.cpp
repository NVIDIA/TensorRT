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
#include "nmsPlugin.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::DetectionOutput;
using nvinfer1::plugin::DetectionOutputDynamic;
using nvinfer1::plugin::DetectionOutputParameters;
using nvinfer1::plugin::NMSPluginCreator;
using nvinfer1::plugin::NMSDynamicPluginCreator;
using nvinfer1::plugin::NMSBasePluginCreator;

namespace
{
const char* NMS_PLUGIN_VERSION{"1"};
const char* NMS_PLUGIN_NAMES[] = {"NMS_TRT", "NMSDynamic_TRT"};
} // namespace

PluginFieldCollection NMSBasePluginCreator::mFC{};
std::vector<PluginField> NMSBasePluginCreator::mPluginAttributes;

// Constrcutor
DetectionOutput::DetectionOutput(DetectionOutputParameters params)
    : param(params)
    , C1(0)
    , C2(0)
    , numPriors(0)
    , mType(DataType::kFLOAT)
    , mScoreBits(16)
{
}

DetectionOutputDynamic::DetectionOutputDynamic(DetectionOutputParameters params)
    : param(params)
    , C1(0)
    , C2(0)
    , numPriors(0)
    , mType(DataType::kFLOAT)
    , mScoreBits(16)
{
}

DetectionOutput::DetectionOutput(DetectionOutputParameters params, int C1, int C2, int numPriors)
    : param(params)
    , C1(C1)
    , C2(C2)
    , numPriors(numPriors)
    , mType(DataType::kFLOAT)
    , mScoreBits(16)
{
}

DetectionOutputDynamic::DetectionOutputDynamic(DetectionOutputParameters params, int C1, int C2, int numPriors)
    : param(params)
    , C1(C1)
    , C2(C2)
    , numPriors(numPriors)
    , mType(DataType::kFLOAT)
    , mScoreBits(16)
{
}

// Parameterized constructor
DetectionOutput::DetectionOutput(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<DetectionOutputParameters>(d);
    // Channel size of the locData tensor
    // numPriors * numLocClasses * 4
    C1 = read<int>(d);
    // Channel size of the confData tensor
    // numPriors * param.numClasses
    C2 = read<int>(d);
    // Number of bounding boxes per sample
    numPriors = read<int>(d);
    // data type of this plugin
    mType = read<DataType>(d);
    // mScoreBits
    mScoreBits = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

DetectionOutputDynamic::DetectionOutputDynamic(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<DetectionOutputParameters>(d);
    // Channel size of the locData tensor
    // numPriors * numLocClasses * 4
    C1 = read<int>(d);
    // Channel size of the confData tensor
    // numPriors * param.numClasses
    C2 = read<int>(d);
    // Number of bounding boxes per sample
    numPriors = read<int>(d);
    // data type of this plugin
    mType = read<DataType>(d);
    // mScoreBits
    mScoreBits = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

int DetectionOutput::getNbOutputs() const noexcept
{
    // Plugin layer has 2 outputs
    return 2;
}

int DetectionOutputDynamic::getNbOutputs() const noexcept
{
    // Plugin layer has 2 outputs
    return 2;
}

int DetectionOutput::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int DetectionOutputDynamic::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void DetectionOutput::terminate() noexcept {}

void DetectionOutputDynamic::terminate() noexcept {}

// Returns output dimensions at given index
Dims DetectionOutput::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 3);
    PLUGIN_ASSERT(index == 0 || index == 1);
    // Output dimensions
    // index 0 : Dimensions 1x param.keepTopK x 7
    // index 1: Dimensions 1x1x1
    if (index == 0)
    {
        return Dims3(1, param.keepTopK, 7);
    }
    return Dims3(1, 1, 1);
}

DimsExprs DetectionOutputDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    // loc data
    PLUGIN_ASSERT(inputs[0].nbDims == 4);
    // conf data
    PLUGIN_ASSERT(inputs[1].nbDims == 4);
    // prior data
    PLUGIN_ASSERT(inputs[2].nbDims == 4);
    const int C1_idx = param.inputOrder[0];
    const int C2_idx = param.inputOrder[1];
    if (inputs[C1_idx].d[0]->isConstant() && inputs[C1_idx].d[1]->isConstant() && inputs[C1_idx].d[2]->isConstant()
        && inputs[C1_idx].d[3]->isConstant())
    {
        C1 = exprBuilder
                 .operation(DimensionOperation::kPROD,
                     *exprBuilder.operation(DimensionOperation::kPROD, *inputs[C1_idx].d[1], *inputs[C1_idx].d[2]),
                     *inputs[C1_idx].d[3])
                 ->getConstantValue();
    }

    if (inputs[C2_idx].d[0]->isConstant() && inputs[C2_idx].d[1]->isConstant() && inputs[C2_idx].d[2]->isConstant())
    {
        C2 = exprBuilder.operation(DimensionOperation::kPROD, *inputs[C2_idx].d[1], *inputs[C2_idx].d[2])
                 ->getConstantValue();
    }
    // Output dimensions
    // index 0 : Dimensions 1x param.keepTopK x 7
    // index 1: Dimensions 1x1x1
    DimsExprs out_dim;
    if (outputIndex == 0)
    {
        // (N, 1, param.keepTopK, 7)
        out_dim.nbDims = 4;
        out_dim.d[0] = inputs[0].d[0];
        out_dim.d[1] = exprBuilder.constant(1);
        out_dim.d[2] = exprBuilder.constant(param.keepTopK);
        out_dim.d[3] = exprBuilder.constant(7);
    }
    else
    {
        out_dim.nbDims = 4;
        out_dim.d[0] = inputs[0].d[0];
        out_dim.d[1] = exprBuilder.constant(1);
        out_dim.d[2] = exprBuilder.constant(1);
        out_dim.d[3] = exprBuilder.constant(1);
    }
    return out_dim;
}

// Returns the workspace size
size_t DetectionOutput::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return detectionInferenceWorkspaceSize(
        param.shareLocation, maxBatchSize, C1, C2, param.numClasses, numPriors, param.topK, mType, mType);
}

size_t DetectionOutputDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return detectionInferenceWorkspaceSize(
        param.shareLocation, inputs[0].dims.d[0], C1, C2, param.numClasses, numPriors, param.topK, mType, mType);
}

// Plugin layer implementation
int DetectionOutput::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // Input order {loc, conf, prior}
    const void* const locData = inputs[param.inputOrder[0]];
    const void* const confData = inputs[param.inputOrder[1]];
    const void* const priorData = inputs[param.inputOrder[2]];

    // Output from plugin index 0: topDetections index 1: keepCount
    void* topDetections = outputs[0];
    void* keepCount = outputs[1];

    pluginStatus_t status = detectionInference(stream, batchSize, C1, C2, param.shareLocation,
        param.varianceEncodedInTarget, param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK,
        param.confidenceThreshold, param.nmsThreshold, param.codeType, mType, locData, priorData, mType, confData,
        keepCount, topDetections, workspace, param.isNormalized, param.confSigmoid, mScoreBits, param.isBatchAgnostic);
    return status;
}

int32_t DetectionOutputDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // Input order {loc, conf, prior}
    const void* const locData = inputs[param.inputOrder[0]];
    const void* const confData = inputs[param.inputOrder[1]];
    const void* const priorData = inputs[param.inputOrder[2]];

    // Output from plugin index 0: topDetections index 1: keepCount
    void* topDetections = outputs[0];
    void* keepCount = outputs[1];

    pluginStatus_t status = detectionInference(stream, inputDesc[0].dims.d[0], C1, C2, param.shareLocation,
        param.varianceEncodedInTarget, param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK,
        param.confidenceThreshold, param.nmsThreshold, param.codeType, mType, locData, priorData, mType, confData,
        keepCount, topDetections, workspace, param.isNormalized, param.confSigmoid, mScoreBits, false);
    return status;
}

// Returns the size of serialized parameters
size_t DetectionOutput::getSerializationSize() const noexcept
{
    // DetectionOutputParameters, C1, C2, numPriors, mType, mScoreBits
    return sizeof(DetectionOutputParameters) + sizeof(int) * 3 + sizeof(DataType) + sizeof(int32_t);
}

size_t DetectionOutputDynamic::getSerializationSize() const noexcept
{
    // DetectionOutputParameters, C1, C2, numPriors, mType, mScoreBits
    return sizeof(DetectionOutputParameters) + sizeof(int) * 3 + sizeof(DataType) + sizeof(int32_t);
}

// Serialization of plugin parameters
void DetectionOutput::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, C1);
    write(d, C2);
    write(d, numPriors);
    write(d, mType);
    write(d, mScoreBits);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void DetectionOutputDynamic::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, C1);
    write(d, C2);
    write(d, numPriors);
    write(d, mType);
    write(d, mScoreBits);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

// Check if the DataType and Plugin format is supported
bool DetectionOutput::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kHALF || type == DataType::kFLOAT) && format == PluginFormat::kLINEAR);
}

bool DetectionOutputDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 3 inputs, 2 outputs, so 5 input/output in total
    PLUGIN_ASSERT(0 <= pos && pos < 5);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const bool consistentFloatPrecision = (in[0].type == in[pos].type);
    switch (pos)
    {
    case 0:
        return (in[0].type == DataType::kHALF || in[0].type == DataType::kFLOAT)
            && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT)
            && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2:
        return (in[2].type == DataType::kHALF || in[2].type == DataType::kFLOAT)
            && in[2].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 3:
        return (out[0].type == DataType::kHALF || out[0].type == DataType::kFLOAT)
            && out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 4: return out[1].type == DataType::kFLOAT && out[1].format == PluginFormat::kLINEAR;
    }
    return false;
}

// Get the plugin type
const char* DetectionOutput::getPluginType() const noexcept
{
    return NMS_PLUGIN_NAMES[0];
}

const char* DetectionOutputDynamic::getPluginType() const noexcept
{
    return NMS_PLUGIN_NAMES[1];
}

// Get the plugin version
const char* DetectionOutput::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

const char* DetectionOutputDynamic::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

// Clean up
void DetectionOutput::destroy() noexcept
{
    delete this;
}

void DetectionOutputDynamic::destroy() noexcept
{
    delete this;
}

void DetectionOutput::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

void DetectionOutputDynamic::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

// Cloning the plugin
IPluginV2Ext* DetectionOutput::clone() const noexcept
{
    try
    {
        // Create a new instance
        auto* plugin = new DetectionOutput(param, C1, C2, numPriors);
        plugin->mType = mType;
        // Set the namespace
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        // set mScoreBits
        plugin->setScoreBits(mScoreBits);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* DetectionOutputDynamic::clone() const noexcept
{
    try
    {
        // Create a new instance
        auto* plugin = new DetectionOutputDynamic(param, C1, C2, numPriors);
        plugin->mType = mType;
        // Set the namespace
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        // set mScoreBits
        plugin->setScoreBits(mScoreBits);
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// Set plugin namespace
void DetectionOutput::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

void DetectionOutputDynamic::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* DetectionOutput::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

const char* DetectionOutputDynamic::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType DetectionOutput::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    noexcept
{
    // Two outputs
    PLUGIN_ASSERT(index == 0 || index == 1);
    PLUGIN_ASSERT(inputTypes[0] == inputTypes[1] && inputTypes[2] == inputTypes[1]);
    // topDetections
    if (index == 0)
    {
        return inputTypes[0];
    }
    // keepCount: use kFLOAT instead as they have same sizeof(type)
    PLUGIN_ASSERT(sizeof(int) == sizeof(float));
    return DataType::kFLOAT;
}

DataType DetectionOutputDynamic::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    noexcept
{
    // Two outputs
    PLUGIN_ASSERT(index == 0 || index == 1);
    PLUGIN_ASSERT(inputTypes[0] == inputTypes[1] && inputTypes[2] == inputTypes[1]);
    // topDetections
    if (index == 0)
    {
        return inputTypes[0];
    }
    // keepCount: use kFLOAT instead as they have same sizeof(type)
    PLUGIN_ASSERT(sizeof(int) == sizeof(float));
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool DetectionOutput::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DetectionOutput::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
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
void DetectionOutput::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 2);

    // Verify all the input dimensions
    for (int i = 0; i < nbInputs; i++)
    {
        PLUGIN_ASSERT(inputDims[i].nbDims == 3);
    }

    // Verify all the output dimensions
    for (int i = 0; i < nbOutputs; i++)
    {
        PLUGIN_ASSERT(outputDims[i].nbDims == 3);
    }

    // Configure C1, C2 and numPriors
    // Input ordering  C1, C2, numPriors
    C1 = inputDims[param.inputOrder[0]].d[0];
    C2 = inputDims[param.inputOrder[1]].d[0];

    const int nbBoxCoordinates = 4;
    numPriors = inputDims[param.inputOrder[2]].d[1] / nbBoxCoordinates;
    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;

    // Verify C1
    PLUGIN_ASSERT(numPriors * numLocClasses * nbBoxCoordinates == inputDims[param.inputOrder[0]].d[0]);

    // Verify C2
    PLUGIN_ASSERT(numPriors * param.numClasses == inputDims[param.inputOrder[1]].d[0]);

    // initialize mType
    mType = inputTypes[0];
}

void DetectionOutputDynamic::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 2);

    // Verify all the input dimensions
    for (int i = 0; i < nbInputs; i++)
    {
        PLUGIN_ASSERT(in[i].desc.dims.nbDims == 4);
    }

    // Verify all the output dimensions
    for (int i = 0; i < nbOutputs; i++)
    {
        PLUGIN_ASSERT(out[i].desc.dims.nbDims == 4);
    }

    // Configure C1, C2 and numPriors
    // Input ordering  C1, C2, numPriors
    C1 = in[param.inputOrder[0]].desc.dims.d[1];
    C2 = in[param.inputOrder[1]].desc.dims.d[1];

    const int nbBoxCoordinates = 4;
    numPriors = in[param.inputOrder[2]].desc.dims.d[2] / nbBoxCoordinates;
    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;

    // Verify C1
    PLUGIN_ASSERT(numPriors * numLocClasses * nbBoxCoordinates == in[param.inputOrder[0]].desc.dims.d[1]);

    // Verify C2
    PLUGIN_ASSERT(numPriors * param.numClasses == in[param.inputOrder[1]].desc.dims.d[1]);

    // initialize mType
    mType = in[0].desc.type;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DetectionOutput::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void DetectionOutput::detachFromContext() noexcept {}

// Plugin creator constructor
NMSBasePluginCreator::NMSBasePluginCreator()
{
    // NMS Plugin field meta data {name,  data, type, length}
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("varianceEncodedInTarget", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("confidenceThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("inputOrder", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("confSigmoid", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("codeType", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreBits", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("isBatchAgnostic", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

NMSPluginCreator::NMSPluginCreator()
{
    mPluginName = NMS_PLUGIN_NAMES[0];
}

NMSDynamicPluginCreator::NMSDynamicPluginCreator()
{
    mPluginName = NMS_PLUGIN_NAMES[1];
}

// Returns the plugin name
const char* NMSBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

// Returns the plugin version
const char* NMSBasePluginCreator::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

// Returns the plugin field names
const PluginFieldCollection* NMSBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

// Creates the NMS plugin
IPluginV2Ext* NMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        // Default init values for TF SSD network
        params.codeType = CodeTypeSSD::TF_CENTER;
        params.inputOrder[0] = 0;
        params.inputOrder[1] = 2;
        params.inputOrder[2] = 1;
        // scoreBits defaults to 16
        mScoreBits = 16;

        // Read configurations from  each fields
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.shareLocation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "varianceEncodedInTarget"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.varianceEncodedInTarget = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "backgroundLabelId"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "numClasses"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "topK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.topK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "keepTopK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "confidenceThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.confidenceThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "nmsThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.nmsThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "confSigmoid"))
            {
                params.confSigmoid = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "isNormalized"))
            {
                params.isNormalized = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "inputOrder"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                const int size = fields[i].length;
                const int* o = static_cast<const int*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    params.inputOrder[j] = *o;
                    o++;
                }
            }
            else if (!strcmp(attrName, "codeType"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.codeType = static_cast<CodeTypeSSD>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "scoreBits"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mScoreBits = *(static_cast<const int32_t*>(fields[i].data));
            }
            else if (!strcmp(attrName, "isBatchAgnostic"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.isBatchAgnostic = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
        }

        DetectionOutput* obj = new DetectionOutput(params);
        obj->setScoreBits(mScoreBits);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* NMSDynamicPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        // Default init values for TF SSD network
        params.codeType = CodeTypeSSD::TF_CENTER;
        params.inputOrder[0] = 0;
        params.inputOrder[1] = 2;
        params.inputOrder[2] = 1;
        // scoreBits defaults to 16
        mScoreBits = 16;

        // Read configurations from  each fields
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.shareLocation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "varianceEncodedInTarget"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.varianceEncodedInTarget = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "backgroundLabelId"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "numClasses"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "topK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.topK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "keepTopK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "confidenceThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.confidenceThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "nmsThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.nmsThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "confSigmoid"))
            {
                params.confSigmoid = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "isNormalized"))
            {
                params.isNormalized = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "inputOrder"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                const int size = fields[i].length;
                const int* o = static_cast<const int*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    params.inputOrder[j] = *o;
                    o++;
                }
            }
            else if (!strcmp(attrName, "codeType"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.codeType = static_cast<CodeTypeSSD>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "scoreBits"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mScoreBits = *(static_cast<const int32_t*>(fields[i].data));
            }
        }

        DetectionOutputDynamic* obj = new DetectionOutputDynamic(params);
        obj->setScoreBits(mScoreBits);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* NMSPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        DetectionOutput* obj = new DetectionOutput(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* NMSDynamicPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        DetectionOutputDynamic* obj = new DetectionOutputDynamic(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
