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

#include "batchedNMSPlugin.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

namespace nvinfer1::plugin
{

#define NVBUG_3321606_WAR 1

namespace
{
char const* const kNMS_PLUGIN_VERSION{"1"};
char const* const kNMS_PLUGIN_NAMES[] = {"BatchedNMS_TRT", "BatchedNMSDynamic_TRT"};
} // namespace

template <>
void write<NMSParameters>(char*& buffer, NMSParameters const& val)
{
    auto* param = reinterpret_cast<NMSParameters*>(buffer);
    std::memset(param, 0, sizeof(NMSParameters));
    param->shareLocation = val.shareLocation;
    param->backgroundLabelId = val.backgroundLabelId;
    param->numClasses = val.numClasses;
    param->topK = val.topK;
    param->keepTopK = val.keepTopK;
    param->scoreThreshold = val.scoreThreshold;
    param->iouThreshold = val.iouThreshold;
    param->isNormalized = val.isNormalized;
    buffer += sizeof(NMSParameters);
}

static inline pluginStatus_t checkParams(NMSParameters const& param)
{
    // NMS plugin supports maximum thread blocksize of 512 and upto 8 blocks at once.
    constexpr int32_t maxTopK{512 * 8};
    if (param.topK > maxTopK)
    {
        plugin::gLogError << "Invalid parameter: NMS topK (" << param.topK << ") exceeds limit (" << maxTopK << ")"
                          << std::endl;
        return STATUS_BAD_PARAM;
    }

    return STATUS_SUCCESS;
}

BatchedNMSPlugin::BatchedNMSPlugin(NMSParameters params)
    : param(params)
{
    gLogWarning << "BatchedNMSPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addNMS() to add an "
                   "INMSLayer OR use EfficientNMS plugin."
                << std::endl;
    mPluginStatus = checkParams(param);
    PLUGIN_VALIDATE(mPluginStatus == STATUS_SUCCESS);
}

BatchedNMSPlugin::BatchedNMSPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    param = read<NMSParameters>(d);
    mBoxesSize = read<int32_t>(d);
    mScoresSize = read<int32_t>(d);
    mNumPriors = read<int32_t>(d);
    mClipBoxes = read<bool>(d);
    mPrecision = read<DataType>(d);
    mScoreBits = read<int32_t>(d);
    mCaffeSemantics = read<bool>(d);
    PLUGIN_VALIDATE(d == a + length);

    mPluginStatus = checkParams(param);
    PLUGIN_VALIDATE(mPluginStatus == STATUS_SUCCESS);
}

BatchedNMSDynamicPlugin::BatchedNMSDynamicPlugin(NMSParameters params)
    : param(params)
{
    mPluginStatus = checkParams(param);
    PLUGIN_VALIDATE(mPluginStatus == STATUS_SUCCESS);
}

BatchedNMSDynamicPlugin::BatchedNMSDynamicPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    param = read<NMSParameters>(d);
    mBoxesSize = read<int32_t>(d);
    mScoresSize = read<int32_t>(d);
    mNumPriors = read<int32_t>(d);
    mClipBoxes = read<bool>(d);
    mPrecision = read<DataType>(d);
    mScoreBits = read<int32_t>(d);
    mCaffeSemantics = read<bool>(d);
    PLUGIN_VALIDATE(d == a + length);

    mPluginStatus = checkParams(param);
    PLUGIN_VALIDATE(mPluginStatus == STATUS_SUCCESS);
}

int32_t BatchedNMSPlugin::getNbOutputs() const noexcept
{
    return 4;
}

int32_t BatchedNMSDynamicPlugin::getNbOutputs() const noexcept
{
    return 4;
}

int32_t BatchedNMSPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int32_t BatchedNMSDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void BatchedNMSPlugin::terminate() noexcept {}

void BatchedNMSDynamicPlugin::terminate() noexcept {}

Dims BatchedNMSPlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputDims == 2);
        PLUGIN_ASSERT(index >= 0 && index < this->getNbOutputs());
        PLUGIN_ASSERT(inputs[0].nbDims == 3);
        PLUGIN_ASSERT(inputs[1].nbDims == 2 || (inputs[1].nbDims == 3 && inputs[1].d[2] == 1));
        // mBoxesSize: number of box coordinates for one sample
        mBoxesSize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // mScoresSize: number of scores for one sample
        mScoresSize = inputs[1].d[0] * inputs[1].d[1];
        // num_detections
        if (index == 0)
        {
            Dims dim0{};
            dim0.nbDims = 0;
            return dim0;
        }
        // nmsed_boxes
        if (index == 1)
        {
            return Dims{2, {param.keepTopK, 4}};
        }
        // nmsed_scores or nmsed_classes
        Dims dim1{};
        dim1.nbDims = 1;
        dim1.d[0] = param.keepTopK;
        return dim1;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return Dims{};
}

DimsExprs BatchedNMSDynamicPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputs == 2);
        PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
        //           shareLocation ==              0               or          1
        // or
        // Dynamic shape: some dimension values may be -1
        PLUGIN_ASSERT(inputs[0].nbDims == 4);

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        // or
        // Dynamic shape: some dimension values may be -1
        PLUGIN_ASSERT(inputs[1].nbDims == 3 || inputs[1].nbDims == 4);

        DimsExprs out_dim;
        // num_detections
        if (outputIndex == 0)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(1);
        }
        // nmsed_boxes
        else if (outputIndex == 1)
        {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
            out_dim.d[2] = exprBuilder.constant(4);
        }
        // nmsed_scores, outputIndex == 2
        // nmsed_classes
        else
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
        }

        return out_dim;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

size_t BatchedNMSPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, mBoxesSize, mScoresSize, param.numClasses,
        mNumPriors, param.topK, mPrecision, mPrecision);
}

size_t BatchedNMSDynamicPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t batchSize = inputs[0].dims.d[0];
    int32_t boxesSize = inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
    int32_t scoreSize = inputs[1].dims.d[1] * inputs[1].dims.d[2];
    int32_t numPriors = inputs[0].dims.d[1];
    return detectionInferenceWorkspaceSize(param.shareLocation, batchSize, boxesSize, scoreSize, param.numClasses,
        numPriors, param.topK, mPrecision, mPrecision);
}

int32_t BatchedNMSPlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        void const* const locData = inputs[0];
        void const* const confData = inputs[1];

        if (mPluginStatus != STATUS_SUCCESS)
        {
            return -1;
        }

        void* keepCount = outputs[0];
        void* nmsedBoxes = outputs[1];
        void* nmsedScores = outputs[2];
        void* nmsedClasses = outputs[3];

        pluginStatus_t status = nmsInference(stream, batchSize, mBoxesSize, mScoresSize, param.shareLocation,
            param.backgroundLabelId, mNumPriors, param.numClasses, param.topK, param.keepTopK, param.scoreThreshold,
            param.iouThreshold, mPrecision, locData, mPrecision, confData, keepCount, nmsedBoxes, nmsedScores,
            nmsedClasses, workspace, param.isNormalized, false, mClipBoxes, mScoreBits, mCaffeSemantics);
        return status == STATUS_SUCCESS ? 0 : -1;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

int32_t BatchedNMSDynamicPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr && workspace != nullptr);

        void const* const locData = inputs[0];
        void const* const confData = inputs[1];

        if (mPluginStatus != STATUS_SUCCESS)
        {
            return -1;
        }

        void* keepCount = outputs[0];
        void* nmsedBoxes = outputs[1];
        void* nmsedScores = outputs[2];
        void* nmsedClasses = outputs[3];

        pluginStatus_t status = nmsInference(stream, inputDesc[0].dims.d[0], mBoxesSize, mScoresSize,
            param.shareLocation, param.backgroundLabelId, mNumPriors, param.numClasses, param.topK, param.keepTopK,
            param.scoreThreshold, param.iouThreshold, mPrecision, locData, mPrecision, confData, keepCount, nmsedBoxes,
            nmsedScores, nmsedClasses, workspace, param.isNormalized, false, mClipBoxes, mScoreBits, mCaffeSemantics);
        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

size_t BatchedNMSPlugin::getSerializationSize() const noexcept
{
    // NMSParameters, mBoxesSize,mScoresSize,mNumPriors
    return sizeof(NMSParameters) + sizeof(int32_t) * 3 + sizeof(bool) * 2 + sizeof(DataType) + sizeof(int32_t);
}

void BatchedNMSPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, mBoxesSize);
    write(d, mScoresSize);
    write(d, mNumPriors);
    write(d, mClipBoxes);
    write(d, mPrecision);
    write(d, mScoreBits);
    write(d, mCaffeSemantics);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

size_t BatchedNMSDynamicPlugin::getSerializationSize() const noexcept
{
    // NMSParameters, mBoxesSize,mScoresSize,mNumPriors
    return sizeof(NMSParameters) + sizeof(int32_t) * 3 + sizeof(bool) * 2 + sizeof(DataType) + sizeof(int32_t);
}

void BatchedNMSDynamicPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, mBoxesSize);
    write(d, mScoresSize);
    write(d, mNumPriors);
    write(d, mClipBoxes);
    write(d, mPrecision);
    write(d, mScoreBits);
    write(d, mCaffeSemantics);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void BatchedNMSPlugin::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims,
    int32_t nbOutputs, DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, nvinfer1::PluginFormat format, int32_t maxBatchSize) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputs == 2);
        PLUGIN_ASSERT(nbOutputs == 4);
        PLUGIN_ASSERT(inputDims[0].nbDims == 3);
        PLUGIN_ASSERT(inputDims[1].nbDims == 2 || (inputDims[1].nbDims == 3 && inputDims[1].d[2] == 1));
        PLUGIN_ASSERT(std::none_of(inputIsBroadcast, inputIsBroadcast + nbInputs, [](bool b) { return b; }));
        PLUGIN_ASSERT(std::none_of(outputIsBroadcast, outputIsBroadcast + nbInputs, [](bool b) { return b; }));

        mBoxesSize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
        mScoresSize = inputDims[1].d[0] * inputDims[1].d[1];
        // num_boxes
        mNumPriors = inputDims[0].d[0];
        const int32_t numLocClasses = param.shareLocation ? 1 : param.numClasses;
        // Third dimension of boxes must be either 1 or num_classes
        PLUGIN_ASSERT(inputDims[0].d[1] == numLocClasses);
        PLUGIN_ASSERT(inputDims[0].d[2] == 4);
        mPrecision = inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void BatchedNMSDynamicPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputs == 2);
        PLUGIN_ASSERT(nbOutputs == 4);

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
        //           shareLocation ==              0               or          1
        const int32_t numLocClasses = param.shareLocation ? 1 : param.numClasses;
        PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);
        PLUGIN_ASSERT(in[0].desc.dims.d[2] == numLocClasses);
        PLUGIN_ASSERT(in[0].desc.dims.d[3] == 4);

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        PLUGIN_ASSERT(in[1].desc.dims.nbDims == 3 || (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));

        mBoxesSize = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        mScoresSize = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
        // num_boxes
        mNumPriors = in[0].desc.dims.d[1];

        mPrecision = in[0].desc.type;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

bool BatchedNMSPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
#if NVBUG_3321606_WAR
    return ((type == DataType::kFLOAT || type == DataType::kINT32) && format == PluginFormat::kLINEAR);
#else
    return ((type == DataType::kHALF || type == DataType::kFLOAT || type == DataType::kINT32)
        && format == PluginFormat::kLINEAR);
#endif // NVBUG_3321606_WAR
}

bool BatchedNMSDynamicPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs <= 2 && nbInputs >= 0);
    PLUGIN_ASSERT(nbOutputs <= 4 && nbOutputs >= 0);
    PLUGIN_ASSERT(pos < 6 && pos >= 0);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;
    bool const consistentFloatPrecision = in[0].type == in[pos].type;
    switch (pos)
    {
    case 0:
        return (in[0].type == DataType::kHALF || in[0].type == DataType::kFLOAT)
            && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT)
            && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2: return out[0].type == DataType::kINT32 && out[0].format == PluginFormat::kLINEAR;
    case 3:
        return (out[1].type == DataType::kHALF || out[1].type == DataType::kFLOAT)
            && out[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 4:
        return (out[2].type == DataType::kHALF || out[2].type == DataType::kFLOAT)
            && out[2].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 5:
        return (out[3].type == DataType::kHALF || out[3].type == DataType::kFLOAT)
            && out[3].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    return false;
}

char const* BatchedNMSPlugin::getPluginType() const noexcept
{
    return kNMS_PLUGIN_NAMES[0];
}

char const* BatchedNMSDynamicPlugin::getPluginType() const noexcept
{
    return kNMS_PLUGIN_NAMES[1];
}

char const* BatchedNMSPlugin::getPluginVersion() const noexcept
{
    return kNMS_PLUGIN_VERSION;
}

char const* BatchedNMSDynamicPlugin::getPluginVersion() const noexcept
{
    return kNMS_PLUGIN_VERSION;
}

void BatchedNMSPlugin::destroy() noexcept
{
    delete this;
}

void BatchedNMSDynamicPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* BatchedNMSPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new BatchedNMSPlugin(param);
        plugin->mBoxesSize = mBoxesSize;
        plugin->mScoresSize = mScoresSize;
        plugin->mNumPriors = mNumPriors;
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setClipParam(mClipBoxes);
        plugin->mPrecision = mPrecision;
        plugin->setScoreBits(mScoreBits);
        plugin->setCaffeSemantics(mCaffeSemantics);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* BatchedNMSDynamicPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new BatchedNMSDynamicPlugin(param);
        plugin->mBoxesSize = mBoxesSize;
        plugin->mScoresSize = mScoresSize;
        plugin->mNumPriors = mNumPriors;
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setClipParam(mClipBoxes);
        plugin->mPrecision = mPrecision;
        plugin->setScoreBits(mScoreBits);
        plugin->setCaffeSemantics(mCaffeSemantics);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void BatchedNMSPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* BatchedNMSPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void BatchedNMSDynamicPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* BatchedNMSDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType BatchedNMSPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

nvinfer1::DataType BatchedNMSDynamicPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

void BatchedNMSPlugin::setClipParam(bool clip) noexcept
{
    mClipBoxes = clip;
}

void BatchedNMSDynamicPlugin::setClipParam(bool clip) noexcept
{
    mClipBoxes = clip;
}

void BatchedNMSPlugin::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

void BatchedNMSDynamicPlugin::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

void BatchedNMSPlugin::setCaffeSemantics(bool caffeSemantics) noexcept
{
    mCaffeSemantics = caffeSemantics;
}

void BatchedNMSDynamicPlugin::setCaffeSemantics(bool caffeSemantics) noexcept
{
    mCaffeSemantics = caffeSemantics;
}

bool BatchedNMSPlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

bool BatchedNMSPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

BatchedNMSBasePluginCreator::BatchedNMSBasePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreBits", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("caffeSemantics", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* BatchedNMSPluginCreator::getPluginName() const noexcept
{
    return kNMS_PLUGIN_NAMES[0];
}

char const* BatchedNMSDynamicPluginCreator::getPluginName() const noexcept
{
    return kNMS_PLUGIN_NAMES[1];
}

char const* BatchedNMSBasePluginCreator::getPluginVersion() const noexcept
{
    return kNMS_PLUGIN_VERSION;
}

PluginFieldCollection const* BatchedNMSBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* BatchedNMSPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning << "BatchedNMSPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addNMS() to add an "
                       "INMSLayer OR use EfficientNMS plugin."
                    << std::endl;
        NMSParameters params;
        PluginField const* fields = fc->fields;
        bool clipBoxes = true;
        int32_t scoreBits = 16;
        bool caffeSemantics = true;

        std::set<std::string> requiredFields{
            "shareLocation",
            "backgroundLabelId",
            "numClasses",
            "topK",
            "keepTopK",
            "scoreThreshold",
            "iouThreshold",
        };
        plugin::validateRequiredAttributesExist(requiredFields, fc);

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation"))
            {
                params.shareLocation = *(static_cast<bool const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "backgroundLabelId"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "numClasses"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "topK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.topK = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "keepTopK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.scoreThreshold = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "iouThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.iouThreshold = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "isNormalized"))
            {
                params.isNormalized = *(static_cast<bool const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "clipBoxes"))
            {
                clipBoxes = *(static_cast<bool const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreBits"))
            {
                scoreBits = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "caffeSemantics"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                caffeSemantics = *(static_cast<bool const*>(fields[i].data));
            }
        }

        auto* plugin = new BatchedNMSPlugin(params);
        plugin->setClipParam(clipBoxes);
        plugin->setScoreBits(scoreBits);
        plugin->setCaffeSemantics(caffeSemantics);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* BatchedNMSDynamicPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning << "BatchedNMSPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addNMS() to add an "
                       "INMSLayer OR use EfficientNMS plugin."
                    << std::endl;
        NMSParameters params;
        PluginField const* fields = fc->fields;
        bool clipBoxes = true;
        int32_t scoreBits = 16;
        bool caffeSemantics = true;

        std::set<std::string> requiredFields{
            "shareLocation",
            "backgroundLabelId",
            "numClasses",
            "topK",
            "keepTopK",
            "scoreThreshold",
            "iouThreshold",
        };
        plugin::validateRequiredAttributesExist(requiredFields, fc);

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation"))
            {
                params.shareLocation = *(static_cast<bool const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "backgroundLabelId"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "numClasses"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "topK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.topK = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "keepTopK"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.scoreThreshold = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "iouThreshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                params.iouThreshold = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "isNormalized"))
            {
                params.isNormalized = *(static_cast<bool const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "clipBoxes"))
            {
                clipBoxes = *(static_cast<bool const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreBits"))
            {
                scoreBits = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "caffeSemantics"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                caffeSemantics = *(static_cast<bool const*>(fields[i].data));
            }
        }

        auto* plugin = new BatchedNMSDynamicPlugin(params);
        plugin->setClipParam(clipBoxes);
        plugin->setScoreBits(scoreBits);
        plugin->setCaffeSemantics(caffeSemantics);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* BatchedNMSPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        gLogWarning << "BatchedNMSPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addNMS() to add an "
                       "INMSLayer OR use EfficientNMS plugin."
                    << std::endl;
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        auto* plugin = new BatchedNMSPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* BatchedNMSDynamicPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        gLogWarning << "BatchedNMSPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addNMS() to add an "
                       "INMSLayer OR use EfficientNMS plugin."
                    << std::endl;
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        auto* plugin = new BatchedNMSDynamicPlugin(serialData, serialLength);
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
