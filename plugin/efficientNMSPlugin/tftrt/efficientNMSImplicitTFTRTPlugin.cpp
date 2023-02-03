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

#include "efficientNMSImplicitTFTRTPlugin.h"
#include "efficientNMSPlugin/efficientNMSInference.h"

// This plugin provides CombinedNMS op compatibility for TF-TRT in Implicit Batch
// mode for legacy back-compatibilty

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::EfficientNMSParameters;
using nvinfer1::plugin::EfficientNMSImplicitTFTRTPlugin;
using nvinfer1::plugin::EfficientNMSImplicitTFTRTPluginCreator;

namespace
{
const char* EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_VERSION{"1"};
const char* EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_NAME{"EfficientNMS_Implicit_TF_TRT"};
} // namespace

EfficientNMSImplicitTFTRTPlugin::EfficientNMSImplicitTFTRTPlugin(EfficientNMSParameters param)
    : mParam(param)
{
}

EfficientNMSImplicitTFTRTPlugin::EfficientNMSImplicitTFTRTPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mParam = read<EfficientNMSParameters>(d);
    PLUGIN_ASSERT(d == a + length);
}

const char* EfficientNMSImplicitTFTRTPlugin::getPluginType() const noexcept
{
    return EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_NAME;
}

const char* EfficientNMSImplicitTFTRTPlugin::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_VERSION;
}

int EfficientNMSImplicitTFTRTPlugin::getNbOutputs() const noexcept
{
    return 4;
}

int EfficientNMSImplicitTFTRTPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void EfficientNMSImplicitTFTRTPlugin::terminate() noexcept {}

size_t EfficientNMSImplicitTFTRTPlugin::getSerializationSize() const noexcept
{
    return sizeof(EfficientNMSParameters);
}

void EfficientNMSImplicitTFTRTPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mParam);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void EfficientNMSImplicitTFTRTPlugin::destroy() noexcept
{
    delete this;
}

void EfficientNMSImplicitTFTRTPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* EfficientNMSImplicitTFTRTPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

Dims EfficientNMSImplicitTFTRTPlugin::getOutputDimensions(int outputIndex, const Dims* inputs, int nbInputs) noexcept
{
    try
    {
        Dims out_dim;

        // When pad per class is set, the output size may need to be reduced:
        // i.e.: outputBoxes = min(outputBoxes, outputBoxesPerClass * numClasses)
        PLUGIN_ASSERT(inputs[1].nbDims == 2);
        if (mParam.padOutputBoxesPerClass && mParam.numOutputBoxesPerClass > 0)
        {
            const int numClasses = inputs[1].d[1];
            if (mParam.numOutputBoxesPerClass * numClasses < mParam.numOutputBoxes)
            {
                mParam.numOutputBoxes = mParam.numOutputBoxesPerClass * numClasses;
            }
        }

        // Standard NMS
        PLUGIN_ASSERT(outputIndex >= 0 && outputIndex <= 3);

        // num_detections
        if (outputIndex == 0)
        {
            out_dim.nbDims = 0;
            out_dim.d[0] = 0;
        }
        // detection_boxes
        else if (outputIndex == 1)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = mParam.numOutputBoxes;
            out_dim.d[1] = 4;
        }
        // detection_scores
        else if (outputIndex == 2)
        {
            out_dim.nbDims = 1;
            out_dim.d[0] = mParam.numOutputBoxes;
        }
        // detection_classes
        else if (outputIndex == 3)
        {
            out_dim.nbDims = 1;
            out_dim.d[0] = mParam.numOutputBoxes;
        }

        return out_dim;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return Dims{};
}

size_t EfficientNMSImplicitTFTRTPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return EfficientNMSWorkspaceSize(maxBatchSize, mParam.numScoreElements, mParam.numClasses, mParam.datatype);
}

int EfficientNMSImplicitTFTRTPlugin::enqueue(int batchSize, void const* const* inputs,
    EfficientNMSImplicitTFTRTOutputsDataType outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        mParam.batchSize = batchSize;

        void const* const boxesInput = inputs[0];
        void const* const scoresInput = inputs[1];
        void const* const anchorsInput = nullptr;

        void* numDetectionsOutput = outputs[0];
        void* nmsBoxesOutput = outputs[1];
        void* nmsScoresOutput = outputs[2];
        void* nmsClassesOutput = outputs[3];

        return EfficientNMSInference(mParam, boxesInput, scoresInput, anchorsInput, numDetectionsOutput, nmsBoxesOutput,
            nmsScoresOutput, nmsClassesOutput, nullptr, workspace, stream);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

bool EfficientNMSImplicitTFTRTPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

DataType EfficientNMSImplicitTFTRTPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    // num_detections and detection_classes use integer outputs
    if (index == 0 || index == 3)
    {
        return DataType::kINT32;
    }
    // All others should use the same datatype as the input
    return inputTypes[0];
}

IPluginV2IOExt* EfficientNMSImplicitTFTRTPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new EfficientNMSImplicitTFTRTPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

bool EfficientNMSImplicitTFTRTPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, bool const* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool EfficientNMSImplicitTFTRTPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept
{
    if (inOut[pos].format != PluginFormat::kLINEAR)
    {
        return false;
    }

    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 4);
    if (nbInputs == 2)
    {
        PLUGIN_ASSERT(0 <= pos && pos <= 5);
    }

    // num_detections and detection_classes output: int
    const int posOut = pos - nbInputs;
    if (posOut == 0 || posOut == 3)
    {
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == PluginFormat::kLINEAR;
    }

    // all other inputs/outputs: fp32 or fp16
    return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT)
        && (inOut[0].type == inOut[pos].type);
}

void EfficientNMSImplicitTFTRTPlugin::configurePlugin(
    const PluginTensorDesc* in, int nbInputs, const PluginTensorDesc* out, int nbOutputs) noexcept
{
    try
    {
        // Inputs: [0] boxes, [1] scores
        PLUGIN_ASSERT(nbInputs == 2);
        PLUGIN_ASSERT(nbOutputs == 4);
        mParam.datatype = in[0].type;

        // Shape of scores input should be
        // [batch_size, num_boxes, num_classes] or [batch_size, num_boxes,
        // num_classes, 1]
        PLUGIN_ASSERT(in[1].dims.nbDims == 2 || (in[1].dims.nbDims == 3 && in[1].dims.d[2] == 1));
        mParam.numScoreElements = in[1].dims.d[0] * in[1].dims.d[1];
        mParam.numClasses = in[1].dims.d[1];

        // Shape of boxes input should be
        // [batch_size, num_boxes, 4] or [batch_size, num_boxes, 1, 4] or [batch_size,
        // num_boxes, num_classes, 4]
        PLUGIN_ASSERT(in[0].dims.nbDims == 2 || in[0].dims.nbDims == 3);
        if (in[0].dims.nbDims == 2)
        {
            PLUGIN_ASSERT(in[0].dims.d[1] == 4);
            mParam.shareLocation = true;
            mParam.numBoxElements = in[0].dims.d[0] * in[0].dims.d[1];
        }
        else
        {
            mParam.shareLocation = (in[0].dims.d[1] == 1);
            PLUGIN_ASSERT(in[0].dims.d[1] == mParam.numClasses || mParam.shareLocation);
            PLUGIN_ASSERT(in[0].dims.d[2] == 4);
            mParam.numBoxElements = in[0].dims.d[0] * in[0].dims.d[1] * in[0].dims.d[2];
        }
        mParam.numAnchors = in[0].dims.d[0];

        if (nbInputs == 2)
        {
            mParam.boxDecoder = false;
        }
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

EfficientNMSImplicitTFTRTPluginCreator::EfficientNMSImplicitTFTRTPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("max_output_size_per_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_total_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("pad_per_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clip_boxes", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientNMSImplicitTFTRTPluginCreator::getPluginName() const noexcept
{
    return EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_NAME;
}

const char* EfficientNMSImplicitTFTRTPluginCreator::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_VERSION;
}

const PluginFieldCollection* EfficientNMSImplicitTFTRTPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2IOExt* EfficientNMSImplicitTFTRTPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "max_output_size_per_class"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxesPerClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_total_size"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_threshold"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "pad_per_class"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.padOutputBoxesPerClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "clip_boxes"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.clipBoxes = *(static_cast<const int*>(fields[i].data));
            }
        }

        auto* plugin = new EfficientNMSImplicitTFTRTPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2IOExt* EfficientNMSImplicitTFTRTPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientNMSImplicitTFTRTPlugin::destroy()
        auto* plugin = new EfficientNMSImplicitTFTRTPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
