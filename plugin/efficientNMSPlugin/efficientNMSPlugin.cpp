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

#include "efficientNMSPlugin.h"
#include "efficientNMSInference.h"

using namespace nvinfer1;
using nvinfer1::plugin::EfficientNMSPlugin;
using nvinfer1::plugin::EfficientNMSParameters;
using nvinfer1::plugin::EfficientNMSPluginCreator;
using nvinfer1::plugin::EfficientNMSONNXPluginCreator;

namespace
{
const char* EFFICIENT_NMS_PLUGIN_VERSION{"1"};
const char* EFFICIENT_NMS_PLUGIN_NAME{"EfficientNMS_TRT"};
const char* EFFICIENT_NMS_ONNX_PLUGIN_VERSION{"1"};
const char* EFFICIENT_NMS_ONNX_PLUGIN_NAME{"EfficientNMS_ONNX_TRT"};
} // namespace

PluginFieldCollection EfficientNMSPluginCreator::mFC{};
PluginFieldCollection EfficientNMSONNXPluginCreator::mFC{};
std::vector<PluginField> EfficientNMSPluginCreator::mPluginAttributes;
std::vector<PluginField> EfficientNMSONNXPluginCreator::mPluginAttributes;

EfficientNMSPlugin::EfficientNMSPlugin(EfficientNMSParameters param)
    : mParam(param)
{
}

EfficientNMSPlugin::EfficientNMSPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mParam = read<EfficientNMSParameters>(d);
    ASSERT(d == a + length);
}

const char* EfficientNMSPlugin::getPluginType() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_NAME;
}

const char* EfficientNMSPlugin::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_VERSION;
}

int EfficientNMSPlugin::getNbOutputs() const noexcept
{
    if (mParam.outputONNXIndices)
    {
        // ONNX NonMaxSuppression Compatibility
        return 1;
    }
    else
    {
        // Standard Plugin Implementation
        return 4;
    }
}

int EfficientNMSPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void EfficientNMSPlugin::terminate() noexcept {}

size_t EfficientNMSPlugin::getSerializationSize() const noexcept
{
    return sizeof(EfficientNMSParameters);
}

void EfficientNMSPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mParam);
    ASSERT(d == a + getSerializationSize());
}

void EfficientNMSPlugin::destroy() noexcept
{
    delete this;
}

void EfficientNMSPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
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

const char* EfficientNMSPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType EfficientNMSPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if (mParam.outputONNXIndices)
    {
        // ONNX NMS uses an integer output
        return nvinfer1::DataType::kINT32;
    }
    else
    {
        // On standard NMS, num_detections and detection_classes use integer outputs
        if (index == 0 || index == 3)
        {
            return nvinfer1::DataType::kINT32;
        }
        // All others should use the same datatype as the input
        return inputTypes[0];
    }
}

IPluginV2DynamicExt* EfficientNMSPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new EfficientNMSPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EfficientNMSPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        DimsExprs out_dim;

        if (mParam.outputONNXIndices)
        {
            // ONNX NMS
            ASSERT(outputIndex == 0);

            // detection_indices
            out_dim.nbDims = 2;
            out_dim.d[0] = exprBuilder.operation(
                DimensionOperation::kPROD, *inputs[0].d[0], *exprBuilder.constant(mParam.numOutputBoxes));
            out_dim.d[1] = exprBuilder.constant(3);
        }
        else
        {
            // Standard NMS
            ASSERT(outputIndex >= 0 && outputIndex <= 3);

            // num_detections
            if (outputIndex == 0)
            {
                out_dim.nbDims = 2;
                out_dim.d[0] = inputs[0].d[0];
                out_dim.d[1] = exprBuilder.constant(1);
            }
            // detection_boxes
            else if (outputIndex == 1)
            {
                out_dim.nbDims = 3;
                out_dim.d[0] = inputs[0].d[0];
                out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
                out_dim.d[2] = exprBuilder.constant(4);
            }
            // detection_scores
            else if (outputIndex == 2)
            {
                out_dim.nbDims = 2;
                out_dim.d[0] = inputs[0].d[0];
                out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
            }
            // detection_classes
            else if (outputIndex == 3)
            {
                out_dim.nbDims = 2;
                out_dim.d[0] = inputs[0].d[0];
                out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
            }
        }

        return out_dim;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool EfficientNMSPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (inOut[pos].format != PluginFormat::kLINEAR)
    {
        return false;
    }

    if (mParam.outputONNXIndices)
    {
        ASSERT(nbInputs == 2);
        ASSERT(nbOutputs == 1);

        // detection_indices output: int
        if (pos == 2)
        {
            return inOut[pos].type == DataType::kINT32;
        }

        // boxes and scores input: fp32 or fp16
        return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT)
            && (inOut[0].type == inOut[pos].type);
    }
    else
    {
        ASSERT(nbInputs == 2 || nbInputs == 3);
        ASSERT(nbOutputs == 4);
        if (nbInputs == 2)
        {
            ASSERT(0 <= pos && pos <= 5);
        }
        if (nbInputs == 3)
        {
            ASSERT(0 <= pos && pos <= 6);
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
}

void EfficientNMSPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    try
    {
        if (mParam.outputONNXIndices)
        {
            // Accepts two inputs
            // [0] boxes, [1] scores
            ASSERT(nbInputs == 2);
            ASSERT(nbOutputs == 1);
        }
        else
        {
            // Accepts two or three inputs
            // If two inputs: [0] boxes, [1] scores
            // If three inputs: [0] boxes, [1] scores, [2] anchors
            ASSERT(nbInputs == 2 || nbInputs == 3);
            ASSERT(nbOutputs == 4);
        }
        mParam.datatype = in[0].desc.type;

        // Shape of scores input should be
        // [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        ASSERT(in[1].desc.dims.nbDims == 3 || (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));
        mParam.numScoreElements = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
        mParam.numClasses = in[1].desc.dims.d[2];

        // Shape of boxes input should be
        // [batch_size, num_boxes, 4] or [batch_size, num_boxes, 1, 4] or [batch_size, num_boxes, num_classes, 4]
        ASSERT(in[0].desc.dims.nbDims == 3 || in[0].desc.dims.nbDims == 4);
        if (in[0].desc.dims.nbDims == 3)
        {
            ASSERT(in[0].desc.dims.d[2] == 4);
            mParam.shareLocation = true;
            mParam.numBoxElements = in[0].desc.dims.d[1] * in[0].desc.dims.d[2];
        }
        else
        {
            mParam.shareLocation = (in[0].desc.dims.d[2] == 1);
            ASSERT(in[0].desc.dims.d[2] == mParam.numClasses || mParam.shareLocation);
            ASSERT(in[0].desc.dims.d[3] == 4);
            mParam.numBoxElements = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        }
        mParam.numAnchors = in[0].desc.dims.d[1];

        if (nbInputs == 2)
        {
            // Only two inputs are used, disable the fused box decoder
            mParam.boxDecoder = false;
        }
        if (nbInputs == 3)
        {
            // All three inputs are used, enable the box decoder
            // Shape of anchors input should be
            // Constant shape: [1, numAnchors, 4] or [batch_size, numAnchors, 4]
            ASSERT(in[2].desc.dims.nbDims == 3);
            mParam.boxDecoder = true;
            mParam.shareAnchors = (in[2].desc.dims.d[0] == 1);
        }
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

size_t EfficientNMSPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[1].dims.d[0];
    int numScoreElements = inputs[1].dims.d[1] * inputs[1].dims.d[2];
    int numClasses = inputs[1].dims.d[2];
    return EfficientNMSWorkspaceSize(batchSize, numScoreElements, numClasses, mParam.datatype);
}

int EfficientNMSPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        mParam.batchSize = inputDesc[0].dims.d[0];

        if (mParam.outputONNXIndices)
        {
            // ONNX NonMaxSuppression Op Support
            const void* const boxesInput = inputs[0];
            const void* const scoresInput = inputs[1];

            void* nmsIndicesOutput = outputs[0];

            return EfficientNMSInference(mParam, boxesInput, scoresInput, nullptr, nullptr, nullptr, nullptr, nullptr,
                nmsIndicesOutput, workspace, stream);
        }
        else
        {
            // Standard NMS Operation
            const void* const boxesInput = inputs[0];
            const void* const scoresInput = inputs[1];
            const void* const anchorsInput = mParam.boxDecoder ? inputs[2] : nullptr;

            void* numDetectionsOutput = outputs[0];
            void* nmsBoxesOutput = outputs[1];
            void* nmsScoresOutput = outputs[2];
            void* nmsClassesOutput = outputs[3];

            return EfficientNMSInference(mParam, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
                nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, nullptr, workspace, stream);
        }
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

EfficientNMSPluginCreator::EfficientNMSPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_output_boxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("background_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_activation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("box_coding", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientNMSPluginCreator::getPluginName() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_NAME;
}

const char* EfficientNMSPluginCreator::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* EfficientNMSPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* EfficientNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "score_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_output_boxes"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "background_class"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.backgroundClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_activation"))
            {
                mParam.scoreSigmoid = *(static_cast<const bool*>(fields[i].data));
            }
            if (!strcmp(attrName, "box_coding"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.boxCoding = *(static_cast<const int*>(fields[i].data));
            }
        }

        auto* plugin = new EfficientNMSPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EfficientNMSPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientNMSPlugin::destroy()
        auto* plugin = new EfficientNMSPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

EfficientNMSONNXPluginCreator::EfficientNMSONNXPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_output_boxes_per_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("center_point_box", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientNMSONNXPluginCreator::getPluginName() const noexcept
{
    return EFFICIENT_NMS_ONNX_PLUGIN_NAME;
}

const char* EfficientNMSONNXPluginCreator::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_ONNX_PLUGIN_VERSION;
}

const PluginFieldCollection* EfficientNMSONNXPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* EfficientNMSONNXPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "score_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_output_boxes_per_class"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxesPerClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "center_point_box"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.boxCoding = *(static_cast<const int*>(fields[i].data));
            }
        }

        // This enables ONNX compatibility mode
        mParam.outputONNXIndices = true;
        mParam.numOutputBoxes = mParam.numOutputBoxesPerClass;

        auto* plugin = new EfficientNMSPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EfficientNMSONNXPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientNMSPlugin::destroy()
        auto* plugin = new EfficientNMSPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
