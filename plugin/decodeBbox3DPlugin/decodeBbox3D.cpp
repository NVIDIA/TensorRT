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

#include "decodeBbox3D.h"
#include <cstring>
#include <iostream>

#define checkCudaErrors(status)                                                                                        \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ << " in file "      \
                      << __FILE__ << " error status: " << status << std::endl;                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    }

using namespace nvinfer1;
using nvinfer1::plugin::DecodeBbox3DPlugin;
using nvinfer1::plugin::DecodeBbox3DPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"DecodeBbox3DPlugin"};

// Static class fields initialization
PluginFieldCollection DecodeBbox3DPluginCreator::mFC{};
std::vector<PluginField> DecodeBbox3DPluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

DecodeBbox3DPlugin::DecodeBbox3DPlugin(float x_min, float x_max, float y_min, float y_max,
            float z_min, float z_max,
            int num_dir_bins, float dir_offset, float dir_limit_offset,
            const std::vector<float>& anchor_bottom_height,
            const std::vector<float>& anchors,
            float score_thresh
) : min_x_range_(x_min), max_x_range_(x_max), min_y_range_(y_min),
    max_y_range_(y_max), min_z_range_(z_min), max_z_range_(z_max),
    num_dir_bins_(num_dir_bins), dir_offset_(dir_offset),
    dir_limit_offset_(dir_limit_offset), score_thresh_(score_thresh)
{
    anchor_bottom_height_.clear();
    for(size_t i = 0; i < anchor_bottom_height.size(); i++)
        anchor_bottom_height_.push_back(anchor_bottom_height[i]);
    anchors_.clear();
    for(size_t i = 0; i < anchors.size(); i++)
        anchors_.push_back(anchors[i]);
    num_classes_ = int(anchor_bottom_height_.size());
    PLUGIN_VALIDATE(num_classes_ > 0);
    PLUGIN_VALIDATE(static_cast<size_t>(num_classes_) * 2 * 4 == anchors_.size());
}

DecodeBbox3DPlugin::DecodeBbox3DPlugin(float x_min, float x_max, float y_min, float y_max,
            float z_min, float z_max,
            int num_dir_bins, float dir_offset, float dir_limit_offset,
            const std::vector<float>& anchor_bottom_height,
            const std::vector<float>& anchors,
            float score_thresh,
            int feature_h, int feature_w
) : min_x_range_(x_min), max_x_range_(x_max), min_y_range_(y_min),
    max_y_range_(y_max), min_z_range_(z_min), max_z_range_(z_max),
    num_dir_bins_(num_dir_bins), dir_offset_(dir_offset),
    dir_limit_offset_(dir_limit_offset),
    score_thresh_(score_thresh), feature_h_(feature_h),
    feature_w_(feature_w)
{
    anchor_bottom_height_.clear();
    for(size_t i = 0; i < anchor_bottom_height.size(); i++)
        anchor_bottom_height_.push_back(anchor_bottom_height[i]);
    anchors_.clear();
    for(size_t i = 0; i < anchors.size(); i++)
        anchors_.push_back(anchors[i]);
    num_classes_ = int(anchor_bottom_height_.size());
    PLUGIN_VALIDATE(num_classes_ > 0);
    PLUGIN_VALIDATE(static_cast<size_t>(num_classes_) * 2 * 4 == anchors_.size());
}

DecodeBbox3DPlugin::DecodeBbox3DPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    min_x_range_ = readFromBuffer<float>(d);
    max_x_range_ = readFromBuffer<float>(d);
    min_y_range_ = readFromBuffer<float>(d);
    max_y_range_ = readFromBuffer<float>(d);
    min_z_range_ = readFromBuffer<float>(d);
    max_z_range_ = readFromBuffer<float>(d);
    num_dir_bins_ = readFromBuffer<int>(d);
    dir_offset_ = readFromBuffer<float>(d);
    dir_limit_offset_ = readFromBuffer<float>(d);
    score_thresh_ = readFromBuffer<float>(d);
    num_classes_ = readFromBuffer<int>(d);
    feature_h_ = readFromBuffer<int>(d);
    feature_w_ = readFromBuffer<int>(d);
    anchor_bottom_height_.clear();
    anchors_.clear();
    for(int i = 0; i < num_classes_; i++)
        anchor_bottom_height_.push_back(readFromBuffer<float>(d));
    for(int i = 0; i < num_classes_ * 2 * 4; i++)
        anchors_.push_back(readFromBuffer<float>(d));
}

nvinfer1::IPluginV2DynamicExt* DecodeBbox3DPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new DecodeBbox3DPlugin(min_x_range_, max_x_range_, min_y_range_, max_y_range_, min_z_range_,
            max_z_range_, num_dir_bins_, dir_offset_, dir_limit_offset_, anchor_bottom_height_, anchors_, score_thresh_,
            feature_h_, feature_w_);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs DecodeBbox3DPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(this->getNbOutputs() == 2);
    PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto feature_h = inputs[0].d[1];
    auto feature_w = inputs[0].d[2];
    auto batch_size = inputs[0].d[0];
    if (outputIndex == 0)
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.operation(
            nvinfer1::DimensionOperation::kPROD,
            feature_h[0],
            exprBuilder.operation(
                nvinfer1::DimensionOperation::kPROD,
                feature_w[0],
                exprBuilder.constant(num_classes_ * 2)[0]
            )[0]
        );
        dim0.d[2] = exprBuilder.constant(9);
        return dim0;
    }
    nvinfer1::DimsExprs dim1{};
    dim1.nbDims = 1;
    dim1.d[0] = batch_size;
    return dim1;
}

bool DecodeBbox3DPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // cls_preds
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // box_preds
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // dir_cls_preds
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // boxes
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       // box_num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void DecodeBbox3DPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    feature_h_ = in[0].desc.dims.d[1];
    feature_w_ = in[0].desc.dims.d[2];
}

size_t DecodeBbox3DPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    size_t anchors_size = num_classes_ * 2 * 4 * sizeof(float);
    size_t anchor_bottom_height_size = num_classes_ * sizeof(float);
    size_t workspaces[2];
    workspaces[0] = anchors_size;
    workspaces[1] = anchor_bottom_height_size;
    return calculateTotalWorkspaceSize(workspaces, 2);
}

int DecodeBbox3DPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // Inputs
    float *cls_input = const_cast<float *>((const float*) (inputs[0]));
    float* box_input = const_cast<float *>((const float *)inputs[1]);
    float *dir_cls_input = const_cast<float *>((const float*) (inputs[2]));
    // Outputs
    float* bndbox_output = (float*) (outputs[0]);
    int* box_num = (int*) (outputs[1]);
    // Initialize workspaces
    float* anchors = (float*) workspace;
    size_t anchors_size = num_classes_ * 2 * 4 * sizeof(float);
    float* anchor_bottom_height = (float*)nextWorkspacePtr((int8_t*)anchors, anchors_size);
    size_t anchor_bottom_height_size = num_classes_ * sizeof(float);
    checkCudaErrors(
        cudaMemcpyAsync(
            anchors,
            &anchors_[0],
            anchors_size,
            cudaMemcpyHostToDevice,
            stream
        )
    );
    checkCudaErrors(
        cudaMemcpyAsync(
            anchor_bottom_height,
            &anchor_bottom_height_[0],
            anchor_bottom_height_size,
            cudaMemcpyHostToDevice,
            stream
        )
    );
    // Initialize box_num to 0
    checkCudaErrors(cudaMemsetAsync(box_num, 0, batchSize * sizeof(int), stream));
    decodeBbox3DLaunch(
        batchSize,
        cls_input,
        box_input,
        dir_cls_input,
        anchors,
        anchor_bottom_height,
        bndbox_output,
        box_num,
        min_x_range_,
        max_x_range_,
        min_y_range_,
        max_y_range_,
        feature_w_,
        feature_h_,
        num_classes_ * 2,
        num_classes_,
        7,
        score_thresh_,
        dir_offset_,
        dir_limit_offset_,
        num_dir_bins_,
        stream
    );
    return 0;
}

nvinfer1::DataType DecodeBbox3DPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if(index == 0)
      return inputTypes[0];
    return nvinfer1::DataType::kINT32;
}

const char* DecodeBbox3DPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* DecodeBbox3DPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int DecodeBbox3DPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int DecodeBbox3DPlugin::initialize() noexcept
{
    return 0;
}

void DecodeBbox3DPlugin::terminate() noexcept
{
}

size_t DecodeBbox3DPlugin::getSerializationSize() const noexcept
{
    size_t scalar_size = 9 * sizeof(float) + 4 * sizeof(int);
    size_t vector_size = num_classes_ * 9 * sizeof(float);
    return scalar_size + vector_size;
}

void DecodeBbox3DPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<float>(d, min_x_range_);
    writeToBuffer<float>(d, max_x_range_);
    writeToBuffer<float>(d, min_y_range_);
    writeToBuffer<float>(d, max_y_range_);
    writeToBuffer<float>(d, min_z_range_);
    writeToBuffer<float>(d, max_z_range_);
    writeToBuffer<int>(d, num_dir_bins_);
    writeToBuffer<float>(d, dir_offset_);
    writeToBuffer<float>(d, dir_limit_offset_);
    writeToBuffer<float>(d, score_thresh_);
    writeToBuffer<int>(d, num_classes_);
    writeToBuffer<int>(d, feature_h_);
    writeToBuffer<int>(d, feature_w_);
    for(int i = 0; i < num_classes_; i++)
        writeToBuffer<float>(d, anchor_bottom_height_[i]);
    for(int i = 0; i < num_classes_ * 2 * 4; i++)
        writeToBuffer<float>(d, anchors_[i]);
}

void DecodeBbox3DPlugin::destroy() noexcept
{
    delete this;
}

void DecodeBbox3DPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* DecodeBbox3DPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DecodeBbox3DPluginCreator::DecodeBbox3DPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("point_cloud_range", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchors", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchor_bottom_height", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("dir_offset", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("dir_limit_offset", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_dir_bins", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_thresh", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DecodeBbox3DPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* DecodeBbox3DPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* DecodeBbox3DPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* DecodeBbox3DPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;
        float point_cloud_range[6] = {0.0f};
        std::vector<float> anchors{};
        std::vector<float> anchor_bottom_height{};
        float dir_offset = 0.78539f;
        float dir_limit_offset = 0.0f;
        int num_dir_bins = 2;
        float score_thresh = 0.1f;
        for (int i = 0; i < nbFields; ++i)
        {
            const char* attr_name = fields[i].name;
            if (!strcmp(attr_name, "point_cloud_range"))
            {
                const float* d = static_cast<const float*>(fields[i].data);
                point_cloud_range[0] = d[0];
                point_cloud_range[1] = d[1];
                point_cloud_range[2] = d[2];
                point_cloud_range[3] = d[3];
                point_cloud_range[4] = d[4];
                point_cloud_range[5] = d[5];
            }
            else if (!strcmp(attr_name, "anchors"))
            {
                const float* as = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < fields[i].length; ++j)
                {
                    anchors.push_back(*as);
                    ++as;
                }
            }
            else if (!strcmp(attr_name, "anchor_bottom_height"))
            {
                const float* ah = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < fields[i].length; ++j)
                {
                    anchor_bottom_height.push_back(*ah);
                    ++ah;
                }
            }
            else if (!strcmp(attr_name, "dir_offset"))
            {
                const float* d = static_cast<const float*>(fields[i].data);
                dir_offset = d[0];
            }
            else if (!strcmp(attr_name, "dir_limit_offset"))
            {
                const float* d = static_cast<const float*>(fields[i].data);
                dir_limit_offset = d[0];
            }
            else if (!strcmp(attr_name, "num_dir_bins"))
            {
                const int* d = static_cast<const int*>(fields[i].data);
                num_dir_bins = d[0];
            }
            else if (!strcmp(attr_name, "score_thresh"))
            {
                const float* d = static_cast<const float*>(fields[i].data);
                score_thresh = d[0];
            }
        }
        IPluginV2* plugin = new DecodeBbox3DPlugin(point_cloud_range[0], point_cloud_range[3], point_cloud_range[1],
            point_cloud_range[4], point_cloud_range[2], point_cloud_range[5], num_dir_bins, dir_offset,
            dir_limit_offset, anchor_bottom_height, anchors, score_thresh);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* DecodeBbox3DPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        return new DecodeBbox3DPlugin(serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void DecodeBbox3DPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* DecodeBbox3DPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
