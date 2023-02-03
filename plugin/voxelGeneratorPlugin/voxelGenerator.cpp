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

#include "voxelGenerator.h"
#include <cstring>
#include <iostream>

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

using namespace nvinfer1;
using nvinfer1::plugin::VoxelGeneratorPlugin;
using nvinfer1::plugin::VoxelGeneratorPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"VoxelGeneratorPlugin"};

// Static class fields initialization
PluginFieldCollection VoxelGeneratorPluginCreator::mFC{};
std::vector<PluginField> VoxelGeneratorPluginCreator::mPluginAttributes;

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

// Mimic np.round as in voxel generator in spconv implementation
int np_round(float x) {
  // half way round to nearest-even
  int x2 = int(x * 2.0f);
  if(x != int(x) && x2 == x * 2.0f) {
    return int(x / 2.0f + 0.5f) * 2;
  }
  return int(x + 0.5f);
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(
    int max_voxels, int max_points, int voxel_features, float x_min,
    float x_max, float y_min, float y_max, float z_min, float z_max,
    float pillar_x, float pillar_y, float pillar_z
) : pillarNum_(max_voxels), pointNum_(max_points), featureNum_(voxel_features),
    min_x_range_(x_min), max_x_range_(x_max), min_y_range_(y_min),
    max_y_range_(y_max), min_z_range_(z_min), max_z_range_(z_max),
    pillar_x_size_(pillar_x), pillar_y_size_(pillar_y),
    pillar_z_size_(pillar_z)
{
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(
    int max_voxels, int max_points, int voxel_features, float x_min,
    float x_max, float y_min, float y_max, float z_min, float z_max,
    float pillar_x, float pillar_y, float pillar_z, int point_features,
    int grid_x, int grid_y, int grid_z
) : pillarNum_(max_voxels), pointNum_(max_points), featureNum_(voxel_features),
    min_x_range_(x_min), max_x_range_(x_max), min_y_range_(y_min),
    max_y_range_(y_max), min_z_range_(z_min), max_z_range_(z_max),
    pillar_x_size_(pillar_x), pillar_y_size_(pillar_y),
    pillar_z_size_(pillar_z), pointFeatureNum_(point_features),
    grid_x_size_(grid_x), grid_y_size_(grid_y), grid_z_size_(grid_z)
{
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    pillarNum_ = readFromBuffer<int>(d);
    pointNum_ = readFromBuffer<int>(d);
    featureNum_ = readFromBuffer<int>(d);
    min_x_range_ = readFromBuffer<float>(d);
    max_x_range_ = readFromBuffer<float>(d);
    min_y_range_ = readFromBuffer<float>(d);
    max_y_range_ = readFromBuffer<float>(d);
    min_z_range_ = readFromBuffer<float>(d);
    max_z_range_ = readFromBuffer<float>(d);
    pillar_x_size_ = readFromBuffer<float>(d);
    pillar_y_size_ = readFromBuffer<float>(d);
    pillar_z_size_ = readFromBuffer<float>(d);
    pointFeatureNum_ = readFromBuffer<int>(d);
    grid_x_size_ = readFromBuffer<int>(d);
    grid_y_size_ = readFromBuffer<int>(d);
    grid_z_size_ = readFromBuffer<int>(d);
}

nvinfer1::IPluginV2DynamicExt* VoxelGeneratorPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new VoxelGeneratorPlugin(pillarNum_, pointNum_, featureNum_, min_x_range_, max_x_range_,
            min_y_range_, max_y_range_, min_z_range_, max_z_range_, pillar_x_size_, pillar_y_size_, pillar_z_size_,
            pointFeatureNum_, grid_x_size_, grid_y_size_, grid_z_size_);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs VoxelGeneratorPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    if (outputIndex == 0)
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 4;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(pillarNum_);
        dim0.d[2] = exprBuilder.constant(pointNum_);
        dim0.d[3] = exprBuilder.constant(featureNum_);
        return dim0;
    }
    if(outputIndex == 1){
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 3;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(pillarNum_);
        dim1.d[2] = exprBuilder.constant(4);
        return dim1;
    }
    nvinfer1::DimsExprs dim2{};
    dim2.nbDims = 1;
    dim2.d[0] = batch_size;
    return dim2;
}

bool VoxelGeneratorPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 3);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // PointCloud Array --- x, y, z, w
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // Point Num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // features, dim: pillarNum x pointNum x featureNum
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // pillarCoords, dim: 1 x 1 x pillarNum x 4
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       // params, dim: 1 x 1 x 1 x 1
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void VoxelGeneratorPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    pointFeatureNum_ = in[0].desc.dims.d[2];
    grid_x_size_ = np_round((max_x_range_ - min_x_range_) / pillar_x_size_);
    grid_y_size_ = np_round((max_y_range_ - min_y_range_) / pillar_y_size_);
    grid_z_size_ = np_round((max_z_range_ - min_z_range_) / pillar_z_size_);
}

size_t VoxelGeneratorPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[0].dims.d[0];
    size_t mask_size = batchSize * grid_z_size_ * grid_y_size_ * grid_x_size_ * sizeof(unsigned int);
    size_t voxels_size = batchSize * grid_z_size_ * grid_y_size_ * grid_x_size_ * pointNum_ * pointFeatureNum_
                   * sizeof(float);
    // the actual max pillar num cannot be determined, use upper bound
    size_t voxel_features_size = voxels_size;
    size_t voxel_num_points_size = mask_size;
    size_t workspaces[4];
    workspaces[0] = mask_size;
    workspaces[1] = voxels_size;
    workspaces[2] = voxel_features_size;
    workspaces[3] = voxel_num_points_size;
    return calculateTotalWorkspaceSize(workspaces, 4);
}

int VoxelGeneratorPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    float * pointCloud = const_cast<float *>((const float *)inputs[0]);
    unsigned int* pointNum = const_cast<unsigned int *>((const unsigned int *)inputs[1]);
    //TRT-output
    float *pillar_features_data = (float *)(outputs[0]);
    unsigned int *coords_data = (unsigned int *)(outputs[1]);
    unsigned int *params_data = (unsigned int *)(outputs[2]);
    int dense_pillar_num = grid_z_size_ * grid_y_size_ * grid_x_size_;
    size_t mask_size = batchSize * dense_pillar_num * sizeof(unsigned int);
    size_t voxels_size = batchSize * dense_pillar_num * pointNum_ * pointFeatureNum_
                   * sizeof(float);
    size_t voxel_features_size = voxels_size;
    size_t voxel_num_points_size = mask_size;
    size_t workspaces[4];
    workspaces[0] = mask_size;
    workspaces[1] = voxels_size;
    workspaces[2] = voxel_features_size;
    workspaces[3] = voxel_num_points_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 4);
    unsigned int* mask_ = static_cast<unsigned int*>(workspace);
    float* voxels_ = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(mask_), mask_size)
    );
    float* voxel_features_ = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(voxels_), voxels_size)
    );
    unsigned int* voxel_num_points_ = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(voxel_features_), voxel_features_size)
    );
    // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(mask_, 0, total_workspace, stream));
    unsigned int pillar_features_data_size = batchSize * pillarNum_ * pointNum_ * featureNum_ * sizeof(float);
    unsigned int coords_data_size = batchSize * pillarNum_ * 4 * sizeof(unsigned int);
    unsigned int params_data_size = batchSize * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(pillar_features_data, 0, pillar_features_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(coords_data, 0, coords_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(params_data, 0, params_data_size, stream));
    // pointcloud + pointNum ---> mask_ + voxel_
    generateVoxels_launch(
          batchSize, maxNumPoints,
          pointCloud, pointNum,
          min_x_range_, max_x_range_,
          min_y_range_, max_y_range_,
          min_z_range_, max_z_range_,
          pillar_x_size_, pillar_y_size_, pillar_z_size_,
          grid_y_size_, grid_x_size_, pointFeatureNum_,
          pointNum_, mask_, voxels_, stream);
    // mask_ + voxel_ ---> params_data + voxel_features_ + voxel_num_points_ + coords_data
    generateBaseFeatures_launch(
        batchSize,
        mask_, voxels_,
        grid_y_size_, grid_x_size_,
        params_data,
        pillarNum_,
        pointNum_,
        pointFeatureNum_,
        voxel_features_,
        voxel_num_points_,
        coords_data, stream);
    generateFeatures_launch(
        batchSize,
        dense_pillar_num,
        voxel_features_,
        voxel_num_points_,
        coords_data,
        params_data,
        pillar_x_size_, pillar_y_size_, pillar_z_size_,
        min_x_range_, min_y_range_, min_z_range_,
        featureNum_, pointNum_, pillarNum_, pointFeatureNum_,
        pillar_features_data, stream);
    return 0;
}

nvinfer1::DataType VoxelGeneratorPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if(index == 0)
      return inputTypes[0];
    return inputTypes[1];
}

const char* VoxelGeneratorPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* VoxelGeneratorPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int VoxelGeneratorPlugin::getNbOutputs() const noexcept
{
    return 3;
}

int VoxelGeneratorPlugin::initialize() noexcept
{
    return 0;
}

void VoxelGeneratorPlugin::terminate() noexcept
{
}

size_t VoxelGeneratorPlugin::getSerializationSize() const noexcept
{
    return 9 * sizeof(float) + 7 * sizeof(int);
}

void VoxelGeneratorPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, pillarNum_);
    writeToBuffer<int>(d, pointNum_);
    writeToBuffer<int>(d, featureNum_);
    writeToBuffer<float>(d, min_x_range_);
    writeToBuffer<float>(d, max_x_range_);
    writeToBuffer<float>(d, min_y_range_);
    writeToBuffer<float>(d, max_y_range_);
    writeToBuffer<float>(d, min_z_range_);
    writeToBuffer<float>(d, max_z_range_);
    writeToBuffer<float>(d, pillar_x_size_);
    writeToBuffer<float>(d, pillar_y_size_);
    writeToBuffer<float>(d, pillar_z_size_);
    writeToBuffer<int>(d, pointFeatureNum_);
    writeToBuffer<int>(d, grid_x_size_);
    writeToBuffer<int>(d, grid_y_size_);
    writeToBuffer<int>(d, grid_z_size_);
}

void VoxelGeneratorPlugin::destroy() noexcept
{
    delete this;
}

void VoxelGeneratorPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* VoxelGeneratorPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

VoxelGeneratorPluginCreator::VoxelGeneratorPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("max_num_points_per_voxel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_voxels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("point_cloud_range", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_feature_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* VoxelGeneratorPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* VoxelGeneratorPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* VoxelGeneratorPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* VoxelGeneratorPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;
        int max_points = 0;
        int max_voxels = 0;
        float point_cloud_range[6] = {0.0f};
        int voxel_feature_num = 0;
        float voxel_size[3] = {0.0f};
        for (int i = 0; i < nbFields; ++i)
        {
            const char* attr_name = fields[i].name;
            if (!strcmp(attr_name, "max_num_points_per_voxel"))
            {
                const int* d = static_cast<const int*>(fields[i].data);
                max_points = d[0];
            }
            else if (!strcmp(attr_name, "max_voxels"))
            {
                const int* d = static_cast<const int*>(fields[i].data);
                max_voxels = d[0];
            }
            else if (!strcmp(attr_name, "point_cloud_range"))
            {
                const float* d = static_cast<const float*>(fields[i].data);
                point_cloud_range[0] = d[0];
                point_cloud_range[1] = d[1];
                point_cloud_range[2] = d[2];
                point_cloud_range[3] = d[3];
                point_cloud_range[4] = d[4];
                point_cloud_range[5] = d[5];
            }
            else if (!strcmp(attr_name, "voxel_feature_num"))
            {
                const int* d = static_cast<const int*>(fields[i].data);
                voxel_feature_num = d[0];
            }
            else if (!strcmp(attr_name, "voxel_size"))
            {
                const float* d = static_cast<const float*>(fields[i].data);
                voxel_size[0] = d[0];
                voxel_size[1] = d[1];
                voxel_size[2] = d[2];
            }
        }
        IPluginV2* plugin = new VoxelGeneratorPlugin(max_voxels, max_points, voxel_feature_num, point_cloud_range[0],
            point_cloud_range[3], point_cloud_range[1], point_cloud_range[4], point_cloud_range[2],
            point_cloud_range[5], voxel_size[0], voxel_size[1], voxel_size[2]);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* VoxelGeneratorPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        return new VoxelGeneratorPlugin(serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void VoxelGeneratorPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* VoxelGeneratorPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
