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

#include "voxelGenerator.h"
#include "common/templates.h"
#include <cmath>
#include <cstring>
#include <iostream>

namespace nvinfer1::plugin
{

using namespace nvinfer1;
using nvinfer1::plugin::VoxelGeneratorPlugin;
using nvinfer1::plugin::VoxelGeneratorPluginCreator;

namespace
{
char const* const kVOXEL_GENERATOR_PLUGIN_VERSION{"1"};
char const* const kVOXEL_GENERATOR_PLUGIN_NAME{"VoxelGeneratorPlugin"};
size_t constexpr kSERIALIZATION_SIZE{9 * sizeof(float) + 7 * sizeof(int32_t)};
} // namespace

// Mimic np.round as in voxel generator in spconv implementation
int32_t npRound(float x)
{
    // half way round to nearest-even
    int32_t x2 = lround(x * 2.0F);
    if (x != static_cast<int32_t>(x) && x2 == x * 2.0F)
    {
        return lround(x / 2.0F + 0.5F) * 2;
    }
    return lround(x);
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(int32_t maxVoxels, int32_t maxPoints, int32_t voxelFeatures, float xMin,
    float xMax, float yMin, float yMax, float zMin, float zMax, float pillarX, float pillarY, float pillarZ)
    : mPillarNum(maxVoxels)
    , mPointNum(maxPoints)
    , mFeatureNum(voxelFeatures)
    , mMinXRange(xMin)
    , mMaxXRange(xMax)
    , mMinYRange(yMin)
    , mMaxYRange(yMax)
    , mMinZRange(zMin)
    , mMaxZRange(zMax)
    , mPillarXSize(pillarX)
    , mPillarYSize(pillarY)
    , mPillarZSize(pillarZ)
{
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(int32_t maxVoxels, int32_t maxPoints, int32_t voxelFeatures, float xMin,
    float xMax, float yMin, float yMax, float zMin, float zMax, float pillarX, float pillarY, float pillarZ,
    int32_t pointFeatures, int32_t gridX, int32_t gridY, int32_t gridZ)
    : mPillarNum(maxVoxels)
    , mPointNum(maxPoints)
    , mFeatureNum(voxelFeatures)
    , mMinXRange(xMin)
    , mMaxXRange(xMax)
    , mMinYRange(yMin)
    , mMaxYRange(yMax)
    , mMinZRange(zMin)
    , mMaxZRange(zMax)
    , mPillarXSize(pillarX)
    , mPillarYSize(pillarY)
    , mPillarZSize(pillarZ)
    , mPointFeatureNum(pointFeatures)
    , mGridXSize(gridX)
    , mGridYSize(gridY)
    , mGridZSize(gridZ)
{
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(void const* data, size_t length)
{
    PLUGIN_ASSERT(data != nullptr);
    uint8_t const* d = reinterpret_cast<uint8_t const*>(data);
    auto const *a = d;
    mPillarNum = readFromBuffer<int32_t>(d);
    mPointNum = readFromBuffer<int32_t>(d);
    mFeatureNum = readFromBuffer<int32_t>(d);
    mMinXRange = readFromBuffer<float>(d);
    mMaxXRange = readFromBuffer<float>(d);
    mMinYRange = readFromBuffer<float>(d);
    mMaxYRange = readFromBuffer<float>(d);
    mMinZRange = readFromBuffer<float>(d);
    mMaxZRange = readFromBuffer<float>(d);
    mPillarXSize = readFromBuffer<float>(d);
    mPillarYSize = readFromBuffer<float>(d);
    mPillarZSize = readFromBuffer<float>(d);
    mPointFeatureNum = readFromBuffer<int32_t>(d);
    mGridXSize = readFromBuffer<int32_t>(d);
    mGridYSize = readFromBuffer<int32_t>(d);
    mGridZSize = readFromBuffer<int32_t>(d);
    PLUGIN_ASSERT(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* VoxelGeneratorPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new VoxelGeneratorPlugin(mPillarNum, mPointNum, mFeatureNum, mMinXRange, mMaxXRange, mMinYRange,
            mMaxYRange, mMinZRange, mMaxZRange, mPillarXSize, mPillarYSize, mPillarZSize, mPointFeatureNum, mGridXSize,
            mGridYSize, mGridZSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs VoxelGeneratorPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        auto batchSize = inputs[0].d[0];
        if (outputIndex == 0)
        {
            nvinfer1::DimsExprs dim0{};
            dim0.nbDims = 4;
            dim0.d[0] = batchSize;
            dim0.d[1] = exprBuilder.constant(mPillarNum);
            dim0.d[2] = exprBuilder.constant(mPointNum);
            dim0.d[3] = exprBuilder.constant(mFeatureNum);
            return dim0;
        }
        if (outputIndex == 1)
        {
            nvinfer1::DimsExprs dim1{};
            dim1.nbDims = 3;
            dim1.d[0] = batchSize;
            dim1.d[1] = exprBuilder.constant(mPillarNum);
            dim1.d[2] = exprBuilder.constant(4);
            return dim1;
        }
        nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 1;
        dim2.d[0] = batchSize;
        return dim2;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

bool VoxelGeneratorPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 3);
        PluginTensorDesc const& in = inOut[pos];
        if (pos == 0) // PointCloud Array --- x, y, z, w
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 1) // Point Num
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 2) // features, dim: pillarNum x pointNum x featureNum
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 3) // pillarCoords, dim: 1 x 1 x pillarNum x 4
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 4) // params, dim: 1 x 1 x 1 x 1
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        return false;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void VoxelGeneratorPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 3);

        mPointFeatureNum = in[0].desc.dims.d[2];
        mGridXSize = npRound((mMaxXRange - mMinXRange) / mPillarXSize);
        mGridYSize = npRound((mMaxYRange - mMinYRange) / mPillarYSize);
        mGridZSize = npRound((mMaxZRange - mMinZRange) / mPillarZSize);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t VoxelGeneratorPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    try
    {
        int32_t batchSize = inputs[0].dims.d[0];
        size_t maskSize = batchSize * mGridZSize * mGridYSize * mGridXSize * sizeof(uint32_t);
        size_t voxelsSize = batchSize * mGridZSize * mGridYSize * mGridXSize * mPointNum * mPointFeatureNum * sizeof(float);
        // the actual max pillar num cannot be determined, use upper bound
        size_t voxelFeaturesSize = voxelsSize;
        size_t voxelNumPointsSize = maskSize;
        size_t workspaces[4];
        workspaces[0] = maskSize;
        workspaces[1] = voxelsSize;
        workspaces[2] = voxelFeaturesSize;
        workspaces[3] = voxelNumPointsSize;
        return calculateTotalWorkspaceSize(workspaces, 4);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0U;
}

int32_t VoxelGeneratorPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr && workspace != nullptr);

        int32_t batchSize = inputDesc[0].dims.d[0];
        int32_t maxNumPoints = inputDesc[0].dims.d[1];
        // TRT-input
        float* pointCloud = const_cast<float*>((float const*) inputs[0]);
        uint32_t* pointNumPtr = const_cast<uint32_t*>((uint32_t const*) inputs[1]);
        // TRT-output
        float* pillarFeaturesData = static_cast<float*>(outputs[0]);
        uint32_t* coordsData = static_cast<uint32_t*>(outputs[1]);
        uint32_t* paramsData = static_cast<uint32_t*>(outputs[2]);
        int32_t densePillarNum = mGridZSize * mGridYSize * mGridXSize;
        size_t maskSize = batchSize * densePillarNum * sizeof(uint32_t);
        size_t voxelsSize = batchSize * densePillarNum * mPointNum * mPointFeatureNum * sizeof(float);
        size_t voxelFeaturesSize = voxelsSize;
        size_t voxelNumPointsSize = maskSize;
        size_t workspaces[4];
        workspaces[0] = maskSize;
        workspaces[1] = voxelsSize;
        workspaces[2] = voxelFeaturesSize;
        workspaces[3] = voxelNumPointsSize;
        size_t totalWorkspace = calculateTotalWorkspaceSize(workspaces, 4);
        uint32_t* mask = static_cast<uint32_t*>(workspace);
        float* voxels = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(mask), maskSize));
        float* voxelFeatures
            = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(voxels), voxelsSize));
        uint32_t* voxelNumPoints = reinterpret_cast<uint32_t*>(
            nextWorkspacePtr(reinterpret_cast<int8_t*>(voxelFeatures), voxelFeaturesSize));
        // Initialize workspace memory
        PLUGIN_CUASSERT(cudaMemsetAsync(mask, 0, totalWorkspace, stream));
        uint32_t pillarFeaturesDataSize = batchSize * mPillarNum * mPointNum * mFeatureNum * sizeof(float);
        uint32_t coordsDataSize = batchSize * mPillarNum * 4 * sizeof(uint32_t);
        uint32_t paramsDataSize = batchSize * sizeof(uint32_t);
        PLUGIN_CUASSERT(cudaMemsetAsync(pillarFeaturesData, 0, pillarFeaturesDataSize, stream));
        PLUGIN_CUASSERT(cudaMemsetAsync(coordsData, 0, coordsDataSize, stream));
        PLUGIN_CUASSERT(cudaMemsetAsync(paramsData, 0, paramsDataSize, stream));
        // pointcloud + pointNum ---> mask_ + voxel_
        generateVoxels_launch(batchSize, maxNumPoints, pointCloud, pointNumPtr, mMinXRange, mMaxXRange, mMinYRange,
            mMaxYRange, mMinZRange, mMaxZRange, mPillarXSize, mPillarYSize, mPillarZSize, mGridYSize, mGridXSize,
            mPointFeatureNum, mPointNum, mask, voxels, stream);
        // mask_ + voxel_ ---> params_data + voxel_features_ + voxel_num_points_ +
        // coords_data
        generateBaseFeatures_launch(batchSize, mask, voxels, mGridYSize, mGridXSize, paramsData, mPillarNum, mPointNum,
            mPointFeatureNum, voxelFeatures, voxelNumPoints, coordsData, stream);
        generateFeatures_launch(batchSize, densePillarNum, voxelFeatures, voxelNumPoints, coordsData, paramsData,
            mPillarXSize, mPillarYSize, mPillarZSize, mMinXRange, mMinYRange, mMinZRange, mFeatureNum, mPointNum, mPillarNum,
            mPointFeatureNum, pillarFeaturesData, stream);
        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

nvinfer1::DataType VoxelGeneratorPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        if (index == 0)
        {
            return inputTypes[0];
        }
        return inputTypes[1];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

char const* VoxelGeneratorPlugin::getPluginType() const noexcept
{
    return kVOXEL_GENERATOR_PLUGIN_NAME;
}

char const* VoxelGeneratorPlugin::getPluginVersion() const noexcept
{
    return kVOXEL_GENERATOR_PLUGIN_VERSION;
}

int32_t VoxelGeneratorPlugin::getNbOutputs() const noexcept
{
    return 3;
}

int32_t VoxelGeneratorPlugin::initialize() noexcept
{
    return 0;
}

void VoxelGeneratorPlugin::terminate() noexcept {}

size_t VoxelGeneratorPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void VoxelGeneratorPlugin::serialize(void* buffer) const noexcept
{

    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t* d = reinterpret_cast<uint8_t*>(buffer);
    auto *a = d;
    writeToBuffer<int32_t>(d, mPillarNum);
    writeToBuffer<int32_t>(d, mPointNum);
    writeToBuffer<int32_t>(d, mFeatureNum);
    writeToBuffer<float>(d, mMinXRange);
    writeToBuffer<float>(d, mMaxXRange);
    writeToBuffer<float>(d, mMinYRange);
    writeToBuffer<float>(d, mMaxYRange);
    writeToBuffer<float>(d, mMinZRange);
    writeToBuffer<float>(d, mMaxZRange);
    writeToBuffer<float>(d, mPillarXSize);
    writeToBuffer<float>(d, mPillarYSize);
    writeToBuffer<float>(d, mPillarZSize);
    writeToBuffer<int32_t>(d, mPointFeatureNum);
    writeToBuffer<int32_t>(d, mGridXSize);
    writeToBuffer<int32_t>(d, mGridYSize);
    writeToBuffer<int32_t>(d, mGridZSize);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void VoxelGeneratorPlugin::destroy() noexcept
{
    delete this;
}

void VoxelGeneratorPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* VoxelGeneratorPlugin::getPluginNamespace() const noexcept
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

char const* VoxelGeneratorPluginCreator::getPluginName() const noexcept
{
    return kVOXEL_GENERATOR_PLUGIN_NAME;
}

char const* VoxelGeneratorPluginCreator::getPluginVersion() const noexcept
{
    return kVOXEL_GENERATOR_PLUGIN_VERSION;
}

PluginFieldCollection const* VoxelGeneratorPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* VoxelGeneratorPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;
        int32_t maxPoints = 0;
        int32_t maxVoxels = 0;
        float pointCloudRange[6]{};
        int32_t voxelFeatureNum = 0;
        float voxelSize[3]{};
        for (int32_t i = 0; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "max_num_points_per_voxel"))
            {
                int32_t const* d = static_cast<int32_t const*>(fields[i].data);
                maxPoints = d[0];
            }
            else if (!strcmp(attrName, "max_voxels"))
            {
                int32_t const* d = static_cast<int32_t const*>(fields[i].data);
                maxVoxels = d[0];
            }
            else if (!strcmp(attrName, "point_cloud_range"))
            {
                float const* d = static_cast<float const*>(fields[i].data);
                pointCloudRange[0] = d[0];
                pointCloudRange[1] = d[1];
                pointCloudRange[2] = d[2];
                pointCloudRange[3] = d[3];
                pointCloudRange[4] = d[4];
                pointCloudRange[5] = d[5];
            }
            else if (!strcmp(attrName, "voxel_feature_num"))
            {
                int32_t const* d = static_cast<int32_t const*>(fields[i].data);
                voxelFeatureNum = d[0];
            }
            else if (!strcmp(attrName, "voxel_size"))
            {
                float const* d = static_cast<float const*>(fields[i].data);
                voxelSize[0] = d[0];
                voxelSize[1] = d[1];
                voxelSize[2] = d[2];
            }
        }
        IPluginV2* plugin = new VoxelGeneratorPlugin(maxVoxels, maxPoints, voxelFeatureNum, pointCloudRange[0],
            pointCloudRange[3], pointCloudRange[1], pointCloudRange[4], pointCloudRange[2], pointCloudRange[5],
            voxelSize[0], voxelSize[1], voxelSize[2]);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* VoxelGeneratorPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
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

void VoxelGeneratorPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* VoxelGeneratorPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
} // namespace nvinfer1::plugin
