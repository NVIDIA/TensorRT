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
#ifndef TRT_PYRAMID_ROIALIGN_PLUGIN_H
#define TRT_PYRAMID_ROIALIGN_PLUGIN_H

#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/kernels/maskRCNNKernels.h"
#include "common/mrcnn_config.h"

namespace nvinfer1
{
namespace plugin
{

class PyramidROIAlign : public IPluginV2Ext
{
public:
    PyramidROIAlign(int32_t pooledSize, int32_t transformCoords, bool absCoords, bool swapCoords, bool plusOneCoords,
        int32_t samplingRatio, bool legacy, xy_t roiRange, int32_t fpnScale);

    PyramidROIAlign(void const* data, size_t length);

    ~PyramidROIAlign() override = default;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override;

    int32_t enqueue(int32_t batch_size, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    void check_valid_inputs(nvinfer1::Dims const* inputs, int32_t nbInputDims);

    static int32_t const mFeatureMapCount = 4;

    xy_t mPooledSize;
    xy_t mImageSize;
    int32_t mFeatureLength;
    int32_t mROICount;
    int32_t mFPNScale;
    int32_t mTransformCoords;
    bool mAbsCoords;
    bool mSwapCoords;
    bool mPlusOneCoords;
    int32_t mSamplingRatio;
    bool mIsLegacy{false};
    xy_t mFeatureSpatialSize[mFeatureMapCount];
    std::string mNameSpace;
};

class PyramidROIAlignPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    PyramidROIAlignPluginCreator();

    ~PyramidROIAlignPluginCreator(){};

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* data, size_t length) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_PYRAMID_ROIALIGN_PLUGIN_H
