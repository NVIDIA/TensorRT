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

#ifndef TRT_MULTILEVEL_CROP_AND_RESIZE_PLUGIN_H
#define TRT_MULTILEVEL_CROP_AND_RESIZE_PLUGIN_H

#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/kernels/maskRCNNKernels.h"
#include "multilevelProposeROI/tlt_mrcnn_config.h"

namespace nvinfer1
{
namespace plugin
{

class MultilevelCropAndResize : public IPluginV2Ext
{
public:
    MultilevelCropAndResize(int pooled_size, const nvinfer1::Dims& image_size);

    MultilevelCropAndResize(const void* data, size_t length);

    ~MultilevelCropAndResize() noexcept override = default;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override;

    int32_t enqueue(
        int32_t batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    void check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims) noexcept;

    xy_t mPooledSize;
    static const int mFeatureMapCount = 5; // p2, p3, p4, p5, p6(Maxpooling)
    int mFeatureLength;
    int mROICount;
    float mThresh;
    int mInputHeight;
    int mInputWidth;
    xy_t mFeatureSpatialSize[mFeatureMapCount];
    std::string mNameSpace;
    DataType mPrecision;
};

class MultilevelCropAndResizePluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    MultilevelCropAndResizePluginCreator() noexcept;

    ~MultilevelCropAndResizePluginCreator() noexcept {};

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* data, size_t length) noexcept override;

private:
    static PluginFieldCollection mFC;
    int mPooledSize;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_MULTILEVEL_CROP_AND_RESIZE_PLUGIN_H
