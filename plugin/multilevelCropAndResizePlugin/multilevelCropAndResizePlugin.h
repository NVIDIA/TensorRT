/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#ifndef TRT_MULTILEVEL_CROP_AND_RESIZE_PLUGIN_H
#define TRT_MULTILEVEL_CROP_AND_RESIZE_PLUGIN_H

#include <cassert>
#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "maskRCNNKernels.h"
#include "tlt_mrcnn_config.h"

namespace nvinfer1
{
namespace plugin
{

class MultilevelCropAndResize : public IPluginV2Ext
{
public:
    MultilevelCropAndResize(int pooled_size);

    MultilevelCropAndResize(const void* data, size_t length);

    ~MultilevelCropAndResize() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    void destroy() override;

    size_t getWorkspaceSize(int) const override;

    int enqueue(
        int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;

private:
    void check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims);

    xy_t mPooledSize;
    static const int mFeatureMapCount = 5; //p2, p3, p4, p5, p6(Maxpooling)
    int mFeatureLength;
    int mROICount;
    float mThresh;
    int mInputHeight;
    int mInputWidth;
    xy_t mFeatureSpatialSize[mFeatureMapCount];
    std::string mNameSpace;
};

class MultilevelCropAndResizePluginCreator : public BaseCreator
{
public:
    MultilevelCropAndResizePluginCreator();

    ~MultilevelCropAndResizePluginCreator(){};

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* data, size_t length) override;

private:
    static PluginFieldCollection mFC;
    int mPooledSize;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_MULTILEVEL_CROP_AND_RESIZE_PLUGIN_H
