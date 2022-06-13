/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_MULTILEVEL_PROPOSE_ROI_PLUGIN_H
#define TRT_MULTILEVEL_PROPOSE_ROI_PLUGIN_H
#include <cuda_runtime_api.h>
#include <memory>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/kernels/maskRCNNKernels.h"

namespace nvinfer1
{
namespace plugin
{

class MultilevelProposeROI : public IPluginV2Ext
{
public:
    MultilevelProposeROI(
        int prenms_topk, int keep_topk, float fg_threshold, float iou_threshold, const nvinfer1::Dims image_size);

    MultilevelProposeROI(const void* data, size_t length);

    ~MultilevelProposeROI() noexcept override = default;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

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
    void generate_pyramid_anchors(nvinfer1::Dims const& imageSize);

    int mBackgroundLabel;
    int mPreNMSTopK;
    int mKeepTopK;
    int mFeatureCnt;
    float mFGThreshold;
    float mIOUThreshold;

    int mMaxBatchSize;
    std::vector<int> mAnchorsCnt;
    std::shared_ptr<CudaBind<int>> mValidCnt; // valid cnt = number of input roi for every image.
    std::vector<std::shared_ptr<CudaBind<float>>>
        mAnchorBoxesDevice; // [N, anchors(261888 for resnet101 + 1024*1024), (y1, x1, y2, x2)]
    std::vector<std::vector<float>> mAnchorBoxesHost;
    std::vector<std::shared_ptr<CudaBind<float>>> mTempScores_float;
    std::vector<std::shared_ptr<CudaBind<float>>> mTempBboxes_float;
    std::vector<std::shared_ptr<CudaBind<uint16_t>>> mTempScores_half;
    std::vector<std::shared_ptr<CudaBind<uint16_t>>> mTempBboxes_half;
    float** mDeviceScores;
    float** mDeviceBboxes;
    std::shared_ptr<CudaBind<float>> mRegWeightDevice;

    nvinfer1::Dims mImageSize;
    nvinfer1::DataType mType;
    RefineNMSParameters mParam;

    std::string mNameSpace;
};

class MultilevelProposeROIPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    MultilevelProposeROIPluginCreator() noexcept;

    ~MultilevelProposeROIPluginCreator() noexcept {};

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* data, size_t length) noexcept override;

private:
    static PluginFieldCollection mFC;
    int mPreNMSTopK;
    int mKeepTopK;
    float mFGThreshold;
    float mIOUThreshold;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_MULTILEVEL_PROPOSE_ROI_PLUGIN_H
