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
#ifndef TRT_PROPOSAL_LAYER_PLUGIN_H
#define TRT_PROPOSAL_LAYER_PLUGIN_H
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

class ProposalLayer : public IPluginV2Ext
{
public:
    ProposalLayer(int prenms_topk, int keep_topk, float iou_threshold, nvinfer1::Dims const& image_size);

    ProposalLayer(void const* data, size_t length);

    ~ProposalLayer() override = default;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, Dims const* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int enqueue(int batch_size, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    // void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs,
    // nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int outputIndex, bool const* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(Dims const* inputDims, int nbInputs, Dims const* outputDims, int nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    void deserialize(int8_t const* data, size_t length);
    void check_valid_inputs(nvinfer1::Dims const* inputs, int nbInputDims);
    void generate_pyramid_anchors(nvinfer1::Dims const& imageDims);

    int32_t mBackgroundLabel{};
    int32_t mPreNMSTopK{};
    int32_t mKeepTopK{};
    float mIOUThreshold{};

    int32_t mMaxBatchSize{};
    int32_t mAnchorsCnt{};
    std::shared_ptr<CudaBind<int32_t>> mValidCnt; // valid cnt = number of input roi for every image.
    std::shared_ptr<CudaBind<float>>
        mAnchorBoxesDevice; // [N, anchors(261888 for resnet101 + 1024*1024), (y1, x1, y2, x2)]
    std::vector<float> mAnchorBoxesHost;

    nvinfer1::DataType mType{};
    nvinfer1::Dims mImageSize{};
    RefineNMSParameters mParam{};

    std::string mNameSpace;
};

class ProposalLayerPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ProposalLayerPluginCreator();

    ~ProposalLayerPluginCreator() override{};

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* data, size_t length) noexcept override;

private:
    static PluginFieldCollection mFC;
    int mPreNMSTopK;
    int mKeepTopK;
    float mIOUThreshold;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_PROPOSAL_LAYER_PLUGIN_H
