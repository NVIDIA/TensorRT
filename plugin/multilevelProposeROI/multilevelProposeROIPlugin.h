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
#ifndef TRT_MULTILEVEL_PROPOSE_ROI_PLUGIN_H
#define TRT_MULTILEVEL_PROPOSE_ROI_PLUGIN_H
#include <cassert>
#include <cuda_runtime_api.h>
#include <memory>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "maskRCNNKernels.h"

namespace nvinfer1
{
namespace plugin
{

class MultilevelProposeROI : public IPluginV2Ext
{
public:
    MultilevelProposeROI(int prenms_topk, int keep_topk, float fg_threshold, float iou_threshold);

    MultilevelProposeROI(const void* data, size_t length);

    ~MultilevelProposeROI() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    void destroy() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

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
    void generate_pyramid_anchors();

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
    std::vector<std::shared_ptr<CudaBind<float>>>
        mTempScores;
    std::vector<std::shared_ptr<CudaBind<float>>>
        mTempBboxes;
    float** mDeviceScores;
    float** mDeviceBboxes;
    std::shared_ptr<CudaBind<float>> mRegWeightDevice;

    nvinfer1::DataType mType;
    RefineNMSParameters mParam;

    std::string mNameSpace;
};

class MultilevelProposeROIPluginCreator : public BaseCreator
{
public:
    MultilevelProposeROIPluginCreator();

    ~MultilevelProposeROIPluginCreator(){};

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* data, size_t length) override;

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
