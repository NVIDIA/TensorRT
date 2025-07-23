/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef TRT_SKIP_LAYER_NORM_PLUGIN_H
#define TRT_SKIP_LAYER_NORM_PLUGIN_H

#include "NvInferPlugin.h"

#include "common/bertCommon.h"
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{
template <bool hasBias>
int32_t computeSkipLayerNormDQQ(cudaStream_t stream, int32_t const ld, int32_t const n, int8_t const* input,
    int8_t const* skip, __half const* beta, __half const* gamma, int8_t* output, __half const* bias,
    float const dqScaleIn, float const dqScaleSkip, float const qScale);

template <typename T, bool hasBias>
int32_t computeSkipLayerNorm(cudaStream_t stream, int32_t const ld, int32_t const n, T const* input, T const* skip,
    T const* beta, T const* gamma, T* output, T const* bias);

class SkipLayerNormPluginV3 : public IPluginV3,
                              public IPluginV3OneCore,
                              public IPluginV3OneBuild,
                              public IPluginV3OneRuntime
{
public:
    SkipLayerNormPluginV3(const std::string name, const nvinfer1::DataType type, int32_t const ld,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& bias);

    // It doesn't make sense to make SkipLayerNormPluginV3 without arguments,
    // so we delete default constructor.
    SkipLayerNormPluginV3() = delete;

    ~SkipLayerNormPluginV3() override;

    // IPluginV3 Methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    IPluginV3* clone() noexcept override;
    // end of IPluginV3 Methods

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
    int32_t getNbOutputs() const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

private:
    // metadata
    const std::string mLayerName;
    std::string mNamespace;

    // members that participate in ser/deserialization
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;
    bert::WeightsWithOwnership mBias;
    nvinfer1::DataType mType;
    nvinfer1::DataType mCfgType;
    int32_t mLd{}; // leading dim
    bool mHasBias{};

    // device-side
    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;
    bert::cuda_unique_ptr<void> mBiasDev;

    // derived member from mCfgType
    size_t mParamWordsize{};

    // serialization data structures
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class SkipLayerNormPluginV3Creator : public nvinfer1::IPluginCreatorV3One
{
public:
    SkipLayerNormPluginV3Creator();
    ~SkipLayerNormPluginV3Creator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

class SkipLayerNormVarSeqlenPluginV3 : public IPluginV3,
                                       public IPluginV3OneCore,
                                       public IPluginV3OneBuild,
                                       public IPluginV3OneRuntime
{
public:
    SkipLayerNormVarSeqlenPluginV3(const std::string name, const nvinfer1::DataType type, nvinfer1::Weights const& beta,
        nvinfer1::Weights const& gamma, nvinfer1::Weights const& bias);

    SkipLayerNormVarSeqlenPluginV3(const std::string name, void const* data, size_t length);

    // It doesn't make sense to make SkipLayerNormVarSeqlenPluginV3 without
    // arguments, so we delete default constructor.
    SkipLayerNormVarSeqlenPluginV3() = delete;

    ~SkipLayerNormVarSeqlenPluginV3() override;

    // IPluginV3 Methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    IPluginV3* clone() noexcept override;
    // end of IPluginV3 Methods

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
    int32_t getNbOutputs() const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

private:
    const std::string mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;
    int32_t mLd{}; // leading dim
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;
    nvinfer1::DataType mType;
    nvinfer1::DataType mCfgType;

    bool mHasBias{};
    bert::cuda_unique_ptr<void> mBiasDev;
    bert::WeightsWithOwnership mBias;

    size_t mParamWordsize{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class SkipLayerNormVarSeqlenPluginV3Creator : public nvinfer1::IPluginCreatorV3One
{
public:
    SkipLayerNormVarSeqlenPluginV3Creator();
    ~SkipLayerNormVarSeqlenPluginV3Creator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SKIP_LAYER_NORM_PLUGIN_H

#endif // CUDA_VERSION >= 10010
