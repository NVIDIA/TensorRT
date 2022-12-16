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

#ifndef TRT_FMHCA_PLUGIN_H
#define TRT_FMHCA_PLUGIN_H
#if defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)

#include "common/bertCommon.h"

#include <NvInfer.h>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class FusedMultiHeadCrossAttentionKernel;
class FMHCAPlugin : public IPluginV2DynamicExt
{
public:
    FMHCAPlugin(std::string const& name);
    FMHCAPlugin(std::string const& name, void const* data, size_t length);
    FMHCAPlugin() = delete;
    ~FMHCAPlugin() = default;

    void init(bool loadCubins = false);

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    DataType getOutputDataType(
        int32_t outputIndex, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;

    void setPluginNamespace(char const* szNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

protected:
    void createMHARunner();

    void allocateSeqlens(int32_t maxBatchSize);
    int32_t initializeSeqlens(int32_t b, int32_t s, void* cuSeqlensDev, cudaStream_t stream = 0);

private:
    // data need serialized into engine.
    struct
    {
        int32_t mOptBatchSize{};
        int32_t mOptSeqLenQ{};
        int32_t mOptSeqLenKV{};
        int32_t mMaxBatchSize{};
        DataType mDataType{DataType::kFLOAT};
    } mSerializationData;

    int32_t mSM{};
    bert::cuda_shared_ptr<void> mCuSeqLensQ;
    bert::cuda_shared_ptr<void> mCuSeqLensKV;
    FusedMultiHeadCrossAttentionKernel const* mKernels{};

    std::string const mLayerName;
    std::string mNamespace;
}; // class FMHCAPlugin

class FMHCAPluginCreator : public IPluginCreator
{
public:
    FMHCAPluginCreator();
    ~FMHCAPluginCreator() = default;

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* szNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
}; // class FMHCAPluginCreator

} // namespace plugin
} // namespace nvinfer1

#endif // defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
#endif // TRT_FMHCA_PLUGIN_H
