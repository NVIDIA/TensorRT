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
#ifndef TRT_SCATTER_PLUGIN_H
#define TRT_SCATTER_PLUGIN_H
#include "common/kernel.h"
#include "common/plugin.h"
#include "cudnn.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class ScatterND : public IPluginV2DynamicExt
{
public:
    ScatterND();        

    ~ScatterND() override = default;

    int getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
        const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;

    virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)  const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;
private:

    //calculate how many slices we need to scatter = reduce_mul(indexTensor.shape[:-1])
    int32_t calculateNumSlices(Dims indexTensorDims) const noexcept;
    int32_t calculateCopySize(const Dims& dataDims) const noexcept;
    void calculateTransformCoeff(const Dims& dataTensorDims, int indexRank, int32_t* transformCoeff) const noexcept;
    std::string mPluginNamespace;    

    static constexpr  int indexTensorIdx = 1;
    static constexpr  int updateTensorIdx = 2;
    static constexpr  int dataTensorIdx = 0;
};

class ScatterNDPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ScatterNDPluginCreator();

    ~ScatterNDPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion()const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SCATTER_PLUGIN_H
