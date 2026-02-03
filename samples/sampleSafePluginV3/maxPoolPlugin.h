/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_MAX_POOL_PLUGIN_H
#define TRT_MAX_POOL_PLUGIN_H

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "NvInferConsistency.h"
#include "NvInferPluginBase.h"
#include "NvInferRuntime.h"
#include "NvInferSafePlugin.h"
#include "NvInferSafeRuntime.h"
#include "maxPoolPluginRuntime.h"

using namespace nvinfer2::safe;
namespace nvinfer1
{
namespace plugin
{

class MaxPoolPlugin : public MaxPoolPluginRuntime, public IPluginV3OneSafeBuildMSS
{
public:
    MaxPoolPlugin(PoolParameters const& params)
        : MaxPoolPluginRuntime(params){};

    ~MaxPoolPlugin() override = default;

    // IPluginV3 Methods

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    IPluginV3* clone() noexcept override;

    // IPluginV3OneSafeBuildMSS Methods

    int32_t getNbOutputs() const noexcept override;

    int32_t configurePlugin(
        TensorDescriptor const* in, int32_t nbInputs, TensorDescriptor const* out, int32_t nbOutputs) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, TensorDescriptor const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        Dims const* inputs, int32_t nbInputs, Dims* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getSymbolicOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) const noexcept override;

    size_t getWorkspaceSize(TensorDescriptor const* inputs, int32_t nbInputs, TensorDescriptor const* outputs,
        int32_t nbOutputs) const noexcept override;
};

} // namespace plugin
} // namespace nvinfer1

namespace nvinfer2::safe::consistency
{

class MaxPoolPluginChecker : public IPluginChecker
{
public:
    bool validate(std::vector<nvinfer2::safe::TensorDescriptor> const& /*Inputs*/,
        std::vector<nvinfer2::safe::TensorDescriptor> const& /*Outputs*/,
        nvinfer1::PluginFieldCollection* /*fc*/) noexcept override
    {
        // Always return true
        return true;
    }
};

} // namespace nvinfer2::safe::consistency

extern "C" nvinfer2::safe::consistency::IPluginChecker* getPluginChecker(char const* name);

#endif // TRT_MAX_POOL_PLUGIN_H
