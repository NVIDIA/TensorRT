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
#ifndef TRT_MAX_POOL_RUNTIME_PLUGIN_H
#define TRT_MAX_POOL_RUNTIME_PLUGIN_H

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "NvInferSafePlugin.h"
#include "NvInferSafeRuntime.h"

// In IPluginV3 interface, the plugin name, version, and name space must be
// specified for the plugin and plugin creator exactly the same.
constexpr char const* const kMAX_POOL_PLUGIN_NAME{"MaxPoolPlugin"};
constexpr char const* const kMAX_POOL_PLUGIN_VERSION{"1"};
constexpr char const* const kMAX_POOL_PLUGIN_NAMESPACE{""};

using namespace nvinfer2::safe;
namespace nvinfer1
{
namespace plugin
{

using AsciiChar = char;

enum class PoolingType : int32_t
{
    kMAX = 0,              // Maximum over elements
    kAVERAGE = 1,          // Average over elements. If the tensor is padded, the count includes the padding
    kMAX_AVERAGE_BLEND = 2 // Blending between max and average pooling: (1-blendFactor)*maxPool + blendFactor*avgPool
};

struct PoolParameters
{
    // Input dimensions
    int32_t C;
    int32_t H;
    int32_t W;
    // Output dimensions
    int32_t H_out;
    int32_t W_out;
    // Kernel size,
    int32_t Kx;
    int32_t Ky;
    // Stride
    int32_t Sx;
    int32_t Sy;
    // Padding
    int32_t Px;
    int32_t Py;
    // Pooling Function
    PoolingType pType;

    // Extra Info
    nvinfer1::DataType dtype;
    size_t dtypeBytes;
    int32_t nbInputs;
    int64_t nbIOProfile;
    std::shared_ptr<Dims> inputDims;

    // Default Constructor
    PoolParameters()
    {
        // To do: Populate Parameters from fc object w/ hard code
        pType = PoolingType::kMAX;
        // stride
        Sx = 3;
        Sy = 3;
        // kernel size
        Kx = 3;
        Ky = 3;
        // padding
        Px = 0;
        Py = 0;
    }

    // Parameterized Constructor
    PoolParameters(nvinfer1::plugin::PoolingType pooling_type, int32_t stride_x, int32_t stride_y, int32_t kernel_x,
        int32_t kernel_y, int32_t padding_x, int32_t padding_y)
    {

        pType = pooling_type;
        // stride
        Sx = stride_x;
        Sy = stride_y;
        // kernel size
        Kx = kernel_x;
        Ky = kernel_y;
        // padding
        Px = padding_x;
        Py = padding_y;
    }
};

class MaxPoolPluginRuntime : public IPluginV3, public IPluginV3OneSafeCore, public IPluginV3OneSafeRuntime
{
public:
    MaxPoolPluginRuntime(PoolParameters const& params);

    ~MaxPoolPluginRuntime() override = default;

    // IPluginV3 Methods

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    // IPluginV3OneSafeCore Methods

    AsciiChar const* getPluginName() const noexcept override;

    AsciiChar const* getPluginVersion() const noexcept override;

    AsciiChar const* getPluginNamespace() const noexcept override;

    // IPluginV3OneSafeRuntime Methods

    int32_t enqueue(TensorDescriptor const* inputDesc, TensorDescriptor const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t initResource(ISafePluginResourceContext const* context) noexcept;

    IPluginV3* clone() noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setSafeRecorder(ISafeRecorder&) noexcept;

    ISafeRecorder* getSafeRecorder() const noexcept;

protected:
    void initFieldsToSerialize();

    // TensorRT plugin parameters.
    PoolParameters mParams;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
    ISafeRecorder* mRecorder;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_MAX_POOL_RUNTIME_PLUGIN_H
