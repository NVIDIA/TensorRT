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

#include "customHardmaxPlugin.h"
#include "NvInferPlugin.h"
#include "common.h" // volume(), ASSERT
#include "logger.h" // sample::gLogError
#include <cuda.h>

using namespace nvinfer1;

#define CUDRIVER_CALL(call)                                                                                            \
    {                                                                                                                  \
        cudaError_enum s_ = call;                                                                                      \
        if (s_ != CUDA_SUCCESS)                                                                                        \
        {                                                                                                              \
            char const *errName_, *errDesc_;                                                                           \
            cuGetErrorName(s_, &errName_);                                                                             \
            cuGetErrorString(s_, &errDesc_);                                                                           \
            sample::gLogError << "CUDA Error: " << errName_ << " " << errDesc_ << std::endl;                           \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

#define CUDA_CALL(call)                                                                                                \
    {                                                                                                                  \
        cudaError_t s_ = call;                                                                                         \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            sample::gLogError << "CUDA Error: " << cudaGetErrorName(s_) << " " << cudaGetErrorString(s_) << std::endl; \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

#define CUBLAS_CALL(call)                                                                                              \
    {                                                                                                                  \
        cublasStatus_t s_ = call;                                                                                      \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            sample::gLogError << "cuBLAS Error: " << s_ << std::endl;                                                  \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(uint8_t*& buffer, T const& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(uint8_t const*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// Static class fields initialization
PluginFieldCollection HardmaxPluginCreator::mFC{};
std::vector<PluginField> HardmaxPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(HardmaxPluginCreator);

namespace
{
constexpr char const* kHARDMAX_NAME{"CustomHardmax"};
constexpr char const* kHARDMAX_VERSION{"1"};
} // namespace

HardmaxPlugin::HardmaxPlugin(int32_t axis)
{
    mAxis = axis;
}

HardmaxPlugin::HardmaxPlugin(void const* serialData, size_t serialLength)
{
    uint8_t const* d = static_cast<uint8_t const*>(serialData);
    uint8_t const* a = d;

    mAxis = readFromBuffer<int32_t>(d);
    mAxisSize = readFromBuffer<int32_t>(d);
    mDimProductOuter = readFromBuffer<int32_t>(d);
    mDimProductInner = readFromBuffer<int32_t>(d);

    ASSERT(d == (a + serialLength));
}

HardmaxPlugin::~HardmaxPlugin()
{
    terminate();
}

int32_t HardmaxPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t HardmaxPlugin::initialize() noexcept
{
    return 0;
}

char const* HardmaxPlugin::getPluginType() const noexcept
{
    return kHARDMAX_NAME;
}

char const* HardmaxPlugin::getPluginVersion() const noexcept
{
    return kHARDMAX_VERSION;
}

nvinfer1::DimsExprs HardmaxPlugin::getOutputDimensions(
    int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    ASSERT(nbInputs == 1);
    ASSERT(index == 0);

    // Dimensions are unchanged
    return inputs[0];
}

void HardmaxPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    ASSERT(
        cublasContext != nullptr && "HardmaxPlugin given a null cuBLAS Context. Was the CUBLAS TacticSource disabled?");
    mCublas = cublasContext;
}

// Detach the plugin object from its execution context.
void HardmaxPlugin::detachFromContext() noexcept {}

int32_t HardmaxPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (inputDesc[0].type != nvinfer1::DataType::kFLOAT)
    {
        return -1;
    }

    CUBLAS_CALL(cublasSetStream(mCublas, stream));

    auto const* data = static_cast<float const*>(inputs[0]);
    auto* result = static_cast<float*>(outputs[0]);

    // Make sure output is initialized to all 0's.
    // Later we will set the correct outputs to be 1's and not touch the rest.
    CUDA_CALL(cudaMemsetAsync(result, 0, mDimProductOuter * mDimProductInner * mAxisSize * sizeof(float), stream));

    // We use the workspace in the case that the first call to 'cublasIsamax' is insufficient.
    // The first half of the workspace we use to copy the values of the axis into, so that we can
    // subtract out the minimum value and call 'cublasIsamax' again. See the comment below.
    // The second half of the workspace will be a costant array of 1's, necessary for our cublasSaxpy call.
    auto* const axisFlat = static_cast<float* const>(workspace);
    float* const ones = axisFlat + mAxisSize;
    float const one = 1.0F;
    CUDRIVER_CALL(cuMemsetD32Async(CUdeviceptr(ones), *reinterpret_cast<int const*>(&one), mAxisSize, stream));

    // This plugin works by parallelizing the argmax operation along a single axis.
    // This is efficient when the axis size is very large compared to the other dimensions.
    //
    // Consider an input shape (1, 512, 3) with axis = 1. This plugin will perform well because
    // the work which is parallelized is over the large 512-element-long axis, and the work that is done
    // serially is over the small 1-element-long and 3-element-long axes.
    //
    // However, when the axis size is small compared to the other dimensions, this plugin will be very
    // inefficient. If the input shape is (1, 512, 3) and the hardmax is over axis = 2, then
    // the work is parallelized over the small 3-element-long axis and the work is done serially over
    // the large 512-element-long axis. A smarter plugin would try to recognize this and parallelize
    // the work which would take longest.
    for (int32_t outer = 0; outer < mDimProductOuter; outer++)
    {
        for (int32_t inner = 0; inner < mDimProductInner; inner++)
        {
            int32_t const axesOffset = outer * mDimProductInner * mAxisSize + inner;
            float const* arr = &data[axesOffset];
            int32_t const stride = mDimProductInner;
            int32_t argmaxResult;
            CUBLAS_CALL(cublasIsamax(mCublas, mAxisSize, arr, stride, &argmaxResult));

            // cublasIsamax returns 1-indexed so convert to 0-indexed
            argmaxResult--;

            // cublasIsamax returns the index of the element with the highest absolute value.
            // If this element is positive, then we know it is also the max.
            // However, if it is negative, we need to
            //      1) Copy the axis into our workspace
            //      2) Subtract the minimum value we found from our array. This ensures that
            //         none of the values are negative, and that the largest element remains
            //         the largest element.
            //      3) Use cublasIsamax to find the largest element again.
            // NOTE: We are using cudaMemcpy instead of cudaMemcpyAsync because we need to know
            //       maxAbsValue before proceeding. However, using synchronous rather than
            //       asynchronous calls inside of enqueue() hurts performance.
            //       This could be fixed by implementing the functionality of this plugin with a kernel
            //       instead of relying only on cuBLAS.
            float maxAbsValue;
            CUDA_CALL(cudaMemcpy(&maxAbsValue, &arr[argmaxResult * stride], sizeof(float), cudaMemcpyDeviceToHost));
            if (maxAbsValue < 0)
            {
                float negMinValue = -maxAbsValue;
                CUBLAS_CALL(cublasScopy(mCublas, mAxisSize, arr, stride, axisFlat, 1));
                CUBLAS_CALL(cublasSaxpy(mCublas, mAxisSize, &negMinValue, ones, 1, axisFlat, 1));
                CUBLAS_CALL(cublasIsamax(mCublas, mAxisSize, axisFlat, 1, &argmaxResult));
                argmaxResult--;
            }

            CUDA_CALL(cudaMemcpyAsync(
                &result[axesOffset + argmaxResult * stride], &one, sizeof(float), cudaMemcpyHostToDevice, stream));
        }
    }
    return cudaPeekAtLastError();
}

size_t HardmaxPlugin::getSerializationSize() const noexcept
{
    return 4 * sizeof(int32_t);
}

void HardmaxPlugin::serialize(void* buffer) const noexcept
{
    // Same order as in deserialize()
    uint8_t* d = static_cast<uint8_t*>(buffer);
    uint8_t* const a = d;

    writeToBuffer(d, mAxis);
    writeToBuffer(d, mAxisSize);
    writeToBuffer(d, mDimProductOuter);
    writeToBuffer(d, mDimProductInner);

    ASSERT(d == a + getSerializationSize());
}

bool HardmaxPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    // No change of type allowed
    if (inOut[0].type != inOut[pos].type)
    {
        return false;
    }

    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
}

void HardmaxPlugin::terminate() noexcept {}

void HardmaxPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* HardmaxPlugin::clone() const noexcept
{
    auto* plugin = new HardmaxPlugin(mAxis);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->mAxisSize = mAxisSize;
    plugin->mDimProductInner = mDimProductInner;
    plugin->mDimProductOuter = mDimProductOuter;
    plugin->mCublas = mCublas;
    return plugin;
}

void HardmaxPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    nvinfer1::Dims const& inDims = in[0].desc.dims;
    nvinfer1::Dims const& outDims = out[0].desc.dims;

    // Check that inputs and outputs have the same dimensions
    ASSERT(inDims.nbDims == outDims.nbDims);
    for (int32_t dim = 0; dim < inDims.nbDims; dim++)
    {
        ASSERT(inDims.d[dim] == outDims.d[dim]);
    }

    // Check that axis is valid
    if (mAxis < 0)
    {
        mAxis += inDims.nbDims;
        ASSERT(mAxis >= 0);
    }
    ASSERT(inDims.nbDims > mAxis);

    // samplesCommon::volume() requires that all dimensions are non-negative.
    // Even in the case of dynamic shapes, the plugin will be configured with
    // resolved shapes before enqueue() is called, so the below member variables
    // will be set correctly.
    if (std::all_of(inDims.d, inDims.d + inDims.nbDims, [](int32_t x) { return x >= 0; }))
    {
        mDimProductOuter = samplesCommon::volume(inDims, 0, mAxis);
        mAxisSize = inDims.d[mAxis];
        mDimProductInner = samplesCommon::volume(inDims, mAxis + 1, inDims.nbDims);
    }
}

nvinfer1::DataType HardmaxPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    ASSERT(inputTypes && nbInputs == 1 && index == 0);
    return inputTypes[0];
}

size_t HardmaxPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // 1st array to store the contents of the working axis
    // 2nd array to store an array of 1's
    return 2 * inputs[0].dims.d[mAxis] * sizeof(float);
}

void HardmaxPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    ASSERT(libNamespace != nullptr);
    mNamespace = libNamespace;
}

char const* HardmaxPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

HardmaxPluginCreator::HardmaxPluginCreator()
{
    mPluginAttributes.clear();

    // Consistent with the ONNX model attr fields
    static auto const axisField = PluginField("axis", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(axisField);

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* HardmaxPluginCreator::getPluginName() const noexcept
{
    return kHARDMAX_NAME;
}

char const* HardmaxPluginCreator::getPluginVersion() const noexcept
{
    return kHARDMAX_VERSION;
}

PluginFieldCollection const* HardmaxPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

char const* HardmaxPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void HardmaxPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    ASSERT(libNamespace != nullptr);
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* HardmaxPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    // Set default value
    int32_t axis = -1;

    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        if (!strcmp(fc->fields[i].name, "axis"))
        {
            ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
            axis = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }

    HardmaxPlugin* plugin = new HardmaxPlugin(axis);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* HardmaxPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    HardmaxPlugin* plugin = new HardmaxPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}
