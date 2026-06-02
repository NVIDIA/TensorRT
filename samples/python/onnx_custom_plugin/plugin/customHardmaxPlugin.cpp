/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <string_view>

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

REGISTER_TENSORRT_PLUGIN(HardmaxPluginCreator);

namespace
{
constexpr char const* kHARDMAX_NAME{"CustomHardmax"};
constexpr char const* kHARDMAX_VERSION{"1"};
} // namespace

HardmaxPlugin::HardmaxPlugin(int32_t axis)
    : mAxis(axis)
{
}

HardmaxPlugin::HardmaxPlugin(HardmaxPlugin const& other)
    : mNamespace(other.mNamespace)
    , mAxisSize(other.mAxisSize)
    , mDimProductOuter(other.mDimProductOuter)
    , mDimProductInner(other.mDimProductInner)
    , mCublas(nullptr)
    , mAxis(other.mAxis)
{
}

HardmaxPlugin::~HardmaxPlugin()
{
    if (mCublas)
    {
        cublasDestroy(mCublas);
    }
}

// IPluginV3 methods

IPluginCapability* HardmaxPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* HardmaxPlugin::clone() noexcept
{
    auto plugin = std::make_unique<HardmaxPlugin>(*this);
    return plugin.release();
}

// IPluginV3OneCore methods

char const* HardmaxPlugin::getPluginName() const noexcept
{
    return kHARDMAX_NAME;
}

char const* HardmaxPlugin::getPluginVersion() const noexcept
{
    return kHARDMAX_VERSION;
}

char const* HardmaxPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// IPluginV3OneBuild methods

int32_t HardmaxPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t HardmaxPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    ASSERT(inputTypes != nullptr);
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t HardmaxPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
    outputs[0] = inputs[0];
    return 0;
}

bool HardmaxPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    // Type changes are not allowed
    if (inOut[0].desc.type != inOut[pos].desc.type)
    {
        return false;
    }

    return inOut[pos].desc.type == DataType::kFLOAT && inOut[pos].desc.format == PluginFormat::kLINEAR;
}

int32_t HardmaxPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    Dims const& inDims = in[0].desc.dims;

    // Normalize negative axis to positive
    if (mAxis < 0)
    {
        mAxis += inDims.nbDims;
        ASSERT(mAxis >= 0);
    }
    ASSERT(inDims.nbDims > mAxis);

    return 0;
}

size_t HardmaxPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    ASSERT(mAxis >= 0);
    // Two arrays are needed:
    // 1. For the contents of the working axis
    // 2. For an array of 1's
    return 2 * inputs[0].max.d[mAxis] * sizeof(float);
}

// IPluginV3OneRuntime methods

int32_t HardmaxPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    Dims const& inDims = in[0].dims;

    // Axis should already be normalized by configurePlugin, but handle it regardless to be safe.
    if (mAxis < 0)
    {
        mAxis += inDims.nbDims;
        ASSERT(mAxis >= 0);
    }
    ASSERT(inDims.nbDims > mAxis);

    mDimProductOuter = samplesCommon::volume(inDims, 0, mAxis);
    mAxisSize = inDims.d[mAxis];
    mDimProductInner = samplesCommon::volume(inDims, mAxis + 1, inDims.nbDims);

    return 0;
}

int32_t HardmaxPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (inputDesc[0].type != DataType::kFLOAT)
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

IPluginV3* HardmaxPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    auto* cloned = static_cast<HardmaxPlugin*>(clone());
    if (cloned == nullptr)
    {
        return nullptr;
    }
    cublasStatus_t ret = cublasCreate(&cloned->mCublas);
    ASSERT(ret == CUBLAS_STATUS_SUCCESS && cloned->mCublas != nullptr && "Failed to create cublasHandle_t.");
    return cloned;
}

PluginFieldCollection const* HardmaxPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("axis", &mAxis, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

void HardmaxPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    ASSERT(libNamespace != nullptr);
    mNamespace = libNamespace;
}

// HardmaxPluginCreator methods

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

IPluginV3* HardmaxPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    using namespace std::string_view_literals;
    // Set default value
    int32_t axis = -1;

    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].name == "axis"sv)
        {
            ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
            axis = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }

    auto plugin = std::make_unique<HardmaxPlugin>(axis);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin.release();
}
