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

#include "fftPlugin.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <set>
#include <span>
#include <string_view>

namespace nvinfer1::plugin
{

using namespace nvinfer1::pluginInternal;

namespace
{
constexpr char const* kFFT_PLUGIN_VERSION{"1"};
constexpr char const* kFFT_PLUGIN_NAME{"FFTPlugin"};

//! Map a TensorRT element type to the cuFFT type for a real-valued sample.
cudaDataType cudaRealType(DataType dt)
{
    switch (dt)
    {
    case DataType::kFLOAT: return CUDA_R_32F;
    case DataType::kHALF: return CUDA_R_16F;
    case DataType::kBF16: return CUDA_R_16BF;
    default: PLUGIN_VALIDATE(false, "FFTPlugin supports only FP32, FP16 and BF16"); return CUDA_R_32F;
    }
}

//! Map a TensorRT element type to the cuFFT type for an interleaved complex sample.
cudaDataType cudaComplexType(DataType dt)
{
    switch (dt)
    {
    case DataType::kFLOAT: return CUDA_C_32F;
    case DataType::kHALF: return CUDA_C_16F;
    case DataType::kBF16: return CUDA_C_16BF;
    default: PLUGIN_VALIDATE(false, "FFTPlugin supports only FP32, FP16 and BF16"); return CUDA_C_32F;
    }
}

void validateCufft(cufftResult status, char const* what)
{
    PLUGIN_VALIDATE(
        status == CUFFT_SUCCESS, (std::string(what) + " failed with cuFFT error " + std::to_string(status)).c_str());
}

} // namespace

void CufftHandleDeleter::operator()(cufftHandle* handle) const noexcept
{
    if (handle != nullptr)
    {
        getCufftWrapper().cufftDestroy(*handle);
        delete handle;
    }
}

FFTPlugin::FFTPlugin(bool inverse, bool onesided, int32_t ndims)
    : mInverse(inverse)
    , mOnesided(onesided)
    , mNdims(ndims)
{
    PLUGIN_VALIDATE(mNdims >= 1 && mNdims <= 3, "FFTPlugin 'ndims' must be 1, 2 or 3");
}

IPluginCapability* FFTPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

FFTPlugin* FFTPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<FFTPlugin>(mInverse, mOnesided, mNdims);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* FFTPlugin::getPluginName() const noexcept
{
    return kFFT_PLUGIN_NAME;
}

char const* FFTPlugin::getPluginVersion() const noexcept
{
    return kFFT_PLUGIN_VERSION;
}

char const* FFTPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void FFTPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

int32_t FFTPlugin::getNbOutputs() const noexcept
{
    return 1;
}

bool FFTPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1 || nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(pos >= 0 && pos < nbInputs + nbOutputs);

        if (inOut[pos].desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }

        if (nbInputs == 2 && pos == kFFT_LENGTH_INPUT_IDX)
        {
            return inOut[pos].desc.type == DataType::kINT64;
        }

        // Signal input (pos 0) and output (pos nbInputs) share an element type.
        auto type = inOut[pos].desc.type;
        return type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return false;
    }
}

int32_t FFTPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1 || nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);

        DimsExprs output = inputs[kINPUT_TENSOR_IDX];

        if (!mInverse && mOnesided)
        {
            // Forward onesided (R2C): real [..., N] -> complex [..., N/2 + 1, 2].
            PLUGIN_VALIDATE(output.nbDims < Dims::MAX_DIMS);
            int32_t const signalIdx = output.nbDims - 1;
            output.d[signalIdx] = exprBuilder.operation(DimensionOperation::kSUM,
                *exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *output.d[signalIdx], *exprBuilder.constant(2)),
                *exprBuilder.constant(1));
            output.d[output.nbDims] = exprBuilder.constant(2);
            output.nbDims += 1;
        }
        else if (mInverse && mOnesided)
        {
            // Inverse onesided (C2R): complex [..., N/2 + 1, 2] -> real [..., N].
            PLUGIN_VALIDATE(output.nbDims > 1 && output.nbDims < Dims::MAX_DIMS);
            output.nbDims -= 1;
            int32_t const signalIdx = output.nbDims - 1;
            if (nbShapeInputs > 0 && shapeInputs[0].nbDims > 0)
            {
                // Explicit length disambiguates even vs odd N.
                output.d[signalIdx] = shapeInputs[0].d[0];
            }
            else
            {
                // Without an explicit length, assume even N: (freq - 1) * 2.
                output.d[signalIdx] = exprBuilder.operation(DimensionOperation::kPROD,
                    *exprBuilder.operation(DimensionOperation::kSUB, *output.d[signalIdx], *exprBuilder.constant(1)),
                    *exprBuilder.constant(2));
            }
        }
        // C2C keeps the input shape unchanged.

        outputs[kOUTPUT_TENSOR_IDX] = output;
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t FFTPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 1 || nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t FFTPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1 || nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        outputTypes[kOUTPUT_TENSOR_IDX] = inputTypes[kINPUT_TENSOR_IDX];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

void FFTPlugin::computePlanShape(PluginTensorDesc const& in, PluginTensorDesc const& out,
    std::array<int64_t, 3>& signalDims, int64_t& batchSize) const
{
    // The transformed (signal) shape excludes the trailing complex pair. For C2R
    // the real output carries the true signal length, so use the output shape.
    Dims signalShape;
    if (mInverse && mOnesided)
    {
        signalShape = out.dims;
    }
    else
    {
        signalShape = in.dims;
        if (!mOnesided)
        {
            // C2C: drop the trailing [2] complex dimension.
            signalShape.nbDims -= 1;
        }
    }

    PLUGIN_VALIDATE(signalShape.nbDims >= mNdims);
    signalDims = {1, 1, 1};
    int32_t const signalStart = signalShape.nbDims - mNdims;
    for (int32_t i = 0; i < mNdims; ++i)
    {
        signalDims[i] = signalShape.d[signalStart + i];
    }

    batchSize = 1;
    for (int32_t i = 0; i < signalStart; ++i)
    {
        batchSize *= signalShape.d[i];
    }
}

FFTPlanContext const& FFTPlugin::ensurePlan(FFTPlanKey const& key) const
{
    // cuFFT half/bf16 transforms require power-of-two extents along every transformed
    // dimension.
    if (key.dtype == DataType::kHALF || key.dtype == DataType::kBF16)
    {
        PLUGIN_VALIDATE(mNdims <= std::ssize(key.signalDims), "mNdims exceeds signalDims extent");
        PLUGIN_VALIDATE(std::ranges::all_of(std::span(key.signalDims).first(mNdims),
                            [](auto d) { return std::has_single_bit(static_cast<uint64_t>(d)); }),
            "FFTPlugin FP16/BF16 transforms require power-of-two signal lengths");
    }

    std::lock_guard<std::mutex> lock(mCacheMutex);
    if (auto it = mPlanCache.find(key); it != mPlanCache.end())
    {
        return it->second;
    }

    auto handle = std::make_unique<cufftHandle>();
    validateCufft(getCufftWrapper().cufftCreate(handle.get()), "cufftCreate");
    validateCufft(getCufftWrapper().cufftSetAutoAllocation(*handle, 0), "cufftSetAutoAllocation");

    cudaDataType inType;
    cudaDataType outType;
    if (!mInverse && mOnesided)
    {
        inType = cudaRealType(key.dtype);
        outType = cudaComplexType(key.dtype);
    }
    else if (mInverse && mOnesided)
    {
        inType = cudaComplexType(key.dtype);
        outType = cudaRealType(key.dtype);
    }
    else
    {
        inType = cudaComplexType(key.dtype);
        outType = cudaComplexType(key.dtype);
    }

    std::array<int64_t, 3> n{};
    std::copy_n(key.signalDims.begin(), mNdims, n.begin());

    size_t workspaceSize = 0;
    validateCufft(getCufftWrapper().cufftXtMakePlanMany(*handle, mNdims, n.data(), nullptr, 1, 0, inType, nullptr, 1, 0,
                      outType, static_cast<int64_t>(key.batchSize), &workspaceSize, inType),
        "cufftXtMakePlanMany");

    FFTPlanContext context;
    context.handle = CufftHandlePtr(handle.release());
    context.workspaceSize = workspaceSize;
    return mPlanCache.emplace(key, std::move(context)).first->second;
}

int32_t FFTPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1 || nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);

        FFTPlanKey key;
        key.dtype = in[kINPUT_TENSOR_IDX].type;
        computePlanShape(in[kINPUT_TENSOR_IDX], out[kOUTPUT_TENSOR_IDX], key.signalDims, key.batchSize);

        FFTPlanContext const& context = ensurePlan(key);
        std::lock_guard<std::mutex> lock(mCacheMutex);
        mCurrentPlan = context.handle.get();
        mCurrentWorkspaceSize = context.workspaceSize;
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

size_t FFTPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    try
    {
        // We're using the workspace memory for cufft by cufftSetWorkArea in enqueue, so
        // we'll have to size for the worst case using the profile's max dims so that it's
        // large enough for every shape the plan cache will later see.
        FFTPlanKey key;
        key.dtype = inputs[kINPUT_TENSOR_IDX].desc.type;
        PluginTensorDesc inMax = inputs[kINPUT_TENSOR_IDX].desc;
        inMax.dims = inputs[kINPUT_TENSOR_IDX].max;
        PluginTensorDesc outMax = outputs[kOUTPUT_TENSOR_IDX].desc;
        outMax.dims = outputs[kOUTPUT_TENSOR_IDX].max;
        computePlanShape(inMax, outMax, key.signalDims, key.batchSize);
        return ensurePlan(key).workspaceSize;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0;
}

int32_t FFTPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        cufftHandle* plan = nullptr;
        {
            // Safe to immediately release the lock after the we got the plan since plans are
            // never evicted from the map.
            std::lock_guard<std::mutex> lock(mCacheMutex);
            plan = mCurrentPlan;
        }
        PLUGIN_VALIDATE(plan != nullptr, "FFTPlugin has no cuFFT plan. onShapeChange did not run");

        validateCufft(getCufftWrapper().cufftSetStream(*plan, stream), "cufftSetStream");
        validateCufft(getCufftWrapper().cufftSetWorkArea(*plan, workspace), "cufftSetWorkArea");

        int32_t const direction = mInverse ? kCUFFT_INVERSE : kCUFFT_FORWARD;
        validateCufft(getCufftWrapper().cufftXtExec(
                          *plan, const_cast<void*>(inputs[kINPUT_TENSOR_IDX]), outputs[kOUTPUT_TENSOR_IDX], direction),
            "cufftXtExec");
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

IPluginV3* FFTPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* FFTPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mInverseField = mInverse ? 1 : 0;
    mOnesidedField = mOnesided ? 1 : 0;
    mDataToSerialize.emplace_back("inverse", &mInverseField, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("onesided", &mOnesidedField, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("ndims", &mNdims, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

//
// FFTPluginCreator
//

FFTPluginCreator::FFTPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("inverse");
    mPluginAttributes.emplace_back("onesided");
    mPluginAttributes.emplace_back("ndims");
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* FFTPluginCreator::getPluginName() const noexcept
{
    return kFFT_PLUGIN_NAME;
}

char const* FFTPluginCreator::getPluginVersion() const noexcept
{
    return kFFT_PLUGIN_VERSION;
}

PluginFieldCollection const* FFTPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

char const* FFTPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void FFTPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

IPluginV3* FFTPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    int32_t inverse = 0;
    int32_t onesided = 0;
    int32_t ndims = 1;

    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        using namespace std::string_view_literals;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PLUGIN_VALIDATE(fc->fields[i].name != nullptr);
            PLUGIN_VALIDATE(fc->fields[i].data != nullptr);
            auto const* data = static_cast<int32_t const*>(fc->fields[i].data);
            if (fc->fields[i].name == "inverse"sv)
            {
                inverse = *data;
            }
            else if (fc->fields[i].name == "onesided"sv)
            {
                onesided = *data;
            }
            else if (fc->fields[i].name == "ndims"sv)
            {
                ndims = *data;
            }
        }

        PLUGIN_VALIDATE(inverse == 0 || inverse == 1, "FFTPlugin 'inverse' must be 0 or 1");
        PLUGIN_VALIDATE(onesided == 0 || onesided == 1, "FFTPlugin 'onesided' must be 0 or 1");
        PLUGIN_VALIDATE(ndims >= 1 && ndims <= 3, "FFTPlugin 'ndims' must be 1, 2 or 3");

        auto plugin = std::make_unique<FFTPlugin>(inverse != 0, onesided != 0, ndims);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

} // namespace nvinfer1::plugin
