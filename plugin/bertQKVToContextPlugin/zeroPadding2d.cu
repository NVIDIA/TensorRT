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

#include "common/checkMacrosPlugin.h"
#include "zeroPadding2d.h"
#include <array>
#include <cstring>

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

constexpr int32_t kMAX_THREADS_PER_BLOCK{256};

template <typename TDataType>
__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    zeroPadding2dKernel(const TDataType* src, int32_t spitch, TDataType* dst, int32_t dpitch, int32_t height)
{
    int32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t numElements = dpitch * height;
    int32_t numThreads = gridDim.x * blockDim.x;

#pragma unroll
    for (; uid < numElements; uid += numThreads)
    {
        int32_t ty = uid / dpitch;
        if (ty >= height)
        {
            return;
        }
        int32_t tx = uid % dpitch;

        TDataType val = 0;
        if (tx < spitch)
        {
            val = src[ty * spitch + tx];
        }

        dst[ty * dpitch + tx] = val;
    }
}

template <>
__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    zeroPadding2dKernel(const int4* src, int32_t spitch, int4* dst, int32_t dpitch, int32_t height)
{
    int32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t numElements = dpitch * height;
    int32_t numThreads = gridDim.x * blockDim.x;

#pragma unroll
    for (; uid < numElements; uid += numThreads)
    {
        int32_t ty = uid / dpitch;
        if (ty >= height)
        {
            continue;
        }
        int32_t tx = uid % dpitch;

        int4 val{0, 0, 0, 0};
        if (tx < spitch)
        {
            val = src[ty * spitch + tx];
        }

        dst[ty * dpitch + tx] = val;
    }
}

cudaError_t zeroPadding2d(
    const void* src, int32_t spitch, void* dst, int32_t dpitch, int32_t height, cudaStream_t stream)
{
    using kernel_ptr_t = void (*)(const void* src, int32_t spitch, void* dst, int32_t dpitch, int32_t height);
    kernel_ptr_t kernels[5]{reinterpret_cast<kernel_ptr_t>(zeroPadding2dKernel<int8_t>),
        reinterpret_cast<kernel_ptr_t>(zeroPadding2dKernel<int16_t>),
        reinterpret_cast<kernel_ptr_t>(zeroPadding2dKernel<int32_t>),
        reinterpret_cast<kernel_ptr_t>(zeroPadding2dKernel<int64_t>),
        reinterpret_cast<kernel_ptr_t>(zeroPadding2dKernel<int4>)};

    auto select = [](size_t width) -> int32_t {
        if (!(width & 0xF))
        {
            return 4;
        }
        if (!(width & 0x7))
        {
            return 3;
        }
        if (!(width & 0x3))
        {
            return 2;
        }
        if (!(width & 0x1))
        {
            return 1;
        }
        return 0;
    };

    auto kernelId = 4; // 128 bit access
    std::array<size_t, 4> checkAlignment{reinterpret_cast<size_t>(src), static_cast<size_t>(spitch),
        reinterpret_cast<size_t>(dst), static_cast<size_t>(dpitch)};
    for (auto size : checkAlignment)
    {
        auto shiftId = select(size);
        if (shiftId < kernelId)
        {
            kernelId = shiftId;
        }
    }

    spitch >>= kernelId;
    dpitch >>= kernelId;

    int32_t devId;
    PLUGIN_CHECK_CUDA(cudaGetDevice(&devId));
    int32_t numSms;
    PLUGIN_CHECK_CUDA(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, devId));
    auto kernel = kernels[kernelId];
    int32_t block = kMAX_THREADS_PER_BLOCK;
    int32_t grid = (dpitch * height + kMAX_THREADS_PER_BLOCK - 1) / kMAX_THREADS_PER_BLOCK;
    int32_t blocksPerSm;
    PLUGIN_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, block, 0));
    grid = std::min(numSms * blocksPerSm, grid);

    kernel<<<grid, block, 0, stream>>>(src, spitch, dst, dpitch, height);
    return cudaPeekAtLastError();
}

QkvPaddingRunner::QkvPaddingRunner(int32_t headSize, DataType dtype)
{
    PLUGIN_ASSERT(headSize > 0 && headSize <= 64);
    mPaddingHeadSize = (headSize <= 32) ? 32 : 64;

    PLUGIN_ASSERT(dtype == DataType::kHALF || dtype == DataType::kINT8);
    mDtypeSize = (dtype == DataType::kHALF) ? 2 : 1;
}

int32_t QkvPaddingRunner::getPaddingHeadSize()
{
    return mPaddingHeadSize;
}

size_t QkvPaddingRunner::getInputSize(int32_t sumSeqLen, int32_t numHeads)
{
    return (3U * sumSeqLen * numHeads * mPaddingHeadSize * mDtypeSize);
}

size_t QkvPaddingRunner::getOutputSize(int32_t sumSeqLen, int32_t numHeads)
{
    return (1U * sumSeqLen * numHeads * mPaddingHeadSize * mDtypeSize);
}

size_t QkvPaddingRunner::getWorkspaceSize(int32_t sumSeqLen, int32_t numHeads)
{
    constexpr int32_t reserveForAlignment = 16;
    return getInputSize(sumSeqLen, numHeads) + getOutputSize(sumSeqLen, numHeads) + reserveForAlignment;
}

void* QkvPaddingRunner::get16BytesAlignedPointer(void* workspace, size_t offset)
{
    auto addr = reinterpret_cast<uintptr_t>(workspace) + offset;
    auto shift = 16 - (addr & 0xF);
    if (shift == 16)
    {
        shift = 0;
    }
    return reinterpret_cast<void*>(addr + shift);
}

cudaError_t QkvPaddingRunner::pad(
    const void* src, void* workspace, int32_t sumSeqLen, int32_t numHeads, int32_t headSize, cudaStream_t stream)
{
    return zeroPadding2d(
        src, headSize * mDtypeSize, workspace, mPaddingHeadSize * mDtypeSize, 3 * sumSeqLen * numHeads, stream);
}

cudaError_t QkvPaddingRunner::unpad(
    const void* workspace, void* dst, int32_t sumSeqLen, int32_t numHeads, int32_t headSize, cudaStream_t stream)
{
    return zeroPadding2d(
        workspace, mPaddingHeadSize * mDtypeSize, dst, headSize * mDtypeSize, sumSeqLen * numHeads, stream);
}

MhaRunParameter QkvPaddingRunner::patchMhaArgs(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* paddingWorkspace, int32_t sumSeqLen, int32_t numHeads)
{
    MhaRunParameter args;

    std::memcpy(args.inputDesc, inputDesc, 4 * sizeof(PluginTensorDesc));
    auto paddingHiddenSize = numHeads * mPaddingHeadSize;
    args.inputDesc[0].dims.d[1] = 3 * paddingHiddenSize;

    args.outputDesc[0] = outputDesc[0];
    args.outputDesc[0].dims.d[1] = paddingHiddenSize;

    std::memcpy(args.inputs, inputs, 4 * sizeof(void*));
    args.inputs[0] = paddingWorkspace;

    args.outputs[0] = get16BytesAlignedPointer(paddingWorkspace, getInputSize(sumSeqLen, numHeads));

    return args;
}

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
