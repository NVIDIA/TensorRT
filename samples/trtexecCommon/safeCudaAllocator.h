/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SAFE_CUDA_ALLOCTOR_H
#define SAFE_CUDA_ALLOCTOR_H

#include "NvInferSafeRuntime.h" // TRTS-10206: NvInferSafeRuntime.h may be refactored
#include "cuda_runtime.h"
#include "safeCommon.h"
#include "safeErrorRecorder.h"
#include <cuda.h>

namespace nvinfer2::safe
{

// In safe runtime, any single allocation size should not be greater than 16_GiB.
constexpr uint64_t kMAXIMUM_SIZE{17179869184U};

class SafeMemAllocator final : public ISafeMemAllocator
{
public:
    static SafeMemAllocator& instance() noexcept
    {
        static SafeMemAllocator sDefaultAllocatorInstance{};
        return sDefaultAllocatorInstance;
    }

    SafeMemAllocator(SafeMemAllocator const&) = delete;
    SafeMemAllocator(SafeMemAllocator&&) = delete;
    SafeMemAllocator& operator=(SafeMemAllocator const&) & = delete;
    SafeMemAllocator& operator=(SafeMemAllocator&&) & = delete;

private:
    constexpr SafeMemAllocator() noexcept = default;
    ~SafeMemAllocator() noexcept final = default;

    void* allocate(uint64_t const size, uint64_t const alignment, MemoryPlacement const flags, MemoryUsage const usage,
        ISafeRecorder& recorder) noexcept final
    {
        void* memory = nullptr;
        SAFE_ASSERT(size <= kMAXIMUM_SIZE && "allocation size should not exceed 16_GiB.");
        SAFE_ASSERT(alignment != 0U);
        SAFE_ASSERT((alignment & (alignment - 1)) == 0 && "Memory alignment has to be power of 2");
        try
        {
            switch (flags)
            {
            case MemoryPlacement::kCPU_PINNED:
            {
                if ((usage != MemoryUsage::kIMMUTABLE) && (usage != MemoryUsage::kIOTENSOR))
                {
                    safeLogWarning(
                        recorder, "Memory usage for cpu-pinned memory should be either IMMUTABLE or IOTENSOR");
                }
                if (cudaHostAlloc(&memory, size, static_cast<uint32_t>(cudaHostAllocMapped)) != cudaSuccess)
                {
                    safeLogError(recorder, "Cannot allocate PINNED CPU/GPU memory with size = " + std::to_string(size));
                    return nullptr;
                }
            }
            break;

            case MemoryPlacement::kGPU:
            {
                if (cudaMalloc(&memory, size) != cudaSuccess)
                {
                    safeLogError(recorder, "Cannot allocate GPU memory with size = " + std::to_string(size));
                    return nullptr;
                }
            }
            break;
            case MemoryPlacement::kCPU:
            {
                // Use posix_memalign for aligned memory allocation on CPU
                // Note: While aligned_alloc is available in C++17, we use posix_memalign
                // for broader platform compatibility
                SAFE_ASSERT(posix_memalign(&memory, alignment, size) == 0);
            }
            break;
            case MemoryPlacement::kMANAGED:
            case MemoryPlacement::kNONE:
            {
                safeLogError(recorder, "MemoryPlacement::kMANAGED and MemoryPlacement::kNONE are not allowed.");
                return nullptr;
            }
            }

            if (reinterpret_cast<uint64_t>(memory) % alignment != 0U)
            {
                safeLogError(recorder, "Allocated memory is not aligned with " + std::to_string(alignment) + " bytes.");
                deallocate(memory, flags, recorder);
                return nullptr;
            }
        }
        catch (std::exception const& e)
        {
            safeLogError(recorder, e.what());
            return nullptr;
        }
        return memory;
    }

    bool deallocate(void* const memory, MemoryPlacement const flags, ISafeRecorder& recorder) noexcept final
    {
        if (memory == nullptr)
        {
            safeLogWarning(recorder, "Attempting to free nullptr memory.");
            return true;
        }
        try
        {
            switch (flags)
            {
            case MemoryPlacement::kCPU_PINNED:
            {
                return cudaFreeHost(memory) == cudaSuccess;
            }
            case MemoryPlacement::kGPU:
            {
                return cudaFree(memory) == cudaSuccess;
            }
            case MemoryPlacement::kCPU:
            {
                free(memory);
                return true;
            }
            case MemoryPlacement::kMANAGED:
            case MemoryPlacement::kNONE:
            {
                safeLogError(recorder, "MemoryPlacement::kMANAGED and MemoryPlacement::kNONE are not allowed.");
                return false;
            }
            }
        }
        catch (std::exception const& e)
        {
            safeLogError(recorder, e.what());
            return false;
        }
        return false;
    }
};

inline ISafeMemAllocator& getSafeMemAllocator() noexcept
{
    return SafeMemAllocator::instance();
}

} // namespace nvinfer2::safe

#endif // SAFE_CUDA_ALLOCTOR_H
