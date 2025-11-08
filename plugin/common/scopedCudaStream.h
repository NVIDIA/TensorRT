/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_SCOPED_CUDA_STREAM_H
#define TRT_SCOPED_CUDA_STREAM_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>

namespace nvinfer1
{
namespace pluginInternal
{

// RAII wrapper for CUDA stream
// Automatically manages CUDA stream lifecycle - creation in constructor, destruction in destructor
// Provides safe stream management and prevents memory leaks
class ScopedCudaStream
{
public:
    // Constructor that creates a new CUDA stream with default flags
    ScopedCudaStream()
        : mStream(nullptr)
    {
        cudaError_t result = cudaStreamCreate(&mStream);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(result)));
        }
    }

    // Constructor that creates a new CUDA stream with custom flags
    explicit ScopedCudaStream(uint32_t const flags)
        : mStream(nullptr)
    {
        cudaError_t result = cudaStreamCreateWithFlags(&mStream, flags);
        if (result != cudaSuccess)
        {
            mStream = nullptr;
        }
    }

    // Destructor - automatically destroys the stream
    ~ScopedCudaStream()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    // Delete copy constructor and assignment
    ScopedCudaStream(ScopedCudaStream const&) = delete;
    ScopedCudaStream& operator=(ScopedCudaStream const&) = delete;

    // Move constructor
    ScopedCudaStream(ScopedCudaStream&& other) noexcept
        : mStream(std::exchange(other.mStream, nullptr))
    {
    }

    // Move assignment
    ScopedCudaStream& operator=(ScopedCudaStream&& other) noexcept
    {
        ScopedCudaStream tmp{std::move(other)};
        std::swap(mStream, tmp.mStream);
        return *this;
    }

    // Get the underlying CUDA stream handle
    cudaStream_t get() const noexcept
    {
        return mStream;
    }

    // Implicit conversion to cudaStream_t
    operator cudaStream_t() const noexcept
    {
        return mStream;
    }

    // Check if the stream is valid
    bool isValid() const noexcept
    {
        return mStream != nullptr;
    }

private:
    cudaStream_t mStream;
};

// Helper function to create a unique_ptr to ScopedCudaStream
inline std::unique_ptr<ScopedCudaStream> makeScopedCudaStream(uint32_t const flags = cudaStreamDefault)
{
    if (flags == cudaStreamDefault)
    {
        return std::make_unique<ScopedCudaStream>();
    }
    return std::make_unique<ScopedCudaStream>(flags);
}

} // namespace pluginInternal
} // namespace nvinfer1

#endif // TRT_SCOPED_CUDA_STREAM_H
