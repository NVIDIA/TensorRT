/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_SAMPLE_DEVICE_H
#define TRT_SAMPLE_DEVICE_H

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

#include "common.h"
#include "sampleUtils.h"

namespace sample
{

class TrtCudaEvent;

namespace
{

void cudaSleep(void* sleep)
{
    std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(*static_cast<float*>(sleep)));
}

} // namespace

//!
//! \class TrtCudaStream
//! \brief Managed CUDA stream
//!
class TrtCudaStream
{
public:
    TrtCudaStream()
    {
        CHECK(cudaStreamCreate(&mStream));
    }

    TrtCudaStream(const TrtCudaStream&) = delete;

    TrtCudaStream& operator=(const TrtCudaStream&) = delete;

    TrtCudaStream(TrtCudaStream&&) = delete;

    TrtCudaStream& operator=(TrtCudaStream&&) = delete;

    ~TrtCudaStream()
    {
        CHECK(cudaStreamDestroy(mStream));
    }

    cudaStream_t get() const
    {
        return mStream;
    }

    void synchronize()
    {
        CHECK(cudaStreamSynchronize(mStream));
    }

    void wait(TrtCudaEvent& event);

    void sleep(float* ms)
    {
        CHECK(cudaLaunchHostFunc(mStream, cudaSleep, ms));
    }

private:
    cudaStream_t mStream{};
};

//!
//! \class TrtCudaEvent
//! \brief Managed CUDA event
//!
class TrtCudaEvent
{
public:
    explicit TrtCudaEvent(bool blocking = true)
    {
        const uint32_t flags = blocking ? cudaEventBlockingSync : cudaEventDefault;
        CHECK(cudaEventCreateWithFlags(&mEvent, flags));
    }

    TrtCudaEvent(const TrtCudaEvent&) = delete;

    TrtCudaEvent& operator=(const TrtCudaEvent&) = delete;

    TrtCudaEvent(TrtCudaEvent&&) = delete;

    TrtCudaEvent& operator=(TrtCudaEvent&&) = delete;

    ~TrtCudaEvent()
    {
        CHECK(cudaEventDestroy(mEvent));
    }

    cudaEvent_t get() const
    {
        return mEvent;
    }

    void record(const TrtCudaStream& stream)
    {
        CHECK(cudaEventRecord(mEvent, stream.get()));
    }

    void synchronize() const
    {
        CHECK(cudaEventSynchronize(mEvent));
    }

    // Returns time elapsed time in milliseconds
    float operator-(const TrtCudaEvent& e) const
    {
        // Synchronize both events to ensure they have completed before calculating elapsed time
        synchronize();
        e.synchronize();

        float time{0};
        CHECK(cudaEventElapsedTime(&time, e.get(), get()));
        return time;
    }

private:
    cudaEvent_t mEvent{};
};

inline void TrtCudaStream::wait(TrtCudaEvent& event)
{
    CHECK(cudaStreamWaitEvent(mStream, event.get(), 0));
}

//!
//! \class TrtCudaGraph
//! \brief Managed CUDA graph
//!
class TrtCudaGraph
{
public:
    explicit TrtCudaGraph() = default;

    TrtCudaGraph(const TrtCudaGraph&) = delete;

    TrtCudaGraph& operator=(const TrtCudaGraph&) = delete;

    TrtCudaGraph(TrtCudaGraph&&) = delete;

    TrtCudaGraph& operator=(TrtCudaGraph&&) = delete;

    ~TrtCudaGraph()
    {
        if (mGraphExec)
        {
            cudaGraphExecDestroy(mGraphExec);
        }
    }

    void beginCapture(TrtCudaStream& stream)
    {
        CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    }

    bool launch(TrtCudaStream& stream)
    {
        return cudaGraphLaunch(mGraphExec, stream.get()) == cudaSuccess;
    }

    void endCapture(TrtCudaStream& stream)
    {
        CHECK(cudaStreamEndCapture(stream.get(), &mGraph));
        CHECK(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
        CHECK(cudaGraphDestroy(mGraph));
    }

    void endCaptureOnError(TrtCudaStream& stream)
    {
        // There are two possibilities why stream capture would fail:
        // (1) stream is in cudaErrorStreamCaptureInvalidated state.
        // (2) TRT reports a failure.
        // In case (1), the returning mGraph should be nullptr.
        // In case (2), the returning mGraph is not nullptr, but it should not be used.
        const auto ret = cudaStreamEndCapture(stream.get(), &mGraph);
        if (ret == cudaErrorStreamCaptureInvalidated)
        {
            assert(mGraph == nullptr);
        }
        else
        {
            CHECK(ret);
            assert(mGraph != nullptr);
            CHECK(cudaGraphDestroy(mGraph));
            mGraph = nullptr;
        }
        // Clean up any CUDA error.
        cudaGetLastError();
        sample::gLogWarning << "The CUDA graph capture on the stream has failed." << std::endl;
    }

private:
    cudaGraph_t mGraph{};
    cudaGraphExec_t mGraphExec{};
};

//!
//! \class TrtCudaBuffer
//! \brief Managed buffer for host and device
//!
template <typename A, typename D>
class TrtCudaBuffer
{
public:
    TrtCudaBuffer() = default;

    TrtCudaBuffer(const TrtCudaBuffer&) = delete;

    TrtCudaBuffer& operator=(const TrtCudaBuffer&) = delete;

    TrtCudaBuffer(TrtCudaBuffer&& rhs)
    {
        reset(rhs.mPtr, rhs.mSize);
        rhs.mPtr = nullptr;
        rhs.mSize = 0;
    }

    TrtCudaBuffer& operator=(TrtCudaBuffer&& rhs)
    {
        if (this != &rhs)
        {
            reset(rhs.mPtr, rhs.mSize);
            rhs.mPtr = nullptr;
            rhs.mSize = 0;
        }
        return *this;
    }

    ~TrtCudaBuffer()
    {
        reset();
    }

    TrtCudaBuffer(size_t size)
    {
        A()(&mPtr, size);
        mSize = size;
    }

    void allocate(size_t size)
    {
        reset();
        A()(&mPtr, size);
        mSize = size;
    }

    void reset(void* ptr = nullptr, size_t size = 0)
    {
        if (mPtr)
        {
            D()(mPtr);
        }
        mPtr = ptr;
        mSize = size;
    }

    void* get() const
    {
        return mPtr;
    }

    size_t getSize() const
    {
        return mSize;
    }

private:
    void* mPtr{nullptr};
    size_t mSize{0};
};

struct DeviceAllocator
{
    void operator()(void** ptr, size_t size)
    {
        CHECK(cudaMalloc(ptr, size));
    }
};

struct DeviceDeallocator
{
    void operator()(void* ptr)
    {
        CHECK(cudaFree(ptr));
    }
};

struct ManagedAllocator
{
    void operator()(void** ptr, size_t size)
    {
        CHECK(cudaMallocManaged(ptr, size));
    }
};

struct HostAllocator
{
    //! Attempts to allocate size bytes on host, pointing *ptr to the start.
    //! First attempts to allocate pinned memory using cudaMallocHost(ptr, size), failing that, warns to gLogWarning and
    //! falls back to ::operator new(size) to allocate pageable memory. If that still fails, an exception may be thrown.
    void operator()(void** ptr, size_t size)
    {
        // Try allocating pinned host memory.
        cudaError_t ret = cudaMallocHost(ptr, size);

        // If we cannot allocate pinned host memory, allocate pageable host memory instead and print a warning.
        if (ret != cudaSuccess)
        {
            // Clean up the last cuda error.
            (void) cudaGetLastError();

            sample::gLogWarning << "cudaMallocHost() call with ptr=" << ptr << " and size=" << size
                                << " returns a cuda error: " << cudaGetErrorString(ret) << std::endl;
            sample::gLogWarning << "Allocate pageable host memory instead of pinned host memory. H2D and D2H copy "
                                   "latencies may become longer."
                                << std::endl;
            *ptr = ::operator new(size);

            // Make sure there is no remaining cuda error at this point.
            CHECK(cudaGetLastError());
        }
    }
};

struct HostDeallocator
{
    //! Attempts to deallocate the host memory allocated by HostAllocator.
    //! It first checks if ptr is a pinned or pageable host memory. If pinned, call cudaFreeHost() to free it. If
    //! pageable, call ::operator delete() to free it. If ptr is neither of them, an error is printed and the program
    //! exits.
    void operator()(void* ptr)
    {
        // Check if the host memory pointer is pinned or pageable.
        cudaPointerAttributes attrs;
        CHECK(cudaPointerGetAttributes(&attrs, ptr));

        // If pinned, call cudaFreeHost() to deallocate it.
        if (attrs.type == cudaMemoryTypeHost)
        {
            CHECK(cudaFreeHost(ptr));
        }
        // If pageable, delete it directly.
        else if (attrs.type == cudaMemoryTypeUnregistered)
        {
            ::operator delete(ptr);
        }
        // The host memory pointer should not be of any other types.
        else
        {
            sample::gLogError << "Unexpected cuda memory type:" << static_cast<int32_t>(attrs.type) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

using TrtDeviceBuffer = TrtCudaBuffer<DeviceAllocator, DeviceDeallocator>;
using TrtManagedBuffer = TrtCudaBuffer<ManagedAllocator, DeviceDeallocator>;

using TrtHostBuffer = TrtCudaBuffer<HostAllocator, HostDeallocator>;

//!
//! \class MirroredBuffer
//! \brief Coupled host and device buffers
//!
class IMirroredBuffer
{
public:
    //!
    //! Allocate memory for the mirrored buffer give the size
    //! of the allocation.
    //!
    virtual void allocate(size_t size) = 0;

    //!
    //! Get the pointer to the device side buffer.
    //!
    //! \return pointer to device memory or nullptr if uninitialized.
    //!
    virtual void* getDeviceBuffer() const = 0;

    //!
    //! Get the pointer to the host side buffer.
    //!
    //! \return pointer to host memory or nullptr if uninitialized.
    //!
    virtual void* getHostBuffer() const = 0;

    //!
    //! Copy the memory from host to device.
    //!
    virtual void hostToDevice(TrtCudaStream& stream) = 0;

    //!
    //! Copy the memory from device to host.
    //!
    virtual void deviceToHost(TrtCudaStream& stream) = 0;

    //!
    //! Interface to get the size of the memory
    //!
    //! \return the size of memory allocated.
    //!
    virtual size_t getSize() const = 0;

    //!
    //! Virtual destructor declaraion
    //!
    virtual ~IMirroredBuffer() = default;

}; // class IMirroredBuffer

//!
//! Class to have a separate memory buffer for discrete device and host allocations.
//!
class DiscreteMirroredBuffer : public IMirroredBuffer
{
public:
    void allocate(size_t size) override
    {
        mSize = size;
        mHostBuffer.allocate(size);
        mDeviceBuffer.allocate(size);
    }

    void* getDeviceBuffer() const override
    {
        return mDeviceBuffer.get();
    }

    void* getHostBuffer() const override
    {
        return mHostBuffer.get();
    }

    void hostToDevice(TrtCudaStream& stream) override
    {
        CHECK(cudaMemcpyAsync(mDeviceBuffer.get(), mHostBuffer.get(), mSize, cudaMemcpyHostToDevice, stream.get()));
    }

    void deviceToHost(TrtCudaStream& stream) override
    {
        CHECK(cudaMemcpyAsync(mHostBuffer.get(), mDeviceBuffer.get(), mSize, cudaMemcpyDeviceToHost, stream.get()));
    }

    size_t getSize() const override
    {
        return mSize;
    }

private:
    size_t mSize{0};
    TrtHostBuffer mHostBuffer;
    TrtDeviceBuffer mDeviceBuffer;
}; // class DiscreteMirroredBuffer

//!
//! Class to have a unified memory buffer for embedded devices.
//!
class UnifiedMirroredBuffer : public IMirroredBuffer
{
public:
    void allocate(size_t size) override
    {
        mSize = size;
        mBuffer.allocate(size);
    }

    void* getDeviceBuffer() const override
    {
        return mBuffer.get();
    }

    void* getHostBuffer() const override
    {
        return mBuffer.get();
    }

    void hostToDevice(TrtCudaStream& stream) override
    {
        // Does nothing since we are using unified memory.
    }

    void deviceToHost(TrtCudaStream& stream) override
    {
        // Does nothing since we are using unified memory.
    }

    size_t getSize() const override
    {
        return mSize;
    }

private:
    size_t mSize{0};
    TrtManagedBuffer mBuffer;
}; // class UnifiedMirroredBuffer

//!
//! Class to allocate memory for outputs with data-dependent shapes. The sizes of those are unknown so pre-allocation is
//! not possible.
//!
class OutputAllocator : public nvinfer1::IOutputAllocator
{
public:
    //! Construct, using buffer as the backing storage:
    explicit OutputAllocator(std::unique_ptr<IMirroredBuffer> buffer)
        : mBuffer{std::move(buffer)}
    {
        ASSERT(mBuffer);
    }

    ~OutputAllocator() override = default;

    void* reallocateOutput(
        char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override
    {
        // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
        // even for empty tensors, so allocate a dummy byte.
        size = std::max(size, static_cast<uint64_t>(1));
        if (size > mSize)
        {
            mBuffer->allocate(roundUp(size, alignment));
            mSize = size;
        }
        return mBuffer->getDeviceBuffer();
    }

    //! IMirroredBuffer does not implement Async allocation, hence this is just a wrap around
    void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment,
        cudaStream_t /*stream*/) noexcept override
    {
        return reallocateOutput(tensorName, currentMemory, size, alignment);
    }

    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override
    {
        mFinalDims = dims;
    }

    IMirroredBuffer* getBuffer()
    {
        return mBuffer.get();
    }

    nvinfer1::Dims getFinalDims()
    {
        return mFinalDims;
    }

private:
    std::unique_ptr<IMirroredBuffer> mBuffer;
    uint64_t mSize{};
    nvinfer1::Dims mFinalDims;
};

//! Set the GPU to run the inference on.
void setCudaDevice(int32_t device, std::ostream& os);

//! Get the CUDA version of the current CUDA driver.
int32_t getCudaDriverVersion();

//! Get the CUDA version of the current CUDA runtime.
int32_t getCudaRuntimeVersion();


} // namespace sample

#endif // TRT_SAMPLE_DEVICE_H
