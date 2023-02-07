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

#ifndef TRT_SAMPLE_DEVICE_H
#define TRT_SAMPLE_DEVICE_H

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

#include "sampleUtils.h"

namespace sample
{

inline void cudaCheck(cudaError_t ret, std::ostream& err = std::cerr)
{
    if (ret != cudaSuccess)
    {
        err << "Cuda failure: " << cudaGetErrorString(ret) << std::endl;
        abort();
    }
}

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
        cudaCheck(cudaStreamCreate(&mStream));
    }

    TrtCudaStream(const TrtCudaStream&) = delete;

    TrtCudaStream& operator=(const TrtCudaStream&) = delete;

    TrtCudaStream(TrtCudaStream&&) = delete;

    TrtCudaStream& operator=(TrtCudaStream&&) = delete;

    ~TrtCudaStream()
    {
        cudaCheck(cudaStreamDestroy(mStream));
    }

    cudaStream_t get() const
    {
        return mStream;
    }

    void synchronize()
    {
        cudaCheck(cudaStreamSynchronize(mStream));
    }

    void wait(TrtCudaEvent& event);

    void sleep(float* ms)
    {
        cudaCheck(cudaLaunchHostFunc(mStream, cudaSleep, ms));
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
        cudaCheck(cudaEventCreateWithFlags(&mEvent, flags));
    }

    TrtCudaEvent(const TrtCudaEvent&) = delete;

    TrtCudaEvent& operator=(const TrtCudaEvent&) = delete;

    TrtCudaEvent(TrtCudaEvent&&) = delete;

    TrtCudaEvent& operator=(TrtCudaEvent&&) = delete;

    ~TrtCudaEvent()
    {
        cudaCheck(cudaEventDestroy(mEvent));
    }

    cudaEvent_t get() const
    {
        return mEvent;
    }

    void record(const TrtCudaStream& stream)
    {
        cudaCheck(cudaEventRecord(mEvent, stream.get()));
    }

    void synchronize()
    {
        cudaCheck(cudaEventSynchronize(mEvent));
    }

    // Returns time elapsed time in milliseconds
    float operator-(const TrtCudaEvent& e) const
    {
        float time{0};
        cudaCheck(cudaEventElapsedTime(&time, e.get(), get()));
        return time;
    }

private:
    cudaEvent_t mEvent{};
};

inline void TrtCudaStream::wait(TrtCudaEvent& event)
{
    cudaCheck(cudaStreamWaitEvent(mStream, event.get(), 0));
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
        cudaCheck(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    }

    bool launch(TrtCudaStream& stream)
    {
        return cudaGraphLaunch(mGraphExec, stream.get()) == cudaSuccess;
    }

    void endCapture(TrtCudaStream& stream)
    {
        cudaCheck(cudaStreamEndCapture(stream.get(), &mGraph));
        cudaCheck(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphDestroy(mGraph));
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
            assert(ret == cudaSuccess);
            assert(mGraph != nullptr);
            cudaCheck(cudaGraphDestroy(mGraph));
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
        reset(rhs.mPtr);
        rhs.mPtr = nullptr;
    }

    TrtCudaBuffer& operator=(TrtCudaBuffer&& rhs)
    {
        if (this != &rhs)
        {
            reset(rhs.mPtr);
            rhs.mPtr = nullptr;
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
    }

    void allocate(size_t size)
    {
        reset();
        A()(&mPtr, size);
    }

    void reset(void* ptr = nullptr)
    {
        if (mPtr)
        {
            D()(mPtr);
        }
        mPtr = ptr;
    }

    void* get() const
    {
        return mPtr;
    }

private:
    void* mPtr{nullptr};
};

struct DeviceAllocator
{
    void operator()(void** ptr, size_t size)
    {
        cudaCheck(cudaMalloc(ptr, size));
    }
};

struct DeviceDeallocator
{
    void operator()(void* ptr)
    {
        cudaCheck(cudaFree(ptr));
    }
};

struct ManagedAllocator
{
    void operator()(void** ptr, size_t size)
    {
        cudaCheck(cudaMallocManaged(ptr, size));
    }
};

struct HostAllocator
{
    void operator()(void** ptr, size_t size)
    {
        cudaCheck(cudaMallocHost(ptr, size));
    }
};

struct HostDeallocator
{
    void operator()(void* ptr)
    {
        cudaCheck(cudaFreeHost(ptr));
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
    void allocate(size_t size)
    {
        mSize = size;
        mHostBuffer.allocate(size);
        mDeviceBuffer.allocate(size);
    }

    void* getDeviceBuffer() const
    {
        return mDeviceBuffer.get();
    }

    void* getHostBuffer() const
    {
        return mHostBuffer.get();
    }

    void hostToDevice(TrtCudaStream& stream)
    {
        cudaCheck(cudaMemcpyAsync(mDeviceBuffer.get(), mHostBuffer.get(), mSize, cudaMemcpyHostToDevice, stream.get()));
    }

    void deviceToHost(TrtCudaStream& stream)
    {
        cudaCheck(cudaMemcpyAsync(mHostBuffer.get(), mDeviceBuffer.get(), mSize, cudaMemcpyDeviceToHost, stream.get()));
    }

    size_t getSize() const
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
    void allocate(size_t size)
    {
        mSize = size;
        mBuffer.allocate(size);
    }

    void* getDeviceBuffer() const
    {
        return mBuffer.get();
    }

    void* getHostBuffer() const
    {
        return mBuffer.get();
    }

    void hostToDevice(TrtCudaStream& stream)
    {
        // Does nothing since we are using unified memory.
    }

    void deviceToHost(TrtCudaStream& stream)
    {
        // Does nothing since we are using unified memory.
    }

    size_t getSize() const
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
    OutputAllocator(IMirroredBuffer* buffer)
        : mBuffer(buffer)
    {
    }

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

    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override {}

    IMirroredBuffer* getBuffer()
    {
        return mBuffer.get();
    }

    virtual ~OutputAllocator() {}

private:
    std::unique_ptr<IMirroredBuffer> mBuffer;
    uint64_t mSize{};
};

inline void setCudaDevice(int device, std::ostream& os)
{
    cudaCheck(cudaSetDevice(device));

    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, device));

    // clang-format off
    os << "=== Device Information ===" << std::endl;
    os << "Selected Device: "      << properties.name                                               << std::endl;
    os << "Compute Capability: "   << properties.major << "." << properties.minor                   << std::endl;
    os << "SMs: "                  << properties.multiProcessorCount                                << std::endl;
    os << "Compute Clock Rate: "   << properties.clockRate / 1000000.0F << " GHz"                   << std::endl;
    os << "Device Global Memory: " << (properties.totalGlobalMem >> 20) << " MiB"                   << std::endl;
    os << "Shared Memory per SM: " << (properties.sharedMemPerMultiprocessor >> 10) << " KiB"       << std::endl;
    os << "Memory Bus Width: "     << properties.memoryBusWidth << " bits"
                        << " (ECC " << (properties.ECCEnabled != 0 ? "enabled" : "disabled") << ")" << std::endl;
    os << "Memory Clock Rate: "    << properties.memoryClockRate / 1000000.0F << " GHz"             << std::endl;
    // clang-format on
}

inline int32_t getCudaDriverVersion()
{
    int32_t version{-1};
    cudaCheck(cudaDriverGetVersion(&version));
    return version;
}

inline int32_t getCudaRuntimeVersion()
{
    int32_t version{-1};
    cudaCheck(cudaRuntimeGetVersion(&version));
    return version;
}

} // namespace sample

#endif // TRT_SAMPLE_DEVICE_H
