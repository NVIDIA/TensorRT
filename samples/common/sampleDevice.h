/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRT_SAMPLE_DEVICE_H
#define TRT_SAMPLE_DEVICE_H

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

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

#if CUDA_VERSION < 10000
void cudaSleep(cudaStream_t stream, cudaError_t status, void* sleep)
#else
void cudaSleep(void* sleep)
#endif
{
    std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(*static_cast<int*>(sleep)));
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

    void sleep(int* ms)
    {
#if CUDA_VERSION < 10000
        cudaCheck(cudaStreamAddCallback(mStream, cudaSleep, ms, 0));
#else
        cudaCheck(cudaLaunchHostFunc(mStream, cudaSleep, ms));
#endif
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
        cudaCheck(cudaGraphCreate(&mGraph, 0));
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
        const auto ret = cudaStreamEndCapture(stream.get(), &mGraph);
        assert(ret == cudaErrorStreamCaptureInvalidated);
        assert(mGraph == nullptr);
        // Clean up the above CUDA error.
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

using TrtHostBuffer = TrtCudaBuffer<HostAllocator, HostDeallocator>;

//!
//! \class MirroredBuffer
//! \brief Coupled host and device buffers
//!
class MirroredBuffer
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

} // namespace sample

#endif // TRT_SAMPLE_DEVICE_H
