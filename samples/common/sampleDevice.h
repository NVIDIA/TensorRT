/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <thread>
#include <cuda_runtime.h>

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

}

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

class TrtCudaEvent
{
public:

    TrtCudaEvent(unsigned int flags)
    {
        cudaCheck(cudaEventCreateWithFlags(&mEvent, flags));
    }

    TrtCudaEvent() = default;

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

    void reset(unsigned int flags = cudaEventDefault)
    {
        cudaCheck(cudaEventDestroy(mEvent));
        cudaCheck(cudaEventCreateWithFlags(&mEvent, flags));
    }

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
    void operator()(void** ptr, size_t size) { cudaCheck(cudaMalloc(ptr, size)); }
};

struct DeviceDeallocator
{
    void operator()(void* ptr) { cudaCheck(cudaFree(ptr)); }
};

struct HostAllocator
{
    void operator()(void** ptr, size_t size) { cudaCheck(cudaMallocHost(ptr, size)); }
};

struct HostDeallocator
{
    void operator()(void* ptr) { cudaCheck(cudaFreeHost(ptr)); }
};

using TrtDeviceBuffer = TrtCudaBuffer<DeviceAllocator, DeviceDeallocator>;

using TrtHostBuffer = TrtCudaBuffer<HostAllocator, HostDeallocator>;

} // namespace sample

#endif // TRT_SAMPLE_DEVICE_H
