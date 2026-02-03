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
#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H

#include "NvInfer.h"
#include "common.h"
#include "half.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

namespace samplesCommon
{

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size)
        , mCapacity(size)
        , mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const
    {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const
    {
        return samplesCommon::getNbBytes(mType, size());
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    //!
    //! \brief Overload of resize that accepts Dims
    //!
    void resize(const nvinfer1::Dims& dims)
    {
        return this->resize(samplesCommon::volume(dims));
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engine, when the I/O tensor volumes
    //! are provided
    //!
    BufferManager(
        std::shared_ptr<nvinfer1::ICudaEngine> engine, std::vector<int64_t> const& volumes, int32_t batchSize = 0)
        : mEngine(engine)
        , mBatchSize(batchSize)
    {
        // Create host and device buffers
        for (int32_t i = 0; i < mEngine->getNbIOTensors(); i++)
        {
            auto const name = engine->getIOTensorName(i);
            mNames[name] = i;

            nvinfer1::DataType type = mEngine->getTensorDataType(name);

            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(volumes[i], type);
            manBuf->hostBuffer = HostBuffer(volumes[i], type);
            void* deviceBuffer = manBuf->deviceBuffer.data();
            mDeviceBindings.emplace_back(deviceBuffer);
            mManagedBuffers.emplace_back(std::move(manBuf));
        }
    }

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engine.
    //!
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, int32_t const batchSize = 0,
        nvinfer1::IExecutionContext const* context = nullptr)
        : mEngine(engine)
        , mBatchSize(batchSize)
    {
        // Create host and device buffers
        for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
        {
            auto const name = engine->getIOTensorName(i);
            mNames[name] = i;

            auto dims = context ? context->getTensorShape(name) : mEngine->getTensorShape(name);
            size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mEngine->getTensorDataType(name);
            int32_t vecDim = mEngine->getTensorVectorizedDim(name);
            if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
            {
                int32_t scalarsPerVec = mEngine->getTensorComponentsPerElement(name);
                dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                vol *= scalarsPerVec;
            }
            vol *= samplesCommon::volume(dims);
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type);
            void* deviceBuffer = manBuf->deviceBuffer.data();
            mDeviceBindings.emplace_back(deviceBuffer);
            mManagedBuffers.emplace_back(std::move(manBuf));
        }
    }

    //!
    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*>& getDeviceBindings()
    {
        return mDeviceBindings;
    }

    //!
    //! \brief Returns a vector of device buffers.
    //!
    std::vector<void*> const& getDeviceBindings() const
    {
        return mDeviceBindings;
    }

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getDeviceBuffer(std::string const& tensorName) const
    {
        return getBuffer(false, tensorName);
    }

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getHostBuffer(std::string const& tensorName) const
    {
        return getBuffer(true, tensorName);
    }

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t size(std::string const& tensorName) const
    {
        auto record = mNames.find(tensorName);
        if (record == mNames.end())
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[record->second]->hostBuffer.nbBytes();
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers synchronously.
    //!
    void copyInputToDevice()
    {
        memcpyBuffers(true, false, false);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    //!
    void copyOutputToHost()
    {
        memcpyBuffers(false, true, false);
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void copyInputToDeviceAsync(cudaStream_t const& stream = 0)
    {
        memcpyBuffers(true, false, true, stream);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void copyOutputToHostAsync(cudaStream_t const& stream = 0)
    {
        memcpyBuffers(false, true, true, stream);
    }

    ~BufferManager() = default;

private:
    void* getBuffer(bool const isHost, std::string const& tensorName) const
    {
        auto record = mNames.find(tensorName);
        if (record == mNames.end())
            return nullptr;
        return (isHost ? mManagedBuffers[record->second]->hostBuffer.data()
                       : mManagedBuffers[record->second]->deviceBuffer.data());
    }

    bool tensorIsInput(const std::string& tensorName) const
    {
        return mEngine->getTensorIOMode(tensorName.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    }

    void memcpyBuffers(bool const copyInput, bool const deviceToHost, bool const async, cudaStream_t const& stream = 0)
    {
        for (auto const& n : mNames)
        {
            void* dstPtr = deviceToHost ? mManagedBuffers[n.second]->hostBuffer.data()
                                        : mManagedBuffers[n.second]->deviceBuffer.data();
            void const* srcPtr = deviceToHost ? mManagedBuffers[n.second]->deviceBuffer.data()
                                              : mManagedBuffers[n.second]->hostBuffer.data();
            size_t const byteSize = mManagedBuffers[n.second]->hostBuffer.nbBytes();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && tensorIsInput(n.first)) || (!copyInput && !tensorIsInput(n.first)))
            {
                if (async)
                    CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                else
                    CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
    int mBatchSize;                                              //!< The batch size for legacy networks, 0 otherwise.
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
    std::vector<void*> mDeviceBindings;              //!< The vector of device buffers needed for engine execution
    std::unordered_map<std::string, int32_t> mNames; //!< The map of tensor name and index pairs
};

} // namespace samplesCommon

#endif // TENSORRT_BUFFERS_H
