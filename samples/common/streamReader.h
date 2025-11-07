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

#ifndef STREAM_READER_H
#define STREAM_READER_H


#include "NvInferRuntime.h"
#include <fstream>
#include "sampleUtils.h"

namespace samplesCommon
{

//! Implements the TensorRT IStreamReader to allow deserializing an engine directly from the plan file.
class FileStreamReader final : public nvinfer1::IStreamReader
{
public:
    bool open(std::string filepath)
    {
        mFile.open(filepath, std::ios::binary);
        return mFile.is_open();
    }

    void close()
    {
        if (mFile.is_open())
        {
            mFile.close();
        }
    }

    ~FileStreamReader() final
    {
        close();
    }

    int64_t read(void* dest, int64_t bytes) final
    {
        if (!mFile.good())
        {
            return -1;
        }
        mFile.read(static_cast<char*>(dest), bytes);
        return mFile.gcount();
    }

    void reset()
    {
        ASSERT(mFile.good());
        mFile.seekg(0);
    }

    bool isOpen() const
    {
        return mFile.is_open();
    }

private:
    std::ifstream mFile;
};

//! Implements the TensorRT IStreamReaderV2 interface to allow deserializing an engine directly from the plan file.
//! Supports seeking to a position within the file, and reading directly to device pointers.
//! This implementation is not optimized, and will not provide performance improvements over the existing reader.
class AsyncStreamReader final : public nvinfer1::IStreamReaderV2
{
public:
    bool open(std::string const& filepath)
    {
        mFile.open(filepath, std::ios::binary);
        return mFile.is_open();
    }

    void close()
    {
        if (mFile.is_open())
        {
            mFile.close();
        }
    }

    ~AsyncStreamReader() final
    {
        close();
    }

    bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept final
    {
        switch (where)
        {
        case (nvinfer1::SeekPosition::kSET): mFile.seekg(offset, std::ios_base::beg); break;
        case (nvinfer1::SeekPosition::kCUR): mFile.seekg(offset, std::ios_base::cur); break;
        case (nvinfer1::SeekPosition::kEND): mFile.seekg(offset, std::ios_base::end); break;
        }
        return mFile.good();
    }

    int64_t read(void* destination, int64_t nbBytes, cudaStream_t stream) noexcept final
    {
        if (!mFile.good())
        {
            return -1;
        }

        cudaPointerAttributes attributes;
        ASSERT(cudaPointerGetAttributes(&attributes, destination) == cudaSuccess);

        // from CUDA 11 onward, host pointers are return cudaMemoryTypeUnregistered
        if (attributes.type == cudaMemoryTypeHost || attributes.type == cudaMemoryTypeUnregistered)
        {
            mFile.read(static_cast<char*>(destination), nbBytes);
            return mFile.gcount();
        }
        else if (attributes.type == cudaMemoryTypeDevice)
        {
            // Set up a temp buffer to read into if reading into device memory.
            std::unique_ptr<char[]> tmpBuf{new char[nbBytes]};
            mFile.read(tmpBuf.get(), nbBytes);
            // cudaMemcpyAsync into device storage.
            ASSERT(cudaMemcpyAsync(destination, tmpBuf.get(), nbBytes, cudaMemcpyHostToDevice, stream) == cudaSuccess);
            // No race between the copying and freeing of tmpBuf, because cudaMemcpyAsync will
            // return once the pageable buffer has been copied to the staging memory for DMA transfer
            // to device memory.
            return mFile.gcount();
        }
        return -1;
    }

    void reset()
    {
        ASSERT(mFile.good());
        mFile.seekg(0);
    }

    bool isOpen() const
    {
        return mFile.is_open();
    }

private:
    std::ifstream mFile;
};


} // namespace samplesCommon

#endif // STREAM_READER_H
