/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "sampleUtils.h"
#include <iostream>

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
        assert(mFile.good());
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
