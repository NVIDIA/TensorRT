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
#ifndef SAMPLE_NMT_PINNED_HOST_BUFFER_
#define SAMPLE_NMT_PINNED_HOST_BUFFER_

#include "cudaError.h"
#include <cuda_runtime_api.h>
#include <memory>

namespace nmtSample
{
/** \class PinnedHostBuffer
 *
 * \brief wrapper for the pinned host memory region
 *
 */
template <typename T>
class PinnedHostBuffer
{
public:
    typedef std::shared_ptr<PinnedHostBuffer<T>> ptr;

    PinnedHostBuffer(size_t elementCount)
        : mBuffer(nullptr)
    {
        CUDA_CHECK(cudaHostAlloc(&mBuffer, elementCount * sizeof(T), cudaHostAllocDefault));
    }

    virtual ~PinnedHostBuffer()
    {
        if (mBuffer)
        {
            cudaFreeHost(mBuffer);
        }
    }

    operator T*()
    {
        return mBuffer;
    }

    operator const T*() const
    {
        return mBuffer;
    }

protected:
    T* mBuffer;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_PINNED_HOST_BUFFER_
