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
#ifndef TRT_PLUGIN_H
#define TRT_PLUGIN_H
#include "NvInferPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "cublasWrapper.h"
#include "cudnnWrapper.h"
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

// Enumerator for status
typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1
{

namespace pluginInternal
{

class BasePlugin : public IPluginV2
{
protected:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    std::string mNamespace;
};

class BaseCreator : public IPluginCreator
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace;
};

std::shared_ptr<nvinfer1::pluginInternal::CudnnWrapper> createPluginCudnnWrapper(
    void* executionContextIdentifier, char const* callerPluginName);
std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> createPluginCublasWrapper(void* executionContextIdentifier);
} // namespace pluginInternal

namespace plugin
{

// Write values into buffer
template <typename Type, typename BufferType>
void write(BufferType*& buffer, Type const& val)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    std::memcpy(buffer, &val, sizeof(Type));
    buffer += sizeof(Type);
}

// Read values from buffer
template <typename OutType, typename BufferType>
OutType read(BufferType const*& buffer)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    OutType val{};
    std::memcpy(&val, static_cast<void const*>(buffer), sizeof(OutType));
    buffer += sizeof(OutType);
    return val;
}

inline int32_t getTrtSmVersionDec(int32_t majorVersion, int32_t minorVersion)
{
    return majorVersion * 10 + minorVersion;
}

//! Represents the compute capability of a device.
//! This pertains to virtual architectures represented by the intermediate PTX format.
//! This is distinct from the SM version.
//! See https://forums.developer.nvidia.com/t/how-should-i-use-correctly-the-sm-xx-and-compute-xx/219160
struct DeviceComputeCapability
{
    int32_t major{};
    int32_t minor{};

    //! \return the compute capability of the CUDA device with the given \p deviceIndex.
    [[nodiscard]] static DeviceComputeCapability forDevice(int32_t deviceIndex)
    {
        int32_t major{0};
        int32_t minor{0};
        PLUGIN_CUASSERT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
        PLUGIN_CUASSERT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));
        return {major, minor};
    }
};

inline int32_t getSmVersion()
{
    int32_t device{-1};
    PLUGIN_CHECK_CUDA(cudaGetDevice(&device));
    auto const cc = DeviceComputeCapability::forDevice(device);
    return getTrtSmVersionDec(cc.major, cc.minor);
}

// Check that all required field names are present in the PluginFieldCollection.
// If not, throw a PluginError with a message stating which fields are missing.
void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc);

template <typename Dtype>
struct CudaBind
{
    size_t mSize;
    void* mPtr;

    CudaBind(size_t size)
    {
        mSize = size;
        PLUGIN_CUASSERT(cudaMalloc(&mPtr, sizeof(Dtype) * mSize));
    }

    ~CudaBind()
    {
        if (mPtr != nullptr)
        {
            PLUGIN_CUASSERT(cudaFree(mPtr));
            mPtr = nullptr;
        }
    }
};

// Convert a 64-bit dimension to a 32-bit dimension.
// Throw exception if it doesn't fit.
int32_t dimToInt32(int64_t);

// Helper function to determine whether memory pool support is available on the device.
bool supportsMemPoolsHelper();

// Wrapper function around the helper to keep the result in a static variable to avoid mulitple calls to CUDA APIs.
bool supportsMemPools();
} // namespace plugin
} // namespace nvinfer1

#ifndef DEBUG

#define PLUGIN_CHECK(status)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
            exit(EXIT_FAILURE);                                                                                                   \
    } while (0)

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_BAD_PARAM;                                                                                   \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_FAILURE;                                                                                     \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)

#else

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__);                                      \
            return STATUS_BAD_PARAM;                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__);                                        \
            return STATUS_FAILURE;                                                                                     \
        }                                                                                                              \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));                        \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CHECK(status)                                                                                           \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status));                      \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                              \
    }

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#endif // DEBUG

#endif // TRT_PLUGIN_H
