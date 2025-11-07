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

#ifndef CUDA_DRIVER_WRAPPER_H
#define CUDA_DRIVER_WRAPPER_H

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <memory>
#include <utility>

#define cuErrCheck(stat, wrap)                                                                                         \
    {                                                                                                                  \
        nvinfer1::cuErrCheck_((stat), wrap, __FILE__, __LINE__);                                                       \
    }

namespace nvinfer1
{
class CUDADriverWrapper
{
public:
    CUDADriverWrapper();

    ~CUDADriverWrapper();

    // Delete default copy constructor and copy assignment constructor
    CUDADriverWrapper(CUDADriverWrapper const&) = delete;
    CUDADriverWrapper& operator=(CUDADriverWrapper const&) = delete;

    CUresult cuGetErrorName(CUresult error, char const** pStr) const;

    CUresult cuGetErrorString(CUresult error, char const** pStr) const;

    CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int32_t value) const;

    CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const;

    CUresult cuModuleUnload(CUmodule hmod) const;

    CUresult cuLinkDestroy(CUlinkState state) const;

    CUresult cuModuleLoadData(CUmodule* module, void const* image) const;

    CUresult cuLinkCreate(uint32_t numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const;

    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, char const* name) const;

    CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, char const* path, uint32_t numOptions,
        CUjit_option* options, void** optionValues) const;

    CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, char const* name,
        uint32_t numOptions, CUjit_option* options, void** optionValues) const;

    CUresult cuLaunchCooperativeKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
        uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
        void** kernelParams) const;

    CUresult cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ, uint32_t blockDimX,
        uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream, void** kernelParams,
        void** extra) const;

#if CUDA_VERSION >= 11060
    CUresult cuLaunchKernelEx(CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra) const;
#endif

#if CUDA_VERSION >= 12000
    CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
        void const* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides,
        cuuint32_t const* boxDim, cuuint32_t const* elementStrides, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) const;
#endif

    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) const;

    CUresult cuDeviceGetAttribute(int32_t* pi, CUdevice_attribute attrib, CUdevice dev) const;

#if CUDA_VERSION >= 12000
    CUresult cuOccupancyMaxActiveClusters(int32_t* maxActiveClusters, CUfunction f, CUlaunchConfig const* config) const;
#endif

private:
    void* handle;
    CUresult (*_cuGetErrorName)(CUresult, char const**);
    CUresult (*_cuGetErrorString)(CUresult, char const**);
    CUresult (*_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int32_t);
    CUresult (*_cuLinkComplete)(CUlinkState, void**, size_t*);
    CUresult (*_cuModuleUnload)(CUmodule);
    CUresult (*_cuLinkDestroy)(CUlinkState);
    CUresult (*_cuLinkCreate)(uint32_t, CUjit_option*, void**, CUlinkState*);
    CUresult (*_cuModuleLoadData)(CUmodule*, void const*);
    CUresult (*_cuModuleGetFunction)(CUfunction*, CUmodule, char const*);
    CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, char const*, uint32_t, CUjit_option*, void**);
    CUresult (*_cuLinkAddData)(
        CUlinkState, CUjitInputType, void*, size_t, char const*, uint32_t, CUjit_option*, void**);
    CUresult (*_cuLaunchCooperativeKernel)(
        CUfunction, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, CUstream, void**);
    CUresult (*_cuLaunchKernel)(CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
        uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
        void** kernelParams, void** extra);
#if CUDA_VERSION >= 11060
    CUresult (*_cuLaunchKernelEx)(CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra);
#endif
#if CUDA_VERSION >= 12000
    CUresult (*_cuTensorMapEncodeTiled)(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
        cuuint32_t tensorRank, void const* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides,
        cuuint32_t const* boxDim, cuuint32_t const* elementStrides, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
#endif
    CUresult (*_cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult (*_cuDeviceGetAttribute)(int32_t*, CUdevice_attribute attrib, CUdevice dev);
#if CUDA_VERSION >= 12000
    CUresult (*_cuOccupancyMaxActiveClusters)(int32_t*, CUfunction f, CUlaunchConfig const* config);
#endif
};

inline void cuErrCheck_(CUresult stat, CUDADriverWrapper const& wrap, char const* file, int32_t line)
{
    if (stat != CUDA_SUCCESS)
    {
        char const* msg = nullptr;
        wrap.cuGetErrorName(stat, &msg);
        fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
    }
}

//! Return CUDA major version
constexpr int32_t getCudaLibVersionMaj() noexcept
{
    return CUDA_VERSION / 1000U;
}

// RAII wrapper for CUDA module
// Automatically manages CUDA module lifecycle - loading in constructor, unloading in destructor
// Provides safe module management and prevents memory leaks
class CudaModule
{
public:
    // Default constructor - creates an uninitialized module
    CudaModule() noexcept
        : mModule(nullptr)
        , mDriverWrapper(nullptr)
        , mLoadResult(CUDA_ERROR_NOT_INITIALIZED)
    {
    }

    // Constructor that loads a CUDA module from data
    CudaModule(CUDADriverWrapper const& driverWrapper, void const* image)
        : mModule(nullptr)
        , mDriverWrapper(&driverWrapper)
        , mLoadResult(CUDA_ERROR_NOT_INITIALIZED)
    {
        mLoadResult = mDriverWrapper->cuModuleLoadData(&mModule, image);
    }

    // Destructor - automatically unloads the module
    ~CudaModule()
    {
        if (mModule != nullptr && mDriverWrapper != nullptr)
        {
            mDriverWrapper->cuModuleUnload(mModule);
        }
    }

    // Delete copy constructor and assignment
    CudaModule(CudaModule const&) = delete;
    CudaModule& operator=(CudaModule const&) = delete;

    // Move constructor
    CudaModule(CudaModule&& other) noexcept
        : mModule(std::exchange(other.mModule, nullptr))
        , mDriverWrapper(std::exchange(other.mDriverWrapper, nullptr))
        , mLoadResult(std::exchange(other.mLoadResult, CUDA_ERROR_NOT_INITIALIZED))
    {
    }

    // Move assignment
    CudaModule& operator=(CudaModule&& other) noexcept
    {
        CudaModule tmp{std::move(other)};
        std::swap(mModule, tmp.mModule);
        std::swap(mDriverWrapper, tmp.mDriverWrapper);
        std::swap(mLoadResult, tmp.mLoadResult);
        return *this;
    }

    //! Get the underlying CUDA module handle (`CUmodule`; a raw pointer).
    [[nodiscard]] CUmodule get() noexcept
    {
        return mModule;
    }

    //! Get the underlying const CUDA module handle (`CUmodule`; a raw pointer).
    //! \note Since `CUmodule` is a raw pointer, we remove the pointer from the
    //! type, add `const`, and re-add the pointer.
    [[nodiscard]] std::remove_pointer_t<CUmodule> const* get() const noexcept
    {
        return mModule;
    }

    // Implicit conversion to CUmodule
    operator CUmodule() const noexcept
    {
        return mModule;
    }

    // Check if the module is valid
    bool isValid() const noexcept
    {
        return mModule != nullptr && mLoadResult == CUDA_SUCCESS;
    }

    // Get function from module
    CUresult getFunction(CUfunction* hfunc, char const* name) const
    {
        if (mModule != nullptr && mDriverWrapper != nullptr)
        {
            return mDriverWrapper->cuModuleGetFunction(hfunc, mModule, name);
        }
        return CUDA_ERROR_INVALID_HANDLE;
    }

private:
    CUmodule mModule;
    CUDADriverWrapper const* mDriverWrapper;
    CUresult mLoadResult;
};

// Helper function to create a unique_ptr to CudaModule
inline std::unique_ptr<CudaModule> makeCudaModule(CUDADriverWrapper const& driverWrapper, void const* image)
{
    return std::make_unique<CudaModule>(driverWrapper, image);
}

} // namespace nvinfer1

#endif // CUDA_DRIVER_WRAPPER_H
