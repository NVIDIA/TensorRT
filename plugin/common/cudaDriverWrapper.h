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

#ifndef CUDA_DRIVER_WRAPPER_H
#define CUDA_DRIVER_WRAPPER_H

#include <cstdint>
#include <cstdio>
#include <cuda.h>

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

private:
    void* handle;
    CUresult (*_cuGetErrorName)(CUresult, char const**);
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

} // namespace nvinfer1

#endif // CUDA_DRIVER_WRAPPER_H
