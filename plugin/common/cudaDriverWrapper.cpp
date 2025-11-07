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
#define CUDA_LIB_NAME "cuda"

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA("nv" name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
#else // defined(_WIN32)
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so.1", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "common/cudaDriverWrapper.h"
#include "common/plugin.h"
#include <cstdint>
#include <cstdio>
#include <cuda.h>

using namespace nvinfer1;

CUDADriverWrapper::CUDADriverWrapper()
{
    handle = dllOpen(CUDA_LIB_NAME);
    PLUGIN_ASSERT(handle != nullptr);

    auto load_sym = [](void* handle, char const* name) {
        void* ret = dllGetSym(handle, name);
        PLUGIN_ASSERT(ret != nullptr);
        return ret;
    };

    _cuGetErrorName = reinterpret_cast<CUresult (*)(CUresult, char const**)>(load_sym(handle, "cuGetErrorName"));
    _cuGetErrorString = reinterpret_cast<CUresult (*)(CUresult, char const**)>(load_sym(handle, "cuGetErrorString"));
    _cuFuncSetAttribute = reinterpret_cast<CUresult (*)(CUfunction, CUfunction_attribute, int32_t)>(
        load_sym(handle, "cuFuncSetAttribute"));
    _cuLinkComplete = reinterpret_cast<CUresult (*)(CUlinkState, void**, size_t*)>(load_sym(handle, "cuLinkComplete"));
    _cuModuleUnload = reinterpret_cast<CUresult (*)(CUmodule)>(load_sym(handle, "cuModuleUnload"));
    _cuLinkDestroy = reinterpret_cast<CUresult (*)(CUlinkState)>(load_sym(handle, "cuLinkDestroy"));
    _cuModuleLoadData = reinterpret_cast<CUresult (*)(CUmodule*, void const*)>(load_sym(handle, "cuModuleLoadData"));
    _cuLinkCreate = reinterpret_cast<CUresult (*)(uint32_t, CUjit_option*, void**, CUlinkState*)>(
        load_sym(handle, "cuLinkCreate_v2"));
    _cuModuleGetFunction
        = reinterpret_cast<CUresult (*)(CUfunction*, CUmodule, char const*)>(load_sym(handle, "cuModuleGetFunction"));
    _cuLinkAddFile
        = reinterpret_cast<CUresult (*)(CUlinkState, CUjitInputType, char const*, uint32_t, CUjit_option*, void**)>(
            load_sym(handle, "cuLinkAddFile_v2"));
    _cuLinkAddData = reinterpret_cast<CUresult (*)(CUlinkState, CUjitInputType, void*, size_t, char const*, uint32_t,
        CUjit_option*, void**)>(load_sym(handle, "cuLinkAddData_v2"));
    _cuLaunchCooperativeKernel = reinterpret_cast<CUresult (*)(CUfunction, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, CUstream, void**)>(load_sym(handle, "cuLaunchCooperativeKernel"));
    _cuLaunchKernel = reinterpret_cast<CUresult (*)(CUfunction, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, CUstream, void**, void**)>(load_sym(handle, "cuLaunchKernel"));
#if CUDA_VERSION >= 11060
    _cuLaunchKernelEx
        = reinterpret_cast<CUresult (*)(CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra)>(
            dllGetSym(handle, "cuLaunchKernelEx"));
#endif
#if CUDA_VERSION >= 12000
    _cuTensorMapEncodeTiled
        = reinterpret_cast<CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void const*, cuuint64_t const*,
            cuuint64_t const*, cuuint32_t const*, cuuint32_t const*, CUtensorMapInterleave, CUtensorMapSwizzle,
            CUtensorMapL2promotion, CUtensorMapFloatOOBfill)>(load_sym(handle, "cuTensorMapEncodeTiled"));
#endif
    _cuMemcpyDtoH = reinterpret_cast<CUresult (*)(void*, CUdeviceptr, size_t)>(load_sym(handle, "cuMemcpyDtoH_v2"));
    _cuDeviceGetAttribute = reinterpret_cast<CUresult (*)(int32_t*, CUdevice_attribute, CUdevice)>(
        load_sym(handle, "cuDeviceGetAttribute"));
#if CUDA_VERSION >= 12000
    _cuOccupancyMaxActiveClusters = reinterpret_cast<CUresult (*)(int32_t*, CUfunction, CUlaunchConfig const*)>(
        load_sym(handle, "cuOccupancyMaxActiveClusters"));
#endif
}

CUDADriverWrapper::~CUDADriverWrapper()
{
    dllClose(handle);
}

CUresult CUDADriverWrapper::cuGetErrorName(CUresult error, char const** pStr) const
{
    return (*_cuGetErrorName)(error, pStr);
}

CUresult CUDADriverWrapper::cuGetErrorString(CUresult error, char const** pStr) const
{
    return (*_cuGetErrorString)(error, pStr);
}

CUresult CUDADriverWrapper::cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int32_t value) const
{
    return (*_cuFuncSetAttribute)(hfunc, attrib, value);
}

CUresult CUDADriverWrapper::cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const
{
    return (*_cuLinkComplete)(state, cubinOut, sizeOut);
}

CUresult CUDADriverWrapper::cuModuleUnload(CUmodule hmod) const
{
    return (*_cuModuleUnload)(hmod);
}

CUresult CUDADriverWrapper::cuLinkDestroy(CUlinkState state) const
{
    return (*_cuLinkDestroy)(state);
}

CUresult CUDADriverWrapper::cuModuleLoadData(CUmodule* module, void const* image) const
{
    return (*_cuModuleLoadData)(module, image);
}

CUresult CUDADriverWrapper::cuLinkCreate(
    uint32_t numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const
{
    return (*_cuLinkCreate)(numOptions, options, optionValues, stateOut);
}

CUresult CUDADriverWrapper::cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, char const* name) const
{
    return (*_cuModuleGetFunction)(hfunc, hmod, name);
}

CUresult CUDADriverWrapper::cuLinkAddFile(CUlinkState state, CUjitInputType type, char const* path, uint32_t numOptions,
    CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddFile)(state, type, path, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size,
    char const* name, uint32_t numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddData)(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLaunchCooperativeKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
    uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes,
    CUstream hStream, void** kernelParams) const
{
    return (*_cuLaunchCooperativeKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult CUDADriverWrapper::cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
    void** kernelParams, void** extra) const
{
    return (*_cuLaunchKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

#if CUDA_VERSION >= 11060
CUresult CUDADriverWrapper::cuLaunchKernelEx(
    CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra) const
{
    PLUGIN_ASSERT(_cuLaunchKernelEx != nullptr);
    return (*_cuLaunchKernelEx)(config, f, kernelParams, extra);
}
#endif

#if CUDA_VERSION >= 12000
CUresult CUDADriverWrapper::cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void const* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides,
    cuuint32_t const* boxDim, cuuint32_t const* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) const
{
    PLUGIN_ASSERT(_cuTensorMapEncodeTiled != nullptr);
    return (*_cuTensorMapEncodeTiled)(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);
}
#endif

CUresult CUDADriverWrapper::cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) const
{
    return (*_cuMemcpyDtoH)(dstHost, srcDevice, ByteCount);
}

CUresult CUDADriverWrapper::cuDeviceGetAttribute(int32_t* pi, CUdevice_attribute attrib, CUdevice dev) const
{
    return (*_cuDeviceGetAttribute)(pi, attrib, dev);
}

#if CUDA_VERSION >= 12000
CUresult CUDADriverWrapper::cuOccupancyMaxActiveClusters(
    int32_t* maxActiveClusters, CUfunction f, CUlaunchConfig const* config) const
{
    PLUGIN_ASSERT(_cuOccupancyMaxActiveClusters != nullptr);
    return (*_cuOccupancyMaxActiveClusters)(maxActiveClusters, f, config);
}
#endif
