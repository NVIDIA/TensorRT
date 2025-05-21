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

#include "cublasLtWrapper.h"
#include "common/checkMacrosPlugin.h"
#include "cudaDriverWrapper.h"

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA(name)
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
auto const kCUBLASLT_PLUGIN_LIBNAME
    = std::string{"cublasLt64_"} + std::to_string(nvinfer1::getCudaLibVersionMaj()) + ".dll";
#else // defined(_WIN32)
#include <dlfcn.h>
#define dllOpen(name) dlopen(name, RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
auto const kCUBLASLT_PLUGIN_LIBNAME = std::string{"libcublasLt.so."} + std::to_string(nvinfer1::getCudaLibVersionMaj());
#endif // defined(_WIN32)

namespace nvinfer1::pluginInternal
{
using namespace nvinfer1;

// If tryLoadingCublasLt failed, the CublasLtWrapper object won't be created.
CublasLtWrapper::CublasLtWrapper(bool initHandle)
    : mLibrary(tryLoadingCublasLt())
{
    PLUGIN_VALIDATE(mLibrary != nullptr);
    auto load_sym = [](void* handle, char const* name) {
        void* ret = dllGetSym(handle, name);
        std::string loadError = "Fail to load symbol " + std::string(name) + " from the cublasLt library.";
        PLUGIN_VALIDATE(ret != nullptr, loadError.c_str());
        return ret;
    };
    *(void**) (&_cublasLtCreate) = load_sym(mLibrary, "cublasLtCreate");
    *(void**) (&_cublasLtDestroy) = load_sym(mLibrary, "cublasLtDestroy");
    *(void**) (&_cublasLtMatmul) = load_sym(mLibrary, "cublasLtMatmul");
    *(void**) (&_cublasLtMatmulDescCreate) = load_sym(mLibrary, "cublasLtMatmulDescCreate");
    *(void**) (&_cublasLtMatmulDescDestroy) = load_sym(mLibrary, "cublasLtMatmulDescDestroy");
    *(void**) (&_cublasLtMatmulPreferenceCreate) = load_sym(mLibrary, "cublasLtMatmulPreferenceCreate");
    *(void**) (&_cublasLtMatmulPreferenceDestroy) = load_sym(mLibrary, "cublasLtMatmulPreferenceDestroy");
    *(void**) (&_cublasLtMatmulPreferenceSetAttribute) = load_sym(mLibrary, "cublasLtMatmulPreferenceSetAttribute");
    *(void**) (&_cublasLtMatmulAlgoInit) = load_sym(mLibrary, "cublasLtMatmulAlgoInit");
    *(void**) (&_cublasLtMatmulAlgoCheck) = load_sym(mLibrary, "cublasLtMatmulAlgoCheck");
    *(void**) (&_cublasLtMatmulAlgoGetIds) = load_sym(mLibrary, "cublasLtMatmulAlgoGetIds");
    *(void**) (&_cublasLtMatrixLayoutCreate) = load_sym(mLibrary, "cublasLtMatrixLayoutCreate");
    *(void**) (&_cublasLtMatrixLayoutDestroy) = load_sym(mLibrary, "cublasLtMatrixLayoutDestroy");
    *(void**) (&_cublasLtMatrixLayoutSetAttribute) = load_sym(mLibrary, "cublasLtMatrixLayoutSetAttribute");
    *(void**) (&_cublasLtMatmulAlgoConfigSetAttribute) = load_sym(mLibrary, "cublasLtMatmulAlgoConfigSetAttribute");
    *(void**) (&_cublasLtMatmulAlgoConfigGetAttribute) = load_sym(mLibrary, "cublasLtMatmulAlgoConfigGetAttribute");
    *(void**) (&_cublasLtMatmulAlgoCapGetAttribute) = load_sym(mLibrary, "cublasLtMatmulAlgoCapGetAttribute");
    *(void**) (&_cublasLtMatmulDescSetAttribute) = load_sym(mLibrary, "cublasLtMatmulDescSetAttribute");

    if (initHandle)
    {
        PLUGIN_VALIDATE(cublasLtCreate(&mHandle) == CUBLAS_STATUS_SUCCESS, "Could not create cublasLt handle.");
        PLUGIN_VALIDATE(mHandle != nullptr);
    }
}

CublasLtWrapper::~CublasLtWrapper()
{
    if (mHandle != nullptr)
    {
        PLUGIN_VALIDATE(cublasLtDestroy(mHandle) == CUBLAS_STATUS_SUCCESS, "Could not destroy cublas handle.");
        mHandle = nullptr;
    }

    if (mLibrary != nullptr)
    {
        dllClose(mLibrary);
    }
}

void* CublasLtWrapper::tryLoadingCublasLt()
{
    void* cublasLtLib = dllOpen(kCUBLASLT_PLUGIN_LIBNAME.c_str());
    std::string errorMsg = "Failed to load " + kCUBLASLT_PLUGIN_LIBNAME + ".";
    PLUGIN_VALIDATE(cublasLtLib != nullptr, errorMsg.c_str());
    return cublasLtLib;
}

cublasLtContext* CublasLtWrapper::getCublasLtHandle()
{
    return mHandle;
}

bool CublasLtWrapper::isValid() const
{
    return mHandle != nullptr;
}

cublasStatus_t CublasLtWrapper::cublasLtCreate(cublasLtHandle_t* handle)
{
    return (*_cublasLtCreate)(handle);
}

cublasStatus_t CublasLtWrapper::cublasLtDestroy(cublasLtHandle_t handle)
{
    return (*_cublasLtDestroy)(handle);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    void const* alpha, void const* A, cublasLtMatrixLayout_t Adesc, void const* B, cublasLtMatrixLayout_t Bdesc,
    void const* beta, void const* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulAlgo_t const* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream)
{
    return (*_cublasLtMatmul)(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo,
        workspace, workspaceSizeInBytes, stream);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType)
{
    return (*_cublasLtMatmulDescCreate)(matmulDesc, computeType, scaleType);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
    return (*_cublasLtMatmulDescDestroy)(matmulDesc);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref)
{
    return (*_cublasLtMatmulPreferenceCreate)(pref);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
    return (*_cublasLtMatmulPreferenceDestroy)(pref);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, void const* buf, size_t sizeInBytes)
{
    return (*_cublasLtMatmulPreferenceSetAttribute)(pref, attr, buf, sizeInBytes);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype,
    int algoId, cublasLtMatmulAlgo_t* algo)
{
    return (*_cublasLtMatmulAlgoInit)(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulAlgoCheck(cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulAlgo_t const* algo,
    cublasLtMatmulHeuristicResult_t* result)
{
    return (*_cublasLtMatmulAlgoCheck)(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo, result);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype,
    int requestedAlgoCount, int algoIdsArray[], int* returnAlgoCount)
{
    return (*_cublasLtMatmulAlgoGetIds)(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype,
        requestedAlgoCount, algoIdsArray, returnAlgoCount);
}

cublasStatus_t CublasLtWrapper::cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)
{
    return (*_cublasLtMatrixLayoutCreate)(matLayout, type, rows, cols, ld);
}

cublasStatus_t CublasLtWrapper::cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
    return (*_cublasLtMatrixLayoutDestroy)(matLayout);
}

cublasStatus_t CublasLtWrapper::cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void const* buf, size_t sizeInBytes)
{
    return (*_cublasLtMatrixLayoutSetAttribute)(matLayout, attr, buf, sizeInBytes);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulAlgoConfigGetAttribute(cublasLtMatmulAlgo_t const* algo,
    cublasLtMatmulAlgoConfigAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten)
{
    return (*_cublasLtMatmulAlgoConfigGetAttribute)(algo, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulAlgoConfigSetAttribute(
    cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, void const* buf, size_t sizeInBytes)
{
    return (*_cublasLtMatmulAlgoConfigSetAttribute)(algo, attr, buf, sizeInBytes);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulAlgoCapGetAttribute(cublasLtMatmulAlgo_t const* algo,
    cublasLtMatmulAlgoCapAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten)
{
    return (*_cublasLtMatmulAlgoCapGetAttribute)(algo, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CublasLtWrapper::cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void const* buf, size_t sizeInBytes)
{
    return (*_cublasLtMatmulDescSetAttribute)(matmulDesc, attr, buf, sizeInBytes);
}

CublasLtWrapper& getCublasLtWrapper()
{
    // Initialize a global cublasLtWrapper instance to be used to call cublasLt functions.
    static CublasLtWrapper sGCublasLtWrapper;
    return sGCublasLtWrapper;
}

} // namespace nvinfer1::pluginInternal
