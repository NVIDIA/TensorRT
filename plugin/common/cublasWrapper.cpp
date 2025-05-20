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

#include "cublasWrapper.h"
#include "common/checkMacrosPlugin.h"
#include "cudaDriverWrapper.h"

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
// Ensure that macros appearing in multiple files are only defined once.
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA(name)
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
auto const kCUBLAS_PLUGIN_LIBNAME
    = std::string{"cublas64_"} + std::to_string(nvinfer1::getCudaLibVersionMaj()) + ".dll";
#else // defined(_WIN32)
#include <dlfcn.h>
#define dllOpen(name) dlopen(name, RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
auto const kCUBLAS_PLUGIN_LIBNAME = std::string{"libcublas.so."} + std::to_string(nvinfer1::getCudaLibVersionMaj());
#endif // defined(_WIN32)

namespace nvinfer1::pluginInternal
{
using namespace nvinfer1;

// If tryLoadingCublas failed, the CublasWrapper object won't be created.
CublasWrapper::CublasWrapper(bool initHandle)
    : mLibrary(tryLoadingCublas())
{
    PLUGIN_VALIDATE(mLibrary != nullptr);
    auto load_sym = [](void* handle, char const* name) {
        void* ret = dllGetSym(handle, name);
        std::string loadError = "Fail to load symbol " + std::string(name) + " from the cublas library.";
        PLUGIN_VALIDATE(ret != nullptr, loadError.c_str());
        return ret;
    };
    *(void**) (&_cublasCreate) = load_sym(mLibrary, "cublasCreate_v2");
    *(void**) (&_cublasDestroy) = load_sym(mLibrary, "cublasDestroy_v2");
    *(void**) (&_cublasSetStream) = load_sym(mLibrary, "cublasSetStream_v2");
    *(void**) (&_cublasGetPointerMode) = load_sym(mLibrary, "cublasGetPointerMode_v2");
    *(void**) (&_cublasSetPointerMode) = load_sym(mLibrary, "cublasSetPointerMode_v2");
    *(void**) (&_cublasGetMathMode) = load_sym(mLibrary, "cublasGetMathMode");
    *(void**) (&_cublasSetMathMode) = load_sym(mLibrary, "cublasSetMathMode");
    *(void**) (&_cublasDscal) = load_sym(mLibrary, "cublasDscal_v2");
    *(void**) (&_cublasSasum) = load_sym(mLibrary, "cublasSasum_v2");
    *(void**) (&_cublasScopy) = load_sym(mLibrary, "cublasScopy_v2");
    *(void**) (&_cublasSscal) = load_sym(mLibrary, "cublasSscal_v2");
    *(void**) (&_cublasSgemm) = load_sym(mLibrary, "cublasSgemm_v2");
    *(void**) (&_cublasHgemm) = load_sym(mLibrary, "cublasHgemm");
    *(void**) (&_cublasHgemmStridedBatched) = load_sym(mLibrary, "cublasHgemmStridedBatched");
    *(void**) (&_cublasSgemmStridedBatched) = load_sym(mLibrary, "cublasSgemmStridedBatched");
    *(void**) (&_cublasGemmEx) = load_sym(mLibrary, "cublasGemmEx");
    *(void**) (&_cublasGemmStridedBatchedEx) = load_sym(mLibrary, "cublasGemmStridedBatchedEx");

    if (initHandle)
    {
        PLUGIN_VALIDATE(cublasCreate(&mHandle) == CUBLAS_STATUS_SUCCESS, "Could not create cublas handle.");
        PLUGIN_VALIDATE(mHandle != nullptr);
    }
}

CublasWrapper::~CublasWrapper()
{
    if (mHandle != nullptr)
    {
        PLUGIN_VALIDATE(cublasDestroy(mHandle) == CUBLAS_STATUS_SUCCESS, "Could not destroy cublas handle.");
        mHandle = nullptr;
    }

    if (mLibrary != nullptr)
    {
        dllClose(mLibrary);
    }
}

void* CublasWrapper::tryLoadingCublas()
{
    void* cublasLib = dllOpen(kCUBLAS_PLUGIN_LIBNAME.c_str());
    std::string errorMsg = "Failed to load " + kCUBLAS_PLUGIN_LIBNAME + ".";
    PLUGIN_VALIDATE(cublasLib != nullptr, errorMsg.c_str());
    return cublasLib;
}

cublasContext* CublasWrapper::getCublasHandle()
{
    return mHandle;
}

bool CublasWrapper::isValid() const
{
    return mHandle != nullptr;
}

cublasStatus_t CublasWrapper::cublasCreate(cublasContext** handle)
{
    return (*_cublasCreate)(handle);
}

cublasStatus_t CublasWrapper::cublasDestroy(cublasContext* handle)
{
    return (*_cublasDestroy)(handle);
}

cublasStatus_t CublasWrapper::cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    return (*_cublasSetStream)(handle, streamId);
}

cublasStatus_t CublasWrapper::cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode)
{
    return (*_cublasGetPointerMode)(handle, mode);
}

cublasStatus_t CublasWrapper::cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode)
{
    return (*_cublasSetPointerMode)(handle, mode);
}

cublasStatus_t CublasWrapper::cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode)
{
    return (*_cublasGetMathMode)(handle, mode);
}

cublasStatus_t CublasWrapper::cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    return (*_cublasSetMathMode)(handle, mode);
}

cublasStatus_t CublasWrapper::cublasDscal(cublasHandle_t handle, int n, float const* alpha, float* x, int incx)
{
    return (*_cublasDscal)(handle, n, alpha, x, incx);
}

cublasStatus_t CublasWrapper::cublasSasum(cublasHandle_t handle, int n, float const* x, int incx, float* result)
{
    return (*_cublasSasum)(handle, n, x, incx, result);
}

cublasStatus_t CublasWrapper::cublasScopy(cublasHandle_t handle, int n, float const* x, int incx, float* y, int incy)
{
    return (*_cublasScopy)(handle, n, x, incx, y, incy);
}

cublasStatus_t CublasWrapper::cublasSscal(cublasHandle_t handle, int n, float const* alpha, float* x, int incx)
{
    return (*_cublasSscal)(handle, n, alpha, x, incx);
}

cublasStatus_t CublasWrapper::cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float const* alpha, float const* A, int lda, float const* B, int ldb, float const* beta,
    float* C, int ldc)
{
    return (*_cublasSgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CublasWrapper::cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, __half const* alpha, __half const* A, int lda, __half const* B, int ldb, __half const* beta,
    __half* C, int ldc)
{
    return (*_cublasHgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CublasWrapper::cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, __half const* alpha, __half const* A, int lda, long long int strideA,
    __half const* B, int ldb, long long int strideB, __half const* beta, __half* C, int ldc, long long int strideC,
    int batchCount)
{
    return (*_cublasHgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t CublasWrapper::cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, float const* alpha, float const* A, int lda, long long int strideA,
    float const* B, int ldb, long long int strideB, float const* beta, float* C, int ldc, long long int strideC,
    int batchCount)
{
    return (*_cublasSgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t CublasWrapper::cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, void const* alpha, void const* A, cudaDataType Atype, int lda, void const* B,
    cudaDataType Btype, int ldb, void const* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType,
    cublasGemmAlgo_t algo)
{
    return (*_cublasGemmEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}

cublasStatus_t CublasWrapper::cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, void const* alpha, void const* A, cudaDataType Atype, int lda,
    long long int strideA, void const* B, cudaDataType Btype, int ldb, long long int strideB, void const* beta, void* C,
    cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo)
{
    return (*_cublasGemmStridedBatchedEx)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb,
        strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
}

CublasWrapper& getCublasWrapper()
{
    // Initialize a global cublasWrapper instance to be used to call cublas functions.
    static CublasWrapper sGCublasWrapper;
    return sGCublasWrapper;
}

} // namespace nvinfer1::pluginInternal
