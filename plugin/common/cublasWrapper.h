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

#ifndef TRT_PLUGIN_CUBLAS_WRAPPER_H
#define TRT_PLUGIN_CUBLAS_WRAPPER_H

#include "NvInferPlugin.h"
#include <cuda_fp16.h>
#include <library_types.h>
#include <string>

namespace nvinfer1
{
namespace pluginInternal
{
/* Copy of CUBLAS status type returns */
enum CublasStatus
{
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
};

/* Copy of CUBLAS math types*/
enum CublasMath
{
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
};

/* Copy of CUBLAS operation types*/
enum cublasOperation
{
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_HERMITAN = 2,
    CUBLAS_OP_CONJG = 3
};

/* Copy of CUBLAS pointer mode types*/
enum cublasPointerMode
{
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
};

/* Copy of CUBLAS compute types*/
enum cublasComputeType
{
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,
};

/* Copy of CUBLAS GEMM algorithm types*/
enum cublasGemmAlgo
{
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,
    CUBLAS_GEMM_ALGO8 = 8,
    CUBLAS_GEMM_ALGO9 = 9,
    CUBLAS_GEMM_ALGO10 = 10,
    CUBLAS_GEMM_ALGO11 = 11,
    CUBLAS_GEMM_ALGO12 = 12,
    CUBLAS_GEMM_ALGO13 = 13,
    CUBLAS_GEMM_ALGO14 = 14,
    CUBLAS_GEMM_ALGO15 = 15,
    CUBLAS_GEMM_ALGO16 = 16,
    CUBLAS_GEMM_ALGO17 = 17,
    CUBLAS_GEMM_ALGO18 = 18,
    CUBLAS_GEMM_ALGO19 = 19,
    CUBLAS_GEMM_ALGO20 = 20,
    CUBLAS_GEMM_ALGO21 = 21,
    CUBLAS_GEMM_ALGO22 = 22,
    CUBLAS_GEMM_ALGO23 = 23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115
};

using cublasStatus_t = CublasStatus;
using cublasMath_t = CublasMath;
using cublasOperation_t = cublasOperation;
using cublasPointerMode_t = cublasPointerMode;
using cublasComputeType_t = cublasComputeType;
using cublasGemmAlgo_t = cublasGemmAlgo;
using cublasDataType_t = cudaDataType;
using cublasHandle_t = struct cublasContext*;

class CublasWrapper
{
public:
    CublasWrapper(bool initHandle = false);
    ~CublasWrapper();

    cublasContext* getCublasHandle();
    bool isValid() const;

    cublasStatus_t cublasCreate(cublasContext** handle);
    cublasStatus_t cublasDestroy(cublasContext* handle);
    cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
    cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode);
    cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);
    cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode);
    cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
    cublasStatus_t cublasDscal(cublasHandle_t handle, int n, float const* alpha, float* x, int incx);
    cublasStatus_t cublasSasum(cublasHandle_t handle, int n, float const* x, int incx, float* result);
    cublasStatus_t cublasScopy(cublasHandle_t handle, int n, float const* x, int incx, float* y, int incy);
    cublasStatus_t cublasSscal(cublasHandle_t handle, int n, float const* alpha, float* x, int incx);
    cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
        int k, float const* alpha, float const* A, int lda, float const* B, int ldb, float const* beta, float* C,
        int ldc);
    cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
        int k, __half const* alpha, __half const* A, int lda, __half const* B, int ldb, __half const* beta, __half* C,
        int ldc);
    cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, __half const* alpha, __half const* A, int lda, long long int strideA, __half const* B,
        int ldb, long long int strideB, __half const* beta, __half* C, int ldc, long long int strideC, int batchCount);
    cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, float const* alpha, float const* A, int lda, long long int strideA, float const* B,
        int ldb, long long int strideB, float const* beta, float* C, int ldc, long long int strideC, int batchCount);
    cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
        int k, void const* alpha, void const* A, cudaDataType Atype, int lda, void const* B, cudaDataType Btype,
        int ldb, void const* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType,
        cublasGemmAlgo_t algo);
    cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, void const* alpha, void const* A, cudaDataType Atype, int lda, long long int strideA,
        void const* B, cudaDataType Btype, int ldb, long long int strideB, void const* beta, void* C,
        cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType,
        cublasGemmAlgo_t algo);

private:
    void* mLibrary{nullptr};
    cublasContext* mHandle{nullptr};
    void* tryLoadingCublas();

    cublasStatus_t (*_cublasCreate)(cublasContext**);
    cublasStatus_t (*_cublasDestroy)(cublasContext*);
    cublasStatus_t (*_cublasSetStream)(cublasHandle_t handle, cudaStream_t streamId);
    cublasStatus_t (*_cublasGetPointerMode)(cublasHandle_t handle, cublasPointerMode_t* mode);
    cublasStatus_t (*_cublasSetPointerMode)(cublasHandle_t handle, cublasPointerMode_t mode);
    cublasStatus_t (*_cublasGetMathMode)(cublasHandle_t handle, cublasMath_t* mode);
    cublasStatus_t (*_cublasSetMathMode)(cublasHandle_t handle, cublasMath_t mode);
    cublasStatus_t (*_cublasDscal)(cublasHandle_t handle, int n, float const* alpha, float* x, int incx);
    cublasStatus_t (*_cublasSasum)(cublasHandle_t handle, int n, float const* x, int incx, float* result);
    cublasStatus_t (*_cublasScopy)(cublasHandle_t handle, int n, float const* x, int incx, float* y, int incy);
    cublasStatus_t (*_cublasSscal)(cublasHandle_t handle, int n, float const* alpha, float* x, int incx);
    cublasStatus_t (*_cublasSgemm)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
        int n, int k, float const* alpha, float const* A, int lda, float const* B, int ldb, float const* beta, float* C,
        int ldc);
    cublasStatus_t (*_cublasHgemm)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
        int n, int k, __half const* alpha, __half const* A, int lda, __half const* B, int ldb, __half const* beta,
        __half* C, int ldc);
    cublasStatus_t (*_cublasHgemmStridedBatched)(cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k, __half const* alpha, __half const* A, int lda,
        long long int strideA, __half const* B, int ldb, long long int strideB, __half const* beta, __half* C, int ldc,
        long long int strideC, int batchCount);
    cublasStatus_t (*_cublasSgemmStridedBatched)(cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k, float const* alpha, float const* A, int lda,
        long long int strideA, float const* B, int ldb, long long int strideB, float const* beta, float* C, int ldc,
        long long int strideC, int batchCount);
    cublasStatus_t (*_cublasGemmEx)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
        int n, int k, void const* alpha, void const* A, cudaDataType Atype, int lda, void const* B, cudaDataType Btype,
        int ldb, void const* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType,
        cublasGemmAlgo_t algo);
    cublasStatus_t (*_cublasGemmStridedBatchedEx)(cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k, void const* alpha, void const* A, cudaDataType Atype, int lda,
        long long int strideA, void const* B, cudaDataType Btype, int ldb, long long int strideB, void const* beta,
        void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType,
        cublasGemmAlgo_t algo);
};

CublasWrapper& getCublasWrapper();

} // namespace pluginInternal
} // namespace nvinfer1

#endif // TRT_PLUGIN_CUBLAS_WRAPPER_H
