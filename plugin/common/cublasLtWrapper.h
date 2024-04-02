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

#ifndef TRT_PLUGIN_CUBLASLT_WRAPPER_H
#define TRT_PLUGIN_CUBLASLT_WRAPPER_H

#include "NvInferPlugin.h"
#include "cublasWrapper.h"
#include <cuda_fp16.h>
#include <library_types.h>
#include <string>

extern "C"
{
    struct cublasLtContext;
}

namespace nvinfer1
{
namespace pluginInternal
{

struct cublasLtMatmulAlgo
{
    uint64_t data[8];
};

using cublasLtMatmulAlgo_t = cublasLtMatmulAlgo;

struct cublasLtMatrixLayoutOpaque
{
    uint64_t data[8];
};

struct cublasLtMatmulPreferenceOpaque
{
    uint64_t data[8];
};

struct cublasLtMatmulDescOpaque
{
    uint64_t data[32];
};

struct cublasLtMatmulHeuristicResult
{
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
};

/* Copy of CUBLASLT cublasLtReductionScheme_t */
enum cublasLtReductionScheme
{
    CUBLASLT_REDUCTION_SCHEME_NONE = 0,
    CUBLASLT_REDUCTION_SCHEME_INPLACE = 1,
    CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE = 2,
    CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE = 4,
    CUBLASLT_REDUCTION_SCHEME_MASK = 0x7,
};

/* Copy of CUBLASLT cublasLtMatrixLayoutAttribute_t */
enum cublasLtMatrixLayoutAttribute
{
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
};

enum cublasLtMatmulAlgoConfigAttributes
{
    CUBLASLT_ALGO_CONFIG_ID = 0,
    CUBLASLT_ALGO_CONFIG_TILE_ID = 1,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM = 2,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME = 3,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING = 4,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION = 5,
    CUBLASLT_ALGO_CONFIG_STAGES_ID = 6,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID = 7,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID = 8,
};

enum cublasLtMatmulPreferenceAttributes
{
    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,
    CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,
    CUBLASLT_MATMUL_PREF_IMPL_MASK = 12,
};

enum cublasLtMatmulTile
{
    CUBLASLT_MATMUL_TILE_UNDEFINED = 0,
    CUBLASLT_MATMUL_TILE_8x8 = 1,
    CUBLASLT_MATMUL_TILE_8x16 = 2,
    CUBLASLT_MATMUL_TILE_16x8 = 3,
    CUBLASLT_MATMUL_TILE_8x32 = 4,
    CUBLASLT_MATMUL_TILE_16x16 = 5,
    CUBLASLT_MATMUL_TILE_32x8 = 6,
    CUBLASLT_MATMUL_TILE_8x64 = 7,
    CUBLASLT_MATMUL_TILE_16x32 = 8,
    CUBLASLT_MATMUL_TILE_32x16 = 9,
    CUBLASLT_MATMUL_TILE_64x8 = 10,
    CUBLASLT_MATMUL_TILE_32x32 = 11,
    CUBLASLT_MATMUL_TILE_32x64 = 12,
    CUBLASLT_MATMUL_TILE_64x32 = 13,
    CUBLASLT_MATMUL_TILE_32x128 = 14,
    CUBLASLT_MATMUL_TILE_64x64 = 15,
    CUBLASLT_MATMUL_TILE_128x32 = 16,
    CUBLASLT_MATMUL_TILE_64x128 = 17,
    CUBLASLT_MATMUL_TILE_128x64 = 18,
    CUBLASLT_MATMUL_TILE_64x256 = 19,
    CUBLASLT_MATMUL_TILE_128x128 = 20,
    CUBLASLT_MATMUL_TILE_256x64 = 21,
    CUBLASLT_MATMUL_TILE_64x512 = 22,
    CUBLASLT_MATMUL_TILE_128x256 = 23,
    CUBLASLT_MATMUL_TILE_256x128 = 24,
    CUBLASLT_MATMUL_TILE_512x64 = 25,
    CUBLASLT_MATMUL_TILE_64x96 = 26,
    CUBLASLT_MATMUL_TILE_96x64 = 27,
    CUBLASLT_MATMUL_TILE_96x128 = 28,
    CUBLASLT_MATMUL_TILE_128x160 = 29,
    CUBLASLT_MATMUL_TILE_160x128 = 30,
    CUBLASLT_MATMUL_TILE_192x128 = 31,
    CUBLASLT_MATMUL_TILE_128x192 = 32,
    CUBLASLT_MATMUL_TILE_128x96 = 33,
    CUBLASLT_MATMUL_TILE_32x256 = 34,
    CUBLASLT_MATMUL_TILE_256x32 = 35,
    CUBLASLT_MATMUL_TILE_END
};

enum cublasLtMatmulAlgoCapAttributes
{
    CUBLASLT_ALGO_CAP_SPLITK_SUPPORT = 0,
    CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK = 1,
    CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT = 2,
    CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT = 3,
    CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT = 4,
    CUBLASLT_ALGO_CAP_UPLO_SUPPORT = 5,
    CUBLASLT_ALGO_CAP_TILE_IDS = 6,
    CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX = 7,
    CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER = 10,
    CUBLASLT_ALGO_CAP_POINTER_MODE_MASK = 11,
    CUBLASLT_ALGO_CAP_EPILOGUE_MASK = 12,
    CUBLASLT_ALGO_CAP_STAGES_IDS = 13,
    CUBLASLT_ALGO_CAP_LD_NEGATIVE = 14,
    CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS = 15,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES = 16,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES = 17,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES = 18,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES = 19,
    CUBLASLT_ALGO_CAP_ATOMIC_SYNC = 20,
};

enum cublasLtMatmulDescAttributes
{
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_TRANSC = 5,
    CUBLASLT_MATMUL_DESC_FILL_MODE = 6,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,
    CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,
    CUBLASLT_MATMUL_DESC_FAST_ACCUM = 25,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = 27,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = 28,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = 29,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = 30,
};

using cublasLtMatrixLayoutOpaque_t = cublasLtMatrixLayoutOpaque;
using cublasLtMatrixLayout_t = cublasLtMatrixLayoutOpaque_t*;
using cublasLtMatmulPreferenceOpaque_t = cublasLtMatmulPreferenceOpaque;
using cublasLtMatmulPreference_t = cublasLtMatmulPreferenceOpaque_t*;
using cublasLtMatmulDescOpaque_t = cublasLtMatmulDescOpaque;
using cublasLtMatmulDesc_t = cublasLtMatmulDescOpaque_t*;
using cublasLtMatmulHeuristicResult_t = cublasLtMatmulHeuristicResult;
using cublasLtReductionScheme_t = cublasLtReductionScheme;
using cublasLtHandle_t = struct cublasLtContext*;
using cublasLtMatrixLayoutAttribute_t = cublasLtMatrixLayoutAttribute;
using cublasLtMatmulAlgoConfigAttributes_t = cublasLtMatmulAlgoConfigAttributes;
using cublasLtMatmulPreferenceAttributes_t = cublasLtMatmulPreferenceAttributes;
using cublasLtMatmulTile_t = cublasLtMatmulTile;
using cublasLtMatmulAlgoCapAttributes_t = cublasLtMatmulAlgoCapAttributes;
using cublasLtMatmulDescAttributes_t = cublasLtMatmulDescAttributes;

/* Copy of CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA */
constexpr auto CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA = 0x01ull << 0;
/* Copy of CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA */
constexpr auto CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA = 0x02ull << 0;

class CublasLtWrapper
{
public:
    CublasLtWrapper(bool initHandle = false);
    ~CublasLtWrapper();

    cublasLtContext* getCublasLtHandle();
    bool isValid() const;

    cublasStatus_t cublasLtCreate(cublasLtHandle_t* handle);
    cublasStatus_t cublasLtDestroy(cublasLtHandle_t handle);
    cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, void const* alpha,
        void const* A, cublasLtMatrixLayout_t Adesc, void const* B, cublasLtMatrixLayout_t Bdesc, void const* beta,
        void const* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
        cublasLtMatmulAlgo_t const* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream);
    cublasStatus_t cublasLtMatmulDescCreate(
        cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType);
    cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
    cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref);
    cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
    cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
        cublasLtMatmulPreferenceAttributes_t attr, void const* buf, size_t sizeInBytes);
    cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
        cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
        cudaDataType_t Dtype, int algoId, cublasLtMatmulAlgo_t* algo);
    cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
        cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc,
        cublasLtMatrixLayout_t Ddesc, cublasLtMatmulAlgo_t const* algo, cublasLtMatmulHeuristicResult_t* result);
    cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
        cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
        cudaDataType_t Dtype, int requestedAlgoCount, int algoIdsArray[], int* returnAlgoCount);
    cublasStatus_t cublasLtMatrixLayoutCreate(
        cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld);
    cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
    cublasStatus_t cublasLtMatrixLayoutSetAttribute(
        cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void const* buf, size_t sizeInBytes);
    cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(cublasLtMatmulAlgo_t const* algo,
        cublasLtMatmulAlgoConfigAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
    cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(
        cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, void const* buf, size_t sizeInBytes);
    cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(cublasLtMatmulAlgo_t const* algo,
        cublasLtMatmulAlgoCapAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
    cublasStatus_t cublasLtMatmulDescSetAttribute(
        cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void const* buf, size_t sizeInBytes);

private:
    void* mLibrary{nullptr};
    cublasLtContext* mHandle{nullptr};
    void* tryLoadingCublasLt();

    cublasStatus_t (*_cublasLtCreate)(cublasLtHandle_t*);
    cublasStatus_t (*_cublasLtDestroy)(cublasLtHandle_t);
    cublasStatus_t (*_cublasLtMatmul)(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, void const* alpha,
        void const* A, cublasLtMatrixLayout_t Adesc, void const* B, cublasLtMatrixLayout_t Bdesc, void const* beta,
        void const* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
        cublasLtMatmulAlgo_t const* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream);
    cublasStatus_t (*_cublasLtMatmulDescCreate)(
        cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType);
    cublasStatus_t (*_cublasLtMatmulDescDestroy)(cublasLtMatmulDesc_t matmulDesc);
    cublasStatus_t (*_cublasLtMatmulPreferenceCreate)(cublasLtMatmulPreference_t* pref);
    cublasStatus_t (*_cublasLtMatmulPreferenceDestroy)(cublasLtMatmulPreference_t pref);
    cublasStatus_t (*_cublasLtMatmulPreferenceSetAttribute)(cublasLtMatmulPreference_t pref,
        cublasLtMatmulPreferenceAttributes_t attr, void const* buf, size_t sizeInBytes);
    cublasStatus_t (*_cublasLtMatmulAlgoInit)(cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
        cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
        cudaDataType_t Dtype, int algoId, cublasLtMatmulAlgo_t* algo);
    cublasStatus_t (*_cublasLtMatmulAlgoCheck)(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
        cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc,
        cublasLtMatrixLayout_t Ddesc, cublasLtMatmulAlgo_t const* algo, cublasLtMatmulHeuristicResult_t* result);
    cublasStatus_t (*_cublasLtMatmulAlgoGetIds)(cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
        cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
        cudaDataType_t Dtype, int requestedAlgoCount, int algoIdsArray[], int* returnAlgoCount);
    cublasStatus_t (*_cublasLtMatrixLayoutCreate)(
        cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld);
    cublasStatus_t (*_cublasLtMatrixLayoutDestroy)(cublasLtMatrixLayout_t matLayout);
    cublasStatus_t (*_cublasLtMatrixLayoutSetAttribute)(
        cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void const* buf, size_t sizeInBytes);
    cublasStatus_t (*_cublasLtMatmulAlgoConfigGetAttribute)(cublasLtMatmulAlgo_t const* algo,
        cublasLtMatmulAlgoConfigAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
    cublasStatus_t (*_cublasLtMatmulAlgoConfigSetAttribute)(
        cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, void const* buf, size_t sizeInBytes);
    cublasStatus_t (*_cublasLtMatmulAlgoCapGetAttribute)(cublasLtMatmulAlgo_t const* algo,
        cublasLtMatmulAlgoCapAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
    cublasStatus_t (*_cublasLtMatmulDescSetAttribute)(
        cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void const* buf, size_t sizeInBytes);
};

CublasLtWrapper& getCublasLtWrapper();

} // namespace pluginInternal
} // namespace nvinfer1

#endif // TRT_PLUGIN_CUBLASLT_WRAPPER_H
