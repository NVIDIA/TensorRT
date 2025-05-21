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

/*
 **************************************************************************
 * Modified from mmcv (https://github.com/open-mmlab/mmcv/tree/master/mmcv)
 * Copyright (c) OpenMMLab. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/open-mmlab/mmcv/blob/master/LICENSE
 **************************************************************************
 */

#include "commonCudaHelper.h"
#include "modulatedDeformConvCudaHelper.h"

using half = __half;
using namespace nvinfer1::pluginInternal;

namespace
{
template <class TScalar>
__global__ void copyPermuteKernel(
    TScalar* dst, TScalar const* src, int32_t n, TensorDesc tsSrcStride, TensorDesc tsDstStride, TensorDesc tsPermute)
{
    int32_t const srcDim = tsSrcStride.dim;
    int32_t const* const srcStride = &tsSrcStride.stride[0];
    int32_t const* const dstStride = &tsDstStride.stride[0];
    int32_t const* const permute = &tsPermute.shape[0];
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        int32_t dstIndex = index;
        int32_t srcIndex = 0;
        for (int32_t i = 0; i < srcDim; ++i)
        {
            int32_t dimIndex = dstIndex / dstStride[i];
            dstIndex = dstIndex % dstStride[i];
            srcIndex += dimIndex * srcStride[permute[i]];
        }
        dst[index] = src[srcIndex];
    }
}
} // namespace

template <class TScalar>
void memcpyPermute(
    TScalar* dst, TScalar const* src, int32_t* srcSize, int32_t* permute, int32_t srcDim, cudaStream_t stream)
{
    int32_t copySize = 1;
    TensorDesc tsPermute;
    memcpy(&(tsPermute.shape[0]), permute, srcDim * sizeof(int));

    TensorDesc tsSrcStride;
    TensorDesc tsDstStride;
    tsSrcStride.dim = srcDim;
    tsDstStride.dim = srcDim;
    int32_t* srcStride = &tsSrcStride.stride[0];
    int32_t* dstStride = &tsDstStride.stride[0];
    int32_t* dstSize = &tsDstStride.shape[0];
    srcStride[srcDim - 1] = 1;
    dstStride[srcDim - 1] = 1;

    for (int32_t i = srcDim - 1; i >= 0; --i)
    {
        dstSize[i] = srcSize[permute[i]];
        if (i < srcDim - 1)
        {
            srcStride[i] = srcStride[i + 1] * srcSize[i + 1];
        }
    }

    for (int32_t i = srcDim - 1; i >= 0; --i)
    {
        copySize *= dstSize[i];
        if (i < srcDim - 1)
        {
            dstStride[i] = dstStride[i + 1] * dstSize[i + 1];
        }
    }

    copyPermuteKernel<TScalar><<<get_blocks(copySize), THREADS_PER_BLOCK, 0, stream>>>(
        dst, src, copySize, tsSrcStride, tsDstStride, tsPermute);
}

template void memcpyPermute<float>(
    float* dst, float const* src, int32_t* srcSize, int32_t* permute, int32_t srcDim, cudaStream_t stream);

template void memcpyPermute<half>(
    half* dst, half const* src, int32_t* srcSize, int32_t* permute, int32_t srcDim, cudaStream_t stream);

template <typename TScalar>
cublasStatus_t cublasGemmWrap(cublasHandle_t handle, cudaStream_t stream, cublasOperation_t transa, cublasOperation_t transb, int32_t m,
    int32_t n, int32_t k, TScalar const* alpha, TScalar const* A, int32_t lda, TScalar const* B, int32_t ldb,
    TScalar const* beta, TScalar* C, int32_t ldc)
{
        return CUBLAS_STATUS_INTERNAL_ERROR;
}

template <>
cublasStatus_t cublasGemmWrap<float>(cublasHandle_t handle, cudaStream_t stream, cublasOperation_t transa, cublasOperation_t transb,
    int32_t m, int32_t n, int32_t k, float const* alpha, float const* A, int32_t lda, float const* B, int32_t ldb,
    float const* beta, float* C, int32_t ldc)
{
    CublasWrapper& wrapper = getCublasWrapper();
    // bind the stream to cublas handle to prevent usage of default stream
    wrapper.cublasSetStream(handle, stream);
    return wrapper.cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasGemmWrap<half>(cublasHandle_t handle, cudaStream_t stream, cublasOperation_t transa, cublasOperation_t transb,
    int32_t m, int32_t n, int32_t k, half const* alpha, half const* A, int32_t lda, half const* B, int32_t ldb,
    half const* beta, half* C, int32_t ldc)
{
    CublasWrapper& wrapper = getCublasWrapper();
    // bind the stream to cublas handle to prevent usage of default stream
    wrapper.cublasSetStream(handle, stream);
    return wrapper.cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
