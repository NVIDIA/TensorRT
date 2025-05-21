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

#ifndef TRT_MODULATED_DEFORM_CONV_CUDA_HELPER_H
#define TRT_MODULATED_DEFORM_CONV_CUDA_HELPER_H
#include "common/cublasWrapper.h"
#include <cstdint>

struct TensorDesc
{
    int32_t shape[10];
    int32_t stride[10];
    int32_t dim;
};

inline int64_t divUp(int64_t m, int32_t n)
{
    return (m + n - 1) / n;
}

template <class TScalar>
void memcpyPermute(
    TScalar* dst, TScalar const* src, int32_t* src_size, int32_t* permute, int32_t src_dim, cudaStream_t stream = 0);

template <typename TScalar>
nvinfer1::pluginInternal::cublasStatus_t cublasGemmWrap(nvinfer1::pluginInternal::cublasHandle_t handle,
    cudaStream_t stream, nvinfer1::pluginInternal::cublasOperation_t transa,
    nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m, int32_t n, int32_t k, TScalar const* alpha,
    TScalar const* A, int32_t lda, TScalar const* B, int32_t ldb, TScalar const* beta, TScalar* C, int32_t ldc);

#endif // TRT_MODULATED_DEFORM_CONV_CUDA_HELPER_H
