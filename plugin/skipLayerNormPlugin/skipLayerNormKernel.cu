/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/serialize.hpp"
#include "skipLayerNormPlugin.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

template <int TPB, int VPT, bool hasBias>
__global__ void skiplnDQQ(const int ld, const int8_t* input, const int8_t* skip, int8_t* output, const __half* beta,
    const __half* gamma, const __half* bias, const float dqScaleIn, const float dqScaleSkip, const float qScale)
{
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    int8_t in_local[VPT];
    int8_t skip_local[VPT];

    __half in_local_dq[VPT]; // dequantized input + skip + bias
    __half bias_local[VPT];  // bias and beta
    __half gamma_local[VPT];
    copy<sizeof(int8_t) * VPT>(&input[idx], in_local);
    copy<sizeof(int8_t) * VPT>(&skip[idx], skip_local);
    copy<sizeof(__half) * VPT>(&bias[threadIdx.x * VPT], bias_local);
    __half2 loc = __floats2half2_rn(0.f, 0.f); // accumulator

    const __half rld = __half(1) / __half(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        // DQ input and skip
        const float tmp_in = in_local[it];
        const float tmp_skip = skip_local[it];
        in_local_dq[it] = dqScaleIn * tmp_in + dqScaleSkip * tmp_skip;

        if (hasBias)
            in_local_dq[it] += bias_local[it];
        const __half tmp = rld * in_local_dq[it];
        const __half2 tmp2 = __halves2half2(tmp, tmp * in_local_dq[it]);
        loc = loc + tmp2;
    }
    // load parameters
    copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT], bias_local);
    copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], gamma_local);

    using BlockReduce = cub::BlockReduce<__half2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ __half mu;     // mean
    __shared__ __half rsigma; // 1 / std.dev.

    const __half2 sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = __low2half(sum2);
        rsigma = rsqrt(__high2half(sum2) - mu * mu );
    }
    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t out_local[VPT/4];
#pragma unroll
    for (int it = 0; it < VPT / 4; it++)
    {
        const float tmp0 = gamma_local[it*4+0] * (in_local_dq[it*4+0] - mu) * rsigma + bias_local[it*4+0];
        const float tmp1 = gamma_local[it*4+1] * (in_local_dq[it*4+1] - mu) * rsigma + bias_local[it*4+1];
        const float tmp2 = gamma_local[it*4+2] * (in_local_dq[it*4+2] - mu) * rsigma + bias_local[it*4+2];
        const float tmp3 = gamma_local[it*4+3] * (in_local_dq[it*4+3] - mu) * rsigma + bias_local[it*4+3];
        out_local[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(out_local, &output[idx]);
}

template <typename T, int TPB, int VPT, bool hasBias>
__global__ void skipln_vec(
    const int ld, const T* input, const T* skip, T* output, const T* beta, const T* gamma, const T* bias)
{
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T in_local[VPT];
    T skip_local[VPT];
    T bias_local[VPT];
    // T gamma_local[VPT];
    copy<sizeof(T) * VPT>(&input[idx], in_local);
    copy<sizeof(T) * VPT>(&skip[idx], skip_local);
    copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], bias_local);
    T local = 0.f;
    T local2 = 0.f;

    const T rld = T(1) / T(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] += skip_local[it];
        if (hasBias)
            in_local[it] += bias_local[it];
        const T tmp = rld * in_local[it];
        local += tmp;
        local2 += tmp * in_local[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], bias_local);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], skip_local);

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<T>(local, local2), cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu );
    }
    __syncthreads();
    ///*
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] = skip_local[it] * (in_local[it] - mu) * rsigma + bias_local[it];
    }
    /* */

    copy<sizeof(T) * VPT>(in_local, &output[idx]);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, const T* bias)
{

    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);
    const int idx = offset + threadIdx.x;
    T val = 0;

    if (threadIdx.x < ld)
    {

        val = input[idx] + skip[idx];
        if (hasBias)
        {
            val += bias[threadIdx.x];
        }

        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    layerNormSmall<T, T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, const T* bias)
{
    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        T val = T(input[idx]) + T(skip[idx]);

        if (hasBias)
        {
            val += T(bias[i]);
        }
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        output[idx] = val;
    }

    layerNorm<T, T, T, TPB>(threadData, ld, offset, beta, gamma, output);
}

template <bool hasBias>
int computeSkipLayerNormDQQ(cudaStream_t stream, const int ld, const int n, const int8_t* input, const int8_t* skip,
    const __half* beta, const __half* gamma, int8_t* output, const __half* bias, const float dqScaleIn,
    const float dqScaleSkip, const float qScale)
{
    // this must be true because n is the total size of the tensor
    PLUGIN_ASSERT(n % ld == 0);

    const int gridSize = n / ld;
    // we're limited by the size of the parameters, i.e. 8-wide instead of 16
    constexpr int VPT = 16 / sizeof(__half);
    if (ld == 768)
    {
        constexpr int TPB = 768 / VPT;
        skiplnDQQ<TPB, VPT, hasBias>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias, dqScaleIn, dqScaleSkip, qScale);
    }
    else if (ld == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        skiplnDQQ<TPB, VPT, hasBias>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias, dqScaleIn, dqScaleSkip, qScale);
    }
    else
    {
        // TODO need to implement this
        gLogError << "SkipLayerNormDQQ - FATAL: unsupported hidden layer size: " << ld << std::endl;
        exit(0);
    }
    PLUGIN_CHECK(cudaPeekAtLastError());

    return 0;
}

template <typename T, bool hasBias>
int computeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* skip, const T* beta,
    const T* gamma, T* output, const T* bias)
{

    // this must be true because n is the total size of the tensor
    PLUGIN_ASSERT(n % ld == 0);
    const int gridSize = n / ld;
    constexpr int VPT = 16 / sizeof(T);
    if (ld <= 32)
    {
        constexpr int blockSize = 32;
        skipLayerNormKernelSmall<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    else if (ld == 768)
    {
        constexpr int TPB = 768 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias);
    }
    else if (ld == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias);
    }
    else
    {
        constexpr int blockSize = 256;
        skipLayerNormKernel<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    PLUGIN_CHECK(cudaPeekAtLastError());

    return 0;
}

template int computeSkipLayerNormDQQ<true>(cudaStream_t stream, const int ld, const int n, const int8_t* input, const int8_t* skip,
    const __half* beta, const __half* gamma, int8_t* output, const __half* bias, const float dqScaleIn,
    const float dqScaleSkip, const float qScale);
template int computeSkipLayerNormDQQ<false>(cudaStream_t stream, const int ld, const int n, const int8_t* input, const int8_t* skip,
    const __half* beta, const __half* gamma, int8_t* output, const __half* bias, const float dqScaleIn,
    const float dqScaleSkip, const float qScale);

template int computeSkipLayerNorm<float, true>(cudaStream_t, const int, const int, const float*, const float*, const float*, const float*, float*, const float*);
template int computeSkipLayerNorm<float, false>(cudaStream_t, const int, const int, const float*, const float*, const float*, const float*, float*, const float*);
template int computeSkipLayerNorm<half, true>(cudaStream_t, const int, const int, const half*, const half*, const half*, const half*, half*, const half*);
template int computeSkipLayerNorm<half, false>(cudaStream_t, const int, const int, const half*, const half*, const half*, const half*, half*, const half*);

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // CUDA_VERSION >= 10010
