/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "skipLayerNormPluginLegacy.h"

#include <cassert>
#include <cstring>
#include <limits>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

template <int32_t TPB, int32_t VPT, bool hasBias>
__global__ void skiplnDQQ(int32_t const ld, int8_t const* input, int8_t const* skip, int8_t* output, __half const* beta,
    __half const* gamma, __half const* bias, float const dqScaleIn, float const dqScaleSkip, float const qScale)
{
    int32_t const idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    int8_t inLocal[VPT];
    int8_t skipLocal[VPT];

    __half inLocalDQ[VPT]; // dequantized input + skip + bias
    __half biasLocal[VPT];  // bias and beta
    __half gammaLocal[VPT];
    copy<sizeof(int8_t) * VPT>(&input[idx], inLocal);
    copy<sizeof(int8_t) * VPT>(&skip[idx], skipLocal);
    copy<sizeof(__half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    __half2 loc = __floats2half2_rn(0.f, 0.f); // accumulator

    const __half rld = __half(1) / __half(ld);
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        // DQ input and skip
        float const tmpIn = inLocal[it];
        float const tmpSkip = skipLocal[it];
        inLocalDQ[it] = dqScaleIn * tmpIn + dqScaleSkip * tmpSkip;

        if (hasBias)
            inLocalDQ[it] += biasLocal[it];
        const __half tmp = rld * inLocalDQ[it];
        const __half2 tmp2 = __halves2half2(tmp, tmp * inLocalDQ[it]);
        loc = loc + tmp2;
    }
    // load parameters
    copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT], biasLocal);
    copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], gammaLocal);

    using BlockReduce = cub::BlockReduce<__half2, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ __half mu;     // mean
    __shared__ __half rsigma; // 1 / std.dev.

    const __half2 sum2 = BlockReduce(tempStorage).Reduce(loc, [](auto const& lhs, auto const& rhs){return lhs + rhs;});

    if (threadIdx.x == 0)
    {
        mu = __low2half(sum2);
        rsigma = rsqrt(__high2half(sum2) - mu * mu + std::numeric_limits<half>::epsilon());
    }
    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t outLocal[VPT / 4U];
#pragma unroll
    for (int32_t it = 0; it < VPT / 4U; it++)
    {
        float const tmp0 = gammaLocal[it * 4 + 0] * (inLocalDQ[it * 4 + 0] - mu) * rsigma + biasLocal[it * 4 + 0];
        float const tmp1 = gammaLocal[it * 4 + 1] * (inLocalDQ[it * 4 + 1] - mu) * rsigma + biasLocal[it * 4 + 1];
        float const tmp2 = gammaLocal[it * 4 + 2] * (inLocalDQ[it * 4 + 2] - mu) * rsigma + biasLocal[it * 4 + 2];
        float const tmp3 = gammaLocal[it * 4 + 3] * (inLocalDQ[it * 4 + 3] - mu) * rsigma + biasLocal[it * 4 + 3];
        outLocal[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
}

template <typename T, int32_t TPB, int32_t VPT, bool hasBias>
__global__ void skipln_vec(
    int32_t const ld, const T* input, const T* skip, T* output, const T* beta, const T* gamma, const T* bias)
{
    int32_t const idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T inLocal[VPT];
    T skipLocal[VPT];
    T biasLocal[VPT];
    // T gammaLocal[VPT];
    copy<sizeof(T) * VPT>(&input[idx], inLocal);
    copy<sizeof(T) * VPT>(&skip[idx], skipLocal);
    copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    T local = 0.f;
    T local2 = 0.f;

    const T rld = T(1) / T(ld);
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        inLocal[it] += skipLocal[it];
        if (hasBias)
            inLocal[it] += biasLocal[it];
        const T tmp = rld * inLocal[it];
        local += tmp;
        local2 += tmp * inLocal[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], biasLocal);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], skipLocal);

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    auto const sumKV = BlockReduce(tempStorage).Reduce(kvp<T>(local, local2), [](auto const& lhs, auto const& rhs){return lhs + rhs;});

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + std::numeric_limits<T>::epsilon());
    }
    __syncthreads();
    ///*
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        inLocal[it] = skipLocal[it] * (inLocal[it] - mu) * rsigma + biasLocal[it];
    }
    /* */

    copy<sizeof(T) * VPT>(inLocal, &output[idx]);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernelSmall(
    int32_t const ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, const T* bias)
{

    const T rld = T(1) / T(ld);
    int32_t const offset = blockIdx.x * ld;

    // reduce x and x^2
    kvp<T> threadData(0, 0);
    int32_t const idx = offset + threadIdx.x;
    T val = 0;

    if (threadIdx.x < ld)
    {

        val = input[idx] + skip[idx];
        if (hasBias)
        {
            val += bias[threadIdx.x];
        }

        const T rldval = rld * val;
        threadData = threadData + kvp<T>(rldval, rldval * val);
    }

    layerNormSmall<T, T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernel(
    int32_t const ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, const T* bias)
{
    const T rld = T(1) / T(ld);
    int32_t const offset = blockIdx.x * ld;

    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int32_t i = threadIdx.x; i < ld; i += TPB)
    {
        int32_t const idx = offset + i;
        T val = T(input[idx]) + T(skip[idx]);

        if (hasBias)
        {
            val += T(bias[i]);
        }
        const T rldval = rld * val;
        threadData = threadData + kvp<T>(rldval, rldval * val);
        output[idx] = val;
    }

    layerNorm<T, T, T, TPB>(threadData, ld, offset, beta, gamma, output);
}

template <bool hasBias>
int32_t computeSkipLayerNormDQQ(cudaStream_t stream, int32_t const ld, int32_t const n, int8_t const* input,
    int8_t const* skip, __half const* beta, __half const* gamma, int8_t* output, __half const* bias,
    float const dqScaleIn, float const dqScaleSkip, float const qScale)
{
    // this must be true because n is the total size of the tensor
    PLUGIN_VALIDATE(n % ld == 0);

    int32_t const gridSize = n / ld;
    // we're limited by the size of the parameters, i.e. 8-wide instead of 16
    constexpr int32_t VPT = 16 / sizeof(__half);
    if (ld == 768)
    {
        constexpr int32_t TPB = 768 / VPT;
        skiplnDQQ<TPB, VPT, hasBias>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias, dqScaleIn, dqScaleSkip, qScale);
    }
    else if (ld == 1024)
    {
        constexpr int32_t TPB = 1024 / VPT;
        skiplnDQQ<TPB, VPT, hasBias>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias, dqScaleIn, dqScaleSkip, qScale);
    }
    else
    {
        // TODO need to implement this
        PLUGIN_ERROR(("SkipLayerNormDQQ - FATAL: unsupported hidden layer size: " + std::to_string(ld)).c_str());
    }
    PLUGIN_CHECK(cudaPeekAtLastError());

    return 0;
}

template <typename T, bool hasBias>
int32_t computeSkipLayerNorm(cudaStream_t stream, int32_t const ld, int32_t const n, const T* input, const T* skip,
    const T* beta, const T* gamma, T* output, const T* bias)
{

    // this must be true because n is the total size of the tensor
    PLUGIN_VALIDATE(n % ld == 0);
    int32_t const gridSize = n / ld;
    constexpr int32_t VPT = 16 / sizeof(T);
    if (ld <= 32)
    {
        constexpr int32_t blockSize = 32;
        skipLayerNormKernelSmall<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    else if (ld == 768)
    {
        constexpr int32_t TPB = 768 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias);
    }
    else if (ld == 1024)
    {
        constexpr int32_t TPB = 1024 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias);
    }
    else
    {
        constexpr int32_t blockSize = 256;
        skipLayerNormKernel<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    PLUGIN_CHECK(cudaPeekAtLastError());

    return 0;
}

template int32_t computeSkipLayerNormDQQ<true>(cudaStream_t stream, int32_t const ld, int32_t const n,
    int8_t const* input, int8_t const* skip, __half const* beta, __half const* gamma, int8_t* output,
    __half const* bias, float const dqScaleIn, float const dqScaleSkip, float const qScale);
template int32_t computeSkipLayerNormDQQ<false>(cudaStream_t stream, int32_t const ld, int32_t const n,
    int8_t const* input, int8_t const* skip, __half const* beta, __half const* gamma, int8_t* output,
    __half const* bias, float const dqScaleIn, float const dqScaleSkip, float const qScale);

template int32_t computeSkipLayerNorm<float, true>(cudaStream_t, int32_t const, int32_t const, float const*,
    float const*, float const*, float const*, float*, float const*);
template int32_t computeSkipLayerNorm<float, false>(cudaStream_t, int32_t const, int32_t const, float const*,
    float const*, float const*, float const*, float*, float const*);
template int32_t computeSkipLayerNorm<half, true>(
    cudaStream_t, int32_t const, int32_t const, half const*, half const*, half const*, half const*, half*, half const*);
template int32_t computeSkipLayerNorm<half, false>(
    cudaStream_t, int32_t const, int32_t const, half const*, half const*, half const*, half const*, half*, half const*);

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // CUDA_VERSION >= 10010
