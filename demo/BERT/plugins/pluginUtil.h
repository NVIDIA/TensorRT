/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_PLUGIN_UTIL_H
#define TRT_PLUGIN_UTIL_H

#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "common.h"
#include <cub/cub.cuh>

namespace bert
{

constexpr uint32_t BDIM = 0; // batch dimension
constexpr uint32_t SDIM = 1; // seq len dimension
constexpr uint32_t HDIM = 2; // hidden dimension

#define DESER(d, m) m = readFromBuffer<decltype(m)>(d)

#define HDI inline __host__ __device__

// Helper function for serializing plugin
template <typename T>
inline void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
inline T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

template <typename T>
inline T* deserToDev(const char*& buffer, size_t nbElem)
{
    T* dev = nullptr;
    const size_t len = sizeof(T) * nbElem;
    CHECK(cudaMalloc(&dev, len));
    CHECK(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return dev;
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    CHECK(cudaMemcpy(buffer, data, len, cudaMemcpyDeviceToHost));
    buffer += len;
}

template <typename T>
__device__ inline T rsqrt(const T& x);

template <>
__device__ inline float rsqrt(const float& x)
{
    return rsqrtf(x);
}

template <>
__device__ inline half rsqrt(const half& x)
{
    return hrsqrt(x);
}

template <typename T>
__device__ inline T tanh(const T& x);

template <>
__device__ inline float tanh(const float& x)
{
    return tanhf(x);
}

template <>
__device__ inline half tanh(const half& x)
{
    const float tmp = tanhf(__half2float(x));
    return __float2half(tmp);
}

template <>
__device__ inline half2 tanh(const half2& x)
{
    // at the moment, there is no half2 tanh builtin
    float2 tmp = (__half22float2(x));
    tmp.x = tanhf(tmp.x);
    tmp.y = tanhf(tmp.y);
    return __float22half2_rn(tmp);
}

template <typename T>
__device__ inline T exp(const T x);

template <>
__device__ inline float exp(const float x)
{
    return expf(x);
}

template <>
__device__ inline half exp(const half x)
{
    return hexp(x);
}

using kv_float = cub::KeyValuePair<float, float>;
using kv_half = cub::KeyValuePair<half, half>;
using kv_half2 = cub::KeyValuePair<half2, half2>;

__device__ inline kv_float operator+(const kv_float& a, const kv_float& b)
{
    return kv_float(a.key + b.key, a.value + b.value);
}

__device__ inline kv_half operator+(const kv_half& a, const kv_half& b)
{
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = __hadd2(a2, b2);
    return kv_half(res.x, res.y);
}

__device__ inline kv_half2 operator+(const kv_half2& a, const kv_half2& b)
{
    return kv_half2(__hadd2(a.key, b.key), __hadd2(a.value, b.value));
}

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <typename T, int TPB>
__device__ inline void layerNorm(
    const kvp<T>& threadData, const int ld, const int offset, const float* beta, const float* gamma, T* output)
{
    // Assuming threadData is already divided by ld

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const T val = output[idx];
        const T g(gamma[i]);
        const T b(beta[i]);
        output[idx] = g * (val - mu) * rsigma + b;
    }
}

template <typename T, int TPB>
__device__ inline void layerNormSmall(const T val, const kvp<T>& threadData, const int ld, const int idx,
    const float* beta, const float* gamma, T* output)
{
    // Assuming threadData is already divided by ld
    // Small settings: the block covers the leading dimension TPB >= ld. The input
    // value is available in a register

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        const T g(gamma[threadIdx.x]);
        const T b(beta[threadIdx.x]);
        output[idx] = g * (val - mu) * rsigma + b;
    }
}

template <typename T, unsigned TPB>
__device__ inline void scaledSoftmaxSmall(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output)
{

    using BlockReduce = cub::BlockReduce<float, TPB>;

    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float rZ;

    const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    const float w(rsqrtHeadSize);
    cub::Sum sum;
    float threadData(0);

    const int idx = offset + threadIdx.x;
    if (threadIdx.x < lastValid)
    {
        const float val = input[idx];
        threadData = exp(val * w);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        rZ = (1.f) / Z;
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        // this will be 0 for threadIdx.x >= lastValid
        output[idx] = T(threadData * rZ);
    }
}

template <typename T, unsigned TPB>
__device__ inline void scaledSoftmax(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output)
{

    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float rZ;

    const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    const float w(rsqrtHeadSize);
    cub::Sum sum;
    float threadData(0);

    for (int i = threadIdx.x; i < lastValid; i += TPB)
    {
        const int idx = offset + i;
        const float val = input[idx];
        threadData += exp(val * w);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        rZ = 1.f / Z;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const float val = (i < lastValid) ? exp(float(input[idx]) * w) * rZ : 0.f;
        output[idx] = T(val);
    }
}

template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}
template <typename IntType>
constexpr HDI IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <typename T>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const T alpha, const T* A, int lda, const T* B, int ldb, const T beta, T* C, int ldc);

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C,
    int ldc)
{

    return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const half alpha, const half* A, int lda, const half* B, int ldb, const half beta, half* C, int ldc)
{
    return cublasHgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount);

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
    int batchCount)
{

    return cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
    const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
    int batchCount)
{
    return cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

struct CublasConfigHelper
{
    cublasPointerMode_t pm;
    cublasMath_t mm;
    cublasHandle_t cublas;
    CublasConfigHelper(cublasHandle_t cublas_)
        : cublas(cublas_)
    {
        cublasGetPointerMode(cublas, &pm);
        cublasGetMathMode(cublas, &mm);
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
        cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
    }
    ~CublasConfigHelper()
    {
        cublasSetMathMode(cublas, mm);
        cublasSetPointerMode(cublas, pm);
    }
};
}
#endif // TRT_PLUGIN_UTIL_H
