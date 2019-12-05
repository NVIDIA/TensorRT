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

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cublas_v2.h"
#include "cuda_fp16.hpp"
#include <cub/cub.cuh>
#include "logging.h"
#include "common.h"
#include "half.h"

extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void setReportableSeverity(Logger::Severity severity);

#define TRT_UNUSED (void)

#include <numeric>
#include <vector>

typedef __half half;
namespace bert
{
constexpr uint32_t BDIM = 1; // batch dimension
constexpr uint32_t SDIM = 0; // seq len dimension
constexpr uint32_t HDIM = 2; // hidden dimension

#define HDI inline __host__ __device__

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

template <typename T, typename R, int TPB>
__device__ inline void layerNorm(
    const kvp<R>& threadData, const int ld, const int offset, const float* beta, const float* gamma, T* output)
{
    // Assuming threadData is already divided by ld

    using BlockReduce = cub::BlockReduce<kvp<R>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ R mu;     // mean
    __shared__ R rsigma; // 1 / std.dev.

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
        const R val = output[idx];
        const R g(gamma[i]);
        const R b(beta[i]);
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

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, float* destDev)
{

    size_t wordSize = sizeof(float);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kFLOAT)
    {
        gLogVerbose << "Float Weights(Host) => Float Array(Device)" << std::endl;
        CHECK(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        gLogVerbose << "Half Weights(Host) => Float Array(Device)" << std::endl;
        std::vector<float> tmp(src.count);
        const half* values = reinterpret_cast<const half*>(src.values);

        for (int it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __half2float(values[it]);
        }

        CHECK(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, half* destDev)
{
    size_t wordSize = sizeof(half);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kHALF)
    {
        gLogVerbose << "Half Weights(Host) => Half Array(Device)" << std::endl;
        CHECK(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        gLogVerbose << "Float Weights(Host) => Half Array(Device)" << std::endl;
        std::vector<half> tmp(src.count);
        const float* values = reinterpret_cast<const float*>(src.values);

        for (int it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __float2half(values[it]);
        }
        CHECK(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype)
{
    switch (ftype)
    {
    case nvinfer1::PluginFieldType::kFLOAT32:
    {
        gLogVerbose << "PluginFieldType is Float32" << std::endl;
        return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16:
    {
        gLogVerbose << "PluginFieldType is Float16" << std::endl;
        return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32:
    {
        gLogVerbose << "PluginFieldType is Int32" << std::endl;
        return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8:
    {
        gLogVerbose << "PluginFieldType is Int8" << std::endl;
        return nvinfer1::DataType::kINT8;
    }
    default: throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
}
#endif // TRT_PLUGIN_UTIL_H
