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

#pragma once

#include "cuda_fp16.h"

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define DESER(d, m) m = readFromBuffer<decltype(m)>(d)

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
inline T* deser2dev(const char*& buffer, size_t n_elem)
{
    T* dev = nullptr;
    size_t len = sizeof(T) * n_elem;
    cudaMalloc(&dev, len);
    cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice);

    buffer += len;
    return dev;
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t n_elem)
{
    size_t len = sizeof(T) * n_elem;
    cudaMemcpy(buffer, data, len, cudaMemcpyDeviceToHost);
    buffer += len;
}

template <typename T>
__device__ inline T myRsqrt(const T& x);

template <>
__device__ inline float myRsqrt(const float& x)
{
    return rsqrtf(x);
}

template <>
__device__ inline half myRsqrt(const half& x)
{
    return hrsqrt(x);
}

template <typename T>
__device__ inline T myTanh(const T& x);

template <>
__device__ inline float myTanh(const float& x)
{
    return tanhf(x);
}

template <>
__device__ inline half myTanh(const half& x)
{
    float tmp = tanhf(__half2float(x));
    return __float2half(tmp);
}

template <>
__device__ inline half2 myTanh(const half2& x)
{
    //at the moment, there is no half2 tanh builtin
    float2 tmp = (__half22float2(x));
    tmp.x = tanhf(tmp.x);
    tmp.y = tanhf(tmp.y);
    return __float22half2_rn(tmp);
}

template <typename T>
__device__ inline T myExp(const T x);

template <>
__device__ inline float myExp(const float x)
{
    return expf(x);
}

template <>
__device__ inline half myExp(const half x)
{
    return hexp(x);
}

typedef cub::KeyValuePair<float, float> kv_float;
typedef cub::KeyValuePair<half, half> kv_half;
typedef cub::KeyValuePair<half2, half2> kv_half2;

__device__ inline kv_float operator+(const kv_float& a, const kv_float& b)
{
    return kv_float(a.key + b.key, a.value + b.value);
}

__device__ inline kv_half operator+(const kv_half& a, const kv_half& b)
{
    half2 a2 = __halves2half2(a.key, a.value);
    half2 b2 = __halves2half2(b.key, b.value);
    half2 res = __hadd2(a2, b2);
    return kv_half(res.x, res.y);
}

__device__ inline kv_half2 operator+(const kv_half2& a, const kv_half2& b)
{
    return kv_half2(__hadd2(a.key, b.key), __hadd2(a.value, b.value));
}

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <typename T, int TPB>
__device__ inline void layer_norm(
    const kvp<T>& thread_data, const int ld, const int offset, const float* beta, const float* gamma, T* output)
{
    // Assuming thread_data is already divided by ld

    typedef cub::BlockReduce<kvp<T>, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    auto sumKV = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = myRsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        int idx = offset + i;
        T val = output[idx];
        T g(gamma[i]);
        T b(beta[i]);
        output[idx] = g * (val - mu) * rsigma + b;
    }
}

template <typename T, int TPB>
__device__ inline void layer_norm_small(const T val, const kvp<T>& thread_data, const int ld, const int idx,
    const float* beta, const float* gamma, T* output)
{
    // Assuming thread_data is already divided by ld
    // Small settings: the block covers the leading dimension TPB >= ld. The input
    // value is available in a register

    typedef cub::BlockReduce<kvp<T>, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    auto sumKV = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = myRsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        T g(gamma[threadIdx.x]);
        T b(beta[threadIdx.x]);
        output[idx] = g * (val - mu) * rsigma + b;
    }
}
