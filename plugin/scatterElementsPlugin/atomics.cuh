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
 *
 * ************************************************************************
 * Modified from pytorch_scatter 
 * Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
 * See https://github.com/rusty1s/pytorch_scatter/blob/master/LICENSE for details
 * ************************************************************************
 */

#ifndef TRT_SCATTER_ELEMENTS_ATOMICS_H
#define TRT_SCATTER_ELEMENTS_ATOMICS_H

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <utility>

#define ATOMIC(NAME)                                                                                                   \
    template <typename TScalar, size_t tSize>                                                                          \
    struct Atomic##NAME##IntegerImpl;                                                                                  \
                                                                                                                       \
    template <typename TScalar>                                                                                        \
    struct Atomic##NAME##IntegerImpl<TScalar, 4>                                                                       \
    {                                                                                                                  \
        inline __device__ void operator()(TScalar* address, TScalar val)                                               \
        {                                                                                                              \
            std::uint32_t* addressAsUI = reinterpret_cast<std::uint32_t*>(address);                                    \
            std::uint32_t old = *addressAsUI;                                                                          \
            std::uint32_t assumed;                                                                                     \
                                                                                                                       \
            do                                                                                                         \
            {                                                                                                          \
                assumed = old;                                                                                         \
                old = atomicCAS(addressAsUI, assumed, OP(val, static_cast<TScalar>(old)));                             \
            } while (assumed != old);                                                                                  \
        }                                                                                                              \
    };                                                                                                                 \
                                                                                                                       \
    template <typename TScalar>                                                                                        \
    struct Atomic##NAME##IntegerImpl<TScalar, 8>                                                                       \
    {                                                                                                                  \
        inline __device__ void operator()(TScalar* address, TScalar val)                                               \
        {                                                                                                              \
            unsigned long long* addressAsULL = reinterpret_cast<unsigned long long*>(address);                         \
            unsigned long long old = *addressAsULL;                                                                    \
            unsigned long long assumed;                                                                                \
                                                                                                                       \
            do                                                                                                         \
            {                                                                                                          \
                assumed = old;                                                                                         \
                old = atomicCAS(addressAsULL, assumed, OP(val, static_cast<TScalar>(old)));                            \
            } while (assumed != old);                                                                                  \
        }                                                                                                              \
    };                                                                                                                 \
                                                                                                                       \
    template <typename TScalar, size_t tSize>                                                                          \
    struct Atomic##NAME##DecimalImpl;                                                                                  \
                                                                                                                       \
    template <typename TScalar>                                                                                        \
    struct Atomic##NAME##DecimalImpl<TScalar, 4>                                                                       \
    {                                                                                                                  \
        inline __device__ void operator()(TScalar* address, TScalar val)                                               \
        {                                                                                                              \
            std::int32_t* addressAsI = reinterpret_cast<std::int32_t*>(address);                                       \
            std::int32_t old = *addressAsI;                                                                            \
            std::int32_t assumed;                                                                                      \
                                                                                                                       \
            do                                                                                                         \
            {                                                                                                          \
                assumed = old;                                                                                         \
                old = atomicCAS(addressAsI, assumed, __float_as_int(OP(val, __int_as_float(assumed))));                \
            } while (assumed != old);                                                                                  \
        }                                                                                                              \
    };                                                                                                                 \
    template <typename TScalar>                                                                                        \
    struct Atomic##NAME##DecimalImpl<TScalar, 2>                                                                       \
    {                                                                                                                  \
        inline __device__ void operator()(TScalar* address, TScalar val)                                               \
        {                                                                                                              \
            uint32_t* addressAsUI = reinterpret_cast<std::uint32_t*>((char*) address - ((std::size_t) address & 2));   \
            std::uint32_t old = *addressAsUI;                                                                          \
            std::uint32_t assumed;                                                                                     \
                                                                                                                       \
            do                                                                                                         \
            {                                                                                                          \
                assumed = old;                                                                                         \
                std::uint16_t hsum_old;                                                                                \
                hsum_old = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);                       \
                auto hsum = OP(*reinterpret_cast<TScalar*>(&hsum_old), val);                                           \
                old = (size_t) address & 2 ? (old & 0xffff) | ((*reinterpret_cast<std::uint16_t*>(&hsum)) << 16)       \
                                           : (old & 0xffff0000) | *reinterpret_cast<std::uint16_t*>(&hsum);            \
                old = atomicCAS(addressAsUI, assumed, old);                                                            \
            } while (assumed != old);                                                                                  \
        }                                                                                                              \
    };

#define OP(X, Y) ((Y) + (X))
ATOMIC(Add)
#undef OP

static inline __device__ void atomAdd(float* address, float val)
{
    atomicAdd(address, val);
}
static inline __device__ void atomAdd(__half* address, __half val)
{
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || CUDA_VERSION < 10000))
  AtomicAddDecimalImpl<__half, sizeof(__half)>()(address, val);
#else
  atomicAdd(address, val);
#endif
}
static inline __device__ void atomAdd(__nv_bfloat16* address, __nv_bfloat16 val)
{
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
  atomicAdd(address, val);
#else
  AtomicAddDecimalImpl<__nv_bfloat16, sizeof(__nv_bfloat16)>()(address, val);
#endif
}
static inline __device__ void atomAdd(std::int32_t* address, std::int32_t val)
{
    atomicAdd(address, val);
}
static inline __device__ void atomAdd(std::int64_t* address, std::int64_t val)
{
    AtomicAddIntegerImpl<std::int64_t, sizeof(std::int64_t)>()(address, val);
}

#define OP(X, Y) ((Y) * (X))
ATOMIC(Mul)
#undef OP
static inline __device__ void atomMul(std::int32_t* address, std::int32_t val)
{
    AtomicMulIntegerImpl<std::int32_t, sizeof(std::int32_t)>()(address, val);
}
static inline __device__ void atomMul(std::int64_t* address, std::int64_t val)
{
    AtomicMulIntegerImpl<std::int64_t, sizeof(std::int64_t)>()(address, val);
}
static inline __device__ void atomMul(float* address, float val)
{
    AtomicMulDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ void atomMul(__half* address, __half val)
{
    AtomicMulDecimalImpl<__half, sizeof(__half)>()(address, val);
}
static inline __device__ void atomMul(__nv_bfloat16* address, __nv_bfloat16 val)
{
    AtomicMulDecimalImpl<__nv_bfloat16, sizeof(__nv_bfloat16)>()(address, val);
}


#define OP(X, Y) ((X) < (Y)) ? (Y) : (X)
ATOMIC(Max)
#undef OP
static inline __device__ void atomMax(std::int32_t* address, std::int32_t val)
{
    atomicMax(address, val);
}
static inline __device__ void atomMax(float* address, float val)
{
    AtomicMaxDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ void atomMax(std::int64_t* address, std::int64_t val)
{
    AtomicMaxIntegerImpl<std::int64_t, sizeof(std::int64_t)>()(address, val);
}
static inline __device__ void atomMax(__half* address, __half val)
{
    AtomicMaxDecimalImpl<__half, sizeof(__half)>()(address, val);
}
static inline __device__ void atomMax(__nv_bfloat16* address, __nv_bfloat16 val)
{
    AtomicMaxDecimalImpl<__nv_bfloat16, sizeof(__nv_bfloat16)>()(address, val);
}

#define OP(X, Y) ((X) > (Y)) ? (Y) : (X)
ATOMIC(Min)
#undef OP
static inline __device__ void atomMin(std::int32_t* address, std::int32_t val)
{
    atomicMin(address, val);
}
static inline __device__ void atomMin(std::int64_t* address, std::int64_t val)
{
    AtomicMinIntegerImpl<std::int64_t, sizeof(std::int64_t)>()(address, val);
}
static inline __device__ void atomMin(float* address, float val)
{
    AtomicMinDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ void atomMin(__half* address, __half val)
{
    AtomicMinDecimalImpl<__half, sizeof(__half)>()(address, val);
}
static inline __device__ void atomMin(__nv_bfloat16* address, __nv_bfloat16 val)
{
    AtomicMinDecimalImpl<__nv_bfloat16, sizeof(__nv_bfloat16)>()(address, val);
}


#endif
