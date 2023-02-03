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

#ifndef INSTANCE_NORM_COMMON_H
#define INSTANCE_NORM_COMMON_H

#include <stdint.h>

#define DEVICE_FUNCTION static inline __device__

template <typename T, int32_t ELEMENTS_PER_LDG>
struct PackedStorage
{
    enum
    {
        PACKED_ELEMENTS_PER_LDG = ELEMENTS_PER_LDG
    };
    typedef T Type;
};

template <int32_t ELEMENTS_PER_LDG>
struct PackedStorage<uint16_t, ELEMENTS_PER_LDG>
{
    enum
    {
        PACKED_ELEMENTS_PER_LDG = ELEMENTS_PER_LDG / 2
    };
    typedef int32_t Type;
};

template <int32_t ELEMENTS_PER_LDG>
struct PackedStorage<int8_t, ELEMENTS_PER_LDG>
{
    enum
    {
        PACKED_ELEMENTS_PER_LDG = ELEMENTS_PER_LDG / 4
    };
    typedef int32_t Type;
};

template <int32_t N>
DEVICE_FUNCTION void fromFloat(int32_t (&dst)[N], float const (&src)[2 * N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        uint16_t lo, hi;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(lo) : "f"(src[2 * i + 0]));
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(hi) : "f"(src[2 * i + 1]));
        asm volatile("mov.b32 %0, {%1, %2};" : "=r"(dst[i]) : "h"(lo), "h"(hi));
    }
}

template <int32_t N>
DEVICE_FUNCTION void fromFloat(int32_t (&dst)[N], float const (&src)[4 * N], float scale)
{
    union Pack_t
    {
        int8_t x[4];
        int32_t val;
    };
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        Pack_t packed;
#pragma unroll
        for (int32_t ii = 0; ii < 4; ii++)
        {
            packed.x[ii] = __float_as_int(min(max(src[4 * i + ii] * scale + 12582912.0F, 12582785.0F), 12583039.0F));
        }
        dst[i] = packed.val;
    }
}

template <int32_t N>
DEVICE_FUNCTION void fromFloat(float (&dst)[N], float const (&src)[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        dst[i] = src[i];
    }
}

template <int32_t N, bool DO_SCALE = false>
DEVICE_FUNCTION void toFloat(float (&dst)[2 * N], int32_t (&src)[N], float scale = 1.f)
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        uint16_t lo, hi;
        asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(src[i]));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dst[2 * i + 0]) : "h"(lo));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dst[2 * i + 1]) : "h"(hi));
    }
}

template <int32_t N, bool DO_SCALE = false>
DEVICE_FUNCTION void toFloat(float (&dst)[4 * N], int32_t (&src)[N], float scale = 1.f)
{
    union Pack_t {
        int8_t x[4];
        int32_t val;
    };
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        Pack_t packed;
        packed.val = src[i];
#pragma unroll
        for (int32_t ii = 0; ii < 4; ++ii)
        {
            dst[4 * i + ii]
                = (DO_SCALE) ? __int2float_rn((int32_t) packed.x[ii]) * scale : __int2float_rn((int32_t) packed.x[ii]);
        }
    }
}

template <int32_t N, bool DO_SCALE = false>
DEVICE_FUNCTION void toFloat(float (&dst)[N], float (&src)[N], float scale = 1.f)
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        dst[i] = (DO_SCALE) ? src[i] * scale : src[i];
    }
}

template <typename T>
DEVICE_FUNCTION void ldg(int32_t (&dst)[1], T const* gmem)
{
    dst[0] = __ldg((int32_t const*) gmem);
}

template <typename T>
DEVICE_FUNCTION void ldgStream(int32_t (&dst)[1], T const* gmem)
{
    uint32_t tmp;
    asm volatile("ld.global.cs.nc.s32 %0, [%1];" : "=r"(tmp) : "l"((const uint32_t*) gmem));
    dst[0] = tmp;
}

template <typename T>
DEVICE_FUNCTION void ldg(int32_t (&dst)[2], T const* gmem)
{
    int2 tmp = __ldg((const int2*) gmem);
    dst[0] = tmp.x;
    dst[1] = tmp.y;
}

template <typename T>
DEVICE_FUNCTION void ldgStream(int32_t (&dst)[2], T const* gmem)
{
    int2 tmp;
    asm volatile("ld.global.cs.nc.v2.s32 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l"((const int2*) gmem));
    dst[0] = tmp.x;
    dst[1] = tmp.y;
}

DEVICE_FUNCTION void ldg(int32_t (&dst)[2], const uint16_t* gmem)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
    int2 tmp = __ldg((const int2*) gmem);
    dst[0] = tmp.x;
    dst[1] = tmp.y;
#endif
}

DEVICE_FUNCTION void ldgStream(int32_t (&dst)[2], const uint16_t* gmem)
{
    int2 tmp;
    asm volatile("ld.global.cs.nc.v2.s32 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l"((const int2*) gmem));
    dst[0] = tmp.x;
    dst[1] = tmp.y;
}

template <int32_t N>
DEVICE_FUNCTION void ldg(float (&dst)[N], const uint16_t* gmem)
{
    int32_t tmp[N / 2];
    ldg(tmp, gmem);
    toFloat(dst, tmp);
}

template <int32_t N>
DEVICE_FUNCTION void ldgStream(float (&dst)[N], const uint16_t* gmem)
{
    int32_t tmp[N / 2];
    ldgStream(tmp, gmem);
    toFloat(dst, tmp);
}

template <int32_t N>
DEVICE_FUNCTION void ldg(float (&dst)[N], const int8_t* gmem)
{
    int32_t tmp[N / 4];
    ldg(tmp, gmem);
    toFloat(dst, tmp);
}

template <int32_t N>
DEVICE_FUNCTION void ldgStream(float (&dst)[N], const int8_t* gmem)
{
    int32_t tmp[N / 4];
    ldgStream(tmp, gmem);
    toFloat(dst, tmp);
}

template <typename T>
DEVICE_FUNCTION void stg(T* gmem, int32_t (&src)[1])
{
    reinterpret_cast<int32_t*>(gmem)[0] = src[0];
}

template <typename T>
DEVICE_FUNCTION void stgStream(T* gmem, int32_t (&src)[1])
{
    uint32_t tmp = src[0];
    asm volatile("st.global.cs.s32 [%0], %1;" ::"l"((uint32_t*) gmem), "r"(tmp));
}

template <typename T>
DEVICE_FUNCTION void stg(T* gmem, int32_t (&src)[2])
{
    reinterpret_cast<int2*>(gmem)[0] = make_int2(src[0], src[1]);
}

template <typename T>
DEVICE_FUNCTION void stgStream(T* gmem, int32_t (&src)[2])
{
    asm volatile("st.global.cs.v2.s32 [%0], {%1,%2};" ::"l"((uint32_t*) gmem), "r"(src[0]), "r"(src[1]));
}

template <int32_t N>
DEVICE_FUNCTION void stg(uint16_t* gmem, float (&src)[N], float scale)
{
    int32_t tmp[N / 2];
    fromFloat(tmp, src);
    stg(gmem, tmp);
}

template <int32_t N>
DEVICE_FUNCTION void stgStream(uint16_t* gmem, float (&src)[N], float scale)
{
    int32_t tmp[N / 2];
    fromFloat(tmp, src);
    stgStream(gmem, tmp);
}

template <int32_t N>
DEVICE_FUNCTION void stg(int8_t* gmem, float (&src)[N], float scale)
{
    int32_t tmp[N / 4];
    fromFloat(tmp, src, scale);
    stg(gmem, tmp);
}

template <int32_t N>
DEVICE_FUNCTION void stgStream(int8_t* gmem, float (&src)[N], float scale)
{
    int32_t tmp[N / 4];
    fromFloat(tmp, src, scale);
    stg(gmem, tmp);
}

DEVICE_FUNCTION void readFromGmem(float (&dst)[2], float const* gmem, int32_t idx)
{
    float2 tmp = __ldg((float2*) &gmem[2 * idx]);
    dst[0] = tmp.x;
    dst[1] = tmp.y;
}

DEVICE_FUNCTION void readFromGmem(float (&dst)[4], float const* gmem, int32_t idx)
{
    float4 tmp = __ldg((float4*) &gmem[4 * idx]);
    dst[0] = tmp.x;
    dst[1] = tmp.y;
    dst[2] = tmp.z;
    dst[3] = tmp.w;
}

template <int32_t N>
DEVICE_FUNCTION void readFromGmem(float (&dst)[N], const __half* gmem, int32_t idx)
{
    int32_t ival[N / 2];
    if (N == 4)
        reinterpret_cast<int2*>(ival)[0] = __ldg((int2*) &gmem[4 * idx]);
    else
        reinterpret_cast<int32_t*>(ival)[0] = __ldg((int32_t*) &gmem[2 * idx]);
#pragma unroll
    for (int32_t i = 0; i < N / 2; ++i)
    {
        uint16_t lo, hi;
        asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(ival[i]));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dst[2 * i + 0]) : "h"(lo));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dst[2 * i + 1]) : "h"(hi));
    }
}

DEVICE_FUNCTION void readFromSmem(float (&x)[2], float const* smem, int32_t idx)
{
    float2 tmp = *(float2 const*) &smem[2 * idx];
    x[0] = tmp.x;
    x[1] = tmp.y;
}

DEVICE_FUNCTION void readFromSmem(float (&x)[4], float const* smem, int32_t idx)
{
    float4 tmp = *(float4 const*) &smem[4 * idx];
    x[0] = tmp.x;
    x[1] = tmp.y;
    x[2] = tmp.z;
    x[3] = tmp.w;
}

DEVICE_FUNCTION void readFromSmem(int32_t (&x)[1], int32_t const* smem, int32_t idx)
{
    x[0] = smem[idx];
}

DEVICE_FUNCTION void readFromSmem(int32_t (&x)[2], int32_t const* smem, int32_t idx)
{
    int2 tmp = *(int2 const*) &smem[2 * idx];
    x[0] = tmp.x;
    x[1] = tmp.y;
}

DEVICE_FUNCTION void writeToGmem(float* gmem, int32_t idx, float const (&src)[2])
{
    reinterpret_cast<float2*>(&gmem[2 * idx])[0] = make_float2(src[0], src[1]);
}

DEVICE_FUNCTION void writeToGmem(float* gmem, int32_t idx, float const (&src)[4])
{
    reinterpret_cast<float4*>(&gmem[4 * idx])[0] = make_float4(src[0], src[1], src[2], src[3]);
}

template <int32_t N>
DEVICE_FUNCTION void writeToGmem(__half* gmem, int32_t idx, float const (&src)[N])
{
    int32_t ival[N / 2];
#pragma unroll
    for (int32_t i = 0; i < N / 2; ++i)
    {
        uint16_t lo;
        uint16_t hi;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(lo) : "f"(src[2 * i + 0]));
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(hi) : "f"(src[2 * i + 1]));
        asm volatile("mov.b32 %0, {%1, %2};" : "=r"(ival[i]) : "h"(lo), "h"(hi));
    }
    if (N == 4)
    {
        reinterpret_cast<int2*>(&gmem[4 * idx])[0] = make_int2(ival[0], ival[1]);
    }
    else
    {
        reinterpret_cast<int32_t*>(&gmem[2 * idx])[0] = ival[0];
    }
}

DEVICE_FUNCTION void writeToSmem(float* smem, int32_t idx, float const (&x)[2])
{
    reinterpret_cast<float2*>(&smem[2 * idx])[0] = make_float2(x[0], x[1]);
}

DEVICE_FUNCTION void writeToSmem(float* smem, int32_t idx, float const (&x)[4])
{
    reinterpret_cast<float4*>(&smem[4 * idx])[0] = make_float4(x[0], x[1], x[2], x[3]);
}

DEVICE_FUNCTION void writeToSmem(int32_t* smem, int32_t idx, int32_t const (&x)[1])
{
    smem[idx] = x[0];
}

static inline __device__ void writeToSmem(int32_t* smem, int32_t idx, int32_t const (&x)[2])
{
    reinterpret_cast<int2*>(&smem[2 * idx])[0] = make_int2(x[0], x[1]);
}

template <int32_t N>
DEVICE_FUNCTION void zero(int32_t (&dst)[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        dst[i] = 0;
    }
}

template <int32_t N>
DEVICE_FUNCTION void zero(float (&dst)[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        dst[i] = 0.f;
    }
}

template <int32_t N>
DEVICE_FUNCTION void add(float (&x)[N], float const (&y)[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        x[i] += y[i];
    }
}

template <int32_t N>
DEVICE_FUNCTION void normalize(float (&x)[N], float const (&bias)[N], float const (&scale)[N], float const (&m1)[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        x[i] = bias[i] + scale[i] * (x[i] - m1[i]);
    }
}

template <typename Storage>
DEVICE_FUNCTION Storage relu(Storage in, Storage alpha)
{
    Storage zero = (Storage) 0.f;
    return (in < zero) ? in * alpha : in;
}

template <int32_t N>
DEVICE_FUNCTION void reluActivation(float (&x)[N], float alpha)
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i)
    {
        x[i] = relu(x[i], alpha);
    }
}

template <int32_t THREADS_PER_CTA>
DEVICE_FUNCTION void parallelSums_16x2(float* smem, float (&x)[4], int32_t nhw)
{

    // The size of a warp.
    int32_t const THREADS_PER_WARP = 32;
    // The number of warps in a CTA.
    int32_t const WARPS_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;
    // The number of threads per pixel.
    int32_t const THREADS_PER_PIXEL = 16;
    // The number of elements per ldg.
    int32_t const ELEMENTS_PER_LDG = 4;
    // The warp decomposition.
    int32_t const warp_id = threadIdx.x / THREADS_PER_WARP;
    int32_t const lane_id = threadIdx.x % THREADS_PER_WARP;

    // Store the values to shared memory.
    writeToSmem(smem, threadIdx.x, x);

    // Compute the parallel sum inside the warp. Use SHFL and reduce the amount of SMEM by 2x?
    __syncwarp();

    // Read the running sum from the other thread in the warp.
    float y[ELEMENTS_PER_LDG];
    if (lane_id < THREADS_PER_PIXEL)
    {
        readFromSmem(y, smem, threadIdx.x + THREADS_PER_PIXEL);
    }

    // Compute the updated sum.
    add(x, y);

    // The data is in SMEM. Do the final reduction.
    __syncthreads();

    // The warp leaders, write to SMEM.
    if (lane_id < THREADS_PER_PIXEL)
    {
        writeToSmem(smem, warp_id * THREADS_PER_PIXEL + lane_id, x);
    }

    // The data is in SMEM. Do the final reduction.
    __syncthreads();

    // The 1st warp does all the work.
    if (warp_id == 0)
    {
        readFromSmem(x, smem, threadIdx.x);
    }

// We do the final reduction each half-warp sequentially reduces the final values.
#pragma unroll
    for (int32_t offset = 1; offset < WARPS_PER_CTA / 2; ++offset)
    {

        // Read the mean and variance from the other pixel.
        if (warp_id == 0)
        {
            readFromSmem(y, smem, threadIdx.x + offset * THREADS_PER_WARP);
        }

        // Compute the updated sum.
        add(x, y);
    }

    // Make sure the data is in SMEM.
    __syncwarp();

    // Store the mean/var for the different pixels. TODO: Use SHFL?
    if (warp_id == 0)
    {
        writeToSmem(smem, threadIdx.x, x);
    }

    // Make sure the data is in SMEM.
    __syncwarp();

    // The first half warp finishes the work.
    if (threadIdx.x < THREADS_PER_PIXEL)
    {
        readFromSmem(y, smem, threadIdx.x + THREADS_PER_PIXEL);
    }

    // Compute the updated sum.
    add(x, y);

    // Make sure the data was read from SMEM.
    __syncwarp();

    // Store the final values.
    if (threadIdx.x < THREADS_PER_PIXEL)
    {
        writeToSmem(smem, threadIdx.x, x);
    }
}

template <int32_t THREADS_PER_CTA>
static inline __device__ void parallelSums_8x4(float* smem, float (&x)[4], int32_t nhw)
{
    // The size of a warp.
    int32_t const THREADS_PER_WARP = 32;
    // The number of warps in a CTA.
    int32_t const WARPS_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;
    // The number of threads per pixel.
    int32_t const THREADS_PER_PIXEL = 8;
    // The number of elements per ldg.
    int32_t const ELEMENTS_PER_LDG = 4;
    // The warp decomposition.
    int32_t const warp_id = threadIdx.x / THREADS_PER_WARP;
    int32_t const lane_id = threadIdx.x % THREADS_PER_WARP;

#pragma unroll
    for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
    {
        x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL + lane_id);
        x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL * 2 + lane_id);
    }

    // The warp leaders, write to SMEM.
    if (lane_id < THREADS_PER_PIXEL)
    {
        writeToSmem(smem, warp_id * THREADS_PER_PIXEL + lane_id, x);
    }

    // The data is in SMEM. Do the final reduction.
    __syncthreads();

    // The 1st warp does all the work.
    // We do the final reduction each half-warp sequentially reduces the final values.
    if (warp_id == 0)
    {
        readFromSmem(x, smem, threadIdx.x);

#pragma unroll
        for (int32_t offset = 1; offset < WARPS_PER_CTA / (THREADS_PER_WARP / THREADS_PER_PIXEL); ++offset)
        {
            float y[ELEMENTS_PER_LDG];
            // Read the mean and variance from the other pixel.
            readFromSmem(y, smem, threadIdx.x + offset * THREADS_PER_WARP);
            // Compute the updated sum.
            add(x, y);
        }

        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL + lane_id);
            x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL * 2 + lane_id);
        }

        // Make sure the data was read from SMEM.
        __syncwarp();

        // Store the final values.
        if (threadIdx.x < THREADS_PER_PIXEL)
        {
            writeToSmem(smem, threadIdx.x, x);
        }
    }
}

template <int32_t THREADS_PER_CTA, int32_t THREADS_PER_PIXEL, int32_t ELEMENTS_PER_LDG>
DEVICE_FUNCTION void parallelSums(float* smem, float (&x)[ELEMENTS_PER_LDG], int32_t nhw)
{

    // The size of a warp.
    int32_t const THREADS_PER_WARP = 32;
    // The number of warps in a CTA.
    int32_t const WARPS_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;
    // The number of pixels computed by a single warp.
    int32_t const PIXELS_PER_WARP = THREADS_PER_WARP / THREADS_PER_PIXEL;

    // The position in the warp.
    int32_t const nhw_in_warp = nhw % PIXELS_PER_WARP;
    // The C in the warp.
    int32_t const c_in_warp = threadIdx.x % THREADS_PER_PIXEL;

    // Store the values to shared memory.
    writeToSmem(smem, threadIdx.x, x);

    // Compute the parallel sums.
    for (int32_t offset = PIXELS_PER_WARP / 2; offset > 0; offset /= 2)
    {

        if ((WARPS_PER_CTA * THREADS_PER_WARP) / THREADS_PER_PIXEL > THREADS_PER_WARP)
        {
            __syncthreads();
        }
        else
        {
            // NOP.
            __syncwarp();
        }

        // Read the running sum from the other thread.
        float y[ELEMENTS_PER_LDG];
        if (nhw_in_warp < offset)
        {
            readFromSmem(y, smem, threadIdx.x + offset * THREADS_PER_PIXEL);
        }

        // Compute the updated sum.
        add(x, y);

        if ((WARPS_PER_CTA * THREADS_PER_WARP) / THREADS_PER_PIXEL > THREADS_PER_WARP)
        {
            __syncthreads();
        }
        else
        {
            // NOP.
            __syncwarp();
        }

        // Update the sum in SMEM.
        if (offset > 1 && nhw_in_warp < offset)
        {
            writeToSmem(smem, threadIdx.x, x);
        }
    }

    // The warps are done. Do the final reduction at the CTA level.
    __syncthreads();

    // The warp leaders, write to SMEM.
    int32_t const idx = (threadIdx.x / THREADS_PER_WARP) * THREADS_PER_PIXEL + c_in_warp;
    if (nhw_in_warp == 0)
    {
        writeToSmem(smem, idx, x);
    }

    // The data is in SMEM. Do the final reduction.
    __syncthreads();

    // Read the 1st element to prepare the work.
    if (nhw < WARPS_PER_CTA / 2)
    {
        readFromSmem(x, smem, threadIdx.x);
    }

    // We have the running mean and running m2. Let's build the mean/var of the CTA.
    for (int32_t offset = WARPS_PER_CTA / 2; offset > 0; offset /= 2)
    {

        if ((WARPS_PER_CTA * THREADS_PER_WARP) / THREADS_PER_PIXEL > THREADS_PER_WARP)
        {
            __syncthreads();
        }
        else
        {
            // NOP.
            __syncwarp();
        }

        // Read the mean and variance from the other pixel.
        float y[ELEMENTS_PER_LDG];
        if (nhw < offset)
        {
            readFromSmem(y, smem, threadIdx.x + offset * THREADS_PER_PIXEL);
        }

        // Compute the updated sum.
        add(x, y);

        if ((WARPS_PER_CTA * THREADS_PER_WARP) / THREADS_PER_PIXEL > THREADS_PER_WARP)
        {
            __syncthreads();
        }
        else
        {
            // NOP.
            __syncwarp();
        }

        // Store the mean/var for the different pixels.
        if (nhw < offset)
        {
            writeToSmem(smem, threadIdx.x, x);
        }
    }
}

template <int32_t THREADS_PER_PIXEL, int32_t ELEMENTS_PER_LDG>
struct ParallelSums
{
    template <int32_t THREADS_PER_CTA>
    DEVICE_FUNCTION void dispatch(float* smem, float (&x)[ELEMENTS_PER_LDG], int32_t nhw)
    {
        parallelSums<THREADS_PER_CTA, THREADS_PER_PIXEL, ELEMENTS_PER_LDG>(smem, x, nhw);
    }
};

template <>
struct ParallelSums<16, 4>
{
    template <int32_t THREADS_PER_CTA>
    DEVICE_FUNCTION void dispatch(float* smem, float (&x)[4], int32_t nhw)
    {
        parallelSums_16x2<THREADS_PER_CTA>(smem, x, nhw);
    }
};

template <>
struct ParallelSums<8, 4>
{
    template <int32_t THREADS_PER_CTA>
    static inline __device__ void dispatch(float* smem, float (&x)[4], int32_t nhw)
    {
        parallelSums_8x4<THREADS_PER_CTA>(smem, x, nhw);
    }
};

#endif // INSTANCE_NORM_COMMON_H
