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

#ifndef TRT_EFFICIENT_NMS_INFERENCE_CUH
#define TRT_EFFICIENT_NMS_INFERENCE_CUH

#include <cuda_fp16.h>

// FP32 Intrinsics

float __device__ __inline__ exp_mp(const float a)
{
    return __expf(a);
}
float __device__ __inline__ sigmoid_mp(const float a)
{
    return __frcp_rn(__fadd_rn(1.f, __expf(-a)));
}
float __device__ __inline__ add_mp(const float a, const float b)
{
    return __fadd_rn(a, b);
}
float __device__ __inline__ sub_mp(const float a, const float b)
{
    return __fsub_rn(a, b);
}
float __device__ __inline__ mul_mp(const float a, const float b)
{
    return __fmul_rn(a, b);
}
bool __device__ __inline__ gt_mp(const float a, const float b)
{
    return a > b;
}
bool __device__ __inline__ lt_mp(const float a, const float b)
{
    return a < b;
}
bool __device__ __inline__ lte_mp(const float a, const float b)
{
    return a <= b;
}
bool __device__ __inline__ gte_mp(const float a, const float b)
{
    return a >= b;
}

#if __CUDA_ARCH__ >= 530

// FP16 Intrinsics

__half __device__ __inline__ exp_mp(const __half a)
{
    return hexp(a);
}
__half __device__ __inline__ sigmoid_mp(const __half a)
{
    return hrcp(__hadd((__half) 1, hexp(__hneg(a))));
}
__half __device__ __inline__ add_mp(const __half a, const __half b)
{
    return __hadd(a, b);
}
__half __device__ __inline__ sub_mp(const __half a, const __half b)
{
    return __hsub(a, b);
}
__half __device__ __inline__ mul_mp(const __half a, const __half b)
{
    return __hmul(a, b);
}
bool __device__ __inline__ gt_mp(const __half a, const __half b)
{
    return __hgt(a, b);
}
bool __device__ __inline__ lt_mp(const __half a, const __half b)
{
    return __hlt(a, b);
}
bool __device__ __inline__ lte_mp(const __half a, const __half b)
{
    return __hle(a, b);
}
bool __device__ __inline__ gte_mp(const __half a, const __half b)
{
    return __hge(a, b);
}

#else

// FP16 Fallbacks on older architectures that lack support

__half __device__ __inline__ exp_mp(const __half a)
{
    return __float2half(exp_mp(__half2float(a)));
}
__half __device__ __inline__ sigmoid_mp(const __half a)
{
    return __float2half(sigmoid_mp(__half2float(a)));
}
__half __device__ __inline__ add_mp(const __half a, const __half b)
{
    return __float2half(add_mp(__half2float(a), __half2float(b)));
}
__half __device__ __inline__ sub_mp(const __half a, const __half b)
{
    return __float2half(sub_mp(__half2float(a), __half2float(b)));
}
__half __device__ __inline__ mul_mp(const __half a, const __half b)
{
    return __float2half(mul_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ gt_mp(const __half a, const __half b)
{
    return __float2half(gt_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ lt_mp(const __half a, const __half b)
{
    return __float2half(lt_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ lte_mp(const __half a, const __half b)
{
    return __float2half(lte_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ gte_mp(const __half a, const __half b)
{
    return __float2half(gte_mp(__half2float(a), __half2float(b)));
}

#endif

template <typename T>
struct __align__(4 * sizeof(T)) BoxCorner;

template <typename T>
struct __align__(4 * sizeof(T)) BoxCenterSize;

template <typename T>
struct __align__(4 * sizeof(T)) BoxCorner
{
    // For NMS/IOU purposes, YXYX coding is identical to XYXY
    T y1, x1, y2, x2;

    __device__ void reorder()
    {
        if (gt_mp(y1, y2))
        {
            // Swap values, so y1 < y2
            y1 = sub_mp(y1, y2);
            y2 = add_mp(y1, y2);
            y1 = sub_mp(y2, y1);
        }
        if (gt_mp(x1, x2))
        {
            // Swap values, so x1 < x2
            x1 = sub_mp(x1, x2);
            x2 = add_mp(x1, x2);
            x1 = sub_mp(x2, x1);
        }
    }

    __device__ BoxCorner<T> clip(T low, T high) const
    {
        return {lt_mp(y1, low) ? low : (gt_mp(y1, high) ? high : y1),
            lt_mp(x1, low) ? low : (gt_mp(x1, high) ? high : x1), lt_mp(y2, low) ? low : (gt_mp(y2, high) ? high : y2),
            lt_mp(x2, low) ? low : (gt_mp(x2, high) ? high : x2)};
    }

    __device__ BoxCorner<T> decode(BoxCorner<T> anchor) const
    {
        return {add_mp(y1, anchor.y1), add_mp(x1, anchor.x1), add_mp(y2, anchor.y2), add_mp(x2, anchor.x2)};
    }

    __device__ float area() const
    {
        T w = sub_mp(x2, x1);
        T h = sub_mp(y2, y1);
        if (lte_mp(h, (T) 0))
        {
            return 0;
        }
        if (lte_mp(w, (T) 0))
        {
            return 0;
        }
        return (float) h * (float) w;
    }

    __device__ operator BoxCenterSize<T>() const
    {
        T w = sub_mp(x2, x1);
        T h = sub_mp(y2, y1);
        return BoxCenterSize<T>{add_mp(y1, mul_mp((T) 0.5, h)), add_mp(x1, mul_mp((T) 0.5, w)), h, w};
    }

    __device__ static BoxCorner<T> intersect(BoxCorner<T> a, BoxCorner<T> b)
    {
        return {gt_mp(a.y1, b.y1) ? a.y1 : b.y1, gt_mp(a.x1, b.x1) ? a.x1 : b.x1, lt_mp(a.y2, b.y2) ? a.y2 : b.y2,
            lt_mp(a.x2, b.x2) ? a.x2 : b.x2};
    }
};

template <typename T>
struct __align__(4 * sizeof(T)) BoxCenterSize
{
    // For NMS/IOU purposes, YXHW coding is identical to XYWH
    T y, x, h, w;

    __device__ void reorder() {}

    __device__ BoxCenterSize<T> clip(T low, T high) const
    {
        return BoxCenterSize<T>(BoxCorner<T>(*this).clip(low, high));
    }

    __device__ BoxCenterSize<T> decode(BoxCenterSize<T> anchor) const
    {
        return {add_mp(mul_mp(y, anchor.h), anchor.y), add_mp(mul_mp(x, anchor.w), anchor.x),
            mul_mp(anchor.h, exp_mp(h)), mul_mp(anchor.w, exp_mp(w))};
    }

    __device__ float area() const
    {
        if (h <= (T) 0)
        {
            return 0;
        }
        if (w <= (T) 0)
        {
            return 0;
        }
        return (float) h * (float) w;
    }

    __device__ operator BoxCorner<T>() const
    {
        T h2 = mul_mp(h, (T) 0.5);
        T w2 = mul_mp(w, (T) 0.5);
        return BoxCorner<T>{sub_mp(y, h2), sub_mp(x, w2), add_mp(y, h2), add_mp(x, w2)};
    }
    __device__ static BoxCenterSize<T> intersect(BoxCenterSize<T> a, BoxCenterSize<T> b)
    {
        return BoxCenterSize<T>(BoxCorner<T>::intersect(BoxCorner<T>(a), BoxCorner<T>(b)));
    }
};

#endif
