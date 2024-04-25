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

#ifndef TRT_SCATTER_ELEMENTS_REDUCER_H
#define TRT_SCATTER_ELEMENTS_REDUCER_H

#include <limits>

#include "atomics.cuh"
#include "scatterElementsPluginKernel.h"

namespace nvinfer1
{
namespace plugin
{

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                                                                       \
    [&] {                                                                                                              \
        switch (reduce)                                                                                                \
        {                                                                                                              \
        case ReductionType::kSUM:                                                                                      \
        {                                                                                                              \
            static constexpr ReductionType REDUCE = ReductionType::kSUM;                                               \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case ReductionType::kMEAN:                                                                                     \
        {                                                                                                              \
            static constexpr ReductionType REDUCE = ReductionType::kMEAN;                                              \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case ReductionType::kMUL:                                                                                      \
        {                                                                                                              \
            static constexpr ReductionType REDUCE = ReductionType::kMUL;                                               \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case ReductionType::kMIN:                                                                                      \
        {                                                                                                              \
            static constexpr ReductionType REDUCE = ReductionType::kMIN;                                               \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        case ReductionType::kMAX:                                                                                      \
        {                                                                                                              \
            static constexpr ReductionType REDUCE = ReductionType::kMAX;                                               \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        }                                                                                                              \
    }()

template <typename TScalar, ReductionType tReduce>
struct Reducer
{
    static inline __host__ __device__ TScalar init()
    {
        if (tReduce == ReductionType::kMUL)
        {
            return TScalar(1);
        }
        else if (tReduce == ReductionType::kMIN)
        {
            return std::numeric_limits<TScalar>::max();
        }
        else if (tReduce == ReductionType::kMAX)
        {
            return std::numeric_limits<TScalar>::lowest();
        }
        else
        {
            return TScalar(0);
        }
    }

    static inline __host__ __device__ void update(TScalar* val, TScalar newVal)
    {
        if (tReduce == ReductionType::kSUM || tReduce == ReductionType::kMEAN)
        {
            *val = *val + newVal;
        }
        else if (tReduce == ReductionType::kMUL)
        {
            *val = *val * newVal;
        }
        else if ((tReduce == ReductionType::kMIN && newVal < *val) || (tReduce == ReductionType::kMAX && newVal > *val))
        {
            *val = newVal;
        }
    }

    static inline __host__ __device__ void update(TScalar* val, TScalar newVal, int64_t* arg, int64_t newArg)
    {
        if (tReduce == ReductionType::kSUM || tReduce == ReductionType::kMEAN)
        {
            *val = *val + newVal;
        }
        else if (tReduce == ReductionType::kMUL)
        {
            *val = *val * newVal;
        }
        else if ((tReduce == ReductionType::kMIN && newVal < *val) || (tReduce == ReductionType::kMAX && newVal > *val))
        {
            *val = newVal;
            *arg = newArg;
        }
    }

    static inline __host__ __device__ void write(
        TScalar* address, TScalar val, int64_t* argAddress, int64_t arg, int count)
    {
        if (tReduce == ReductionType::kSUM || tReduce == ReductionType::kMUL)
        {
            *address = val;
        }
        else if (tReduce == ReductionType::kMEAN)
        {
            *address = val / (TScalar) (count > 0 ? count : 1);
        }
        else if (tReduce == ReductionType::kMIN || tReduce == ReductionType::kMAX)
        {
            if (count > 0)
            {
                *address = val;
                *argAddress = arg;
            }
            else
            {
                *address = (TScalar) 0;
            }
        }
    }

    static inline __device__ void atomic_write(TScalar* address, TScalar val)
    {
        if (tReduce == ReductionType::kSUM || tReduce == ReductionType::kMEAN)
        {
            atomAdd(address, val);
        }
        else if (tReduce == ReductionType::kMUL)
        {
            atomMul(address, val);
        }
        else if (tReduce == ReductionType::kMIN)
        {
            atomMin(address, val);
        }
        else if (tReduce == ReductionType::kMAX)
        {
            atomMax(address, val);
        }
    }
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SCATTER_ELEMENTS_REDUCER_H
