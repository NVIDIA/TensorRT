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

#ifndef TRT_SCATTER_ELEMENTS_TENSOR_INFO_H
#define TRT_SCATTER_ELEMENTS_TENSOR_INFO_H

#include "common/plugin.h"

namespace nvinfer1
{
namespace plugin
{
namespace detail
{

static constexpr int32_t kMAX_TENSORINFO_DIMS = 25;

// CUDA kernel argument that defines tensor layout
template <typename TScalar, typename TIndex>
struct TensorInfo
{
    TensorInfo();
    TensorInfo(const TScalar* p, int32_t dim, TIndex sz[kMAX_TENSORINFO_DIMS], TIndex st[kMAX_TENSORINFO_DIMS]);

    // Contiguous tensors of more than one dimension are collapsed down
    // to one tensor
    __host__ __device__ inline bool isContiguous() const
    {
        return (dims == 1 && strides[0] == 1);
    }

    const TScalar* data;
    TIndex sizes[kMAX_TENSORINFO_DIMS];
    TIndex strides[kMAX_TENSORINFO_DIMS];
    int32_t dims;
};

// Creates TensorInfo object from PluginTensorDesc and data address
template <typename TScalar, typename TIndex>
TensorInfo<TScalar, TIndex> getTensorInfo(const void* d, PluginTensorDesc const& t)
{
    TIndex sz[kMAX_TENSORINFO_DIMS];
    TIndex st[kMAX_TENSORINFO_DIMS];

    int32_t dims = t.dims.nbDims;
    for (int32_t i = 0; i < dims; ++i)
    {
        sz[i] = t.dims.d[i];
    }
    for (int32_t i = dims; i < kMAX_TENSORINFO_DIMS; ++i)
    {
        sz[i] = static_cast<TIndex>(0);
    }
    // calculate strides
    st[dims - 1] = 1;
    for (int32_t i = dims - 2; i >= 0; --i)
    {
        st[i] = st[i + 1] * sz[i + 1];
    }
    return TensorInfo<TScalar, TIndex>(reinterpret_cast<const TScalar*>(d), dims, sz, st);
}

template <typename TScalar, typename TIndex>
TensorInfo<TScalar, TIndex>::TensorInfo()
{
    data = nullptr;
    dims = 0;
}

template <typename TScalar, typename TIndex>
TensorInfo<TScalar, TIndex>::TensorInfo(
    const TScalar* p, int32_t dim, TIndex sz[kMAX_TENSORINFO_DIMS], TIndex st[kMAX_TENSORINFO_DIMS])
{
    data = p;
    dims = dim;
    for (int32_t i = 0; i < dim; ++i)
    {
        sizes[i] = sz[i];
        strides[i] = st[i];
    }
}

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename TScalar, typename TIndex, int tDims>
struct IndexToOffset
{
    static __host__ __device__ TIndex get(TIndex linearId, const TensorInfo<TScalar, TIndex>& info)
    {

        TIndex offset = 0;

        // Uses static dims
        for (int32_t i = tDims - 1; i > 0; --i)
        {
            TIndex curDimIndex = linearId % info.sizes[i];
            TIndex curDimOffset = curDimIndex * info.strides[i];
            offset += curDimOffset;
            linearId /= info.sizes[i];
        }

        return offset + linearId * info.strides[0];
    }
};

// Uses dynamic (runtime) instead of static (compiletime) dims
template <typename TScalar, typename TIndex>
struct IndexToOffset<TScalar, TIndex, -1>
{
    static inline __host__ __device__ TIndex get(TIndex linearId, const TensorInfo<TScalar, TIndex>& info)
    {

        TIndex offset = 0;

        for (int32_t i = info.dims - 1; i > 0; --i)
        {
            TIndex curDimIndex = linearId % info.sizes[i];
            TIndex curDimOffset = curDimIndex * info.strides[i];
            offset += curDimOffset;
            linearId /= info.sizes[i];
        }

        return offset + linearId * info.strides[0];
    }
};

} // namespace detail
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SCATTER_ELEMENTS_TENSOR_INFO_H
