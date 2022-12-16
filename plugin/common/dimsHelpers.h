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

#ifndef TRT_PLUGIN_DIMS_HELPERS_H
#define TRT_PLUGIN_DIMS_HELPERS_H

#include "common/plugin.h" // purely for assertions

#include <algorithm> // all of
#include <functional>
#include <numeric>

namespace nvinfer1
{

namespace pluginInternal
{

//! Return number of elements in the given dimensions in the range [start, stop).
//! Does not include padding added for vectorized formats.
//!
//! \param dims dimensions whose partial volume needs to be computed
//! \param start inclusive start axis
//! \param stop exclusive stop axis
//!
//! Expects 0 <= start <= stop <= dims.nbDims.
//! For i in the range [start,stop), dims.d[i] must be non-negative.
//!
inline int64_t volume(Dims const& dims, int32_t start, int32_t stop)
{
    // The signature is int32_t start (and not uint32_t start) because int32_t is used
    // for indexing everywhere
    ASSERT_PARAM(start >= 0);
    ASSERT_PARAM(start <= stop);
    ASSERT_PARAM(stop <= dims.nbDims);
    ASSERT_PARAM(std::all_of(dims.d + start, dims.d + stop, [](int32_t x) { return x >= 0; }));
    return std::accumulate(dims.d + start, dims.d + stop, int64_t{1}, std::multiplies<int64_t>{});
}

//! Shorthand for volume(dims, 0, dims.nbDims).
inline int64_t volume(Dims const& dims)
{
    return volume(dims, 0, dims.nbDims);
}

} // namespace pluginInternal

} // namespace nvinfer1

#endif // TRT_PLUGIN_DIMS_HELPERS_H
