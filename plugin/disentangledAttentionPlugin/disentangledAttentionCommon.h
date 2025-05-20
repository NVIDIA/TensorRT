/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_DISENTANGLED_ATTENTION_COMMON_H
#define TRT_DISENTANGLED_ATTENTION_COMMON_H

#include "NvInferPlugin.h"
#include <cstdint>

namespace nvinfer1
{
namespace plugin
{

// Version 1: regular relative position index
// Version 2: log bucket relative position index
#define kDISENTANGLED_VERSION 2
#if kDISENTANGLED_VERSION == 1
constexpr int32_t kDISENTANGLED_TILESIZE = 32;
constexpr int32_t kDISENTANGLED_BLOCKDIMY = 8;
#elif kDISENTANGLED_VERSION == 2
constexpr int32_t kDISENTANGLED_TILESIZE = 64;
constexpr int32_t kDISENTANGLED_BLOCKDIMY = 4;
#endif

template <typename TDataType, int32_t tTileSize, int32_t tBlockDimY>
void disentangled_kernel_wrapper(TDataType const* data0, TDataType const* data1, TDataType const* data2,
    TDataType* result, dim3 dimData0, dim3 dimData1, dim3 dimData2, dim3 dimResult, TDataType factor, int32_t span,
    dim3 block, dim3 grid, cudaStream_t stream);

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_DISENTANGLED_ATTENTION_COMMON_H
