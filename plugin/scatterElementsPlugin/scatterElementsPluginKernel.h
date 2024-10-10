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
#ifndef TRT_SCATTER_ELEMENTS_KERNEL_PLUGIN_H
#define TRT_SCATTER_ELEMENTS_KERNEL_PLUGIN_H

#include "common/plugin.h"
#include "scatterElementsCommon.h"

namespace nvinfer1
{
namespace plugin
{

bool hasBfloat16AtomicAdd();

void runScatterElementsKernel(void* outDataPtr, void const* dataDataPtr, void const* updatesDataPtr,
    void const* indicesDataPtr, PluginTensorDesc const& outDesc, PluginTensorDesc const& dataDesc,
    PluginTensorDesc const& updatesDesc, PluginTensorDesc const& indicesDesc, int64_t axis, ReductionType reduction,
    cudaStream_t stream);

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SCATTER_ELEMENTS_KERNEL_PLUGIN_H
