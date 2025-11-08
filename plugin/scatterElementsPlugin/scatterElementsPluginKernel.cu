/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "TensorInfo.cuh"
#include "common/dimsHelpers.h"
#include "reducer.cuh"
#include "scatterElementsPluginKernel.h"
#include <thrust/device_vector.h>

namespace nvinfer1
{
namespace plugin
{

#define THREADS 256
#define BLOCKS(N) (N + THREADS - 1) / THREADS

using detail::TensorInfo;
using detail::getTensorInfo;
using nvinfer1::pluginInternal::volume;

template <typename TScalar, ReductionType tReduce>
__global__ void scatterElements_kernel(const TScalar* updatesData, const TensorInfo<int64_t, int32_t> indexInfo,
    TScalar* outData, int32_t nE, int32_t nK, int32_t nN, int32_t nbElements)
{

    int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t b = thread_idx / (nE * nK);
    int32_t k = thread_idx % nK;

    if (thread_idx < nbElements)
    {
        int32_t offset = detail::IndexToOffset<int64_t, int32_t, -1>::get(thread_idx, indexInfo);
        int64_t idx = indexInfo.data[offset];

        Reducer<TScalar, tReduce>::atomic_write(outData + b * nN * nK + idx * nK + k, updatesData[thread_idx]);
    }
}

bool hasBfloat16AtomicAdd()
{
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);
  return deviceProp.major >= 8;
}

inline uint32_t getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4:
    case nvinfer1::DataType::kE8M0:
        PLUGIN_FAIL("Unsupported data type");
    }
    return 0;
}

template <typename TScalar>
void dispatchScatterElementsKernel(void* outDataPtr, void const* dataDataPtr, void const* updatesDataPtr,
    void const* indicesDataPtr, PluginTensorDesc const& outDesc, PluginTensorDesc const& dataDesc,
    PluginTensorDesc const& updatesDesc, PluginTensorDesc const& indicesDesc, int64_t axis, ReductionType reduction,
    cudaStream_t stream)
{
    auto updatesNumEl = volume(updatesDesc.dims);
    auto nB = 1;
    for (auto i = 0; i < axis; i++)
    {
        nB *= updatesDesc.dims.d[i];
    }
    auto nE = updatesDesc.dims.d[axis];
    auto nK = updatesNumEl / (nB * nE);
    auto nN = outDesc.dims.d[axis];

    auto indexInfo = getTensorInfo<int64_t, int32_t>(indicesDataPtr, indicesDesc);

    auto updatesData = (TScalar*) updatesDataPtr;
    auto outData = (TScalar*) outDataPtr;

    AT_DISPATCH_REDUCTION_TYPES(reduction, [&] {
        scatterElements_kernel<TScalar, REDUCE>
            <<<BLOCKS(updatesNumEl), THREADS, 0, stream>>>(updatesData, indexInfo, outData, nE, nK, nN, updatesNumEl);
    });
}

#define DISPATCH_RUN_KERNEL(TYPE)                                                                                      \
    dispatchScatterElementsKernel<TYPE>(outDataPtr, dataDataPtr, updatesDataPtr, indicesDataPtr, outDesc, dataDesc,    \
        updatesDesc, indicesDesc, axis, reduction, stream)

void runScatterElementsKernel(void* outDataPtr, void const* dataDataPtr, void const* updatesDataPtr,
    void const* indicesDataPtr, PluginTensorDesc const& outDesc, PluginTensorDesc const& dataDesc,
    PluginTensorDesc const& updatesDesc, PluginTensorDesc const& indicesDesc, int64_t axis, ReductionType reduction,
    cudaStream_t stream)

{
    auto updatesNumEl = volume(updatesDesc.dims);
    auto outNumEl = volume(outDesc.dims);

    // copy dataDataPtr data to outDataPtr area first
    cudaMemcpyAsync(outDataPtr, dataDataPtr, getElementSize(outDesc.type) * outNumEl, cudaMemcpyDeviceToDevice, stream);

    if (updatesNumEl == 0)
    {
        return;
    }

    switch (outDesc.type)
    {
    case nvinfer1::DataType::kFLOAT: DISPATCH_RUN_KERNEL(float); break;
    case nvinfer1::DataType::kHALF: DISPATCH_RUN_KERNEL(__half); break;
    case nvinfer1::DataType::kINT32: DISPATCH_RUN_KERNEL(int32_t); break;
    case nvinfer1::DataType::kINT64: DISPATCH_RUN_KERNEL(int64_t); break;
    case nvinfer1::DataType::kBF16:  DISPATCH_RUN_KERNEL(__nv_bfloat16); break;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kFP4:
    case nvinfer1::DataType::kE8M0:
      std::ostringstream stream;
      stream << "Unsupported data type:" << (int)outDesc.type << std::endl;
      PLUGIN_FAIL(stream.str().c_str());
      break;
    }
}

} // namespace plugin
} // namespace nvinfer1
