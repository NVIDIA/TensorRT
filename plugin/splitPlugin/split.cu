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
 */

#include <algorithm>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#include "split.h"

namespace nvinfer1
{
namespace plugin
{
struct SplitPluginDeviceVectors
{
    thrust::device_vector<int32_t> _d_segment_offsets;
    thrust::device_vector<float*> _d_output_ptrs;
};
} // namespace plugin
} // namespace nvinfer1

using namespace nvinfer1;
using nvinfer1::plugin::SplitPlugin;

template<typename T>
__device__
int upper_bound(T const* vals, int n, T const& key)
{
  int i = 0;
  while( n > 0 )
  {
    int m = n / 2;
    int j = i + m;
    if( !(key < vals[j]) )
    {
      i  = j + 1;
      n -= m + 1;
    }
    else
    {
      n = m;
    }
  }
  return i;
}

template<typename T>
__global__
void split_kernel(int nsegment,
                  int const* __restrict__ segment_offsets,
                  T   const* __restrict__ idata,
                  T*  const* odatas,
                  int nx,
                  int src_ny,
                  int nz)
{
  int x0     = threadIdx.x + blockIdx.x * blockDim.x;
  int src_y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0     = threadIdx.z + blockIdx.z * blockDim.z;
  for( int z=z0; z<nz; z+=blockDim.z*gridDim.z )
  {
    for( int src_y=src_y0; src_y<src_ny; src_y+=blockDim.y*gridDim.y )
    {
      for( int x=x0; x<nx; x+=blockDim.x*gridDim.x )
      {
        int segment = upper_bound(segment_offsets, nsegment, src_y) - 1;
        int dst_y = src_y - segment_offsets[segment];
        int dst_ny = segment_offsets[segment + 1] - segment_offsets[segment];
        odatas[segment][x + nx*(dst_y + dst_ny*z)] =
                  idata[x + nx*(src_y + src_ny*z)];
      }
    }
  }
}

bool SplitPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return (inOut[pos].format == nvinfer1::PluginFormat::kLINEAR);
}

nvinfer1::DataType SplitPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    noexcept
{
    PLUGIN_ASSERT(inputTypes && nbInputs > 0);
    return inputTypes[0];
}

int SplitPlugin::initialize() noexcept 
{
  return 0;
}

void SplitPlugin::terminate() noexcept
{

}

void SplitPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
  deviceVectors.reset(new SplitPluginDeviceVectors);
  std::vector<int> segment_offsets(1, 0);
  for( int i = 0; i < nbOutputs; ++i )
  {
    segment_offsets.push_back(segment_offsets.back() + _output_lengths[i]);
  }
  deviceVectors->_d_segment_offsets = segment_offsets;

  for (int i = 0; i < nbInputs; i++)
  {
      for (int j = 0; j < in[0].desc.dims.nbDims; j++)
      {
          // Do not support dynamic dimensions
          PLUGIN_ASSERT(in[0].desc.dims.d[j] != -1);
      }
  }

  nvinfer1::Dims dims = in[0].desc.dims;
  _nx = 1;
  for( int i = dims.nbDims-1; i > _axis; --i )
  {
    _nx *= dims.d[i];
  }
  _ny = dims.d[_axis];
  _nz = 1;
  for( int i = _axis-1; i >= 0; --i )
  {
    _nz *= dims.d[i];
  }
  deviceVectors->_d_output_ptrs.resize(nbOutputs, nullptr);
}

int SplitPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* /* outputDesc */,
                         const void* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
  PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);

  int const* d_segment_offsets_ptr =
    thrust::raw_pointer_cast(&(deviceVectors->_d_segment_offsets)[0]);
  float  const* idata    = reinterpret_cast<float  const*>(inputs[0]);
  float* const* h_odatas = reinterpret_cast<float* const*>(outputs);
  float** odatas = thrust::raw_pointer_cast(&(deviceVectors->_d_output_ptrs)[0]);
  cudaError_t cuda_status =
    cudaMemcpyAsync(odatas, h_odatas,
                    (deviceVectors->_d_output_ptrs).size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);
  if( cuda_status != cudaSuccess )
  {
    return 1;
  }
  int nz = _nz * inputDesc[0].dims.d[0];
  dim3 block(32, 16);
  dim3 grid(std::min((_nx - 1) / block.x + 1, 65535u),
            std::min((_ny - 1) / block.y + 1, 65535u),
            std::min((_nz - 1) / block.z + 1, 65535u));
  if (inputDesc[0].type==nvinfer1::DataType::kFLOAT)
  {
    split_kernel<<<grid, block, 0, stream>>>
      ((deviceVectors->_d_segment_offsets).size(), d_segment_offsets_ptr, idata, odatas,
       _nx, _ny, nz);
  }
  else
  {
    split_kernel<<<grid, block, 0, stream>>>
      ((deviceVectors->_d_segment_offsets).size(), d_segment_offsets_ptr, (__half const*)idata, (__half**)odatas,
       _nx, _ny, nz);
  }
  return cudaGetLastError() != cudaSuccess;
}

nvinfer1::DimsExprs SplitPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
  nvinfer1::DimsExprs output(inputs[0]);
  output.d[_axis] = exprBuilder.constant(_output_lengths[outputIndex]);
  return output;
}
