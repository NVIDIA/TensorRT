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

#include <algorithm>
#include <cuda_fp16.h>

#include "splitPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::SplitPlugin;

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
  std::vector<int> segment_offsets(1, 0);
  for( int i = 0; i < nbOutputs; ++i )
  {
    segment_offsets.push_back(segment_offsets.back() + _output_lengths[i]);
  }
  _d_segment_offsets = segment_offsets;

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
  //_d_output_ptrs.resize(nbOutputs, nullptr);
}

nvinfer1::DimsExprs SplitPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
  nvinfer1::DimsExprs output(inputs[0]);
  output.d[_axis] = exprBuilder.constant(_output_lengths[outputIndex]);
  return output;
}
