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

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

const int PILLARS_PER_BLOCK = 64;
const int PILLAR_FEATURE_SIZE = 64;

template <typename Element>
__global__ void scatterBEV_kernel(const Element *pillar_features_data,
          const unsigned int *coords_data, const unsigned int *params_data,
          unsigned int featureX, unsigned int featureY,
          Element *spatial_feature_data)
{
    int pillar_idx = blockIdx.x * PILLARS_PER_BLOCK + threadIdx.x;
    int valid_pillars_inBlock = PILLARS_PER_BLOCK;
    const int num_pillars = params_data[0];
    int valid_blocks = (num_pillars+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK;
    if(blockIdx.x >= valid_blocks) return;
    if(blockIdx.x == (valid_blocks-1)) {
      valid_pillars_inBlock = num_pillars % PILLARS_PER_BLOCK;
    }
    valid_pillars_inBlock = (valid_pillars_inBlock==0) ? PILLARS_PER_BLOCK : valid_pillars_inBlock;
    __shared__ Element pillarSM[PILLARS_PER_BLOCK][PILLAR_FEATURE_SIZE]; //pillar*64
    for (int i = 0; i < valid_pillars_inBlock; i++)
    {
      pillarSM[i][threadIdx.x] = pillar_features_data[ (blockIdx.x * PILLARS_PER_BLOCK +i)*PILLAR_FEATURE_SIZE + threadIdx.x];
    }
    __syncthreads();
    if(pillar_idx >= num_pillars) return;
    int4 coord = ((const int4 *)coords_data)[pillar_idx];
    int x = coord.w;
    int y = coord.z;
    for (int i = 0; i < PILLAR_FEATURE_SIZE; i++)
    {
      spatial_feature_data[i*featureY*featureX + y*featureX + x] = pillarSM[threadIdx.x][i];
    }
}

template <typename Element>
int pillarScatterKernelLaunch(
  int batch_size,
  int max_pillar_num,
  int num_features,
  const Element *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  Element *spatial_feature_data,
  cudaStream_t stream)
{
    dim3 blocks( (featureX*featureY+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK);
    dim3 threads(PILLARS_PER_BLOCK);
    for (int b = 0; b < batch_size; b++) {
      scatterBEV_kernel<Element><<<blocks, threads, 0, stream>>>
        (pillar_features_data + b*max_pillar_num*num_features,
         coords_data + b*max_pillar_num*4,
         params_data + b,
         featureX,
         featureY,
         spatial_feature_data + b*num_features*featureX*featureY
        );
      auto err = cudaGetLastError();
      if (cudaSuccess != err) {
          fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
          return -1;
      }
    }
    return 0;
}

template int pillarScatterKernelLaunch<half>(
  int batch_size,
  int max_pillar_num,
  int num_features,
  const half *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  half *spatial_feature_data,
  cudaStream_t stream);

template int pillarScatterKernelLaunch<float>(
  int batch_size,
  int max_pillar_num,
  int num_features,
  const float *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  float *spatial_feature_data,
  cudaStream_t stream);
