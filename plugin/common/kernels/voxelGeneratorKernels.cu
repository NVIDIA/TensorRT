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

__global__ void generateVoxels_kernel(
        int max_num_points,
        float *points, unsigned int* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size, int num_point_values,
        int max_points_per_voxel,
        unsigned int *mask, float *voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = point_idx / max_num_points;
  int point_idx_in_frame = point_idx % max_num_points;
  if(point_idx_in_frame >= points_size[batch_idx]) return;
  float px = points[num_point_values * point_idx];
  float py = points[num_point_values * point_idx + 1];
  float pz = points[num_point_values * point_idx + 2];
  float pw = points[num_point_values * point_idx + 3];
  float pt;
  if (num_point_values == 5) {
    pt = points[num_point_values * point_idx + 4];
  }
  if(px<min_x_range||px>=max_x_range
    || py<min_y_range||py>=max_y_range
    || pz<min_z_range||pz>=max_z_range) return;
  int voxel_idx = floorf((px - min_x_range)/pillar_x_size);
  int voxel_idy = floorf((py - min_y_range)/pillar_y_size);
  unsigned int voxel_index = (batch_idx * grid_y_size + voxel_idy) * grid_x_size + voxel_idx;
  unsigned int point_id = atomicAdd(&(mask[voxel_index]), 1);
  if(point_id >= max_points_per_voxel) return;
  float *address = voxels + (voxel_index*max_points_per_voxel + point_id)*num_point_values;
  atomicExch(address+0, px);
  atomicExch(address+1, py);
  atomicExch(address+2, pz);
  atomicExch(address+3, pw);
  if (num_point_values == 5) {
    atomicExch(address+4, pt);
  }
}

__global__ void generateBaseFeatures_kernel(
        int batch_size,
        unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        int max_pillar_num,
        int max_points_per_voxel,
        int num_point_values,
        float *voxel_features,
        unsigned int *voxel_num_points,
        unsigned int *coords)
{
  int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
  int voxel_idx = voxel_id % grid_x_size;
  int voxel_idy = (voxel_id / grid_x_size) % grid_y_size;
  int batch_id = voxel_id / (grid_y_size * grid_x_size);
  if (batch_id >= batch_size) return;
  unsigned int count = mask[voxel_id];
  if( !(count>0) ) return;
  count = count<max_points_per_voxel?count:max_points_per_voxel;
  int current_pillarId = 0;
  current_pillarId = atomicAdd(pillar_num + batch_id, 1);
  voxel_num_points[batch_id * grid_y_size * grid_x_size + current_pillarId] = count;
  int4 coord = {0, 0, voxel_idy, voxel_idx};
  ((int4*)coords)[batch_id * max_pillar_num + current_pillarId] = coord;
  for (int i=0; i<count; i++){
    int inIndex = voxel_id*max_points_per_voxel + i;
    int outIndex = (batch_id * grid_x_size * grid_y_size + current_pillarId)*max_points_per_voxel + i;
    if (num_point_values == 4) {
      ((float4*)voxel_features)[outIndex] = ((float4*)voxels)[inIndex];
    }
    else if (num_point_values == 5) {
      for(int k=0; k<5;k++)
          voxel_features[5 * outIndex + k] = voxels[5 * inIndex + k];
    }
  }
}

void generateVoxels_launch(
        int batch_size, int max_num_points,
        float *points, unsigned int* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size, int num_point_values,
        int max_points_per_voxel,
        unsigned int *mask, float *voxels,
        cudaStream_t stream)
{
  int threadNum = 256;
  dim3 blocks((batch_size * max_num_points + threadNum - 1) / threadNum);
  dim3 threads(threadNum);
  generateVoxels_kernel<<<blocks, threads, 0, stream>>>
      (max_num_points,
        points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        pillar_x_size, pillar_y_size, pillar_z_size,
        grid_y_size, grid_x_size, num_point_values,
        max_points_per_voxel,
        mask, voxels);
}

void generateBaseFeatures_launch(
        int batch_size,
        unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        int max_pillar_num,
        int max_points_per_voxel,
        int num_point_values,
        float *voxel_features,
        unsigned int *voxel_num_points,
        unsigned int *coords,
        cudaStream_t stream)
{
  int blockSize = 1024;
  dim3 threads(blockSize);
  dim3 blocks((batch_size * grid_y_size * grid_x_size + blockSize - 1) / blockSize);
  generateBaseFeatures_kernel<<<blocks, threads, 0, stream>>>
      (
        batch_size,
        mask, voxels, grid_y_size, grid_x_size,
        pillar_num,
        max_pillar_num,
        max_points_per_voxel,
        num_point_values,
        voxel_features,
        voxel_num_points,
        coords
      );
}

__global__ void generateFeatures_kernel(
    int batch_size,
    int dense_pillar_num,
    float* voxel_features,
    unsigned int* voxel_num_points,
    unsigned int* coords, unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    unsigned int voxel_features_size, unsigned int max_points,
    unsigned int max_voxels,
    float* features)
{
    int warp_size = max_points;
    int pillar_idx = blockIdx.x * 4 + threadIdx.x/warp_size;
    int point_idx = threadIdx.x % warp_size;
    // In case the actual number of points is less than warp_size
    // E.g., warp_size=32, max_points=20
    if (point_idx >= max_points) return;
    int batch_idx = pillar_idx / max_voxels;
    if (batch_idx >= batch_size) return;
    int pillar_idx_in_frame = pillar_idx % max_voxels;
    int dense_pillar_idx = pillar_idx_in_frame + dense_pillar_num * batch_idx;
    int pillar_idx_inBlock = threadIdx.x/warp_size;
    // Limit number of voxels to max_voxels
    unsigned int num_pillars = params[batch_idx] > max_voxels ? max_voxels : params[batch_idx];
    // Update max_voxel to actual number
    if (pillar_idx_in_frame == 0 && point_idx == 0) {
      params[batch_idx] = num_pillars;
    }
    if (pillar_idx_in_frame >= num_pillars) return;

    //load src
    __shared__ float pillarSM[4][64][5]; // up to 64 points per pillar
    __shared__ float4 pillarSumSM[4]; //4*4
    __shared__ int4 cordsSM[4]; //4*4
    __shared__ int pointsNumSM[4]; //4
    __shared__ float pillarOutSM[4][64][11]; // up to 11 features per point

    if (point_idx == 0) {
      pointsNumSM[pillar_idx_inBlock] = voxel_num_points[dense_pillar_idx];
      cordsSM[pillar_idx_inBlock] = ((int4*)coords)[dense_pillar_idx];
      pillarSumSM[pillar_idx_inBlock] = {0,0,0,0};
    }
    for(int k=0; k<5; k++) {
      pillarSM[pillar_idx_inBlock][point_idx][k] = voxel_features[5 * (dense_pillar_idx*max_points + point_idx) + k];
    }
    __syncthreads();
    //calculate sm
    if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
      atomicAdd(&(pillarSumSM[pillar_idx_inBlock].x),  pillarSM[pillar_idx_inBlock][point_idx][0]);
      atomicAdd(&(pillarSumSM[pillar_idx_inBlock].y),  pillarSM[pillar_idx_inBlock][point_idx][1]);
      atomicAdd(&(pillarSumSM[pillar_idx_inBlock].z),  pillarSM[pillar_idx_inBlock][point_idx][2]);
    }
    __syncthreads();
    //feature-mean
    float4 mean;
    float validPoints = pointsNumSM[pillar_idx_inBlock];
    mean.x = pillarSumSM[pillar_idx_inBlock].x / validPoints;
    mean.y = pillarSumSM[pillar_idx_inBlock].y / validPoints;
    mean.z = pillarSumSM[pillar_idx_inBlock].z / validPoints;
    mean.x  = pillarSM[pillar_idx_inBlock][point_idx][0] - mean.x;
    mean.y  = pillarSM[pillar_idx_inBlock][point_idx][1] - mean.y;
    mean.z  = pillarSM[pillar_idx_inBlock][point_idx][2] - mean.z;
    //calculate offset
    float x_offset = voxel_x / 2.0f + cordsSM[pillar_idx_inBlock].w * voxel_x + range_min_x;
    float y_offset = voxel_y / 2.0f + cordsSM[pillar_idx_inBlock].z * voxel_y + range_min_y;
    float z_offset = voxel_z / 2.0f + cordsSM[pillar_idx_inBlock].y * voxel_z + range_min_z;
    //feature-offset
    float4 center;
    center.x  = pillarSM[pillar_idx_inBlock][point_idx][0] - x_offset;
    center.y  = pillarSM[pillar_idx_inBlock][point_idx][1] - y_offset;
    center.z  = pillarSM[pillar_idx_inBlock][point_idx][2] - z_offset;
    //store output
    if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
      for(int k=0; k<5; k++)
        pillarOutSM[pillar_idx_inBlock][point_idx][k] = pillarSM[pillar_idx_inBlock][point_idx][k];
      pillarOutSM[pillar_idx_inBlock][point_idx][5] = mean.x;
      pillarOutSM[pillar_idx_inBlock][point_idx][5 + 1] = mean.y;
      pillarOutSM[pillar_idx_inBlock][point_idx][5 + 2] = mean.z;
      pillarOutSM[pillar_idx_inBlock][point_idx][5 + 3] = center.x;
      pillarOutSM[pillar_idx_inBlock][point_idx][5 + 4] = center.y;
      if (5 + 5 < voxel_features_size)
        pillarOutSM[pillar_idx_inBlock][point_idx][warp_size + 5] = center.z;
    } else {
      for (int k = 0; k < voxel_features_size; k++)
        pillarOutSM[pillar_idx_inBlock][point_idx][k] = 0;
    }
    __syncthreads();
    for(int i = 0; i < voxel_features_size; i ++) {
      int outputSMId = pillar_idx_inBlock*64*11 + point_idx * 11 + i;
      int outputId = pillar_idx*max_points*voxel_features_size + point_idx * voxel_features_size + i;
      features[outputId] = ((float*)pillarOutSM)[outputSMId] ;
    }

}

__global__ void generateFeatures_kernel_4x(
  int batch_size,
  int dense_pillar_num,
  float* voxel_features,
  unsigned int* voxel_num_points, unsigned int* coords,
  unsigned int *params,
  float voxel_x, float voxel_y, float voxel_z,
  float range_min_x, float range_min_y, float range_min_z,
  unsigned int voxel_features_size, unsigned int max_points,
  unsigned int max_voxels,
  float* features)
{
  int warp_size = max_points;
  int pillar_idx = blockIdx.x * 4 + threadIdx.x / warp_size;
  int point_idx = threadIdx.x % warp_size;
  // In case the actual number of points is less than warp_size
  // E.g., warp_size=32, max_points=20
  if (point_idx >= max_points) return;
  int batch_idx = pillar_idx / max_voxels;
  if (batch_idx >= batch_size) return;
  int pillar_idx_in_frame = pillar_idx % max_voxels;
  int dense_pillar_idx = pillar_idx_in_frame + dense_pillar_num * batch_idx;
  int pillar_idx_inBlock = threadIdx.x / warp_size;
  // Limit number of voxels to max_voxels
  unsigned int num_pillars = params[batch_idx] > max_voxels ? max_voxels : params[batch_idx];
  // Update max_voxel to actual number
  if (pillar_idx_in_frame == 0 && point_idx == 0) {
    params[batch_idx] = num_pillars;
  }
  if (pillar_idx_in_frame >= num_pillars) return;
  //load src
  __shared__ float4 pillarSM[4][64]; // up to 64 points per pillar
  __shared__ float4 pillarSumSM[4]; //4*4
  __shared__ int4 cordsSM[4]; //4*4
  __shared__ int pointsNumSM[4]; //4
  __shared__ float pillarOutSM[4][64][11]; // up to 11 output features per point

  if (point_idx == 0) {
    pointsNumSM[pillar_idx_inBlock] = voxel_num_points[dense_pillar_idx];
    cordsSM[pillar_idx_inBlock] = ((int4*)coords)[pillar_idx];
    pillarSumSM[pillar_idx_inBlock] = {0,0,0,0};
  }
  pillarSM[pillar_idx_inBlock][point_idx] = ((float4*)voxel_features)[dense_pillar_idx*max_points + point_idx];
  __syncthreads();
  //calculate sm
  if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
    atomicAdd(&(pillarSumSM[pillar_idx_inBlock].x),  pillarSM[pillar_idx_inBlock][point_idx].x);
    atomicAdd(&(pillarSumSM[pillar_idx_inBlock].y),  pillarSM[pillar_idx_inBlock][point_idx].y);
    atomicAdd(&(pillarSumSM[pillar_idx_inBlock].z),  pillarSM[pillar_idx_inBlock][point_idx].z);
  }
  __syncthreads();
  //feature-mean
  float4 mean;
  float validPoints = pointsNumSM[pillar_idx_inBlock];
  mean.x = pillarSumSM[pillar_idx_inBlock].x / validPoints;
  mean.y = pillarSumSM[pillar_idx_inBlock].y / validPoints;
  mean.z = pillarSumSM[pillar_idx_inBlock].z / validPoints;
  mean.x  = pillarSM[pillar_idx_inBlock][point_idx].x - mean.x;
  mean.y  = pillarSM[pillar_idx_inBlock][point_idx].y - mean.y;
  mean.z  = pillarSM[pillar_idx_inBlock][point_idx].z - mean.z;
  //calculate offset
  float x_offset = voxel_x / 2.0f + cordsSM[pillar_idx_inBlock].w * voxel_x + range_min_x;
  float y_offset = voxel_y / 2.0f + cordsSM[pillar_idx_inBlock].z * voxel_y + range_min_y;
  float z_offset = voxel_z / 2.0f + cordsSM[pillar_idx_inBlock].y * voxel_z + range_min_z;
  //feature-offset
  float4 center;
  center.x  = pillarSM[pillar_idx_inBlock][point_idx].x - x_offset;
  center.y  = pillarSM[pillar_idx_inBlock][point_idx].y - y_offset;
  center.z  = pillarSM[pillar_idx_inBlock][point_idx].z - z_offset;
  //store output
  if (point_idx < pointsNumSM[pillar_idx_inBlock]) {
    pillarOutSM[pillar_idx_inBlock][point_idx][0] = pillarSM[pillar_idx_inBlock][point_idx].x;
    pillarOutSM[pillar_idx_inBlock][point_idx][1] = pillarSM[pillar_idx_inBlock][point_idx].y;
    pillarOutSM[pillar_idx_inBlock][point_idx][2] = pillarSM[pillar_idx_inBlock][point_idx].z;
    pillarOutSM[pillar_idx_inBlock][point_idx][3] = pillarSM[pillar_idx_inBlock][point_idx].w;
    pillarOutSM[pillar_idx_inBlock][point_idx][4] = mean.x;
    pillarOutSM[pillar_idx_inBlock][point_idx][5] = mean.y;
    pillarOutSM[pillar_idx_inBlock][point_idx][6] = mean.z;
    pillarOutSM[pillar_idx_inBlock][point_idx][7] = center.x;
    pillarOutSM[pillar_idx_inBlock][point_idx][8] = center.y;
    pillarOutSM[pillar_idx_inBlock][point_idx][9] = center.z;

  } else {
    pillarOutSM[pillar_idx_inBlock][point_idx][0] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][1] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][2] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][3] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][4] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][5] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][6] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][7] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][8] = 0;
    pillarOutSM[pillar_idx_inBlock][point_idx][9] = 0;
  }
  __syncthreads();
  for(int i = 0; i < voxel_features_size; i ++) {
    int outputSMId = pillar_idx_inBlock*64*11 + point_idx * 11 + i;
    int outputId = pillar_idx*max_points*voxel_features_size + point_idx * voxel_features_size + i;
    features[outputId] = ((float*)pillarOutSM)[outputSMId] ;
  }

}

int generateFeatures_launch(
    int batch_size,
    int dense_pillar_num,
    float* voxel_features,
    unsigned int* voxel_num_points,
    unsigned int* coords,
    unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    unsigned int voxel_features_size, unsigned int max_points,
    unsigned int max_voxels, unsigned int num_point_values,
    float* features,
    cudaStream_t stream)
{
    unsigned int warp_size = max_points;
    dim3 blocks((batch_size * max_voxels + 3) / 4);
    dim3 threads(4*warp_size);
    if (num_point_values == 4) {
      generateFeatures_kernel_4x<<<blocks, threads, 0, stream>>>
      (batch_size,
      dense_pillar_num,
      voxel_features,
      voxel_num_points,
      coords,
      params,
      voxel_x, voxel_y, voxel_z,
      range_min_x, range_min_y, range_min_z,
      voxel_features_size, max_points,
      max_voxels, 
      features);
    }
    else {
      generateFeatures_kernel<<<blocks, threads, 0, stream>>>
      (batch_size,
      dense_pillar_num,
      voxel_features,
      voxel_num_points,
      coords,
      params,
      voxel_x, voxel_y, voxel_z,
      range_min_x, range_min_y, range_min_z,
      voxel_features_size, max_points,
      max_voxels, 
      features);
    }
    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return err;
}
