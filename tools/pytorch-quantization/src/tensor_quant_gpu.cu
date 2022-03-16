/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <ATen/ATen.h>
#include <torch/extension.h>

#include <math.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

__host__ __device__ float bits_to_bound(int num_bits, int is_unsigned) {
  float bound = (1 << (num_bits - 1 + int(is_unsigned))) - 1;
  return bound;
}

template <typename T> __device__ T fake_tensor_quant_device(T input, T amax, int bound);

template <>
__device__ float fake_tensor_quant_device(float input, float amax, int bound) {
  float scale = bound / amax;
  float output = round(input * scale);
  output = output > bound ? bound : output;
  output = output < -bound ? -bound : output;

  return output / scale;
}

template <>
__device__ at::Half fake_tensor_quant_device(at::Half input, at::Half amax, int bound) {
  float output = fake_tensor_quant_device(__half2float(input), __half2float(amax), bound);

  return __float2half(output);
}

// Sepcialize double only to pass Aten dispatch macros
template <>
__device__ double fake_tensor_quant_device(double input, double amax, int bound) {
  float output = fake_tensor_quant_device(input, amax, bound);

  return output;
}

template <typename T>
__global__ void fake_tensor_quant_kernel(
    const T *inputs, size_t n, T *outputs,
    const T *amax, int num_bits=8, bool is_unsigned=false) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    float bound = bits_to_bound(num_bits, is_unsigned);
    outputs[tid] = fake_tensor_quant_device(inputs[tid], amax[0], bound);
  }
}

void fake_tensor_quant_cuda_inplace(at::Tensor inputs, at::Tensor amax, int num_bits=8, bool is_unsigned=false) {
  size_t numel = inputs.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs.type().scalarType(), "fake_tensor_quant_cuda_inplace", [&] {
    fake_tensor_quant_kernel<<<numel/BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        inputs.data_ptr<scalar_t>(), numel, inputs.data_ptr<scalar_t>(),
        amax.data_ptr<scalar_t>(), num_bits, is_unsigned);
  });
}

at::Tensor fake_tensor_quant_cuda(at::Tensor inputs, at::Tensor amax, int num_bits=8, bool is_unsigned=false) {
  size_t numel = inputs.numel();
  auto outputs = torch::zeros_like(inputs);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs.type().scalarType(), "fake_tensor_quant_cuda", [&] {
    fake_tensor_quant_kernel<<<numel/BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>(),
        amax.data_ptr<scalar_t>(), num_bits, is_unsigned);
  });

  return outputs;
}

__global__ void fake_tensor_quant_with_axis_cuda_kernel(
    const float *inputs, size_t n, float *outputs,
    const float *amax, int axis_size, int outer_size, int num_bits=8, bool is_unsigned=false) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float bound = bits_to_bound(num_bits, is_unsigned);

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    int axis_idx = (idx / outer_size) % axis_size;

    outputs[idx] = fake_tensor_quant_device(inputs[idx], amax[axis_idx], bound);
  }
}

at::Tensor fake_tensor_quant_with_axis_cuda(
    at::Tensor inputs, at::Tensor amax, int axis, int num_bits=8, bool is_unsigned=false) {
  auto outputs = torch::empty_like(inputs);
  size_t numel = inputs.numel();
  int axis_size = inputs.size(axis);

  int outer_size = 1;
  for (int i = axis + 1; i < inputs.dim(); ++i) {
    outer_size *= inputs.size(i);
  }

  fake_tensor_quant_with_axis_cuda_kernel<<<numel / (BLOCK_SIZE*4) + 1, BLOCK_SIZE>>>(
      inputs.data_ptr<float>(), numel, outputs.data_ptr<float>(), amax.data_ptr<float>(), axis_size, outer_size,
      num_bits, is_unsigned);
  return outputs;
}

