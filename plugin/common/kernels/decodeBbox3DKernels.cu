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

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

__device__ float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void postprocess_kernal(const float *cls_input,
                                        float *box_input,
                                        const float *dir_cls_input,
                                        float *anchors,
                                        float *anchors_bottom_height,
                                        float *bndbox_output,
                                        int *object_counter,
                                        const float min_x_range,
                                        const float max_x_range,
                                        const float min_y_range,
                                        const float max_y_range,
                                        const int feature_x_size,
                                        const int feature_y_size,
                                        const int num_anchors,
                                        const int num_classes,
                                        const int num_box_values,
                                        const float score_thresh,
                                        const float dir_offset,
                                        const float dir_limit_offset,
                                        const int num_dir_bins)
{
  int max_box_num = feature_x_size * feature_y_size * num_anchors;
  int loc_index =blockIdx.x;
  int batch_idx = blockIdx.x / (feature_x_size * feature_y_size);
  int loc_index_in_frame = blockIdx.x % (feature_x_size * feature_y_size);
  int ith_anchor = threadIdx.x;
  if (ith_anchor >= num_anchors)
  {
      return;
  }
  int col = loc_index_in_frame % feature_x_size;
  int row = loc_index_in_frame / feature_x_size;
  float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
  float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);
  int cls_offset = loc_index * num_classes * num_anchors + ith_anchor * num_classes;
  float dev_cls[2] = {-1, 0};
  const float *scores = cls_input + cls_offset;
  float max_score = sigmoid(scores[0]);
  int cls_id = 0;
  for (int i = 1; i < num_classes; i++) {
    float cls_score = sigmoid(scores[i]);
    if (cls_score > max_score) {
      max_score = cls_score;
      cls_id = i;
    }
  }
  dev_cls[0] = static_cast<float>(cls_id);
  dev_cls[1] = max_score;
  if (dev_cls[1] >= score_thresh)
  {
    int box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values;
    int dir_cls_offset = loc_index * num_anchors * 2 + ith_anchor * 2;
    float *anchor_ptr = anchors + ith_anchor * 4;
    float z_offset = anchor_ptr[2] / 2 + anchors_bottom_height[ith_anchor / 2];
    float anchor[7] = {x_offset, y_offset, z_offset, anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};
    float *box_encodings = box_input + box_offset;
    float xa = anchor[0];
    float ya = anchor[1];
    float za = anchor[2];
    float dxa = anchor[3];
    float dya = anchor[4];
    float dza = anchor[5];
    float ra = anchor[6];
    float diagonal = sqrtf(dxa * dxa + dya * dya);
    float be0 = box_encodings[0] * diagonal + xa;
    float be1 = box_encodings[1] * diagonal + ya;
    float be2 = box_encodings[2] * dza + za;
    float be3 = expf(box_encodings[3]) * dxa;
    float be4 = expf(box_encodings[4]) * dya;
    float be5 = expf(box_encodings[5]) * dza;
    float be6 = box_encodings[6] + ra;
    float yaw;
    int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1] ? 0 : 1;
    float period = 2.0f * float(M_PI) / num_dir_bins;
    float val = be6 - dir_offset;
    float dir_rot = val - floor(val / period + dir_limit_offset) * period;
    yaw = dir_rot + dir_offset + period * dir_label;
    int resCount = atomicAdd(object_counter + batch_idx, 1);
    float *data = bndbox_output + (batch_idx * max_box_num + resCount) * 9;
    data[0] = be0;
    data[1] = be1;
    data[2] = be2;
    data[3] = be3;
    data[4] = be4;
    data[5] = be5;
    data[6] = yaw;
    data[7] = dev_cls[0];
    data[8] = dev_cls[1];
  }
}


void  decodeBbox3DLaunch(
  const int batch_size,
  const float *cls_input,
  float *box_input,
  const float *dir_cls_input,
  float *anchors,
  float *anchors_bottom_height,
  float *bndbox_output,
  int *object_counter,
  const float min_x_range,
  const float max_x_range,
  const float min_y_range,
  const float max_y_range,
  const int feature_x_size,
  const int feature_y_size,
  const int num_anchors,
  const int num_classes,
  const int num_box_values,
  const float score_thresh,
  const float dir_offset,
  const float dir_limit_offset,
  const int num_dir_bins,
  cudaStream_t stream)
{
  int bev_size = batch_size * feature_x_size * feature_y_size;
  dim3 threads (num_anchors);
  dim3 blocks (bev_size);
  postprocess_kernal<<<blocks, threads, 0, stream>>>
                (cls_input,
                 box_input,
                 dir_cls_input,
                 anchors,
                 anchors_bottom_height,
                 bndbox_output,
                 object_counter,
                 min_x_range,
                 max_x_range,
                 min_y_range,
                 max_y_range,
                 feature_x_size,
                 feature_y_size,
                 num_anchors,
                 num_classes,
                 num_box_values,
                 score_thresh,
                 dir_offset,
                 dir_limit_offset,
                 num_dir_bins);
  checkCudaErrors(cudaGetLastError());
}
