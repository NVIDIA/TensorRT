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

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void cropAndResizeKernel(const int nthreads, const T* image_ptr, const float* boxes_ptr,
                                    int num_boxes, int batch, int image_height, int image_width,
                                    int crop_height, int crop_width, int depth,
                                    float extrapolation_value, float* crops_ptr)
{
    for (int out_idx = threadIdx.x + blockIdx.x * blockDim.x ; out_idx < nthreads;
            out_idx += blockDim.x * gridDim.x)
    {
        int idx =  out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;
        const float y1 = boxes_ptr[b * 4];
        const float x1 = boxes_ptr[b * 4 + 1];
        const float y2 = boxes_ptr[b * 4 + 2];
        const float x2 = boxes_ptr[b * 4 + 3];
        //each image has num_boxes of boxes, so we simply divide to get the box index.
        const int b_in = b / num_boxes;

        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
            : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;
        const float in_y = (crop_height > 1)
                            ? y1 * (image_height - 1) + y * height_scale
                            : 0.5 * (y1 + y2) * (image_height - 1);

        if (in_y < 0 || in_y > image_height - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1)
                           ? x1 * (image_width - 1) + x * width_scale
                           : 0.5 * (x1 + x2) * (image_width - 1);

        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;
        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;
        const float top_left(static_cast<float>(
                                     image_ptr[((b_in * depth + d) * image_height +
                                               top_y_index) *
                                               image_width +
                                               left_x_index]));
        const float top_right(static_cast<float>(
                                     image_ptr[((b_in * depth + d) * image_height +
                                                top_y_index) *
                                                image_width +
                                                right_x_index]));
        const float bottom_left(static_cast<float>(
                                     image_ptr[((b_in * depth + d) * image_height +
                                               bottom_y_index) *
                                               image_width +
                                               left_x_index]));
        const float bottom_right(static_cast<float>(
                                     image_ptr[((b_in * depth + d) * image_height +
                                               bottom_y_index) *
                                               image_width +
                                               right_x_index]));
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    }
}





int cropAndResizeInference(
    cudaStream_t stream,
    int n,
    const void* image,
    const void* rois,
    int batch_size,
    int input_height,
    int input_width,
    int num_boxes,
    int crop_height,
    int crop_width,
    int depth,
    void* output)
{
    int output_volume = batch_size * num_boxes * crop_height * crop_width * depth;
    int block_size = 1024;
    int grid_size = (output_volume + block_size - 1 ) / block_size;
    cropAndResizeKernel<float> <<< grid_size, block_size, 0, stream>>>(output_volume,
            static_cast<const float*>(image),
            static_cast<const float*>(rois),
            num_boxes,
            batch_size,
            input_height,
            input_width,
            crop_height,
            crop_width,
            depth,
            0.0f,
            static_cast<float*>(output));
    return 0;
}
