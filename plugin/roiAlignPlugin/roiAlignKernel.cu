/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <float.h>

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height, const int width, T y, T x)
{
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width)
        return 0;

    if (y <= 0)
        y = 0;
    if (x <= 0)
        x = 0;

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high;
    int x_high;

    if (y_low >= height - 1)
    {
        y_high = y_low = height - 1;
        y = (T) y_low;
    }
    else
    {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1)
    {
        x_high = x_low = width - 1;
        x = (T) x_low;
    }
    else
    {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = input[y_low * width + x_low];
    T v2 = input[y_low * width + x_high];
    T v3 = input[y_high * width + x_low];
    T v4 = input[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename T>
__global__ void roi_align_forward_cuda_kernel(const int nthreads, const T* input, const T* rois, const int roi_cols,
    const int* batch_indices, T* output, const int pooled_height, const int pooled_width, const T spatial_scale,
    const int sampling_ratio,
    const int pool_mode,            // 0 - max pool, 1 - avg pool
    const int coord_transform_mode, // 0 - output_half_pixel, 1 - half_pixel
    const int channels, const int height, const int width)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const T* offset_rois = rois + n * roi_cols;
        int roi_batch_ind = batch_indices[n];

        // Do not using rounding; this implementation detail is critical
        T offset = coord_transform_mode ? (T) 0.5 : (T) 0.0;
        T roi_start_w = offset_rois[0] * spatial_scale - offset;
        T roi_start_h = offset_rois[1] * spatial_scale - offset;
        T roi_end_w = offset_rois[2] * spatial_scale - offset;
        T roi_end_h = offset_rois[3] * spatial_scale - offset;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;
        if (!coord_transform_mode)
        { // for backward-compatibility only
            roi_width = max(roi_width, (T) 1.);
            roi_height = max(roi_height, (T) 1.);
        }

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        const T* offset_input = input + (roi_batch_ind * channels + c) * height * width;

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h
            = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_height / pooled_height));
        int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_width / pooled_width));
        T output_val = 0.;
        bool max_flag = false;
        for (int iy = 0; iy < roi_bin_grid_h; iy++)
        {
            const T y = roi_start_h + ph * bin_size_h
                + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++)
            {
                const T x = roi_start_w + pw * bin_size_w
                    + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
                T val = bilinear_interpolate(offset_input, height, width, y, x);
                if (pool_mode == 1)
                {
                    // We do avg pooling
                    output_val += val;
                }
                else
                {
                    // We do max pooling
                    if (!max_flag)
                    {
                        output_val = val;
                        max_flag = true;
                    }
                    else
                    {
                        output_val = max(output_val, val);
                    }
                }
            }
        }
        if (pool_mode == 1)
        {
            const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
            output_val /= count;
        }
        output[index] = output_val;
    }
}

template <typename scalar_t>
void TRTRoIAlignForwardCUDAKernelLauncher(const scalar_t* input, const scalar_t* rois, const int roi_cols,
    const int* batch_indices, scalar_t* output, int output_size, int channels, int height, int width,
    int aligned_height, int aligned_width, scalar_t spatial_scale, int sampling_ratio, int pool_mode,
    int coord_transform_mode, cudaStream_t stream)
{
    int thread_per_block = 512;
    int block_size = (output_size + thread_per_block - 1) / thread_per_block;
    int max_block_num = 4096;
    block_size = min(block_size, max_block_num);
    roi_align_forward_cuda_kernel<scalar_t><<<block_size, thread_per_block, 0, stream>>>(output_size, input, rois,
        roi_cols, batch_indices, output, aligned_height, aligned_width, static_cast<scalar_t>(spatial_scale),
        sampling_ratio, pool_mode, coord_transform_mode, channels, height, width);
}

void TRTRoIAlignForwardCUDAKernelLauncher_float(const float* input, const float* rois, const int roi_cols,
    const int* batch_indices, float* output, int output_size, int channels, int height, int width, int aligned_height,
    int aligned_width, float spatial_scale, int sampling_ratio, int pool_mode, int coord_transform_mode,
    cudaStream_t stream)
{
    TRTRoIAlignForwardCUDAKernelLauncher<float>(input, rois, roi_cols, batch_indices, output, output_size, channels,
        height, width, aligned_height, aligned_width, spatial_scale, sampling_ratio, pool_mode, coord_transform_mode,
        stream);
}
