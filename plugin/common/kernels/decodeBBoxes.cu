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
#include "common/kernel.h"
#include "common/kernels/saturate.h"
#include "cuda_fp16.h"
#include <array>
using namespace nvinfer1;
using namespace nvinfer1::plugin;
// overloading exp for half type
inline __device__ __half exp(__half a)
{
#if __CUDA_ARCH__ >= 530
    return hexp(a);
#else
    return exp(float(a));
#endif
}

inline __device__ __half add_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a + b;
#else
    return __float2half(__half2float(a) + __half2float(b));
#endif
}

inline __device__ float add_fb(const float & a, const float & b) {
    return a + b;
}

inline __device__ __half minus_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a - b;
#else
    return __float2half(__half2float(a) - __half2float(b));
#endif
}

inline __device__ float minus_fb(const float & a, const float & b) {
    return a - b;
}

inline __device__ __half mul_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a * b;
#else
    return __float2half(__half2float(a) * __half2float(b));
#endif
}

inline __device__ float mul_fb(const float & a, const float & b) {
    return a * b;
}

inline __device__ __half div_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a / b;
#else
    return __float2half(__half2float(a) / __half2float(b));
#endif
}

inline __device__ float div_fb(const float & a, const float & b) {
    return a / b;
}

template <typename T_BBOX, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void decodeBBoxes_kernel(
        const int nthreads,
        const CodeTypeSSD code_type,
        const bool variance_encoded_in_target,
        const int num_priors,
        const bool share_location,
        const int num_loc_classes,
        const int background_label_id,
        const bool clip_bbox,
        const T_BBOX* loc_data,
        const T_BBOX* prior_data,
        T_BBOX* bbox_data,
        const bool batch_agnostic)
{
    for (int index = blockIdx.x * nthds_per_cta + threadIdx.x;
         index < nthreads;
         index += nthds_per_cta * gridDim.x)
    {
        // Bounding box coordinate index {0, 1, 2, 3}
        // loc_data: (N, R, C, 4)
        // prior_data: (N, 2, R, 4)
        const int i = index % 4;
        // Bounding box class index
        const int c = (index / 4) % num_loc_classes;
        // Prior box id corresponding to the bounding box
        const int d = (index / 4 / num_loc_classes) % num_priors;
        // batch dim
        const int batch = index / (4 * num_loc_classes * num_priors);
        // If bounding box was not shared among all the classes and the bounding box is corresponding to the background class
        if (!share_location && c == background_label_id)
        {
            // Ignore background class if not share_location.
            return;
        }
        // Index to the right anchor box corresponding to the current bounding box
        // do not assume each images' anchor boxes are identical
        // e.g., in FasterRCNN, priors are ROIs from proposal layer and are different
        // for each image.
        const int pi = batch_agnostic ? d * 4 : (batch * 2 * num_priors + d) * 4;
        // Index to the right variances corresponding to the current bounding box
        const int vi = pi + num_priors * 4;
        // Encoding method: CodeTypeSSD::CORNER
        //if (code_type == PriorBoxParameter_CodeType_CORNER){
        if (code_type == CodeTypeSSD::CORNER)
        {
            // Do not want to use variances to adjust the bounding box decoding
            if (variance_encoded_in_target)
            {
                // variance is encoded in target, we simply need to add the offset
                // predictions.
                // prior_data[pi + i]: prior box coordinates corresponding to the current bounding box coordinate
                bbox_data[index] = add_fb(prior_data[pi + i], loc_data[index]);
            }
            else
            {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                // prior_data[vi + i]: variance corresponding to the current bounding box coordinate
                bbox_data[index] = add_fb(prior_data[pi + i], mul_fb(loc_data[index], prior_data[vi + i]));
            }
            //} else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
        }
        // Encoding method: CodeTypeSSD::CENTER_SIZE
        else if (code_type == CodeTypeSSD::CENTER_SIZE)
        {
            // Get prior box coordinates
            const T_BBOX p_xmin = prior_data[pi];
            const T_BBOX p_ymin = prior_data[pi + 1];
            const T_BBOX p_xmax = prior_data[pi + 2];
            const T_BBOX p_ymax = prior_data[pi + 3];
            // Calculate prior box center, height, and width
            const T_BBOX prior_width = minus_fb(p_xmax, p_xmin);
            const T_BBOX prior_height = minus_fb(p_ymax, p_ymin);
            const T_BBOX prior_center_x = div_fb(add_fb(p_xmin, p_xmax), T_BBOX(2));
            const T_BBOX prior_center_y = div_fb(add_fb(p_ymin, p_ymax), T_BBOX(2));

            // Get the current bounding box coordinates
            const T_BBOX xmin = loc_data[index - i];
            const T_BBOX ymin = loc_data[index - i + 1];
            const T_BBOX xmax = loc_data[index - i + 2];
            const T_BBOX ymax = loc_data[index - i + 3];

            // Declare decoded bounding box coordinates
            T_BBOX decode_bbox_center_x, decode_bbox_center_y;
            T_BBOX decode_bbox_width, decode_bbox_height;
            // Do not want to use variances to adjust the bounding box decoding
            if (variance_encoded_in_target)
            {
                // variance is encoded in target, we simply need to retore the offset
                // predictions.
                decode_bbox_center_x = add_fb(mul_fb(xmin, prior_width), prior_center_x);
                decode_bbox_center_y = add_fb(mul_fb(ymin, prior_height), prior_center_y);
                decode_bbox_width = mul_fb(exp(xmax), prior_width);
                decode_bbox_height = mul_fb(exp(ymax), prior_height);
            }
            else
            {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decode_bbox_center_x = add_fb(mul_fb(mul_fb(prior_data[vi], xmin), prior_width), prior_center_x);
                decode_bbox_center_y = add_fb(mul_fb(mul_fb(prior_data[vi + 1], ymin), prior_height), prior_center_y);
                decode_bbox_width = mul_fb(exp(mul_fb(prior_data[vi + 2], xmax)), prior_width);
                decode_bbox_height = mul_fb(exp(mul_fb(prior_data[vi + 3], ymax)), prior_height);
            }

            // Use [x_topleft, y_topleft, x_bottomright, y_bottomright] as coordinates for final decoded bounding box output
            switch (i)
            {
            case 0:
                bbox_data[index] = minus_fb(decode_bbox_center_x, div_fb(decode_bbox_width, T_BBOX(2)));
                break;
            case 1:
                bbox_data[index] = minus_fb(decode_bbox_center_y, div_fb(decode_bbox_height, T_BBOX(2)));
                break;
            case 2:
                bbox_data[index] = add_fb(decode_bbox_center_x, div_fb(decode_bbox_width, T_BBOX(2)));
                break;
            case 3:
                bbox_data[index] = add_fb(decode_bbox_center_y, div_fb(decode_bbox_height, T_BBOX(2)));
                break;
            }
            //} else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
        }
        // Encoding method: CodeTypeSSD::CORNER_SIZE
        else if (code_type == CodeTypeSSD::CORNER_SIZE)
        {
            // Get prior box coordinates
            const T_BBOX p_xmin = prior_data[pi];
            const T_BBOX p_ymin = prior_data[pi + 1];
            const T_BBOX p_xmax = prior_data[pi + 2];
            const T_BBOX p_ymax = prior_data[pi + 3];
            // Get prior box width and height
            const T_BBOX prior_width = minus_fb(p_xmax, p_xmin);
            const T_BBOX prior_height = minus_fb(p_ymax, p_ymin);
            T_BBOX p_size;
            if (i == 0 || i == 2)
            {
                p_size = prior_width;
            }
            else
            {
                p_size = prior_height;
            }
            // Do not want to use variances to adjust the bounding box decoding
            if (variance_encoded_in_target)
            {
                // variance is encoded in target, we simply need to add the offset
                // predictions.
                bbox_data[index] = add_fb(prior_data[pi + i], mul_fb(loc_data[index], p_size));
            }
            else
            {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                bbox_data[index] = add_fb(prior_data[pi + i], mul_fb(mul_fb(loc_data[index], prior_data[vi + i]), p_size));
            }
        }
        // Exactly the same to CodeTypeSSD::CENTER_SIZE with using variance to adjust the bounding box decoding
        else if (code_type == CodeTypeSSD::TF_CENTER)
        {
            const T_BBOX pXmin = prior_data[pi];
            const T_BBOX pYmin = prior_data[pi + 1];
            const T_BBOX pXmax = prior_data[pi + 2];
            const T_BBOX pYmax = prior_data[pi + 3];
            const T_BBOX priorWidth = minus_fb(pXmax, pXmin);
            const T_BBOX priorHeight = minus_fb(pYmax, pYmin);
            const T_BBOX priorCenterX = div_fb(add_fb(pXmin, pXmax), T_BBOX(2));
            const T_BBOX priorCenterY = div_fb(add_fb(pYmin, pYmax), T_BBOX(2));

            const T_BBOX ymin = loc_data[index - i];
            const T_BBOX xmin = loc_data[index - i + 1];
            const T_BBOX ymax = loc_data[index - i + 2];
            const T_BBOX xmax = loc_data[index - i + 3];

            T_BBOX bboxCenterX, bboxCenterY;
            T_BBOX bboxWidth, bboxHeight;

            bboxCenterX = add_fb(mul_fb(mul_fb(prior_data[vi], xmin), priorWidth), priorCenterX);
            bboxCenterY = add_fb(mul_fb(mul_fb(prior_data[vi + 1], ymin), priorHeight), priorCenterY);
            bboxWidth = mul_fb(exp(mul_fb(prior_data[vi + 2], xmax)), priorWidth);
            bboxHeight = mul_fb(exp(mul_fb(prior_data[vi + 3], ymax)), priorHeight);

            switch (i)
            {
            case 0:
                bbox_data[index] = minus_fb(bboxCenterX, div_fb(bboxWidth, T_BBOX(2)));
                break;
            case 1:
                bbox_data[index] = minus_fb(bboxCenterY, div_fb(bboxHeight, T_BBOX(2)));
                break;
            case 2:
                bbox_data[index] = add_fb(bboxCenterX, div_fb(bboxWidth, T_BBOX(2)));
                break;
            case 3:
                bbox_data[index] = add_fb(bboxCenterY, div_fb(bboxHeight, T_BBOX(2)));
                break;
            }
        }
        else
        {
            // Unknown code type.
            assert("Unknown Box decode code type");
        }
        // Clip bounding box or not
        if (clip_bbox)
        {
            bbox_data[index] = saturate(bbox_data[index]);
        }
    }
}

template <typename T_BBOX>
pluginStatus_t decodeBBoxes_gpu(
    cudaStream_t stream,
    const int nthreads,
    const CodeTypeSSD code_type,
    const bool variance_encoded_in_target,
    const int num_priors,
    const bool share_location,
    const int num_loc_classes,
    const int background_label_id,
    const bool clip_bbox,
    const void* loc_data,
    const void* prior_data,
    void* bbox_data,
    const bool batch_agnostic)
{
    const int BS = 512;
    const int GS = (nthreads + BS - 1) / BS;
    decodeBBoxes_kernel<T_BBOX, BS><<<GS, BS, 0, stream>>>(nthreads, code_type, variance_encoded_in_target,
                                                           num_priors, share_location, num_loc_classes,
                                                           background_label_id, clip_bbox,
                                                           (const T_BBOX*) loc_data, (const T_BBOX*) prior_data,
                                                           (T_BBOX*) bbox_data, batch_agnostic);
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// decodeBBoxes LAUNCH CONFIG
typedef pluginStatus_t (*dbbFunc)(cudaStream_t,
                               const int,
                               const CodeTypeSSD,
                               const bool,
                               const int,
                               const bool,
                               const int,
                               const int,
                               const bool,
                               const void*,
                               const void*,
                               void*,
                               const bool);

struct dbbLaunchConfig
{
    DataType t_bbox;
    dbbFunc function;

    dbbLaunchConfig(DataType t_bbox)
        : t_bbox(t_bbox)
        , function(nullptr)
    {
    }
    dbbLaunchConfig(DataType t_bbox, dbbFunc function)
        : t_bbox(t_bbox)
        , function(function)
    {
    }
    bool operator==(const dbbLaunchConfig& other)
    {
        return t_bbox == other.t_bbox;
    }
};

static std::array<dbbLaunchConfig, 2> dbbLCOptions = {
    dbbLaunchConfig(DataType::kFLOAT, decodeBBoxes_gpu<float>),
    dbbLaunchConfig(DataType::kHALF, decodeBBoxes_gpu<__half>)
};

pluginStatus_t decodeBBoxes(
    cudaStream_t stream,
    const int nthreads,
    const CodeTypeSSD code_type,
    const bool variance_encoded_in_target,
    const int num_priors,
    const bool share_location,
    const int num_loc_classes,
    const int background_label_id,
    const bool clip_bbox,
    const DataType DT_BBOX,
    const void* loc_data,
    const void* prior_data,
    void* bbox_data,
    const bool batch_agnostic)
{
    dbbLaunchConfig lc = dbbLaunchConfig(DT_BBOX);
    for (unsigned i = 0; i < dbbLCOptions.size(); ++i)
    {
        if (lc == dbbLCOptions[i])
        {
            DEBUG_PRINTF("decodeBBox kernel %d\n", i);
            return dbbLCOptions[i].function(stream,
                                          nthreads,
                                          code_type,
                                          variance_encoded_in_target,
                                          num_priors,
                                          share_location,
                                          num_loc_classes,
                                          background_label_id,
                                          clip_bbox,
                                          loc_data,
                                          prior_data,
                                          bbox_data,
                                          batch_agnostic);
        }
    }
    return STATUS_BAD_PARAM;
}
