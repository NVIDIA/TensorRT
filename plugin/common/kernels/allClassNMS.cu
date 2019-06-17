/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "kernel.h"
#include "bboxUtils.h"
#include <vector>

template <typename T_BBOX>
__device__ T_BBOX bboxSize(
    const Bbox<T_BBOX>& bbox,
    const bool normalized)
{
    if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
    {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0;
    }
    else
    {
        T_BBOX width = bbox.xmax - bbox.xmin;
        T_BBOX height = bbox.ymax - bbox.ymin;
        if (normalized)
        {
            return width * height;
        }
        else
        {
            // If bbox is not within range [0, 1].
            return (width + 1) * (height + 1);
        }
    }
}

template <typename T_BBOX>
__device__ void intersectBbox(
    const Bbox<T_BBOX>& bbox1,
    const Bbox<T_BBOX>& bbox2,
    Bbox<T_BBOX>* intersect_bbox)
{
    if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin || bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin)
    {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->xmin = T_BBOX(0);
        intersect_bbox->ymin = T_BBOX(0);
        intersect_bbox->xmax = T_BBOX(0);
        intersect_bbox->ymax = T_BBOX(0);
    }
    else
    {
        intersect_bbox->xmin = max(bbox1.xmin, bbox2.xmin);
        intersect_bbox->ymin = max(bbox1.ymin, bbox2.ymin);
        intersect_bbox->xmax = min(bbox1.xmax, bbox2.xmax);
        intersect_bbox->ymax = min(bbox1.ymax, bbox2.ymax);
    }
}

template <typename T_BBOX>
__device__ float jaccardOverlap(
    const Bbox<T_BBOX>& bbox1,
    const Bbox<T_BBOX>& bbox2,
    const bool normalized)
{
    Bbox<T_BBOX> intersect_bbox;
    intersectBbox(bbox1, bbox2, &intersect_bbox);
    float intersect_width, intersect_height;
    if (normalized)
    {
        intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
        intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
    }
    else
    {
        intersect_width = intersect_bbox.xmax - intersect_bbox.xmin + 1;
        intersect_height = intersect_bbox.ymax - intersect_bbox.ymin + 1;
    }
    if (intersect_width > 0 && intersect_height > 0)
    {
        float intersect_size = intersect_width * intersect_height;
        float bbox1_size = bboxSize(bbox1, normalized);
        float bbox2_size = bboxSize(bbox2, normalized);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    }
    else
    {
        return 0.;
    }
}

template <typename T_BBOX>
__device__ void emptyBboxInfo(
    BboxInfo<T_BBOX>* bbox_info)
{
    bbox_info->conf_score = T_BBOX(0);
    bbox_info->label = -2; // -1 is used for all labels when shared_location is ture
    bbox_info->bbox_idx = -1;
    bbox_info->kept = false;
}
/********** new NMS for only score and index array **********/

template <typename T_SCORE, typename T_BBOX, int TSIZE>
__global__ void allClassNMS_kernel(
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k,
    const float nms_threshold,
    const bool share_location,
    const bool isNormalized,
    T_BBOX* bbox_data, // bbox_data should be float to preserve location information
    T_SCORE* beforeNMS_scores,
    int* beforeNMS_index_array,
    T_SCORE* afterNMS_scores,
    int* afterNMS_index_array,
    bool flipXY = false)
{
    //__shared__ bool kept_bboxinfo_flag[CAFFE_CUDA_NUM_THREADS * TSIZE];
    extern __shared__ bool kept_bboxinfo_flag[];

    for (int i = 0; i < num; i++)
    {
        const int offset = i * num_classes * num_preds_per_class + blockIdx.x * num_preds_per_class;
        const int max_idx = offset + top_k; // put top_k bboxes into NMS calculation
        const int bbox_idx_offset = share_location ? (i * num_preds_per_class) : (i * num_classes * num_preds_per_class);

        // local thread data
        int loc_bboxIndex[TSIZE];
        Bbox<T_BBOX> loc_bbox[TSIZE];

// initialize Bbox, Bboxinfo, kept_bboxinfo_flag
#pragma unroll
        for (int t = 0; t < TSIZE; t++)
        {
            const int cur_idx = threadIdx.x + blockDim.x * t;
            const int item_idx = offset + cur_idx;

            if (item_idx < max_idx)
            {
                loc_bboxIndex[t] = beforeNMS_index_array[item_idx];

                if (loc_bboxIndex[t] != -1)
                {
                    const int bbox_data_idx = share_location ? (loc_bboxIndex[t] % num_preds_per_class + bbox_idx_offset) : loc_bboxIndex[t];

                    loc_bbox[t].xmin = flipXY ? bbox_data[bbox_data_idx * 4 + 1] 
                      : bbox_data[bbox_data_idx * 4 + 0];
                    loc_bbox[t].ymin = flipXY ? bbox_data[bbox_data_idx * 4 + 0] 
                      : bbox_data[bbox_data_idx * 4 + 1];
                    loc_bbox[t].xmax = flipXY ? bbox_data[bbox_data_idx * 4 + 3] 
                      : bbox_data[bbox_data_idx * 4 + 2];
                    loc_bbox[t].ymax = flipXY ? bbox_data[bbox_data_idx * 4 + 2]
                      : bbox_data[bbox_data_idx * 4 + 3];
                    kept_bboxinfo_flag[cur_idx] = true;
                }
                else
                {
                    kept_bboxinfo_flag[cur_idx] = false;
                }
            }
            else
            {
                kept_bboxinfo_flag[cur_idx] = false;
            }
        }

        // filter out overlapped boxes with lower scores
        int ref_item_idx = offset;
        int ref_bbox_idx = share_location ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset) : beforeNMS_index_array[ref_item_idx];

        while ((ref_bbox_idx != -1) && ref_item_idx < max_idx)
        {
            Bbox<T_BBOX> ref_bbox;
            ref_bbox.xmin = flipXY ? bbox_data[ref_bbox_idx * 4 + 1] : bbox_data[ref_bbox_idx * 4 + 0];
            ref_bbox.ymin = flipXY ? bbox_data[ref_bbox_idx * 4 + 0] : bbox_data[ref_bbox_idx * 4 + 1];
            ref_bbox.xmax = flipXY ? bbox_data[ref_bbox_idx * 4 + 3] : bbox_data[ref_bbox_idx * 4 + 2];
            ref_bbox.ymax = flipXY ? bbox_data[ref_bbox_idx * 4 + 2] : bbox_data[ref_bbox_idx * 4 + 3];

            for (int t = 0; t < TSIZE; t++)
            {
                const int cur_idx = threadIdx.x + blockDim.x * t;
                const int item_idx = offset + cur_idx;

                if ((kept_bboxinfo_flag[cur_idx]) && (item_idx > ref_item_idx))
                {
                    // TODO: may need to add bool normalized as argument, HERE true means normalized
                    if (jaccardOverlap(ref_bbox, loc_bbox[t], isNormalized) > nms_threshold)
                    {
                        kept_bboxinfo_flag[cur_idx] = false;
                    }
                }
            }
            __syncthreads();

            do
            {
                ref_item_idx++;
            } while (ref_item_idx < max_idx && !kept_bboxinfo_flag[ref_item_idx - offset]);

            ref_bbox_idx = share_location ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset) : beforeNMS_index_array[ref_item_idx];
        }

        // store data
        for (int t = 0; t < TSIZE; t++)
        {
            const int cur_idx = threadIdx.x + blockDim.x * t;
            const int read_item_idx = offset + cur_idx;
            const int write_item_idx = (i * num_classes * top_k + blockIdx.x * top_k) + cur_idx;
            /*
             * If not not keeping the bbox
             * Set the score to 0
             * Set the bounding box index to -1
             */
            if (read_item_idx < max_idx)
            {
                afterNMS_scores[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? beforeNMS_scores[read_item_idx] : 0.0f;
                afterNMS_index_array[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? loc_bboxIndex[t] : -1;
            }
        }
    }
}

template <typename T_SCORE, typename T_BBOX>
pluginStatus_t allClassNMS_gpu(
    cudaStream_t stream,
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k,
    const float nms_threshold,
    const bool share_location,
    const bool isNormalized,
    void* bbox_data,
    void* beforeNMS_scores,
    void* beforeNMS_index_array,
    void* afterNMS_scores,
    void* afterNMS_index_array,
    bool flipXY = false)
{
#define P(tsize) allClassNMS_kernel<T_SCORE, T_BBOX, (tsize)>

    void (*kernel[8])(const int, const int, const int, const int, const float,
                      const bool, const bool, float*, T_SCORE*, int*, T_SCORE*,
                      int*, bool)
        = {
            P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(8),
        };

    const int BS = 512;
    const int GS = num_classes;
    const int t_size = (top_k + BS - 1) / BS;

    kernel[t_size - 1]<<<GS, BS, BS * t_size * sizeof(bool), stream>>>(num, num_classes, num_preds_per_class,
                                                                       top_k, nms_threshold, share_location, isNormalized,
                                                                       (T_BBOX*) bbox_data,
                                                                       (T_SCORE*) beforeNMS_scores, 
                                                                       (int*) beforeNMS_index_array,
                                                                       (T_SCORE*) afterNMS_scores,
                                                                       (int*) afterNMS_index_array,
                                                                       flipXY);

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// allClassNMS LAUNCH CONFIG 
typedef pluginStatus_t (*nmsFunc)(cudaStream_t,
                               const int,
                               const int,
                               const int,
                               const int,
                               const float,
                               const bool,
                               const bool,
                               void*,
                               void*,
                               void*,
                               void*,
                               void*,
                               bool);

struct nmsLaunchConfigSSD
{
    DataType t_score;
    DataType t_bbox;
    nmsFunc function;

    nmsLaunchConfigSSD(DataType t_score, DataType t_bbox)
        : t_score(t_score)
        , t_bbox(t_bbox)
    {
    }
    nmsLaunchConfigSSD(DataType t_score, DataType t_bbox, nmsFunc function)
        : t_score(t_score)
        , t_bbox(t_bbox)
        , function(function)
    {
    }
    bool operator==(const nmsLaunchConfigSSD& other)
    {
        return t_score == other.t_score && t_bbox == other.t_bbox;
    }
};

static std::vector<nmsLaunchConfigSSD> nmsFuncVec;

bool nmsInit()
{
    nmsFuncVec.push_back(nmsLaunchConfigSSD(DataType::kFLOAT, DataType::kFLOAT,
                                            allClassNMS_gpu<float, float>));
    return true;
}

static bool initialized = nmsInit();


pluginStatus_t allClassNMS(cudaStream_t stream,
                        const int num,
                        const int num_classes,
                        const int num_preds_per_class,
                        const int top_k,
                        const float nms_threshold,
                        const bool share_location,
                        const bool isNormalized,
                        const DataType DT_SCORE,
                        const DataType DT_BBOX,
                        void* bbox_data,
                        void* beforeNMS_scores,
                        void* beforeNMS_index_array,
                        void* afterNMS_scores,
                        void* afterNMS_index_array,
                        bool flipXY)
{
    nmsLaunchConfigSSD lc = nmsLaunchConfigSSD(DT_SCORE, DT_BBOX, allClassNMS_gpu<float, float>);
    for (unsigned i = 0; i < nmsFuncVec.size(); ++i)
    {
        if (lc == nmsFuncVec[i])
        {
            DEBUG_PRINTF("all class nms kernel %d\n", i);
            return nmsFuncVec[i].function(stream,
                                          num,
                                          num_classes,
                                          num_preds_per_class,
                                          top_k,
                                          nms_threshold,
                                          share_location,
                                          isNormalized,
                                          bbox_data,
                                          beforeNMS_scores,
                                          beforeNMS_index_array,
                                          afterNMS_scores,
                                          afterNMS_index_array,
                                          flipXY);
        }
    }
    return STATUS_BAD_PARAM;
}
