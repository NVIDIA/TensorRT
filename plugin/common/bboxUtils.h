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
#ifndef TRT_BBOX_UTILS_H
#define TRT_BBOX_UTILS_H

#include "common/plugin.h"

template <typename T>
struct Bbox
{
    T xmin, ymin, xmax, ymax;
    Bbox(T xmin, T ymin, T xmax, T ymax)
        : xmin(xmin)
        , ymin(ymin)
        , xmax(xmax)
        , ymax(ymax)
    {
    }
    Bbox() = default;
};

template <typename T>
struct BboxInfo
{
    T conf_score;
    int label;
    int bbox_idx;
    bool kept;
    BboxInfo(T conf_score, int label, int bbox_idx, bool kept)
        : conf_score(conf_score)
        , label(label)
        , bbox_idx(bbox_idx)
        , kept(kept)
    {
    }
    BboxInfo() = default;
};

template <typename TFloat>
bool operator<(const Bbox<TFloat>& lhs, const Bbox<TFloat>& rhs)
{
    return lhs.x1 < rhs.x1;
}

template <typename TFloat>
bool operator==(const Bbox<TFloat>& lhs, const Bbox<TFloat>& rhs)
{
    return lhs.x1 == rhs.x1 && lhs.y1 == rhs.y1 && lhs.x2 == rhs.x2 && lhs.y2 == rhs.y2;
}
// }}}

int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

size_t dataTypeSize(nvinfer1::DataType dtype);

void setUniformOffsets(cudaStream_t stream, int num_segments, int offset, int* d_offsets);

#endif
