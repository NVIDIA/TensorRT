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

#ifndef TRT_SAMPLE_UTILS_H
#define TRT_SAMPLE_UTILS_H

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

#include "NvInfer.h"

#include "common.h"
#include "logger.h"

#define SMP_RETVAL_IF_FALSE(condition, msg, retval, err)                                                               \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            (err) << (msg) << std::endl;                                                                               \
            return retval;                                                                                             \
        }                                                                                                              \
    }

namespace sample
{

size_t dataTypeSize(nvinfer1::DataType dataType);

template <typename T>
inline T roundUp(T m, T n)
{
    return ((m + n - 1) / n) * n;
}

int64_t volume(nvinfer1::Dims const& d);

//! comps is the number of components in a vector. Ignored if vecDim < 0.
int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps, int32_t batch);

int64_t volume(nvinfer1::Dims const& dims, int32_t vecDim, int32_t comps, int32_t batch);

nvinfer1::Dims toDims(std::vector<int32_t> const& vec);

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void fillBuffer(void* buffer, int64_t volume, T min, T max);

template <typename T, typename std::enable_if<!std::is_integral<T>::value, int32_t>::type = 0>
void fillBuffer(void* buffer, int64_t volume, T min, T max);

template <typename T>
void dumpBuffer(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);

void loadFromFile(std::string const& fileName, char* dst, size_t size);

bool broadcastIOFormats(std::vector<IOFormat> const& formats, size_t nbBindings, bool isInput = true);

int32_t getCudaDriverVersion();

int32_t getCudaRuntimeVersion();

void sparsify(nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights);
void sparsify(Weights const& weights, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

// Walk the weights elements and overwrite (at most) 2 out of 4 elements to 0.
template <typename T>
void sparsify(T const* values, int64_t count, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

template <typename L>
void setSparseWeights(L& l, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

// Sparsify the weights of Constant layers that are fed to MatMul via Shuffle layers.
// Forward analysis on the API graph to determine which weights to sparsify.
void sparsifyMatMulKernelWeights(
    nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights);

template <typename T>
void transpose2DWeights(void* dst, void const* src, int32_t const m, int32_t const n);

} // namespace sample

#endif // TRT_SAMPLE_UTILS_H
