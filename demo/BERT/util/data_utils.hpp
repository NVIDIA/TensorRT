/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <NvInfer.h>
#include <map>
#include <string>
#include <vector>
namespace bert
{
// Loads a dictionary of weights
// The Weights in the dictionary own the storage behind the Weights::values pointer.
// It is therefore the callers responsibility to free it
void load_weights(const std::string& path, std::map<std::string, nvinfer1::Weights>& weight_dict);

void load_inputs(const std::string& wts_path, int& Bmax, int& S, std::vector<nvinfer1::Weights>& in_ids,
    std::vector<nvinfer1::Weights>& in_masks, std::vector<nvinfer1::Weights>& segment_ids,
    std::vector<nvinfer1::Dims>& dims);

void infer_network_sizes(const std::map<std::string, nvinfer1::Weights>& init_dict, int& hidden_size,
    int& intermediate_size, int& num_hidden_layers);

void alloc_bindings(const nvinfer1::ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, nvinfer1::Weights>& dict, int verbose);

void alloc_bindings(const nvinfer1::ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, std::vector<float>>& dict, int verbose);

void upload(const nvinfer1::ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, nvinfer1::Weights>& dict, cudaStream_t stream);

void download(const nvinfer1::ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    std::map<std::string, std::vector<float>>& dict, cudaStream_t stream);
}
