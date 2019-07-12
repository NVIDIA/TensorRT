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

#include "data_utils.hpp"
#include <attention_keys.hpp>
#include <cassert>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <numeric>

namespace bert
{

using namespace nvinfer1;
using std::cout;
using std::endl;

void getDims(Dims& d, std::ifstream& input, const std::string& name)
{
    const int n_dims = d.nbDims;
    printf("%s: D=%d, (", name.c_str(), n_dims);
    for (int it = 0; it < n_dims; it++)
    {
        input >> d.d[it];
        printf("%d,", d.d[it]);
    }
    printf(")\n");
    assert(input.peek() == ' ');
    input.get();
}

void transposeMatrix(float* data, const Dims& d)
{
    // data represents a d0xd1 RowMajor matrix, i.e. d1xd0 ColMajor
    // which we transpose into d0xd1 ColMajor
    assert(d.nbDims == 2);
    int len = d.d[0] * d.d[1];
    std::vector<float> tmp(data, data + len);

    int idx = 0; // result matrix linear idx
    for (int c = 0; c < d.d[1]; c++)
    { // columns of the result matrix
        for (int r = 0; r < d.d[0]; r++)
        { // rows of the result matrix
            int src_idx = r * d.d[1] + c;
            data[idx++] = tmp[src_idx];
        }
    }
}

int type2bytes(DataType dt)
{
    if (dt == DataType::kINT8)
        return 1;
    if (dt == DataType::kHALF)
        return 2;
    return 4;
}

template <typename>
struct T2DT;
template <>
struct T2DT<float>
{
    const static DataType value;
};
template <>
struct T2DT<int>
{
    const static DataType value;
};
const DataType T2DT<float>::value = DataType::kFLOAT;
const DataType T2DT<int>::value = DataType::kINT32;

template <typename T>
void load_row(std::string& name, T*& data, int& param_size, Dims& d, std::ifstream& input)
{
    int32_t type, n_dims, dim;
    input >> name >> std::dec >> type >> n_dims;
    assert(n_dims > 0 && n_dims <= Dims::MAX_DIMS);
    d.nbDims = n_dims;
    getDims(d, input, name);

    DataType dtype = static_cast<DataType>(type);
    assert(dtype == T2DT<T>::value);

    param_size = std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
    data = new T[param_size]; // should be freed by caller
    char* bytes = reinterpret_cast<char*>(data);
    input.read(bytes, param_size * sizeof(T));

    assert(input.peek() == '\n');
    input.get();
}

void load_weights(const std::string& wts_path, std::map<std::string, nvinfer1::Weights>& weight_dict)
{

    std::ifstream input(wts_path, std::ios_base::binary);
    int32_t count;
    input >> count;
    cout << "Number of parameters: " << count << endl;

    for (int it = 0; it < count; it++)
    {
        int param_size = 0;
        std::string name;
        float* data = nullptr;
        Dims d;
        load_row(name, data, param_size, d, input);
        assert(data);
        assert(param_size);

        // Need to be careful here. This is highly dependent on the TF implementation.
        // The TF squad output does not use a fully connected layer but a matmul op with a transpose, which names the
        // output as squad_output_weights
        if (name.find("kernel") != std::string::npos)
        {
            printf("Transposing\n");
            transposeMatrix(data, d);
        }
        Weights tmp;
        tmp.type = DataType::kFLOAT;
        tmp.values = data;
        tmp.count = param_size;

        weight_dict[name] = tmp;
    }

    input.close();

    for (auto& kv : weight_dict)
    {

        int pos = kv.first.find(BQ); // starting pos of BQ
        if (pos != std::string::npos)
        {
            int hidden_size = kv.second.count;
            std::string prefix = kv.first.substr(0, pos);

            const Weights& Bq_ = kv.second;
            const Weights Bk_ = weight_dict.at(prefix + BK);
            const Weights Bv_ = weight_dict.at(prefix + BV);
            const Weights Wq_ = weight_dict.at(prefix + WQ);
            const Weights Wk_ = weight_dict.at(prefix + WK);
            const Weights Wv_ = weight_dict.at(prefix + WV);

            int mat_size = hidden_size * hidden_size;
            int wcount = 3 * mat_size;
            float* Wall_ptr = (float*) malloc(wcount * sizeof(float));

            int bcount = 3 * hidden_size;
            float* Ball_ptr = (float*) malloc(bcount * sizeof(float));

            std::copy((float*) Wq_.values, ((float*) Wq_.values) + mat_size, Wall_ptr);
            std::copy((float*) Wk_.values, ((float*) Wk_.values) + mat_size, Wall_ptr + mat_size);
            std::copy((float*) Wv_.values, ((float*) Wv_.values) + mat_size, Wall_ptr + 2 * mat_size);

            std::copy((float*) Bq_.values, ((float*) Bq_.values) + hidden_size, Ball_ptr);
            std::copy((float*) Bk_.values, ((float*) Bk_.values) + hidden_size, Ball_ptr + hidden_size);
            std::copy((float*) Bv_.values, ((float*) Bv_.values) + hidden_size, Ball_ptr + 2 * hidden_size);

            weight_dict[prefix + WQKV] = {DataType::kFLOAT, Wall_ptr, wcount};
            weight_dict[prefix + BQKV] = {DataType::kFLOAT, Ball_ptr, bcount};
        }
    }
}

void load_inputs(const std::string& wts_path, int& Bmax, int& S, std::vector<nvinfer1::Weights>& in_ids,
    std::vector<nvinfer1::Weights>& in_masks, std::vector<nvinfer1::Weights>& segment_ids,
    std::vector<nvinfer1::Dims>& dims)
{
    std::ifstream input(wts_path, std::ios_base::binary);
    int32_t count;
    input >> count;
    cout << "Number of buffers: " << count << endl;
    assert(count % 3 == 0);
    S = 0;
    Bmax = 0;

    for (int it = 0; it < count; it++)
    {
        int param_size = 0;
        std::string name;
        int* data = nullptr;
        Dims d;
        load_row(name, data, param_size, d, input);
        assert(data);
        assert(param_size);
        dims.push_back(d);

        Bmax = std::max(Bmax, d.d[0]); // find the largest batch size in the dataset

        if (S == 0)
            S = d.d[1];
        else
            assert(S == d.d[1]); // all inputs should have the same sequence length

        Weights tmp;
        tmp.type = DataType::kINT32;
        tmp.values = data;
        tmp.count = param_size;

        if (name.find("input_id") != std::string::npos)
        {
            in_ids.push_back(tmp);
            continue;
        }

        if (name.find("input_mask") != std::string::npos)
        {
            in_masks.push_back(tmp);
            continue;
        }
        segment_ids.push_back(tmp);
    }
    input.close();
    assert(in_ids.size() == count / 3);
    assert(in_masks.size() == in_ids.size());
    assert(segment_ids.size() == in_ids.size());
    assert(dims.size() == count);
    assert(S);
    assert(Bmax);
}

void infer_network_sizes(const std::map<std::string, nvinfer1::Weights>& init_dict, int& hidden_size,
    int& intermediate_size, int& num_hidden_layers)
{
    for (auto& kv : init_dict)
    {
        if (kv.first.find("beta") != std::string::npos)
        {
            hidden_size = kv.second.count;
            break;
        }
    }
    for (auto& kv : init_dict)
    {
        if (kv.first.find("intermediate_dense_bias") != std::string::npos)
        {
            intermediate_size = kv.second.count;
            break;
        }
    }
    num_hidden_layers = 0;
    for (auto& kv : init_dict)
    {
        if (kv.first[0] == 'l')
        {
            std::string tok = kv.first.substr(1, kv.first.find("_") - 1);
            int layer = std::stoi(tok);
            num_hidden_layers = std::max(num_hidden_layers, layer + 1);
            std::cout << kv.first << " " << tok << " " << layer << std::endl;
        }
    }
}

void alloc_bindings(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, nvinfer1::Weights>& dict, int verbose)
{

    Weights W;
    std::string name;

    for (auto& kv : dict)
    {
        std::tie(name, W) = kv;
        const int idx = engine.getBindingIndex(name.c_str());
        if (verbose)
            printf(" idx %d name %s\n", idx, name.c_str());

        assert(idx >= 0);
        int outlen = W.count * type2bytes(W.type);
        (cudaMalloc(&buffers[idx], batchSize * outlen));
        if (verbose)
            printf(" idx %d allocated %d bytes\n", idx, batchSize * outlen);
    }
}

void alloc_bindings(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, std::vector<float>>& dict, int verbose)
{

    for (auto& kv : dict)
    {
        const int idx = engine.getBindingIndex(kv.first.c_str());
        if (verbose)
            printf(" idx %d name %s\n", idx, kv.first.c_str());

        assert(idx >= 0);
        int outlen = sizeof(float) * kv.second.size();
        (cudaMalloc(&buffers[idx], batchSize * outlen));
        if (verbose)
            printf(" idx %d allocated %d bytes\n", idx, batchSize * outlen);
    }
}

void upload(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, Weights>& dict, cudaStream_t stream)
{

    Weights W;
    std::string name;

    for (auto kv : dict)
    {
        std::tie(name, W) = kv;
        const int idx = engine.getBindingIndex(name.c_str());
        int len = W.count * type2bytes(W.type);
        (cudaMemcpyAsync(buffers[idx], W.values, batchSize * len, cudaMemcpyHostToDevice, stream));
    }
}

void download(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    std::map<std::string, std::vector<float>>& dict, cudaStream_t stream)
{
    for (auto& kv : dict)
    {
        const int idx = engine.getBindingIndex(kv.first.c_str());
        int len = kv.second.size() * sizeof(float);
        cudaMemcpyAsync(&kv.second[0], buffers[idx], batchSize * len, cudaMemcpyDeviceToHost, stream);
    }
}
}
