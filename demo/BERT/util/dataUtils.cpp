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

#include <cassert>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <numeric>

#include "attentionKeys.h"
#include "common.h"
#include "dataUtils.h"

namespace bert
{

using namespace nvinfer1;
using namespace samplesCommon;
using std::cout;
using std::endl;

void parseDims(std::ifstream& input, const std::string& name, Dims& d)
{
    const int n_dims = d.nbDims;
    for (int it = 0; it < n_dims; it++)
    {
        input >> d.d[it];
    }
    cout << name << ": nbDim=" << d.nbDims << " dim: " << d << endl;
    assert(input.peek() == ' ');
    input.get();
}

//! \brief Inplace transpose of a column major matrix in host memory
//! \param data dense storage of matrix in column major format
//! \param d dimensions of matrix
void transposeMatrix(float* data, const Dims& d)
{
    // data represents a d0xd1 RowMajor matrix, i.e. d1xd0 ColMajor
    // which we transpose into d0xd1 ColMajor
    assert(d.nbDims == 2);
    const int len = d.d[0] * d.d[1];
    std::vector<float> tmp(data, data + len);

    int idx = 0; // result matrix linear idx
    for (int c = 0; c < d.d[1]; c++)
    { // columns of the result matrix
        for (int r = 0; r < d.d[0]; r++)
        { // rows of the result matrix
            const int srcIdx = r * d.d[1] + c;
            data[idx++] = tmp[srcIdx];
        }
    }
}

template <typename T>
DataType getDType()
{
    if (std::is_same<T, float>::value)
    {
        return DataType::kFLOAT;
    }
    else if (std::is_same<T, int>::value)
    {
        return DataType::kINT32;
    }
    throw std::runtime_error("Invalid DataType.");
}

template <typename T>
void loadRow(std::ifstream& input, std::string& name, T*& data, int& nbWeights, Dims& d)
{
    int32_t type;
    int32_t nbDims;
    int32_t dim;
    input >> name >> std::dec >> type >> nbDims;
    assert(nbDims > 0 && nbDims <= Dims::MAX_DIMS);
    d.nbDims = nbDims;
    parseDims(input, name, d);

    const DataType dtype = static_cast<DataType>(type);
    assert(dtype == getDType<T>());

    nbWeights = std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
    data = new T[nbWeights]; // should be freed by caller
    char* bytes = reinterpret_cast<char*>(data);
    input.read(bytes, nbWeights * sizeof(T));

    assert(input.peek() == '\n');
    input.get();
}

void loadWeights(const std::string& wts_path, WeightMap& weightMap)
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
        loadRow(input, name, data, param_size, d);
        assert(data);
        assert(param_size);

        // Need to be careful here. This is highly dependent on the TF implementation.
        // The TF squad output does not use a fully connected layer but a matmul op with a transpose, which names the
        // output as squad_output_weights
        if (name.find("kernel") != std::string::npos)
        {
            cout << "Transposing\n";
            transposeMatrix(data, d);
        }
        Weights tmp;
        tmp.type = DataType::kFLOAT;
        tmp.values = data;
        tmp.count = param_size;

        weightMap[name] = tmp;
    }

    input.close();

    for (auto& kv : weightMap)
    {

        const int pos = kv.first.find(BQ); // starting pos of BQ
        if (pos != std::string::npos)
        {
            const int hidden_size = kv.second.count;
            const std::string prefix = kv.first.substr(0, pos);

            const Weights& Bq_ = kv.second;
            const Weights Bk_ = weightMap.at(prefix + BK);
            const Weights Bv_ = weightMap.at(prefix + BV);
            const Weights Wq_ = weightMap.at(prefix + WQ);
            const Weights Wk_ = weightMap.at(prefix + WK);
            const Weights Wv_ = weightMap.at(prefix + WV);

            const int mat_size = hidden_size * hidden_size;
            const int wcount = 3 * mat_size;
            float* Wall_ptr = new float[wcount];

            const int bcount = 3 * hidden_size;
            float* Ball_ptr = new float[bcount];

            std::copy(reinterpret_cast<const float*>(Wq_.values),
                (reinterpret_cast<const float*>(Wq_.values)) + mat_size, Wall_ptr);
            std::copy(reinterpret_cast<const float*>(Wk_.values),
                (reinterpret_cast<const float*>(Wk_.values)) + mat_size, Wall_ptr + mat_size);
            std::copy(reinterpret_cast<const float*>(Wv_.values),
                (reinterpret_cast<const float*>(Wv_.values)) + mat_size, Wall_ptr + 2 * mat_size);
            std::copy(reinterpret_cast<const float*>(Bq_.values),
                (reinterpret_cast<const float*>(Bq_.values)) + hidden_size, Ball_ptr);
            std::copy(reinterpret_cast<const float*>(Bk_.values),
                (reinterpret_cast<const float*>(Bk_.values)) + hidden_size, Ball_ptr + hidden_size);
            std::copy(reinterpret_cast<const float*>(Bv_.values),
                (reinterpret_cast<const float*>(Bv_.values)) + hidden_size, Ball_ptr + 2 * hidden_size);

            weightMap[prefix + WQKV] = {DataType::kFLOAT, Wall_ptr, wcount};
            weightMap[prefix + BQKV] = {DataType::kFLOAT, Ball_ptr, bcount};
        }
    }
}

void loadInputs(const std::string& weightsPath, int& Bmax, int& S, std::vector<nvinfer1::Weights>& inputIds,
    std::vector<nvinfer1::Weights>& inputMasks, std::vector<nvinfer1::Weights>& segmentIds,
    std::vector<nvinfer1::Dims>& inputDims)
{
    std::ifstream input(weightsPath, std::ios_base::binary);
    int32_t count;
    input >> count;
    cout << "Number of buffers: " << count << endl;
    assert(count % 3 == 0);
    S = 0;
    Bmax = 0;

    std::vector<nvinfer1::Dims> maskDims;
    std::vector<nvinfer1::Dims> segIdDims;
    for (int it = 0; it < count; it++)
    {
        int nbWeights = 0;
        std::string name;
        int* data = nullptr;
        Dims d;
        loadRow(input, name, data, nbWeights, d);
        assert(data);
        assert(nbWeights);

        Bmax = std::max(Bmax, d.d[0]); // find the largest batch size in the dataset

        if (S == 0)
        {
            S = d.d[1];
        }
        else
        {
            assert(S == d.d[1]);
        } // all inputs should have the same sequence length

        Weights tmp;
        tmp.type = DataType::kINT32;
        tmp.values = data;
        tmp.count = nbWeights;

        if (name.find("input_id") != std::string::npos)
        {
            inputIds.push_back(tmp);
            inputDims.push_back(d);
            continue;
        }

        if (name.find("input_mask") != std::string::npos)
        {
            inputMasks.push_back(tmp);
            maskDims.push_back(d);
            continue;
        }
        segmentIds.push_back(tmp);
        segIdDims.push_back(d);
    }
    input.close();
    assert(inputIds.size() == count / 3);
    assert(inputMasks.size() == inputIds.size());
    assert(segmentIds.size() == inputIds.size());
    assert(inputDims.size() == inputIds.size());
    assert(S);
    assert(Bmax);

    for (int it = 0; it < inputIds.size(); it++)
    {
        assert(inputIds[it].count == inputMasks[it].count);
        assert(inputIds[it].count == segmentIds[it].count);
        assert(inputDims[it] == maskDims[it]);
        assert(inputDims[it] == segIdDims[it]);
    }
}

void inferNetworkSizes(const WeightMap& weightMap, int& hiddenSize, int& intermediateSize, int& numHiddenLayers)
{
    for (const auto& kv : weightMap)
    {
        if (kv.first.find("beta") != std::string::npos)
        {
            hiddenSize = kv.second.count;
            break;
        }
    }
    for (const auto& kv : weightMap)
    {
        if (kv.first.find("intermediate_dense_bias") != std::string::npos)
        {
            intermediateSize = kv.second.count;
            break;
        }
    }
    numHiddenLayers = 0;
    for (const auto& kv : weightMap)
    {
        if (kv.first[0] == 'l')
        {
            const std::string tok = kv.first.substr(1, kv.first.find("_") - 1);
            const int layer = std::stoi(tok);
            numHiddenLayers = std::max(numHiddenLayers, layer + 1);
        }
    }
}

void allocBindingsFromWeights(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, nvinfer1::Weights>& dict, int verbose)
{

    Weights W;
    std::string name;

    for (auto& kv : dict)
    {
        std::tie(name, W) = kv;
        const int idx = engine.getBindingIndex(name.c_str());
        if (verbose)
        {
            printf(" idx %d name %s\n", idx, name.c_str());
        }
        assert(idx >= 0);
        const int outlen = W.count * getElementSize(W.type);
        CHECK(cudaMalloc(&buffers[idx], outlen));
        if (verbose)
        {
            printf(" idx %d allocated %d bytes\n", idx, outlen);
        }
    }
}

void allocBindingsFromVectors(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, std::vector<float>>& dict, int verbose)
{

    for (auto& kv : dict)
    {
        const int idx = engine.getBindingIndex(kv.first.c_str());
        if (verbose)
        {
            printf(" idx %d name %s\n", idx, kv.first.c_str());
        }
        assert(idx >= 0);
        const int outlen = sizeof(float) * kv.second.size();
        CHECK(cudaMalloc(&buffers[idx], outlen));
        if (verbose)
        {
            printf(" idx %d allocated %d bytes\n", idx, outlen);
        }
    }
}

void copyToDeviceBindings(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    const std::map<std::string, Weights>& dict, cudaStream_t stream)
{

    Weights W;
    std::string name;

    for (auto kv : dict)
    {
        std::tie(name, W) = kv;
        const int idx = engine.getBindingIndex(name.c_str());
        const int len = W.count * getElementSize(W.type);
        CHECK(cudaMemcpyAsync(buffers[idx], W.values, len, cudaMemcpyHostToDevice, stream));
    }
}

void copyFromDeviceBindings(const ICudaEngine& engine, std::vector<void*>& buffers, const int batchSize,
    std::map<std::string, std::vector<float>>& dict, cudaStream_t stream)
{
    for (auto& kv : dict)
    {
        const int idx = engine.getBindingIndex(kv.first.c_str());
        const int len = kv.second.size() * sizeof(float);
        CHECK(cudaMemcpyAsync(&kv.second[0], buffers[idx], len, cudaMemcpyDeviceToHost, stream));
        printf("Binding %s idx %d downloading %d bytes\n", kv.first.c_str(), idx, len);
    }
}

void transposeLogits(std::vector<float>& logits, const int B, const int S)
{
    // BxSx2 => 2xBxS
    assert(logits.size() == B * S * 2);
    std::vector<float> tmp(logits);
    std::copy(logits.begin(), logits.end(), tmp.begin());
    for (int b = 0; b < B; b++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int p = 0; p < 2; p++)
            {
                const int inIdx = p + s * 2 + b * 2 * S;
                const int outIdx = s + b * S + p * B * S;
                logits[outIdx] = tmp[inIdx];
            }
        }
    }
}

} // namespace bert
