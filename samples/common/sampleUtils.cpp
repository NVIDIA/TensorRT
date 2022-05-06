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

#include "sampleUtils.h"
#include "half.h"

using namespace nvinfer1;

namespace sample
{

size_t dataTypeSize(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4U;
    case nvinfer1::DataType::kHALF: return 2U;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1U;
    }
    return 0;
}

int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps, int32_t batch)
{
    int32_t maxNbElems = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        // Get effective length of axis.
        int32_t d = dims.d[i];
        // Any dimension is 0, it is an empty tensor.
        if (d == 0)
        {
            return 0;
        }
        if (i == vecDim)
        {
            d = samplesCommon::divUp(d, comps);
        }
        maxNbElems = std::max(maxNbElems, d * strides.d[i]);
    }
    return static_cast<int64_t>(maxNbElems) * batch * (vecDim < 0 ? 1 : comps);
}

int64_t volume(nvinfer1::Dims dims, int32_t vecDim, int32_t comps, int32_t batch)
{
    if (vecDim != -1)
    {
        dims.d[vecDim] = roundUp(dims.d[vecDim], comps);
    }
    return volume(dims) * std::max(batch, 1);
}

nvinfer1::Dims toDims(std::vector<int32_t> const& vec)
{
    int32_t limit = static_cast<int32_t>(nvinfer1::Dims::MAX_DIMS);
    if (static_cast<int32_t>(vec.size()) > limit)
    {
        sample::gLogWarning << "Vector too long, only first 8 elements are used in dimension." << std::endl;
    }
    // Pick first nvinfer1::Dims::MAX_DIMS elements
    nvinfer1::Dims dims{std::min(static_cast<int32_t>(vec.size()), limit), {}};
    std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
    return dims;
}

void loadFromFile(std::string const& fileName, char* dst, size_t size)
{
    ASSERT(dst);

    std::ifstream file(fileName, std::ios::in | std::ios::binary);
    if (file.is_open())
    {
        file.read(dst, size);
        file.close();
    }
    else
    {
        std::stringstream msg;
        msg << "Cannot open file " << fileName << "!";
        throw std::invalid_argument(msg.str());
    }
}

bool broadcastIOFormats(std::vector<IOFormat> const& formats, size_t nbBindings, bool isInput /*= true*/)
{
    bool broadcast = formats.size() == 1;
    bool validFormatsCount = broadcast || (formats.size() == nbBindings);
    if (!formats.empty() && !validFormatsCount)
    {
        if (isInput)
        {
            throw std::invalid_argument(
                "The number of inputIOFormats must match network's inputs or be one for broadcasting.");
        }
        else
        {
            throw std::invalid_argument(
                "The number of outputIOFormats must match network's outputs or be one for broadcasting.");
        }
    }
    return broadcast;
}

void sparsifyMatMulKernelWeights(nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights)
{
    using TensorToLayer = std::unordered_map<nvinfer1::ITensor*, nvinfer1::ILayer*>;
    using LayerToTensor = std::unordered_map<nvinfer1::ILayer*, nvinfer1::ITensor*>;

    // 1. Collect layers and tensors information from the network.
    TensorToLayer matmulI2L;
    TensorToLayer constO2L;
    TensorToLayer shuffleI2L;
    LayerToTensor shuffleL2O;
    auto collectMappingInfo = [&](int32_t const idx)
    {
        ILayer* l = network.getLayer(idx);
        switch (l->getType())
        {
        case nvinfer1::LayerType::kMATRIX_MULTIPLY:
        {
            // assume weights on the second input.
            matmulI2L.insert({l->getInput(1), l});
            break;
        }
        case nvinfer1::LayerType::kCONSTANT:
        {
            DataType const dtype = static_cast<nvinfer1::IConstantLayer*>(l)->getWeights().type;
            if (dtype == nvinfer1::DataType::kFLOAT || dtype == nvinfer1::DataType::kHALF)
            {
                // Sparsify float only.
                constO2L.insert({l->getOutput(0), l});
            }
            break;
        }
        case nvinfer1::LayerType::kSHUFFLE:
        {
            shuffleI2L.insert({l->getInput(0), l});
            shuffleL2O.insert({l, l->getOutput(0)});
            break;
        }
        default: break;
        }
    };
    int32_t const nbLayers = network.getNbLayers();
    for (int32_t i = 0; i < nbLayers; ++i)
    {
        collectMappingInfo(i);
    }
    if (matmulI2L.size() == 0 || constO2L.size() == 0)
    {
        // No MatrixMultiply or Constant layer found, no weights to sparsify.
        return;
    }

    // Helper for analysis
    auto isTranspose
        = [](nvinfer1::Permutation const& perm) -> bool { return (perm.order[0] == 1 && perm.order[1] == 0); };
    auto is2D = [](nvinfer1::Dims const& dims) -> bool { return dims.nbDims == 2; };
    auto isIdenticalReshape = [](nvinfer1::Dims const& dims) -> bool
    {
        for (int32_t i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] != i || dims.d[i] != -1)
            {
                return false;
            }
        }
        return true;
    };
    auto tensorReachedViaTranspose = [&](nvinfer1::ITensor* t, bool& needTranspose) -> ITensor*
    {
        while (shuffleI2L.find(t) != shuffleI2L.end())
        {
            nvinfer1::IShuffleLayer* s = static_cast<nvinfer1::IShuffleLayer*>(shuffleI2L.at(t));
            if (!is2D(s->getInput(0)->getDimensions()) || !is2D(s->getReshapeDimensions())
                || !isIdenticalReshape(s->getReshapeDimensions()))
            {
                break;
            }

            if (isTranspose(s->getFirstTranspose()))
            {
                needTranspose = !needTranspose;
            }
            if (isTranspose(s->getSecondTranspose()))
            {
                needTranspose = !needTranspose;
            }

            t = shuffleL2O.at(s);
        }
        return t;
    };

    // 2. Forward analysis to collect the Constant layers connected to MatMul via Transpose
    std::unordered_map<nvinfer1::IConstantLayer*, bool> constantLayerToSparse;
    for (auto& o2l : constO2L)
    {
        // If need to transpose the weights of the Constant layer.
        // Need to transpose by default due to semantic difference.
        bool needTranspose{true};
        ITensor* t = tensorReachedViaTranspose(o2l.first, needTranspose);
        if (matmulI2L.find(t) == matmulI2L.end())
        {
            continue;
        }

        // check MatMul params...
        IMatrixMultiplyLayer* mm = static_cast<nvinfer1::IMatrixMultiplyLayer*>(matmulI2L.at(t));
        bool const twoInputs = mm->getNbInputs() == 2;
        bool const all2D = is2D(mm->getInput(0)->getDimensions()) && is2D(mm->getInput(1)->getDimensions());
        bool const isSimple = mm->getOperation(0) == nvinfer1::MatrixOperation::kNONE
            && mm->getOperation(1) != nvinfer1::MatrixOperation::kVECTOR;
        if (!(twoInputs && all2D && isSimple))
        {
            continue;
        }
        if (mm->getOperation(1) == nvinfer1::MatrixOperation::kTRANSPOSE)
        {
            needTranspose = !needTranspose;
        }

        constantLayerToSparse.insert({static_cast<IConstantLayer*>(o2l.second), needTranspose});
    }

    // 3. Finally, sparsify the weights
    auto sparsifyConstantWeights = [&sparseWeights](nvinfer1::IConstantLayer* layer, bool const needTranspose)
    {
        Dims dims = layer->getOutput(0)->getDimensions();
        ASSERT(dims.nbDims == 2);
        int32_t const idxN = needTranspose ? 1 : 0;
        int32_t const n = dims.d[idxN];
        int32_t const k = dims.d[1 - idxN];
        sparseWeights.emplace_back();
        std::vector<int8_t>& spw = sparseWeights.back();
        Weights w = layer->getWeights();
        DataType const dtype = w.type;
        ASSERT(dtype == nvinfer1::DataType::kFLOAT
            || dtype == nvinfer1::DataType::kHALF); // non-float weights should have been ignored.

        if (needTranspose)
        {
            if (dtype == nvinfer1::DataType::kFLOAT)
            {
                spw.resize(w.count * sizeof(float));
                transpose2DWeights<float>(spw.data(), w.values, k, n);
            }
            else if (dtype == nvinfer1::DataType::kHALF)
            {
                spw.resize(w.count * sizeof(half_float::half));
                transpose2DWeights<half_float::half>(spw.data(), w.values, k, n);
            }

            w.values = spw.data();
            std::vector<int8_t> tmpW;
            sparsify(w, n, 1, tmpW);

            if (dtype == nvinfer1::DataType::kFLOAT)
            {
                transpose2DWeights<float>(spw.data(), tmpW.data(), n, k);
            }
            else if (dtype == nvinfer1::DataType::kHALF)
            {
                transpose2DWeights<half_float::half>(spw.data(), tmpW.data(), n, k);
            }
        }
        else
        {
            sparsify(w, n, 1, spw);
        }

        w.values = spw.data();
        layer->setWeights(w);
    };
    for (auto& l : constantLayerToSparse)
    {
        sparsifyConstantWeights(l.first, l.second);
    }
}

template <typename L>
void setSparseWeights(L& l, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights)
{
    auto weights = l.getKernelWeights();
    sparsify(weights, k, rs, sparseWeights);
    weights.values = sparseWeights.data();
    l.setKernelWeights(weights);
}

// Explicit instantiation
template void setSparseWeights<IConvolutionLayer>(
    IConvolutionLayer& l, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);
template void setSparseWeights<IFullyConnectedLayer>(
    IFullyConnectedLayer& l, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

void sparsify(nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights)
{
    for (int32_t l = 0; l < network.getNbLayers(); ++l)
    {
        auto* layer = network.getLayer(l);
        auto const t = layer->getType();
        if (t == nvinfer1::LayerType::kCONVOLUTION)
        {
            auto& conv = *static_cast<IConvolutionLayer*>(layer);
            auto const& dims = conv.getKernelSizeNd();
            if (dims.nbDims > 2)
            {
                continue;
            }
            auto const k = conv.getNbOutputMaps();
            auto const rs = dims.d[0] * dims.d[1];
            sparseWeights.emplace_back();
            setSparseWeights(conv, k, rs, sparseWeights.back());
        }
        else if (t == nvinfer1::LayerType::kFULLY_CONNECTED)
        {
            auto& fc = *static_cast<nvinfer1::IFullyConnectedLayer*>(layer);
            auto const k = fc.getNbOutputChannels();
            sparseWeights.emplace_back();
            setSparseWeights(fc, k, 1, sparseWeights.back());
        }
    }

    sparsifyMatMulKernelWeights(network, sparseWeights);
}

void sparsify(Weights const& weights, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights)
{
    switch (weights.type)
    {
    case DataType::kFLOAT:
        sparsify(static_cast<float const*>(weights.values), weights.count, k, rs, sparseWeights);
        break;
    case DataType::kHALF:
        sparsify(static_cast<half_float::half const*>(weights.values), weights.count, k, rs, sparseWeights);
        break;
    case DataType::kINT8:
    case DataType::kINT32:
    case DataType::kBOOL: break;
    }
}

template <typename T>
void dumpBuffer(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv)
{
    int64_t const volume = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
    T const* typedBuffer = static_cast<T const*>(buffer);
    std::string sep;
    for (int64_t v = 0; v < volume; ++v)
    {
        int64_t curV = v;
        int32_t dataOffset = 0;
        for (int32_t dimIndex = dims.nbDims - 1; dimIndex >= 0; --dimIndex)
        {
            int32_t dimVal = curV % dims.d[dimIndex];
            if (dimIndex == vectorDim)
            {
                dataOffset += (dimVal / spv) * strides.d[dimIndex] * spv + dimVal % spv;
            }
            else
            {
                dataOffset += dimVal * strides.d[dimIndex] * (vectorDim == -1 ? 1 : spv);
            }
            curV /= dims.d[dimIndex];
            ASSERT(curV >= 0);
        }

        os << sep << typedBuffer[dataOffset];
        sep = separator;
    }
}

// Explicit instantiation
template void dumpBuffer<bool>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<int32_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<int8_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<float>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<__half>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);

template <typename T>
void sparsify(T const* values, int64_t count, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights)
{
    auto const c = count / (k * rs);
    sparseWeights.resize(count * sizeof(T));
    auto* sparseValues = reinterpret_cast<T*>(sparseWeights.data());

    constexpr int32_t window = 4;
    constexpr int32_t nonzeros = 2;

    int32_t const crs = c * rs;
    auto const getIndex = [=](int32_t ki, int32_t ci, int32_t rsi) { return ki * crs + ci * rs + rsi; };

    for (int64_t ki = 0; ki < k; ++ki)
    {
        for (int64_t rsi = 0; rsi < rs; ++rsi)
        {
            int32_t w = 0;
            int32_t nz = 0;
            for (int64_t ci = 0; ci < c; ++ci)
            {
                auto const index = getIndex(ki, ci, rsi);
                if (nz < nonzeros)
                {
                    sparseValues[index] = values[index];
                    ++nz;
                }
                else
                {
                    sparseValues[index] = 0;
                }
                if (++w == window)
                {
                    w = 0;
                    nz = 0;
                }
            }
        }
    }
}

// Explicit instantiation
template void sparsify<float>(
    float const* values, int64_t count, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);
template void sparsify<half_float::half>(
    half_float::half const* values, int64_t count, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

template <typename T>
void transpose2DWeights(void* dst, void const* src, int32_t const m, int32_t const n)
{
    ASSERT(dst != src);
    T* tdst = reinterpret_cast<T*>(dst);
    T const* tsrc = reinterpret_cast<T const*>(src);
    for (int32_t mi = 0; mi < m; ++mi)
    {
        for (int32_t ni = 0; ni < n; ++ni)
        {
            int32_t const isrc = mi * n + ni;
            int32_t const idst = ni * m + mi;
            tdst[idst] = tsrc[isrc];
        }
    }
}

// Explicit instantiation
template void transpose2DWeights<float>(void* dst, void const* src, int32_t const m, int32_t const n);
template void transpose2DWeights<half_float::half>(void* dst, void const* src, int32_t const m, int32_t const n);

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type>
void fillBuffer(void* buffer, int64_t volume, T min, T max)
{
    T* typedBuffer = static_cast<T*>(buffer);
    std::default_random_engine engine;
    std::uniform_int_distribution<int32_t> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, int32_t>::type>
void fillBuffer(void* buffer, int64_t volume, T min, T max)
{
    T* typedBuffer = static_cast<T*>(buffer);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
}

// Explicit instantiation
template void fillBuffer<bool>(void* buffer, int64_t volume, bool min, bool max);
template void fillBuffer<float>(void* buffer, int64_t volume, float min, float max);
template void fillBuffer<int32_t>(void* buffer, int64_t volume, int32_t min, int32_t max);
template void fillBuffer<int8_t>(void* buffer, int64_t volume, int8_t min, int8_t max);
template void fillBuffer<__half>(void* buffer, int64_t volume, __half min, __half max);

} // namespace sample
