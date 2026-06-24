/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "bfloat16.h"
#include "common.h"
#include "half.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <type_traits>

#if CUDA_VERSION >= 11060
#include <cuda_fp8.h>
#endif

using namespace nvinfer1;
using samplesCommon::startsWith;

namespace sample
{

using TensorToLayer = std::unordered_map<nvinfer1::ITensor*, nvinfer1::ILayer*>;
using LayerToTensor = std::unordered_map<nvinfer1::ILayer*, nvinfer1::ITensor*>;
using TensorToTensor = std::unordered_map<nvinfer1::ITensor*, nvinfer1::ITensor*>;

int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps, int32_t batch)
{
    int64_t maxNbElems = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        // Get effective length of axis.
        int64_t d = dims.d[i];
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
    return maxNbElems * batch * (vecDim < 0 ? 1 : comps);
}

nvinfer1::Dims toDims(std::vector<int64_t> const& vec)
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
        file.seekg(0, std::ios::end);
        int64_t fileSize = static_cast<int64_t>(file.tellg());
        // Due to change from int32_t to int64_t VC engines created with earlier versions
        // may expect input of the half of the size
        if (fileSize != static_cast<int64_t>(size) && fileSize != static_cast<int64_t>(size * 2))
        {
            std::ostringstream msg;
            msg << "Unexpected file size for input file: " << fileName << ". Note: Input binding size is: " << size
                << " bytes but the file size is " << fileSize
                << " bytes. Double check the size and datatype of the provided data.";
            throw std::invalid_argument(msg.str());
        }
        // Move file pointer back to the beginning after reading file size.
        file.seekg(0, std::ios::beg);
        file.read(dst, size);
        size_t const nbBytesRead = file.gcount();
        file.close();
        if (nbBytesRead != size)
        {
            std::ostringstream msg;
            msg << "Unexpected file size for input file: " << fileName << ". Note: Expected: " << size
                << " bytes but only read: " << nbBytesRead << " bytes";
            throw std::invalid_argument(msg.str());
        }
    }
    else
    {
        std::ostringstream msg;
        msg << "Cannot open file " << fileName << "!";
        throw std::invalid_argument(msg.str());
    }
}

std::vector<std::string> splitToStringVec(std::string const& s, char separator, int64_t maxSplit)
{
    std::vector<std::string> splitted;

    for (size_t start = 0; start < s.length();)
    {
        // If maxSplit is specified and we have reached maxSplit, emplace back the rest of the string and break the
        // loop.
        if (maxSplit >= 0 && static_cast<int64_t>(splitted.size()) == maxSplit)
        {
            splitted.emplace_back(s.substr(start, s.length() - start));
            break;
        }

        size_t separatorIndex = s.find(separator, start);
        if (separatorIndex == std::string::npos)
        {
            separatorIndex = s.length();
        }
        splitted.emplace_back(s.substr(start, separatorIndex - start));

        // If the separator is the last character, then we should push an empty string at the end.
        if (separatorIndex == s.length() - 1)
        {
            splitted.emplace_back("");
        }

        start = separatorIndex + 1;
    }

    return splitted;
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

        throw std::invalid_argument(
            "The number of outputIOFormats must match network's outputs or be one for broadcasting.");
    }
    return broadcast;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void sparsifyMatMulKernelWeights(nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights)
{
    // 1. Collect layers and tensors information from the network.
    TensorToLayer matmulI2L;
    TensorToLayer constO2L;
    TensorToLayer shuffleI2L;
    LayerToTensor shuffleL2O;
    auto collectMappingInfo = [&](int32_t const idx) {
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
    auto isIdenticalReshape = [](nvinfer1::Dims const& dims) -> bool {
        for (int32_t i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] != i || dims.d[i] != -1)
            {
                return false;
            }
        }
        return true;
    };
    auto tensorReachedViaTranspose = [&](nvinfer1::ITensor* t, bool& needTranspose) -> ITensor* {
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
    auto sparsifyConstantWeights = [&sparseWeights](nvinfer1::IConstantLayer* layer, bool const needTranspose) {
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
void setSparseWeights(L& l, int32_t k, int32_t trs, std::vector<int8_t>& sparseWeights)
{
    auto weights = l.getKernelWeights();
    sparsify(weights, k, trs, sparseWeights);
    weights.values = sparseWeights.data();
    l.setKernelWeights(weights);
}

// Explicit instantiation
template void setSparseWeights<IConvolutionLayer>(
    IConvolutionLayer& l, int32_t k, int32_t trs, std::vector<int8_t>& sparseWeights);

//! \brief Sparsify conv weights fed via Q/DQ chains (companion to sparsifyMatMulKernelWeights).
//!
//! Strongly-typed Q/DQ networks attach the conv weight as a tensor input rather than
//! static kernelWeights. Walks the chain forward from each FP Constant:
//!     Constant -> Shuffle* -> Q? -> Shuffle* -> DQ -> Shuffle* -> Conv.input(1)
//! If the chain terminates at a Conv weight input, sparsify the constant in place.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void sparsifyQDQConvKernelWeights(
    nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights)
{
    TensorToLayer convWeightI2L;
    TensorToLayer constO2L;
    TensorToTensor dqI2O;
    TensorToTensor qI2O;
    TensorToTensor shuffleI2O;
    auto collectMappingInfo = [&](ILayer& l) {
        switch (l.getType())
        {
        case nvinfer1::LayerType::kCONVOLUTION:
            // Conv with weights as a tensor input (vs. static kernelWeights).
            if (l.getNbInputs() >= 2 && l.getInput(1) != nullptr)
            {
                convWeightI2L.try_emplace(l.getInput(1), &l);
            }
            break;
        case nvinfer1::LayerType::kCONSTANT:
        {
            DataType const dtype = static_cast<nvinfer1::IConstantLayer&>(l).getWeights().type;
            auto const floatDTypes = {nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF, nvinfer1::DataType::kBF16};
            if (std::any_of(floatDTypes.begin(), floatDTypes.end(), [dtype](auto t) { return t == dtype; }))
            {
                constO2L.try_emplace(l.getOutput(0), &l);
            }
            break;
        }
        case nvinfer1::LayerType::kDEQUANTIZE: dqI2O.try_emplace(l.getInput(0), l.getOutput(0)); break;
        case nvinfer1::LayerType::kQUANTIZE: qI2O.try_emplace(l.getInput(0), l.getOutput(0)); break;
        case nvinfer1::LayerType::kSHUFFLE: shuffleI2O.try_emplace(l.getInput(0), l.getOutput(0)); break;
        default: break;
        }
    };
    int32_t const nbLayers = network.getNbLayers();
    for (int32_t i = 0; i < nbLayers; ++i)
    {
        collectMappingInfo(*network.getLayer(i));
    }
    if (convWeightI2L.size() == 0 || constO2L.size() == 0 || dqI2O.size() == 0)
    {
        return;
    }

    //! Skip past any Shuffle layers consuming t and return the tensor at the chain's end.
    //! Returns t unchanged if no Shuffle reads it.
    auto walkShuffleChain = [&](nvinfer1::ITensor* t) -> ITensor* {
        while (true)
        {
            auto const it = shuffleI2O.find(t);
            if (it == shuffleI2O.end())
            {
                break;
            }
            t = it->second;
        }
        return t;
    };

    //! Follow Constant -> Shuffle* -> Q? -> Shuffle* -> DQ -> Shuffle* -> Conv.input(1) chain.
    //! Returns the terminating IConvolutionLayer*, or nullptr if the chain breaks.
    auto walkShuffleQDQChain = [&](nvinfer1::ITensor* t) -> IConvolutionLayer* {
        t = walkShuffleChain(t);
        if (auto const qI2OIt = qI2O.find(t); qI2OIt != qI2O.end())
        {
            t = walkShuffleChain(qI2OIt->second);
        }
        auto const dqI2OIt = dqI2O.find(t);
        if (dqI2OIt == dqI2O.end())
        {
            return nullptr;
        }
        t = walkShuffleChain(dqI2OIt->second);
        auto const convWeightI2LIt = convWeightI2L.find(t);
        if (convWeightI2LIt == convWeightI2L.end())
        {
            return nullptr;
        }
        ASSERT(convWeightI2LIt->second->getType() == nvinfer1::LayerType::kCONVOLUTION);
        return static_cast<nvinfer1::IConvolutionLayer*>(convWeightI2LIt->second);
    };

    for (auto& o2l : constO2L)
    {
        IConvolutionLayer* const conv = walkShuffleQDQChain(o2l.first);
        if (conv == nullptr)
        {
            continue;
        }
        ASSERT(o2l.second->getType() == nvinfer1::LayerType::kCONSTANT);
        IConstantLayer* constLayer = static_cast<nvinfer1::IConstantLayer*>(o2l.second);
        Weights w = constLayer->getWeights();
        if (w.count == 0)
        {
            continue;
        }
        Dims const kernelDims = conv->getKernelSizeNd();
        int32_t const k = conv->getNbOutputMaps();
        int64_t const trs = samplesCommon::volume(kernelDims);
        // sparsify() reconstructs c (input channels) via c = count / (k*trs); fail loudly if
        // the constant's element count doesn't match the KCRS layout this routine assumes.
        ASSERT(k > 0 && 0 < trs && trs <= std::numeric_limits<int32_t>::max()
            && w.count % (static_cast<int64_t>(k) * trs) == 0);
        sparseWeights.emplace_back();
        sparsify(w, k, static_cast<int32_t>(trs), sparseWeights.back());
        w.values = sparseWeights.back().data();
        constLayer->setWeights(w);
    }
}

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
            ASSERT(dims.nbDims == 2 || dims.nbDims == 3);
            auto const k = conv.getNbOutputMaps();
            auto const trs = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int32_t>());
            sparseWeights.emplace_back();
            setSparseWeights(conv, k, trs, sparseWeights.back());
        }
    }

    sparsifyMatMulKernelWeights(network, sparseWeights);
    sparsifyQDQConvKernelWeights(network, sparseWeights);
    sample::gLogVerbose << "--sparsity=force pruned " << sparseWeights.size() << " weights to be sparsity pattern."
                        << std::endl;
    sample::gLogVerbose << "--sparsity=force has been deprecated. Please use <polygraphy surgeon prune> to rewrite the "
                           "weights to a sparsity pattern and then run with --sparsity=enable"
                        << std::endl;
}

void sparsify(Weights const& weights, int32_t k, int32_t trs, std::vector<int8_t>& sparseWeights)
{
    switch (weights.type)
    {
    case DataType::kFLOAT:
        sparsify(static_cast<float const*>(weights.values), weights.count, k, trs, sparseWeights);
        break;
    case DataType::kHALF:
        sparsify(static_cast<half_float::half const*>(weights.values), weights.count, k, trs, sparseWeights);
        break;
    case DataType::kBF16:
        sparsify(static_cast<BFloat16 const*>(weights.values), weights.count, k, trs, sparseWeights);
        break;
    case DataType::kINT8:
    case DataType::kINT32:
    case DataType::kUINT8:
    case DataType::kBOOL:
    case DataType::kINT4:
    case DataType::kFP8:
    case DataType::kINT64:
    case DataType::kFP4: ASSERT(false && "Unsupported data type");
    case DataType::kE8M0: ASSERT(false && "E8M0 is not supported");
    }
}

template <typename T>
void print(std::ostream& os, T v)
{
    os << v;
}

void print(std::ostream& os, int8_t v)
{
    os << static_cast<int32_t>(v);
}

void print(std::ostream& os, uint8_t v)
{
    os << static_cast<uint32_t>(v);
}

void print(std::ostream& os, __half v)
{
    os << static_cast<float>(v);
}

#if CUDA_VERSION >= 11060
void print(std::ostream& os, __nv_fp8_e4m3 v)
{
    os << static_cast<float>(v);
}
#endif

int32_t dataOffsetFromDims(int64_t v, Dims const& dims, Dims const& strides, int32_t vectorDim, int32_t spv)
{
    int32_t dataOffset = 0;
    for (int32_t dimIndex = dims.nbDims - 1; dimIndex >= 0; --dimIndex)
    {
        int32_t dimVal = v % dims.d[dimIndex];
        if (dimIndex == vectorDim)
        {
            dataOffset += (dimVal / spv) * strides.d[dimIndex] * spv + dimVal % spv;
        }
        else
        {
            dataOffset += dimVal * strides.d[dimIndex] * (vectorDim == -1 ? 1 : spv);
        }
        v /= dims.d[dimIndex];
        ASSERT(v >= 0);
    }

    return dataOffset;
}

template <typename T>
void dumpBuffer(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv)
{
    auto const vol = volume(dims);
    T const* typedBuffer = static_cast<T const*>(buffer);
    for (int64_t v = 0; v < vol; ++v)
    {
        int32_t dataOffset = dataOffsetFromDims(v, dims, strides, vectorDim, spv);
        if (v > 0)
        {
            os << separator;
        }
        print(os, typedBuffer[dataOffset]);
    }
}

void dumpInt4Buffer(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv)
{
    auto const vol = volume(dims);
    uint8_t const* typedBuffer = static_cast<uint8_t const*>(buffer);
    for (int64_t v = 0; v < vol; ++v)
    {
        int32_t dataOffset = dataOffsetFromDims(v, dims, strides, vectorDim, spv);
        if (v > 0)
        {
            os << separator;
        }

        auto value = typedBuffer[dataOffset / 2];
        if (dataOffset % 2 == 0)
        {
            // Cast to int8_t before right shift, so right-shift will sign-extend.
            // Left shift on int8_t can be undefined behaviour, must perform left shift on uint8_t.
            os << (static_cast<int8_t>(value << 4) >> 4);
        }
        else
        {
            os << (static_cast<int8_t>(value) >> 4);
        }
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
template void dumpBuffer<BFloat16>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
#if CUDA_VERSION >= 11060
template void dumpBuffer<__nv_fp8_e4m3>(void const* buffer, std::string const& separator, std::ostream& os,
    Dims const& dims, Dims const& strides, int32_t vectorDim, int32_t spv);
#endif
template void dumpBuffer<uint8_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<int64_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);

template <typename T>
void sparsify(T const* values, int64_t count, int32_t k, int32_t trs, std::vector<int8_t>& sparseWeights)
{
    auto const c = count / (k * trs);
    sparseWeights.resize(count * sizeof(T));
    auto* sparseValues = reinterpret_cast<T*>(sparseWeights.data());

    constexpr int32_t window = 4;
    constexpr int32_t nonzeros = 2;

    int32_t const crs = c * trs;
    auto const getIndex = [=](int32_t ki, int32_t ci, int32_t rsi) { return ki * crs + ci * trs + rsi; };

    for (int64_t ki = 0; ki < k; ++ki)
    {
        for (int64_t rsi = 0; rsi < trs; ++rsi)
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
    float const* values, int64_t count, int32_t k, int32_t trs, std::vector<int8_t>& sparseWeights);
template void sparsify<half_float::half>(
    half_float::half const* values, int64_t count, int32_t k, int32_t trs, std::vector<int8_t>& sparseWeights);

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
void fillBuffer(void* buffer, int64_t volume, int32_t min, int32_t max)
{
    T* typedBuffer = static_cast<T*>(buffer);
    std::default_random_engine engine;
    std::uniform_int_distribution<int32_t> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type>
void fillBuffer(void* buffer, int64_t volume, float min, float max)
{
    T* typedBuffer = static_cast<T*>(buffer);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
}

// Explicit instantiation
template void fillBuffer<bool>(void* buffer, int64_t volume, int32_t min, int32_t max);
template void fillBuffer<int32_t>(void* buffer, int64_t volume, int32_t min, int32_t max);
template void fillBuffer<int8_t>(void* buffer, int64_t volume, int32_t min, int32_t max);
template void fillBuffer<float>(void* buffer, int64_t volume, float min, float max);
template void fillBuffer<__half>(void* buffer, int64_t volume, float min, float max);
template void fillBuffer<BFloat16>(void* buffer, int64_t volume, float min, float max);
#if CUDA_VERSION >= 11060
template void fillBuffer<__nv_fp8_e4m3>(void* buffer, int64_t volume, float min, float max);
#endif
template void fillBuffer<uint8_t>(void* buffer, int64_t volume, int32_t min, int32_t max);
template void fillBuffer<int64_t>(void* buffer, int64_t volume, int32_t min, int32_t max);

bool matchStringWithOneWildcard(std::string const& pattern, std::string const& target)
{
    auto const splitPattern = splitToStringVec(pattern, '*', 1);

    // If there is no wildcard, return if the two strings match exactly.
    if (splitPattern.size() == 1)
    {
        return pattern == target;
    }

    // Otherwise, target must follow prefix+anything+postfix pattern.
    return target.size() >= (splitPattern[0].size() + splitPattern[1].size()) && target.find(splitPattern[0]) == 0
        && target.rfind(splitPattern[1]) == (target.size() - splitPattern[1].size());
}

//! @brief Sanitizes the remote auto tuning config string by removing sensitive credentials
//!
//! This function removes usernames and passwords from URL-style configuration strings
//! to prevent sensitive authentication information from appearing in logs or debug output.
//! The credentials section (username:password) is replaced with "***" for security.
//!
//! Config format: protocol://username[:password]@hostname[:port]?param1=value1&param2=value2
//! Supported protocols: ssh, http, https, etc.
//!
//! Examples:
//!   Input:  "ssh://admin:secretpass@server.com:22?timeout=30"
//!   Output: "ssh://***@server.com:22?timeout=30"
//!
//! @param config The configuration string to sanitize
//! @return Sanitized configuration string with passwords and usernames replaced by ***
std::string sanitizeRemoteAutoTuningConfig(std::string const& config)
{
    if (config.empty())
    {
        return config;
    }

    try
    {
        // Find the protocol part (before ://)
        size_t protocolEnd = config.find("://");
        if (protocolEnd == std::string::npos)
        {
            return config; // Invalid format, return as is
        }

        // Find the credentials part (between :// and @)
        size_t credentialsStart = protocolEnd + 3;
        if (credentialsStart >= config.length())
        {
            return config; // Truncated after protocol
        }

        size_t credentialsEnd = config.find('@', credentialsStart);
        if (credentialsEnd == std::string::npos)
        {
            return config; // No credentials, return as is
        }

        // Extract parts and sanitize
        std::string protocol = config.substr(0, protocolEnd);
        std::string hostAndParams = config.substr(credentialsEnd);

        // Return sanitized version
        return protocol + "://***" + hostAndParams;
    }
    catch (std::exception const& e)
    {
        sample::gLogError << "Exception in sanitizeRemoteAutoTuningConfig: " << e.what() << std::endl;
        return config; // Return original on error
    }
    catch (...)
    {
        sample::gLogError << "Unknown exception in sanitizeRemoteAutoTuningConfig" << std::endl;
        return config; // Return original on error
    }
}

bool validateNonEmpty(std::string const& value, std::string const& flagName)
{
    if (value.empty())
    {
        sample::gLogError << flagName << " cannot be empty" << std::endl;
        return false;
    }
    return true;
}

bool validateRemoteAutoTuningConfig(std::string const& config)
{
    if (config.find("://") == std::string::npos)
    {
        sample::gLogError << "Invalid remote auto tuning config format. Expected format: "
                             "protocol://username[:password]@hostname[:port]?param1=value1&param2=value2"
                          << std::endl;
        return false;
    }
    return true;
}

std::vector<std::string> sanitizeArgv(int32_t argc, char** argv)
{
    std::vector<std::string> sanitizedArgs;
    sanitizedArgs.reserve(argc);

    for (int32_t i = 0; i < argc; ++i)
    {
        std::string arg = argv[i];

        // Sanitize remoteAutoTuningConfig argument
        if (auto const flag = std::string("--remoteAutoTuningConfig=");
            arg.size() > flag.size() && arg.substr(0, flag.size()) == flag)
        {
            arg = std::string(flag) + sanitizeRemoteAutoTuningConfig(arg.substr(flag.size()));
        }

        sanitizedArgs.push_back(arg);
    }

    return sanitizedArgs;
}

// ============================================================================
// Accuracy Validator Implementations
// ============================================================================

template <typename T>
double L0AccuracyValidator<T>::calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference)
{
    // Uses PyTorch/NumPy allclose formula: |a - b| <= atol + rtol * |b|
    // See: https://docs.pytorch.org/docs/stable/generated/torch.allclose.html
    // and infer_ref_check/infer_ref_check.cpp::torchIsClose()
    ASSERT(actual.size() == reference.size());
    ASSERT(actual.size() != 0);
    int64_t mismatchCount = 0;
    for (uint64_t i = 0; i < actual.size(); ++i)
    {
        double const absDiff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(reference[i]));
        double const refAbs = std::abs(static_cast<double>(reference[i]));
        double const tolerance = mAtol + mRtol * refAbs;
        if (absDiff > tolerance)
        {
            mismatchCount++;
        }
    }
    return static_cast<double>(mismatchCount) / actual.size();
}

template <typename T>
double L1AccuracyValidator<T>::calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference)
{
    ASSERT(actual.size() == reference.size());
    ASSERT(actual.size() != 0);
    double sum = 0.0;
    for (uint64_t i = 0; i < actual.size(); ++i)
    {
        sum += std::abs(static_cast<double>(actual[i]) - static_cast<double>(reference[i]));
    }
    return sum / actual.size();
}

template <typename T>
double L2AccuracyValidator<T>::calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference)
{
    ASSERT(actual.size() == reference.size());
    ASSERT(actual.size() != 0);
    double sum = 0.0;
    for (uint64_t i = 0; i < actual.size(); ++i)
    {
        double diff = static_cast<double>(actual[i]) - static_cast<double>(reference[i]);
        sum += diff * diff;
    }
    return sum / actual.size();
}

template <typename T>
double LInfAccuracyValidator<T>::calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference)
{
    ASSERT(actual.size() == reference.size());
    ASSERT(actual.size() != 0);
    double maxDiff = 0.0;
    for (uint64_t i = 0; i < actual.size(); ++i)
    {
        double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(reference[i]));
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

template <typename T>
double CosineSimilarityValidator<T>::calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference)
{
    ASSERT(actual.size() == reference.size());
    ASSERT(actual.size() != 0);
    double dotProduct = 0.0;
    double normActual = 0.0;
    double normRef = 0.0;
    for (uint64_t i = 0; i < actual.size(); ++i)
    {
        double a = static_cast<double>(actual[i]);
        double r = static_cast<double>(reference[i]);
        dotProduct += a * r;
        normActual += a * a;
        normRef += r * r;
    }
    double denominator = std::sqrt(normActual) * std::sqrt(normRef);
    if (denominator < 1e-12)
    {
        return 1.0; // Handle zero vectors
    }
    double cosineSim = dotProduct / denominator;
    return 1.0 - cosineSim; // Return as cost (0 = perfect match)
}

// Explicit template instantiations for supported types
template class L0AccuracyValidator<float>;
template class L0AccuracyValidator<int32_t>;
template class L0AccuracyValidator<int8_t>;
template class L0AccuracyValidator<half_float::half>;

template class L1AccuracyValidator<float>;
template class L1AccuracyValidator<int32_t>;
template class L1AccuracyValidator<int8_t>;
template class L1AccuracyValidator<half_float::half>;

template class L2AccuracyValidator<float>;
template class L2AccuracyValidator<int32_t>;
template class L2AccuracyValidator<int8_t>;
template class L2AccuracyValidator<half_float::half>;

template class LInfAccuracyValidator<float>;
template class LInfAccuracyValidator<int32_t>;
template class LInfAccuracyValidator<int8_t>;
template class LInfAccuracyValidator<half_float::half>;

template class CosineSimilarityValidator<float>;
template class CosineSimilarityValidator<int32_t>;
template class CosineSimilarityValidator<int8_t>;
template class CosineSimilarityValidator<half_float::half>;

bool peekArg(int32_t argc, char** argv, char const* flag)
{
    auto const flagLen = std::strlen(flag);
    for (int32_t i = 1; i < argc; ++i)
    {
        if (argv[i] == nullptr)
        {
            continue;
        }
        // Match either bare flag (--continue) or flag=value (--tuneBuildRoutes=...).
        if (std::strncmp(argv[i], flag, flagLen) == 0 && (argv[i][flagLen] == '\0' || argv[i][flagLen] == '='))
        {
            return true;
        }
    }
    return false;
}

std::string buildShellQuotedCmdLine(int32_t argc, char** argv)
{
    std::string cmdLine;
    for (int32_t i = 0; i < argc; ++i)
    {
        if (i > 0)
        {
            cmdLine += " ";
        }
        std::string arg = argv[i];
        bool const needsQuoting = arg.find_first_of(" \t|[]{}()&;'\"\\") != std::string::npos;
        if (needsQuoting)
        {
            std::string escaped;
            for (char c : arg)
            {
                if (c == '\'')
                {
                    escaped += "'\\''";
                }
                else
                {
                    escaped += c;
                }
            }
            cmdLine += "'" + escaped + "'";
        }
        else
        {
            cmdLine += arg;
        }
    }
    return cmdLine;
}

//! \brief Resolve file paths in argv to absolute for cache storage.
//!
//! File-path flags that get resolved: --onnx=, --saveEngine=, --loadInputs=,
//! --loadRefOutputs=, --tuneBuildRouteFile=, --loadEngine=. All others are stored as-is.
//! --loadInputs and --loadRefOutputs have format "name:path,name:path" so each
//! path component is resolved separately.
namespace
{
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::vector<std::string> resolveArgvPaths(int32_t argc, char** argv)
{
    static std::vector<std::string> const kSIMPLE_PATH_FLAGS
        = {"--onnx=", "--saveEngine=", "--tuneBuildRouteFile=", "--loadEngine="};
    static std::vector<std::string> const kMAPPED_PATH_FLAGS = {"--loadInputs=", "--loadRefOutputs="};

    std::vector<std::string> result;
    for (int32_t i = 0; i < argc; ++i)
    {
        std::string arg(argv[i]);

        // Check simple path flags (--flag=path -> --flag=<absolute path>)
        bool resolved = false;
        for (auto const& prefix : kSIMPLE_PATH_FLAGS)
        {
            if (startsWith(arg, prefix))
            {
                result.push_back(prefix + resolveAbsolutePath(arg.substr(prefix.size())));
                resolved = true;
                break;
            }
        }
        if (resolved)
        {
            continue;
        }

        // Check mapped path flags (--flag=name:path,name:path -> resolve each path)
        for (auto const& prefix : kMAPPED_PATH_FLAGS)
        {
            if (startsWith(arg, prefix))
            {
                std::string value = arg.substr(prefix.size());
                // Split on ',' to get individual name:path pairs
                auto pairs = splitToStringVec(value, ',');
                std::string resolvedValue;
                for (uint64_t p = 0; p < pairs.size(); ++p)
                {
                    if (p > 0)
                    {
                        resolvedValue += ",";
                    }
                    // Split each pair on ':' to separate name from path
                    auto nameAndPath = splitToStringVec(pairs[p], ':', 1);
                    if (nameAndPath.size() == 2)
                    {
                        resolvedValue += nameAndPath[0] + ":" + resolveAbsolutePath(nameAndPath[1]);
                    }
                    else
                    {
                        resolvedValue += pairs[p]; // Malformed pair, keep as-is
                    }
                }
                result.push_back(prefix + resolvedValue);
                resolved = true;
                break;
            }
        }
        if (resolved)
        {
            continue;
        }

        result.push_back(arg);
    }
    return result;
}
} // anonymous namespace

void writeTuningCacheHeader(std::string const& cacheFilePath, AllOptions const& options, int32_t argc, char** argv,
    std::string const& tunerVersion, std::string const& defaultBuildRoute)
{
    // Use ordered_json to preserve insertion order matching best_config.json.example:
    // tuner_version, accuracy_algorithm, accuracy_parameter, searching_algorithm,
    // command_line, default_build_route, tuning_expr, files, argv
    nlohmann::ordered_json header;

    header["tuner_version"] = tunerVersion;
    header["accuracy_algorithm"] = getAlgorithmName(options.inference.accuracyValidationAlgorithm);

    nlohmann::ordered_json accParam;
    accParam["atol"] = options.inference.atol;
    accParam["rtol"] = options.inference.rtol;
    accParam["epsilon"] = options.inference.accuracyThresholdEndToEnd;
    header["accuracy_parameter"] = accParam;

    header["searching_algorithm"] = toString(options.tuning.tuningSearchAlgorithm);

    // Reconstruct command line for reference, with shell-safe quoting for arguments
    // that contain spaces or metacharacters (e.g. --tuneBuildRoutes values).
    std::string cmdLine = buildShellQuotedCmdLine(argc, argv);
    header["command_line"] = cmdLine;
    header["default_build_route"] = defaultBuildRoute;

    // Store the expanded tuning expression. This is the already-expanded string
    // (handles --tuneBuildRouteFile case where the file may not exist at resume time).
    header["tuning_expr"] = options.tuning.tuningExpr;

    // Store absolute paths to all file-based options for human readability and
    // as a cross-check. The authoritative source for --continue reconstruction
    // is the "argv" field below.
    {
        nlohmann::ordered_json files;
        if (!options.model.baseModel.model.empty())
        {
            files["onnx"] = resolveAbsolutePath(options.model.baseModel.model);
        }
        if (!options.build.engine.empty())
        {
            files["save_engine"] = resolveAbsolutePath(options.build.engine);
        }
        // Input files: map of tensor_name → absolute path
        if (!options.inference.refPairs.empty())
        {
            nlohmann::ordered_json inputs;
            for (auto const& [name, path] : options.inference.refPairs[0].first)
            {
                inputs[name] = resolveAbsolutePath(path);
            }
            if (!inputs.empty())
            {
                files["inputs"] = inputs;
            }

            nlohmann::ordered_json refOutputs;
            for (auto const& [name, path] : options.inference.refPairs[0].second)
            {
                refOutputs[name] = resolveAbsolutePath(path);
            }
            if (!refOutputs.empty())
            {
                files["ref_outputs"] = refOutputs;
            }
        }
        header["files"] = files;
    }

    // Store argv with file-path arguments resolved to absolute paths.
    // This is the machine-readable source of truth for --continue reconstruction.
    // When resuming, the stored argv is replayed to reconstruct all options
    // (--iterations, --duration, --fp16, etc.) without enumerating each one.
    {
        auto resolvedArgv = resolveArgvPaths(argc, argv);
        nlohmann::ordered_json argvArray(resolvedArgv);
        header["argv"] = argvArray;
    }

    std::ofstream file(cacheFilePath, std::ios::trunc);
    if (!file)
    {
        sample::gLogError << "Cannot open tuning cache file for writing header: " << cacheFilePath << std::endl;
        return;
    }
    file << header.dump() << std::endl;
}

void writeTuningCacheIteration(std::string const& cacheFilePath, uint64_t iter, std::string const& buildRoute,
    bool crashed, std::string const& errorMessage, std::unordered_map<std::string, double> const& accuracyLossValues,
    double gpuTimeMs)
{
    // Use ordered_json to preserve insertion order matching best_config.json.example:
    // iter, build_route, crash, error_message, accuracy_loss, gpu_time
    nlohmann::ordered_json result;
    result[tuningCache::kIter] = iter;
    result[tuningCache::kBuildRoute] = buildRoute;
    result[tuningCache::kCrash] = crashed;
    result[tuningCache::kErrorMessage] = errorMessage;

    // accuracy_loss is a per-output map: {"output_name": accuracy_value, ...}
    // When crashed, accuracy values are unavailable so we write null.
    if (crashed || accuracyLossValues.empty())
    {
        result[tuningCache::kAccuracyLoss] = nullptr;
    }
    else
    {
        nlohmann::ordered_json accMap;
        for (auto const& [name, value] : accuracyLossValues)
        {
            accMap[name] = value;
        }
        result[tuningCache::kAccuracyLoss] = accMap;
    }
    result[tuningCache::kGpuTime] = crashed ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(gpuTimeMs);

    std::ofstream file(cacheFilePath, std::ios::app);
    if (!file)
    {
        sample::gLogError << "Cannot open tuning cache file to append iteration " << iter << ": " << cacheFilePath
                          << std::endl;
        return;
    }
    file << result.dump() << std::endl;
}

std::vector<std::string> reconstructArgvFromCacheHeader(
    TuningCacheHeader const& header, std::string const& currentExePath, std::string const& cacheFilePath)
{
    std::vector<std::string> newArgv;

    // Use current executable path as argv[0], not the one stored in the cache
    // (the binary may have been rebuilt or moved since the original run).
    newArgv.push_back(currentExePath);

    // Iterate over stored argv (skip stored argv[0]).
    for (uint64_t i = 1; i < header.argv.size(); ++i)
    {
        std::string const& arg = header.argv[i];

        // Replace --tuneBuildRoutes or --tuneBuildRouteFile with the stored tuning_expr.
        // This handles the case where --tuneBuildRouteFile was used originally but the
        // file no longer exists — the expanded expression is stored in tuning_expr.
        if (startsWith(arg, "--tuneBuildRoutes=") || startsWith(arg, "--tuneBuildRouteFile="))
        {
            continue; // Will be re-added below with the stored tuning_expr.
        }

        // Remove --continue and --tuningCacheFile from the stored argv to avoid
        // recursion (the stored run may itself have been a --continue run).
        if (arg == "--continue" || startsWith(arg, "--tuningCacheFile="))
        {
            continue;
        }

        newArgv.push_back(arg);
    }

    // Add back the tuning expression and cache file path.
    newArgv.push_back("--tuneBuildRoutes=" + header.tuningExpr);
    newArgv.push_back("--tuningCacheFile=" + cacheFilePath);

    return newArgv;
}

std::string resolveAbsolutePath(std::string const& path)
{
    if (path.empty())
    {
        return path;
    }
#if defined(_WIN32)
    // On Windows, path resolution is not needed (tuning features are not supported on Windows).
    // Return the path unchanged so the code compiles.
    return path;
#else
    // POSIX realpath() resolves symlinks and relative components to an absolute path.
    // Returns nullptr if the file does not exist or another error occurs.
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved) != nullptr)
    {
        return std::string(resolved);
    }
    return path;
#endif
}

std::optional<TuningCacheHeader> readTuningCacheHeader(std::string const& cacheFilePath)
{
    std::ifstream file(cacheFilePath);
    if (!file.is_open())
    {
        return std::nullopt;
    }

    // First line is the JSON header.
    std::string headerLine;
    if (!std::getline(file, headerLine) || headerLine.empty())
    {
        return std::nullopt;
    }

    try
    {
        auto headerJson = nlohmann::json::parse(headerLine);

        TuningCacheHeader header;

        // Extract argv array → vector<string>
        if (headerJson.contains("argv") && headerJson["argv"].is_array())
        {
            for (auto const& elem : headerJson["argv"])
            {
                header.argv.push_back(elem.get<std::string>());
            }
        }
        else
        {
            // argv field is required for --continue reconstruction.
            sample::gLogError << "Tuning cache header missing 'argv' field" << std::endl;
            return std::nullopt;
        }

        // Extract tuning_expr string.
        if (headerJson.contains("tuning_expr") && headerJson["tuning_expr"].is_string())
        {
            header.tuningExpr = headerJson["tuning_expr"].get<std::string>();
        }
        else
        {
            sample::gLogError << "Tuning cache header missing 'tuning_expr' field" << std::endl;
            return std::nullopt;
        }

        // Count remaining non-empty lines as completed iterations.
        header.completedIterations = 0;
        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty())
            {
                ++header.completedIterations;
            }
        }

        return header;
    }
    catch (nlohmann::json::exception const& e)
    {
        sample::gLogError << "Failed to parse tuning cache header: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::vector<CachedIterationResult> readCachedIterationResults(std::string const& cacheFilePath, int64_t maxIterations)
{
    std::vector<CachedIterationResult> results;
    std::ifstream file(cacheFilePath);
    if (!file.is_open())
    {
        return results;
    }

    std::string line;
    // Skip header line.
    if (!std::getline(file, line))
    {
        return results;
    }

    // Read iteration lines, extracting crash and gpu_time fields.
    while (std::getline(file, line) && static_cast<int64_t>(results.size()) < maxIterations)
    {
        if (line.empty())
        {
            continue;
        }
        try
        {
            auto j = nlohmann::json::parse(line);
            CachedIterationResult r;
            r.crashed = j.value(tuningCache::kCrash, true);
            r.gpuTimeMs = j.contains(tuningCache::kGpuTime) && j[tuningCache::kGpuTime].is_number()
                ? j[tuningCache::kGpuTime].get<double>()
                : 0.0;
            results.push_back(r);
        }
        catch (nlohmann::json::exception const&)
        {
            // Malformed line — treat as crashed.
            results.push_back({true, 0.0});
        }
    }

    return results;
}

} // namespace sample
