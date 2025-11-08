/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef BERT_COMMON_H
#define BERT_COMMON_H

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "common/checkMacrosPlugin.h"
#include "common/cublasWrapper.h"
#include "common/plugin.h"
#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cuda_runtime_api.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#define TRT_UNUSED (void)

#define BERT_PRINT_DEBUG_MSG 0

#if BERT_PRINT_DEBUG_MSG
#define BERT_DEBUG_MSG(msg) (gLogVerbose << (msg) << std::endl)
#define BERT_DEBUG_VALUE(key, value) (gLogVerbose << key << value << std::endl)
#else
#define BERT_DEBUG_MSG(msg) TRT_UNUSED(msg)
#define BERT_DEBUG_VALUE(key, value)                                                                                   \
    TRT_UNUSED(key);                                                                                                   \
    TRT_UNUSED(value)
#endif

using half = __half;

constexpr uint32_t BDIM = 1; // batch dimension
constexpr uint32_t SDIM = 0; // seq len dimension
constexpr uint32_t HDIM = 2; // hidden dimension

constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_87 = 87;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;
constexpr int32_t kSM_100 = 100;
constexpr int32_t kSM_120 = 120;

// For full mask mode, we must produce the compressed mask format expected by the fused attention path. Currently, only
// two sequence lengths are supported. We hard code the sizes here.
// The number of threads per CTA: warps_m * warps_n * warps_k * 32;
constexpr size_t threadsPerCta128 = 2 * 2 * 32;
constexpr size_t threadsPerCta384 = 1 * 8 * 32;

// The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension: (s + 16*warps_m - 1)
// / (16*warps_m);
constexpr size_t xmmasM128 = 4;
constexpr size_t xmmasM384 = 24;

// Packed mask size per batch. Layout is XMMAS_M * THREADS_PER_CTA.
constexpr size_t unfusedMaskSize = 1;
constexpr size_t packedMaskSize64 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize96 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;

namespace nvinfer1
{
namespace pluginInternal
{
template <typename T>
struct CudaDeleter
{
    void operator()(T* buf)
    {
        PLUGIN_CUASSERT(cudaFree(buf));
    }
};

} // namespace pluginInternal
namespace plugin
{

namespace bert
{

//! \brief Checks if the first argument matches any of the list items.
//! \return True if v is a member of list.
template <typename TElem, typename Container = std::initializer_list<TElem>>
bool elem(TElem const& v, Container const& list)
{
    return std::any_of(std::begin(list), std::end(list), [&v](TElem const& t) { return t == v; });
}

inline int32_t getMHAMaskPackedSize(int32_t smVersion, nvinfer1::DataType dataType, int32_t sequenceLength)
{
    // this code must match EmbLayerNormPluginDynamic::getOutputDimensions in embLayerNormPlugin.cpp
    int32_t packedSize = unfusedMaskSize;
    bool const isSmOK = elem(smVersion, {kSM_75, kSM_80, kSM_86, kSM_87, kSM_89, kSM_90, kSM_100, kSM_120});
    bool isPrecisionOK = (dataType == nvinfer1::DataType::kINT8 || dataType == nvinfer1::DataType::kHALF);
    if (isSmOK && isPrecisionOK)
    {
        if (sequenceLength == 64)
        {
            packedSize = packedMaskSize64;
        }
        else if (sequenceLength == 96)
        {
            packedSize = packedMaskSize96;
        }
        else if (sequenceLength == 128)
        {
            packedSize = packedMaskSize128;
        }
        else if (sequenceLength == 384)
        {
            packedSize = packedMaskSize384;
        }
    }
    return packedSize;
}

inline uint32_t getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4:
    case nvinfer1::DataType::kE8M0: PLUGIN_FAIL("Element size is not implemented for sub-byte data-types");
    }
    return 0;
}

inline int64_t getWeightsSize(nvinfer1::Weights const& w, nvinfer1::DataType type)
{
    return w.count * getElementSize(type);
}

inline int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

//! Check if the hardware supports BERT Multi-Head Attention plugins
//! The plugin calls precompiled cubins (compiled from fmha_v2/xmma kernels)
//! that are SM-specific.
inline bool doesHwSupportBertMHAPlugin() noexcept
{
    int32_t device{-1};
    cudaGetDevice(&device);
    int32_t smMajor{0};
    int32_t smMinor{0};
    cudaDeviceGetAttribute(&smMajor, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&smMinor, cudaDevAttrComputeCapabilityMinor, device);
    int32_t smVersion = (smMajor << 4) | (smMinor);
    // Turing and above
    static constexpr int32_t kSM_TURING_HEX{0x75};
    static constexpr int32_t kSM_BLACKWELL_100_HEX{0xA0};
    static constexpr int32_t kSM_BLACKWELL_120_HEX{0xC0};
    static constexpr int32_t kSM_ORIN_HEX{0x87};
    bool isAuto = smVersion == kSM_ORIN_HEX;
    bool isSm100OrLower = smVersion >= kSM_TURING_HEX && smVersion <= kSM_BLACKWELL_100_HEX;
    bool isHardwareSupported = (isSm100OrLower || smVersion == kSM_BLACKWELL_120_HEX) && !isAuto;

    return isHardwareSupported;
}

template <typename IntType>
constexpr IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}
template <typename IntType>
constexpr IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <typename T>
inline T* deserToDev(char const*& buffer, size_t nbElem)
{
    void* dev{nullptr};
    const size_t len = sizeof(T) * nbElem;
    PLUGIN_CUASSERT(cudaMalloc(&dev, len));
    PLUGIN_CUASSERT(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return static_cast<T*>(dev);
}

template <typename T>
inline void serFromDev(char*& buffer, T const* data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    PLUGIN_CUASSERT(cudaMemcpy(buffer, static_cast<void const*>(data), len, cudaMemcpyDeviceToHost));
    buffer += len;
}

template <typename T>
inline T* devToDev(T const* data, size_t nbElem)
{
    void* dev{nullptr};
    const size_t len = sizeof(T) * nbElem;
    PLUGIN_CUASSERT(cudaMalloc(&dev, len));
    PLUGIN_CUASSERT(cudaMemcpy(dev, static_cast<void const*>(data), len, cudaMemcpyDeviceToDevice));
    return static_cast<T*>(dev);
}

template <typename T>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemm(nvinfer1::pluginInternal::cublasHandle_t handle,
    nvinfer1::pluginInternal::cublasOperation_t transa, nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m,
    int32_t n, int32_t k, const T alpha, T const* A, int32_t lda, T const* B, int32_t ldb, const T beta, T* C,
    int32_t ldc);

template <>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemm(nvinfer1::pluginInternal::cublasHandle_t handle,
    nvinfer1::pluginInternal::cublasOperation_t transa, nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m,
    int32_t n, int32_t k, float const alpha, float const* A, int32_t lda, float const* B, int32_t ldb, float const beta,
    float* C, int32_t ldc)
{
    nvinfer1::pluginInternal::CublasWrapper& wrapper = nvinfer1::pluginInternal::getCublasWrapper();
    return wrapper.cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemm(nvinfer1::pluginInternal::cublasHandle_t handle,
    nvinfer1::pluginInternal::cublasOperation_t transa, nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m,
    int32_t n, int32_t k, const half alpha, half const* A, int32_t lda, half const* B, int32_t ldb, const half beta,
    half* C, int32_t ldc)
{
    nvinfer1::pluginInternal::CublasWrapper& wrapper = nvinfer1::pluginInternal::getCublasWrapper();
    return wrapper.cublasHgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <typename T>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemmStridedBatchedEx(
    nvinfer1::pluginInternal::cublasHandle_t handle, nvinfer1::pluginInternal::cublasOperation_t transa,
    nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m, int32_t n, int32_t k, const T alpha, T const* A,
    int32_t lda, int64_t strideA, T const* B, int32_t ldb, int64_t strideB, const T beta, T* C, int32_t ldc,
    int64_t strideC, int32_t batchCount, nvinfer1::pluginInternal::cublasGemmAlgo_t algo);

template <>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemmStridedBatchedEx(
    nvinfer1::pluginInternal::cublasHandle_t handle, nvinfer1::pluginInternal::cublasOperation_t transa,
    nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m, int32_t n, int32_t k, float const alpha,
    float const* A, int32_t lda, int64_t strideA, float const* B, int32_t ldb, int64_t strideB, float const beta,
    float* C, int32_t ldc, int64_t strideC, int32_t batchCount, nvinfer1::pluginInternal::cublasGemmAlgo_t algo)
{
    nvinfer1::pluginInternal::CublasWrapper& wrapper = nvinfer1::pluginInternal::getCublasWrapper();
    return wrapper.cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
        CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC, batchCount, CUDA_R_32F, algo);
}

template <>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemmStridedBatchedEx(
    nvinfer1::pluginInternal::cublasHandle_t handle, nvinfer1::pluginInternal::cublasOperation_t transa,
    nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m, int32_t n, int32_t k, const half alpha,
    half const* A, int32_t lda, int64_t strideA, half const* B, int32_t ldb, int64_t strideB, const half beta, half* C,
    int32_t ldc, int64_t strideC, int32_t batchCount, nvinfer1::pluginInternal::cublasGemmAlgo_t algo)
{
    nvinfer1::pluginInternal::CublasWrapper& wrapper = nvinfer1::pluginInternal::getCublasWrapper();
    return wrapper.cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, &beta, C, CUDA_R_16F, ldc, strideC, batchCount, CUDA_R_16F, algo);
}

template <typename T>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemmStridedBatched(
    nvinfer1::pluginInternal::cublasHandle_t handle, nvinfer1::pluginInternal::cublasOperation_t transa,
    nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m, int32_t n, int32_t k, const T alpha, T const* A,
    int32_t lda, int64_t strideA, T const* B, int32_t ldb, int64_t strideB, const T beta, T* C, int32_t ldc,
    int64_t strideC, int32_t batchCount);

template <>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemmStridedBatched(
    nvinfer1::pluginInternal::cublasHandle_t handle, nvinfer1::pluginInternal::cublasOperation_t transa,
    nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m, int32_t n, int32_t k, float const alpha,
    float const* A, int32_t lda, int64_t strideA, float const* B, int32_t ldb, int64_t strideB, float const beta,
    float* C, int32_t ldc, int64_t strideC, int32_t batchCount)
{
    nvinfer1::pluginInternal::CublasWrapper& wrapper = nvinfer1::pluginInternal::getCublasWrapper();
    return wrapper.cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <>
nvinfer1::pluginInternal::cublasStatus_t inline cublasGemmStridedBatched(
    nvinfer1::pluginInternal::cublasHandle_t handle, nvinfer1::pluginInternal::cublasOperation_t transa,
    nvinfer1::pluginInternal::cublasOperation_t transb, int32_t m, int32_t n, int32_t k, const half alpha,
    half const* A, int32_t lda, int64_t strideA, half const* B, int32_t ldb, int64_t strideB, const half beta, half* C,
    int32_t ldc, int64_t strideC, int32_t batchCount)
{
    nvinfer1::pluginInternal::CublasWrapper& wrapper = nvinfer1::pluginInternal::getCublasWrapper();
    return wrapper.cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

struct CublasConfigHelper
{
    nvinfer1::pluginInternal::cublasPointerMode_t pm;
    nvinfer1::pluginInternal::cublasMath_t mm;
    nvinfer1::pluginInternal::cublasHandle_t cublas;
    nvinfer1::pluginInternal::CublasWrapper& wrapper = nvinfer1::pluginInternal::getCublasWrapper();
    CublasConfigHelper(nvinfer1::pluginInternal::cublasHandle_t cublas_)
        : cublas(cublas_)
    {
        PLUGIN_CUBLASASSERT(wrapper.cublasGetPointerMode(cublas, &pm));
        PLUGIN_CUBLASASSERT(wrapper.cublasGetMathMode(cublas, &mm));
        PLUGIN_CUBLASASSERT(wrapper.cublasSetPointerMode(cublas, nvinfer1::pluginInternal::CUBLAS_POINTER_MODE_HOST));
        PLUGIN_CUBLASASSERT(wrapper.cublasSetMathMode(cublas, nvinfer1::pluginInternal::CUBLAS_TENSOR_OP_MATH));
    }
    ~CublasConfigHelper()
    {
        wrapper.cublasSetMathMode(cublas, mm);
        wrapper.cublasSetPointerMode(cublas, pm);
    }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, pluginInternal::CudaDeleter<T>>;

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem)
{
    ptr.reset(static_cast<T*>(cudaMem), pluginInternal::CudaDeleter<T>());
}

struct WeightsWithOwnership : public nvinfer1::Weights
{
    WeightsWithOwnership()
    {
        values = nullptr;
        count = 0;
    }
    ~WeightsWithOwnership()
    {
        operator delete[](const_cast<void*>(values));
    }

    WeightsWithOwnership(WeightsWithOwnership const&) = delete;
    WeightsWithOwnership operator=(WeightsWithOwnership const&) = delete;
    WeightsWithOwnership(WeightsWithOwnership const&&) = delete;
    WeightsWithOwnership operator=(WeightsWithOwnership const&&) = delete;

    void convertAndCopy(nvinfer1::Weights const& src, nvinfer1::DataType type)
    {
        this->type = type;
        this->count = src.count;

        if (type == nvinfer1::DataType::kFLOAT)
        {
            auto destBuf = new float[src.count];
            this->values = destBuf;

            if (src.type == nvinfer1::DataType::kFLOAT)
            {
                BERT_DEBUG_MSG("Float Weights(Host) => Float Array(Host)");
                std::copy_n(static_cast<float const*>(src.values), src.count, destBuf);
            }
            else
            {
                PLUGIN_ASSERT(src.type == nvinfer1::DataType::kHALF);

                BERT_DEBUG_MSG("Half Weights(Host) => Float Array(Host)");
                auto const s = static_cast<half const*>(src.values);
                auto d = static_cast<float*>(const_cast<void*>(this->values));

                for (auto it = 0; it < src.count; it++)
                {
                    d[it] = __half2float(s[it]);
                }
            }
        }
        else if (type == nvinfer1::DataType::kHALF)
        {
            auto destBuf = new half[src.count];
            this->values = destBuf;

            if (src.type == nvinfer1::DataType::kHALF)
            {
                BERT_DEBUG_MSG("Half Weights(Host) => Half Array(Host)");
                std::copy_n(static_cast<half const*>(src.values), src.count, destBuf);
            }
            else
            {
                PLUGIN_ASSERT(src.type == nvinfer1::DataType::kFLOAT);

                BERT_DEBUG_MSG("Float Weights(Host) => Half Array(Host)");
                auto const s = static_cast<float const*>(src.values);
                auto d = static_cast<half*>(const_cast<void*>(this->values));

                for (auto it = 0; it < src.count; it++)
                {
                    d[it] = __float2half(s[it]);
                }
            }
        }
        else
        {
            throw std::runtime_error("Unsupported DataType specified for plugin.");
        }
    }

    void convertAndCopy(char const*& srcBuf, size_t count, nvinfer1::DataType type) noexcept
    {
        this->type = type;
        this->count = count;
        auto const nbBytes = getWeightsSize(*this, type);
        auto destBuf = new char[nbBytes];
        this->values = destBuf;

        std::copy_n(srcBuf, nbBytes, destBuf);
        srcBuf += nbBytes;
    }
};

template <typename T>
inline void copyToDevice(WeightsWithOwnership& hostWeights, size_t nbBytes, cuda_unique_ptr<T>& cudaWeights)
{
    if (hostWeights.values)
    {
        void* cudaMem{nullptr};
        PLUGIN_CUASSERT(cudaMalloc(&cudaMem, nbBytes));
        PLUGIN_CUASSERT(cudaMemcpy(cudaMem, hostWeights.values, nbBytes, cudaMemcpyHostToDevice));
        cudaWeights.reset(static_cast<T*>(cudaMem));
    }
}

inline void convertAndCopyToDevice(nvinfer1::Weights const& src, float* destDev)
{

    size_t wordSize = sizeof(float);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kFLOAT)
    {
        BERT_DEBUG_MSG("Float Weights(Host) => Float Array(Device)");
        PLUGIN_CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        BERT_DEBUG_MSG("Half Weights(Host) => Float Array(Device)");
        std::vector<float> tmp(src.count);
        half const* values = reinterpret_cast<half const*>(src.values);

        for (size_t it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __half2float(values[it]);
        }

        PLUGIN_CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline void convertAndCopyToDevice(nvinfer1::Weights const& src, half* destDev)
{
    size_t wordSize = sizeof(half);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kHALF)
    {
        BERT_DEBUG_MSG("Half Weights(Host) => Half Array(Device)");
        PLUGIN_CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        BERT_DEBUG_MSG("Float Weights(Host) => Half Array(Device)");
        std::vector<half> tmp(src.count);
        float const* values = reinterpret_cast<float const*>(src.values);

        for (size_t it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __float2half(values[it]);
        }
        PLUGIN_CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype)
{
    switch (ftype)
    {
    case nvinfer1::PluginFieldType::kFLOAT32:
    {
        BERT_DEBUG_MSG("PluginFieldType is Float32");
        return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16:
    {
        BERT_DEBUG_MSG("PluginFieldType is Float16");
        return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32:
    {
        BERT_DEBUG_MSG("PluginFieldType is Int32");
        return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8:
    {
        BERT_DEBUG_MSG("PluginFieldType is Int8");
        return nvinfer1::DataType::kINT8;
    }
    default: throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // BERT_COMMON_H

#endif // CUDA_VERSION >= 10010
