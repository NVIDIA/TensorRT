/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef BERT_COMMON_H
#define BERT_COMMON_H

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "plugin.h"
#include "pluginLogger.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#define TRT_UNUSED (void)

using half = __half;

constexpr uint32_t BDIM = 1; // batch dimension
constexpr uint32_t SDIM = 0; // seq len dimension
constexpr uint32_t HDIM = 2; // hidden dimension

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t getWeightsSize(const nvinfer1::Weights& w, nvinfer1::DataType type)
{
    return w.count * getElementSize(type);
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

namespace bert
{

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
inline T* deserToDev(const char*& buffer, size_t nbElem)
{
    void* dev{nullptr};
    const size_t len = sizeof(T) * nbElem;
    CUASSERT(cudaMalloc(&dev, len));
    CUASSERT(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return static_cast<T*>(dev);
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    CUASSERT(cudaMemcpy(buffer, static_cast<const void*>(data), len, cudaMemcpyDeviceToHost));
    buffer += len;
}

template <typename T>
inline T* devToDev(const T* data, size_t nbElem)
{
    void* dev{nullptr};
    const size_t len = sizeof(T) * nbElem;
    CUASSERT(cudaMalloc(&dev, len));
    CUASSERT(cudaMemcpy(dev, static_cast<const void*>(data), len, cudaMemcpyDeviceToDevice));
    return static_cast<T*>(dev);
}

template <typename T>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const T alpha, const T* A, int lda, const T* B, int ldb, const T beta, T* C, int ldc);

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C,
    int ldc)
{

    return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const half alpha, const half* A, int lda, const half* B, int ldb, const half beta, half* C, int ldc)
{
    return cublasHgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount,
    cublasGemmAlgo_t algo);

template <>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
    int batchCount, cublasGemmAlgo_t algo)
{

    return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
        CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC, batchCount, CUDA_R_32F, algo);
}

template <>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
    const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
    int batchCount, cublasGemmAlgo_t algo)
{
    return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, &beta, C, CUDA_R_16F, ldc, strideC, batchCount, CUDA_R_16F, algo);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount);

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
    int batchCount)
{

    return cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
    const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
    int batchCount)
{
    return cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

struct CublasConfigHelper
{
    cublasPointerMode_t pm;
    cublasMath_t mm;
    cublasHandle_t cublas;
    CublasConfigHelper(cublasHandle_t cublas_)
        : cublas(cublas_)
    {
        cublasGetPointerMode(cublas, &pm);
        cublasGetMathMode(cublas, &mm);
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
        cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
    }
    ~CublasConfigHelper()
    {
        cublasSetMathMode(cublas, mm);
        cublasSetPointerMode(cublas, pm);
    }
};

template <typename T>
struct CudaDeleter
{
    void operator()(T* buf)
    {
        CUASSERT(cudaFree(buf));
    }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, bert::CudaDeleter<T>>;

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem)
{
    ptr.reset(static_cast<T*>(cudaMem), bert::CudaDeleter<T>());
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

    WeightsWithOwnership(const WeightsWithOwnership&) = delete;
    WeightsWithOwnership operator=(const WeightsWithOwnership&) = delete;
    WeightsWithOwnership(const WeightsWithOwnership&&) = delete;
    WeightsWithOwnership operator=(const WeightsWithOwnership&&) = delete;

    void convertAndCopy(const nvinfer1::Weights& src, nvinfer1::DataType type)
    {
        this->type = type;
        this->count = src.count;

        if (type == nvinfer1::DataType::kFLOAT)
        {
            auto destBuf = new float[src.count];
            this->values = destBuf;

            if (src.type == nvinfer1::DataType::kFLOAT)
            {
                gLogVerbose << "Float Weights(Host) => Float Array(Host)\n";
                std::copy_n(static_cast<const float*>(src.values), src.count, destBuf);
            }
            else
            {
                assert(src.type == nvinfer1::DataType::kHALF);

                gLogVerbose << "Half Weights(Host) => Float Array(Host)\n";
                const auto s = static_cast<const half*>(src.values);
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
                gLogVerbose << "Half Weights(Host) => Half Array(Host)\n";
                std::copy_n(static_cast<const half*>(src.values), src.count, destBuf);
            }
            else
            {
                assert(src.type == nvinfer1::DataType::kFLOAT);

                gLogVerbose << "Float Weights(Host) => Half Array(Host)\n";
                const auto s = static_cast<const float*>(src.values);
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

    void convertAndCopy(const char*& srcBuf, size_t count, nvinfer1::DataType type)
    {
        this->type = type;
        this->count = count;
        const auto nbBytes = getWeightsSize(*this, type);
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
        CUASSERT(cudaMalloc(&cudaMem, nbBytes));
        CUASSERT(cudaMemcpy(cudaMem, hostWeights.values, nbBytes, cudaMemcpyHostToDevice));
        cudaWeights.reset(static_cast<T*>(cudaMem));
    }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, float* destDev)
{

    size_t wordSize = sizeof(float);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kFLOAT)
    {
        gLogVerbose << "Float Weights(Host) => Float Array(Device)" << std::endl;
        CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        gLogVerbose << "Half Weights(Host) => Float Array(Device)" << std::endl;
        std::vector<float> tmp(src.count);
        const half* values = reinterpret_cast<const half*>(src.values);

        for (size_t it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __half2float(values[it]);
        }

        CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, half* destDev)
{
    size_t wordSize = sizeof(half);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kHALF)
    {
        gLogVerbose << "Half Weights(Host) => Half Array(Device)" << std::endl;
        CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        gLogVerbose << "Float Weights(Host) => Half Array(Device)" << std::endl;
        std::vector<half> tmp(src.count);
        const float* values = reinterpret_cast<const float*>(src.values);

        for (size_t it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __float2half(values[it]);
        }
        CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype)
{
    switch (ftype)
    {
    case nvinfer1::PluginFieldType::kFLOAT32:
    {
        gLogVerbose << "PluginFieldType is Float32" << std::endl;
        return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16:
    {
        gLogVerbose << "PluginFieldType is Float16" << std::endl;
        return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32:
    {
        gLogVerbose << "PluginFieldType is Int32" << std::endl;
        return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8:
    {
        gLogVerbose << "PluginFieldType is Int8" << std::endl;
        return nvinfer1::DataType::kINT8;
    }
    default: throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

} // namespace bert
#endif // BERT_COMMON_H

#endif // CUDA_VERSION >= 10010
