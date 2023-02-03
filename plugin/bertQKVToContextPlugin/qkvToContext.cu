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

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/serialize.hpp"
#include "qkvToContextPlugin.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

#include "bertQKVToContextPlugin/fused_multihead_attention_v2/include/fused_multihead_attention_v2.h"
using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{
inline uint32_t asUInt32(float const& val)
{
    return *reinterpret_cast<uint32_t const*>(reinterpret_cast<void const*>(&val));
}

template <typename T, int TPB, int VPT>
__global__ void maskedSoftmax(const float rsqrtHeadSize, const T* input, T* output, const int* maskIdx)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;

    union SMem
    {
        T shm[VPT * TPB];
        typename BlockReduce::TempStorage reduce;
        SMem() {}
    };
    __shared__ SMem tmp;

    // grid: (NxS, B)
    const int b = blockIdx.y;
    const int blockOffset = (b * gridDim.x + blockIdx.x) * TPB;
    __shared__ int lastValid;
    if (threadIdx.x == 0)
    {
        lastValid = min(TPB, maskIdx[b]);
    }
    __syncthreads();
    float local[VPT];

    __shared__ float rZ;
    __shared__ float fMax[VPT];

    const int idx = (blockOffset + threadIdx.x) * VPT;
    T* myshm = &tmp.shm[threadIdx.x * VPT];
    copy<sizeof(T) * VPT>(&input[idx], myshm);

    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = (threadIdx.x < lastValid) ? float(tmp.shm[it * TPB + threadIdx.x]) : -FLT_MAX;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        float maxElem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
        if (threadIdx.x == 0)
        {
            fMax[it] = maxElem;
        }
        __syncthreads();
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = (threadIdx.x < lastValid) ? myExp<float>(rsqrtHeadSize * (local[it] - fMax[it])) : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

        if (threadIdx.x == 0)
        {
            rZ = (1.f) / Z;
        }
        __syncthreads();
        local[it] *= rZ;
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        tmp.shm[it * TPB + threadIdx.x] = local[it];
    }
    __syncthreads();
    copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, int TPB, int VPT>
__global__ void softmax(const float rsqrtHeadSize, const T* input, T* output)
{
    float local[VPT];

    using BlockReduce = cub::BlockReduce<float, TPB>;

    union SMem
    {
        T shm[VPT * TPB];
        typename BlockReduce::TempStorage reduce;
        SMem() {}
    };
    __shared__ SMem tmp;

    __shared__ float rZ;
    __shared__ float fMax[VPT];

    const int idx = (TPB * blockIdx.x + threadIdx.x) * VPT;
    T* myshm = &tmp.shm[threadIdx.x * VPT];
    copy<sizeof(T) * VPT>(&input[idx], myshm);

    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = float(tmp.shm[it * TPB + threadIdx.x]);
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        float maxElem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
        if (threadIdx.x == 0)
        {
            fMax[it] = maxElem;
        }
        __syncthreads();
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = myExp<float>(rsqrtHeadSize * (local[it] - fMax[it]));
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {

        const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

        if (threadIdx.x == 0)
        {
            rZ = 1.f / Z;
        }
        __syncthreads();
        local[it] *= rZ;
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        tmp.shm[it * TPB + threadIdx.x] = local[it];
    }
    __syncthreads();
    copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, unsigned TPB>
__global__ void scaledSoftmaxKernelSmall(const int ld, const float rsqrtHeadSize, const T* input, T* output)
{
    scaledSoftmaxSmall<T, TPB>(ld, ld, rsqrtHeadSize, input, output);
}

template <typename T, unsigned TPB>
__global__ void scaledSoftmaxKernel(const int ld, const float rsqrtHeadSize, const T* input, T* output)
{
    scaledSoftmax<T, TPB>(ld, ld, rsqrtHeadSize, input, output);
}

template <typename T>
int computeScaledSoftmax(
    cudaStream_t stream, const int ld, const int B, const int N, const float rsqrtHeadSize, const T* input, T* output)
{

    constexpr int VPT = 16 / sizeof(T);

    const dim3 grid(ld * N, B, 1);

    if (ld <= 32)
    {
        const int blockSize = 32;
        scaledSoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
    }
    else if (ld < 128)
    {
        const int blockSize = 128;
        scaledSoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
    }
    else if (ld == 128)
    {
        const int grid = B * N * ld / (VPT);
        softmax<T, 128, VPT><<<grid, 128, 0, stream>>>(rsqrtHeadSize, input, output);
    }

    else if (ld == 384)
    {
        const int grid = B * N * ld / (VPT);
        softmax<T, 384, VPT><<<grid, 384, 0, stream>>>(rsqrtHeadSize, input, output);
    }
    else
    {
        const int blockSize = 256;

        scaledSoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
    }

    PLUGIN_CHECK(cudaPeekAtLastError());
    return 0;
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernelSmall(
    const int ld, const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output)
{
    __shared__ int lastValid;

    if (threadIdx.x == 0)
    {
        lastValid = min(ld, maskIdx[blockIdx.y]);
    }
    __syncthreads();

    scaledSoftmaxSmall<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernel(
    const int ld, const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output)
{

    __shared__ int lastValid;

    if (threadIdx.x == 0)
    {
        lastValid = min(ld, maskIdx[blockIdx.y]);
    }
    __syncthreads();
    scaledSoftmax<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T>
int computeMaskedScaledSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrtHeadSize,
    const int* maskIdx, const T* input, T* output)
{
    // Mask idx is of length B and assumes the valid region is contiguous starting
    // from the beginning of the sequence

    const dim3 grid(ld * N, B, 1);
    // for smaller problems, e.g. BERT base B=1, this is not optimal
    if (ld <= 32)
    {
        constexpr int blockSize = 32;
        maskedScaledSoftmaxKernelSmall<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
    }
    else if (ld < 128)
    {
        constexpr int blockSize = 128;
        maskedScaledSoftmaxKernelSmall<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
    }
    else if (ld == 128)
    {
        if (B == 1)
        {
            constexpr int VPT = 4 / sizeof(T);
            constexpr int blockSize = 128;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
        else
        {
            constexpr int VPT = 16 / sizeof(T);
            constexpr int blockSize = 128;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
    }
    else if (ld == 384)
    {
        if (B == 1)
        {
            constexpr int VPT = 4 / sizeof(T);
            constexpr int blockSize = 384;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
        else
        {
            constexpr int VPT = 16 / sizeof(T);
            constexpr int blockSize = 384;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
    }
    else
    {
        constexpr int blockSize = 256;
        maskedScaledSoftmaxKernel<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
    }

    PLUGIN_CHECK(cudaPeekAtLastError());
    return 0;
}

std::pair<int, int> tuneBatchedGemm(
    const int B, const int S, const int numHeads, const int headSize, const int smVersion)
{
    const int nruns = 500;
    cublasHandle_t cublas;
    PLUGIN_CUBLASASSERT(cublasCreate(&cublas));
    cudaStream_t stream;
    PLUGIN_CUASSERT(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    PLUGIN_CUASSERT(cudaEventCreate(&start));
    PLUGIN_CUASSERT(cudaEventCreate(&stop));
    PLUGIN_CUBLASASSERT(cublasSetStream(cublas, stream));
    PLUGIN_CUBLASASSERT(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

    using T = half;
    const int omatSize = S * S;
    const int numMats = B * numHeads;
    const int ldQKV = 3 * B * numHeads * headSize;
    const int strideQKV = 3 * headSize;
    const int ldOut = B * numHeads * headSize;
    const int strideOut = headSize;

    const size_t inBytes = S * B * 3 * numHeads * headSize * sizeof(T);
    const size_t qkBytes = S * S * B * numHeads * sizeof(T);
    const size_t outBytes = S * B * numHeads * headSize * sizeof(T);

    T* input = nullptr;
    T* qkptr = nullptr;
    T* output = nullptr;
    PLUGIN_CUASSERT(cudaMalloc(&input, inBytes));
    PLUGIN_CUASSERT(cudaMalloc(&qkptr, qkBytes));
    PLUGIN_CUASSERT(cudaMalloc(&output, outBytes));
    PLUGIN_CUASSERT(cudaMemset(input, 1, inBytes));
    PLUGIN_CUASSERT(cudaMemset(qkptr, 1, qkBytes));

    // input: SxBx3xNxH
    const T* qptr = input;
    const T* kptr = qptr + headSize;
    const T* vptr = kptr + headSize;

    const int startAlgo = (int) CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    const int endAlgo = (int) CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int best1 = startAlgo;
    int best2 = startAlgo;
    float ms1 = 1000000;
    float ms2 = 1000000;

    PLUGIN_ASSERT(smVersion >= kSM_53);
    for (int a = startAlgo; a <= endAlgo; a++)
    {
        cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(a);
        float ms1_, ms2_;
        // qkptr: BxNxSxS
        PLUGIN_CUASSERT(cudaEventRecord(start, stream));
        for (int r = 0; r < nruns; r++)
        {
            PLUGIN_CUBLASASSERT(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, headSize, T(1.f),
                kptr, ldQKV, strideQKV, qptr, ldQKV, strideQKV, T(0.f), qkptr, S, omatSize, numMats, algo));
        }

        PLUGIN_CUASSERT(cudaEventRecord(stop, stream));
        PLUGIN_CUASSERT(cudaStreamSynchronize(stream));
        PLUGIN_CUASSERT(cudaEventElapsedTime(&ms1_, start, stop));
        if (ms1_ < ms1)
        {
            best1 = algo;
            ms1 = ms1_;
        }

        // pptr: BxNxSxS
        // output: SxBxNxH
        PLUGIN_CUASSERT(cudaEventRecord(start, stream));
        for (int r = 0; r < nruns; r++)
        {
            PLUGIN_CUBLASASSERT(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, headSize, S, S, 1.f,
                vptr, ldQKV, strideQKV, qkptr, S, omatSize, 0.f, output, ldOut, strideOut, numMats, algo));
        }

        PLUGIN_CUASSERT(cudaEventRecord(stop, stream));
        PLUGIN_CUASSERT(cudaStreamSynchronize(stream));
        PLUGIN_CUASSERT(cudaEventElapsedTime(&ms2_, start, stop));

        if (ms2_ < ms2)
        {
            best2 = algo;
            ms2 = ms2_;
        }
    }

    PLUGIN_CUASSERT(cudaFree(input));
    PLUGIN_CUASSERT(cudaFree(qkptr));
    PLUGIN_CUASSERT(cudaFree(output));
    PLUGIN_CUASSERT(cudaEventDestroy(start));
    PLUGIN_CUASSERT(cudaEventDestroy(stop));
    PLUGIN_CUASSERT(cudaStreamDestroy(stream));
    PLUGIN_CUBLASASSERT(cublasDestroy(cublas));
    return std::make_pair(best1, best2);
}

template int computeScaledSoftmax<float>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const float* input, float* output);
template int computeScaledSoftmax<half>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const half* input, half* output);

template int computeMaskedScaledSoftmax<float>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const int* maskIdx, const float* input, float* output);
template int computeMaskedScaledSoftmax<half>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const int* maskIdx, const half* input, half* output);

size_t MHARunner::getSerializationSize() const noexcept
{
    return sizeof(mS) + sizeof(mB);
}

void MHARunner::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mB);
}

void MHARunner::deserialize(const void* data, size_t length)
{
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);
    setup(mS, mB);
}

UnfusedMHARunner::UnfusedMHARunner(const nvinfer1::DataType type, const int numHeads, const int headSize, const int sm)
    : MHARunner(type, numHeads, headSize)
    , mIsBestAlgoFound(false)
    , mAlgoBatchedEx1(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    , mAlgoBatchedEx2(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    , mSm(sm)
{
    PLUGIN_CUBLASASSERT(cublasCreate(&mCublas));
}

UnfusedMHARunner::~UnfusedMHARunner()
{
    PLUGIN_CUBLASASSERT(cublasDestroy(mCublas));
}

size_t UnfusedMHARunner::getSerializationSize() const noexcept
{
    return sizeof(mAlgoBatchedEx1) + sizeof(mAlgoBatchedEx2) + MHARunner::getSerializationSize();
}

void UnfusedMHARunner::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mAlgoBatchedEx1);
    serialize_value(&buffer, mAlgoBatchedEx2);
    MHARunner::serialize(buffer);
}

void UnfusedMHARunner::deserialize(const void* data, size_t length)
{
    mIsBestAlgoFound = true;
    deserialize_value(&data, &length, &mAlgoBatchedEx1);
    deserialize_value(&data, &length, &mAlgoBatchedEx2);
    MHARunner::deserialize(data, length);
}

void UnfusedMHARunner::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    if (mType == DataType::kHALF && !mIsBestAlgoFound)
    {
        std::tie(mAlgoBatchedEx1, mAlgoBatchedEx2) = tuneBatchedGemm(B, S, mNumHeads, mHeadSize, mSm);
        mIsBestAlgoFound = true;

        BERT_DEBUG_VALUE("QKV Plugin - Selected Algo 1 for batch gemms: ", mAlgoBatchedEx1);
        BERT_DEBUG_VALUE("QKV Plugin - Selected Algo 2 for batch gemms: ", mAlgoBatchedEx2);
    }
}

size_t UnfusedMHARunner::getWorkspaceSize() const
{
    return 2UL * mWordSize * mOmatSize * mNumMats;
}

void UnfusedMHARunner::run(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    this->run(inputDesc[0], outputDesc[0], inputs[0], inputs[1], outputs[0], workspace, stream);
}

void UnfusedMHARunner::run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc, const void* qkvPtr,
    const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
{
    const int* maskIdx = static_cast<const int*>(maskPtr);

    PLUGIN_CUBLASASSERT(cublasSetStream(mCublas, stream));

    // Q, K, V: BxNxSxH (inputs)
    // Q * K': BxNxSxS (-> scratch1)
    // P: BxNxSxS (-> scratch2)
    // P * V: BxNxSxH (output)

    if (mType == DataType::kHALF)
    {
        CublasConfigHelper helper(mCublas);
        const half* qptr = static_cast<const half*>(qkvPtr);
        const half* kptr = qptr + mHeadSize;
        const half* vptr = kptr + mHeadSize;
        half* qkptr = static_cast<half*>(workspace);
        half* pptr = qkptr + mOmatSize * mNumMats;
        half alpha = 1.f;
        half beta = 0.f;
        PLUGIN_CUBLASASSERT(::cublasGemmStridedBatchedEx(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mS, mS, mHeadSize, &alpha,
            kptr, CUDA_R_16F, mLdQKV, mStrideQKV, qptr, CUDA_R_16F, mLdQKV, mStrideQKV, &beta, qkptr, CUDA_R_16F, mS,
            mOmatSize, mNumMats, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx1)));

        // apply softmax
        if (maskIdx)
        { // if we have a mask
            computeMaskedScaledSoftmax<half>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, maskIdx, qkptr, pptr);
        }
        else
        { // if we don't have a mask
            computeScaledSoftmax<half>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, qkptr, pptr);
        }

        // compute P*V (as V*P)
        PLUGIN_CUBLASASSERT(cublasGemmStridedBatchedEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, mS, mS, &alpha,
            vptr, CUDA_R_16F, mLdQKV, mStrideQKV, pptr, CUDA_R_16F, mS, mOmatSize, &beta, output, CUDA_R_16F, mLdOut,
            mStrideOut, mNumMats, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx2)));
    }
    else
    {

        const float* qptr = static_cast<const float*>(qkvPtr);
        const float* kptr = qptr + mHeadSize;
        const float* vptr = kptr + mHeadSize;
        float* qkptr = static_cast<float*>(workspace);
        float* pptr = qkptr + mOmatSize * mNumMats;
        float* outptr = static_cast<float*>(output);
        PLUGIN_CUBLASASSERT(cublasGemmStridedBatched<float>(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mS, mS, mHeadSize, 1.f,
            kptr, mLdQKV, mStrideQKV, qptr, mLdQKV, mStrideQKV, 0.f, qkptr, mS, mOmatSize, mNumMats));

        // apply softmax
        if (maskIdx)
        { // if we have a mask
            computeMaskedScaledSoftmax<float>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, maskIdx, qkptr, pptr);
        }
        else
        { // if we don't have a mask
            computeScaledSoftmax<float>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, qkptr, pptr);
        }

        PLUGIN_CUBLASASSERT(cublasGemmStridedBatched<float>(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, mS, mS, 1.f,
            vptr, mLdQKV, mStrideQKV, pptr, mS, mOmatSize, 0.f, outptr, mLdOut, mStrideOut, mNumMats));
    }
}

bool UnfusedMHARunner::isValid(int s) const
{
    return mType != DataType::kINT8;
}

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        half2 h2 = __float2half2_rn(norm);
        alpha = reinterpret_cast<const uint32_t&>(h2);
    }
    else if (dtype == DATA_TYPE_FP32)
    {
        alpha = reinterpret_cast<const uint32_t&>(norm);
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<const uint32_t&>(inorm);
    }
    else
    {
        assert(false);
    }
}

class FusedMHARunnerFP16::mhaImpl
{
public:
    mhaImpl(FusedMHARunnerFP16* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernels(DATA_TYPE_FP16, sm))
        , xmmas_m(0U)
        , xmmas_n(0U)
        , threads_per_cta(1U)
    {
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        // check that we initialized
        assert(xmmas_m > 0);
        assert(threads_per_cta > 0);
        assert(interface->mB > 0);
        return interface->mB * xmmas_m * threads_per_cta * sizeof(uint32_t);
    }

    void setup(const int S, const int B)
    {
        // TODO these implementation details might be better centralized into the XMMA code, since they are needed in
        // several places (also outside of this plugin)
        size_t warps_m{1U};
        size_t warps_n{1U};
        size_t warps_k{1U};
        if (S == 64 || S == 96 || S == 128)
        {
            warps_m = 2;
            warps_n = 2;
        }
        else if (S == 384)
        {
            warps_m = 1;
            warps_n = 8;
        }
        else
        {
            assert(false && "Unsupporte seqlen");
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

        const float scale_bmm1 = interface->mRsqrtHeadSize;
        const float scale_softmax = 1.f; // Seems to be only required for int8
        const float scale_bmm2 = 1.f;

        Data_type scale_type = DATA_TYPE_FP16;
        set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;

        params.qkv_stride_in_bytes = get_size_in_bytes(interface->mLdQKV, DATA_TYPE_FP16);
        params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
        params.o_stride_in_bytes = get_size_in_bytes(interface->mLdOut, DATA_TYPE_FP16);
    }

    void run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc, const void* qkvPtr,
        const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
    {
        params.qkv_ptr = const_cast<void*>(qkvPtr);

        params.packed_mask_ptr = const_cast<void*>(maskPtr);

        params.o_ptr = output;

        xmmaKernel->run(params, stream);

        PLUGIN_CHECK(cudaPeekAtLastError());
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

private:
    FusedMHARunnerFP16* interface;
    Fused_multihead_attention_params params;
    int sm;
    const FusedMultiHeadAttentionXMMAKernel* xmmaKernel;
    size_t xmmas_m;
    size_t xmmas_n;
    size_t threads_per_cta;
};

FusedMHARunnerFP16::FusedMHARunnerFP16(const int numHeads, const int headSize, const int sm)
    : MHARunner(DataType::kHALF, numHeads, headSize)
    , mSm(sm)
    , pimpl(new mhaImpl(this))
{
}

void FusedMHARunnerFP16::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    pimpl->setup(S, B);
}

size_t FusedMHARunnerFP16::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerFP16::deserialize(const void* data, size_t length)
{
    MHARunner::deserialize(data, length);
    setup(mS, mB);
}

void FusedMHARunnerFP16::run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc, const void* qkvPtr,
    const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
{
    pimpl->run(inputDesc, outputDesc, qkvPtr, maskPtr, output, workspace, stream);
}

void FusedMHARunnerFP16::run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    assert(false && "not implemented");
}

bool FusedMHARunnerFP16::isValid(int s) const
{
    return pimpl->isValid(s);
}

// Int8 starts here: TODO refactor the duplicate stuff

class FusedMHARunnerInt8::mhaImpl
{

public:
    mhaImpl(FusedMHARunnerInt8* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernels(DATA_TYPE_INT8, sm))
        , mDqProbs(interface->mDqProbs)
        , xmmas_m(0U)
        , xmmas_n(0U)
        , threads_per_cta(1U)
    {
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        assert(xmmas_m > 0);
        assert(threads_per_cta > 0);
        assert(interface->mB > 0);
        return interface->mB * xmmas_m * threads_per_cta * sizeof(uint32_t);
    }

    void setup(const int S, const int B)
    {
        size_t warps_m{1U};
        size_t warps_n{1U};
        size_t warps_k{1U};
        if (S == 128)
        {
            warps_m = 2;
            warps_n = 2;
        }
        else if (S == 384)
        {
            warps_m = 1;
            warps_n = 8;
        }
        else
        {
            assert(false && "Unsupporte seqlen");
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);


        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;

        params.qkv_stride_in_bytes = get_size_in_bytes(interface->mLdQKV, DATA_TYPE_INT8);
        params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
        params.o_stride_in_bytes = get_size_in_bytes(interface->mLdOut, DATA_TYPE_INT8);
    }

    void run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc, const void* qkvPtr,
        const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
    {
        float scaleQkv = inputDesc.scale;
        float scaleCtx = outputDesc.scale;

        float scaleBmm1 = scaleQkv * scaleQkv * interface->mRsqrtHeadSize;
        float scaleBmm2 = mDqProbs * scaleQkv / scaleCtx;
        float scaleSoftmax = 1.f / mDqProbs;

        params.scale_bmm1 = asUInt32(scaleBmm1);
        params.scale_bmm2 = asUInt32(scaleBmm2);
        params.scale_softmax = asUInt32(scaleSoftmax);

        params.enable_i2f_trick = -double(1 << 22) * double(scaleBmm2) <= -128.f
            && double(1 << 22) * double(scaleBmm2) >= 127.f;

        params.qkv_ptr = const_cast<void*>(qkvPtr);

        params.packed_mask_ptr = const_cast<void*>(maskPtr);

        params.o_ptr = output;

        xmmaKernel->run(params, stream);
        PLUGIN_CHECK(cudaPeekAtLastError());
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

private:
    float mDqProbs;
    FusedMHARunnerInt8* interface;
    Fused_multihead_attention_params params;
    int sm;
    const FusedMultiHeadAttentionXMMAKernel* xmmaKernel;
    size_t xmmas_m;
    size_t xmmas_n;
    size_t threads_per_cta;
};

FusedMHARunnerInt8::FusedMHARunnerInt8(const int numHeads, const int headSize, const int sm, const float dqProbs)
    : MHARunner(DataType::kINT8, numHeads, headSize)
    , mSm(sm)
    , pimpl(new mhaImpl(this))
    , mDqProbs(dqProbs)
{
}

void FusedMHARunnerInt8::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    pimpl->setup(S, B);
}

size_t FusedMHARunnerInt8::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerInt8::deserialize(const void* data, size_t length)
{
    MHARunner::deserialize(data, length);
    setup(mS, mB);
}

void FusedMHARunnerInt8::run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc, const void* qkvPtr,
    const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
{
    pimpl->run(inputDesc, outputDesc, qkvPtr, maskPtr, output, workspace, stream);
}

void FusedMHARunnerInt8::run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    assert(false && "not implemented");
}

bool FusedMHARunnerInt8::isValid(int s) const
{
    return pimpl->isValid(s);
}

class FusedMHARunnerFP16v2::mhaImpl
{
public:
    mhaImpl(FusedMHARunnerFP16v2* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernelsV2(DATA_TYPE_FP16, sm))
    {
        assert((sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 || sm == kSM_87 || sm == kSM_89 || sm == kSM_90)
            && "Unsupported architecture");
        params.clear();
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        // check that we initialized
        assert(xmmas_m > 0);
        assert(threads_per_cta > 0);
        assert(interface->mB > 0);
        return interface->mB * xmmas_m * threads_per_cta * sizeof(uint32_t);
    }

    void setup(const int S, const int B)
    {
        // TODO these implementation details might be better centralized into the XMMA code, since they are needed in
        // several places (also outside of this plugin)
        size_t warps_m{1U};
        size_t warps_n{1U};
        size_t warps_k{1U};
        if (S == 64 || S == 96 || S == 128)
        {
            warps_m = 2;
            warps_n = 2;
        }
        else if (S == 256 || S == 192)
        {
            warps_m = 1;
            warps_n = 4;
        }
        else if (S == 384 || S == 512)
        {
            warps_m = 1;
            warps_n = 8;
        }

        else
        {
            assert(false && "Unsupporte seqlen");
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

        const float scale_bmm1 = interface->mRsqrtHeadSize;
        const float scale_softmax = 1.f; // Seems to be only required for int8
        const float scale_bmm2 = 1.f;

        Data_type scale_type = DATA_TYPE_FP16;
        set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;

        // mLdQKV = 3 * B * mNumHeads * mHeadSize;
        // mLdOut = B * mNumHeads * mHeadSize;

        params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
        params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
        params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(half);
    }

    void run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc, const void* qkvPtr,
        const void* maskPtr, const void* cuSeqlenPtr, void* output, void* workspace, cudaStream_t stream)
    {

        params.qkv_ptr = const_cast<void*>(qkvPtr);

        // dummy input in V2/V3 because now we use cu_seqlens
        params.packed_mask_ptr = nullptr;

        params.o_ptr = output;

        params.cu_seqlens = static_cast<int*>(const_cast<void*>(cuSeqlenPtr));
        xmmaKernel->run(params, stream);
        PLUGIN_CHECK(cudaPeekAtLastError());
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

private:
    FusedMHARunnerFP16v2* interface;
    Fused_multihead_attention_params_v2 params;
    int sm;
    const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
    size_t xmmas_m;
    size_t xmmas_n;
    size_t threads_per_cta;
};

FusedMHARunnerFP16v2::FusedMHARunnerFP16v2(const int numHeads, const int headSize, const int sm)
    : MHARunner(DataType::kHALF, numHeads, headSize)
    , mSm(sm)
    , pimpl(new mhaImpl(this))
{
}

void FusedMHARunnerFP16v2::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    pimpl->setup(S, B);
}

size_t FusedMHARunnerFP16v2::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerFP16v2::deserialize(const void* data, size_t length)
{
    MHARunner::deserialize(data, length);
    setup(mS, mB);
}

void FusedMHARunnerFP16v2::run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc,
    const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
{
    assert(false && "not implemented");
    // pimpl->run(inputDesc, outputDesc, qkvPtr, maskPtr, output, workspace, stream);
}

void FusedMHARunnerFP16v2::run(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    pimpl->run(inputDesc[0], outputDesc[0], inputs[0], inputs[1], inputs[2], outputs[0], workspace, stream);
}

bool FusedMHARunnerFP16v2::isValid(int s) const
{
    return pimpl->isValid(s);
}

// Int8 starts here: TODO refactor the duplicate stuff

class FusedMHARunnerInt8v2::mhaImpl
{

public:
    mhaImpl(FusedMHARunnerInt8v2* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernelsV2(DATA_TYPE_INT8, sm))
        , mDqProbs(interface->mDqProbs)
        , xmmas_m(0U)
        , xmmas_n(0U)
        , threads_per_cta(1U)
    {
        assert((sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 || sm == kSM_87 || sm == kSM_89 || sm == kSM_90)
            && "Unsupported architecture");
        params.clear();
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        assert(xmmas_m > 0);
        assert(threads_per_cta > 0);
        assert(interface->mB > 0);
        return interface->mB * xmmas_m * threads_per_cta * sizeof(uint32_t);
    }

    void setup(const int S, const int B)
    {
        size_t warps_m{1U};
        size_t warps_n{1U};
        size_t warps_k{1U};
        if (S == 128)
        {
            warps_m = 2;
            warps_n = 2;
        }
        else if (S == 256 || S == 192)
        {
            warps_m = 1;
            warps_n = 4;
        }
        else if (S == 384 || S == 512)
        {
            warps_m = 1;
            warps_n = 8;
        }

        else
        {
            assert(false && "Unsupported seqlen.");
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;
        params.use_int8_scale_max = interface->mUseInt8ScaleMax;
        params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
        params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(int8_t);
        params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(int8_t);
    }

    void run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc, const void* qkvPtr,
        const void* maskPtr, const void* cuSeqlenPtr, void* output, void* workspace, cudaStream_t stream)
    {
        float scaleQkv = inputDesc.scale;
        float scaleCtx = outputDesc.scale;

        float scaleBmm1 = scaleQkv * scaleQkv * interface->mRsqrtHeadSize;
        float scaleBmm2 = mDqProbs * scaleQkv / scaleCtx;
        float scaleSoftmax = 1.f / mDqProbs;

        params.scale_bmm1 = asUInt32(scaleBmm1);
        params.scale_bmm2 = asUInt32(scaleBmm2);
        params.scale_softmax = asUInt32(scaleSoftmax);

        params.enable_i2f_trick
            = -double(1 << 22) * double(scaleBmm2) <= -128.f && double(1 << 22) * double(scaleBmm2) >= 127.f;

        params.qkv_ptr = const_cast<void*>(qkvPtr);

        // dummy input in V2/V3 because now we use cu_seqlens
        params.packed_mask_ptr = nullptr;

        params.use_int8_scale_max = interface->mUseInt8ScaleMax;

        params.o_ptr = output;

        params.cu_seqlens = static_cast<int*>(const_cast<void*>(cuSeqlenPtr));

        xmmaKernel->run(params, stream);
        PLUGIN_CHECK(cudaPeekAtLastError());
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

private:
    float mDqProbs;
    FusedMHARunnerInt8v2* interface;
    Fused_multihead_attention_params_v2 params;
    int sm;
    const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
    size_t xmmas_m;
    size_t xmmas_n;
    size_t threads_per_cta;
};

FusedMHARunnerInt8v2::FusedMHARunnerInt8v2(const int numHeads, const int headSize, const int sm, const float dqProbs, bool const useInt8ScaleMax)
    : MHARunner(DataType::kINT8, numHeads, headSize)
    , mSm(sm)
    , pimpl(new mhaImpl(this))
    , mDqProbs(dqProbs)
    , mUseInt8ScaleMax(useInt8ScaleMax)
{
}

void FusedMHARunnerInt8v2::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    pimpl->setup(S, B);
}

size_t FusedMHARunnerInt8v2::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerInt8v2::deserialize(const void* data, size_t length)
{
    MHARunner::deserialize(data, length);
    setup(mS, mB);
}

void FusedMHARunnerInt8v2::run(const PluginTensorDesc& inputDesc, const PluginTensorDesc& outputDesc,
    const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
{
    assert(false && "Not implemented");
}

void FusedMHARunnerInt8v2::run(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    pimpl->run(inputDesc[0], outputDesc[0], inputs[0], inputs[1], inputs[2], outputs[0], workspace, stream);
}

bool FusedMHARunnerInt8v2::isValid(int s) const
{
    return pimpl->isValid(s);
}

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
