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

#include "NvInfer.h"
#include "bertCommon.h"
#include "common.h"
#include "qkvToContextPlugin.h"
#include "serialize.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;

namespace bert
{

template <typename T, int TPB, int VPT>
__global__ void maskedSoftmax(const float rsqrtHeadSize, const T* input, T* output, const int* maskIdx)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ union
    {
        T shm[VPT * TPB];
        typename BlockReduce::TempStorage reduce;
    } tmp;

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

    const int idx = (blockOffset + threadIdx.x) * VPT;
    T* myshm = &tmp.shm[threadIdx.x * VPT];
    copy<sizeof(T) * VPT>(&input[idx], myshm);

    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it]
            = (threadIdx.x < lastValid) ? myExp<float>((rsqrtHeadSize) * float(tmp.shm[it * TPB + threadIdx.x])) : 0.f;
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

    __shared__ union
    {
        T shm[VPT * TPB];
        typename BlockReduce::TempStorage reduce;
    } tmp;

    __shared__ float rZ;

    const int idx = (TPB * blockIdx.x + threadIdx.x) * VPT;
    T* myshm = &tmp.shm[threadIdx.x * VPT];
    copy<sizeof(T) * VPT>(&input[idx], myshm);

    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = myExp<float>(rsqrtHeadSize * float(tmp.shm[it * TPB + threadIdx.x]));
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

    CHECK(cudaPeekAtLastError());
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

    CHECK(cudaPeekAtLastError());
    return 0;
}

std::pair<int, int> tuneBatchedGemm(const int B, const int S, const int numHeads, const int headSize)
{
    const int nruns = 500;
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasSetStream(cublas, stream);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

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
    cudaMalloc(&input, inBytes);
    cudaMalloc(&qkptr, qkBytes);
    cudaMalloc(&output, outBytes);
    cudaMemset(input, 1, inBytes);
    cudaMemset(qkptr, 1, qkBytes);

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
    for (int a = startAlgo; a <= endAlgo; a++)
    {
        cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(a);
        float ms1_, ms2_;
        // qkptr: BxNxSxS
        cudaEventRecord(start, stream);
        for (int r = 0; r < nruns; r++)
        {
            CHECK(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, headSize, T(1.f), kptr, ldQKV,
                strideQKV, qptr, ldQKV, strideQKV, T(0.f), qkptr, S, omatSize, numMats, algo));
        }

        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&ms1_, start, stop);
        if (ms1_ < ms1)
        {
            best1 = algo;
            ms1 = ms1_;
        }

        // pptr: BxNxSxS
        // output: SxBxNxH
        cudaEventRecord(start, stream);
        for (int r = 0; r < nruns; r++)
        {
            CHECK(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, headSize, S, S, 1.f, vptr, ldQKV,
                strideQKV, qkptr, S, omatSize, 0.f, output, ldOut, strideOut, numMats, algo));
        }

        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&ms2_, start, stop);

        if (ms2_ < ms2)
        {
            best2 = algo;
            ms2 = ms2_;
        }
    }

    cudaFree(input);
    cudaFree(qkptr);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas);
    return std::make_pair(best1, best2);
}

template <typename T>
int QKVToContextPluginDynamic::qkvToCtx(cublasHandle_t& cublas, const int B, const int S, const int numHeads,
    const int headSize, const float rsqrtHeadSize, const T* input, T* output, T* qkptr, T* pptr, cudaStream_t stream,
    const int* maskIdx)
{

    const int omatSize = S * S;
    const int numMats = B * numHeads;
    const T* qptr = input;
    const T* kptr = qptr + headSize;
    const T* vptr = kptr + headSize;

    cublasSetStream(cublas, stream);
    CublasConfigHelper helper(cublas);

    // Q, K, V: BxNxSxH (inputs)
    // Q * K': BxNxSxS (-> scratch1)
    // P: BxNxSxS (-> scratch2)
    // P * V: BxNxSxH (output)

    const int ldQKV = 3 * B * numHeads * headSize;
    const int strideQKV = 3 * headSize;
    
    if (mType == DataType::kHALF)
    {
        CHECK(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, headSize, 1.f, kptr, ldQKV,
            strideQKV, qptr, ldQKV, strideQKV, 0.f, qkptr, S, omatSize, numMats,
            static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx1)));
    }
    else 
    {

        CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, headSize, 1.f, kptr, ldQKV, strideQKV,
            qptr, ldQKV, strideQKV, 0.f, qkptr, S, omatSize, numMats));
    }

    // apply softmax
    if (maskIdx)
    { // if we have a mask
        computeMaskedScaledSoftmax<T>(stream, S, B, numHeads, rsqrtHeadSize, maskIdx, qkptr, pptr);
    }
    else
    { // if we don't have a mask
        computeScaledSoftmax<T>(stream, S, B, numHeads, rsqrtHeadSize, qkptr, pptr);
    }

    // compute P*V (as V*P)

    const int ldOut = B * numHeads * headSize;
    const int strideOut = headSize;
    if (mType == DataType::kHALF)
    {

        CHECK(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, headSize, S, S, 1.f, vptr, ldQKV,
            strideQKV, pptr, S, omatSize, 0.f, output, ldOut, strideOut, numMats,
            static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx2)));
    }
    else 
    {

        CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, headSize, S, S, 1.f, vptr, ldQKV, strideQKV,
            pptr, S, omatSize, 0.f, output, ldOut, strideOut, numMats));
    }
    return 0;
}

namespace
{
static const char* QKV_TO_CONTEXT_PLUGIN_VERSION{"1"};
static const char* QKV_TO_CONTEXT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection QKVToContextPluginDynamicCreator::mFC{};
std::vector<PluginField> QKVToContextPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKVToContextPluginDynamicCreator);

constexpr size_t kAlignment = 256;
constexpr uint32_t IIDX = 0; // index of the input tensor
constexpr uint32_t MIDX = 1; // index of the mask

QKVToContextPluginDynamic::QKVToContextPluginDynamic(
    const std::string name, const DataType type, const int hiddenSize, const int numHeads, bool hasImask)
    : mLayerName(name)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mHasImask(hasImask)
    , mType(type)
      , mAlgoBatchedEx1(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
      , mAlgoBatchedEx2(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
{
    assert(hiddenSize % numHeads == 0);
    mHeadSize = hiddenSize / numHeads;
    mRsqrtHeadSize = 1.f / sqrt(float(mHeadSize));
}

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "QKV Deser Start" << std::endl;
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mRsqrtHeadSize);
    deserialize_value(&data, &length, &mHasImask);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mAlgoBatchedEx1);
    deserialize_value(&data, &length, &mAlgoBatchedEx2);
    gLogVerbose << "QKV Deser done" << std::endl;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextPluginDynamic::clone() const
{
    gLogVerbose << "QKV Clone" << std::endl;
    auto ret = new QKVToContextPluginDynamic(mLayerName, mType, mHiddenSize, mNumHeads, mHasImask);
    ret->initialize();
    gLogVerbose << "QKV Clone done" << std::endl;
    return ret;
}

DimsExprs QKVToContextPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    // Input is BxSx3*N*H, output should be BxSxN*H
    assert(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    auto three = exprBuilder.constant(3);
    output.d[HDIM] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[IIDX].d[HDIM], *three);
    return output;
}
bool QKVToContextPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    assert(pos >= 0);
    assert(pos < 2 + mHasImask);
    assert(nbInputs == 1 + mHasImask);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    if (pos == 0)
    {
        // must not check descriptions > pos
        return (in->type == mType) &&                // precision
            (in->format == TensorFormat::kLINEAR) && // format
            (in->dims.nbDims == 5) &&                // num dims
            ((in->dims.d[HDIM] % 3) == 0) &&         // see getOutputDimensions
            ((in->dims.d[3]) == 1) &&                // for fc
            ((in->dims.d[4]) == 1)                   // for fc
            ;
    }
    else
    { // pos==1
        if ((mHasImask && pos == 1))
        {
            const auto* inMask = &inOut[1];
            return (inMask->type == DataType::kINT32) &&     // precision
                (inMask->format == TensorFormat::kLINEAR) && // format
                (inMask->dims.nbDims == 1) &&                // num dims
                ((inMask->dims.d[0]) == in->dims.d[BDIM])    // check B
                ;
        }
        if (!mHasImask || (pos == 2))
        {
            return (in->type == out->type) &&                      // precision
                (out->format == TensorFormat::kLINEAR) &&          // format
                (out->dims.nbDims == 5) &&                         // num dims
                ((in->dims.d[HDIM] / 3) == (out->dims.d[HDIM])) && // div 3
                ((out->dims.d[3]) == 1) &&                         // for fc
                ((out->dims.d[4]) == 1) &&                         // for fc
                ((out->dims.d[BDIM]) == in->dims.d[BDIM]) &&       // check B
                ((out->dims.d[SDIM]) == in->dims.d[SDIM])          // check S
                ;
        }
    }
    return false;
}
void QKVToContextPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(nbInputs == 1 + mHasImask);
    assert(nbOutputs == 1);
    const PluginTensorDesc& inDesc = in[IIDX].desc;
    TRT_UNUSED inDesc;
    const PluginTensorDesc& outDesc = out->desc;
    TRT_UNUSED outDesc;
    assert(mType == inDesc.type);
    assert(mType == outDesc.type);
    assert(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
    assert(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
    assert(inDesc.dims.d[HDIM] == 3 * outDesc.dims.d[HDIM]);
    if (mHasImask)
    {
        const PluginTensorDesc& maskDesc = in[MIDX].desc;
        TRT_UNUSED maskDesc;
        assert(maskDesc.type == DataType::kINT32);
        assert(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
    }

    const int S = in->max.d[SDIM];
    const int B = in->max.d[BDIM];
    std::tie(mAlgoBatchedEx1, mAlgoBatchedEx2) = tuneBatchedGemm(B, S, mNumHeads, mHeadSize);
    gLogVerbose << "QKV Plugin - Selected Algos for batch gemms: " << mAlgoBatchedEx1 << ", " << mAlgoBatchedEx2 << "\n";
}

size_t QKVToContextPluginDynamic::scratchSize(const int B, const int S) const
{
    const size_t wordSize = samplesCommon::getElementSize(mType);
    const size_t len = B * mNumHeads * S * S;
    const size_t bytes = len * wordSize;

    return bytes;
}

size_t QKVToContextPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    const int B = inputs->dims.d[BDIM];
    const int S = inputs->dims.d[SDIM];

    const size_t bytesAligned = alignTo<size_t>(scratchSize(B, S), kAlignment);
    const size_t ws = 2UL * bytesAligned;

    return ws;
}

// IPluginV2Ext Methods
DataType QKVToContextPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* QKVToContextPluginDynamic::getPluginType() const
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPluginDynamic::getPluginVersion() const
{
    return QKV_TO_CONTEXT_PLUGIN_VERSION;
}

int QKVToContextPluginDynamic::getNbOutputs() const
{
    return 1;
}

int QKVToContextPluginDynamic::initialize()
{
    cublasCreate(&cublas);
    return 0;
}

void QKVToContextPluginDynamic::terminate()
{
    CHECK(cublasDestroy(cublas));
}

size_t QKVToContextPluginDynamic::getSerializationSize() const
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mRsqrtHeadSize) + sizeof(mHasImask)
        + sizeof(mHiddenSize) + sizeof(mAlgoBatchedEx1) + sizeof(mAlgoBatchedEx2);
}

void QKVToContextPluginDynamic::serialize(void* buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mNumHeads);
    serialize_value(&buffer, mHeadSize);
    serialize_value(&buffer, mRsqrtHeadSize);
    serialize_value(&buffer, mHasImask);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mAlgoBatchedEx1);
    serialize_value(&buffer, mAlgoBatchedEx2);
}

void QKVToContextPluginDynamic::destroy()
{
    delete this;
}

void QKVToContextPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

int QKVToContextPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{

    const int batchSize = inputDesc->dims.d[BDIM];
    const int S = inputDesc->dims.d[SDIM];

    const size_t bytesAligned = alignTo<size_t>(scratchSize(batchSize, S), kAlignment);
    char* scratch1 = static_cast<char*>(workspace);
    char* scratch2 = scratch1 + bytesAligned;

    const int* maskIdx = mHasImask ? static_cast<const int*>(inputs[1]) : nullptr;

    int status = -1;
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        float* scr1 = reinterpret_cast<float*>(scratch1);
        float* scr2 = reinterpret_cast<float*>(scratch2);

        status = qkvToCtx(
            cublas, batchSize, S, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, stream, maskIdx);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        half* output = static_cast<half*>(outputs[0]);
        half* scr1 = reinterpret_cast<half*>(scratch1);
        half* scr2 = reinterpret_cast<half*>(scratch2);

        status = qkvToCtx(
            cublas, batchSize, S, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, stream, maskIdx);
    }
    else
    {
        assert(false);
    }

    return status;
}

QKVToContextPluginDynamicCreator::QKVToContextPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKVToContextPluginDynamicCreator::getPluginName() const
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPluginDynamicCreator::getPluginVersion() const
{
    return QKV_TO_CONTEXT_PLUGIN_VERSION;
}

const PluginFieldCollection* QKVToContextPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* QKVToContextPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating QKV2ContextPlugin...\n";

    int hiddenSize = 0;
    int numHeads = 0;
    bool hasMask = false;
    int typeId = -1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building typeId: " << typeId << std::endl;
        }
        if (field_name.compare("hidden_size") == 0)
        {
            hiddenSize = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building hiddenSize: " << hiddenSize << std::endl;
        }
        if (field_name.compare("num_heads") == 0)
        {
            numHeads = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building numHeads: " << numHeads << std::endl;
        }
        if (field_name.compare("has_mask") == 0)
        {
            hasMask = *static_cast<const bool*>(fc->fields[i].data);
            gLogVerbose << "Building hasMask: " << hasMask << std::endl;
        }
    }
    if (typeId < 0 || typeId > 3)
    {
        gLogError << "QKV: Invalid TypeId " << typeId << std::endl;
    }

    if (hiddenSize <= 0)
    {
        gLogError << "QKV: Invalid hiddenSize " << hiddenSize << std::endl;
    }

    if (numHeads <= 0)
    {
        gLogError << "QKV: Invalid numHeads " << numHeads << std::endl;
    }

    gLogVerbose << "Building the Plugin...\n";
    DataType type = static_cast<DataType>(typeId);
    QKVToContextPluginDynamic* p = new QKVToContextPluginDynamic(name, type, hiddenSize, numHeads, hasMask);
    return p;
}

IPluginV2* QKVToContextPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextPluginDynamic::destroy()
    return new QKVToContextPluginDynamic(name, serialData, serialLength);
}

void QKVToContextPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
} // namespace bert
