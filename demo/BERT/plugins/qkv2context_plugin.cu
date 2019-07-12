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
#include "logger.h"
#include "plugin_kernels.hpp"
#include "plugin_util.hpp"
#include "qkv2context_plugin.hpp"

#include <cassert>
#include <cstring>
#include <half.h>
#include <vector>

using namespace nvinfer1;
#define HDI inline __host__ __device__

template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}
template <typename IntType>
constexpr HDI IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

constexpr size_t my_align = 256;


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
int compute_qkv2ctx(cublasHandle_t& cublas, const int mB, const int mS, const int mNumHeads, const int mHeadSize,
    const float mRsqrtHeadSize, const T* input, T* output, T* qkptr, T* pptr, cudaStream_t stream,
    const int* mask_idx = nullptr)
{
    // input should be 3xBxNxSxH

    cublasSetStream(cublas, stream);

    int tsize = mB * mNumHeads * mS * mHeadSize;
    int imat_size = mS * mHeadSize;
    int omat_size = mS * mS;
    int num_mats = mB * mNumHeads;

    const T* qptr = input;
    const T* kptr = input + tsize;
    const T* vptr = input + 2 * tsize;

    CublasConfigHelper helper(cublas);

    // Q, K, V: BxNxSxH (inputs)
    // Q * K': BxNxSxS (-> scratch1)
    // P: BxNxSxS (-> scratch2)
    // P * V: BxNxSxH (output)

    // compute Q*K' (as K'*Q)
    CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, mS, mS, mHeadSize, 1.f, kptr, mHeadSize,
        imat_size, qptr, mHeadSize, imat_size, 0.f, qkptr, mS, omat_size, num_mats));

    // apply softmax
    if (mask_idx)
    { // if we have a mask
        compute_masked_scaled_softmax<T>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, mask_idx, qkptr, pptr);
    }
    else
    { // if we don't have a mask
        compute_scaled_softmax<T>(stream, mS, num_mats * omat_size, mRsqrtHeadSize, qkptr, pptr);
    }

    // compute P*V (as V*P)
    CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, mS, mS, 1.f, vptr, mHeadSize,
        imat_size, pptr, mS, omat_size, 0.f, output, mHeadSize, imat_size, num_mats));

    return 0;
}

namespace
{
static const char* QKV2CONTEXT_PLUGIN_VERSION{"1"};
static const char* QKV2CONTEXT_PLUGIN_NAME{"CustomQKV2ContextPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection QKV2ContextPluginCreator::mFC{};
std::vector<PluginField> QKV2ContextPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKV2ContextPluginCreator);

QKV2ContextPlugin::QKV2ContextPlugin(
    const std::string name, const int hidden_size, const int num_heads, const int B, const int S, bool has_imask)
    : mLayerName(name)
    , mB(B)
    , mHiddenSize(hidden_size)
    , mNumHeads(num_heads)
    , mS(S)
    , mHasImask(has_imask)
{
    assert(hidden_size % num_heads == 0);
    mHeadSize = hidden_size / num_heads;
    mRsqrtHeadSize = 1.f / sqrt(float(mHeadSize));
}

QKV2ContextPlugin::QKV2ContextPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{

    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    gLogInfo << "QKV Deser Start" << std::endl;

    DESER(d, mType);
    DESER(d, mB);
    DESER(d, mS);
    DESER(d, mNumHeads);
    DESER(d, mHeadSize);
    DESER(d, mRsqrtHeadSize);
    DESER(d, mHasImask);

    gLogInfo << "QKV Deser done" << std::endl;

    assert(d == (a + length));
}

const char* QKV2ContextPlugin::getPluginType() const
{
    return QKV2CONTEXT_PLUGIN_NAME;
}

const char* QKV2ContextPlugin::getPluginVersion() const
{
    return QKV2CONTEXT_PLUGIN_VERSION;
}

int QKV2ContextPlugin::getNbOutputs() const
{
    return 1;
}

Dims QKV2ContextPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1 + mHasImask);
    assert(index == 0);

    Dims ret{inputs->nbDims - 1};
    for (int it = 0; it < inputs->nbDims - 1; it++)
    {
        ret.d[it] = inputs->d[it + 1];
    }

    return ret;
}

void QKV2ContextPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas_, IGpuAllocator* alloc)
{
    gLogInfo << "QKV AttachToContext" << std::endl;
}

int QKV2ContextPlugin::initialize()
{
    gLogInfo << "QKV Initialize" << std::endl;
    cublasCreate(&cublas);

    return 0;
}

int QKV2ContextPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    size_t bytes_aligned = alignTo<size_t>(scratchSize(batchSize), my_align);
    char* scratch_bytes = reinterpret_cast<char*>(workspace);

    char* scratch1 = scratch_bytes;
    char* scratch2 = scratch_bytes + bytes_aligned;

    int status = -1;
    const int* mask_idx = nullptr;
    if (mHasImask)
    {
        mask_idx = static_cast<const int*>(inputs[1]);
    }

    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        float* scr1 = reinterpret_cast<float*>(scratch1);
        float* scr2 = reinterpret_cast<float*>(scratch2);

        status = compute_qkv2ctx(
            cublas, mB, mS, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, stream, mask_idx);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        half* output = static_cast<half*>(outputs[0]);
        half* scr1 = reinterpret_cast<half*>(scratch1);
        half* scr2 = reinterpret_cast<half*>(scratch2);

        status = compute_qkv2ctx(
            cublas, mB, mS, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, stream, mask_idx);
    }
    else
    {
        assert(false);
    }

    return status;
}

size_t QKV2ContextPlugin::getSerializationSize() const
{
    return sizeof(mB) + sizeof(mNumHeads) + sizeof(mS) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mRsqrtHeadSize)
        + sizeof(mHasImask);
}

void QKV2ContextPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    writeToBuffer(d, mType);
    writeToBuffer(d, mB);
    writeToBuffer(d, mS);
    writeToBuffer(d, mNumHeads);
    writeToBuffer(d, mHeadSize);
    writeToBuffer(d, mRsqrtHeadSize);
    writeToBuffer(d, mHasImask);

    assert(d == a + getSerializationSize());
}

DataType QKV2ContextPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    DataType type = inputTypes[0];
    if (type == DataType::kFLOAT || type == DataType::kHALF)
    {
        return type;
    }
    type = DataType::kFLOAT;
    return type;
}

void QKV2ContextPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    // Validate input arguments
    assert(nbInputs == 1 + mHasImask);
    assert(nbOutputs == 1);
    assert(inputDims[0].nbDims == 5);
    assert(inputDims[0].d[0] == 3);
    assert(inputDims[0].d[1] == mB);
    assert(inputDims[0].d[2] == mNumHeads);
    assert(inputDims[0].d[3] == mS);
    assert(inputDims[0].d[4] == mHeadSize);

    assert(outputDims[0].nbDims == 4);
    assert(outputDims[0].d[0] == mB);
    assert(outputDims[0].d[1] == mNumHeads);
    assert(outputDims[0].d[2] == mS);
    assert(outputDims[0].d[3] == mHeadSize);

    mType = outputTypes[0];
    if (!(mType == DataType::kHALF || mType == DataType::kFLOAT))
        mType = DataType::kFLOAT;
    if (mHasImask)
    {
        assert(inputTypes[1] == DataType::kINT32);
    }
}

bool QKV2ContextPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    if (type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT32)
        return format == PluginFormat::kNCHW;
    else
        return false;
}

void QKV2ContextPlugin::terminate()
{
    gLogInfo << "QKV Terminate " << std::endl;
    cublasDestroy(cublas);
    gLogInfo << "QKV Terminate done" << std::endl;
}

size_t QKV2ContextPlugin::scratchSize(int batchsize) const
{
    int word_size = sizeof(float);
    if (mType == DataType::kHALF)
        word_size /= 2;
    size_t len = mB * mNumHeads * mS * mS;
    size_t bytes = len * word_size;

    return bytes;
}

size_t QKV2ContextPlugin::getWorkspaceSize(int batchsize) const
{
    size_t bytes = scratchSize(batchsize);
    size_t bytes_aligned = alignTo<size_t>(bytes, my_align);
    size_t two = 2;
    size_t ws = two * bytes_aligned;

    return ws;
}

void QKV2ContextPlugin::destroy()
{
}

IPluginV2Ext* QKV2ContextPlugin::clone() const
{
    gLogInfo << "QKV Clone" << std::endl;
    auto ret = new QKV2ContextPlugin(mLayerName, mHiddenSize, mNumHeads, mB, mS, mHasImask);
    ret->mType = mType;
    ret->initialize();
    gLogInfo << "QKV Clone done" << std::endl;
    return ret;
}

void QKV2ContextPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKV2ContextPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

QKV2ContextPluginCreator::QKV2ContextPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKV2ContextPluginCreator::getPluginName() const
{
    return QKV2CONTEXT_PLUGIN_NAME;
}

const char* QKV2ContextPluginCreator::getPluginVersion() const
{
    return QKV2CONTEXT_PLUGIN_VERSION;
}

const PluginFieldCollection* QKV2ContextPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* QKV2ContextPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogError << "QKV2ContextPluginCreator::createPlugin not implemented\n";
    assert(false);
    return nullptr;
}

IPluginV2* QKV2ContextPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call QKV2ContextPlugin::destroy()
    return new QKV2ContextPlugin(name, serialData, serialLength);
}

void QKV2ContextPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKV2ContextPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
