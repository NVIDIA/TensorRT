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
#include "pluginKernels.h"
#include "pluginUtil.h"
#include "qkvToContextPlugin.h"

#include <cassert>
#include <common.h>
#include <cstring>
#include <half.h>
#include <vector>

namespace bert
{

using namespace nvinfer1;

constexpr size_t kAlignment = 256;

template <typename T>
__global__ void transposeCtx(const int H, const T* input, T* output)
{
    // Input:  HxSxNxB
    // Output: HxNxSxB

    int n = threadIdx.y;
    int s = blockIdx.x;
    int b = blockIdx.y;

    int N = blockDim.y;
    int S = gridDim.x;
    // B = gridDim.y

    const int NH = N * H;
    const int NHS = NH * S;
    const int in_offset = s * H + n * S * H + b * NHS;
    const int out_offset = n * H + s * NH + b * NHS;

    const int i = threadIdx.x;
    if (i < H)
    {
        output[out_offset + i] = input[in_offset + i];
    }
}

void launchTransCtx(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
    const float* input, float* output)
{

    const dim3 grid(S, B, 1);
    if (0 == (headSize & 1))
    {
        const int H = headSize / 2;
        const float2* input2 = reinterpret_cast<const float2*>(input);
        float2* output2 = reinterpret_cast<float2*>(output);
        const dim3 block(H, numHeads, 1);
        transposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
        CHECK(cudaPeekAtLastError());
    }
    else
    {
        const dim3 block(headSize, numHeads, 1);
        transposeCtx<float><<<grid, block, 0, stream>>>(headSize, input, output);
        CHECK(cudaPeekAtLastError());
    }
}

void launchTransCtx(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
    const half* input, half* output)
{
    const dim3 grid(S, B, 1);
    if (0 == (headSize % 4))
    {
        const int H = headSize / 4;
        const dim3 block(H, numHeads, 1);
        const float2* input2 = reinterpret_cast<const float2*>(input);
        float2* output2 = reinterpret_cast<float2*>(output);
        transposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
    else if (0 == (headSize & 1))
    {
        const int H = headSize / 2;
        const dim3 block(H, numHeads, 1);
        const half2* input2 = reinterpret_cast<const half2*>(input);
        half2* output2 = reinterpret_cast<half2*>(output);
        transposeCtx<half2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
    else
    { // this should be an "odd" case. probably not worth catching it in the half2 kernel.
        const dim3 block(headSize, numHeads, 1);
        transposeCtx<half><<<grid, block, 0, stream>>>(headSize, input, output);
    }
    CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void transposeQKV(const int H, const T* input, T* output)
{
    // Input:  HxNx3xSxB
    // Output: HxSxNxBx3

    int n = threadIdx.y;
    int s = blockIdx.x;
    int b = blockIdx.y;
    int m = blockIdx.z; // matrix id

    const int N = blockDim.y;

    const int S = gridDim.x;
    const int B = gridDim.y;
    const int NH = N * H;
    const int NHS = NH * S;
    const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
    const int out_offset = s * H + n * S * H + b * NHS + m * NHS * B;

    const int i = threadIdx.x;
    if (i < H)
    {
        output[out_offset + i] = input[in_offset + i];
    }
}

void launchTransQkv(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
    const float* input, float* output)
{

    const dim3 grid(S, B, 3);
    if (0 == (headSize & 1))
    {
        const int H = headSize / 2;
        const float2* input2 = reinterpret_cast<const float2*>(input);
        float2* output2 = reinterpret_cast<float2*>(output);
        const dim3 block(H, numHeads, 1);
        transposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
    else
    {
        const dim3 block(headSize, numHeads, 1);
        transposeQKV<float><<<grid, block, 0, stream>>>(headSize, input, output);
    }
    CHECK(cudaPeekAtLastError());
}

void launchTransQkv(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
    const half* input, half* output)
{
    const dim3 grid(S, B, 3);
    if (0 == (headSize % 4))
    {
        const int H = headSize / 4;
        const dim3 block(H, numHeads, 1);
        const float2* input2 = reinterpret_cast<const float2*>(input);
        float2* output2 = reinterpret_cast<float2*>(output);
        transposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
    else if (0 == (headSize & 1))
    {
        const int H = headSize / 2;
        const dim3 block(H, numHeads, 1);
        const half2* input2 = reinterpret_cast<const half2*>(input);
        half2* output2 = reinterpret_cast<half2*>(output);
        transposeQKV<half2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
    else
    { // this should be an "odd" case. probably not worth catching it in the half2 kernel..
        const dim3 block(headSize, numHeads, 1);
        transposeQKV<half><<<grid, block, 0, stream>>>(headSize, input, output);
    }
    CHECK(cudaPeekAtLastError());
}

template <typename T>
int qkvToCtx(cublasHandle_t& cublas, const int B, const int S, const int numHeads, const int headSize,
    const float rsqrtHeadSize, const T* input, T* output, T* qkptr, T* pptr, T* tptr, cudaStream_t stream,
    const int* maskIdx = nullptr)
{
    // input should be BxSx3xNxH => tptr: 3xBxNxSxH
    launchTransQkv(stream, S, B, headSize, numHeads, input, tptr);

    const int tsize = B * numHeads * S * headSize;
    const int imatSize = S * headSize;
    const int omatSize = S * S;
    const int numMats = B * numHeads;
    const T* qptr = tptr;
    const T* kptr = qptr + tsize;
    const T* vptr = kptr + tsize;

    cublasSetStream(cublas, stream);
    CublasConfigHelper helper(cublas);

    // Q, K, V: BxNxSxH (inputs)
    // Q * K': BxNxSxS (-> scratch1)
    // P: BxNxSxS (-> scratch2)
    // P * V: BxNxSxH (output)

    // compute Q*K' (as K'*Q)
    CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, headSize, 1.f, kptr, headSize, imatSize,
        qptr, headSize, imatSize, 0.f, qkptr, S, omatSize, numMats));

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
    CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, headSize, S, S, 1.f, vptr, headSize, imatSize,
        pptr, S, omatSize, 0.f, tptr, headSize, imatSize, numMats));

    // tptr is 3xBxNxSxH, so 3x output
    launchTransCtx(stream, S, B, headSize, numHeads, tptr, output);
    return 0;
}

namespace
{
static const char* QKVToCONTEXT_PLUGIN_VERSION{"1"};
static const char* QKVToCONTEXT_PLUGIN_NAME{"CustomQKVToContextPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection QKVToContextPluginCreator::mFC{};
std::vector<PluginField> QKVToContextPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKVToContextPluginCreator);

QKVToContextPlugin::QKVToContextPlugin(
    const std::string name, const int hiddenSize, const int numHeads, const int S, bool hasImask)
    : mLayerName(name)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mS(S)
    , mHasImask(hasImask)
{
    assert(hiddenSize % numHeads == 0);
    mHeadSize = hiddenSize / numHeads;
    mRsqrtHeadSize = 1.f / sqrt(float(mHeadSize));
}

QKVToContextPlugin::QKVToContextPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{

    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    gLogVerbose << "QKV Deser Start" << std::endl;

    DESER(d, mType);
    DESER(d, mS);
    DESER(d, mNumHeads);
    DESER(d, mHeadSize);
    DESER(d, mRsqrtHeadSize);
    DESER(d, mHasImask);

    gLogVerbose << "QKV Deser done" << std::endl;

    assert(d == (a + length));
}

const char* QKVToContextPlugin::getPluginType() const
{
    return QKVToCONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPlugin::getPluginVersion() const
{
    return QKVToCONTEXT_PLUGIN_VERSION;
}

int QKVToContextPlugin::getNbOutputs() const
{
    return 1;
}

Dims QKVToContextPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1 + mHasImask);
    assert(index == 0);

    return Dims4{mS, mNumHeads * mHeadSize, 1, 1};
}

void QKVToContextPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas_, IGpuAllocator* alloc)
{
    gLogVerbose << "QKV AttachToContext" << std::endl;
}

int QKVToContextPlugin::initialize()
{
    gLogVerbose << "QKV Initialize" << std::endl;
    cublasCreate(&cublas);

    return 0;
}

int QKVToContextPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    const size_t bytesAligned = alignTo<size_t>(scratchSize(batchSize), kAlignment);
    char* scratchBytes = reinterpret_cast<char*>(workspace);

    char* scratch1 = scratchBytes;
    char* scratch2 = scratchBytes + bytesAligned;
    char* scratch3 = scratch2 + bytesAligned;

    const int* maskIdx = mHasImask ? static_cast<const int*>(inputs[1]) : nullptr;

    int status = -1;
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        float* scr1 = reinterpret_cast<float*>(scratch1);
        float* scr2 = reinterpret_cast<float*>(scratch2);
        float* scr3 = reinterpret_cast<float*>(scratch3);

        status = qkvToCtx(cublas, batchSize, mS, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, scr3,
            stream, maskIdx);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        half* output = static_cast<half*>(outputs[0]);
        half* scr1 = reinterpret_cast<half*>(scratch1);
        half* scr2 = reinterpret_cast<half*>(scratch2);
        half* scr3 = reinterpret_cast<half*>(scratch3);

        status = qkvToCtx(cublas, batchSize, mS, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, scr3,
            stream, maskIdx);
    }
    else
    {
        assert(false);
    }

    return status;
}

size_t QKVToContextPlugin::getSerializationSize() const
{
    return sizeof(mNumHeads) + sizeof(mS) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mRsqrtHeadSize)
        + sizeof(mHasImask);
}

void QKVToContextPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    writeToBuffer(d, mType);
    writeToBuffer(d, mS);
    writeToBuffer(d, mNumHeads);
    writeToBuffer(d, mHeadSize);
    writeToBuffer(d, mRsqrtHeadSize);
    writeToBuffer(d, mHasImask);

    assert(d == a + getSerializationSize());
}

DataType QKVToContextPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    DataType type = inputTypes[0];
    if (type == DataType::kFLOAT || type == DataType::kHALF)
    {
        return type;
    }
    type = DataType::kFLOAT;
    return type;
}

void QKVToContextPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    // Validate input arguments
    assert(nbInputs == 1 + mHasImask);
    assert(nbOutputs == 1);

    assert(inputDims[0].nbDims == 4);
    assert(inputDims[0].d[0] == mS);
    assert(inputDims[0].d[1] == 3 * mHeadSize * mNumHeads);
    assert(inputDims[0].d[2] == 1);
    assert(inputDims[0].d[3] == 1);

    assert(outputDims[0].nbDims == 4);
    assert(outputDims[0].d[0] == mS);
    assert(outputDims[0].d[1] == mNumHeads * mHeadSize);
    assert(outputDims[0].d[2] == 1);
    assert(outputDims[0].d[3] == 1);
    mType = outputTypes[0];
    if (!(mType == DataType::kHALF || mType == DataType::kFLOAT))
        mType = DataType::kFLOAT;
    if (mHasImask)
    {
        assert(inputTypes[1] == DataType::kINT32);
    }
}

bool QKVToContextPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    if (type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT32)
    {
        return format == PluginFormat::kNCHW;
    }
    else
    {
        return false;
    }
}

void QKVToContextPlugin::terminate()
{
    gLogVerbose << "QKV Terminate " << std::endl;
    CHECK(cublasDestroy(cublas));
    gLogVerbose << "QKV Terminate done" << std::endl;
}

size_t QKVToContextPlugin::scratchSize(int batchsize) const
{
    size_t wordSize = samplesCommon::getElementSize(mType);
    const size_t len = batchsize * mNumHeads * mS * mS;
    const size_t bytes = len * wordSize;

    return bytes;
}

size_t QKVToContextPlugin::getWorkspaceSize(int batchsize) const
{
    const size_t bytes = scratchSize(batchsize);
    const size_t bytesAligned = alignTo<size_t>(bytes, kAlignment);
    const size_t two = 2;
    const size_t ws = two * bytesAligned;

    size_t wordSize = samplesCommon::getElementSize(mType);
    const size_t tp = 3 * batchsize * mS * mNumHeads * mHeadSize * wordSize;

    return ws + tp;
}

void QKVToContextPlugin::destroy() {}

IPluginV2Ext* QKVToContextPlugin::clone() const
{
    gLogVerbose << "QKV Clone" << std::endl;
    auto ret = new QKVToContextPlugin(mLayerName, mHiddenSize, mNumHeads, mS, mHasImask);
    ret->mType = mType;
    ret->initialize();
    gLogVerbose << "QKV Clone done" << std::endl;
    return ret;
}

void QKVToContextPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

QKVToContextPluginCreator::QKVToContextPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKVToContextPluginCreator::getPluginName() const
{
    return QKVToCONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPluginCreator::getPluginVersion() const
{
    return QKVToCONTEXT_PLUGIN_VERSION;
}

const PluginFieldCollection* QKVToContextPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* QKVToContextPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating QKV2ContextPlugin...\n";

    int hidden_size;
    int num_heads;
    int S;
    bool has_mask;

    for(int i=0; i< fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("hidden_size")==0)
        {
            hidden_size = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building hidden_size: " << hidden_size << std::endl;
        }
        if (field_name.compare("num_heads")==0)
        {
            num_heads =  *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building num_heads: " << num_heads << std::endl;
        }
        if (field_name.compare("S")==0)
        {
            S =  *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building S: " << S << std::endl;
        }
        if (field_name.compare("has_mask")==0)
        {
            has_mask =  *static_cast<const bool*>(fc->fields[i].data);
            gLogVerbose << "Building has_mask: " << has_mask  << std::endl;
        }
    }

    gLogVerbose << "Building the Plugin...\n";
    QKVToContextPlugin* p =  new QKVToContextPlugin(name, hidden_size, num_heads, S, has_mask);
    return p;
}

IPluginV2* QKVToContextPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextPlugin::destroy()
    return new QKVToContextPlugin(name, serialData, serialLength);
}

void QKVToContextPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
}
