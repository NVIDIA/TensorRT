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
#include <iostream>

using namespace nvinfer1;

namespace bert
{

namespace test
{

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

inline void launchTransCtx(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
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

inline void launchTransCtx(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
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

inline void launchTransQkv(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
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

inline void launchTransQkv(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
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
inline int qkvToCtx(cublasHandle_t& cublas, const int B, const int S, const int numHeads, const int headSize,
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


QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const int hiddenSize, const int numHeads, bool hasImask)
    : mLayerName(name)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mHasImask(hasImask)
{
    assert(hiddenSize % numHeads == 0);
    mHeadSize = hiddenSize / numHeads;
    mRsqrtHeadSize = 1.f / sqrt(float(mHeadSize));
}

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{

    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    gLogVerbose << "QKV Deser Start" << std::endl;

    DESER(d, mType);
    DESER(d, mNumHeads);
    DESER(d, mHeadSize);
    DESER(d, mRsqrtHeadSize);
    DESER(d, mHasImask);
    DESER(d, mHiddenSize);

    gLogVerbose << "QKV Deser done" << std::endl;

    assert(d == (a + length));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextPluginDynamic::clone() const
{
    gLogVerbose << "QKV Clone" << std::endl;
    auto ret = new QKVToContextPluginDynamic(mLayerName, mHiddenSize, mNumHeads,  mHasImask);
    ret->mType = mType;
    ret->initialize();
    gLogVerbose << "QKV Clone done" << std::endl;
    return ret;
}

DimsExprs QKVToContextPluginDynamic::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
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
bool QKVToContextPluginDynamic::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    assert(pos >= 0);
    assert(pos < 2 + mHasImask);
    assert(nbInputs == 1 + mHasImask);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    if (pos == 0)
    {
        // must not check descriptions > pos
        return (in->type == DataType::kFLOAT || in->type == DataType::kHALF) && // precision
            (in->format == TensorFormat::kLINEAR) &&                            // format
            (in->dims.nbDims == 5) &&                                           // num dims
            ((in->dims.d[HDIM] % 3) == 0) &&                                    // see getOutputDimensions
            ((in->dims.d[3]) == 1) &&                                           // for fc
            ((in->dims.d[4]) == 1)                                              // for fc
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
                ((inMask->dims.d[BDIM]) == in->dims.d[BDIM]) // check B
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
void QKVToContextPluginDynamic::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(nbInputs == 1 + mHasImask);
    assert(nbOutputs == 1);
    const PluginTensorDesc& inDesc = in[IIDX].desc;
    const PluginTensorDesc& outDesc = out->desc;
    mType = inDesc.type;
    assert(mType == outDesc.type);
    assert(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
    assert(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
    assert(inDesc.dims.d[HDIM] == 3 * outDesc.dims.d[HDIM]);
    if (mHasImask)
    {
        const PluginTensorDesc& maskDesc = in[MIDX].desc;
        assert(maskDesc.type == DataType::kINT32);
        assert(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
    }
}

size_t QKVToContextPluginDynamic::scratchSize(const int B, const int S) const
{
    size_t wordSize = samplesCommon::getElementSize(mType);
    const size_t len = B * mNumHeads * S * S;
    const size_t bytes = len * wordSize;

    return bytes;
}

size_t QKVToContextPluginDynamic::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    const int B = inputs->dims.d[BDIM];
    const int S = inputs->dims.d[SDIM];
    const size_t bytes = scratchSize(B, S);
    const size_t bytesAligned = alignTo<size_t>(bytes, kAlignment);
    const size_t two = 2;
    const size_t ws = two * bytesAligned;

    const size_t wordSize = samplesCommon::getElementSize(mType);
    const size_t tp = 3 * B * S * mNumHeads * mHeadSize * wordSize;

    return ws + tp;
}

// IPluginV2Ext Methods
DataType QKVToContextPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
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
    return sizeof(mNumHeads) +  sizeof(mHeadSize) + sizeof(DataType) + sizeof(mRsqrtHeadSize)
        + sizeof(mHasImask) + sizeof(mHiddenSize);
}

void QKVToContextPluginDynamic::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    writeToBuffer(d, mType);
    writeToBuffer(d, mNumHeads);
    writeToBuffer(d, mHeadSize);
    writeToBuffer(d, mRsqrtHeadSize);
    writeToBuffer(d, mHasImask);
    writeToBuffer(d, mHiddenSize);

    assert(d == a + getSerializationSize());
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

        status = qkvToCtx(cublas, batchSize, S, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, scr3,
            stream, maskIdx);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        half* output = static_cast<half*>(outputs[0]);
        half* scr1 = reinterpret_cast<half*>(scratch1);
        half* scr2 = reinterpret_cast<half*>(scratch2);
        half* scr3 = reinterpret_cast<half*>(scratch3);

        status = qkvToCtx(cublas, batchSize, S, mNumHeads, mHeadSize, mRsqrtHeadSize, input, output, scr1, scr2, scr3,
            stream, maskIdx);
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

    int hidden_size;
    int num_heads;
    bool has_mask;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("hidden_size") == 0)
        {
            hidden_size = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building hidden_size: " << hidden_size << std::endl;
        }
        if (field_name.compare("num_heads") == 0)
        {
            num_heads = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building num_heads: " << num_heads << std::endl;
        }
        if (field_name.compare("has_mask") == 0)
        {
            has_mask = *static_cast<const bool*>(fc->fields[i].data);
            gLogVerbose << "Building has_mask: " << has_mask << std::endl;
        }
    }

    gLogVerbose << "Building the Plugin...\n";
    QKVToContextPluginDynamic* p = new QKVToContextPluginDynamic(name, hidden_size, num_heads, has_mask);
    return p;
}

IPluginV2* QKVToContextPluginDynamicCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
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
}
}
