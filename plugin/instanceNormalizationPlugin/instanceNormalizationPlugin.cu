/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "checkMacrosPlugin.h"
#include "instanceNormalizationPlugin.h"
#include <algorithm>
#include <cuda_fp16.h>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::InstanceNormalizationPlugin;
using nvinfer1::plugin::InstanceNormalizationPluginCreator;

template <typename T, int32_t THREADS_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void in3dReluActivation(
    T* __restrict dst, T* __restrict src, float alpha, int32_t count)
{
    int32_t idx = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    if (idx >= count)
        return;

    float val = src[idx];
    dst[idx] = (val < 0.f) ? val * alpha : val;
}

cudnnStatus_t convertTrt2cudnnDtype(nvinfer1::DataType trt_dtype, cudnnDataType_t* cudnn_dtype)
{
    switch (trt_dtype)
    {
    case nvinfer1::DataType::kFLOAT: *cudnn_dtype = CUDNN_DATA_FLOAT; break;
    case nvinfer1::DataType::kHALF: *cudnn_dtype = CUDNN_DATA_HALF; break;
    default: return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

namespace
{
constexpr const char* INSTANCE_PLUGIN_VERSION{"1"};
constexpr const char* INSTANCE_PLUGIN_NAME{"InstanceNormalization_TRT"};
} // namespace

PluginFieldCollection InstanceNormalizationPluginCreator::mFC{};
std::vector<PluginField> InstanceNormalizationPluginCreator::mPluginAttributes;

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    float epsilon, const std::vector<float>& scale, const std::vector<float>& bias, int32_t relu, float alpha)
    : mEpsilon(epsilon)
    , mAlpha(alpha)
    , mRelu(relu)
    , mNchan(scale.size())
    , mHostScale(scale)
    , mHostBias(bias)
{
    ASSERT(scale.size() == bias.size());
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias, int32_t relu, float alpha)
    : mEpsilon(epsilon)
    , mAlpha(alpha)
    , mRelu(relu)
    , mNchan(scale.count)
{
    ASSERT(scale.count == bias.count);
    const auto copyWeights = [](nvinfer1::Weights const& input, std::vector<float>& output) {
        output.reserve(input.count);
        if (input.type == nvinfer1::DataType::kFLOAT)
        {
            output.assign(static_cast<const float*>(input.values), static_cast<const float*>(input.values) + input.count);
        }
        else if (input.type == nvinfer1::DataType::kHALF)
        {
            for (int32_t c = 0; c < input.count; ++c)
            {
                const auto value = static_cast<const unsigned short*>(input.values);
                output.push_back(__internal_half2float(value[c]));
            }
        }
        else
        {
            throw std::runtime_error("Unsupported scale/bias dtype");
        }
    };

    copyWeights(scale, mHostScale);
    copyWeights(bias, mHostBias);
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mEpsilon);
    deserialize_value(&serialData, &serialLength, &mNchan);
    deserialize_value(&serialData, &serialLength, &mHostScale);
    deserialize_value(&serialData, &serialLength, &mHostBias);
    deserialize_value(&serialData, &serialLength, &mRelu);
    deserialize_value(&serialData, &serialLength, &mAlpha);
}

InstanceNormalizationPlugin::~InstanceNormalizationPlugin()
{
    terminate();
}

// InstanceNormalizationPlugin returns one output.
int32_t InstanceNormalizationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs InstanceNormalizationPlugin::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

int32_t InstanceNormalizationPlugin::initialize() noexcept
{
    if (!mInitialized)
    {
        CHECK_CUDNN(cudnnCreate(&mCudnnHandle));

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mBDescriptor));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mXDescriptor));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mYDescriptor));

        // NDHWC path
        // Device info.
        int32_t device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, device));

        mContext.sm_count = props.multiProcessorCount;
        mContext.sm_shared_size = props.sharedMemPerMultiprocessor;
        mContext.sm_version = props.major * 100 + props.minor * 10;

        CHECK_CUDA(cudaMalloc(&mDeviceScale, mNchan * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mDeviceBias, mNchan * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mDeviceScale, &mHostScale[0], mNchan * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mDeviceBias, &mHostBias[0], mNchan * sizeof(float), cudaMemcpyHostToDevice));
    }
    mInitialized = true;

    return 0;
}

void InstanceNormalizationPlugin::terminate() noexcept
{
    if (mInitialized)
    {
        cudnnDestroyTensorDescriptor(mYDescriptor);
        cudnnDestroyTensorDescriptor(mXDescriptor);
        cudnnDestroyTensorDescriptor(mBDescriptor);

        cudnnDestroy(mCudnnHandle);

        CUASSERT(cudaFree(mDeviceBias));
        CUASSERT(cudaFree(mDeviceScale));
    }
    mInitialized = false;
}

size_t InstanceNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    nvinfer1::Dims input_dims = inputs[0].dims;
    ASSERT(input_dims.nbDims == 4 || input_dims.nbDims == 5);

    if (inputs[0].format == nvinfer1::PluginFormat::kLINEAR)
    {
        nvinfer1::Dims input_dims = inputs[0].dims;

        int32_t n = input_dims.d[0];
        int32_t c = input_dims.d[1];

        size_t nchan_bytes = c * sizeof(float);
        size_t scale_size = n * nchan_bytes;
        size_t bias_size = n * nchan_bytes;

        size_t total_wss = scale_size + bias_size;

        return total_wss;
    }
    else if (inputs[0].format == nvinfer1::PluginFormat::kDHWC8 || inputs[0].format == nvinfer1::PluginFormat::kCDHW32)
    {
        ASSERT(input_dims.nbDims == 5);
        int32_t input_data_type = (inputs[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
        int32_t output_data_type = (outputs[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
        nvinfer1::Dims input_dims = inputs[0].dims;

        int32_t n = input_dims.d[0];
        int32_t c = input_dims.d[1];
        int32_t d = input_dims.d[2];
        int32_t h = input_dims.d[3];
        int32_t w = input_dims.d[4];

        InstanceNormFwdParams params;
        // only these parameters are required for workspace computation
        params.nhw = d * h * w;
        params.c = c;
        params.n = n;
        // Reserve memory for the workspaces.
        size_t size_sums, size_counts, size_retired_ctas;
        instanceNormBufferSizesDispatch(
            mContext, params, size_sums, size_counts, size_retired_ctas, input_data_type, output_data_type);
        size_t size_nc = n * c * sizeof(float);
        size_nc = ((size_nc + 256 - 1) / 256) * 256;
        return size_sums + size_counts + size_retired_ctas + 4 * size_nc;
    }
    else
    {
        ASSERT(0);
    }
    return 0;
}

int32_t InstanceNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    // early return for empty tensor
    if (std::any_of(input_dims.d, input_dims.d + input_dims.nbDims, [](int32_t d) { return d == 0; }))
    {
        return 0;
    }

    const auto callRelu = [this, &stream](void* inOut, int32_t count, nvinfer1::DataType type) {
        if (mRelu > 0)
        {
            const int32_t kBLOCK_SZ = 256;
            switch (type)
            {
            case nvinfer1::DataType::kFLOAT:
                in3dReluActivation<float, kBLOCK_SZ><<<(count + kBLOCK_SZ - 1) / kBLOCK_SZ, kBLOCK_SZ, 0, stream>>>(
                    static_cast<float*>(inOut), static_cast<float*>(inOut), mAlpha, count);
                break;
            case nvinfer1::DataType::kHALF:
                in3dReluActivation<__half, kBLOCK_SZ><<<(count + kBLOCK_SZ - 1) / kBLOCK_SZ, kBLOCK_SZ, 0, stream>>>(
                    static_cast<__half*>(inOut), static_cast<__half*>(inOut), mAlpha, count);
                break;
            default: ASSERT(0);
            }
        }
    };

    if (input_dims.nbDims <= 4)
    {
        nvinfer1::Dims input_dims = inputDesc[0].dims;
        int32_t n = input_dims.d[0];
        int32_t c = input_dims.d[1];
        int32_t h = input_dims.d[2];
        int32_t w = input_dims.nbDims > 3 ? input_dims.d[3] : 1;
        size_t nchan_bytes = c * sizeof(float);

        float* _d_array = static_cast<float*>(workspace);
        float* d_scale = &_d_array[0];
        float* d_bias = &_d_array[n * c];
        for (int32_t i = 0; i < n; ++i)
        {
            CUASSERT(cudaMemcpyAsync(d_scale + i * c, mDeviceScale, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
            CUASSERT(cudaMemcpyAsync(d_bias + i * c, mDeviceBias, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
        }

        CUDNNASSERT(cudnnSetTensor4dDescriptor(mBDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1));
        cudnnDataType_t cudnn_dtype{};
        CUDNNASSERT(convertTrt2cudnnDtype(inputDesc[0].type, &cudnn_dtype));
        CUDNNASSERT(cudnnSetTensor4dDescriptor(mXDescriptor, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
        CUDNNASSERT(cudnnSetTensor4dDescriptor(mYDescriptor, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
        float alpha = 1;
        float beta = 0;
        void const* x_ptr = inputs[0];
        void* y_ptr = outputs[0];
        CUDNNASSERT(cudnnSetStream(mCudnnHandle, stream));
        // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
        //       overflows (NaNs) for fp32 data in some circumstances. The lower-
        //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
        //       acceptable.
        CUDNNASSERT(cudnnBatchNormalizationForwardTraining(mCudnnHandle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha,
            &beta, mXDescriptor, x_ptr, mYDescriptor, y_ptr, mBDescriptor, d_scale, d_bias, 1., nullptr, nullptr,
            mEpsilon, nullptr, nullptr));

        callRelu(y_ptr, n * c * h * w, inputDesc[0].type);
    }
    else
    {
        if (inputDesc[0].format == nvinfer1::PluginFormat::kLINEAR)
        {
            CHECK_CUDNN(cudnnSetStream(mCudnnHandle, stream));
            nvinfer1::Dims input_dims = inputDesc[0].dims;
            int32_t n = input_dims.d[0];
            int32_t c = input_dims.d[1];
            int32_t d = input_dims.d[2];
            int32_t h = input_dims.d[3];
            int32_t w = input_dims.d[4];
            size_t nchan_bytes = c * sizeof(float);

            // Note: We repeat the data for each batch entry so that we can do the full
            //       computation in a single CUDNN call in enqueue().
            float* _d_array = (float*) workspace;
            float* d_scale = &_d_array[0];
            float* d_bias = &_d_array[n * c];
            for (int32_t i = 0; i < n; ++i)
            {
                CHECK_CUDA(
                    cudaMemcpyAsync(d_scale + i * c, mDeviceScale, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
                CHECK_CUDA(cudaMemcpyAsync(d_bias + i * c, mDeviceBias, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
            }

            int32_t nc_dimA[] = {1, n * c, 1, 1, 1};
            int32_t nc_strideA[] = {nc_dimA[1] * nc_dimA[2] * nc_dimA[3] * nc_dimA[4],
                nc_dimA[2] * nc_dimA[3] * nc_dimA[4], nc_dimA[3] * nc_dimA[4], nc_dimA[4], 1};
            int32_t img_dimA[] = {1, n * c, d, h, w};
            int32_t img_strideA[] = {img_dimA[1] * img_dimA[2] * img_dimA[3] * img_dimA[4],
                img_dimA[2] * img_dimA[3] * img_dimA[4], img_dimA[3] * img_dimA[4], img_dimA[4], 1};

            CHECK_CUDNN(cudnnSetTensorNdDescriptor(mBDescriptor, CUDNN_DATA_FLOAT, 5, nc_dimA, nc_strideA));
            cudnnDataType_t cudnn_dtype;
            CHECK_CUDNN(convertTrt2cudnnDtype(inputDesc[0].type, &cudnn_dtype));
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(mXDescriptor, cudnn_dtype, 5, img_dimA, img_strideA));
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(mYDescriptor, cudnn_dtype, 5, img_dimA, img_strideA));
            float alpha = 1;
            float beta = 0;

            void const* x_ptr = inputs[0];
            void* y_ptr = outputs[0];
            // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
            //       overflows (NaNs) for fp32 data in some circumstances. The lower-
            //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
            //       acceptable.
            CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(mCudnnHandle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha,
                &beta, mXDescriptor, x_ptr, mYDescriptor, y_ptr, mBDescriptor, d_scale, d_bias, 1., nullptr, nullptr,
                mEpsilon, nullptr, nullptr));

            callRelu(y_ptr, n * c * d * h * w, inputDesc[0].type);
        }
        else if (inputDesc[0].format == nvinfer1::PluginFormat::kDHWC8
            || inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32)
        {
            int32_t input_data_type = (inputDesc[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
            int32_t output_data_type = (outputDesc[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;

            nvinfer1::Dims input_dims = inputDesc[0].dims;
            int32_t n = input_dims.d[0];
            int32_t c = input_dims.d[1];
            int32_t d = input_dims.d[2];
            int32_t h = input_dims.d[3];
            int32_t w = input_dims.d[4];

            InstanceNormFwdParams params;
            params.nhw = d * h * w;
            params.c = c;
            params.n = n;

            size_t size_sums, size_counts, size_retired_ctas;
            instanceNormBufferSizesDispatch(
                mContext, params, size_sums, size_counts, size_retired_ctas, input_data_type, output_data_type);

            size_t size_nc = n * c * sizeof(float);
            size_nc = ((size_nc + 256 - 1) / 256) * 256;

            char* d_buf = static_cast<char*>(workspace);

            params.gmem_sums = reinterpret_cast<GMEM_SUMS_TYPE*>(d_buf);
            d_buf += size_sums;
            params.gmem_counts = reinterpret_cast<int32_t*>(d_buf);
            d_buf += size_counts;
            params.gmem_retired_ctas = reinterpret_cast<int32_t*>(d_buf);
            d_buf += size_retired_ctas;
            params.gmem_running_mean = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;
            params.gmem_running_var = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;
            params.gmem_saved_mean = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;
            params.gmem_saved_var = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;

            params.gmem_src = inputs[0];
            params.gmem_dst = outputs[0];
            params.gmem_bias = mDeviceBias;
            params.gmem_scale = mDeviceScale;

            params.var_eps = mEpsilon;
            params.exp_avg_factor = 1.F; //(float)exp_avg_factor;
            params.use_relu = mRelu;     // use_relu;
            params.relu_alpha = mAlpha;  // relu_alpha;

            params.in_scale = inputDesc[0].scale;
            ASSERT(outputDesc[0].scale != 0.F);
            params.out_scale = 1.F / outputDesc[0].scale;

            instanceNormFwdDispatch(mContext, params, stream, input_data_type, output_data_type);
        }
        else
        {
            ASSERT(false && "Unexpected input format");
        }
    }
    return 0;
}

size_t InstanceNormalizationPlugin::getSerializationSize() const noexcept
{
    return (serialized_size(mEpsilon) + serialized_size(mNchan) + serialized_size(mHostScale)
        + serialized_size(mHostBias) + serialized_size(mRelu) + serialized_size(mAlpha));
}

void InstanceNormalizationPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNchan);
    serialize_value(&buffer, mHostScale);
    serialize_value(&buffer, mHostBias);
    serialize_value(&buffer, mRelu);
    serialize_value(&buffer, mAlpha);
}

bool InstanceNormalizationPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    ASSERT(pos == 0 || pos == 1);

    // For 4-D or 3-D tensor (nbSpatialDims == 1 or 2), only FP32_Linear and FP16_Linear are supported.
    // For 5-D tensor (nbSpatialDims == 3), FP32_Linear, FP16_Linear, FP16_DHWC8, and INT8_CDHW32 are supported.
    // This is because we have special InstanceNorm3D kernels for vectorized formats from MLPerf-Inference.

    const int32_t nbDims = inOut[pos].dims.nbDims;
    ASSERT(nbDims >= 3);
    ASSERT(nbDims <= 5);
    const bool is3DInstanceNorm = (nbDims == 5);

    const bool isFP32Linear
        = (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    const bool isFP16Linear
            = (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
                && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    const bool isFP16DHWC8
        = (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::PluginFormat::kDHWC8
            && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    const bool isINT8CDHW32
        = (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::PluginFormat::kCDHW32
            && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    const bool isFormatOK = isFP32Linear || isFP16Linear || (is3DInstanceNorm && (isFP16DHWC8 || isINT8CDHW32));

    // Kernels for vectorized formats only support the case of C % spv == 0.
    int32_t spv{1};
    switch (inOut[pos].format)
    {
    case nvinfer1::PluginFormat::kDHWC8: spv = 8; break;
    case nvinfer1::PluginFormat::kCDHW32: spv = 32; break;
    default: break;
    }
    const int32_t isAlignmentOK = (inOut[pos].dims.d[1] % spv == 0);

    return isFormatOK && isAlignmentOK;
}

const char* InstanceNormalizationPlugin::getPluginType() const noexcept
{
    return INSTANCE_PLUGIN_NAME;
}

const char* InstanceNormalizationPlugin::getPluginVersion() const noexcept
{
    return INSTANCE_PLUGIN_VERSION;
}

void InstanceNormalizationPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* InstanceNormalizationPlugin::clone() const noexcept
{
    auto* plugin = new InstanceNormalizationPlugin{mEpsilon, mHostScale, mHostBias, mRelu, mAlpha};
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
}

// Set plugin namespace
void InstanceNormalizationPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* InstanceNormalizationPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

nvinfer1::DataType InstanceNormalizationPlugin::getOutputDataType(
    int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void InstanceNormalizationPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void InstanceNormalizationPlugin::detachFromContext() noexcept {}

void InstanceNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    // Not support dynamic shape in C dimension
    ASSERT(nbInputs == 1 && in[0].desc.dims.d[1] != -1);
}

// InstanceNormalizationPluginCreator methods
InstanceNormalizationPluginCreator::InstanceNormalizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("relu", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* InstanceNormalizationPluginCreator::getPluginName() const noexcept
{
    return INSTANCE_PLUGIN_NAME;
}

const char* InstanceNormalizationPluginCreator::getPluginVersion() const noexcept
{
    return INSTANCE_PLUGIN_VERSION;
}

const PluginFieldCollection* InstanceNormalizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    std::vector<float> scaleValues;
    std::vector<float> biasValues;
    float epsilon{};
    int32_t relu{};
    float alpha{};
    const PluginField* fields = fc->fields;
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "epsilon"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            epsilon = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scales"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int32_t size = fields[i].length;
            scaleValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int32_t j = 0; j < size; j++)
            {
                scaleValues.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "bias"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int32_t size = fields[i].length;
            biasValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int32_t j = 0; j < size; j++)
            {
                biasValues.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "relu"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            relu = *(static_cast<const int32_t*>(fields[i].data));
        }
        else if (!strcmp(attrName, "alpha"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            alpha = *(static_cast<const float*>(fields[i].data));
        }
    }

    Weights scaleWeights{DataType::kFLOAT, scaleValues.data(), (int64_t) scaleValues.size()};
    Weights biasWeights{DataType::kFLOAT, biasValues.data(), (int64_t) biasValues.size()};

    InstanceNormalizationPlugin* obj = new InstanceNormalizationPlugin(epsilon, scaleWeights, biasWeights, relu, alpha);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    InstanceNormalizationPlugin* obj = new InstanceNormalizationPlugin{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
