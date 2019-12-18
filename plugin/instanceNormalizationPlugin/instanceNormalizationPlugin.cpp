/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <stdexcept>
#include "instanceNormalizationPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::InstanceNormalizationPlugin;
using nvinfer1::plugin::InstanceNormalizationPluginCreator;

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

inline bool is_CHW(nvinfer1::Dims const& dims)
{
    return (dims.nbDims == 3 && dims.type[0] == nvinfer1::DimensionType::kCHANNEL
        && dims.type[1] == nvinfer1::DimensionType::kSPATIAL && dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

// This is derived from: https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
inline float half_to_float_fast(unsigned short value)
{
    union F32
    {
        unsigned int u;
        float f;
    };
    static const F32 magic = {(254 - 15) << 23};
    static const F32 was_infnan = {(127 + 16) << 23};
    F32 result;
    result.u = (value & 0x7fff) << 13; // exponent/mantissa bits
    result.f *= magic.f;               // exponent adjust
    if (result.f >= was_infnan.f)
    { // make sure Inf/NaN survive
        result.u |= 255 << 23;
    }
    result.u |= (value & 0x8000) << 16; // sign bit
    return result.f;
}

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype, cudnnDataType_t* cudnn_dtype)
{
    switch (trt_dtype)
    {
    case nvinfer1::DataType::kFLOAT: *cudnn_dtype = CUDNN_DATA_FLOAT; break;
    case nvinfer1::DataType::kHALF: *cudnn_dtype = CUDNN_DATA_HALF; break;
    default: return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

namespace {
    constexpr const char* INSTANCE_PLUGIN_VERSION{"001"};
    constexpr const char* INSTANCE_PLUGIN_NAME{"InstanceNormalization_TRT"};
}

PluginFieldCollection InstanceNormalizationPluginCreator::mFC{};
std::vector<PluginField> InstanceNormalizationPluginCreator::mPluginAttributes;


InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    float epsilon, const std::vector<float>& scale, const std::vector<float>& bias)
    : _epsilon(epsilon)
    , _nchan(scale.size())
    , _h_scale(scale)
    , _h_bias(bias)
    , _initialized(false)
{
    ASSERT(scale.size() == bias.size());
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias)
    : _epsilon(epsilon)
    , _nchan(scale.count)
    , _initialized(false)
{
    ASSERT(scale.count == bias.count);
    if (scale.type == nvinfer1::DataType::kFLOAT)
    {
        _h_scale.assign((float*) scale.values, (float*) scale.values + scale.count);
    }
    else if (scale.type == nvinfer1::DataType::kHALF)
    {
        _h_scale.reserve(_nchan);
        for (int c = 0; c < _nchan; ++c)
        {
            unsigned short value = ((unsigned short*) scale.values)[c];
            _h_scale.push_back(half_to_float_fast(value));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported scale dtype");
    }
    if (bias.type == nvinfer1::DataType::kFLOAT)
    {
        _h_bias.assign((float*) bias.values, (float*) bias.values + bias.count);
    }
    else if (bias.type == nvinfer1::DataType::kHALF)
    {
        _h_bias.reserve(_nchan);
        for (int c = 0; c < _nchan; ++c)
        {
            unsigned short value = ((unsigned short*) bias.values)[c];
            _h_bias.push_back(half_to_float_fast(value));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported bias dtype");
    }
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(void const* serialData, size_t serialLength) : _initialized(false)
{
    deserialize_value(&serialData, &serialLength, &_epsilon);
    deserialize_value(&serialData, &serialLength, &_nchan);
    deserialize_value(&serialData, &serialLength, &_h_scale);
    deserialize_value(&serialData, &serialLength, &_h_bias);
}

InstanceNormalizationPlugin::~InstanceNormalizationPlugin()
{
    terminate();
}

// InstanceNormalizationPlugin returns one output.
int InstanceNormalizationPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs InstanceNormalizationPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

int InstanceNormalizationPlugin::initialize()
{
    _initialized = true;
    return 0;
}

void InstanceNormalizationPlugin::terminate()
{
    if (!_initialized)
    {
        return;
    }
    cudnnDestroyTensorDescriptor(_y_desc);
    cudnnDestroyTensorDescriptor(_x_desc);
    cudnnDestroyTensorDescriptor(_b_desc);
    cudaFree(_d_bias);
    cudaFree(_d_scale);
    cudnnDestroy(_cudnn_handle);
    _initialized = false;
}

size_t InstanceNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{ 
    return 0; 
}


int InstanceNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int n = input_dims.d[0];
    int c = input_dims.d[1];
    int h = input_dims.d[2];
    int w = input_dims.d[3];
    size_t nchan_bytes = c * sizeof(float);

    // Note: We repeat the data for each batch entry so that we can do the full
    //       computation in a single CUDNN call in enqueue().
    CHECK_CUDA(cudaMalloc((void**) &_d_scale, n * nchan_bytes));
    CHECK_CUDA(cudaMalloc((void**) &_d_bias, n * nchan_bytes));
    for (int i = 0; i < n; ++i)
    {
        CHECK_CUDA(cudaMemcpy(_d_scale + i * c, _h_scale.data(), nchan_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(_d_bias + i * c, _h_bias.data(), nchan_bytes, cudaMemcpyHostToDevice));
    }
    CHECK_CUDNN(cudnnCreate(&_cudnn_handle));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&_b_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&_x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&_y_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1));
    cudnnDataType_t cudnn_dtype;
    CHECK_CUDNN(convert_trt2cudnn_dtype(inputDesc[0].type, &cudnn_dtype));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(_y_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
    float alpha = 1;
    float beta = 0;
    void const* x_ptr = inputs[0];
    void* y_ptr = outputs[0];
    CHECK_CUDNN(cudnnSetStream(_cudnn_handle, stream));
    // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
    //       overflows (NaNs) for fp32 data in some circumstances. The lower-
    //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
    //       acceptable.
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(_cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta,
        _x_desc, x_ptr, _y_desc, y_ptr, _b_desc, _d_scale, _d_bias, 1., nullptr, nullptr, _epsilon, nullptr, nullptr));
    return 0;
}

size_t InstanceNormalizationPlugin::getSerializationSize() const
{
    return (serialized_size(_epsilon) +
            serialized_size(_nchan) +
            serialized_size(_h_scale) +
            serialized_size(_h_bias));
}

void InstanceNormalizationPlugin::serialize(void *buffer) const
{
    serialize_value(&buffer, _epsilon);
    serialize_value(&buffer, _nchan);
    serialize_value(&buffer, _h_scale);
    serialize_value(&buffer, _h_bias);
}

bool InstanceNormalizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
        && inOut[pos].format == nvinfer1::PluginFormat::kNCHW);
}

const char* InstanceNormalizationPlugin::getPluginType() const
{
    return INSTANCE_PLUGIN_NAME;
}

const char* InstanceNormalizationPlugin::getPluginVersion() const
{
    return INSTANCE_PLUGIN_VERSION;
}

void InstanceNormalizationPlugin::destroy()
{ 
    delete this;
}

IPluginV2DynamicExt* InstanceNormalizationPlugin::clone() const
{ 
    return new InstanceNormalizationPlugin{_epsilon, _h_scale, _h_bias};
}

// Set plugin namespace
void InstanceNormalizationPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* InstanceNormalizationPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType InstanceNormalizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void InstanceNormalizationPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void InstanceNormalizationPlugin::detachFromContext() {}

void InstanceNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    for (int i = 0; i < nbInputs; i++)
    {
      for (int j = 0; j < in[0].desc.dims.nbDims; j++)
      {
        // Do not support dynamic dimensions
        ASSERT(in[0].desc.dims.d[j] != -1);
      }
    }
}

// InstanceNormalizationPluginCreator methods
InstanceNormalizationPluginCreator::InstanceNormalizationPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* InstanceNormalizationPluginCreator::getPluginName() const
{
    return INSTANCE_PLUGIN_NAME;
}

const char* InstanceNormalizationPluginCreator::getPluginVersion() const
{
    return INSTANCE_PLUGIN_VERSION;
}

const PluginFieldCollection* InstanceNormalizationPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    std::vector<float> scaleValues;
    std::vector<float> biasValues;
    float epsilon {};
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "epsilon"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            epsilon= *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scales"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            scaleValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                scaleValues.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "bias"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            biasValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                biasValues.push_back(*w);
                w++;
            }
        }
    }

    Weights scaleWeights{DataType::kFLOAT, scaleValues.data(), (int64_t) scaleValues.size()};
    Weights biasWeights{DataType::kFLOAT, biasValues.data(), (int64_t) biasValues.size()};

    InstanceNormalizationPlugin* obj = new InstanceNormalizationPlugin(epsilon, scaleWeights, biasWeights);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    InstanceNormalizationPlugin* obj = new InstanceNormalizationPlugin{serialData, serialLength}; 
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
