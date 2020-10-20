/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "specialSlicePlugin.h"
#include "maskRCNNKernels.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::SpecialSlice;
using nvinfer1::plugin::SpecialSlicePluginCreator;

namespace
{
const char* SPECIALSLICE_PLUGIN_VERSION{"1"};
const char* SPECIALSLICE_PLUGIN_NAME{"SpecialSlice_TRT"};
} // namespace

PluginFieldCollection SpecialSlicePluginCreator::mFC{};
std::vector<PluginField> SpecialSlicePluginCreator::mPluginAttributes;

SpecialSlicePluginCreator::SpecialSlicePluginCreator()
{

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SpecialSlicePluginCreator::getPluginName() const
{
    return SPECIALSLICE_PLUGIN_NAME;
};

const char* SpecialSlicePluginCreator::getPluginVersion() const
{
    return SPECIALSLICE_PLUGIN_VERSION;
};

const PluginFieldCollection* SpecialSlicePluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2Ext* SpecialSlicePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    return new SpecialSlice();
};

IPluginV2Ext* SpecialSlicePluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new SpecialSlice(data, length);
};

size_t SpecialSlice::getWorkspaceSize(int) const
{
    return 0;
}

bool SpecialSlice::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
};

const char* SpecialSlice::getPluginType() const
{
    return "SpecialSlice_TRT";
};

const char* SpecialSlice::getPluginVersion() const
{
    return "1";
};

IPluginV2Ext* SpecialSlice::clone() const
{
    auto plugin = new SpecialSlice(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
};

void SpecialSlice::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* SpecialSlice::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

size_t SpecialSlice::getSerializationSize() const
{
    return sizeof(int);
};

void SpecialSlice::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mBboxesCnt);
    ASSERT(d == a + getSerializationSize());
};

SpecialSlice::SpecialSlice(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mBboxesCnt = read<int>(d);
    assert(d == a + length);
};

SpecialSlice::SpecialSlice(){

};

int SpecialSlice::initialize()
{
    return 0;
};

int SpecialSlice::getNbOutputs() const
{
    return 1;
};

void SpecialSlice::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{

    assert(nbInputDims == 1);
    // detections: [N, anchors, (y1, x1, y2, x2, class_id, score)]
    assert(inputs[0].nbDims == 2 && inputs[0].d[1] == 6);
}

Dims SpecialSlice::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
{

    assert(index == 0);
    assert(nbInputs == 1);
    check_valid_inputs(inputDims, nbInputs);

    nvinfer1::Dims output;
    output.nbDims = inputDims[0].nbDims;
    // number of anchors
    output.d[0] = inputDims[0].d[0];
    //(y1, x1, y2, x2)
    output.d[1] = 4;

    return output;
};

int SpecialSlice::enqueue(
    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    specialSlice(stream, batch_size, mBboxesCnt, inputs[0], outputs[0]);

    return cudaGetLastError() != cudaSuccess;
};

// Return the DataType of the plugin output at the requested index
DataType SpecialSlice::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 1 input and 1 output from the plugin layer
    ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool SpecialSlice::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool SpecialSlice::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void SpecialSlice::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    assert(nbInputs == 1);

    assert(nbOutputs == 1);

    mBboxesCnt = inputDims[0].d[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void SpecialSlice::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void SpecialSlice::detachFromContext() {}
