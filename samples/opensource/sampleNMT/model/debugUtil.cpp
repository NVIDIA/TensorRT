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

#include "debugUtil.h"

#include <cassert>
#include <cuda_runtime_api.h>

#include "../cudaError.h"

namespace nmtSample
{
std::list<DebugUtil::DumpTensorPlugin::ptr> DebugUtil::mPlugins;

DebugUtil::DumpTensorPlugin::DumpTensorPlugin(std::shared_ptr<std::ostream> out)
    : mOut(out)
{
}

int DebugUtil::DumpTensorPlugin::getNbOutputs() const
{
    return 1;
}

nvinfer1::Dims DebugUtil::DumpTensorPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    return inputs[0];
}

void DebugUtil::DumpTensorPlugin::configure(
    const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize)
{
    mDims = inputDims[0];

    *mOut << "Max batch size = " << maxBatchSize << std::endl;
    *mOut << "Tensor dimensions = ";
    mTensorVolume = 1;
    for (int i = 0; i < mDims.nbDims; ++i)
    {
        if (i > 0)
            *mOut << "x";
        *mOut << mDims.d[i];
        mTensorVolume *= mDims.d[i];
    }
    mElemsPerRow = 1;
    for (int i = mDims.nbDims - 1; i >= 0; --i)
    {
        if (mElemsPerRow == 1)
            mElemsPerRow *= mDims.d[i];
    }
    *mOut << std::endl;

    mData = std::make_shared<PinnedHostBuffer<float>>(mTensorVolume * maxBatchSize);
}

int DebugUtil::DumpTensorPlugin::initialize()
{
    return 0;
}

void DebugUtil::DumpTensorPlugin::terminate()
{
    mOut.reset();
    mData.reset();
}

size_t DebugUtil::DumpTensorPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int DebugUtil::DumpTensorPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    int totalElems = batchSize * mTensorVolume;

    CUDA_CHECK(cudaMemcpyAsync(*mData, inputs[0], totalElems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], totalElems * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    *mOut << "Batch size = " << batchSize << "\n";
    int rowCount = totalElems / mElemsPerRow;
    for (int rowId = 0; rowId < rowCount; ++rowId)
    {
        for (int i = 0; i < mElemsPerRow; ++i)
        {
            if (i > 0)
                *mOut << " ";
            *mOut << (*mData)[rowId * mElemsPerRow + i];
        }
        *mOut << "\n";
    }
    *mOut << std::endl;

    return 0;
}

size_t DebugUtil::DumpTensorPlugin::getSerializationSize()
{
    assert(0);
    return 0;
}

void DebugUtil::DumpTensorPlugin::serialize(void* buffer)
{
    assert(0);
}

void DebugUtil::addDumpTensorToStream(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
    nvinfer1::ITensor** output, std::shared_ptr<std::ostream> out)
{
    assert(!input->getBroadcastAcrossBatch());
    auto plugin = std::make_shared<DumpTensorPlugin>(out);
    nvinfer1::ITensor* inputTensors[] = {input};
    auto pluginLayer = network->addPlugin(inputTensors, 1, *plugin);
    assert(pluginLayer != nullptr);
    *output = pluginLayer->getOutput(0);
    assert(*output != nullptr);
    mPlugins.push_back(plugin);
}
} // namespace nmtSample
