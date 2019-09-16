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

#include "driver.h"
#include "cuda_profiler_api.h"
#include <iostream>

using namespace nvinfer1;
using namespace samplesCommon;

namespace bert
{

HostTensor::HostTensor(void* data, const DataType type, const vector<size_t>& shape)
    : mShape(shape)
    , mData(data)
    , mType(type)
{
    mSize = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
    mNbBytes = mSize * samplesCommon::getElementSize(type);
}

Driver::Driver(const int maxBatchSize, const bool useFp16, const size_t maxWorkspaceSize)
    : mMaxBatchSize(maxBatchSize)
    , mMaxWorkspaceSize(maxWorkspaceSize)
    , mUseFp16(useFp16)
{
}

NetworkDefinitionCreationFlags Driver::getNetworkFlags() const
{
    return 1 << static_cast<size_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
}

IBuilderConfig* Driver::getBuilderConfig() const
{
    IBuilderConfig* config = mBuilder->createBuilderConfig();
    config->setMaxWorkspaceSize(mMaxWorkspaceSize);
    if (mUseFp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    return config;
}

void Driver::allocateBindings()
{
    // Static sizes with implicit batch size: allocation sizes known to engine
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        size_t vol = samplesCommon::volume(mEngine->getBindingDimensions(i));
        size_t elementSize = samplesCommon::getElementSize(mEngine->getBindingDataType(i));
        size_t allocationSize = static_cast<size_t>(mMaxBatchSize) * vol * elementSize;
        mDeviceBuffers.emplace_back(DeviceBuffer(allocationSize, mEngine->getBindingDataType(i)));
        mBuffers.emplace_back(mDeviceBuffers.back().data());
    }
}

void Driver::init(const HostTensorMap& params)
{
    mBuilder = createInferBuilder(gLogger.getTRTLogger());

    const NetworkDefinitionCreationFlags flags = getNetworkFlags();
    INetworkDefinition* network{mBuilder->createNetworkV2(flags)};

    buildNetwork(network, params);
    assert(network);

    IBuilderConfig* config = getBuilderConfig();
    gLogInfo << "Building Engine..." << endl;
    // Build the engine
    mEngine = (mBuilder->buildEngineWithConfig(*network, *config));
    gLogInfo << "Done building engine." << endl;

    assert(mEngine);
    mContext = (mEngine->createExecutionContext());
    assert(mContext);

    allocateBindings();
}

void Driver::buildNetwork(INetworkDefinition* network, const HostTensorMap& params)
{
    auto inputTensor = network->addInput("input", DataType::kFLOAT, Dims3{768, 1, 1});
    auto W_ = *params.at("l0_attention_self_query_kernel");
    auto B_ = *params.at("l0_attention_self_query_bias");
    Weights weights{DataType::kFLOAT, W_.mData, static_cast<int64_t>(W_.mSize)};
    Weights bias{DataType::kFLOAT, nullptr, 0};

    auto fc = network->addFullyConnected(*inputTensor, 768, weights, bias);
    fc->getOutput(0)->setName("output");
    network->markOutput(*fc->getOutput(0));
}

void Driver::h2d(const HostTensorMap& hostBuffers, cudaStream_t stream)
{
    for (auto& kv : hostBuffers)
    {
        const int idx = mEngine->getBindingIndex(kv.first.c_str());
        assert(idx >= 0);
        assert(mEngine->getBindingDataType(idx) == kv.second->mType);
        const size_t len = kv.second->mNbBytes;
        CHECK(cudaMemcpyAsync(mBuffers[idx], kv.second->mData, len, cudaMemcpyHostToDevice, stream));
        gLogVerbose << "Binding: " << kv.first << ", idx: " << idx << ", uploading " << len << " bytes" << std::endl;
    }
}

void Driver::d2h(HostTensorMap& hostBuffers, cudaStream_t stream)
{
    for (auto& kv : hostBuffers)
    {
        const int idx = mEngine->getBindingIndex(kv.first.c_str());
        assert(idx >= 0);
        assert(mEngine->getBindingDataType(idx) == kv.second->mType);
        const size_t len = kv.second->mNbBytes;
        CHECK(cudaMemcpyAsync(kv.second->mData, mBuffers[idx], len, cudaMemcpyDeviceToHost, stream));
        gLogVerbose << "Binding: " << kv.first << ", idx: " << idx << ", downloading " << len << " bytes" << std::endl;
    }
}

void Driver::benchmark(const HostTensorMap& inCfg, HostTensorMap& outCfg, const int batchSize, cudaStream_t stream,
    vector<float>& timesTotal, vector<float>& timesCompute, const bool withMemcpy)
{
    const int numRuns = timesTotal.size();
    assert(numRuns == timesCompute.size());
    assert(numRuns > 0);

    void** bs = mBuffers.data();

    vector<cudaEvent_t> startsTotal(numRuns);
    vector<cudaEvent_t> stopsTotal(numRuns);
    vector<cudaEvent_t> startsCompute(numRuns);
    vector<cudaEvent_t> stopsCompute(numRuns);

    for (int it = 0; it < numRuns; it++)
    {
        cudaEventCreate(&startsTotal[it]);
        cudaEventCreate(&stopsTotal[it]);
        cudaEventCreate(&startsCompute[it]);
        cudaEventCreate(&stopsCompute[it]);
    }

    cudaProfilerStart();
    if (withMemcpy)
    {
        for (int it = 0; it < numRuns; it++)
        {
            CHECK(cudaEventRecord(startsTotal[it], stream));
            h2d(inCfg, stream);
            CHECK(cudaEventRecord(startsCompute[it], stream));
            infer(batchSize, stream);
            CHECK(cudaEventRecord(stopsCompute[it], stream));
            d2h(outCfg, stream);
            CHECK(cudaEventRecord(stopsTotal[it], stream));
        }
    }
    else
    {
        for (int it = 0; it < numRuns; it++)
        {
            CHECK(cudaEventRecord(startsCompute[it], stream));
            infer(batchSize, stream);
            CHECK(cudaEventRecord(stopsCompute[it], stream));
        }
    }
    CHECK(cudaDeviceSynchronize());

    cudaProfilerStop();
    float msCompute = 0;
    float msTotal = 0;
    for (int it = 0; it < numRuns; it++)
    {
        cudaEventElapsedTime(&msCompute, startsCompute[it], stopsCompute[it]);
        timesCompute[it] = msCompute;

        msTotal = msCompute;
        if (withMemcpy)
        {
            cudaEventElapsedTime(&msTotal, startsTotal[it], stopsTotal[it]);
        }
        timesTotal[it] = msTotal;

        cudaEventDestroy(startsTotal[it]);
        cudaEventDestroy(stopsTotal[it]);
        cudaEventDestroy(startsCompute[it]);
        cudaEventDestroy(stopsCompute[it]);

        gLogInfo << "Run " << it << "; Total: " << timesTotal[it] << "ms Comp.only: " << timesCompute[it] << "ms" << std::endl;
    }
}

void Driver::infer(const int batchSize, cudaStream_t stream)
{
    mContext->enqueueV2(mBuffers.data(), stream, nullptr);
}

void Driver::infer(const HostTensorMap& inCfg, HostTensorMap& outCfg, const int batchSize, cudaStream_t stream)
{
    h2d(inCfg, stream);
    infer(batchSize, stream);
    d2h(outCfg, stream);
}

Driver::~Driver()
{
    for (auto b : mBuffers)
    {
        CHECK(cudaFree(b));
    }
}

void Driver::serializeEngine(const std::string& enginePath) const
{
    ofstream engineFile(enginePath, ios::binary);
    if (!engineFile)
    {
        gLogError << "Cannot open engine file: " << enginePath << endl;
    }

    IHostMemory* serializedEngine{mContext->getEngine().serialize()};
    if (serializedEngine == nullptr)
    {
        gLogError << "Engine serialization failed" << endl;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    serializedEngine->destroy();
}

Driver::Driver(const std::string& enginePath)
    : mMaxBatchSize(-1)
    , mMaxWorkspaceSize(-1)
    , mUseFp16(false)
{
    ifstream input(enginePath, ios::binary);
    if (!input)
    {
        gLogError << "Invalid engine file";
    }
    vector<char> bytes(istreambuf_iterator<char>(input), {});

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(bytes.data(), bytes.size(), nullptr);
    assert(engine);

    mContext = (engine->createExecutionContext());
    assert(mContext);
    mMaxBatchSize = engine->getMaxBatchSize();

    engine->destroy();
    runtime->destroy();
}

DynamicDriver::DynamicDriver(
     const bool useFp16, const size_t maxWorkspaceSize, const OptProfiles& optProfiles)
    : Driver(1, useFp16, maxWorkspaceSize)
    , mOptProfiles(optProfiles)
{
}

DynamicDriver::DynamicDriver(const std::string& enginePath)
    : Driver(enginePath)
{
}

NetworkDefinitionCreationFlags DynamicDriver::getNetworkFlags() const
{
    return (1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
}

IBuilderConfig* DynamicDriver::getBuilderConfig() const
{
    auto config = Driver::getBuilderConfig();
    for (auto& optProfile : mOptProfiles)
    {
        auto profile = mBuilder->createOptimizationProfile();
        for (auto& kv : optProfile)
        {
            profile->setDimensions(kv.first.c_str(), OptProfileSelector::kMIN, get<OPIDX_MIN>(kv.second));
            profile->setDimensions(kv.first.c_str(), OptProfileSelector::kMAX, get<OPIDX_MAX>(kv.second));
            profile->setDimensions(kv.first.c_str(), OptProfileSelector::kOPT, get<OPIDX_OPT>(kv.second));
        }
        config->addOptimizationProfile(profile);
    }
    return config;
}

void DynamicDriver::allocateBindings()
{
    // dynamic shapes: setting each input binding to its maximum binding dimensions
    // there should be a opt profile for each input
    for (auto kv : mOptProfiles[0])//assuming there is only one opt profile - take its max dims
    {
        auto iidx = mEngine->getBindingIndex(kv.first.c_str());
        mContext->setBindingDimensions(iidx, get<OPIDX_MAX>(kv.second));
    }
    assert(mContext->allInputDimensionsSpecified());

    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        auto bDims = mContext->getBindingDimensions(i);

        size_t vol = samplesCommon::volume(bDims);
        size_t elementSize = samplesCommon::getElementSize(mEngine->getBindingDataType(i));
        size_t allocationSize = vol * elementSize;
        gLogVerbose  << "Binding " << mEngine->getBindingName(i) << ": vol=" << vol << " wordSize="
            << elementSize << " allocSize=" << allocationSize << " bytes" << std::endl;
        mDeviceBuffers.emplace_back(DeviceBuffer(allocationSize, mEngine->getBindingDataType(i)));
        mBuffers.emplace_back(mDeviceBuffers.back().data());
    }
}
}
