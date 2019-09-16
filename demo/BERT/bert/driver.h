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

#ifndef TRT_DRIVER_H
#define TRT_DRIVER_H

#include "buffers.h"
#include <typeinfo>
#include <vector>

namespace bert
{

struct HostTensor
{
    void* mData{nullptr};
    size_t mNbBytes;
    size_t mSize;
    nvinfer1::DataType mType;
    std::vector<size_t> mShape;

    HostTensor(void* data, const nvinfer1::DataType type, const std::vector<size_t>& shape);
};

using HostTensorMap = std::map<std::string, std::shared_ptr<HostTensor>>;

struct InferDeleter1
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter1>;

struct Driver
{
    std::vector<void*> mBuffers;
    std::vector<samplesCommon::DeviceBuffer> mDeviceBuffers;

    nvinfer1::IBuilder* mBuilder{nullptr};
    nvinfer1::ICudaEngine* mEngine{nullptr};
    nvinfer1::IExecutionContext* mContext{nullptr};

    int mMaxBatchSize;
    size_t mMaxWorkspaceSize;
    bool mUseFp16;

    Driver(const int maxBatchSize, const bool useFp16, const size_t maxWorkspaceSize);

    Driver(const std::string& enginePath);

    virtual ~Driver();

    virtual void buildNetwork(INetworkDefinition* network, const HostTensorMap& in);

    virtual nvinfer1::NetworkDefinitionCreationFlags getNetworkFlags() const;
    virtual nvinfer1::IBuilderConfig* getBuilderConfig() const;
    virtual void allocateBindings();

    void init(const HostTensorMap& params);

    void h2d(const HostTensorMap& inCfg, cudaStream_t stream);

    void d2h(HostTensorMap& outCfg, cudaStream_t stream);

    void infer(const int batchSize, cudaStream_t stream);

    void infer(const HostTensorMap& inCfg, HostTensorMap& outCfg, const int batchSize, cudaStream_t stream);

    void benchmark(const HostTensorMap& inCfg, HostTensorMap& outCfg, const int batchSize, cudaStream_t stream,
        std::vector<float>& timesTotal, std::vector<float>& timesCompute, const bool withMemcpy = true);

    void serializeEngine(const std::string& enginePath) const;
};

constexpr uint32_t OPIDX_MIN = 0;
constexpr uint32_t OPIDX_MAX = 1;
constexpr uint32_t OPIDX_OPT = 2;

using OptProfile = std::tuple<nvinfer1::Dims, nvinfer1::Dims, nvinfer1::Dims>;
using OptProfileMap = std::map<std::string, OptProfile>;
using OptProfiles = std::vector<OptProfileMap>;

struct DynamicDriver : Driver
{

    OptProfiles mOptProfiles;

    DynamicDriver(const bool useFp16, const size_t maxWorkspaceSize, const OptProfiles& optProfiles);

    DynamicDriver(const std::string& enginePath);

    nvinfer1::NetworkDefinitionCreationFlags getNetworkFlags() const override;
    nvinfer1::IBuilderConfig* getBuilderConfig() const override;
    void allocateBindings() override;
};
}
#endif // TRT_DRIVER_H
