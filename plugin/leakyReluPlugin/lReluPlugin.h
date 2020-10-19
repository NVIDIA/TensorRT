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

#ifndef TRT_L_RELU_PLUGIN_H
#define TRT_L_RELU_PLUGIN_H
#include "NvInfer.h"
#include "kernel.h"
#include "plugin.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class LReLU : public BasePlugin
{
public:
    LReLU(float negSlope);

    LReLU(const void* buffer, size_t length);

    ~LReLU() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type,
        PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2* clone() const override;

private:
    float mNegSlope;
    int mBatchDim;
};

class LReluPluginCreator : public BaseCreator
{
public:
    LReluPluginCreator();

    ~LReluPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    float negSlope{};
    static std::vector<PluginField> mPluginAttributes;
};

typedef LReLU PReLU; // Temporary. For backward compatibilty.
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_L_RELU_PLUGIN_H
