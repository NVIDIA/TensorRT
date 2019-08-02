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
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
using namespace nvinfer1;
using namespace nvinfer1::plugin;

#include "batchedNMSPlugin/batchedNMSPlugin.h"
#include "batchTilePlugin/batchTilePlugin.h"
#include "cropAndResizePlugin/cropAndResizePlugin.h"
#include "flattenConcat/flattenConcat.h"
#include "gridAnchorPlugin/gridAnchorPlugin.h"
#include "nmsPlugin/nmsPlugin.h"
#include "normalizePlugin/normalizePlugin.h"
#include "nvFasterRCNN/nvFasterRCNNPlugin.h"
#include "priorBoxPlugin/priorBoxPlugin.h"
#include "proposalPlugin/proposalPlugin.h"
#include "regionPlugin/regionPlugin.h"
#include "reorgPlugin/reorgPlugin.h"

using nvinfer1::plugin::RPROIParams;

namespace nvinfer1
{

namespace plugin
{
ILogger* gLogger {};

// Instances of this class are statically constructed in initializePlugin.
// This ensures that each plugin is only registered a single time, as further calls to
// initializePlugin will be no-ops.
template <typename CreatorType>
class InitializePlugin
{
public:
    InitializePlugin(void* logger, const char* libNamespace)
        : mCreator{new CreatorType{}}
    {
        mCreator->setPluginNamespace(libNamespace);
        bool status = getPluginRegistry()->registerCreator(*mCreator, libNamespace);
        if (logger)
        {
            nvinfer1::plugin::gLogger = static_cast<nvinfer1::ILogger*>(logger);
            if (!status)
            {
                std::string errorMsg{"Could not register plugin creator:  " + std::string(mCreator->getPluginName())
                    + " in namespace: " + std::string{mCreator->getPluginNamespace()}};
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
            }
            else
            {
                std::string verboseMsg{
                    "Plugin Creator registration succeeded - " + std::string{mCreator->getPluginName()}};
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
            }
        }
    }

    InitializePlugin(const InitializePlugin&) = delete;
    InitializePlugin(InitializePlugin&&) = delete;

private:
    std::unique_ptr<CreatorType> mCreator;
};

template <typename CreatorType>
void initializePlugin(void* logger, const char* libNamespace)
{
    static InitializePlugin<CreatorType> plugin{logger, libNamespace};
}

} // namespace plugin
} // namespace nvinfer1

extern "C" {
bool initLibNvInferPlugins(void* logger, const char* libNamespace)
{
    initializePlugin<nvinfer1::plugin::GridAnchorPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::NMSPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::ReorgPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::RegionPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::PriorBoxPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::NormalizePluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::RPROIPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::BatchedNMSPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::FlattenConcatPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::CropAndResizePluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::ProposalPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::BatchTilePluginCreator>(logger, libNamespace);
    return true;
}
} // extern "C"
