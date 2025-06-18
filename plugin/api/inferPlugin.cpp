/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/plugin.h"
#include "roiAlignPlugin/roiAlignPlugin.h"
#include "batchTilePlugin/batchTilePlugin.h"
#include "batchedNMSPlugin/batchedNMSPlugin.h"
#include "clipPlugin/clipPlugin.h"
#include "coordConvACPlugin/coordConvACPlugin.h"
#include "cropAndResizePlugin/cropAndResizePlugin.h"
#include "cropAndResizePlugin/cropAndResizePluginLegacy.h"
#include "decodeBbox3DPlugin/decodeBbox3D.h"
#include "detectionLayerPlugin/detectionLayerPlugin.h"
#include "efficientNMSPlugin/efficientNMSPlugin.h"
#include "efficientNMSPlugin/tftrt/efficientNMSExplicitTFTRTPlugin.h"
#include "efficientNMSPlugin/tftrt/efficientNMSImplicitTFTRTPlugin.h"
#include "flattenConcat/flattenConcat.h"
#include "generateDetectionPlugin/generateDetectionPlugin.h"
#include "gridAnchorPlugin/gridAnchorPlugin.h"
#include "instanceNormalizationPlugin/instanceNormalizationPlugin.h"
#include "instanceNormalizationPlugin/instanceNormalizationPluginLegacy.h"
#include "leakyReluPlugin/lReluPlugin.h"
#include "modulatedDeformConvPlugin/modulatedDeformConvPlugin.h"
#include "modulatedDeformConvPlugin/modulatedDeformConvPluginLegacy.h"
#include "multilevelCropAndResizePlugin/multilevelCropAndResizePlugin.h"
#include "multilevelProposeROI/multilevelProposeROIPlugin.h"
#include "multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPlugin.h"
#include "multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPluginLegacy.h"
#include "nmsPlugin/nmsPlugin.h"
#include "normalizePlugin/normalizePlugin.h"
#include "nvFasterRCNN/nvFasterRCNNPlugin.h"
#include "pillarScatterPlugin/pillarScatter.h"
#include "priorBoxPlugin/priorBoxPlugin.h"
#include "proposalLayerPlugin/proposalLayerPlugin.h"
#include "proposalPlugin/proposalPlugin.h"
#include "pyramidROIAlignPlugin/pyramidROIAlignPlugin.h"
#include "regionPlugin/regionPlugin.h"
#include "reorgPlugin/reorgPlugin.h"
#include "resizeNearestPlugin/resizeNearestPlugin.h"
#include "roiAlignPlugin/roiAlignPlugin.h"
#include "roiAlignPlugin/roiAlignPluginLegacy.h"
#include "scatterElementsPlugin/scatterElementsPlugin.h"
#include "scatterElementsPlugin/scatterElementsPluginLegacy.h"
#include "scatterPlugin/scatterPlugin.h"
#include "specialSlicePlugin/specialSlicePlugin.h"
#include "splitPlugin/split.h"
#include "voxelGeneratorPlugin/voxelGenerator.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_set>
using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::pluginInternal;

using nvinfer1::plugin::RPROIParams;

namespace nvinfer1::plugin
{

extern ILogger* gLogger;

} // namespace nvinfer1::plugin

namespace
{
// This singleton ensures that each plugin is only registered once for a given
// namespace and type, and attempts of duplicate registration are ignored.
class PluginCreatorRegistry
{
public:
    static PluginCreatorRegistry& getInstance()
    {
        static PluginCreatorRegistry instance;
        return instance;
    }

    template <typename CreatorType>
    void addPluginCreator(void* logger, char const* libNamespace)
    {
        // Make accesses to the plugin creator registry thread safe
        std::lock_guard<std::mutex> lock(mRegistryLock);

        std::string errorMsg;
        std::string verboseMsg;

        std::unique_ptr<CreatorType> pluginCreator{new CreatorType{}};
        pluginCreator->setPluginNamespace(libNamespace);

        nvinfer1::plugin::gLogger = static_cast<nvinfer1::ILogger*>(logger);
        std::string pluginType = std::string{pluginCreator->getPluginNamespace()}
            + "::" + std::string{pluginCreator->getPluginName()} + " version "
            + std::string{pluginCreator->getPluginVersion()};

        if (mRegistryList.find(pluginType) == mRegistryList.end())
        {
            bool status = getPluginRegistry()->registerCreator(*pluginCreator, libNamespace);
            if (status)
            {
                mRegistry.push(std::move(pluginCreator));
                mRegistryList.insert(pluginType);
                verboseMsg = "Registered plugin creator - " + pluginType;
            }
            else
            {
                errorMsg = "Could not register plugin creator -  " + pluginType;
            }
        }
        else
        {
            verboseMsg = "Plugin creator already registered - " + pluginType;
        }

        if (logger)
        {
            if (!errorMsg.empty())
            {
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
            }
            if (!verboseMsg.empty())
            {
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
            }
        }
    }

    ~PluginCreatorRegistry()
    {
        std::lock_guard<std::mutex> lock(mRegistryLock);

        // Release pluginCreators in LIFO order of registration.
        while (!mRegistry.empty())
        {
            mRegistry.pop();
        }
        mRegistryList.clear();
    }

private:
    PluginCreatorRegistry() {}

    std::mutex mRegistryLock;
    std::stack<std::unique_ptr<IPluginCreatorInterface>> mRegistry;
    std::unordered_set<std::string> mRegistryList;

public:
    PluginCreatorRegistry(PluginCreatorRegistry const&) = delete;
    void operator=(PluginCreatorRegistry const&) = delete;
};

template <typename CreatorType>
void initializePlugin(void* logger, char const* libNamespace)
{
    PluginCreatorRegistry::getInstance().addPluginCreator<CreatorType>(logger, libNamespace);
}

} // namespace
// New Plugin APIs

extern "C"
{
    bool initLibNvInferPlugins(void* logger, char const* libNamespace)
    {
        initializePlugin<nvinfer1::plugin::ROIAlignV3PluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CropAndResizeDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::InstanceNormalizationV3PluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ScatterElementsPluginV3Creator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::MultiscaleDeformableAttnPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ModulatedDeformableConvPluginDynamicCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::BatchedNMSDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::BatchedNMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::BatchTilePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ClipPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CoordConvACPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CropAndResizeDynamicPluginLegacyCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CropAndResizePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::DecodeBbox3DPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::DetectionLayerPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSExplicitTFTRTPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSImplicitTFTRTPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSONNXPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::FlattenConcatPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::GenerateDetectionPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::GridAnchorPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::GridAnchorRectPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::InstanceNormalizationPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::InstanceNormalizationPluginCreatorV2>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::LReluPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ModulatedDeformableConvPluginDynamicLegacyCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::MultilevelCropAndResizePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::MultilevelProposeROIPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::MultiscaleDeformableAttnPluginCreatorLegacy>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::NMSDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::NMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::NormalizePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::PillarScatterPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::PriorBoxPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalLayerPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::PyramidROIAlignPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::RegionPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ReorgDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ReorgStaticPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ResizeNearestPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ROIAlignPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::RPROIPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ScatterElementsPluginV2Creator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ScatterNDPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::SpecialSlicePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::SplitPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::VoxelGeneratorPluginCreator>(logger, libNamespace);
        return true;
    }
} // extern "C"
