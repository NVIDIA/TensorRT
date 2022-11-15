/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "batchTilePlugin/batchTilePlugin.h"
#include "batchedNMSPlugin/batchedNMSPlugin.h"
#include "clipPlugin/clipPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/plugin.h"
#include "coordConvACPlugin/coordConvACPlugin.h"
#include "cropAndResizePlugin/cropAndResizePlugin.h"
#include "decodeBbox3DPlugin/decodeBbox3D.h"
#include "detectionLayerPlugin/detectionLayerPlugin.h"
#include "efficientNMSPlugin/efficientNMSPlugin.h"
#include "efficientNMSPlugin/tftrt/efficientNMSExplicitTFTRTPlugin.h"
#include "efficientNMSPlugin/tftrt/efficientNMSImplicitTFTRTPlugin.h"
#include "flattenConcat/flattenConcat.h"
#include "generateDetectionPlugin/generateDetectionPlugin.h"
#include "gridAnchorPlugin/gridAnchorPlugin.h"
#include "instanceNormalizationPlugin/instanceNormalizationPlugin.h"
#include "multilevelCropAndResizePlugin/multilevelCropAndResizePlugin.h"
#include "multilevelProposeROI/multilevelProposeROIPlugin.h"
#include "multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPlugin.h"
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

#include "batchTilePlugin.h"
#include "batchedNMSPlugin.h"
#include "clipPlugin.h"
#include "coordConvACPlugin.h"
#include "cropAndResizePlugin.h"
#include "decodeBbox3D.h"
#include "detectionLayerPlugin.h"
#include "efficientNMSPlugin.h"
#include "tftrt/efficientNMSImplicitTFTRTPlugin.h"
#include "tftrt/efficientNMSExplicitTFTRTPlugin.h"
#include "flattenConcat.h"
#include "fmhcaPlugin.h"
#include "generateDetectionPlugin.h"
#include "gridAnchorPlugin.h"
#include "groupNormPlugin.h"
#include "instanceNormalizationPlugin.h"
#include "layerNormPlugin.h"
#include "lReluPlugin.h"
#include "multiHeadFlashAttentionPlugin/fmhaPlugin.h"
#include "multilevelCropAndResizePlugin.h"
#include "multilevelProposeROIPlugin.h"
#include "multiscaleDeformableAttnPlugin.h"
#include "nmsPlugin.h"
#include "normalizePlugin.h"
#include "nvFasterRCNNPlugin.h"
#include "pillarScatter.h"
#include "priorBoxPlugin.h"
#include "proposalLayerPlugin.h"
#include "proposalPlugin.h"
#include "pyramidROIAlignPlugin.h"
#include "regionPlugin.h"
#include "reorgPlugin.h"
#include "resizeNearestPlugin.h"
#include "roiAlignPlugin.h"
#include "scatterPlugin.h"
#include "seqLen2SpatialPlugin.h"
#include "specialSlicePlugin.h"
#include "split.h"
#include "splitGeLUPlugin.h"
#include "voxelGenerator.h"

using nvinfer1::plugin::RPROIParams;

namespace nvinfer1
{

namespace plugin
{

extern ILogger* gLogger;

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
    std::stack<std::unique_ptr<IPluginCreator>> mRegistry;
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

} // namespace plugin
} // namespace nvinfer1
// New Plugin APIs

extern "C"
{
<<<<<<< HEAD:plugin/api/InferPlugin.cpp
    bool initLibNvInferPlugins(void* logger, const char* libNamespace)
    {
        initializePlugin<nvinfer1::plugin::BatchTilePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::BatchedNMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::BatchedNMSDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ClipPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CoordConvACPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CropAndResizePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CropAndResizeDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::DecodeBbox3DPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::DetectionLayerPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSONNXPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSExplicitTFTRTPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::EfficientNMSImplicitTFTRTPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::FlattenConcatPluginCreator>(logger, libNamespace);
#if defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
        initializePlugin<nvinfer1::plugin::FMHAPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::FMHCAPluginCreator>(logger, libNamespace);
#endif // defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
        initializePlugin<nvinfer1::plugin::GenerateDetectionPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::GridAnchorPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::GridAnchorRectPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::GroupNormPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::InstanceNormalizationPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::LayerNormPluginCreator>(logger, libNamespace);
=======
    IPluginV2* createRPNROIPlugin(int32_t featureStride, int32_t preNmsTop, int32_t nmsMaxOut, float iouThreshold,
        float minBoxSize, float spatialScale, nvinfer1::DimsHW pooling, nvinfer1::Weights anchorRatios,
        nvinfer1::Weights anchorScales)
    {
        PLUGIN_API_CHECK_RETVAL(anchorRatios.count > 0 && anchorScales.count > 0, nullptr);
        PLUGIN_API_CHECK_RETVAL(pooling.d[0] > 0 && pooling.d[1] > 0, nullptr);
        return new RPROIPlugin(RPROIParams{pooling.d[0], pooling.d[1], featureStride, preNmsTop, nmsMaxOut,
                                   static_cast<int>(anchorRatios.count), static_cast<int>(anchorScales.count),
                                   iouThreshold, minBoxSize, spatialScale},
            const_cast<float*>((float const*) anchorRatios.values),
            const_cast<float*>((float const*) anchorScales.values));
    }

    IPluginV2* createNormalizePlugin(
        nvinfer1::Weights const* weights, bool acrossSpatial, bool channelShared, float eps)
    {
        PLUGIN_API_CHECK_RETVAL(weights[0].count >= 1, nullptr);
        return new Normalize(weights, 1, acrossSpatial, channelShared, eps);
    }

    IPluginV2* createPriorBoxPlugin(PriorBoxParameters param)
    {
        PLUGIN_API_CHECK_RETVAL(param.numMinSize > 0 && param.minSize != nullptr, nullptr);
        return new PriorBox(param);
    }

    IPluginV2* createAnchorGeneratorPlugin(GridAnchorParameters* param, int32_t numLayers)
    {
        PLUGIN_API_CHECK_RETVAL(numLayers > 0, nullptr);
        PLUGIN_API_CHECK_RETVAL(param != nullptr, nullptr);
        return new GridAnchorGenerator(param, numLayers, "GridAnchor_TRT");
    }

    IPluginV2* createNMSPlugin(DetectionOutputParameters param)
    {
        return new DetectionOutput(param);
    }

    IPluginV2* createReorgPlugin(int32_t stride)
    {
        PLUGIN_API_CHECK_RETVAL(stride >= 0, nullptr);
        return new Reorg(stride);
    }

    IPluginV2* createRegionPlugin(RegionParameters params)
    {
        return new Region(params);
    }

    IPluginV2* createBatchedNMSPlugin(NMSParameters params)
    {
        return new BatchedNMSPlugin(params);
    }

    IPluginV2* createSplitPlugin(int32_t axis, int32_t* output_lengths, int32_t noutput)
    {
        return new SplitPlugin(axis, output_lengths, noutput);
    }

    IPluginV2* createInstanceNormalizationPlugin(
        float epsilon, nvinfer1::Weights scale_weights, nvinfer1::Weights bias_weights)
    {
        return new InstanceNormalizationPlugin(epsilon, scale_weights, bias_weights);
    }

    bool initLibNvInferPlugins(void* logger, char const* libNamespace)
    {
        initializePlugin<nvinfer1::plugin::BatchedNMSDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::BatchedNMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::BatchTilePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ClipPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CoordConvACPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::CropAndResizeDynamicPluginCreator>(logger, libNamespace);
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
>>>>>>> 707a6ff02f... TRT-18379: add getPluginCreators and LoggerFinder for vfc plugin:plugin/api/inferPlugin.cpp
        initializePlugin<nvinfer1::plugin::LReluPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::MultilevelCropAndResizePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::MultilevelProposeROIPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::MultiscaleDeformableAttnPluginCreator>(logger, libNamespace);
<<<<<<< HEAD:plugin/api/InferPlugin.cpp
        initializePlugin<nvinfer1::plugin::NMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::NMSDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::NormalizePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::PillarScatterPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::PriorBoxPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalLayerPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalDynamicPluginCreator>(logger, libNamespace);
=======
        initializePlugin<nvinfer1::plugin::NMSDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::NMSPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::NormalizePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::PillarScatterPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::PriorBoxPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalDynamicPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalLayerPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ProposalPluginCreator>(logger, libNamespace);
>>>>>>> 707a6ff02f... TRT-18379: add getPluginCreators and LoggerFinder for vfc plugin:plugin/api/inferPlugin.cpp
        initializePlugin<nvinfer1::plugin::PyramidROIAlignPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::RegionPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ReorgPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ResizeNearestPluginCreator>(logger, libNamespace);
<<<<<<< HEAD:plugin/api/InferPlugin.cpp
        initializePlugin<nvinfer1::plugin::RPROIPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ROIAlignPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ScatterNDPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::SeqLen2SpatialPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::SpecialSlicePluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::SplitGeLUPluginCreator>(logger, libNamespace);
=======
        initializePlugin<nvinfer1::plugin::ROIAlignPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::RPROIPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::ScatterNDPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::SpecialSlicePluginCreator>(logger, libNamespace);
>>>>>>> 707a6ff02f... TRT-18379: add getPluginCreators and LoggerFinder for vfc plugin:plugin/api/inferPlugin.cpp
        initializePlugin<nvinfer1::plugin::SplitPluginCreator>(logger, libNamespace);
        initializePlugin<nvinfer1::plugin::VoxelGeneratorPluginCreator>(logger, libNamespace);
        return true;
    }
<<<<<<< HEAD:plugin/api/InferPlugin.cpp
=======

>>>>>>> 707a6ff02f... TRT-18379: add getPluginCreators and LoggerFinder for vfc plugin:plugin/api/inferPlugin.cpp
} // extern "C"
