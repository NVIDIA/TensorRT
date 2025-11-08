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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "ErrorRecorder.h"
#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleOptions.h"
#include "sampleUtils.h"

#if ENABLE_UNIFIED_BUILDER
#include "NvInferConsistency.h"
#include "safeErrorRecorder.h"
#endif

using namespace nvinfer1;

namespace sample
{

namespace
{
class FileStreamWriter final : public nvinfer1::IStreamWriter
{
protected:
    std::ofstream mStream;
    int64_t mTotalWrittenSize;

public:
    FileStreamWriter(std::string const& path)
        : mStream(path, std::ios::binary)
        , mTotalWrittenSize(0)
    {
    }

    virtual int64_t write(void const* data, int64_t nbBytes) final
    {
        SMP_RETVAL_IF_FALSE(
            (mStream.is_open() && mStream.good()), "Cannot write to FileStreamWriter", -1, sample::gLogError);
        auto const* src = reinterpret_cast<char const*>(data);
        mStream.write(src, nbBytes);
        mTotalWrittenSize += nbBytes;
        return nbBytes;
    }

    int64_t finalize()
    {
        mStream.close();
        return mTotalWrittenSize;
    }
};

std::map<std::string, float> readScalesFromCalibrationCache(std::string const& calibrationFile)
{
    std::map<std::string, float> tensorScales;
    std::ifstream cache{calibrationFile};
    if (!cache.is_open())
    {
        sample::gLogError << "[TRT] Can not open provided calibration cache file" << std::endl;
        return tensorScales;
    }
    std::string line;
    while (std::getline(cache, line))
    {
        auto colonPos = line.find_last_of(':');
        if (colonPos != std::string::npos)
        {
            // Scales should be stored in calibration cache as 32-bit floating numbers encoded as 32-bit integers
            int32_t scalesAsInt = std::stoi(line.substr(colonPos + 2, 8), nullptr, 16);
            auto const tensorName = line.substr(0, colonPos);
            tensorScales[tensorName] = *reinterpret_cast<float*>(&scalesAsInt);
        }
    }
    cache.close();
    return tensorScales;
}
} // namespace

nvinfer1::ICudaEngine* LazilyDeserializedEngine::get()
{
    SMP_RETVAL_IF_FALSE(
        !mIsSafe, "Safe mode is enabled, but trying to get standard engine!", nullptr, sample::gLogError);

    if (mEngine == nullptr)
    {
        SMP_RETVAL_IF_FALSE(getAsyncFileReader().isOpen() || getFileReader().isOpen() || !getBlob().empty(),
            "Engine is empty. Nothing to deserialize!", nullptr, sample::gLogError);
        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using duration = std::chrono::duration<float>;
        time_point const deserializeStartTime{std::chrono::high_resolution_clock::now()};

        if (mLeanDLLPath.empty())
        {
            mRuntime.reset(createRuntime());
        }
        else
        {
            mParentRuntime.reset(createRuntime());
            ASSERT(mParentRuntime != nullptr);

            mRuntime.reset(mParentRuntime->loadRuntime(mLeanDLLPath.c_str()));
        }
        ASSERT(mRuntime != nullptr);
        if (mVersionCompatible)
        {
            // Application needs to opt into allowing deserialization of engines with embedded lean runtime.
            mRuntime->setEngineHostCodeAllowed(true);
        }

        if (!mTempdir.empty())
        {
            mRuntime->setTemporaryDirectory(mTempdir.c_str());
        }

        mRuntime->setTempfileControlFlags(mTempfileControls);
        SMP_RETVAL_IF_FALSE(mRuntime != nullptr, "runtime creation failed", nullptr, sample::gLogError);
        if (mDLACore != -1)
        {
            mRuntime->setDLACore(mDLACore);
        }
        mRuntime->setErrorRecorder(&gRecorder);
        for (auto const& pluginPath : mDynamicPlugins)
        {
            mRuntime->getPluginRegistry().loadLibrary(pluginPath.c_str());
        }

        if (getAsyncFileReader().isOpen())
        {
            mEngine.reset(mRuntime->deserializeCudaEngine(getAsyncFileReader()));
        }
        else if (getFileReader().isOpen())
        {
            mEngine.reset(mRuntime->deserializeCudaEngine(getFileReader()));
        }
        else
        {
            auto const& engineBlob = getBlob();
            mEngine.reset(mRuntime->deserializeCudaEngine(engineBlob.data, engineBlob.size));
        }
        SMP_RETVAL_IF_FALSE(mEngine != nullptr, "Engine deserialization failed", nullptr, sample::gLogError);

        time_point const deserializeEndTime{std::chrono::high_resolution_clock::now()};
        sample::gLogInfo << "Engine deserialized in " << duration(deserializeEndTime - deserializeStartTime).count()
                         << " sec." << std::endl;
    }

    return mEngine.get();
}

nvinfer1::ICudaEngine* LazilyDeserializedEngine::release()
{
    return mEngine.release();
}

bool LazilyDeserializedEngine::checkDLASafe()
{
    ASSERT(sample::hasSafeRuntime());

    SMP_RETVAL_IF_FALSE(mDLACore == -1, "Safe DLA engine built with kDLA_STANDALONE should not be run via TRT!", false,
        sample::gLogError);

    return true;
}

void setTensorScalesFromCalibration(nvinfer1::INetworkDefinition& network, std::vector<IOFormat> const& inputFormats,
    std::vector<IOFormat> const& outputFormats, std::string const& calibrationFile)
{
    auto const tensorScales = readScalesFromCalibrationCache(calibrationFile);
    bool const broadcastInputFormats = broadcastIOFormats(inputFormats, network.getNbInputs());
    for (int32_t i = 0, n = network.getNbInputs(); i < n; ++i)
    {
        int32_t formatIdx = broadcastInputFormats ? 0 : i;
        if (!inputFormats.empty() && inputFormats[formatIdx].first == DataType::kINT8)
        {
            auto* input = network.getInput(i);
            auto const calibScale = tensorScales.at(input->getName());
            input->setDynamicRange(-127 * calibScale, 127 * calibScale);
        }
    }
    bool const broadcastOutputFormats = broadcastIOFormats(outputFormats, network.getNbOutputs());
    for (int32_t i = 0, n = network.getNbOutputs(); i < n; ++i)
    {
        int32_t formatIdx = broadcastOutputFormats ? 0 : i;
        if (!outputFormats.empty() && outputFormats[formatIdx].first == DataType::kINT8)
        {
            auto* output = network.getOutput(i);
            auto const calibScale = tensorScales.at(output->getName());
            output->setDynamicRange(-127 * calibScale, 127 * calibScale);
        }
    }
}


//!
//! \brief Generate a network definition for a given model
//!
//! \param[in] model Model options for this network
//! \param[in,out] network Network storing the parsed results
//! \param[in,out] err Error stream
//! \param[out] vcPluginLibrariesUsed If not nullptr, will be populated with paths to VC plugin libraries required by
//! the parsed network.
//!
//! \return Parser The parser used to initialize the network and that holds the weights for the network, or an invalid
//! parser (the returned parser converts to false if tested)
//!
//! Constant input dimensions in the model must not be changed in the corresponding
//! network definition, because its correctness may rely on the constants.
//!
//! \see Parser::operator bool()
//!
Parser modelToNetwork(ModelOptions const& model, BuildOptions const& build, nvinfer1::INetworkDefinition& network,
    std::ostream& err, std::vector<std::string>* vcPluginLibrariesUsed)
{
    sample::gLogInfo << "Start parsing network model." << std::endl;
    auto const tBegin = std::chrono::high_resolution_clock::now();

    Parser parser;
    switch (model.baseModel.format)
    {
    case ModelFormat::kONNX:
    {
        using namespace nvonnxparser;
        parser.onnxParser.reset(createONNXParser(network));
        ASSERT(parser.onnxParser != nullptr);
        // kNATIVE_INSTANCENORM is ON by default in the parser and must be cleared to use the plugin implementation.
        if (build.pluginInstanceNorm)
        {
            parser.onnxParser->clearFlag(OnnxParserFlag::kNATIVE_INSTANCENORM);
        }
        if (build.enableUInt8AsymmetricQuantizationDLA)
        {
            parser.onnxParser->setFlag(OnnxParserFlag::kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA);
        }
        if (!parser.onnxParser->parseFromFile(
                model.baseModel.model.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity())))
        {
            err << "Failed to parse onnx file" << std::endl;
            parser.onnxParser.reset();
        }
        if (vcPluginLibrariesUsed && parser.onnxParser.get())
        {
            int64_t nbPluginLibs;
            char const* const* pluginLibArray = parser.onnxParser->getUsedVCPluginLibraries(nbPluginLibs);
            if (nbPluginLibs >= 0)
            {
                vcPluginLibrariesUsed->reserve(nbPluginLibs);
                for (int64_t i = 0; i < nbPluginLibs; ++i)
                {
                    sample::gLogInfo << "Using VC plugin library " << pluginLibArray[i] << std::endl;
                    vcPluginLibrariesUsed->emplace_back(std::string{pluginLibArray[i]});
                }
            }
            else
            {
                sample::gLogWarning << "Failure to query VC plugin libraries required by parsed ONNX network"
                                    << std::endl;
            }
        }
        break;
    }
    case ModelFormat::kANY: break;
    }

    auto const tEnd = std::chrono::high_resolution_clock::now();
    float const parseTime = std::chrono::duration<float>(tEnd - tBegin).count();

    sample::gLogInfo << "Finished parsing network model. Parse time: " << parseTime << std::endl;
    return parser;
}

namespace
{

class RndInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    RndInt8Calibrator(int32_t batches, std::vector<int64_t>& elemCount, std::string const& cacheFile,
        nvinfer1::INetworkDefinition const& network, std::ostream& err);

    ~RndInt8Calibrator() override
    {
        for (auto& elem : mInputDeviceBuffers)
        {
            CHECK_WITH_STREAM(cudaFree(elem.second), mErr);
        }
    }

    bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override;

    int32_t getBatchSize() const noexcept override
    {
        return 1;
    }

    void const* readCalibrationCache(size_t& length) noexcept override;

    void writeCalibrationCache(void const*, size_t) noexcept override {}

private:
    int32_t mBatches{};
    int32_t mCurrentBatch{};
    std::string mCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
    std::ostream& mErr;
};

RndInt8Calibrator::RndInt8Calibrator(int32_t batches, std::vector<int64_t>& elemCount, std::string const& cacheFile,
    INetworkDefinition const& network, std::ostream& err)
    : mBatches(batches)
    , mCurrentBatch(0)
    , mCacheFile(cacheFile)
    , mErr(err)
{
    std::ifstream tryCache(cacheFile, std::ios::binary);
    if (tryCache.good())
    {
        return;
    }

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    auto gen = [&generator, &distribution]() { return distribution(generator); };

    for (int32_t i = 0; i < network.getNbInputs(); i++)
    {
        auto* input = network.getInput(i);
        std::vector<float> rnd_data(elemCount[i]);
        std::generate_n(rnd_data.begin(), elemCount[i], gen);

        void* data;
        CHECK_WITH_STREAM(cudaMalloc(&data, elemCount[i] * sizeof(float)), mErr);
        CHECK_WITH_STREAM(
            cudaMemcpy(data, rnd_data.data(), elemCount[i] * sizeof(float), cudaMemcpyHostToDevice), mErr);

        mInputDeviceBuffers.insert(std::make_pair(input->getName(), data));
    }
}

bool RndInt8Calibrator::getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
    if (mCurrentBatch >= mBatches)
    {
        return false;
    }

    for (int32_t i = 0; i < nbBindings; ++i)
    {
        bindings[i] = mInputDeviceBuffers[names[i]];
    }

    ++mCurrentBatch;

    return true;
}

void const* RndInt8Calibrator::readCalibrationCache(size_t& length) noexcept
{
    mCalibrationCache.clear();
    std::ifstream input(mCacheFile, std::ios::binary);
    input >> std::noskipws;
    if (input.good())
    {
        std::copy(
            std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
    }

    length = mCalibrationCache.size();
    return !mCalibrationCache.empty() ? mCalibrationCache.data() : nullptr;
}

bool setTensorDynamicRange(INetworkDefinition const& network, float inRange = 2.0F, float outRange = 4.0F)
{
    // Ensure that all layer inputs have a dynamic range.
    for (int32_t l = 0; l < network.getNbLayers(); l++)
    {
        auto* layer = network.getLayer(l);
        for (int32_t i = 0; i < layer->getNbInputs(); i++)
        {
            ITensor* input{layer->getInput(i)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input && !input->dynamicRangeIsSet())
            {
                // Concat should propagate dynamic range from outputs to inputs to avoid
                // Re-quantization during the concatenation
                auto dynRange = (layer->getType() == LayerType::kCONCATENATION) ? outRange : inRange;
                if (!input->setDynamicRange(-dynRange, dynRange))
                {
                    return false;
                }}
        }
        for (int32_t o = 0; o < layer->getNbOutputs(); o++)
        {
            ITensor* output{layer->getOutput(o)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output dynamic range.
                if (layer->getType() == LayerType::kPOOLING)
                {
                    if (!output->setDynamicRange(-inRange, inRange))
                    {
                        return false;
                    }
                }
                else
                {
                    if (!output->setDynamicRange(-outRange, outRange))
                    {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool isNonActivationType(nvinfer1::DataType const type)
{
    return type == nvinfer1::DataType::kINT32 || type == nvinfer1::DataType::kINT64 || type == nvinfer1::DataType::kBOOL
        || type == nvinfer1::DataType::kUINT8;
}
void setLayerPrecisions(INetworkDefinition& network, LayerPrecisions const& layerPrecisions)
{
    bool hasLayerPrecisionSkipped{false};
    for (int32_t layerIdx = 0; layerIdx < network.getNbLayers(); ++layerIdx)
    {
        auto* layer = network.getLayer(layerIdx);
        auto const layerName = layer->getName();
        auto exactMatch = layerPrecisions.find(layerName);
        auto plausibleMatch = findPlausible(layerPrecisions, layerName);
        if (exactMatch != layerPrecisions.end())
        {
            sample::gLogInfo << "Set layer " << layerName << " to precision " << exactMatch->second << std::endl;
            layer->setPrecision(exactMatch->second);
        }
        else if (plausibleMatch != layerPrecisions.end())
        {
            if (isNonActivationType(layer->getPrecision()))
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because the "
                                    << " default layer precision is of non-activation type." << std::endl;
                continue;
            }
            if (layer->getType() == nvinfer1::LayerType::kCONSTANT
                && (isNonActivationType(static_cast<IConstantLayer*>(layer)->getWeights().type)))
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because this "
                                    << "constant layer has weights of non-activation type." << std::endl;
                continue;
            }
            if (layer->getNbInputs() >= 1 && layer->getInput(0)->isShapeTensor())
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because this layer "
                                    << "operates on a shape tensor." << std::endl;
                continue;
            }
            if (layer->getNbInputs() >= 1 && isNonActivationType(layer->getInput(0)->getType())
                && layer->getNbOutputs() >= 1 && isNonActivationType(layer->getOutput(0)->getType()))
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because this "
                                    << "layer has input and output of non-activation type." << std::endl;
                continue;
            }
            // All heuristics passed. Set the layer precision.
            sample::gLogInfo << "Set layer " << layerName << " to precision " << plausibleMatch->second << std::endl;
            layer->setPrecision(plausibleMatch->second);
        }
    }

    if (hasLayerPrecisionSkipped)
    {
        sample::gLogInfo << "Skipped setting precisions for some layers. Check verbose logs for more details."
                         << std::endl;
    }
}

void setLayerOutputTypes(INetworkDefinition& network, LayerOutputTypes const& layerOutputTypes)
{
    bool const hasGlobalOutputType{layerOutputTypes.find("*") != layerOutputTypes.end()};
    auto const globalOutputType = hasGlobalOutputType ? layerOutputTypes.at("*").at(0) : nvinfer1::DataType::kFLOAT;
    bool hasLayerOutputTypeSkipped{false};
    for (int32_t layerIdx = 0; layerIdx < network.getNbLayers(); ++layerIdx)
    {
        auto* layer = network.getLayer(layerIdx);
        auto const layerName = layer->getName();
        auto const nbOutputs = layer->getNbOutputs();
        auto exactMatch = layerOutputTypes.find(layerName);
        auto plausibleMatch = findPlausible(layerOutputTypes, layerName);
        if (exactMatch != layerOutputTypes.end())
        {
            auto const& outputTypes = exactMatch->second;
            bool const isBroadcast = (outputTypes.size() == 1);
            if (!isBroadcast && static_cast<int32_t>(outputTypes.size()) != nbOutputs)
            {
                sample::gLogError << "Layer " << layerName << " has " << nbOutputs << " outputs but "
                                  << outputTypes.size() << " output types are given in --layerOutputTypes flag."
                                  << std::endl;
                throw std::invalid_argument("Invalid --layerOutputTypes flag.");
            }
            for (int32_t outputIdx = 0; outputIdx < nbOutputs; ++outputIdx)
            {
                auto const outputType = outputTypes.at(isBroadcast ? 0 : outputIdx);
                sample::gLogInfo << "Set output " << outputIdx << " of layer " << layerName << " to type " << outputType
                                 << std::endl;
                layer->setOutputType(outputIdx, outputType);
            }
        }
        else if (plausibleMatch != layerOutputTypes.end())
        {
            auto const& outputTypes = plausibleMatch->second;
            bool const isBroadcast = (outputTypes.size() == 1);

            // We should not set the layer output types if its default precision is INT32 or Bool.
            if (layer->getPrecision() == nvinfer1::DataType::kINT32
                || layer->getPrecision() == nvinfer1::DataType::kBOOL)
            {
                hasLayerOutputTypeSkipped = true;
                sample::gLogVerbose << "Skipped setting output types for layer " << layerName << " because the "
                                    << " default layer precision is INT32 or Bool." << std::endl;
                continue;
            }
            // We should not set the constant layer output types if its weights are in INT32.
            if (layer->getType() == nvinfer1::LayerType::kCONSTANT
                && static_cast<IConstantLayer*>(layer)->getWeights().type == nvinfer1::DataType::kINT32)
            {
                hasLayerOutputTypeSkipped = true;
                sample::gLogVerbose << "Skipped setting output types for layer " << layerName << " because this "
                                    << "constant layer has INT32 weights." << std::endl;
                continue;
            }
            for (int32_t outputIdx = 0; outputIdx < nbOutputs; ++outputIdx)
            {
                // We should not set the output type if the output is a shape tensor.
                if (layer->getOutput(0)->isShapeTensor())
                {
                    hasLayerOutputTypeSkipped = true;
                    sample::gLogVerbose << "Skipped setting output type for output " << outputIdx << " of layer "
                                        << layerName << " because it is a shape tensor." << std::endl;
                    continue;
                }

                auto const outputType = outputTypes.at(isBroadcast ? 0 : outputIdx);
                sample::gLogInfo << "Set output " << outputIdx << " of layer " << layerName << " to type " << outputType
                                 << std::endl;
                layer->setOutputType(outputIdx, globalOutputType);
            }
        }
    }

    if (hasLayerOutputTypeSkipped)
    {
        sample::gLogInfo << "Skipped setting output types for some layers. Check verbose logs for more details."
                         << std::endl;
    }
}

void setLayerDeviceTypes(
    INetworkDefinition const& network, IBuilderConfig& config, LayerDeviceTypes const& layerDeviceTypes)
{
    for (int32_t layerIdx = 0; layerIdx < network.getNbLayers(); ++layerIdx)
    {
        auto* layer = network.getLayer(layerIdx);
        auto const layerName = layer->getName();
        auto match = findPlausible(layerDeviceTypes, layerName);
        if (match != layerDeviceTypes.end())
        {
            DeviceType const deviceType = match->second;
            sample::gLogInfo << "Set layer " << layerName << " to device type " << deviceType << std::endl;
            config.setDeviceType(layer, deviceType);
        }
    }
}

void setDecomposables(INetworkDefinition& network, DecomposableAttentions const& decomposableAttentions)
{
    for (int32_t layerIdx = 0; layerIdx < network.getNbLayers(); ++layerIdx)
    {
        auto* layer = network.getLayer(layerIdx);
        if (layer->getType() == LayerType::kATTENTION_INPUT)
        {
            auto* attention = static_cast<const nvinfer1::IAttentionInputLayer*>(layer)->getAttention();
            auto const attentionName = attention->getName();
            auto match = findPlausible(decomposableAttentions, attentionName);
            if (match != decomposableAttentions.end())
            {
                attention->setDecomposable(match->second);
                sample::gLogInfo << "Set attention " << attentionName
                                 << " to decomposable = " << ((match->second) ? "true" : "false") << std::endl;
            }
        }
    }
}

void markDebugTensors(INetworkDefinition& network, StringSet const& debugTensors)
{
    for (int64_t inputIndex = 0; inputIndex < network.getNbInputs(); ++inputIndex)
    {
        auto* t = network.getInput(inputIndex);
        auto const tensorName = t->getName();
        if (debugTensors.count(tensorName) > 0)
        {
            network.markDebug(*t);
        }
    }
    for (int64_t layerIndex = 0; layerIndex < network.getNbLayers(); ++layerIndex)
    {
        auto* layer = network.getLayer(layerIndex);
        for (int64_t outputIndex = 0; outputIndex < layer->getNbOutputs(); ++outputIndex)
        {
            auto* t = layer->getOutput(outputIndex);
            auto const tensorName = t->getName();
            if (debugTensors.count(tensorName) > 0)
            {
                network.markDebug(*t);
            }
        }
    }
}
void setMemoryPoolLimits(IBuilderConfig& config, BuildOptions const& build)
{
    auto const roundToBytes = [](double const size, bool fromMB = true) {
        return static_cast<size_t>(size * (fromMB ? 1.0_MiB : 1.0_KiB));
    };
    if (build.workspace >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, roundToBytes(build.workspace));
    }
    if (build.dlaSRAM >= 0)
    {
        size_t const sizeInBytes = roundToBytes(build.dlaSRAM);
        size_t sizeInPowerOf2{1};
        // Using 2^30 bytes as a loose upper bound to prevent the possibility of overflows and infinite loops.
        while (sizeInPowerOf2 < 31 && (static_cast<size_t>(1) << sizeInPowerOf2) <= sizeInBytes)
        {
            ++sizeInPowerOf2;
        }
        --sizeInPowerOf2;
        if (sizeInPowerOf2 == 30)
        {
            sample::gLogWarning
                << "User-specified DLA managed SRAM size is too large and has been clipped to 2^30 bytes. "
                << "Please make sure that this is the intended managed SRAM size." << std::endl;
        }
        config.setMemoryPoolLimit(MemoryPoolType::kDLA_MANAGED_SRAM, static_cast<size_t>(1) << sizeInPowerOf2);
    }
    if (build.dlaLocalDRAM >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kDLA_LOCAL_DRAM, roundToBytes(build.dlaLocalDRAM));
    }
    if (build.dlaGlobalDRAM >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kDLA_GLOBAL_DRAM, roundToBytes(build.dlaGlobalDRAM));
    }
    if (build.tacticSharedMem >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, roundToBytes(build.tacticSharedMem, false));
    }
}

void setPreviewFeatures(IBuilderConfig& config, BuildOptions const& build)
{
    auto const setFlag = [&](PreviewFeature feat) {
        int32_t featVal = static_cast<int32_t>(feat);
        if (build.previewFeatures.find(featVal) != build.previewFeatures.end())
        {
            config.setPreviewFeature(feat, build.previewFeatures.at(featVal));
        }
    };
    setFlag(PreviewFeature::kALIASED_PLUGIN_IO_10_03);
    setFlag(PreviewFeature::kRUNTIME_ACTIVATION_RESIZE_10_10);
}

[[nodiscard]] bool setupTilingSettings(BuildOptions const& build, IBuilderConfig& config, std::ostream& err)
{
    if (!config.setTilingOptimizationLevel(static_cast<TilingOptimizationLevel>(build.tilingOptimizationLevel)))
    {
        err << "Can not set tilingOptimizationLevel(" << build.tilingOptimizationLevel << ")" << std::endl;
        return false;
    }

    if (build.l2LimitForTiling != -1)
    {
        if (!config.setL2LimitForTiling(build.l2LimitForTiling))
        {
            err << "Can not set l2LimitForTiling(" << build.l2LimitForTiling << ")" << std::endl;
            return false;
        }
    }

    return true;
}

bool setupNetworkAndConfig(BuildOptions const& build, SystemOptions const& sys, IBuilder& builder,
    INetworkDefinition& network, IBuilderConfig& config, std::unique_ptr<nvinfer1::IInt8Calibrator>& calibrator,
    std::ostream& err, std::vector<std::vector<int8_t>>& sparseWeights)
{
    std::vector<IOptimizationProfile*> profiles{};
    profiles.resize(build.optProfiles.size());
    for (auto& profile : profiles)
    {
        profile = builder.createOptimizationProfile();
    }

    bool hasDynamicShapes{false};

    bool broadcastInputFormats = broadcastIOFormats(build.inputFormats, network.getNbInputs());

    // Check if the provided input tensor names match the input tensors of the engine.
    // Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
    for (auto const& shapes : build.optProfiles)
    {
        for (auto const& shape : shapes)
        {
            bool tensorNameFound{false};
            for (int32_t i = 0; i < network.getNbInputs(); ++i)
            {
                if (matchStringWithOneWildcard(shape.first, network.getInput(i)->getName()))
                {
                    tensorNameFound = true;
                    break;
                }
            }
            if (!tensorNameFound)
            {
                sample::gLogError << "Cannot find input tensor with name \"" << shape.first << "\" in the network "
                                  << "inputs! Please make sure the input tensor names are correct." << std::endl;
                return false;
            }
        }
    }

    for (uint32_t i = 0, n = network.getNbInputs(); i < n; i++)
    {
        // Set formats and data types of inputs
        auto* input = network.getInput(i);
        if (!build.inputFormats.empty())
        {
            int32_t inputFormatIndex = broadcastInputFormats ? 0 : i;
            input->setType(build.inputFormats[inputFormatIndex].first);
            input->setAllowedFormats(build.inputFormats[inputFormatIndex].second);
        }

        auto const dims = input->getDimensions();
        auto const isScalar = dims.nbDims == 0;
        auto const isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; })
            || input->isShapeTensor();
        if (isDynamicInput)
        {
            hasDynamicShapes = true;
            for (size_t i = 0; i < build.optProfiles.size(); i++)
            {
                auto const& optShapes = build.optProfiles[i];
                auto profile = profiles[i];
                auto const tensorName = input->getName();
                auto shape = findPlausible(optShapes, tensorName);
                ShapeRange shapes{};

                // If no shape is provided, set dynamic dimensions to 1.
                if (shape == optShapes.end())
                {
                    constexpr int32_t kDEFAULT_DIMENSION{1};
                    std::vector<int64_t> staticDims;
                    if (input->isShapeTensor())
                    {
                        if (isScalar)
                        {
                            staticDims.push_back(1);
                        }
                        else
                        {
                            staticDims.resize(dims.d[0]);
                            std::fill(staticDims.begin(), staticDims.end(), kDEFAULT_DIMENSION);
                        }
                    }
                    else
                    {
                        staticDims.resize(dims.nbDims);
                        std::transform(dims.d, dims.d + dims.nbDims, staticDims.begin(),
                            [&](int dimension) { return dimension > 0 ? dimension : kDEFAULT_DIMENSION; });
                    }
                    sample::gLogWarning << "Dynamic dimensions required for input: " << tensorName
                                        << ", but no shapes were provided. Automatically overriding shape to: "
                                        << staticDims << std::endl;
                    std::fill(shapes.begin(), shapes.end(), staticDims);
                }
                else
                {
                    shapes = shape->second;
                }

                std::vector<int64_t> profileDims{};
                if (input->isShapeTensor())
                {
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValuesV2(tensorName, OptProfileSelector::kMIN,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values MIN", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValuesV2(tensorName, OptProfileSelector::kOPT,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values OPT", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValuesV2(tensorName, OptProfileSelector::kMAX,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values MAX", false, err);
                    sample::gLogInfo << "Set input shape tensor " << tensorName << " for optimization profile " << i
                                     << " to:"
                                     << " MIN=" << shapes[static_cast<size_t>(OptProfileSelector::kMIN)]
                                     << " OPT=" << shapes[static_cast<size_t>(OptProfileSelector::kOPT)]
                                     << " MAX=" << shapes[static_cast<size_t>(OptProfileSelector::kMAX)] << std::endl;
                }
                else
                {
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(tensorName, OptProfileSelector::kMIN, toDims(profileDims)),
                        "Error in set dimensions to profile MIN", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(tensorName, OptProfileSelector::kOPT, toDims(profileDims)),
                        "Error in set dimensions to profile OPT", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(tensorName, OptProfileSelector::kMAX, toDims(profileDims)),
                        "Error in set dimensions to profile MAX", false, err);
                    sample::gLogInfo << "Set shape of input tensor " << tensorName << " for optimization profile " << i
                                     << " to:"
                                     << " MIN=" << shapes[static_cast<size_t>(OptProfileSelector::kMIN)]
                                     << " OPT=" << shapes[static_cast<size_t>(OptProfileSelector::kOPT)]
                                     << " MAX=" << shapes[static_cast<size_t>(OptProfileSelector::kMAX)] << std::endl;
                }
            }
        }
    }

    for (uint32_t i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        auto* output = network.getOutput(i);
        auto const dims = output->getDimensions();
        // A shape tensor output with known static dimensions may have dynamic shape values inside it.
        auto const isDynamicOutput = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; })
            || output->isShapeTensor();
        if (isDynamicOutput)
        {
            hasDynamicShapes = true;
        }
    }

    if (!hasDynamicShapes && !build.optProfiles[0].empty())
    {
        sample::gLogError << "Static model does not take explicit shapes since the shape of inference tensors will be "
                             "determined by the model itself"
                          << std::endl;
        return false;
    }

    if (hasDynamicShapes)
    {
        for (auto profile : profiles)
        {
            SMP_RETVAL_IF_FALSE(profile->isValid(), "Required optimization profile is invalid", false, err);
            SMP_RETVAL_IF_FALSE(
                config.addOptimizationProfile(profile) != -1, "Error in add optimization profile", false, err);
        }
    }

    bool broadcastOutputFormats = broadcastIOFormats(build.outputFormats, network.getNbOutputs(), false);

    for (uint32_t i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        // Set formats and data types of outputs
        auto* output = network.getOutput(i);
        if (!build.outputFormats.empty())
        {
            int32_t outputFormatIndex = broadcastOutputFormats ? 0 : i;
            output->setType(build.outputFormats[outputFormatIndex].first);
            output->setAllowedFormats(build.outputFormats[outputFormatIndex].second);
        }
    }

    setMemoryPoolLimits(config, build);

    setPreviewFeatures(config, build);

    if (build.builderOptimizationLevel != defaultBuilderOptimizationLevel)
    {
        config.setBuilderOptimizationLevel(build.builderOptimizationLevel);
    }

    if (build.maxTactics != defaultMaxTactics)
    {
        config.setMaxNbTactics(build.maxTactics);
    }

    if (build.timingCacheMode == TimingCacheMode::kDISABLE)
    {
        config.setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
    }

    if (build.disableCompilationCache)
    {
        config.setFlag(BuilderFlag::kDISABLE_COMPILATION_CACHE);
    }

    if (build.errorOnTimingCacheMiss)
    {
        config.setFlag(BuilderFlag::kERROR_ON_TIMING_CACHE_MISS);
    }

    if (!build.tf32)
    {
        config.clearFlag(BuilderFlag::kTF32);
    }

    if (build.refittable)
    {
        config.setFlag(BuilderFlag::kREFIT);
    }

    if (build.stripWeights)
    {
        // The kREFIT_IDENTICAL is enabled by default when kSTRIP_PLAN is on.
        config.setFlag(BuilderFlag::kSTRIP_PLAN);
    }

    if (build.versionCompatible)
    {
        config.setFlag(BuilderFlag::kVERSION_COMPATIBLE);
    }
    std::vector<char const*> pluginPaths;
    for (auto const& pluginPath : sys.setPluginsToSerialize)
    {
        sample::gLogVerbose << "Setting plugin to serialize: " << pluginPath << std::endl;
        pluginPaths.push_back(pluginPath.c_str());
    }
    if (!pluginPaths.empty())
    {
        config.setPluginsToSerialize(pluginPaths.data(), pluginPaths.size());
    }
    if (build.excludeLeanRuntime)
    {
        config.setFlag(BuilderFlag::kEXCLUDE_LEAN_RUNTIME);
    }

    if (build.sparsity != SparsityFlag::kDISABLE)
    {
        config.setFlag(BuilderFlag::kSPARSE_WEIGHTS);
        if (build.sparsity == SparsityFlag::kFORCE)
        {
            sparsify(network, sparseWeights);
        }
    }

    if (build.enableMonitorMemory)
    {
        config.setFlag(BuilderFlag::kMONITOR_MEMORY);
    }

    if (build.distributiveIndependence)
    {
        config.setFlag(BuilderFlag::kDISTRIBUTIVE_INDEPENDENCE);
    }

    config.setProfilingVerbosity(build.profilingVerbosity);
    config.setAvgTimingIterations(build.avgTiming);
    if (build.fp16)
    {
        config.setFlag(BuilderFlag::kFP16);
    }
    if (build.int8)
    {
        config.setFlag(BuilderFlag::kINT8);
    }
    if (build.bf16)
    {
        config.setFlag(BuilderFlag::kBF16);
    }

    SMP_RETVAL_IF_FALSE(!(build.int8 && build.fp8), "FP8 and INT8 precisions have been specified", false, err);

    if (build.fp8)
    {
        config.setFlag(BuilderFlag::kFP8);
    }

    if (build.int4)
    {
        config.setFlag(BuilderFlag::kINT4);
    }

    if (build.int8 && !build.fp16)
    {
        sample::gLogInfo
            << "FP32 and INT8 precisions have been specified - more performance might be enabled by additionally "
               "specifying --fp16 or --best"
            << std::endl;
    }
    auto isInt8 = [](IOFormat const& format) { return format.first == DataType::kINT8; };
    auto int8IO = std::count_if(build.inputFormats.begin(), build.inputFormats.end(), isInt8)
        + std::count_if(build.outputFormats.begin(), build.outputFormats.end(), isInt8);

    auto hasQDQLayers = [](INetworkDefinition& network) {
        // Determine if our network has QDQ layers.
        auto const nbLayers = network.getNbLayers();
        for (int32_t i = 0; i < nbLayers; i++)
        {
            auto const& layer = network.getLayer(i);
            if (layer->getType() == LayerType::kQUANTIZE || layer->getType() == LayerType::kDEQUANTIZE)
            {
                return true;
            }
        }
        return false;
    };

    if (!hasQDQLayers(network) && (build.int8 || int8IO) && build.calibration.empty())
    {
        // Explicitly set int8 scales if no calibrator is provided and if I/O tensors use int8,
        // because auto calibration does not support this case.
        SMP_RETVAL_IF_FALSE(setTensorDynamicRange(network), "Error in set tensor dynamic range.", false, err);
    }
    else if (build.int8)
    {
        if (!hasQDQLayers(network) && int8IO)
        {
            try
            {
                // Set dynamic ranges of int8 inputs / outputs to match scales loaded from calibration cache
                // TODO http://nvbugs/3262234 Change the network validation so that this workaround can be removed
                setTensorScalesFromCalibration(network, build.inputFormats, build.outputFormats, build.calibration);
            }
            catch (std::exception&)
            {
                sample::gLogError
                    << "Int8IO was specified but impossible to read tensor scales from provided calibration cache file"
                    << std::endl;
                return false;
            }
        }

        IOptimizationProfile* profileCalib{nullptr};
        if (!build.shapesCalib.empty())
        {
            profileCalib = builder.createOptimizationProfile();
            for (uint32_t i = 0, n = network.getNbInputs(); i < n; i++)
            {
                auto* input = network.getInput(i);
                Dims profileDims{};
                auto const tensorName = input->getName();
                auto shape = findPlausible(build.shapesCalib, tensorName);

                if (shape == build.shapesCalib.end())
                {
                    std::ostringstream msg;
                    msg << "Calibration profile for tensor " << tensorName << " cannot be found!";
                    throw std::invalid_argument(msg.str());
                }

                auto shapesCalib = shape->second;
                profileDims = toDims(shapesCalib[static_cast<size_t>(OptProfileSelector::kOPT)]);
                // Here we check only kMIN as all profileDims are the same.
                SMP_RETVAL_IF_FALSE(profileCalib->setDimensions(tensorName, OptProfileSelector::kMIN, profileDims),
                    "Error in set dimensions to calibration profile OPT", false, err);
                profileCalib->setDimensions(tensorName, OptProfileSelector::kOPT, profileDims);
                profileCalib->setDimensions(tensorName, OptProfileSelector::kMAX, profileDims);
                sample::gLogInfo << "Set calibration profile for input tensor " << tensorName << " to " << profileDims
                                 << std::endl;
            }
            SMP_RETVAL_IF_FALSE(profileCalib->isValid(), "Calibration profile is invalid", false, err);
            SMP_RETVAL_IF_FALSE(
                config.setCalibrationProfile(profileCalib), "Error in set calibration profile", false, err);
        }

        std::vector<int64_t> elemCount{};
        for (int i = 0; i < network.getNbInputs(); i++)
        {
            auto* input = network.getInput(i);
            auto const dims = input->getDimensions();
            auto const isDynamicInput
                = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; });

            if (profileCalib)
            {
                elemCount.push_back(volume(profileCalib->getDimensions(input->getName(), OptProfileSelector::kOPT)));
            }
            else if (!profiles.empty() && isDynamicInput)
            {
                elemCount.push_back(
                    volume(profiles[build.calibProfile]->getDimensions(input->getName(), OptProfileSelector::kOPT)));
            }
            else
            {
                elemCount.push_back(volume(input->getDimensions()));
            }
        }

        calibrator.reset(new RndInt8Calibrator(1, elemCount, build.calibration, network, err));
        config.setInt8Calibrator(calibrator.get());
    }

    if (build.directIO)
    {
        config.setFlag(BuilderFlag::kDIRECT_IO);
    }

    switch (build.precisionConstraints)
    {
    case PrecisionConstraints::kNONE:
        // It's the default for TensorRT.
        break;
    case PrecisionConstraints::kOBEY: config.setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS); break;
    case PrecisionConstraints::kPREFER: config.setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS); break;
    }

    if (!build.layerPrecisions.empty() && build.precisionConstraints != PrecisionConstraints::kNONE)
    {
        setLayerPrecisions(network, build.layerPrecisions);
    }

    if (!build.layerOutputTypes.empty() && build.precisionConstraints != PrecisionConstraints::kNONE)
    {
        setLayerOutputTypes(network, build.layerOutputTypes);
    }

    if (!build.layerDeviceTypes.empty())
    {
        setLayerDeviceTypes(network, config, build.layerDeviceTypes);
    }

    if (!build.decomposableAttentions.empty())
    {
        setDecomposables(network, build.decomposableAttentions);
    }

    if (!build.debugTensors.empty())
    {
        markDebugTensors(network, build.debugTensors);
    }

    if (build.markUnfusedTensorsAsDebugTensors)
    {
        network.markUnfusedTensorsAsDebugTensors();
    }

    if (build.safe && sys.DLACore == -1)
    {
        config.setEngineCapability(EngineCapability::kSAFETY);
    }

    if (build.restricted)
    {
        config.setFlag(BuilderFlag::kSAFETY_SCOPE);
    }

    if (sys.DLACore != -1)
    {
        if (sys.DLACore < builder.getNbDLACores())
        {
            config.setDefaultDeviceType(DeviceType::kDLA);
            config.setDLACore(sys.DLACore);
            config.setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            if (build.buildDLAStandalone)
            {
                config.setEngineCapability(EngineCapability::kDLA_STANDALONE);
            }
            if (build.allowGPUFallback)
            {
                config.setFlag(BuilderFlag::kGPU_FALLBACK);
            }
            else
            {
                // Reformatting runs on GPU, so avoid I/O reformatting.
                config.setFlag(BuilderFlag::kDIRECT_IO);
            }
            if (!build.int8)
            {
                config.setFlag(BuilderFlag::kFP16);
            }
        }
        else
        {
            err << "Cannot create DLA engine, " << sys.DLACore << " not available" << std::endl;
            return false;
        }
    }
    if (build.enabledTactics || build.disabledTactics)
    {
        TacticSources tacticSources = config.getTacticSources();
        tacticSources |= build.enabledTactics;
        tacticSources &= ~build.disabledTactics;
        config.setTacticSources(tacticSources);
    }

    config.setHardwareCompatibilityLevel(build.hardwareCompatibilityLevel);


    config.setRuntimePlatform(build.runtimePlatform);

    if (build.maxAuxStreams != defaultMaxAuxStreams)
    {
        config.setMaxAuxStreams(build.maxAuxStreams);
    }

    if (build.allowWeightStreaming)
    {
        config.setFlag(BuilderFlag::kWEIGHT_STREAMING);
    }

    if (!setupTilingSettings(build, config, err))
    {
        return false;
    }

    if (!build.remoteAutoTuningConfig.empty())
    {
        SMP_RETVAL_IF_FALSE(config.setRemoteAutoTuningConfig(build.remoteAutoTuningConfig.c_str()),
            "Failed to set remote auto tuning config", false, err);
    }

    return true;
}

} // namespace

//!
//! \brief Create a serialized engine for a network definition
//!
//! \return Whether the engine creation succeeds or fails.
//!
bool networkToSerializedEngine(
    BuildOptions const& build, SystemOptions const& sys, IBuilder& builder, BuildEnvironment& env, std::ostream& err)
{
    std::unique_ptr<IBuilderConfig> config{builder.createBuilderConfig()};
    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    std::vector<std::vector<int8_t>> sparseWeights;
    SMP_RETVAL_IF_FALSE(config != nullptr, "Config creation failed", false, err);
    SMP_RETVAL_IF_FALSE(
        setupNetworkAndConfig(build, sys, builder, *env.network, *config, calibrator, err, sparseWeights),
        "Network And Config setup failed", false, err);

    std::unique_ptr<ITimingCache> timingCache{};
    // Try to load cache from file. Create a fresh cache if the file doesn't exist
    if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
    {
        timingCache = samplesCommon::buildTimingCacheFromFile(gLogger.getTRTLogger(), *config, build.timingCacheFile);
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    SMP_RETVAL_IF_FALSE(profileStream != nullptr, "Cuda stream creation failed", false, err);
    config->setProfileStream(*profileStream);

    auto const tBegin = std::chrono::high_resolution_clock::now();

    if (!(build.safe || build.buildDLAStandalone) && build.save)
    {
        auto const engineFile = build.engine;
        FileStreamWriter writer(engineFile);
        builder.buildSerializedNetworkToStream(*env.network, *config, writer);
        auto const engineSize = writer.finalize();
        std::vector<uint8_t> streamEngine(engineSize, 0);
        std::ifstream reader(engineFile, std::ios::binary);
        SMP_RETVAL_IF_FALSE((reader.is_open() && reader.good()), "Failed to open engine file for reading", false, err);
        reader.read(reinterpret_cast<char*>(streamEngine.data()), engineSize);
        SMP_RETVAL_IF_FALSE((!reader.fail()), "Error when reading engine file", false, err);
        reader.close();
        sample::gLogInfo << "Created engine with size: " << (engineSize / 1.0_MiB) << " MiB" << std::endl;
        env.engine.setBlob(std::move(streamEngine));
    }
    else
    {
        IHostMemory* serializedEngine{nullptr};
        if (build.safe && build.save && build.dumpKernelText)
        {
            IHostMemory* kernelText{nullptr};
            serializedEngine = builder.buildSerializedNetwork(*env.network, *config, kernelText);
            if (kernelText != nullptr && kernelText->size() > 0)
            {
                std::unique_ptr<IHostMemory> kernelTextPtr(kernelText);
                env.kernelText.setBlob(kernelTextPtr);
                sample::gLogInfo << "Created kernel CPP with size: " << (kernelText->size() / 1.0_MiB) << " MiB"
                                 << std::endl;
            }
            else
            {
                sample::gLogError << "Failed to create kernel CPP." << std::endl;
            }
        }
        else
        {
            serializedEngine = builder.buildSerializedNetwork(*env.network, *config);
        }
        SMP_RETVAL_IF_FALSE(serializedEngine != nullptr, "Engine could not be created from network", false, err);
        sample::gLogInfo << "Created engine with size: " << (serializedEngine->size() / 1.0_MiB) << " MiB" << std::endl;

        if (build.safe && build.consistency)
        {
            std::vector<std::string> pluginBuildLibPaths;
#if ENABLE_UNIFIED_BUILDER
            pluginBuildLibPaths.reserve(sys.safetyPlugins.size());
            std::transform(sys.safetyPlugins.begin(), sys.safetyPlugins.end(), std::back_inserter(pluginBuildLibPaths),
                [](auto const& sp) { return sp.libraryName; });
#endif
            if (!checkSafeEngine(serializedEngine->data(), serializedEngine->size(), pluginBuildLibPaths))
            {
                return false;
            }
        }
        std::unique_ptr<IHostMemory> serializedEnginePtr(serializedEngine);
        env.engine.setBlob(serializedEnginePtr);
    }

    auto const tEnd = std::chrono::high_resolution_clock::now();
    float const buildTime = std::chrono::duration<float>(tEnd - tBegin).count();
    sample::gLogInfo << "Engine built in " << buildTime << " sec." << std::endl;

    if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
    {
        auto timingCache = config->getTimingCache();
        samplesCommon::updateTimingCacheFile(gLogger.getTRTLogger(), build.timingCacheFile, timingCache, builder);
    }

    return true;
}


//!
//! \brief Parse a given model, create a network and an engine.
//!
bool modelToBuildEnv(
    ModelOptions const& model, BuildOptions const& build, SystemOptions& sys, BuildEnvironment& env, std::ostream& err)
{
    env.builder.reset(createBuilder());
    SMP_RETVAL_IF_FALSE(env.builder != nullptr, "Builder creation failed", false, err);
    env.builder->setErrorRecorder(&gRecorder);
    auto networkFlags = (build.stronglyTyped)
        ? 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)
        : 0U;
    for (auto const& pluginPath : sys.dynamicPlugins)
    {
        env.builder->getPluginRegistry().loadLibrary(pluginPath.c_str());
    }
    env.network.reset(env.builder->createNetworkV2(networkFlags));

    std::vector<std::string> vcPluginLibrariesUsed;
    SMP_RETVAL_IF_FALSE(env.network != nullptr, "Network creation failed", false, err);
    env.parser
        = modelToNetwork(model, build, *env.network, err, build.versionCompatible ? &vcPluginLibrariesUsed : nullptr);
    SMP_RETVAL_IF_FALSE(env.parser.operator bool(), "Parsing model failed", false, err);

    if (build.versionCompatible && !sys.ignoreParsedPluginLibs && !vcPluginLibrariesUsed.empty())
    {
        sample::gLogInfo << "The following plugin libraries were identified by the parser as required for a "
                            "version-compatible engine:"
                         << std::endl;
        for (auto const& lib : vcPluginLibrariesUsed)
        {
            sample::gLogInfo << "    " << lib << std::endl;
        }
        if (!build.excludeLeanRuntime)
        {
            sample::gLogInfo << "These libraries will be added to --setPluginsToSerialize since --excludeLeanRuntime "
                                "was not specified."
                             << std::endl;
            std::copy(vcPluginLibrariesUsed.begin(), vcPluginLibrariesUsed.end(),
                std::back_inserter(sys.setPluginsToSerialize));
        }
        sample::gLogInfo << "These libraries will be added to --dynamicPlugins for use at inference time." << std::endl;
        std::copy(vcPluginLibrariesUsed.begin(), vcPluginLibrariesUsed.end(), std::back_inserter(sys.dynamicPlugins));

        // Implicitly-added plugins from ONNX parser should be loaded into plugin registry as well.
        for (auto const& pluginPath : vcPluginLibrariesUsed)
        {
            env.builder->getPluginRegistry().loadLibrary(pluginPath.c_str());
        }

        sample::gLogInfo << "Use --ignoreParsedPluginLibs to disable this behavior." << std::endl;
    }

    SMP_RETVAL_IF_FALSE(
        networkToSerializedEngine(build, sys, *env.builder, env, err), "Building engine failed", false, err);
    return true;
}

namespace
{
std::pair<std::vector<std::string>, std::vector<WeightsRole>> getLayerWeightsRolePair(IRefitter& refitter)
{
    // Get number of refittable items.
    auto const nbAll = refitter.getAll(0, nullptr, nullptr);
    std::vector<char const*> layerNames(nbAll);
    // Allocate buffers for the items and get them.
    std::vector<nvinfer1::WeightsRole> weightsRoles(nbAll);
    refitter.getAll(nbAll, layerNames.data(), weightsRoles.data());
    std::vector<std::string> layerNameStrs(nbAll);
    std::transform(layerNames.begin(), layerNames.end(), layerNameStrs.begin(), [](char const* name) {
        if (name == nullptr)
        {
            return std::string{};
        }
        return std::string{name};
    });
    return {layerNameStrs, weightsRoles};
}

std::pair<std::vector<std::string>, std::vector<WeightsRole>> getMissingLayerWeightsRolePair(IRefitter& refitter)
{
    // Get number of refittable items.
    auto const nbMissing = refitter.getMissing(0, nullptr, nullptr);
    std::vector<char const*> layerNames(nbMissing);
    // Allocate buffers for the items and get them.
    std::vector<nvinfer1::WeightsRole> weightsRoles(nbMissing);
    refitter.getMissing(nbMissing, layerNames.data(), weightsRoles.data());
    std::vector<std::string> layerNameStrs(nbMissing);
    std::transform(layerNames.begin(), layerNames.end(), layerNameStrs.begin(), [](char const* name) {
        if (name == nullptr)
        {
            return std::string{};
        }
        return std::string{name};
    });
    return {layerNameStrs, weightsRoles};
}
} // namespace

bool loadStreamingEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env, std::ostream& err)
{
    auto& reader = env.engine.getFileReader();
    SMP_RETVAL_IF_FALSE(reader.open(filepath), "", false, err << "Error opening engine file: " << filepath);
    return true;
}

bool loadAsyncStreamingEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env, std::ostream& err)
{
    auto& asyncReader = env.engine.getAsyncFileReader();
    SMP_RETVAL_IF_FALSE(asyncReader.open(filepath), "", false, err << "Error opening engine file: " << filepath);
    return true;
}


bool loadEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env, std::ostream& err,
    SystemOptions const& sys, bool const enableConsistency)
{
    auto const tBegin = std::chrono::high_resolution_clock::now();
    std::ifstream engineFile(filepath, std::ios::binary);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error opening engine file: " << filepath);
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> engineBlob(fsize);
    engineFile.read(reinterpret_cast<char*>(engineBlob.data()), fsize);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error loading engine file: " << filepath);
    auto const tEnd = std::chrono::high_resolution_clock::now();
    float const loadTime = std::chrono::duration<float>(tEnd - tBegin).count();
    sample::gLogInfo << "Engine loaded in " << loadTime << " sec." << std::endl;
    sample::gLogInfo << "Loaded engine with size: " << (fsize / 1.0_MiB) << " MiB" << std::endl;

    if (enableConsistency)
    {
        std::vector<std::string> pluginBuildLibPaths;
#if ENABLE_UNIFIED_BUILDER
        pluginBuildLibPaths.reserve(sys.safetyPlugins.size());
        std::transform(sys.safetyPlugins.begin(), sys.safetyPlugins.end(), std::back_inserter(pluginBuildLibPaths),
            [](auto const& sp) { return sp.libraryName; });
#endif
        if (!checkSafeEngine(engineBlob.data(), fsize, pluginBuildLibPaths))
        {
            sample::gLogError << "Consistency validation is not enabled." << std::endl;
            return false;
        }
    }

    env.engine.setBlob(std::move(engineBlob));

    return true;
}

bool printPlanVersion(BuildEnvironment& env, std::ostream& err)
{
    constexpr int64_t kPLAN_SIZE{28};
    std::vector<uint8_t> data(kPLAN_SIZE);
    auto blob = data.data();

    auto& reader = env.engine.getFileReader();
    auto& asyncReader = env.engine.getAsyncFileReader();
    if (reader.isOpen())
    {
        SMP_RETVAL_IF_FALSE(reader.read(data.data(), kPLAN_SIZE) == kPLAN_SIZE, "Failed to read plan file", false, err);
    }
    else if (asyncReader.isOpen())
    {
        SMP_RETVAL_IF_FALSE(asyncReader.read(data.data(), kPLAN_SIZE, cudaStream_t{}) == kPLAN_SIZE,
            "Failed to read plan file", false, err);
    }
    else
    {
        SMP_RETVAL_IF_FALSE(env.engine.getBlob().data != nullptr, "Plan file is empty", false, err);
        SMP_RETVAL_IF_FALSE(env.engine.getBlob().size >= 28, "Plan file is incorrect", false, err);
        blob = static_cast<uint8_t*>(env.engine.getBlob().data);
    }
    auto blob32 = reinterpret_cast<uint32_t*>(blob);

    //! Correct TensorRT plan file starts with this tag
    constexpr uint32_t kPLAN_FILE_TAG{0x74727466U};
    SMP_RETVAL_IF_FALSE(blob32[0] == kPLAN_FILE_TAG, "Failed to verify a plan tag.", false, err);
    switch (blob32[1])
    {
    case 0U:
    {
        // Blob index to store the plan version may depend on the serialization version.
        sample::gLogInfo << "Plan was created with TensorRT version " << static_cast<int32_t>(blob[24])
        << "." << static_cast<int32_t>(blob[25]) << "." << static_cast<int32_t>(blob[26])
        << "." << static_cast<int32_t>(blob[27]) << std::endl;
        return true;
    }
    }
    sample::gLogError << "Serialization version is not supported." << std::endl;
    return false;
}

void dumpRefittable(nvinfer1::ICudaEngine& engine)
{
    std::unique_ptr<IRefitter> refitter{createRefitter(engine)};
    if (refitter == nullptr)
    {
        sample::gLogError << "Failed to create a refitter." << std::endl;
        return;
    }

    auto const& layerWeightsRolePair = getLayerWeightsRolePair(*refitter);
    auto const& layerNames = layerWeightsRolePair.first;
    auto const& weightsRoles = layerWeightsRolePair.second;
    auto const nbAll = layerWeightsRolePair.first.size();
    for (size_t i = 0; i < nbAll; ++i)
    {
        sample::gLogInfo << layerNames[i] << " " << weightsRoles[i] << std::endl;
    }
}

ICudaEngine* loadEngine(std::string const& engine, int32_t DLACore, std::ostream& err)
{
    BuildEnvironment env(/* isSafe */ false, /* versionCompatible */ false, DLACore, "", getTempfileControlDefaults());
    SystemOptions sys;
    return loadEngineToBuildEnv(engine, env, err, sys, false) ? env.engine.release() : nullptr;
}

bool saveEngine(ICudaEngine const& engine, std::string const& fileName, std::ostream& err)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        err << "Cannot open engine file: " << fileName << std::endl;
        return false;
    }

    std::unique_ptr<IHostMemory> serializedEngine{engine.serialize()};
    if (serializedEngine == nullptr)
    {
        err << "Engine serialization failed" << std::endl;
        return false;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

bool getEngineBuildEnv(
    ModelOptions const& model, BuildOptions const& build, SystemOptions& sys, BuildEnvironment& env, std::ostream& err)
{
    bool createEngineSuccess{false};

    if (build.load)
    {
        if (build.safe)
        {
            createEngineSuccess = loadEngineToBuildEnv(build.engine, env, err, sys, build.safe && build.consistency);
        }
        else
        {
            if (build.asyncFileReader)
            {
                createEngineSuccess = loadAsyncStreamingEngineToBuildEnv(build.engine, env, err);
            }
            else
            {
                createEngineSuccess = loadStreamingEngineToBuildEnv(build.engine, env, err);
            }
        }
    }
    else
    {
        createEngineSuccess = modelToBuildEnv(model, build, sys, env, err);
    }

    SMP_RETVAL_IF_FALSE(createEngineSuccess, "Failed to create engine from model or file.", false, err);

    if (build.getPlanVersionOnly && build.load)
    {
        SMP_RETVAL_IF_FALSE(printPlanVersion(env, err), "Failed to get plan file version.", false, err);
        return true;
    }

    if (build.save)
    {
        std::ofstream engineFile(build.engine, std::ios::binary);
        auto& engineBlob = env.engine.getBlob();
        engineFile.write(static_cast<char const*>(engineBlob.data), engineBlob.size);
        SMP_RETVAL_IF_FALSE(!engineFile.fail(), "Saving engine to file failed.", false, err);
        engineFile.flush();
        engineFile.close();
        if (!build.safe)
        {
            env.engine.releaseBlob();
            if (build.asyncFileReader)
            {
                SMP_RETVAL_IF_FALSE(loadAsyncStreamingEngineToBuildEnv(build.engine, env, err),
                    "Reading engine file via async stream reader failed.", false, err);
            }
            else
            {
                SMP_RETVAL_IF_FALSE(loadStreamingEngineToBuildEnv(build.engine, env, err),
                    "Reading engine file via stream reader failed.", false, err);
            }
        }
        if (build.safe && build.dumpKernelText)
        {
            auto& kernelTextBlob = env.kernelText.getBlob();
            if (kernelTextBlob.data != nullptr)
            {
                std::ofstream engineTextFile(build.engine + ".txt");
                engineTextFile.write(static_cast<char const*>(kernelTextBlob.data), kernelTextBlob.size);
                SMP_RETVAL_IF_FALSE(!engineTextFile.fail(), "Saving engine kernel text to file failed.", false, err);
                engineTextFile.close();
            }
        }
    }

    return true;
}

// There is not a getWeightsName API, so we need to use WeightsRole.
std::vector<std::pair<WeightsRole, Weights>> getAllRefitWeightsForLayer(ILayer const& l)
{
    switch (l.getType())
    {
    case LayerType::kCONSTANT:
    {
        auto const& layer = static_cast<nvinfer1::IConstantLayer const&>(l);
        auto const weights = layer.getWeights();
        switch (weights.type)
        {
        case DataType::kFLOAT:
        case DataType::kHALF:
        case DataType::kBF16:
        case DataType::kINT8:
        case DataType::kINT32:
        case DataType::kINT64: return {std::make_pair(WeightsRole::kCONSTANT, weights)};
        case DataType::kBOOL:
        case DataType::kUINT8:
        case DataType::kFP8:
        case DataType::kINT4:
        case DataType::kFP4:
        case DataType::kE8M0:
            // Refit not supported for these types.
            break;
        }
        break;
    }
    case LayerType::kCONVOLUTION:
    {
        auto const& layer = static_cast<nvinfer1::IConvolutionLayer const&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kDECONVOLUTION:
    {
        auto const& layer = static_cast<nvinfer1::IDeconvolutionLayer const&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kSCALE:
    {
        auto const& layer = static_cast<nvinfer1::IScaleLayer const&>(l);
        return {std::make_pair(WeightsRole::kSCALE, layer.getScale()),
            std::make_pair(WeightsRole::kSHIFT, layer.getShift())};
    }
    case LayerType::kACTIVATION:
    case LayerType::kATTENTION_INPUT:
    case LayerType::kATTENTION_OUTPUT:
    case LayerType::kASSERTION:
    case LayerType::kCAST:
    case LayerType::kCONCATENATION:
    case LayerType::kCONDITION:
    case LayerType::kCONDITIONAL_INPUT:
    case LayerType::kCONDITIONAL_OUTPUT:
    case LayerType::kCUMULATIVE:
    case LayerType::kDEQUANTIZE:
    case LayerType::kDYNAMIC_QUANTIZE:
    case LayerType::kEINSUM:
    case LayerType::kELEMENTWISE:
    case LayerType::kFILL:
    case LayerType::kGATHER:
    case LayerType::kGRID_SAMPLE:
    case LayerType::kIDENTITY:
    case LayerType::kITERATOR:
    case LayerType::kLOOP_OUTPUT:
    case LayerType::kLRN:
    case LayerType::kMATRIX_MULTIPLY:
    case LayerType::kNMS:
    case LayerType::kNON_ZERO:
    case LayerType::kNORMALIZATION:
    case LayerType::kONE_HOT:
    case LayerType::kPADDING:
    case LayerType::kPARAMETRIC_RELU:
    case LayerType::kPLUGIN:
    case LayerType::kPLUGIN_V2:
    case LayerType::kPLUGIN_V3:
    case LayerType::kPOOLING:
    case LayerType::kQUANTIZE:
    case LayerType::kRAGGED_SOFTMAX:
    case LayerType::kRECURRENCE:
    case LayerType::kREDUCE:
    case LayerType::kRESIZE:
    case LayerType::kREVERSE_SEQUENCE:
    case LayerType::kSCATTER:
    case LayerType::kSELECT:
    case LayerType::kSHAPE:
    case LayerType::kSHUFFLE:
    case LayerType::kSLICE:
    case LayerType::kSOFTMAX:
    case LayerType::kSQUEEZE:
    case LayerType::kTOPK:
    case LayerType::kTRIP_LIMIT:
    case LayerType::kUNARY:
    case LayerType::kUNSQUEEZE: return {};
    }
    return {};
}

bool refitFromOnnx(nvinfer1::ICudaEngine& engine, std::string onnxModelFile, bool multiThreading)
{
    sample::gLogInfo << "Refitting engine from ONNX model " << onnxModelFile << std::endl;
    std::unique_ptr<IRefitter> refitter{createRefitter(engine)};
    if (multiThreading && !refitter->setMaxThreads(10))
    {
        sample::gLogError << "Failed to set max threads to refitter." << std::endl;
        return false;
    }
    std::unique_ptr<nvonnxparser::IParserRefitter> parserRefitter{createONNXRefitter(*refitter)};

    if (!parserRefitter->refitFromFile(onnxModelFile.c_str()))
    {
        return false;
    }
    TrtCudaStream stream;
    if (!refitter->refitCudaEngineAsync(stream.get()))
    {
        return false;
    }
    stream.synchronize();

    sample::gLogInfo << "Engine successfully refitted from ONNX model " << onnxModelFile << std::endl;
    return true;
}

bool timeRefit(INetworkDefinition const& network, nvinfer1::ICudaEngine& engine, bool multiThreading)
{
    using time_point = std::chrono::time_point<std::chrono::steady_clock>;
    using durationMs = std::chrono::duration<float, std::milli>;

    auto const nbLayers = network.getNbLayers();
    std::unique_ptr<IRefitter> refitter{createRefitter(engine)};
    // Set max threads that can be used by refitter.
    if (multiThreading && !refitter->setMaxThreads(10))
    {
        sample::gLogError << "Failed to set max threads to refitter." << std::endl;
        return false;
    }
    auto const& layerWeightsRolePair = getLayerWeightsRolePair(*refitter);
    // We use std::string instead of char const* since we can have copies of layer names.
    std::set<std::pair<std::string, WeightsRole>> layerRoleSet;

    auto const& layerNames = layerWeightsRolePair.first;
    auto const& weightsRoles = layerWeightsRolePair.second;

    std::transform(layerNames.begin(), layerNames.end(), weightsRoles.begin(),
        std::inserter(layerRoleSet, layerRoleSet.begin()),
        [](std::string const& layerName, WeightsRole const role) { return std::make_pair(layerName, role); });

    auto const isRefittable = [&layerRoleSet](char const* layerName, WeightsRole const role) {
        return layerRoleSet.find(std::make_pair(layerName, role)) != layerRoleSet.end();
    };

    auto const setWeights = [&] {
        for (int32_t i = 0; i < nbLayers; i++)
        {
            auto const layer = network.getLayer(i);
            auto const roleWeightsVec = getAllRefitWeightsForLayer(*layer);
            for (auto const& roleWeights : roleWeightsVec)
            {
                if (isRefittable(layer->getName(), roleWeights.first))
                {
                    bool const success = refitter->setWeights(layer->getName(), roleWeights.first, roleWeights.second);
                    if (!success)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    auto const reportMissingWeights = [&] {
        auto const& missingPair = getMissingLayerWeightsRolePair(*refitter);
        auto const& layerNames = missingPair.first;
        auto const& weightsRoles = missingPair.second;
        for (size_t i = 0; i < layerNames.size(); ++i)
        {
            sample::gLogError << "Missing (" << layerNames[i] << ", " << weightsRoles[i] << ") for refitting."
                              << std::endl;
        }
        return layerNames.empty();
    };

    // Skip weights validation since we are confident that the new weights are similar to the weights used to build
    // engine.
    refitter->setWeightsValidation(false);

    // Warm up and report missing weights
    // We only need to set weights for the first time and that can be reused in later refitting process.
    bool const success = setWeights() && reportMissingWeights() && refitter->refitCudaEngine();
    if (!success)
    {
        return false;
    }

    TrtCudaStream stream;
    constexpr int32_t kLOOP = 10;
    time_point const refitStartTime{std::chrono::steady_clock::now()};
    {
        for (int32_t l = 0; l < kLOOP; l++)
        {
            if (!refitter->refitCudaEngineAsync(stream.get()))
            {
                return false;
            }
        }
    }
    stream.synchronize();
    time_point const refitEndTime{std::chrono::steady_clock::now()};

    sample::gLogInfo << "Engine refitted"
                     << " in " << durationMs(refitEndTime - refitStartTime).count() / kLOOP << " ms." << std::endl;
    return true;
}

namespace
{
void* initSafeRuntime()
{
    void* handle{nullptr};
    // Currently libnvinfer_safe_debug.so for samplesCommon::isDebug() is not ready.
#if !defined(_WIN32)
    std::string const dllName{"libnvinfer_safe.so"};
#if SANITIZER_BUILD
    handle = dlopen(dllName.c_str(), RTLD_LAZY | RTLD_NODELETE);
#else
    // RTLD_GLOBAL is used for symbol resolution of subsequently loaded plugin libraries
    handle = dlopen(dllName.c_str(), RTLD_LAZY | RTLD_GLOBAL);
#endif
#endif
    return handle;
}

void* initConsistencyCheckerLibrary()
{
    void* handle{nullptr};
#if !defined(_WIN32)
    std::string const dllName{"libnvinfer_checker_shared.so"};
#if SANITIZER_BUILD
    handle = dlopen(dllName.c_str(), RTLD_LAZY | RTLD_NODELETE);
#else
    handle = dlopen(dllName.c_str(), RTLD_LAZY);
#endif
#endif
    return handle;
}

#if !defined(_WIN32)
struct DllDeleter
{
    void operator()(void* handle)
    {
        if (handle != nullptr)
        {
            dlclose(handle);
        }
    }
};
const std::unique_ptr<void, DllDeleter> safeRuntimeLibrary{initSafeRuntime()};
const std::unique_ptr<void, DllDeleter> consistencyCheckerLibrary{initConsistencyCheckerLibrary()};
#endif
} // namespace

bool hasSafeRuntime()
{
#if defined(_WIN32)
    return false;
#else
    return (safeRuntimeLibrary != nullptr);
#endif
}

bool hasConsistencyChecker()
{
#if defined(_WIN32)
    return false;
#else
    return (consistencyCheckerLibrary != nullptr);
#endif
}

#if ENABLE_UNIFIED_BUILDER

nvinfer2::safe::consistency::IConsistencyChecker* createConsistencyChecker(sample::SampleSafeRecorder& recorder,
    void const* serializedEngine, int32_t const engineSize, std::vector<std::string> const& pluginBuildLibPath) noexcept
{
    nvinfer2::safe::consistency::IConsistencyChecker* checker{nullptr};

    if (serializedEngine == nullptr || engineSize == 0)
    {
        return checker;
    }

#if !defined(_WIN32)
    constexpr char symbolName[] = "createConsistencyChecker";
    typedef ErrorCode (*CreateCheckerFn)(nvinfer2::safe::consistency::IConsistencyChecker * &checker,
        sample::SampleSafeRecorder & recorder, void const* data, size_t size,
        std::vector<std::string> const& pluginBuildLibPath);
    if (hasSafeRuntime())
    {
        auto createFn = reinterpret_cast<CreateCheckerFn>(dlsym(consistencyCheckerLibrary.get(), symbolName));
        if (createFn != nullptr)
        {
            ErrorCode errorCode = createFn(checker, recorder, serializedEngine, engineSize, pluginBuildLibPath);
            if (errorCode != ErrorCode::kSUCCESS)
            {
                return nullptr;
            }
        }
    }
#endif
    return checker;
}
#endif

bool checkSafeEngine(
    void const* serializedEngine, int64_t const engineSize, std::vector<std::string> const& pluginBuildLibPath)
{
    if (!hasConsistencyChecker())
    {
        sample::gLogError << "Cannot perform consistency check because the checker is not loaded.." << std::endl;
        return false;
    }

#if ENABLE_UNIFIED_BUILDER
    sample::SampleSafeRecorder recorder{nvinfer2::safe::Severity::kINFO};
    auto checker = std::unique_ptr<nvinfer2::safe::consistency::IConsistencyChecker>(
        createConsistencyChecker(recorder, serializedEngine, engineSize, pluginBuildLibPath));
    if (checker.get() == nullptr)
    {
        sample::gLogError << "Failed to create consistency checker." << std::endl;
        return false;
    }
    sample::gLogInfo << "Start consistency checking." << std::endl;
    if (!checker->validate())
    {
        sample::gLogError << "Consistency validation failed." << std::endl;
        return false;
    }
    sample::gLogInfo << "Consistency validation passed." << std::endl;
    return true;
#else
    return false;
#endif
}

} // namespace sample
