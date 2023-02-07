/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "ErrorRecorder.h"
#include "common.h"
#include "half.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleOptions.h"
#include "sampleUtils.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

using namespace nvinfer1;

namespace sample
{

namespace
{

struct CaffeBufferShutter
{
    ~CaffeBufferShutter()
    {
        nvcaffeparser1::shutdownProtobufLibrary();
    }
};

struct UffBufferShutter
{
    ~UffBufferShutter()
    {
        nvuffparser::shutdownProtobufLibrary();
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
        SMP_RETVAL_IF_FALSE(
            !mEngineBlob.empty(), "Engine blob is empty. Nothing to deserialize!", nullptr, sample::gLogError);

        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using duration = std::chrono::duration<float>;
        time_point const deserializeStartTime{std::chrono::high_resolution_clock::now()};

        std::unique_ptr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
        SMP_RETVAL_IF_FALSE(runtime != nullptr, "Runtime creation failed", nullptr, sample::gLogError);
        if (mDLACore != -1)
        {
            runtime->setDLACore(mDLACore);
        }
        runtime->setErrorRecorder(&gRecorder);
        mEngine.reset(runtime->deserializeCudaEngine(mEngineBlob.data(), mEngineBlob.size()));
        SMP_RETVAL_IF_FALSE(mEngine != nullptr, "Engine deserialization failed", nullptr, sample::gLogError);

        time_point const deserializeEndTime{std::chrono::high_resolution_clock::now()};
        sample::gLogInfo << "Engine deserialized in "
                            << duration(deserializeEndTime - deserializeStartTime).count() << " sec." << std::endl;
    }

    return mEngine.get();
}

nvinfer1::ICudaEngine* LazilyDeserializedEngine::release()
{
    auto* engine = get();
    mEngine.release();
    return engine;
}

nvinfer1::safe::ICudaEngine* LazilyDeserializedEngine::getSafe()
{
    SMP_RETVAL_IF_FALSE(
        mIsSafe, "Safe mode is not enabled, but trying to get safe engine!", nullptr, sample::gLogError);

    ASSERT(sample::hasSafeRuntime());
    if (mSafeEngine == nullptr)
    {
        SMP_RETVAL_IF_FALSE(
            !mEngineBlob.empty(), "Engine blob is empty. Nothing to deserialize!", nullptr, sample::gLogError);

        SMP_RETVAL_IF_FALSE(
            mDLACore == -1, "Safe DLA engine built with kDLA_STANDALONE should not be deserialized in TRT!", nullptr,
            sample::gLogError);

        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using duration = std::chrono::duration<float>;
        time_point const deserializeStartTime{std::chrono::high_resolution_clock::now()};

        std::unique_ptr<safe::IRuntime> safeRuntime{sample::createSafeInferRuntime(sample::gLogger.getTRTLogger())};
        SMP_RETVAL_IF_FALSE(safeRuntime != nullptr, "SafeRuntime creation failed", nullptr, sample::gLogError);
        safeRuntime->setErrorRecorder(&gRecorder);
        mSafeEngine.reset(
            safeRuntime->deserializeCudaEngine(mEngineBlob.data(), mEngineBlob.size()));
        SMP_RETVAL_IF_FALSE(mSafeEngine != nullptr, "SafeEngine deserialization failed", nullptr, sample::gLogError);

        time_point const deserializeEndTime{std::chrono::high_resolution_clock::now()};
        sample::gLogInfo << "SafeEngine deserialized in "
                            << duration(deserializeEndTime - deserializeStartTime).count() << " sec." << std::endl;
    }

    return mSafeEngine.get();
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
    bool const broadcastOutputFormats = broadcastIOFormats(outputFormats, network.getNbInputs());
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

Parser modelToNetwork(const ModelOptions& model, nvinfer1::INetworkDefinition& network, std::ostream& err)
{
    sample::gLogInfo << "Start parsing network model" << std::endl;
    Parser parser;
    std::string const& modelName = model.baseModel.model;
    switch (model.baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        using namespace nvcaffeparser1;
        parser.caffeParser.reset(createCaffeParser());
        CaffeBufferShutter bufferShutter;
        auto const* const blobNameToTensor = parser.caffeParser->parse(
            model.prototxt.c_str(), modelName.empty() ? nullptr : modelName.c_str(), network, DataType::kFLOAT);
        if (!blobNameToTensor)
        {
            err << "Failed to parse caffe model or prototxt, tensors blob not found" << std::endl;
            parser.caffeParser.reset();
            break;
        }

        for (auto const& s : model.outputs)
        {
            if (blobNameToTensor->find(s.c_str()) == nullptr)
            {
                err << "Could not find output blob " << s << std::endl;
                parser.caffeParser.reset();
                break;
            }
            network.markOutput(*blobNameToTensor->find(s.c_str()));
        }
        break;
    }
    case ModelFormat::kUFF:
    {
        using namespace nvuffparser;
        parser.uffParser.reset(createUffParser());
        UffBufferShutter bufferShutter;
        for (auto const& s : model.uffInputs.inputs)
        {
            if (!parser.uffParser->registerInput(
                    s.first.c_str(), s.second, model.uffInputs.NHWC ? UffInputOrder::kNHWC : UffInputOrder::kNCHW))
            {
                err << "Failed to register input " << s.first << std::endl;
                parser.uffParser.reset();
                break;
            }
        }

        for (auto const& s : model.outputs)
        {
            if (!parser.uffParser->registerOutput(s.c_str()))
            {
                err << "Failed to register output " << s << std::endl;
                parser.uffParser.reset();
                break;
            }
        }

        if (!parser.uffParser->parse(model.baseModel.model.c_str(), network))
        {
            err << "Failed to parse uff file" << std::endl;
            parser.uffParser.reset();
            break;
        }
        break;
    }
    case ModelFormat::kONNX:
    {
        using namespace nvonnxparser;
        parser.onnxParser.reset(createParser(network, sample::gLogger.getTRTLogger()));
        if (!parser.onnxParser->parseFromFile(
                model.baseModel.model.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity())))
        {
            err << "Failed to parse onnx file" << std::endl;
            parser.onnxParser.reset();
        }
        break;
    }
    case ModelFormat::kANY: break;
    }

    sample::gLogInfo << "Finish parsing network model" << std::endl;
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
            cudaCheck(cudaFree(elem.second), mErr);
        }
    }

    bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override;

    int32_t getBatchSize() const noexcept override
    {
        return 1;
    }

    const void* readCalibrationCache(size_t& length) noexcept override;

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
        cudaCheck(cudaMalloc(&data, elemCount[i] * sizeof(float)), mErr);
        cudaCheck(cudaMemcpy(data, rnd_data.data(), elemCount[i] * sizeof(float), cudaMemcpyHostToDevice), mErr);

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

const void* RndInt8Calibrator::readCalibrationCache(size_t& length) noexcept
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
                }
            }
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

void setLayerPrecisions(INetworkDefinition& network, LayerPrecisions const& layerPrecisions)
{
    bool const hasGlobalPrecision{layerPrecisions.find("*") != layerPrecisions.end()};
    auto const globalPrecision = hasGlobalPrecision ? layerPrecisions.at("*") : nvinfer1::DataType::kFLOAT;
    bool hasLayerPrecisionSkipped{false};
    for (int32_t layerIdx = 0; layerIdx < network.getNbLayers(); ++layerIdx)
    {
        auto* layer = network.getLayer(layerIdx);
        auto const layerName = layer->getName();
        if (layerPrecisions.find(layer->getName()) != layerPrecisions.end())
        {
            layer->setPrecision(layerPrecisions.at(layer->getName()));
        }
        else if (hasGlobalPrecision)
        {
            // We should not set the layer precision if its default precision is INT32 or Bool.
            if (layer->getPrecision() == nvinfer1::DataType::kINT32
                || layer->getPrecision() == nvinfer1::DataType::kBOOL)
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because the "
                                    << " default layer precision is INT32 or Bool." << std::endl;
                continue;
            }
            // We should not set the constant layer precision if its weights are in INT32.
            if (layer->getType() == nvinfer1::LayerType::kCONSTANT
                && static_cast<IConstantLayer*>(layer)->getWeights().type == nvinfer1::DataType::kINT32)
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because this "
                                    << "constant layer has INT32 weights." << std::endl;
                continue;
            }
            // We should not set the layer precision if the layer operates on a shape tensor.
            if (layer->getNbInputs() >= 1 && layer->getInput(0)->isShapeTensor())
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because this layer "
                                    << "operates on a shape tensor." << std::endl;
                continue;
            }
            if (layer->getNbInputs() >= 1 && layer->getInput(0)->getType() == nvinfer1::DataType::kINT32
                && layer->getNbOutputs() >= 1 && layer->getOutput(0)->getType() == nvinfer1::DataType::kINT32)
            {
                hasLayerPrecisionSkipped = true;
                sample::gLogVerbose << "Skipped setting precision for layer " << layerName << " because this "
                                    << "layer has INT32 input and output." << std::endl;
                continue;
            }
            // All heuristics passed. Set the layer precision.
            layer->setPrecision(globalPrecision);
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
        if (layerOutputTypes.find(layer->getName()) != layerOutputTypes.end())
        {
            auto const& outputTypes = layerOutputTypes.at(layer->getName());
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
                layer->setOutputType(outputIdx, outputTypes.at(isBroadcast ? 0 : outputIdx));
            }
        }
        else if (hasGlobalOutputType)
        {
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

void setMemoryPoolLimits(IBuilderConfig& config, BuildOptions const& build)
{
    auto const roundToBytes = [](double const sizeInMB) { return static_cast<size_t>(sizeInMB * (1 << 20)); };
    if (build.workspace >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, roundToBytes(build.workspace));
    }
    if (build.dlaSRAM >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kDLA_MANAGED_SRAM, roundToBytes(build.dlaSRAM));
    }
    if (build.dlaLocalDRAM >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kDLA_LOCAL_DRAM, roundToBytes(build.dlaLocalDRAM));
    }
    if (build.dlaGlobalDRAM >= 0)
    {
        config.setMemoryPoolLimit(MemoryPoolType::kDLA_GLOBAL_DRAM, roundToBytes(build.dlaGlobalDRAM));
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
    setFlag(PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805);
    setFlag(PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805);
}

} // namespace

bool setupNetworkAndConfig(BuildOptions const& build, SystemOptions const& sys, IBuilder& builder,
    INetworkDefinition& network, IBuilderConfig& config, std::unique_ptr<nvinfer1::IInt8Calibrator>& calibrator,
    std::ostream& err, std::vector<std::vector<int8_t>>& sparseWeights)
{
    IOptimizationProfile* profile{nullptr};
    if (build.maxBatch)
    {
        builder.setMaxBatchSize(build.maxBatch);
    }
    else
    {
        profile = builder.createOptimizationProfile();
    }

    bool hasDynamicShapes{false};

    bool broadcastInputFormats = broadcastIOFormats(build.inputFormats, network.getNbInputs());

    if (profile)
    {
        // Check if the provided input tensor names match the input tensors of the engine.
        // Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
        for (auto const& shape : build.shapes)
        {
            bool tensorNameFound{false};
            for (int32_t i = 0; i < network.getNbInputs(); ++i)
            {
                if (network.getInput(i)->getName() == shape.first)
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
            int inputFormatIndex = broadcastInputFormats ? 0 : i;
            input->setType(build.inputFormats[inputFormatIndex].first);
            input->setAllowedFormats(build.inputFormats[inputFormatIndex].second);
        }
        else
        {
            switch (input->getType())
            {
            case DataType::kINT32:
            case DataType::kBOOL:
            case DataType::kHALF:
            case DataType::kUINT8:
                // Leave these as is.
                break;
            case DataType::kFLOAT:
            case DataType::kINT8:
                // User did not specify a floating-point format.  Default to kFLOAT.
                input->setType(DataType::kFLOAT);
                break;
            }
            input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }

        if (profile)
        {
            auto const dims = input->getDimensions();
            auto const isScalar = dims.nbDims == 0;
            auto const isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; })
                || input->isShapeTensor();
            if (isDynamicInput)
            {
                hasDynamicShapes = true;
                auto shape = build.shapes.find(input->getName());
                ShapeRange shapes{};

                // If no shape is provided, set dynamic dimensions to 1.
                if (shape == build.shapes.end())
                {
                    constexpr int DEFAULT_DIMENSION = 1;
                    std::vector<int> staticDims;
                    if (input->isShapeTensor())
                    {
                        if (isScalar)
                        {
                            staticDims.push_back(1);
                        }
                        else
                        {
                            staticDims.resize(dims.d[0]);
                            std::fill(staticDims.begin(), staticDims.end(), DEFAULT_DIMENSION);
                        }
                    }
                    else
                    {
                        staticDims.resize(dims.nbDims);
                        std::transform(dims.d, dims.d + dims.nbDims, staticDims.begin(),
                            [&](int dimension) { return dimension > 0 ? dimension : DEFAULT_DIMENSION; });
                    }
                    sample::gLogWarning << "Dynamic dimensions required for input: " << input->getName()
                                        << ", but no shapes were provided. Automatically overriding shape to: "
                                        << staticDims << std::endl;
                    std::fill(shapes.begin(), shapes.end(), staticDims);
                }
                else
                {
                    shapes = shape->second;
                }

                std::vector<int> profileDims{};
                if (input->isShapeTensor())
                {
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kMIN,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values MIN", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kOPT,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values OPT", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kMAX,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values MAX", false, err);
                }
                else
                {
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(input->getName(), OptProfileSelector::kMIN, toDims(profileDims)),
                        "Error in set dimensions to profile MIN", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(input->getName(), OptProfileSelector::kOPT, toDims(profileDims)),
                        "Error in set dimensions to profile OPT", false, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(input->getName(), OptProfileSelector::kMAX, toDims(profileDims)),
                        "Error in set dimensions to profile MAX", false, err);
                }
            }
        }
    }

    for (uint32_t i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        auto* output = network.getOutput(i);
        if (profile)
        {
            auto const dims = output->getDimensions();
            // A shape tensor output with known static dimensions may have dynamic shape values inside it.
            auto const isDynamicOutput = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; })
                || output->isShapeTensor();
            if (isDynamicOutput)
            {
                hasDynamicShapes = true;
            }
        }
    }

    if (!hasDynamicShapes && !build.shapes.empty())
    {
        sample::gLogError << "Static model does not take explicit shapes since the shape of inference tensors will be "
                             "determined by the model itself"
                          << std::endl;
        return false;
    }

    if (profile && hasDynamicShapes)
    {
        SMP_RETVAL_IF_FALSE(profile->isValid(), "Required optimization profile is invalid", false, err);
        SMP_RETVAL_IF_FALSE(
            config.addOptimizationProfile(profile) != -1, "Error in add optimization profile", false, err);
    }

    bool broadcastOutputFormats = broadcastIOFormats(build.outputFormats, network.getNbOutputs(), false);

    for (uint32_t i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        // Set formats and data types of outputs
        auto* output = network.getOutput(i);
        if (!build.outputFormats.empty())
        {
            int outputFormatIndex = broadcastOutputFormats ? 0 : i;
            output->setType(build.outputFormats[outputFormatIndex].first);
            output->setAllowedFormats(build.outputFormats[outputFormatIndex].second);
        }
        else
        {
            output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }
    }

    setMemoryPoolLimits(config, build);

    setPreviewFeatures(config, build);

    if (build.heuristic)
    {
        config.setFlag(BuilderFlag::kENABLE_TACTIC_HEURISTIC);
    }

    if (build.timingCacheMode == TimingCacheMode::kDISABLE)
    {
        config.setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
    }

    if (!build.tf32)
    {
        config.clearFlag(BuilderFlag::kTF32);
    }

    if (build.refittable)
    {
        config.setFlag(BuilderFlag::kREFIT);
    }

    if (build.sparsity != SparsityFlag::kDISABLE)
    {
        config.setFlag(BuilderFlag::kSPARSE_WEIGHTS);
        if (build.sparsity == SparsityFlag::kFORCE)
        {
            sparsify(network, sparseWeights);
        }
    }

    config.setProfilingVerbosity(build.profilingVerbosity);
    config.setMinTimingIterations(build.minTiming);
    config.setAvgTimingIterations(build.avgTiming);

    if (build.fp16)
    {
        config.setFlag(BuilderFlag::kFP16);
    }

    if (build.int8)
    {
        config.setFlag(BuilderFlag::kINT8);
    }

    if (build.int8 && !build.fp16)
    {
        sample::gLogInfo
            << "FP32 and INT8 precisions have been specified - more performance might be enabled by additionally "
               "specifying --fp16 or --best"
            << std::endl;
    }

    auto isInt8 = [](const IOFormat& format) { return format.first == DataType::kINT8; };
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
                auto shape = build.shapesCalib.find(input->getName());
                ShapeRange shapesCalib{};
                shapesCalib = shape->second;

                profileDims = toDims(shapesCalib[static_cast<size_t>(OptProfileSelector::kOPT)]);
                // Here we check only kMIN as all profileDims are the same.
                SMP_RETVAL_IF_FALSE(
                    profileCalib->setDimensions(input->getName(), OptProfileSelector::kMIN, profileDims),
                    "Error in set dimensions to calibration profile OPT", false, err);
                profileCalib->setDimensions(input->getName(), OptProfileSelector::kOPT, profileDims);
                profileCalib->setDimensions(input->getName(), OptProfileSelector::kMAX, profileDims);
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
            else if (profile && isDynamicInput)
            {
                elemCount.push_back(volume(profile->getDimensions(input->getName(), OptProfileSelector::kOPT)));
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
    case PrecisionConstraints::kOBEY:
        config.setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
        break;
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

    if (build.safe)
    {
        config.setEngineCapability(sys.DLACore != -1 ? EngineCapability::kDLA_STANDALONE : EngineCapability::kSAFETY);
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

            if (sys.fallback)
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

    return true;
}

//!
//! \brief Create a serialized engine for a network defintion
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

    std::unique_ptr<ITimingCache> timingCache{nullptr};
    // Try to load cache from file. Create a fresh cache if the file doesn't exist
    if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
    {
        std::vector<char> loadedCache = samplesCommon::loadTimingCacheFile(build.timingCacheFile);
        timingCache.reset(config->createTimingCache(static_cast<const void*>(loadedCache.data()), loadedCache.size()));
        SMP_RETVAL_IF_FALSE(timingCache != nullptr, "TimingCache creation failed", false, err);
        config->setTimingCache(*timingCache, false);
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    SMP_RETVAL_IF_FALSE(profileStream != nullptr, "Cuda stream creation failed", false, err);
    config->setProfileStream(*profileStream);

    std::unique_ptr<IHostMemory> serializedEngine{builder.buildSerializedNetwork(*env.network, *config)};
    SMP_RETVAL_IF_FALSE(serializedEngine != nullptr, "Engine could not be created from network", false, err);

    env.engine.setBlob(serializedEngine->data(), serializedEngine->size());

    if (build.safe && build.consistency)
    {
        checkSafeEngine(serializedEngine->data(), serializedEngine->size());
    }

    if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
    {
        auto timingCache = config->getTimingCache();
        samplesCommon::updateTimingCacheFile(build.timingCacheFile, timingCache);
    }

    return true;
}

//!
//! \brief Parse a given model, create a network and an engine.
//!
bool modelToBuildEnv(const ModelOptions& model, BuildOptions const& build, SystemOptions const& sys,
    BuildEnvironment& env, std::ostream& err)
{
    env.builder.reset(createInferBuilder(sample::gLogger.getTRTLogger()));
    SMP_RETVAL_IF_FALSE(env.builder != nullptr, "Builder creation failed", false, err);
    env.builder->setErrorRecorder(&gRecorder);
    auto networkFlags
        = (build.maxBatch) ? 0U : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    env.network.reset(env.builder->createNetworkV2(networkFlags));
    SMP_RETVAL_IF_FALSE(env.network != nullptr, "Network creation failed", false, err);
    env.parser = modelToNetwork(model, *env.network, err);
    SMP_RETVAL_IF_FALSE(env.parser.operator bool(), "Parsing model failed", false, err);
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

bool loadEngineToBuildEnv(std::string const& engine, bool enableConsistency, BuildEnvironment& env, std::ostream& err)
{
    std::ifstream engineFile(engine, std::ios::binary);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error opening engine file: " << engine);
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> engineBlob(fsize);
    engineFile.read(reinterpret_cast<char*>(engineBlob.data()), fsize);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error loading engine file: " << engine);

    if (enableConsistency)
    {
        checkSafeEngine(engineBlob.data(), fsize);
    }

    env.engine.setBlob(engineBlob.data(), engineBlob.size());

    return true;
}

void dumpRefittable(nvinfer1::ICudaEngine& engine)
{
    std::unique_ptr<IRefitter> refitter{createInferRefitter(engine, sample::gLogger.getTRTLogger())};
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
    BuildEnvironment env(false, DLACore);
    return loadEngineToBuildEnv(engine, false, env, err) ? env.engine.release() : nullptr;
}

bool saveEngine(const ICudaEngine& engine, std::string const& fileName, std::ostream& err)
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

bool getEngineBuildEnv(const ModelOptions& model, BuildOptions const& build, SystemOptions const& sys,
    BuildEnvironment& env, std::ostream& err)
{
    bool createEngineSuccess {false};

    if (build.load)
    {
        createEngineSuccess = loadEngineToBuildEnv(build.engine, build.safe && build.consistency, env, err);
    }
    else
    {
        createEngineSuccess = modelToBuildEnv(model, build, sys, env, err);
    }

    SMP_RETVAL_IF_FALSE(createEngineSuccess, "Failed to create engine from model or file.", false, err);

    if (build.save)
    {
        std::ofstream engineFile(build.engine, std::ios::binary);
        engineFile.write(reinterpret_cast<char const*>(env.engine.getBlob().data()), env.engine.getBlob().size());
        SMP_RETVAL_IF_FALSE(!engineFile.fail(), "Saving engine to file failed.", false, err);
    }

    return true;
}

// There is not a getWeightsName API, so we need to use WeightsRole.
std::vector<std::pair<WeightsRole, Weights>> getAllRefitWeightsForLayer(const ILayer& l)
{
    switch (l.getType())
    {
    case LayerType::kCONSTANT:
    {
        auto const& layer = static_cast<const nvinfer1::IConstantLayer&>(l);
        auto const weights = layer.getWeights();
        switch (weights.type)
        {
        case DataType::kFLOAT:
        case DataType::kHALF:
        case DataType::kINT8:
        case DataType::kINT32: return {std::make_pair(WeightsRole::kCONSTANT, weights)};
        case DataType::kBOOL:
        case DataType::kUINT8:
            // Refit not supported for these types.
            break;
        }
        break;
    }
    case LayerType::kCONVOLUTION:
    {
        auto const& layer = static_cast<const nvinfer1::IConvolutionLayer&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kDECONVOLUTION:
    {
        auto const& layer = static_cast<const nvinfer1::IDeconvolutionLayer&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kFULLY_CONNECTED:
    {
        auto const& layer = static_cast<const nvinfer1::IFullyConnectedLayer&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kSCALE:
    {
        auto const& layer = static_cast<const nvinfer1::IScaleLayer&>(l);
        return {std::make_pair(WeightsRole::kSCALE, layer.getScale()),
            std::make_pair(WeightsRole::kSHIFT, layer.getShift())};
    }
    case LayerType::kACTIVATION:
    case LayerType::kASSERTION:
    case LayerType::kCONCATENATION:
    case LayerType::kCONDITION:
    case LayerType::kCONDITIONAL_INPUT:
    case LayerType::kCONDITIONAL_OUTPUT:
    case LayerType::kDEQUANTIZE:
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
    case LayerType::kONE_HOT:
    case LayerType::kPADDING:
    case LayerType::kPARAMETRIC_RELU:
    case LayerType::kPLUGIN:
    case LayerType::kPLUGIN_V2:
    case LayerType::kPOOLING:
    case LayerType::kQUANTIZE:
    case LayerType::kRAGGED_SOFTMAX:
    case LayerType::kRECURRENCE:
    case LayerType::kREDUCE:
    case LayerType::kRESIZE:
    case LayerType::kRNN_V2:
    case LayerType::kSCATTER:
    case LayerType::kSELECT:
    case LayerType::kSHAPE:
    case LayerType::kSHUFFLE:
    case LayerType::kSLICE:
    case LayerType::kSOFTMAX:
    case LayerType::kTOPK:
    case LayerType::kTRIP_LIMIT:
    case LayerType::kUNARY: return {};
    }
    return {};
}

bool timeRefit(INetworkDefinition const& network, nvinfer1::ICudaEngine& engine, bool multiThreading)
{
    using time_point = std::chrono::time_point<std::chrono::steady_clock>;
    using durationMs = std::chrono::duration<float, std::milli>;

    auto const nbLayers = network.getNbLayers();
    std::unique_ptr<IRefitter> refitter{createInferRefitter(engine, sample::gLogger.getTRTLogger())};
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

    // Warm up and report missing weights
    bool const success = setWeights() && reportMissingWeights() && refitter->refitCudaEngine();
    if (!success)
    {
        return false;
    }

    constexpr int32_t loop = 5;
    time_point const refitStartTime{std::chrono::steady_clock::now()};
    {
        for (int32_t l = 0; l < loop; l++)
        {
            bool const success = setWeights() && refitter->refitCudaEngine();
            if (!success)
            {
                return false;
            }
        }
    }
    time_point const refitEndTime{std::chrono::steady_clock::now()};

    sample::gLogInfo << "Engine refitted"
        << " in " << durationMs(refitEndTime - refitStartTime).count() / loop << " ms." << std::endl;
    return true;
}

namespace
{
void* initSafeRuntime()
{
    void* handle{nullptr};
#if !defined(_WIN32)
    std::string const dllName{samplesCommon::isDebug() ? "libnvinfer_safe_debug.so.8" : "libnvinfer_safe.so.8"};
#if SANITIZER_BUILD
    handle = dlopen(dllName.c_str(), RTLD_LAZY | RTLD_NODELETE);
#else
    handle = dlopen(dllName.c_str(), RTLD_LAZY);
#endif
#endif
    return handle;
}

void* initConsistencyCheckerLibrary()
{
    void* handle{nullptr};
#if !defined(_WIN32)
    std::string const dllName{samplesCommon::isDebug() ? "libnvinfer_checker_debug.so.8" : "libnvinfer_checker.so.8"};
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
    bool ret{false};
#if !defined(_WIN32)
    ret = (safeRuntimeLibrary != nullptr);
#endif
    return ret;
}

nvinfer1::safe::IRuntime* createSafeInferRuntime(nvinfer1::ILogger& logger) noexcept
{
    nvinfer1::safe::IRuntime* runtime{nullptr};
#if !defined(_WIN32)
    constexpr char symbolName[] = "_ZN8nvinfer14safe18createInferRuntimeERNS_7ILoggerE";
    typedef nvinfer1::safe::IRuntime* (*CreateInferRuntimeFn)(nvinfer1::ILogger & logger);
    if (hasSafeRuntime())
    {
        auto createFn = reinterpret_cast<CreateInferRuntimeFn>(dlsym(safeRuntimeLibrary.get(), symbolName));
        if (createFn != nullptr)
        {
            runtime = createFn(logger);
        }
    }
#endif
    return runtime;
}

bool hasConsistencyChecker()
{
    bool ret{false};
#if !defined(_WIN32)
    ret = (consistencyCheckerLibrary != nullptr);
#endif
    return ret;
}

nvinfer1::consistency::IConsistencyChecker* createConsistencyChecker(
    nvinfer1::ILogger& logger, void const* serializedEngine, int32_t const engineSize) noexcept
{
    nvinfer1::consistency::IConsistencyChecker* checker{nullptr};

    if (serializedEngine == nullptr || engineSize == 0)
    {
        return checker;
    }

#if !defined(_WIN32)
    constexpr char symbolName[] = "createConsistencyChecker_INTERNAL";
    typedef nvinfer1::consistency::IConsistencyChecker* (*CreateCheckerFn)(
        nvinfer1::ILogger * logger, void const* data, size_t size, uint32_t version);
    if (hasSafeRuntime())
    {
        auto createFn = reinterpret_cast<CreateCheckerFn>(dlsym(consistencyCheckerLibrary.get(), symbolName));
        if (createFn != nullptr)
        {
            checker = createFn(&logger, serializedEngine, engineSize, NV_TENSORRT_VERSION);
        }
    }
#endif
    return checker;
}

bool checkSafeEngine(void const* serializedEngine, int32_t const engineSize)
{
    if (!hasConsistencyChecker())
    {
        sample::gLogError << "Cannot perform consistency check because the checker is not loaded.." << std::endl;
        return false;
    }
    auto checker = std::unique_ptr<nvinfer1::consistency::IConsistencyChecker>(
        createConsistencyChecker(sample::gLogger.getTRTLogger(), serializedEngine, engineSize));
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
}
} // namespace sample
