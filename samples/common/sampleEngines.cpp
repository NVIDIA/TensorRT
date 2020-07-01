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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "logger.h"
#include "sampleEngines.h"
#include "sampleOptions.h"
#include "sampleUtils.h"

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

} // namespace

#define SMP_RETVAL_IF_FALSE(condition, msg, retval, err)                                                               \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            err << msg << std::endl;                                                                                   \
            return retval;                                                                                             \
        }                                                                                                              \
    }

Parser modelToNetwork(const ModelOptions& model, nvinfer1::INetworkDefinition& network, std::ostream& err)
{
    Parser parser;
    const std::string& modelName = model.baseModel.model;
    switch (model.baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        using namespace nvcaffeparser1;
        parser.caffeParser.reset(createCaffeParser());
        CaffeBufferShutter bufferShutter;
        const auto blobNameToTensor = parser.caffeParser->parse(
            model.prototxt.c_str(), modelName.empty() ? nullptr : modelName.c_str(), network, DataType::kFLOAT);
        if (!blobNameToTensor)
        {
            err << "Failed to parse caffe model or prototxt, tensors blob not found" << std::endl;
            parser.caffeParser.reset();
            break;
        }

        for (const auto& s : model.outputs)
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
        for (const auto& s : model.uffInputs.inputs)
        {
            if (!parser.uffParser->registerInput(
                    s.first.c_str(), s.second, model.uffInputs.NHWC ? UffInputOrder::kNHWC : UffInputOrder::kNCHW))
            {
                err << "Failed to register input " << s.first << std::endl;
                parser.uffParser.reset();
                break;
            }
        }

        for (const auto& s : model.outputs)
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

    return parser;
}

namespace
{

class RndInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    RndInt8Calibrator(int batches, std::vector<int>& elemCount, const std::string& cacheFile,
        const nvinfer1::INetworkDefinition& network, std::ostream& err);

    ~RndInt8Calibrator()
    {
        for (auto& elem : mInputDeviceBuffers)
        {
            cudaCheck(cudaFree(elem.second), mErr);
        }
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

    int getBatchSize() const override
    {
        return 1;
    }

    const void* readCalibrationCache(size_t& length) override;

    virtual void writeCalibrationCache(const void*, size_t) override {}

private:
    int mBatches{};
    int mCurrentBatch{};
    std::string mCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
    std::ostream& mErr;
};

RndInt8Calibrator::RndInt8Calibrator(int batches, std::vector<int>& elemCount, const std::string& cacheFile,
    const INetworkDefinition& network, std::ostream& err)
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

    for (int i = 0; i < network.getNbInputs(); i++)
    {
        auto input = network.getInput(i);
        std::vector<float> rnd_data(elemCount[i]);
        std::generate_n(rnd_data.begin(), elemCount[i], gen);

        void* data;
        cudaCheck(cudaMalloc(&data, elemCount[i] * sizeof(float)), mErr);
        cudaCheck(cudaMemcpy(data, rnd_data.data(), elemCount[i] * sizeof(float), cudaMemcpyHostToDevice), mErr);

        mInputDeviceBuffers.insert(std::make_pair(input->getName(), data));
    }
}

bool RndInt8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if (mCurrentBatch >= mBatches)
    {
        return false;
    }

    for (int i = 0; i < nbBindings; ++i)
    {
        bindings[i] = mInputDeviceBuffers[names[i]];
    }

    ++mCurrentBatch;

    return true;
}

const void* RndInt8Calibrator::readCalibrationCache(size_t& length)
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
    return mCalibrationCache.size() ? mCalibrationCache.data() : nullptr;
}

bool setTensorScales(const INetworkDefinition& network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int l = 0; l < network.getNbLayers(); l++)
    {
        auto layer = network.getLayer(l);
        for (int i = 0; i < layer->getNbInputs(); i++)
        {
            ITensor* input{layer->getInput(i)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input && !input->dynamicRangeIsSet())
            {
                if (!input->setDynamicRange(-inScales, inScales))
                {
                    return false;
                }
            }
        }
        for (int o = 0; o < layer->getNbOutputs(); o++)
        {
            ITensor* output{layer->getOutput(o)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == LayerType::kPOOLING)
                {
                    if (!output->setDynamicRange(-inScales, inScales))
                    {
                        return false;
                    }
                }
                else
                {
                    if (!output->setDynamicRange(-outScales, outScales))
                    {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

} // namespace

ICudaEngine* networkToEngine(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder,
    INetworkDefinition& network, std::ostream& err)
{
    TrtUniquePtr<IBuilderConfig> config{builder.createBuilderConfig()};

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
    if (!build.inputFormats.empty() && build.inputFormats.size() != static_cast<size_t>(network.getNbInputs()))
    {
        throw std::invalid_argument("The number of inputIOFormats must match network's inputs.");
    }
    for (unsigned int i = 0, n = network.getNbInputs(); i < n; i++)
    {
        // Set formats and data types of inputs
        auto input = network.getInput(i);
        if (!build.inputFormats.empty())
        {
            input->setType(build.inputFormats[i].first);
            input->setAllowedFormats(build.inputFormats[i].second);
        }
        else
        {
            switch (input->getType())
            {
            case DataType::kINT32:
            case DataType::kBOOL:
            case DataType::kHALF:
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
            Dims dims = input->getDimensions();
            const bool isScalar = dims.nbDims == 0;
            const bool isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; })
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
                            [&DEFAULT_DIMENSION](int dim) { return dim > 0 ? dim : DEFAULT_DIMENSION; });
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
                        "Error in set shape values MIN", nullptr, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kOPT,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values OPT", nullptr, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
                    SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kMAX,
                                            profileDims.data(), static_cast<int>(profileDims.size())),
                        "Error in set shape values MAX", nullptr, err);
                }
                else
                {
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(input->getName(), OptProfileSelector::kMIN, toDims(profileDims)),
                        "Error in set dimensions to profile MIN", nullptr, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(input->getName(), OptProfileSelector::kOPT, toDims(profileDims)),
                        "Error in set dimensions to profile OPT", nullptr, err);
                    profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
                    SMP_RETVAL_IF_FALSE(
                        profile->setDimensions(input->getName(), OptProfileSelector::kMAX, toDims(profileDims)),
                        "Error in set dimensions to profile MAX", nullptr, err);
                }
            }
        }
    }

    if (profile && hasDynamicShapes)
    {
        SMP_RETVAL_IF_FALSE(profile->isValid(), "Required optimization profile is invalid", nullptr, err);
        SMP_RETVAL_IF_FALSE(
            config->addOptimizationProfile(profile) != -1, "Error in add optimization profile", nullptr, err);
    }

    if (!build.outputFormats.empty() && build.outputFormats.size() != static_cast<size_t>(network.getNbOutputs()))
    {
        throw std::invalid_argument("The number of outputIOFormats must match network's outputs.");
    }
    for (unsigned int i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        // Set formats and data types of outputs
        auto output = network.getOutput(i);
        if (!build.outputFormats.empty())
        {
            output->setType(build.outputFormats[i].first);
            output->setAllowedFormats(build.outputFormats[i].second);
        }
        else
        {
            output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }
    }

    config->setMaxWorkspaceSize(static_cast<size_t>(build.workspace) << 20);

    if (!build.builderCache)
    {
        config->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
    }

    if (!build.tf32)
    {
        config->clearFlag(BuilderFlag::kTF32);
    }

    config->setProfilingVerbosity(build.nvtxMode);
    config->setMinTimingIterations(build.minTiming);
    config->setAvgTimingIterations(build.avgTiming);

    if (build.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    if (build.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
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

    if ((build.int8 && build.calibration.empty()) || int8IO)
    {
        // Explicitly set int8 scales if no calibrator is provided and if I/O tensors use int8,
        // because auto calibration does not support this case.
        SMP_RETVAL_IF_FALSE(setTensorScales(network), "Error in set tensor scales.", nullptr, err);
    }
    else if (build.int8)
    {
        IOptimizationProfile* profileCalib{nullptr};
        if (!build.shapesCalib.empty())
        {
            profileCalib = builder.createOptimizationProfile();
            for (unsigned int i = 0, n = network.getNbInputs(); i < n; i++)
            {
                auto input = network.getInput(i);
                Dims profileDims{};
                auto shape = build.shapesCalib.find(input->getName());
                ShapeRange shapesCalib{};
                shapesCalib = shape->second;

                profileDims = toDims(shapesCalib[static_cast<size_t>(OptProfileSelector::kOPT)]);
                // Here we check only kMIN as all profileDims are the same.
                SMP_RETVAL_IF_FALSE(
                    profileCalib->setDimensions(input->getName(), OptProfileSelector::kMIN, profileDims),
                    "Error in set dimensions to calibration profile OPT", nullptr, err);
                profileCalib->setDimensions(input->getName(), OptProfileSelector::kOPT, profileDims);
                profileCalib->setDimensions(input->getName(), OptProfileSelector::kMAX, profileDims);
            }
            SMP_RETVAL_IF_FALSE(profileCalib->isValid(), "Calibration profile is invalid", nullptr, err);
            SMP_RETVAL_IF_FALSE(
                config->setCalibrationProfile(profileCalib), "Error in set calibration profile", nullptr, err);
        }

        std::vector<int> elemCount{};
        for (int i = 0; i < network.getNbInputs(); i++)
        {
            auto input = network.getInput(i);
            if (profileCalib)
            {
                elemCount.push_back(volume(profileCalib->getDimensions(input->getName(), OptProfileSelector::kOPT)));
            }
            else if (profile)
            {
                elemCount.push_back(volume(profile->getDimensions(input->getName(), OptProfileSelector::kOPT)));
            }
            else
            {
                elemCount.push_back(volume(input->getDimensions()));
            }
        }

        config->setInt8Calibrator(new RndInt8Calibrator(1, elemCount, build.calibration, network, err));
    }

    if (build.safe)
    {
        config->setEngineCapability(sys.DLACore != -1 ? EngineCapability::kSAFE_DLA : EngineCapability::kSAFE_GPU);
    }

    if (sys.DLACore != -1)
    {
        if (sys.DLACore < builder.getNbDLACores())
        {
            config->setDefaultDeviceType(DeviceType::kDLA);
            config->setDLACore(sys.DLACore);
            config->setFlag(BuilderFlag::kSTRICT_TYPES);

            if (sys.fallback)
            {
                config->setFlag(BuilderFlag::kGPU_FALLBACK);
            }
            if (!build.int8)
            {
                config->setFlag(BuilderFlag::kFP16);
            }
        }
        else
        {
            err << "Cannot create DLA engine, " << sys.DLACore << " not available" << std::endl;
            return nullptr;
        }
    }

    return builder.buildEngineWithConfig(network, *config);
}

ICudaEngine* modelToEngine(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err)
{
    TrtUniquePtr<IBuilder> builder{createInferBuilder(sample::gLogger.getTRTLogger())};
    if (builder == nullptr)
    {
        err << "Builder creation failed" << std::endl;
        return nullptr;
    }
    const bool isOnnxModel = model.baseModel.format == ModelFormat::kONNX;
    auto batchFlag = (build.maxBatch && !isOnnxModel)
        ? 0U
        : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<INetworkDefinition> network{builder->createNetworkV2(batchFlag)};
    if (!network)
    {
        err << "Network creation failed" << std::endl;
        return nullptr;
    }
    Parser parser = modelToNetwork(model, *network, err);
    if (!parser)
    {
        err << "Parsing model failed" << std::endl;
        return nullptr;
    }

    return networkToEngine(build, sys, *builder, *network, err);
}

ICudaEngine* loadEngine(const std::string& engine, int DLACore, std::ostream& err)
{
    std::ifstream engineFile(engine, std::ios::binary);
    if (!engineFile)
    {
        err << "Error opening engine file: " << engine << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        err << "Error loading engine file: " << engine << std::endl;
        return nullptr;
    }

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (DLACore != -1)
    {
        runtime->setDLACore(DLACore);
    }

    return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}

bool saveEngine(const ICudaEngine& engine, const std::string& fileName, std::ostream& err)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        err << "Cannot open engine file: " << fileName << std::endl;
        return false;
    }

    TrtUniquePtr<IHostMemory> serializedEngine{engine.serialize()};
    if (serializedEngine == nullptr)
    {
        err << "Engine serialization failed" << std::endl;
        return false;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

TrtUniquePtr<nvinfer1::ICudaEngine> getEngine(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err)
{
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    if (build.load)
    {
        engine.reset(loadEngine(build.engine, sys.DLACore, err));
    }
    else
    {
        engine.reset(modelToEngine(model, build, sys, err));
    }
    if (!engine)
    {
        err << "Engine creation failed" << std::endl;
        return nullptr;
    }
    if (build.save && !saveEngine(*engine, build.engine, err))
    {
        err << "Saving engine to file failed" << std::endl;
        return nullptr;
    }
    return engine;
}

} // namespace sample
