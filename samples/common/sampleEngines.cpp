/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <set>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "common.h"
#include "ErrorRecorder.h"
#include "half.h"
#include "logger.h"
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

std::map<std::string, float> readScalesFromCalibrationCache(const std::string& calibrationFile)
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
            const auto tensorName = line.substr(0, colonPos);
            tensorScales[tensorName] = *reinterpret_cast<float*>(&scalesAsInt);
        }
    }
    cache.close();
    return tensorScales;
}
} // namespace

void setTensorScalesFromCalibration(nvinfer1::INetworkDefinition& network, const std::vector<IOFormat>& inputFormats,
    const std::vector<IOFormat>& outputFormats, const std::string& calibrationFile)
{
    const auto tensorScales = readScalesFromCalibrationCache(calibrationFile);
    const bool broadcastInputFormats = broadcastIOFormats(inputFormats, network.getNbInputs());
    for (int32_t i = 0, n = network.getNbInputs(); i < n; ++i)
    {
        int32_t formatIdx = broadcastInputFormats ? 0 : i;
        if (!inputFormats.empty() && inputFormats[formatIdx].first == DataType::kINT8)
        {
            auto* input = network.getInput(i);
            const auto calibScale = tensorScales.at(input->getName());
            input->setDynamicRange(-127 * calibScale, 127 * calibScale);
        }
    }
    const bool broadcastOutputFormats = broadcastIOFormats(outputFormats, network.getNbInputs());
    for (int32_t i = 0, n = network.getNbOutputs(); i < n; ++i)
    {
        int32_t formatIdx = broadcastOutputFormats ? 0 : i;
        if (!outputFormats.empty() && outputFormats[formatIdx].first == DataType::kINT8)
        {
            auto* output = network.getOutput(i);
            const auto calibScale = tensorScales.at(output->getName());
            output->setDynamicRange(-127 * calibScale, 127 * calibScale);
        }
    }
}

#define SMP_RETVAL_IF_FALSE(condition, msg, retval, err)                                                               \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            (err) << (msg) << std::endl;                                                                               \
            return retval;                                                                                             \
        }                                                                                                              \
    }

Parser modelToNetwork(const ModelOptions& model, nvinfer1::INetworkDefinition& network, std::ostream& err)
{
    sample::gLogInfo << "Start parsing network model" << std::endl;
    Parser parser;
    const std::string& modelName = model.baseModel.model;
    switch (model.baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        using namespace nvcaffeparser1;
        parser.caffeParser.reset(createCaffeParser());
        CaffeBufferShutter bufferShutter;
        const auto* const blobNameToTensor = parser.caffeParser->parse(
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

    sample::gLogInfo << "Finish parsing network model" << std::endl;
    return parser;
}

namespace
{

class RndInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    RndInt8Calibrator(int batches, std::vector<int64_t>& elemCount, const std::string& cacheFile,
        const nvinfer1::INetworkDefinition& network, std::ostream& err);

    ~RndInt8Calibrator()
    {
        for (auto& elem : mInputDeviceBuffers)
        {
            cudaCheck(cudaFree(elem.second), mErr);
        }
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    int getBatchSize() const noexcept override
    {
        return 1;
    }

    const void* readCalibrationCache(size_t& length) noexcept override;

    virtual void writeCalibrationCache(const void*, size_t) noexcept override {}

private:
    int mBatches{};
    int mCurrentBatch{};
    std::string mCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
    std::ostream& mErr;
};

RndInt8Calibrator::RndInt8Calibrator(int batches, std::vector<int64_t>& elemCount, const std::string& cacheFile,
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
        auto* input = network.getInput(i);
        std::vector<float> rnd_data(elemCount[i]);
        std::generate_n(rnd_data.begin(), elemCount[i], gen);

        void* data;
        cudaCheck(cudaMalloc(&data, elemCount[i] * sizeof(float)), mErr);
        cudaCheck(cudaMemcpy(data, rnd_data.data(), elemCount[i] * sizeof(float), cudaMemcpyHostToDevice), mErr);

        mInputDeviceBuffers.insert(std::make_pair(input->getName(), data));
    }
}

bool RndInt8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
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

bool setTensorDynamicRange(const INetworkDefinition& network, float inRange = 2.0F, float outRange = 4.0F)
{
    // Ensure that all layer inputs have a dynamic range.
    for (int l = 0; l < network.getNbLayers(); l++)
    {
        auto* layer = network.getLayer(l);
        for (int i = 0; i < layer->getNbInputs(); i++)
        {
            ITensor* input{layer->getInput(i)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input && !input->dynamicRangeIsSet())
            {
                if (!input->setDynamicRange(-inRange, inRange))
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

template <typename T>
void sparsify(const T* values, int64_t count, int32_t k, int32_t rs, std::vector<char>& sparseWeights)
{
    const auto c = count / (k * rs);
    sparseWeights.resize(count * sizeof(T));
    auto* sparseValues = reinterpret_cast<T*>(sparseWeights.data());

    constexpr int32_t window = 4;
    constexpr int32_t nonzeros = 2;

    const int32_t crs = c * rs;
    const auto getIndex = [=](int32_t ki, int32_t ci, int32_t rsi) { return ki * crs + ci * rs + rsi; };

    for (int64_t ki = 0; ki < k; ++ki)
    {
        for (int64_t rsi = 0; rsi < rs; ++rsi)
        {
            int32_t w = 0;
            int32_t nz = 0;
            for (int64_t ci = 0; ci < c; ++ci)
            {
                const auto index = getIndex(ki, ci, rsi);
                if (nz < nonzeros)
                {
                    sparseValues[index] = values[index];
                    ++nz;
                }
                else
                {
                    sparseValues[index] = 0;
                }
                if (++w == window)
                {
                    w = 0;
                    nz = 0;
                }
            }
        }
    }
}

void sparsify(const Weights& weights, int32_t k, int32_t rs, std::vector<char>& sparseWeights)
{
    switch (weights.type)
    {
    case DataType::kFLOAT:
        sparsify(static_cast<const float*>(weights.values), weights.count, k, rs, sparseWeights);
        break;
    case DataType::kHALF:
        sparsify(static_cast<const half_float::half*>(weights.values), weights.count, k, rs, sparseWeights);
        break;
    case DataType::kINT8:
    case DataType::kINT32:
    case DataType::kBOOL: break;
    }
}

template <typename L>
void setSparseWeights(L& l, int32_t k, int32_t rs, std::vector<char>& sparseWeights)
{
    auto weights = l.getKernelWeights();
    sparsify(weights, k, rs, sparseWeights);
    weights.values = sparseWeights.data();
    l.setKernelWeights(weights);
}

void sparsify(INetworkDefinition& network, std::vector<std::vector<char>>& sparseWeights)
{
    for (int32_t l = 0; l < network.getNbLayers(); ++l)
    {
        auto* layer = network.getLayer(l);
        const auto t = layer->getType();
        if (t == LayerType::kCONVOLUTION)
        {
            auto& conv = *static_cast<IConvolutionLayer*>(layer);
            const auto& dims = conv.getKernelSizeNd();
            if (dims.nbDims > 2)
            {
                continue;
            }
            const auto k = conv.getNbOutputMaps();
            const auto rs = dims.d[0] * dims.d[1];
            sparseWeights.emplace_back();
            setSparseWeights(conv, k, rs, sparseWeights.back());
        }
        else if (t == LayerType::kFULLY_CONNECTED)
        {
            auto& fc = *static_cast<IFullyConnectedLayer*>(layer);
            const auto k = fc.getNbOutputChannels();
            sparseWeights.emplace_back();
            setSparseWeights(fc, k, 1, sparseWeights.back());
        }
    }
}

} // namespace

bool setupNetworkAndConfig(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder,
    INetworkDefinition& network, IBuilderConfig& config, std::ostream& err,
    std::vector<std::vector<char>>& sparseWeights)
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
        for (const auto& shape : build.shapes)
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

    config.setMaxWorkspaceSize(static_cast<size_t>(build.workspace) << 20);

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
        const auto nbLayers = network.getNbLayers();
        for (int32_t i = 0; i < nbLayers; i++)
        {
            const auto& layer = network.getLayer(i);
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
            if (profileCalib)
            {
                elemCount.push_back(volume(profileCalib->getDimensions(input->getName(), OptProfileSelector::kOPT)));
            }
            else if (profile && hasDynamicShapes)
            {
                elemCount.push_back(volume(profile->getDimensions(input->getName(), OptProfileSelector::kOPT)));
            }
            else
            {
                elemCount.push_back(volume(input->getDimensions()));
            }
        }

        config.setInt8Calibrator(new RndInt8Calibrator(1, elemCount, build.calibration, network, err));
    }

    if (build.strictTypes)
    {
        config.setFlag(BuilderFlag::kSTRICT_TYPES);
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
            config.setFlag(BuilderFlag::kSTRICT_TYPES);

            if (sys.fallback)
            {
                config.setFlag(BuilderFlag::kGPU_FALLBACK);
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
//! \brief Create an engine for a network defintion
//!
//! \return Pointer to the engine created or nullptr if the creation failed
//!
bool networkToEngine(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder,
    BuildEnvironment& env, std::ostream& err)
{
    TrtUniquePtr<IBuilderConfig> config{builder.createBuilderConfig()};
    std::vector<std::vector<char>> sparseWeights;
    SMP_RETVAL_IF_FALSE(config != nullptr, "Config creation failed", false, err);
    SMP_RETVAL_IF_FALSE(setupNetworkAndConfig(build, sys, builder, *env.network, *config, err, sparseWeights),
        "Network And Config setup failed", false, err);

    std::unique_ptr<ITimingCache> timingCache{nullptr};
    // Try to load cache from file. Create a fresh cache if the file doesn't exist
    if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
    {
        std::vector<char> loadedCache = loadTimingCacheFile(build.timingCacheFile);
        timingCache.reset(config->createTimingCache(static_cast<const void*>(loadedCache.data()), loadedCache.size()));
        SMP_RETVAL_IF_FALSE(timingCache != nullptr, "TimingCache creation failed", false, err);
        config->setTimingCache(*timingCache, false);
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    SMP_RETVAL_IF_FALSE(profileStream != nullptr, "Cuda stream creation failed", false, err);
    config->setProfileStream(*profileStream);

    env.serializedEngine.reset(builder.buildSerializedNetwork(*env.network, *config));
    SMP_RETVAL_IF_FALSE(env.serializedEngine != nullptr, "Engine could not be created from network", false, err);

    if (build.safe)
    {
        ASSERT(sample::hasSafeRuntime());
        std::unique_ptr<safe::IRuntime> safeRuntime{sample::createSafeInferRuntime(sample::gLogger.getTRTLogger())};
        SMP_RETVAL_IF_FALSE(safeRuntime != nullptr, "SafeRuntime creation failed", false, err);
        safeRuntime->setErrorRecorder(&gRecorder);
        env.safeEngine.reset(
            safeRuntime->deserializeCudaEngine(env.serializedEngine->data(), env.serializedEngine->size()));
        if (build.consistency)
        {
            checkSafeEngine(env.serializedEngine->data(), env.serializedEngine->size());
        }
        SMP_RETVAL_IF_FALSE(env.safeEngine != nullptr, "SafeEngine deserialization failed", false, err);
    }
    else
    {
        TrtUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
        SMP_RETVAL_IF_FALSE(runtime != nullptr, "Runtime creation failed", false, err);
        runtime->setErrorRecorder(&gRecorder);
        env.engine.reset(runtime->deserializeCudaEngine(env.serializedEngine->data(), env.serializedEngine->size()));
        SMP_RETVAL_IF_FALSE(env.engine != nullptr, "Engine deserialization failed", false, err);
        if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
        {
            auto const& timingCache = config->getTimingCache();
            std::unique_ptr<IHostMemory> timingCacheHostData{timingCache->serialize()};
            SMP_RETVAL_IF_FALSE(timingCacheHostData != nullptr, "Timing Cache serialization failed", false, err);
            saveTimingCacheFile(build.timingCacheFile, timingCacheHostData.get());
        }
        if (config->getInt8Calibrator())
        {
            delete config->getInt8Calibrator();
        }
    }
    return true;
}

//!
//! \brief Parse a given model, create a network and an engine.
//!
bool modelToBuildEnv(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, BuildEnvironment& env, std::ostream& err)
{
    TrtUniquePtr<IBuilder> builder{createInferBuilder(sample::gLogger.getTRTLogger())};
    SMP_RETVAL_IF_FALSE(builder != nullptr, "Builder creation failed", false, err);
    builder->setErrorRecorder(&gRecorder);
    auto networkFlags
        = (build.maxBatch) ? 0U : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    env.network.reset(builder->createNetworkV2(networkFlags));
    SMP_RETVAL_IF_FALSE(env.network != nullptr, "Network creation failed", false, err);
    env.parser = modelToNetwork(model, *env.network, err);
    SMP_RETVAL_IF_FALSE(env.parser.operator bool(), "Parsing model failed", false, err);
    SMP_RETVAL_IF_FALSE(networkToEngine(build, sys, *builder, env, err), "Building engine failed", false, err);
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
    std::vector<const char*> layerNames(nbMissing);
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

bool loadEngineToEnv(const std::string& engine, int DLACore, bool safe, bool enableConsistency, BuildEnvironment& env, std::ostream& err)
{
    std::ifstream engineFile(engine, std::ios::binary);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error opening engine file: " << engine);
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error loading engine file: " << engine);
    // TODO: copy engine blob to env.serializedEngine

    if (safe)
    {
        ASSERT(sample::hasSafeRuntime());
        std::unique_ptr<safe::IRuntime> safeRuntime{sample::createSafeInferRuntime(sample::gLogger.getTRTLogger())};
        safeRuntime->setErrorRecorder(&gRecorder);
        env.safeEngine.reset(safeRuntime->deserializeCudaEngine(engineData.data(), fsize));
        bool result = env.safeEngine != nullptr;
        if (result && enableConsistency)
        {
            checkSafeEngine(engineData.data(), fsize);
        }
        return result;
    }

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (DLACore != -1)
    {
        runtime->setDLACore(DLACore);
    }
    runtime->setErrorRecorder(&gRecorder);
    env.engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    return env.engine != nullptr;
}
} // namespace

void dumpRefittable(nvinfer1::ICudaEngine& engine)
{
    TrtUniquePtr<IRefitter> refitter{createInferRefitter(engine, sample::gLogger.getTRTLogger())};
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

ICudaEngine* loadEngine(const std::string& engine, int DLACore, std::ostream& err)
{
    BuildEnvironment env;
    return loadEngineToEnv(engine, DLACore, false, false, env, err) ? env.engine.release() : nullptr;
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

bool getEngineBuildEnv(const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys,
    BuildEnvironment& env, std::ostream& err)
{
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<INetworkDefinition> network;
    Parser parser;

    bool createEngineSuccess {false};

    if (build.load)
    {
        createEngineSuccess = loadEngineToEnv(build.engine, sys.DLACore, build.safe, build.consistency, env, err);
    }
    else
    {
        createEngineSuccess = modelToBuildEnv(model, build, sys, env, err);
    }

    SMP_RETVAL_IF_FALSE(createEngineSuccess, "Failed to create engine from model.", false, err);

    if (build.save)
    {
        std::ofstream engineFile(build.engine, std::ios::binary);
        engineFile.write(static_cast<char*>(env.serializedEngine->data()), env.serializedEngine->size());
        SMP_RETVAL_IF_FALSE(!engineFile.fail(), "Saving engine to file failed.", false, err);
    }
    return true;
}

IHostMemory* networkToSerialized(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder,
    INetworkDefinition& network, std::ostream& err)
{
    TrtUniquePtr<IBuilderConfig> config{builder.createBuilderConfig()};
    std::vector<std::vector<char>> sparseWeights;
    SMP_RETVAL_IF_FALSE(config != nullptr, "Config creation failed", nullptr, err);
    SMP_RETVAL_IF_FALSE(setupNetworkAndConfig(build, sys, builder, network, *config, err, sparseWeights),
        "Network And Config setup failed", nullptr, err);
    return builder.buildSerializedNetwork(network, *config);
}

IHostMemory* modelToSerialized(
    const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err)
{
    TrtUniquePtr<IBuilder> builder{createInferBuilder(sample::gLogger.getTRTLogger())};
    SMP_RETVAL_IF_FALSE(builder != nullptr, "Builder creation failed", nullptr, err);
    builder->setErrorRecorder(&gRecorder);

    auto networkFlags
        = (build.maxBatch) ? 0U : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    TrtUniquePtr<INetworkDefinition> network{builder->createNetworkV2(networkFlags)};
    SMP_RETVAL_IF_FALSE(network != nullptr, "Network creation failed", nullptr, err);

    Parser parser = modelToNetwork(model, *network, err);
    SMP_RETVAL_IF_FALSE(parser.operator bool(), "Parsing model failed", nullptr, err);

    return networkToSerialized(build, sys, *builder, *network, err);
}

bool serializeAndSave(const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys, std::ostream& err)
{
    TrtUniquePtr<IHostMemory> serialized{modelToSerialized(model, build, sys, err)};
    SMP_RETVAL_IF_FALSE(serialized != nullptr, "Network serialization failed", false, err);

    std::ofstream engineFile(build.engine, std::ios::binary);
    SMP_RETVAL_IF_FALSE(!!engineFile, "Cannot open a file to save a serialize network", false, err);
    engineFile.write(static_cast<char*>(serialized->data()), serialized->size());
    return !engineFile.fail();
}

// There is not a getWeightsName API, so we need to use WeightsRole.
std::vector<std::pair<WeightsRole, Weights>> getAllRefitWeightsForLayer(const ILayer& l)
{
    switch (l.getType())
    {
    case LayerType::kCONSTANT:
    {
        const auto& layer = static_cast<const nvinfer1::IConstantLayer&>(l);
        return {std::make_pair(WeightsRole::kCONSTANT, layer.getWeights())};
    }
    case LayerType::kCONVOLUTION:
    {
        const auto& layer = static_cast<const nvinfer1::IConvolutionLayer&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kDECONVOLUTION:
    {
        const auto& layer = static_cast<const nvinfer1::IDeconvolutionLayer&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kFULLY_CONNECTED:
    {
        const auto& layer = static_cast<const nvinfer1::IFullyConnectedLayer&>(l);
        return {std::make_pair(WeightsRole::kKERNEL, layer.getKernelWeights()),
            std::make_pair(WeightsRole::kBIAS, layer.getBiasWeights())};
    }
    case LayerType::kSCALE:
    {
        const auto& layer = static_cast<const nvinfer1::IScaleLayer&>(l);
        return {std::make_pair(WeightsRole::kSCALE, layer.getScale()),
            std::make_pair(WeightsRole::kSHIFT, layer.getShift())};
    }
    case LayerType::kRNN_V2:
    case LayerType::kACTIVATION:
    case LayerType::kPOOLING:
    case LayerType::kLRN:
    case LayerType::kSOFTMAX:
    case LayerType::kSHUFFLE:
    case LayerType::kCONCATENATION:
    case LayerType::kELEMENTWISE:
    case LayerType::kPLUGIN:
    case LayerType::kUNARY:
    case LayerType::kPADDING:
    case LayerType::kREDUCE:
    case LayerType::kTOPK:
    case LayerType::kGATHER:
    case LayerType::kMATRIX_MULTIPLY:
    case LayerType::kRAGGED_SOFTMAX:
    case LayerType::kIDENTITY:
    case LayerType::kPLUGIN_V2:
    case LayerType::kSLICE:
    case LayerType::kFILL:
    case LayerType::kSHAPE:
    case LayerType::kPARAMETRIC_RELU:
    case LayerType::kRESIZE:
    case LayerType::kTRIP_LIMIT:
    case LayerType::kRECURRENCE:
    case LayerType::kITERATOR:
    case LayerType::kLOOP_OUTPUT:
    case LayerType::kSELECT:
    case LayerType::kQUANTIZE:
    case LayerType::kDEQUANTIZE:
    case LayerType::kCONDITION:
    case LayerType::kCONDITIONAL_INPUT:
    case LayerType::kCONDITIONAL_OUTPUT:
    case LayerType::kSCATTER:
    case LayerType::kEINSUM:
    case LayerType::kASSERTION: return {};
    }
    return {};
}

bool timeRefit(INetworkDefinition const& network, nvinfer1::ICudaEngine& engine)
{
    using time_point = std::chrono::time_point<std::chrono::steady_clock>;
    using durationMs = std::chrono::duration<float, std::milli>;

    auto const nbLayers = network.getNbLayers();
    TrtUniquePtr<IRefitter> refitter{createInferRefitter(engine, sample::gLogger.getTRTLogger())};
    auto const& layerWeightsRolePair = getLayerWeightsRolePair(*refitter);
    // We use std::string instead of const char* since we can have copies of layer names.
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

    constexpr int32_t loop = 10;
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
