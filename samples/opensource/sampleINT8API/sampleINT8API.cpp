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

//! sampleINT8API.cpp
//! This file contains implementation showcasing usage of INT8 calibration and precision APIs.
//! It creates classification networks such as mobilenet, vgg19, resnet-50 from onnx model file.
//! This sample showcae setting per tensor dynamic range overriding calibrator generated scales if it exists.
//! This sample showcase how to set computation precision of layer. It involves forcing output tensor type of the layer
//! to particular precision. It can be run with the following command line: Command: ./sample_int8_api [-h or --help]
//! [-m modelfile] [-s per_tensor_dynamic_range_file] [-i image_file] [-r reference_file] [-d path/to/data/dir]
//! [--verbose] [-useDLA <id>]

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

const std::string gSampleName = "TensorRT.sample_int8_api";

struct SampleINT8APIPreprocessing
{
    // Preprocessing values are available here: https://github.com/onnx/models/tree/master/models/image_classification/resnet
    std::vector<int> inputDims{1,3,224,224};
};

//!
//! \brief The SampleINT8APIParams structure groups the additional parameters required by
//!         the INT8 API sample
//!
struct SampleINT8APIParams
{
    bool verbose{false};
    bool writeNetworkTensors{false};
    int dlaCore{-1};
    int batchSize{1};

    SampleINT8APIPreprocessing mPreproc;
    std::string modelFileName;
    std::vector<std::string> dataDirs;
    std::string dynamicRangeFileName;
    std::string imageFileName;
    std::string referenceFileName;
    std::string networkTensorsFileName;
};

//!
//! \brief The SampleINT8API class implements INT8 inference on classification networks.
//!
//! \details INT8 API usage for setting custom int8 range for each input layer. API showcase how
//!           to perform INT8 inference without calibration table
//!
class SampleINT8API
{
private:
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleINT8API(const SampleINT8APIParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    Logger::TestResult build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    Logger::TestResult infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    Logger::TestResult teardown();

    SampleINT8APIParams mParams; //!< Stores Sample Parameter

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

    std::map<std::string, std::string> mInOut; //!< Input and output mapping of the network

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network

    std::unordered_map<std::string, float>
        mPerTensorDynamicRangeMap; //!< Mapping from tensor name to max absolute dynamic range values

    void getInputOutputNames(); //!< Populates input and output mapping of the network

    //!
    //! \brief Reads the ppm input image, preprocesses, and stores the result in a managed buffer
    //!
    bool prepareInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers) const;

    //!
    //! \brief Populate per tensor dynamic range values
    //!
    bool readPerTensorDynamicRangeValues();

    //!
    //! \brief  Sets custom dynamic range for network tensors
    //!
    bool setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief  Sets computation precision for network layers
    //!
    void setLayerPrecision(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief  Write network tensor names to a file.
    //!
    void writeNetworkTensorNames(const SampleUniquePtr<nvinfer1::INetworkDefinition>& network);
};

//!
//! \brief  Populates input and output mapping of the network
//!
void SampleINT8API::getInputOutputNames()
{
    int nbindings = mEngine.get()->getNbBindings();
    assert(nbindings == 2);
    for (int b = 0; b < nbindings; ++b)
    {
        nvinfer1::Dims dims = mEngine.get()->getBindingDimensions(b);
        if (mEngine.get()->bindingIsInput(b))
        {
            if (mParams.verbose)
            {
                gLogInfo << "Found input: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            mInOut["input"] = mEngine.get()->getBindingName(b);
        }
        else
        {
            if (mParams.verbose)
            {
                gLogInfo << "Found output: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            mInOut["output"] = mEngine.get()->getBindingName(b);
        }
    }
}

//!
//! \brief Populate per tensor dyanamic range values
//!
bool SampleINT8API::readPerTensorDynamicRangeValues()
{
    std::ifstream iDynamicRangeStream(mParams.dynamicRangeFileName);
    if (!iDynamicRangeStream)
    {
        gLogError << "Could not find per tensor scales file: " << mParams.dynamicRangeFileName << std::endl;
        return false;
    }

    std::string line;
    char delim = ':';
    while (std::getline(iDynamicRangeStream, line))
    {
        std::istringstream iline(line);
        std::string token;
        std::getline(iline, token, delim);
        std::string tensorName = token;
        std::getline(iline, token, delim);
        float dynamicRange = std::stof(token);
        mPerTensorDynamicRangeMap[tensorName] = dynamicRange;
    }
    return true;
}

//!
//! \brief  Sets computation precision for network layers
//!
void SampleINT8API::setLayerPrecision(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    gLogInfo << "Setting Per Layer Computation Precision" << std::endl;
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto layer = network->getLayer(i);
        if (mParams.verbose)
        {
            std::string layerName = layer->getName();
            gLogInfo << "Layer: " << layerName << ". Precision: INT8" << std::endl;
        }

        // Don't set the precision on non-computation layers as they don't support
        // int8.
        if (layer->getType() != LayerType::kCONSTANT
                && layer->getType() != LayerType::kCONCATENATION
                && layer->getType() != LayerType::kSHAPE)
        {
            // set computation precision of the layer
            layer->setPrecision(nvinfer1::DataType::kINT8);
        }

        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            if (mParams.verbose)
            {
                std::string tensorName = layer->getOutput(j)->getName();
                gLogInfo << "Tensor: " << tensorName << ". OutputType: INT8" << std::endl;
            }
            // set output type of execution tensors and not shape tensors.
            if (layer->getOutput(j)->isExecutionTensor())
            {
                layer->setOutputType(j, nvinfer1::DataType::kINT8);
            }
        }
    }
}

//!
//! \brief  Write network tensor names to a file.
//!
void SampleINT8API::writeNetworkTensorNames(const SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    gLogInfo << "Sample requires to run with per tensor dynamic range." << std::endl;
    gLogInfo << "In order to run Int8 inference without calibration, user will need to provide dynamic range for all "
                "the network tensors."
             << std::endl;

    std::ofstream tensorsFile{mParams.networkTensorsFileName};

    // Iterate through network inputs to write names of input tensors.
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        string tName = network->getInput(i)->getName();
        tensorsFile << "TensorName: " << tName << std::endl;
        if (mParams.verbose)
        {
            gLogInfo << "TensorName: " << tName << std::endl;
        }
    }

    // Iterate through network layers.
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        // Write output tensors of a layer to the file.
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
        {
            string tName = network->getLayer(i)->getOutput(j)->getName();
            tensorsFile << "TensorName: " << tName << std::endl;
            if (mParams.verbose)
            {
                gLogInfo << "TensorName: " << tName << std::endl;
            }
        }
    }
    tensorsFile.close();
    gLogInfo << "Successfully generated network tensor names. Writing: " << mParams.networkTensorsFileName << std::endl;
    gLogInfo << "Use the generated tensor names file to create dynamic range file for Int8 inference. Follow README.md "
                "for instructions to generate dynamic_ranges.txt file."
             << std::endl;
}

//!
//! \brief  Sets custom dynamic range for network tensors
//!
bool SampleINT8API::setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // populate per tensor dynamic range
    if (!readPerTensorDynamicRangeValues())
    {
        return false;
    }

    gLogInfo << "Setting Per Tensor Dynamic Range" << std::endl;
    if (mParams.verbose)
    {
        gLogInfo << "If dynamic range for a tensor is missing, TensorRT will run inference assuming dynamic range for "
                    "the tensor as optional."
                 << std::endl;
        gLogInfo << "If dynamic range for a tensor is required then inference will fail. Follow README.md to generate "
                    "missing per tensor dynamic range."
                 << std::endl;
    }
    // set dynamic range for network input tensors
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        string tName = network->getInput(i)->getName();
        if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
        {
            network->getInput(i)->setDynamicRange(-mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
        }
        else
        {
            if (mParams.verbose)
            {
                gLogWarning << "Missing dynamic range for tensor: " << tName << std::endl;
            }
        }
    }

     // set dynamic range for layer output tensors
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto lyr = network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j)
        {
            string tName = lyr->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
            {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                lyr->getOutput(j)->setDynamicRange(
                    -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
            }
            else if (lyr->getType() == LayerType::kCONSTANT)
            {
                IConstantLayer* cLyr = static_cast<IConstantLayer*>(lyr);
                if (mParams.verbose)
                {
                    gLogWarning << "Computing missing dynamic range for tensor, " << tName << ", from weights."
                                << std::endl;
                }
                auto wts = cLyr->getWeights();
                double max = std::numeric_limits<double>::min();
                for (int64_t wb = 0, we = wts.count; wb < we; ++wb)
                {
                    double val;
                    switch (wts.type)
                    {
                    case DataType::kFLOAT: val = static_cast<const float*>(wts.values)[wb]; break;
                    case DataType::kBOOL: val = static_cast<const bool*>(wts.values)[wb]; break;
                    case DataType::kINT8: val = static_cast<const int8_t*>(wts.values)[wb]; break;
                    case DataType::kHALF: val = static_cast<const half_float::half*>(wts.values)[wb]; break;
                    case DataType::kINT32: val = static_cast<const int32_t*>(wts.values)[wb]; break;
                    }
                    max = std::max(max, std::abs(val));
                }

                lyr->getOutput(j)->setDynamicRange(-max, max);
            }
            else
            {
                if (mParams.verbose)
                {
                    gLogWarning << "Missing dynamic range for tensor: " << tName << std::endl;
                }
            }
        }
    }

    // set dynamic range for layer output tensors
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
        {
            string tName = network->getLayer(i)->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
            {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                network->getLayer(i)->getOutput(j)->setDynamicRange(
                    -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
            }
            else
            {
                if (mParams.verbose)
                {
                    gLogWarning << "Missing dynamic range for tensor: " << tName << std::endl;
                }
            }
        }
    }

    if (mParams.verbose)
    {
        gLogInfo << "Per Tensor Dynamic Range Values for the Network:" << std::endl;
        for (auto iter = mPerTensorDynamicRangeMap.begin(); iter != mPerTensorDynamicRangeMap.end(); ++iter)
            gLogInfo << "Tensor: " << iter->first << ". Max Absolute Dynamic Range: " << iter->second << std::endl;
    }
    return true;
}

//!
//! \brief Preprocess inputs and allocate host/device input buffers
//!
bool SampleINT8API::prepareInput(const samplesCommon::BufferManager& buffers)
{
    if (samplesCommon::toLower(samplesCommon::getFileType(mParams.imageFileName)).compare("ppm") != 0)
    {
        gLogError << "Wrong format: " << mParams.imageFileName << " is not a ppm file." << std::endl;
        return false;
    }

    int channels = mParams.mPreproc.inputDims.at(1);
    int height = mParams.mPreproc.inputDims.at(2);
    int width = mParams.mPreproc.inputDims.at(3);
    int max{0};
    std::string magic{""};

    vector<uint8_t> fileData(channels * height * width);
    // Prepardde PPM Buffer to read the input image
    // samplesCommon::PPM<channels, height, width> ppm;
    // samplesCommon::readPPMFile(mParams.imageFileName, ppm);

    std::ifstream infile(mParams.imageFileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(fileData.data()), width * height * channels);

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(mInOut["input"]));

    // Convert HWC to CHW and Normalize
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = c * height * width + h * width + w;
                int srcIdx = h * width * channels + w * channels + c;
                // This equation include 3 steps
                // 1. Scale Image to range [0.f, 1.0f]
                // 2. Normalize Image using per channel Mean and per channel Standard Deviation
                // 3. Shuffle HWC to CHW form
                hostInputBuffer[dstIdx] = (2.0 / 255.0) * static_cast<const float>(fileData[srcIdx]) - 1.0;
            }
        }
    }
    return true;
}

//!
//! \brief Verifies that the output is correct and prints it
//!
bool SampleINT8API::verifyOutput(const samplesCommon::BufferManager& buffers) const
{
    // copy output host buffer data for further processing
    const float* probPtr = static_cast<const float*>(buffers.getHostBuffer(mInOut.at("output")));
    vector<float> output(probPtr, probPtr + mOutputDims.d[1] * mParams.batchSize);

    auto inds = samplesCommon::argsort(output.cbegin(), output.cend(), true);

    // read reference lables to generate prediction lables
    vector<string> referenceVector;
    if (!samplesCommon::readReferenceFile(mParams.referenceFileName, referenceVector))
    {
        gLogError << "Unable to read reference file: " << mParams.referenceFileName << std::endl;
        return false;
    }

    vector<string> top5Result = samplesCommon::classify(referenceVector, output, 5);

    gLogInfo << "SampleINT8API result: Detected:" << std::endl;
    for (int i = 1; i <= 5; ++i)
    {
        gLogInfo << "[" << i << "]  " << top5Result[i - 1] << std::endl;
    }

    return true;
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates INT8 classification network by parsing the onnx model and builds
//!          the engine that will be used to run INT8 inference (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
Logger::TestResult SampleINT8API::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        gLogError << "Unable to create builder object." << std::endl;
        return Logger::TestResult::kFAILED;
    }
    
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        gLogError << "Unable to create network object." << mParams.referenceFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        gLogError << "Unable to create config object." << mParams.referenceFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        gLogError << "Unable to create parser object." << mParams.referenceFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    // Parse ONNX model file to populate TensorRT INetwork
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    if (!parser->parseFromFile(mParams.modelFileName.c_str(), verbosity))
    {
        gLogError << "Unable to parse ONNX model file: " << mParams.modelFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    if (mParams.writeNetworkTensors)
    {
        writeNetworkTensorNames(network);
        return Logger::TestResult::kWAIVED;
    }

    if (!builder->platformHasFastInt8())
    {
        gLogError << "Platform does not support INT8 inference. sampleINT8API can only run in INT8 Mode." << std::endl;
        return Logger::TestResult::kWAIVED;
    }

    // Configure buider
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setMaxWorkspaceSize(1_GiB);

    // Enable INT8 model. Required to set custom per tensor dynamic range or INT8 Calibration
    config->setFlag(BuilderFlag::kINT8);
    // Mark calibrator as null. As user provides dynamic range for each tensor, no calibrator is required
    config->setInt8Calibrator(nullptr);

    auto maxBatchSize = mParams.batchSize;
    if (mParams.dlaCore >= 0)
    {
        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
        if (maxBatchSize > builder->getMaxDLABatchSize())
        {
            std::cerr << "Requested batch size " << maxBatchSize << " is greater than the max DLA batch size of "
                      << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
            maxBatchSize = builder->getMaxDLABatchSize();
        }
    }
    builder->setMaxBatchSize(maxBatchSize);

    // force layer to execute with required precision
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    setLayerPrecision(network);

    // set INT8 Per Tensor Dynamic range
    if (!setDynamicRange(network))
    {
        gLogError << "Unable to set per tensor dynamic range." << std::endl;
        return Logger::TestResult::kFAILED;
    }

    // build TRT engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        gLogError << "Unable to build cuda engine." << std::endl;
        return Logger::TestResult::kFAILED;
    }

    // populates input output map structure
    getInputOutputNames();

    // derive input/output dims from engine bindings
    const int inputIndex = mEngine.get()->getBindingIndex(mInOut["input"].c_str());
    mInputDims = mEngine.get()->getBindingDimensions(inputIndex);

    const int outputIndex = mEngine.get()->getBindingIndex(mInOut["output"].c_str());
    mOutputDims = mEngine.get()->getBindingDimensions(outputIndex);

    return Logger::TestResult::kRUNNING;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output
//!
Logger::TestResult SampleINT8API::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return Logger::TestResult::kFAILED;
    }

    // Read the input data into the managed buffers
    // There should be just 1 input tensor

    if (!prepareInput(buffers))
    {
        return Logger::TestResult::kFAILED;
    }

    // Create CUDA stream for the execution of this inference
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueueV2(buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return Logger::TestResult::kFAILED;
    }

    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    // Release stream
    cudaStreamDestroy(stream);

    // Check and print the output of the inference
    return verifyOutput(buffers) ? Logger::TestResult::kRUNNING : Logger::TestResult::kFAILED;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
Logger::TestResult SampleINT8API::teardown()
{
    return Logger::TestResult::kRUNNING;
}

//!
//! \brief The SampleINT8APIArgs structures groups the additional arguments required by
//!         the INT8 API sample
//!
struct SampleINT8APIArgs : public samplesCommon::Args
{
    bool verbose{false};
    bool writeNetworkTensors{false};
    std::string modelFileName{"resnet50.onnx"};
    std::string imageFileName{"airliner.ppm"};
    std::string referenceFileName{"reference_labels.txt"};
    std::string dynamicRangeFileName{"resnet50_per_tensor_dynamic_range.txt"};
    std::string networkTensorsFileName{"network_tensors.txt"};
};

//! \brief This function parses arguments specific to SampleINT8API
//!
bool parseSampleINT8APIArgs(SampleINT8APIArgs& args, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--model=", 8))
        {
            args.modelFileName = (argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--image=", 8))
        {
            args.imageFileName = (argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--reference=", 12))
        {
            args.referenceFileName = (argv[i] + 12);
        }
        else if (!strncmp(argv[i], "--write_tensors", 15))
        {
            args.writeNetworkTensors = true;
        }
        else if (!strncmp(argv[i], "--network_tensors_file=", 23))
        {
            args.networkTensorsFileName = (argv[i] + 23);
        }
        else if (!strncmp(argv[i], "--ranges=", 9))
        {
            args.dynamicRangeFileName = (argv[i] + 9);
        }
        else if (!strncmp(argv[i], "--int8", 6))
        {
            args.runInInt8 = true;
        }
        else if (!strncmp(argv[i], "--fp16", 6))
        {
            args.runInFp16 = true;
        }
        else if (!strncmp(argv[i], "--useDLACore=", 13))
        {
            args.useDLACore = std::stoi(argv[i] + 13);
        }
        else if (!strncmp(argv[i], "--data=", 7))
        {
            std::string dirPath = (argv[i] + 7);
            if (dirPath.back() != '/')
            {
                dirPath.push_back('/');
            }
            args.dataDirs.push_back(dirPath);
        }
        else if (!strncmp(argv[i], "--verbose", 9) || !strncmp(argv[i], "-v", 2))
        {
            args.verbose = true;
        }
        else if (!strncmp(argv[i], "--help", 6) || !strncmp(argv[i], "-h", 2))
        {
            args.help = true;
        }
        else
        {
            gLogError << "Invalid Argument: " << argv[i] << std::endl;
            return false;
        }
    }
    return true;
}

void validateInputParams(SampleINT8APIParams& params)
{
    gLogInfo << "Please follow README.md to generate missing input files." << std::endl;
    gLogInfo << "Validating input parameters. Using following input files for inference." << std::endl;
    params.modelFileName = locateFile(params.modelFileName, params.dataDirs);
    gLogInfo << "    Model File: " << params.modelFileName << std::endl;
    if (params.writeNetworkTensors)
    {
        gLogInfo << "    Writing Network Tensors File to: " << params.networkTensorsFileName << std::endl;
        return;
    }
    params.imageFileName = locateFile(params.imageFileName, params.dataDirs);
    gLogInfo << "    Image File: " << params.imageFileName << std::endl;
    params.referenceFileName = locateFile(params.referenceFileName, params.dataDirs);
    gLogInfo << "    Reference File: " << params.referenceFileName << std::endl;
    params.dynamicRangeFileName = locateFile(params.dynamicRangeFileName, params.dataDirs);
    gLogInfo << "    Dynamic Range File: " << params.dynamicRangeFileName << std::endl;
    return;
}

//!
//! \brief This function initializes members of the params struct using the command line args
//!
SampleINT8APIParams initializeSampleParams(SampleINT8APIArgs args)
{
    SampleINT8APIParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/samples/int8_api/");
        params.dataDirs.push_back("data/int8_api/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.dataDirs.push_back(""); //! In case of absolute path search
    params.batchSize = 1;
    params.verbose = args.verbose;
    params.modelFileName = args.modelFileName;
    params.imageFileName = args.imageFileName;
    params.referenceFileName = args.referenceFileName;
    params.dynamicRangeFileName = args.dynamicRangeFileName;
    params.dlaCore = args.useDLACore;
    params.writeNetworkTensors = args.writeNetworkTensors;
    params.networkTensorsFileName = args.networkTensorsFileName;
    validateInputParams(params);
    return params;
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_int8_api [-h or --help] [--model=<model file>] "
                 "[--ranges=<dynamic range file>] [--image=<image file>] [--reference=<reference file>] "
                 "[--data=<path to data directory>] [--useDLACore=<int>] [-v or --verbose]\n";
    std::cout << "--help, -h                        Display This help information" << std::endl;
    std::cout << "--model=<model file>              Path to the model file (default = resnet50.onnx)"
              << std::endl;
    std::cout << "--image=<image file>              Path to the image file to infer (default = airlines.ppm)"
              << std::endl;
    std::cout << "--reference=<reference file>      Path to the reference labels file (default = reference_labels.txt)"
              << std::endl;
    std::cout << "--ranges=<dynamic range file>     Path to the custom per tensor dynamic range file for the network "
                 "(default = resnet50_per_tensor_dynamic_range.txt)"
              << std::endl;
    std::cout << "--write_tensors                   Option to generate file containing network tensors name. By default writes to "
                 "network_tensors.txt file. To provide user defined file name use additional option "
                 "--network_tensors_file. See --network_tensors_file option usage for more detail."
              << std::endl;
    std::cout << "--network_tensors_file=<file>     Path to the network tensors file. This option "
                 "needs to be used with --write_tensors option. Specify file name (will write to current execution "
                 "directory) or absolute path to file name to write network tensor names file. Dynamic range "
                 "corresponding to each network tensor is required to run the sample (default = network_tensors.txt)"
              << std::endl;
    std::cout << "--data=<path to data dir>         Path to the data directory to search for above files in case absolute paths to "
                 "files are not provided (default both data/samples/int8_api/ and data/int8_api/)"
              << std::endl;
    std::cout << "--useDLACore=N                    Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--verbose                         Enable verbose prints" << std::endl;
}

int main(int argc, char** argv)
{
    SampleINT8APIArgs args;
    bool argsOK = parseSampleINT8APIArgs(args, argc, argv);

    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (args.verbose)
    {
        gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleINT8APIParams params;
    params = initializeSampleParams(args);

    SampleINT8API sample(params);
    gLogInfo << "Building and running a INT8 GPU inference engine for " << params.modelFileName << std::endl;

    auto buildStatus = sample.build();
    if (buildStatus == Logger::TestResult::kWAIVED)
    {
        return gLogger.reportWaive(sampleTest);
    }
    else if (buildStatus == Logger::TestResult::kFAILED)
    {
        return gLogger.reportFail(sampleTest);
    }

    if (sample.infer() != Logger::TestResult::kRUNNING)
    {
        return gLogger.reportFail(sampleTest);
    }

    if (sample.teardown() != Logger::TestResult::kRUNNING)
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
