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

//!
//! sampleMNISTAPI.cpp
//! This file contains the implementation of the MNIST API sample. It creates the network
//! for MNIST classification using the API.
//! It can be run with the following command line:
//! Command: ./sample_mnist_api [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_mnist_api";

//!
//! \brief The SampleMNISTAPIParams structure groups the additional parameters required by
//!         the SampleMNISTAPI sample.
//!
struct SampleMNISTAPIParams : public samplesCommon::SampleParams
{
    int inputH;                  //!< The input height
    int inputW;                  //!< The input width
    int outputSize;              //!< The output size
    std::string weightsFile;     //!< The filename of the weights file
    std::string mnistMeansProto; //!< The proto file containing means
};

//! \brief  The SampleMNISTAPI class implements the MNIST API sample
//!
//! \details It creates the network for MNIST classification using the API
//!
class SampleMNISTAPI
{
public:
    SampleMNISTAPI(const SampleMNISTAPIParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleMNISTAPIParams mParams; //!< The parameters for the sample.

    int mNumber{0}; //!< The number to classify

    std::map<std::string, nvinfer1::Weights> mWeightMap; //!< The weight name to weight value map

    std::vector<std::unique_ptr<samplesCommon::HostMemory>> weightsMemory; //!< Host weights memory holder

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Uses the API to create the MNIST Network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Loads weights from weights file
    //!
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MNIST network by using the API to create a model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleMNISTAPI::build()
{
    mWeightMap = loadWeights(locateFile(mParams.weightsFile, mParams.dataDirs));

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatchFlag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatchFlag));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    auto inputDims = network->getInput(0)->getDimensions();
    ASSERT(inputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    auto outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 4);

    return true;
}

//!
//! \brief Uses the API to create the MNIST Network
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleMNISTAPI::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    // Create input tensor of shape { 1, 1, 28, 28 }
    ITensor* data = network->addInput(
        mParams.inputTensorNames[0].c_str(), DataType::kFLOAT, Dims4{1, 1, mParams.inputH, mParams.inputW});
    ASSERT(data);

    // Create scale layer with default power/shift and specified scale parameter.
    const float scaleParam = 0.0125f;
    const Weights power{DataType::kFLOAT, nullptr, 0};
    const Weights shift{DataType::kFLOAT, nullptr, 0};
    const Weights scale{DataType::kFLOAT, &scaleParam, 1};
    IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
    ASSERT(scale_1);

    // Add convolution layer with 20 outputs and a 5x5 filter.
    IConvolutionLayer* conv1 = network->addConvolutionNd(
        *scale_1->getOutput(0), 20, Dims{2, {5, 5}}, mWeightMap["conv1filter"], mWeightMap["conv1bias"]);
    ASSERT(conv1);
    conv1->setStride(DimsHW{1, 1});

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kMAX, Dims{2, {2, 2}});
    ASSERT(pool1);
    pool1->setStride(DimsHW{2, 2});

    // Add second convolution layer with 50 outputs and a 5x5 filter.
    IConvolutionLayer* conv2 = network->addConvolutionNd(
        *pool1->getOutput(0), 50, Dims{2, {5, 5}}, mWeightMap["conv2filter"], mWeightMap["conv2bias"]);
    ASSERT(conv2);
    conv2->setStride(DimsHW{1, 1});

    // Add second max pooling layer with stride of 2x2 and kernel size of 2x3>
    IPoolingLayer* pool2 = network->addPoolingNd(*conv2->getOutput(0), PoolingType::kMAX, Dims{2, {2, 2}});
    ASSERT(pool2);
    pool2->setStride(DimsHW{2, 2});

    // Utility for use MatMul as FC
    auto addMatMulasFCLayer
        = [&network](ITensor* input, int32_t const outputs, Weights& filterWeights, Weights& biasWeights) -> ILayer* {
        Dims inputDims = input->getDimensions();
        int32_t const m = inputDims.d[0];
        int32_t const k
            = std::accumulate(inputDims.d + 1, inputDims.d + inputDims.nbDims, 1, std::multiplies<int32_t>());
        int32_t const n = static_cast<int32_t>(filterWeights.count / static_cast<int64_t>(k));
        ASSERT(static_cast<int64_t>(n) * static_cast<int64_t>(k) == filterWeights.count);
        ASSERT(static_cast<int64_t>(n) == biasWeights.count);
        ASSERT(n == outputs);

        IShuffleLayer* inputReshape = network->addShuffle(*input);
        ASSERT(inputReshape);
        inputReshape->setReshapeDimensions(Dims{2, {m, k}});

        IConstantLayer* filterConst = network->addConstant(Dims{2, {n, k}}, filterWeights);
        ASSERT(filterConst);
        IMatrixMultiplyLayer* mm = network->addMatrixMultiply(*inputReshape->getOutput(0), MatrixOperation::kNONE,
            *filterConst->getOutput(0), MatrixOperation::kTRANSPOSE);
        ASSERT(mm);

        IConstantLayer* biasConst = network->addConstant(Dims{2, {1, n}}, biasWeights);
        ASSERT(biasConst);
        IElementWiseLayer* biasAdd
            = network->addElementWise(*mm->getOutput(0), *biasConst->getOutput(0), ElementWiseOperation::kSUM);
        ASSERT(biasAdd);

        IShuffleLayer* outputReshape = network->addShuffle(*biasAdd->getOutput(0));
        ASSERT(outputReshape);
        outputReshape->setReshapeDimensions(Dims{4, {m, n, 1, 1}});

        return outputReshape;
    };

    // Add fully connected layer with 500 outputs.
    ILayer* ip1 = addMatMulasFCLayer(pool2->getOutput(0), 500, mWeightMap["ip1filter"], mWeightMap["ip1bias"]);
    ASSERT(ip1);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
    ASSERT(relu1);

    // Add second fully connected layer with 20 outputs.
    ILayer* ip2
        = addMatMulasFCLayer(relu1->getOutput(0), mParams.outputSize, mWeightMap["ip2filter"], mWeightMap["ip2bias"]);
    ASSERT(ip2);

    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
    ASSERT(prob);
    prob->getOutput(0)->setName(mParams.outputTensorNames[0].c_str());
    network->markOutput(*prob->getOutput(0));

    // Build engine
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 64.0f, 64.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleMNISTAPI::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleMNISTAPI::processInput(const samplesCommon::BufferManager& buffers)
{
    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(mParams.inputH * mParams.inputW);
    mNumber = rand() % mParams.outputSize;
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), mParams.inputH,
        mParams.inputW);

    // Print ASCII representation of digit image
    std::cout << "\nInput:\n" << std::endl;
    for (int i = 0; i < mParams.inputH * mParams.inputW; i++)
    {
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % mParams.inputW) ? "" : "\n");
    }

    // Parse mean file
    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
    {
        return false;
    }

    auto meanBlob = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(
        parser->parseBinaryProto(locateFile(mParams.mnistMeansProto, mParams.dataDirs).c_str()));
    if (!meanBlob)
    {
        return false;
    }

    const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());
    if (!meanData)
    {
        return false;
    }

    // Subtract mean from image
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < mParams.inputH * mParams.inputW; i++)
    {
        hostDataBuffer[i] = float(fileData[i]) - meanData[i];
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleMNISTAPI::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float* prob = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    std::cout << "\nOutput:\n" << std::endl;
    float maxVal{0.0f};
    int idx{0};
    for (int i = 0; i < mParams.outputSize; i++)
    {
        if (maxVal < prob[i])
        {
            maxVal = prob[i];
            idx = i;
        }
        std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << std::endl;
    }
    std::cout << std::endl;

    return idx == mNumber && maxVal > 0.9f;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleMNISTAPI::teardown()
{
    return true;
}

//!
//! \brief Loads weights from weights file
//!
//! \details TensorRT weight files have a simple space delimited format
//!          [type] [size] <data x size in hex>
//!
std::map<std::string, nvinfer1::Weights> SampleMNISTAPI::loadWeights(const std::string& file)
{
    sample::gLogInfo << "Loading weights: " << file << std::endl;

    // Open weights file
    std::ifstream input(file, std::ios::binary);
    ASSERT(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    ASSERT(count > 0 && "Invalid weight map file.");

    std::map<std::string, nvinfer1::Weights> weightMap;
    while (count--)
    {
        nvinfer1::Weights wt{DataType::kFLOAT, nullptr, 0};
        int type;
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);

        // Load blob
        if (wt.type == DataType::kFLOAT)
        {
            // Use uint32_t to create host memory to avoid additional conversion.
            auto mem = new samplesCommon::TypedHostMemory<uint32_t, nvinfer1::DataType::kFLOAT>(size);
            weightsMemory.emplace_back(mem);
            uint32_t* val = mem->raw();
            for (uint32_t x = 0; x < size; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        else if (wt.type == DataType::kHALF)
        {
            // HalfMemory's raw type is uint16_t
            auto mem = new samplesCommon::HalfMemory(size);
            weightsMemory.emplace_back(mem);
            auto val = mem->raw();
            for (uint32_t x = 0; x < size; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleMNISTAPIParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleMNISTAPIParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.inputTensorNames.push_back("data");
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.inputH = 28;
    params.inputW = 28;
    params.outputSize = 10;
    params.weightsFile = "mnistapi.wts";
    params.mnistMeansProto = "mnist_mean.binaryproto";

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_mnist_api [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleMNISTAPI sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for MNIST API" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
