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

//!
//! SampleMLP.cpp
//! This file contains the implementation of the MLP sample. It creates the network
//! for MNIST classification using the API.
//! It can be run with the following command line:
//! Command: ./sample_mlp [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir] [--useDLACore=<int>]
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

const std::string gSampleName = "TensorRT.sample_mlp";

//!
//! \brief The SampleMLPParams structure groups the additional parameters required by
//!         the MLP sample.
//!
struct SampleMLPParams : public samplesCommon::SampleParams
{
    int inputH;              //!< The input height
    int inputW;              //!< The input width
    int outputSize;          //!< The output size
    std::string weightsFile; //!< The filename of the weights file
};

//! \brief  The SampleMLP class implements the MNIST API sample
//!
//! \details It creates the network for MNIST classification using the API
//!
class SampleMLP
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMLP(const SampleMLPParams& params)
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
    SampleMLPParams mParams; //!< The parameters for the sample.

    int mNumber{0}; //!< The number to classify

    std::map<std::string, std::pair<nvinfer1::Dims, nvinfer1::Weights>>
        mWeightMap; //!< The weight name to weight value map

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Uses the API to create the MLP Network
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
    std::map<std::string, std::pair<nvinfer1::Dims, nvinfer1::Weights>> loadWeights(const std::string& file);

    //!
    //! \brief Loads shape from weights file
    //!
    nvinfer1::Dims loadShape(std::ifstream& input);

    //!
    //! \brief Transpose weights
    //!
    void transposeWeights(nvinfer1::Weights& wts, int hiddenSize);

    //!
    //! \brief Add an MLP layer
    //!
    nvinfer1::ILayer* addMLPLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& inputTensor,
        int32_t hiddenSize, nvinfer1::Weights wts, nvinfer1::Weights bias, nvinfer1::ActivationType actType, int idx);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MLP network by using the API to create a model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleMLP::build()
{
    mWeightMap = loadWeights(locateFile(mParams.weightsFile, mParams.dataDirs));

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
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

    assert(network->getNbInputs() == 1);
    auto inputDims = network->getInput(0)->getDimensions();
    assert(inputDims.nbDims == 3);

    assert(network->getNbOutputs() == 1);
    auto outputDims = network->getOutput(0)->getDimensions();
    assert(outputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses the API to create the MLP Network
//!
//! \param network Pointer to the network that will be populated with the MLP network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleMLP::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    // FC layers must still have 3 dimensions, so we create a {C, 1, 1,} matrix.
    // Currently the mnist example is only trained in FP32 mode.
    auto input = network->addInput(mParams.inputTensorNames[0].c_str(), nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims3{(mParams.inputH * mParams.inputW), 1, 1});
    assert(input != nullptr);

    for (int i = 0; i < 2; ++i)
    {
        std::stringstream weightStr, biasStr;
        weightStr << "hiddenWeights" << i;
        biasStr << "hiddenBias" << i;
        // Transpose hidden layer weights
        transposeWeights(mWeightMap[weightStr.str()].second, 256);
        auto mlpLayer = addMLPLayer(network.get(), *input, 256, mWeightMap[weightStr.str()].second,
            mWeightMap[biasStr.str()].second, nvinfer1::ActivationType::kSIGMOID, i);
        input = mlpLayer->getOutput(0);
    }
    // Transpose output layer weights
    transposeWeights(mWeightMap["outputWeights"].second, mParams.outputSize);

    auto finalLayer = addMLPLayer(network.get(), *input, mParams.outputSize, mWeightMap["outputWeights"].second,
        mWeightMap["outputBias"].second, nvinfer1::ActivationType::kSIGMOID, -1);
    assert(finalLayer != nullptr);
    // Run topK to get the final result
    auto topK = network->addTopK(*finalLayer->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, 0x1);
    assert(topK != nullptr);
    topK->setName("OutputTopK");
    topK->getOutput(1)->setName(mParams.outputTensorNames[0].c_str());
    network->markOutput(*topK->getOutput(1));
    topK->getOutput(1)->setType(nvinfer1::DataType::kINT32);

    // Build engine
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    builder->setFp16Mode(mParams.fp16);
    builder->setInt8Mode(mParams.int8);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 64.0f, 64.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
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
bool SampleMLP::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
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
bool SampleMLP::processInput(const samplesCommon::BufferManager& buffers)
{
    // Read a random digit file
    srand(unsigned(time(nullptr)));
    mNumber = rand() % mParams.outputSize;
    std::vector<uint8_t> fileData(mParams.inputH * mParams.inputW);
    // read a random digit file from the data directory for use as input.
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), mParams.inputH,
        mParams.inputW);

    // print the ascii representation of the file that was loaded.
    gLogInfo << "Input:\n";
    for (int i = 0; i < (mParams.inputH * mParams.inputW); i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % mParams.inputW) ? "" : "\n");
    }
    gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Normalize the data the same way TensorFlow does.
    for (int i = 0; i < mParams.inputH * mParams.inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i]) / 255.0f;
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleMLP::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    int idx = static_cast<int*>(buffers.getHostBuffer(mParams.outputTensorNames[0]))[0];
    bool pass = (idx == mNumber);
    if (pass)
    {
        gLogInfo << "Algorithm chose " << idx << std::endl;
    }
    else
    {
        gLogInfo << "Algorithm chose " << idx << " but expected " << mNumber << "." << std::endl;
    }

    return pass;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleMLP::teardown()
{
    // Release weights host memory
    for (auto& mem : mWeightMap)
    {
        auto weight = mem.second.second;
        {
            delete[] static_cast<const float*>(weight.values);
        }
    }

    return true;
}

//!
//! \brief Loads weights from weights file
//!
//! \details Our weight files are in a very simple space delimited format.
//!          type is the integer value of the DataType enum in NvInfer.h.
//!          <number of buffers>
//!          for each buffer: [name] [type] [size] <data x size in hex>
//!
std::map<std::string, std::pair<nvinfer1::Dims, nvinfer1::Weights>> SampleMLP::loadWeights(const std::string& file)
{
    std::map<std::string, std::pair<nvinfer1::Dims, nvinfer1::Weights>> weightMap;
    std::ifstream input(file, std::ios_base::binary);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--)
    {
        std::pair<nvinfer1::Dims, nvinfer1::Weights> wt{};
        std::int32_t type;
        std::string name;
        input >> name >> std::dec >> type;
        wt.first = loadShape(input);
        wt.second.type = static_cast<nvinfer1::DataType>(type);
        wt.second.count = std::accumulate(wt.first.d, wt.first.d + wt.first.nbDims, 1, std::multiplies<int32_t>());
        assert(wt.second.type == nvinfer1::DataType::kFLOAT);

        float* value = new float[wt.second.count];
        input.read(reinterpret_cast<char*>(value), wt.second.count * sizeof(float));
        assert(input.peek() == '\n');
        // Consume the newline at the end of the data blob.
        input.get();
        wt.second.values = value;
        weightMap[name] = wt;
    }
    return weightMap;
}

//!
//! \brief Loads shape from weights file
//!
nvinfer1::Dims SampleMLP::loadShape(std::ifstream& input)
{
    // Initial format is "(A, B, C,...,Y [,])"
    nvinfer1::Dims shape{};
    std::string shapeStr;

    // Convert to "(A,B,C,...,Y[,])"
    do
    {
        std::string tmp;
        input >> tmp;
        shapeStr += tmp;
    } while (*shapeStr.rbegin() != ')');
    assert(input.peek() == ' ');

    // Consume the space between the shape and the data buffer.
    input.get();

    // Convert to "A,B,C,...,Y[,]"
    assert(*shapeStr.begin() == '(');
    shapeStr.erase(0, 1); //
    assert(*shapeStr.rbegin() == ')');
    shapeStr.pop_back();

    // Convert to "A,B,C,...,Y"
    if (*shapeStr.rbegin() == ',')
    {
        shapeStr.pop_back(); // Remove the excess ',' character
    }

    std::vector<std::string> shapeDim;
    std::size_t begin = 0, size = shapeStr.size();
    while (begin != std::string::npos)
    {
        std::size_t found = shapeStr.find_first_of(",", begin);
        // Handle case of two or more delimiters in a row.
        if (found != begin)
        {
            shapeDim.push_back(shapeStr.substr(begin, found - begin));
        }
        begin = found + 1;
        // Handle case of no more tokens.
        if (found == std::string::npos)
        {
            break;
        }
        // Handle case of delimiter being last or first token.
        if (begin >= size)
        {
            break;
        }
    }

    // Convert to {A, B, C,...,Y}
    assert(shapeDim.size() <= shape.MAX_DIMS);
    assert(shapeDim.size() > 0);
    assert(shape.nbDims == 0);
    std::for_each(
        shapeDim.begin(), shapeDim.end(), [&](std::string& val) { shape.d[shape.nbDims++] = std::stoi(val); });
    return shape;
}

//!
//! \brief Transpose weights
//!
void SampleMLP::transposeWeights(nvinfer1::Weights& wts, int hiddenSize)
{
    int d = 0;
    int dim0 = hiddenSize;       // 256 or 10
    int dim1 = wts.count / dim0; // 784 or 256
    std::vector<uint32_t> trans_wts(wts.count);
    for (int d0 = 0; d0 < dim0; ++d0)
    {
        for (int d1 = 0; d1 < dim1; ++d1)
        {
            trans_wts[d] = *((uint32_t*) wts.values + d1 * dim0 + d0);
            d++;
        }
    }

    for (int k = 0; k < wts.count; ++k)
    {
        *((uint32_t*) wts.values + k) = trans_wts[k];
    }
}

//!
//! \brief Add an MLP layer
//!
//! \details
//!     The addMLPLayer function is a simple helper function that creates the combination required for an
//!     MLP layer. By replacing the implementation of this sequence with various implementations, then
//!     then it can be shown how TensorRT optimizations those layer sequences.
//!
nvinfer1::ILayer* SampleMLP::addMLPLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& inputTensor,
    int32_t hiddenSize, nvinfer1::Weights wts, nvinfer1::Weights bias, nvinfer1::ActivationType actType, int idx)
{
    std::string baseName("MLP Layer" + (idx == -1 ? "Output" : std::to_string(idx)));
    auto fc = network->addFullyConnected(inputTensor, hiddenSize, wts, bias);
    assert(fc != nullptr);
    std::string fcName = baseName + "FullyConnected";
    fc->setName(fcName.c_str());
    auto act = network->addActivation(*fc->getOutput(0), actType);
    assert(act != nullptr);
    std::string actName = baseName + "Activation";
    act->setName(actName.c_str());
    return act;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleMLPParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleMLPParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
        params.dataDirs.push_back("data/mlp/");
        params.dataDirs.push_back("data/samples/mlp/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.inputTensorNames.push_back("input");
    params.batchSize = 1;
    params.outputTensorNames.push_back("output");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.inputH = 28;
    params.inputW = 28;
    params.outputSize = 10;
    params.weightsFile = "sampleMLP.wts2";

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_mlp [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mlp/, data/mlp/)"
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
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleMLP sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for MLP" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
