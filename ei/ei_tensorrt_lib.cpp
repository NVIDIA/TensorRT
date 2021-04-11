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

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params, const char* model_file_name, float* output, int output_size)
        : mParams(params)
        , mEngine(nullptr)
        , model_file_name(model_file_name)
        , output_size(output_size)
        , output(output)
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

    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    std::string input_name;
    std::string output_name;
    std::string model_file_name;
    float* output;
    int output_size;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    sample::gLogInfo << "Input tensor name: " << network->getInput(0)->getName() << std::endl;
    sample::gLogInfo << "Output tensor name: " << network->getOutput(0)->getName() << std::endl;
    input_name = network->getInput(0)->getName();
    output_name = network->getOutput(0)->getName();
    // assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    // assert(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(model_file_name.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(512_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    auto start = chrono::steady_clock::now();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    auto end = chrono::steady_clock::now();
 
    sample::gLogInfo << "Elapsed inference time in nanoseconds : "
        << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
        << " ns" << endl;
 
    sample::gLogInfo << "Elapsed time in microseconds : "
        << chrono::duration_cast<chrono::microseconds>(end - start).count()
        << " µs" << endl;

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
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int chans = mInputDims.d[3];

    // // Read a random digit file
    // srand(unsigned(time(nullptr)));
    // std::vector<uint8_t> fileData(inputH * inputW);
    // mNumber = rand() % 10;
    // readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // // Print an ascii representation
    // sample::gLogInfo << "Input:" << std::endl;
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    // }
    // sample::gLogInfo << std::endl;

    sample::gLogInfo << mParams.dataDirs[0] << endl;
    std::ifstream infile(mParams.dataDirs[0], std::ifstream::binary);
    if( infile.fail() ) {
        sample::gLogError << "Failed to open input file" << endl;
        return false;
    }

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(input_name));
    if( hostDataBuffer ) {
        int i;
        sample::gLogInfo << "h,w,c: " << inputH << ", " << inputW << ", " << chans << endl;
        char comma;
        for (i = 0; i < inputH * inputW * chans; i++)
        {
            infile >> hostDataBuffer[i] >> comma;
        }
        sample::gLogInfo << hostDataBuffer[0] << " \n" << hostDataBuffer[i-1] << " " << i << endl;
    } else {
        sample::gLogInfo << "failed to get buffer by input tensor name." << endl;
    }
 
    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float* output_from_engine = static_cast<float*>(buffers.getHostBuffer(output_name));

    if( output_from_engine ) {
        memcpy(output, output_from_engine, output_size*sizeof(float)); //use output size provided in call.
        // output size derived by TensorRT parser is always -1.  Even though web onnx parser says 1x2 (or 1x3, etc)
        sample::gLogInfo << "Output size (according to onnx parser, currently not used): " << mOutputDims.d[1] << endl;
        float val{0.0f};
        int idx{0};

        // For some other model, we had to calculate the softmax manually here
        // For EI model, we already do the softmax in the model
        sample::gLogInfo << "Output:" << std::endl;
        for (int i = 0; i < output_size; i++)
        {
            sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                            << " "
                            << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
                            << std::endl;
        }
        sample::gLogInfo << std::endl;
        return idx == mNumber && val > 0.9f;
    } else {
        sample::gLogError << " Failed to get buffer by output tensor name";
        return false;
    }  
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams()
{
    samplesCommon::OnnxSampleParams params;
    // params.onnxFileName = "mnist.onnx"; //moved to where it's loaded
    params.dlaCore = -1;
    params.int8 = false;
    params.fp16 = false;

    return params;
}

int ei_infer(const char* fileName, const char* model_file_name, float* output, int output_size)
{
    auto sampleTest = sample::gLogger.defineTest("ei_lib","ei_lib");

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST sample(initializeSampleParams(), model_file_name, output, output_size);

    sample.mParams.dataDirs.push_back(fileName);

    sample::gLogInfo << "EI TensorRT lib" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}

// int main()
// {
//     ei_infer("calc5c.rgb");
// }