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
//! SampleUffSSD.cpp
//! This file contains the implementation of the Uff SSD sample. It creates the network using
//! the SSD UFF model.
//! It can be run with the following command line:
//! Command: ./sample_uff_ssd [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_uff_ssd";

//!
//! \brief The SampleUffSSDParams structure groups the additional parameters required by
//!         the Uff SSD sample.
//!
struct SampleUffSSDParams : public samplesCommon::SampleParams
{
    std::string uffFileName;    //!< The file name of the UFF model to use
    std::string labelsFileName; //!< The file namefo the class labels
    int outputClsSize;          //!< The number of output classes
    int calBatchSize;           //!< The size of calibration batch
    int nbCalBatches;           //!< The number of batches for calibration
    int keepTopK;               //!< The maximum number of detection post-NMS
    float visualThreshold;      //!< The minimum score threshold to consider a detection
};

//! \brief  The SampleUffSSD class implements the SSD sample
//!
//! \details It creates the network using an UFF model
//!
class SampleUffSSD
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleUffSSD(const SampleUffSSDParams& params)
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
    SampleUffSSDParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::vector<samplesCommon::PPM<3, 300, 300>> mPPMs; //!< PPMs of test images

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an UFF model for SSD and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvuffparser::IUffParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the SSD network by parsing the UFF model and builds
//!          the engine that will be used to run SSD (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleUffSSD::build()
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

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

    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 2);

    return true;
}

//!
//! \brief Uses a UFF parser to create the SSD Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the SSD network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleUffSSD::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvuffparser::IUffParser>& parser)
{
    parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(3, 300, 300), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputTensorNames[0].c_str());

    auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.int8)
    {
        gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        const std::string listFileName = "list.txt";
        const int imageC = 3;
        const int imageH = 300;
        const int imageW = 300;
        nvinfer1::DimsNCHW imageDims{};
        imageDims = nvinfer1::DimsNCHW{mParams.calBatchSize, imageC, imageH, imageW};
        BatchStream calibrationStream(
            mParams.calBatchSize, mParams.nbCalBatches, imageDims, listFileName, mParams.dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(
            calibrationStream, 0, "UffSSD", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }

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
bool SampleUffSSD::infer()
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

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleUffSSD::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleUffSSD::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;

    // Available images
    std::vector<std::string> imageList = {"dog.ppm", "bus.ppm"};
    mPPMs.resize(batchSize);
    assert(mPPMs.size() <= imageList.size());
    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(imageList[i], mParams.dataDirs), mPPMs[i]);
    }

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Host memory for input buffer
    for (int i = 0, volImg = inputC * inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j]
                    = (2.0 / 255.0) * float(mPPMs[i].buffer[j * inputC + c]) - 1.0;
            }
        }
    }

    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleUffSSD::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;
    const int keepTopK = mParams.keepTopK;
    const float visualThreshold = mParams.visualThreshold;
    const int outputClsSize = mParams.outputClsSize;

    const float* detectionOut = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    const int* keepCount = static_cast<const int*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    std::vector<std::string> classes(outputClsSize);

    // Gather class labels
    std::ifstream labelFile(locateFile(mParams.labelsFileName, mParams.dataDirs));
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        classes[id++] = line;
    }

    bool pass = true;

    for (int p = 0; p < batchSize; ++p)
    {
        int numDetections = 0;
        // at least one correct detection
        bool correctDetection = false;

        for (int i = 0; i < keepCount[p]; ++i)
        {
            const float* det = &detectionOut[0] + (p * keepTopK + i) * 7;
            if (det[2] < visualThreshold)
            {
                continue;
            }

            // Output format for each detection is stored in the below order
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            int detection = det[1];
            assert(detection < outputClsSize);
            std::string storeName = classes[detection] + "-" + std::to_string(det[2]) + ".ppm";

            numDetections++;
            if ((p == 0 && classes[detection] == "dog")
                || (p == 1 && (classes[detection] == "truck" || classes[detection] == "car")))
            {
                correctDetection = true;
            }

            gLogInfo << "Detected " << classes[detection].c_str() << " in the image " << int(det[0]) << " ("
                     << mPPMs[p].fileName.c_str() << ")"
                     << " with confidence " << det[2] * 100.f << " and coordinates (" << det[3] * inputW << ","
                     << det[4] * inputH << ")"
                     << ",(" << det[5] * inputW << "," << det[6] * inputH << ")." << std::endl;

            gLogInfo << "Result stored in " << storeName.c_str() << "." << std::endl;

            samplesCommon::writePPMFileWithBBox(
                storeName, mPPMs[p], {det[3] * inputW, det[4] * inputH, det[5] * inputW, det[6] * inputH});
        }
        pass &= correctDetection;
        pass &= numDetections >= 1;
    }

    return pass;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleUffSSDParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleUffSSDParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/ssd/");
        params.dataDirs.push_back("data/ssd/VOC2007/");
        params.dataDirs.push_back("data/ssd/VOC2007/PPMImages/");
        params.dataDirs.push_back("data/samples/ssd/");
        params.dataDirs.push_back("data/samples/ssd/VOC2007/");
        params.dataDirs.push_back("data/samples/ssd/VOC2007/PPMImages/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.uffFileName = "sample_ssd_relu6.uff";
    params.labelsFileName = "ssd_coco_labels.txt";
    params.inputTensorNames.push_back("Input");
    params.batchSize = 2;
    params.outputTensorNames.push_back("NMS");
    params.outputTensorNames.push_back("NMS_1");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.outputClsSize = 91;
    params.calBatchSize = 10;
    params.nbCalBatches = 10;
    params.keepTopK = 100;
    params.visualThreshold = 0.5;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_uff_ssd [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/ssd/ and data/ssd/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8          Specify to run in int8 mode." << std::endl;
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

    SampleUffSSD sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for SSD" << std::endl;

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
