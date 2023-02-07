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

using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_uff_ssd";
const std::vector<std::string> gImgFnames = {"dog.ppm", "bus.ppm"};

//!
//! \brief The SampleUffSSDParams structure groups the additional parameters required by
//!         the Uff SSD sample.
//!
struct SampleUffSSDParams : public samplesCommon::SampleParams
{
    std::string uffFileName;    //!< The file name of the UFF model to use
    std::string labelsFileName; //!< The file namefo the class labels
    int32_t outputClsSize;      //!< The number of output classes
    int32_t calBatchSize;       //!< The size of calibration batch
    int32_t nbCalBatches;       //!< The number of batches for calibration
    int32_t keepTopK;           //!< The maximum number of detection post-NMS
    float visualThreshold;      //!< The minimum score threshold to consider a detection
};

//! \brief  The SampleUffSSD class implements the SSD sample
//!
//! \details It creates the network using an UFF model
//!
class SampleUffSSD
{
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
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleUffSSD::build()
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
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

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 3);

    ASSERT(network->getNbOutputs() == 2);

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
    parser->registerInput(mParams.inputTensorNames[0].c_str(), Dims3(3, 300, 300), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputTensorNames[0].c_str());

    auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.int8)
    {
        sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        const std::string listFileName = "list.txt";
        const int32_t imageC = 3;
        const int32_t imageH = 300;
        const int32_t imageW = 300;
        nvinfer1::Dims4 imageDims{};
        imageDims = nvinfer1::Dims4{mParams.calBatchSize, imageC, imageH, imageW};
        BatchStream calibrationStream(
            mParams.calBatchSize, mParams.nbCalBatches, imageDims, listFileName, mParams.dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(
            calibrationStream, 0, "UffSSD", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }

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
//!          sets inputs, executes the engine and verifies the detection outputs.
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
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    const bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
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
    const int32_t inputC = mInputDims.d[0];
    const int32_t inputH = mInputDims.d[1];
    const int32_t inputW = mInputDims.d[2];
    const int32_t batchSize = mParams.batchSize;

    mPPMs.resize(batchSize);
    assert(mPPMs.size() == gImgFnames.size());
    for (int32_t i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(gImgFnames[i], mParams.dataDirs), mPPMs[i]);
    }

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Host memory for input buffer
    for (int32_t i = 0, volImg = inputC * inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (int32_t c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (uint32_t j = 0, volChl = inputH * inputW; j < volChl; ++j)
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
    const int32_t inputH = mInputDims.d[1];
    const int32_t inputW = mInputDims.d[2];
    const int32_t batchSize = mParams.batchSize;
    const int32_t keepTopK = mParams.keepTopK;
    const float visualThreshold = mParams.visualThreshold;
    const int32_t outputClsSize = mParams.outputClsSize;

    const float* detectionOut = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    const int32_t* keepCount = static_cast<const int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    // Read COCO class labels from file
    std::vector<std::string> classes(outputClsSize);
    {
        std::ifstream labelFile(locateFile(mParams.labelsFileName, mParams.dataDirs));
        std::string line;
        int32_t id = 0;
        while (getline(labelFile, line))
        {
            classes[id++] = line;
        }
    }

    bool pass = true;

    for (int32_t bi = 0; bi < batchSize; ++bi)
    {
        int32_t numDetections = 0;
        bool correctDetection = false;

        for (int32_t i = 0; i < keepCount[bi]; ++i)
        {
            const float* det = &detectionOut[0] + (bi * keepTopK + i) * 7;
            if (det[2] < visualThreshold)
            {
                continue;
            }

            // Output format for each detection is stored in the below order
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            const int32_t detection = det[1];
            assert(detection < outputClsSize);
            const std::string outFname = classes[detection] + "-" + std::to_string(det[2]) + ".ppm";

            numDetections++;

            if ((bi == 0 && classes[detection] == "dog")
                || (bi == 1 && (classes[detection] == "truck" || classes[detection] == "car")))
            {
                correctDetection = true;
            }

            sample::gLogInfo << "Detected " << classes[detection].c_str() << " in image "
                             << static_cast<int32_t>(det[0]) << " (" << mPPMs[bi].fileName.c_str() << ")"
                             << " with confidence " << det[2] * 100.f << " and coordinates (" << det[3] * inputW << ", "
                             << det[4] * inputH << ")"
                             << ", (" << det[5] * inputW << ", " << det[6] * inputH << ")." << std::endl;

            sample::gLogInfo << "Result stored in: " << outFname.c_str() << std::endl;

            samplesCommon::writePPMFileWithBBox(
                outFname, mPPMs[bi], {det[3] * inputW, det[4] * inputH, det[5] * inputW, det[6] * inputH});
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
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/ssd/");
        params.dataDirs.push_back("data/ssd/VOC2007/");
        params.dataDirs.push_back("data/ssd/VOC2007/PPMImages/");
        params.dataDirs.push_back("data/samples/ssd/");
        params.dataDirs.push_back("data/samples/ssd/VOC2007/");
        params.dataDirs.push_back("data/samples/ssd/VOC2007/PPMImages/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.uffFileName = "sample_ssd_relu6.uff";
    params.labelsFileName = "ssd_coco_labels.txt";
    params.inputTensorNames.push_back("Input");
    params.batchSize = gImgFnames.size();
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
    std::cout << "Usage: ./sample_uff_ssd [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<N>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/ssd/ and data/ssd/"
              << std::endl;
    std::cout << "--useDLACore    Specify a DLA engine for layers that support DLA. Value can range from 0 to N-1, "
                 "where N is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8          Specify to run in int8 mode." << std::endl;
}

int32_t main(int32_t argc, char** argv)
{
    samplesCommon::Args args;
    const bool argsOK = samplesCommon::parseArgs(args, argc, argv);

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

    SampleUffSSD sample(initializeSampleParams(args));

    sample::gLogInfo << "Building inference engine for SSD" << std::endl;
    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogInfo << "Running inference engine for SSD" << std::endl;
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
