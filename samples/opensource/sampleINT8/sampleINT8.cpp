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
//! SampleINT8.cpp
//! This file contains the implementation of the sample. It creates the network using
//! the caffe model.
//! It can be run with the following command line:
//! Command: ./sample_int8 [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "BatchStream.h"
#include "EntropyCalibrator.h"
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

const std::string gSampleName = "TensorRT.sample_int8";

//!
//! \brief The SampleINT8Params structure groups the additional parameters required by
//!         the INT8 sample.
//!
struct SampleINT8Params : public samplesCommon::CaffeSampleParams
{
    int nbCalBatches;        //!< The number of batches for calibration
    int calBatchSize;        //!< The calibration batch size
    std::string networkName; //!< The name of the network
};

//! \brief  The SampleINT8 class implements the INT8 sample
//!
//! \details It creates the network using a caffe model
//!
class SampleINT8
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleINT8(const SampleINT8Params& params)
        : mParams(params)
        , mEngine(nullptr)
    {
        initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build(DataType dataType);

    //!
    //! \brief Checks if the platform supports the data type.
    //!
    //! \return Returns true if the platform supports the data type.
    //!
    bool isSupported(DataType dataType);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(std::pair<float, float>& score, int firstScoreBatch, int nbScoreBatches);

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleINT8Params mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses a Caffe model and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, DataType dataType);

    //!
    //! \brief Reads the input and stores it in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const float* data);

    //!
    //! \brief Scores model
    //!
    int calculateScore(
        const samplesCommon::BufferManager& buffers, float* labels, int batchSize, int outputSize, int threshold);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the network by parsing the caffe model and builds
//!          the engine that will be used to run the model (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleINT8::build(DataType dataType)
{

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

    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
    {
        return false;
    }

    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser, dataType);
    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Checks if the platform supports the data type.
//!
//! \return Returns true if the platform supports the data type.
//!
bool SampleINT8::isSupported(DataType dataType)
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
    {
        return false;
    }

    return true;
}

//!
//! \brief Uses a caffe parser to create the network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleINT8::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, DataType dataType)
{
    mEngine = nullptr;
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor
        = parser->parse(locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
            locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(), *network,
            dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);

    for (auto& s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);
    config->setMaxWorkspaceSize(1_GiB);
    config->setFlag(BuilderFlag::kDEBUG);
    if (dataType == DataType::kHALF)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (dataType == DataType::kINT8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }
    builder->setMaxBatchSize(mParams.batchSize);

    if (dataType == DataType::kINT8)
    {
        MNISTBatchStream calibrationStream(mParams.calBatchSize, mParams.nbCalBatches, "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte", mParams.dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<MNISTBatchStream>(
            calibrationStream, 0, mParams.networkName.c_str(), mParams.inputTensorNames[0].c_str()));
        config->setInt8Calibrator(calibrator.get());
    }

    if (mParams.dlaCore >= 0)
    {
        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
        if (mParams.batchSize > builder->getMaxDLABatchSize())
        {
            gLogError << "Requested batch size " << mParams.batchSize << " is greater than the max DLA batch size of "
                      << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
            return false;
        }
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
bool SampleINT8::infer(std::pair<float, float>& score, int firstScoreBatch, int nbScoreBatches)
{
    float ms{0.0f};

    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    MNISTBatchStream batchStream(
        mParams.batchSize, nbScoreBatches + firstScoreBatch, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", mParams.dataDirs);
    batchStream.skip(firstScoreBatch);

    Dims outputDims = context->getEngine().getBindingDimensions(
        context->getEngine().getBindingIndex(mParams.outputTensorNames[0].c_str()));
    int outputSize = samplesCommon::volume(outputDims);
    int top1{0}, top5{0};
    float totalTime{0.0f};

    while (batchStream.next())
    {
        // Read the input data into the managed buffers
        assert(mParams.inputTensorNames.size() == 1);
        if (!processInput(buffers, batchStream.getBatch()))
        {
            return false;
        }

        // Memcpy from host input buffers to device input buffers
        buffers.copyInputToDevice();

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Use CUDA events to measure inference time
        cudaEvent_t start, end;
        CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
        CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
        cudaEventRecord(start, stream);

        bool status = context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr);
        if (!status)
        {
            return false;
        }

        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        totalTime += ms;

        // Memcpy from device output buffers to host output buffers
        buffers.copyOutputToHost();

        CHECK(cudaStreamDestroy(stream));

        top1 += calculateScore(buffers, batchStream.getLabels(), mParams.batchSize, outputSize, 1);
        top5 += calculateScore(buffers, batchStream.getLabels(), mParams.batchSize, outputSize, 5);

        if (batchStream.getBatchesRead() % 100 == 0)
        {
            gLogInfo << "Processing next set of max 100 batches" << std::endl;
        }
    }

    int imagesRead = (batchStream.getBatchesRead() - firstScoreBatch) * mParams.batchSize;
    score.first = float(top1) / float(imagesRead);
    score.second = float(top5) / float(imagesRead);

    gLogInfo << "Top1: " << score.first << ", Top5: " << score.second << std::endl;
    gLogInfo << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and "
             << totalTime / batchStream.getBatchesRead() << " ms/batch." << std::endl;

    return true;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleINT8::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads the input and stores it in a managed buffer
//!
bool SampleINT8::processInput(const samplesCommon::BufferManager& buffers, const float* data)
{
    // Fill data buffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    std::memcpy(hostDataBuffer, data, mParams.batchSize * samplesCommon::volume(mInputDims) * sizeof(float));
    return true;
}

//!
//! \brief Scores model
//!
int SampleINT8::calculateScore(
    const samplesCommon::BufferManager& buffers, float* labels, int batchSize, int outputSize, int threshold)
{
    float* probs = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    int success = 0;
    for (int i = 0; i < batchSize; i++)
    {
        float *prob = probs + outputSize * i, correct = prob[(int) labels[i]];

        int better = 0;
        for (int j = 0; j < outputSize; j++)
        {
            if (prob[j] >= correct)
            {
                better++;
            }
        }
        if (better <= threshold)
        {
            success++;
        }
    }
    return success;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleINT8Params initializeSampleParams(const samplesCommon::Args& args, int batchSize)
{
    SampleINT8Params params;
    // Use directories provided by the user, in addition to default directories.
    params.dataDirs = args.dataDirs;
    params.dataDirs.emplace_back("data/mnist/");
    params.dataDirs.emplace_back("int8/mnist/");
    params.dataDirs.emplace_back("samples/mnist/");
    params.dataDirs.emplace_back("data/samples/mnist/");
    params.dataDirs.emplace_back("data/int8/mnist/");
    params.dataDirs.emplace_back("data/int8_samples/mnist/");

    params.batchSize = batchSize;
    params.dlaCore = args.useDLACore;
    params.nbCalBatches = 10;
    params.calBatchSize = 50;
    params.inputTensorNames.push_back("data");
    params.outputTensorNames.push_back("prob");
    params.prototxtFileName = "deploy.prototxt";
    params.weightsFileName = "mnist_lenet.caffemodel";
    params.networkName = "mnist";
    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_int8 [-h or --help] [-d or --datadir=<path to data directory>] "
                 "[--useDLACore=<int>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories."
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "batch=N         Set batch size (default = 32)." << std::endl;
    std::cout << "start=N         Set the first batch to be scored (default = 100). All batches before this batch will "
                 "be used for calibration."
              << std::endl;
    std::cout << "score=N         Set the number of batches to be scored (default = 400)." << std::endl;
}

int main(int argc, char** argv)
{
    if (argc >= 2 && (!strncmp(argv[1], "help", 4) || !strncmp(argv[1], "--help", 6) || !strncmp(argv[1], "--h", 3) || !strncmp(argv[1], "-h", 2)))
    {
        printHelpInfo();
        return EXIT_FAILURE;
    }

    // By default we score over 57600 images starting at 512, so we don't score those used to search calibration
    int batchSize = 32;
    int firstScoreBatch = 16;
    int nbScoreBatches = 1800;

    // Parse extra arguments
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "batch=", 6))
        {
            batchSize = atoi(argv[i] + 6);
        }
        else if (!strncmp(argv[i], "start=", 6))
        {
            firstScoreBatch = atoi(argv[i] + 6);
        }
        else if (!strncmp(argv[i], "score=", 6))
        {
            nbScoreBatches = atoi(argv[i] + 6);
        }
    }

    if (batchSize > 128)
    {
        gLogError << "Please provide batch size <= 128" << std::endl;
        return EXIT_FAILURE;
    }

    if ((firstScoreBatch + nbScoreBatches) * batchSize > 60000)
    {
        gLogError << "Only 60000 images available" << std::endl;
        return EXIT_FAILURE;
    }

    samplesCommon::Args args;
    samplesCommon::parseArgs(args, argc, argv);

    SampleINT8 sample(initializeSampleParams(args, batchSize));

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    gLogInfo << "Building and running a GPU inference engine for INT8 sample" << std::endl;

    std::vector<std::string> dataTypeNames = {"FP32", "FP16", "INT8"};
    std::vector<DataType> dataTypes = {DataType::kFLOAT, DataType::kHALF, DataType::kINT8};
    std::vector<std::pair<float, float>> scores(3, std::make_pair(0.0f, 0.0f));
    for (size_t i = 0; i < dataTypes.size(); i++)
    {
        gLogInfo << dataTypeNames[i] << " run:" << nbScoreBatches << " batches of size " << batchSize << " starting at "
                 << firstScoreBatch << std::endl;

        if (!sample.build(dataTypes[i]))
        {
            if (!sample.isSupported(dataTypes[i]))
            {
                gLogWarning << "Skipping " << dataTypeNames[i] << " since the platform does not support this data type."
                            << std::endl;
                continue;
            }
            return gLogger.reportFail(sampleTest);
        }
        if (!sample.infer(scores[i], firstScoreBatch, nbScoreBatches))
        {
            return gLogger.reportFail(sampleTest);
        }
    }

    auto isApproximatelyEqual = [](float a, float b, double tolerance) { return (std::abs(a - b) <= tolerance); };
    double fp16tolerance{0.5}, int8tolerance{0.01};

    if (scores[1].first != 0.0f && !isApproximatelyEqual(scores[0].first, scores[1].first, fp16tolerance))
    {
        gLogError << "FP32(" << scores[0].first << ") and FP16(" << scores[1].first
                  << ") Top1 accuracy differ by more than " << fp16tolerance << "." << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    if (scores[2].first != 0.0f && !isApproximatelyEqual(scores[0].first, scores[2].first, int8tolerance))
    {
        gLogError << "FP32(" << scores[0].first << ") and Int8(" << scores[2].first
                  << ") Top1 accuracy differ by more than " << int8tolerance << "." << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    if (scores[1].second != 0.0f && !isApproximatelyEqual(scores[0].second, scores[1].second, fp16tolerance))
    {
        gLogError << "FP32(" << scores[0].second << ") and FP16(" << scores[1].second
                  << ") Top5 accuracy differ by more than " << fp16tolerance << "." << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    if (scores[2].second != 0.0f && !isApproximatelyEqual(scores[0].second, scores[2].second, int8tolerance))
    {
        gLogError << "FP32(" << scores[0].second << ") and INT8(" << scores[2].second
                  << ") Top5 accuracy differ by more than " << int8tolerance << "." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
