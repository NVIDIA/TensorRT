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
//! sampleFasterRCNN.cpp
//! This file contains the implementation of the FasterRCNN sample. It creates the network using
//! the FasterRCNN caffe model.
//! It can be run with the following command line:
//! Command: ./sample_fasterRCNN [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
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
#include <unordered_map>

using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_fasterRCNN";

//!
//! \brief The SampleFasterRCNNParams structure groups the additional parameters required by
//!         the FasterRCNN sample.
//!
struct SampleFasterRCNNParams : public samplesCommon::CaffeSampleParams
{
    int outputClsSize; //!< The number of output classes
    int nmsMaxOut;     //!< The maximum number of detection post-NMS
    std::string dynamicRangeFileName; //!< The name of dynamic range file
};

//! \brief  The SampleFasterRCNN class implements the FasterRCNN sample
//!
//! \details It creates the network using a caffe model
//!
class SampleFasterRCNN
{
public:
    SampleFasterRCNN(const SampleFasterRCNNParams& params)
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
    SampleFasterRCNNParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    static const int kIMG_CHANNELS = 3;
    static const int kIMG_H = 375;
    static const int kIMG_W = 500;
    std::vector<samplesCommon::PPM<kIMG_CHANNELS, kIMG_H, kIMG_W>> mPPMs; //!< PPMs of test images

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses a Caffe model for FasterRCNN and creates a TensorRT network
    //!
    void constructNetwork(SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser,
        SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections, handles post-processing of bounding boxes and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Performs inverse bounding box transform and clipping
    //!
    void bboxTransformInvAndClip(const float* rois, const float* deltas, float* predBBoxes, const float* imInfo,
        const int N, const int nmsMaxOut, const int numCls);

    //!
    //! \brief Performs non maximum suppression on final bounding boxes
    //!
    std::vector<int> nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
        const int classNum, const int numClasses, const float nmsThreshold);

    //!
    //! \brief Sets per-tensor DynamicRange for int8
    //!
    bool setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief Reads per-tensor DynamicRange for int8
    //!
    bool readPerTensorDynamicRangeValues(std::unordered_map<std::string, float>& dynamicRangeMap) const;
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the FasterRCNN network by parsing the caffe model and builds
//!          the engine that will be used to run FasterRCNN (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleFasterRCNN::build()
{
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

    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    constructNetwork(parser, builder, network, config);

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

    ASSERT(network->getNbInputs() == 2);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a caffe parser to create the FasterRCNN network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the FasterRCNN network
//!
//! \param builder Pointer to the engine builder
//!
void SampleFasterRCNN::constructNetwork(SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser,
    SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
    SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor
        = parser->parse(locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
            locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(), *network, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
    {
        // when marking the plugin output rois as network output, TRT propagates the FP32 requirement
        // from plugin output to plugin input, and finally only FP32 -> FP32 is selected.
        // If we want to enable int8 for plugin, we should use addIdentity to break FP32 propagation.
        if (mParams.int8 && s == "rois")
        {
            sample::gLogInfo << "Add an identity layer after the rois tensor to enable INT8 I/O plugin." << std::endl;
            auto rois_old = blobNameToTensor->find(s.c_str());
            rois_old->setName("rois_");
            auto rois_new = network->addIdentity(*rois_old)->getOutput(0);
            rois_new->setName(s.c_str());
            network->markOutput(*rois_new); 
        }
        else
        {
            network->markOutput(*blobNameToTensor->find(s.c_str()));  
        }

    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);

    if (mParams.int8)
    {
        // Enable INT8 model. Required to set custom per tensor dynamic range or INT8 Calibration
        config->setFlag(BuilderFlag::kINT8);
        // Mark calibrator as null. As user provides dynamic range for each tensor, no calibrator is required
        config->setInt8Calibrator(nullptr);
        if (!setDynamicRange(network))
        {
            sample::gLogError << "Unable to set per tensor dynamic range. The sample will continue, "
                            <<"but you may get wrong detection results. Please try FP32 precision." << std::endl;
        }
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleFasterRCNN::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 2);
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
bool SampleFasterRCNN::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads per tensor dyanamic range values
//!
bool SampleFasterRCNN::readPerTensorDynamicRangeValues(std::unordered_map<std::string, float>& dynamicRangeMap) const
{
    std::ifstream iDynamicRangeStream(locateFile(mParams.dynamicRangeFileName, mParams.dataDirs));
    if (!iDynamicRangeStream)
    {
        sample::gLogError << "Could not find per tensor dynamic range file: " << mParams.dynamicRangeFileName << std::endl;
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
        dynamicRangeMap[tensorName] = dynamicRange;
    }
    return true;
}

//!
//! \brief  Sets custom dynamic range for network tensors
//!
bool SampleFasterRCNN::setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    std::unordered_map<std::string, float> PerTensorDynamicRangeMap;
    if (!readPerTensorDynamicRangeValues(PerTensorDynamicRangeMap))
    {
        return false;
    }

    sample::gLogInfo << "Setting Per Tensor Dynamic Range" << std::endl;
    // set dynamic range for network input tensors
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        std::string tName = network->getInput(i)->getName();
        if (PerTensorDynamicRangeMap.find(tName) != PerTensorDynamicRangeMap.end())
        {
            if (!network->getInput(i)->setDynamicRange(
                    -PerTensorDynamicRangeMap.at(tName), PerTensorDynamicRangeMap.at(tName)))
            {
                return false;
            }
        }
    }
    // set dynamic range for layer output tensors
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto lyr = network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j)
        {
            std::string tName = lyr->getOutput(j)->getName();
            if (PerTensorDynamicRangeMap.find(tName) != PerTensorDynamicRangeMap.end())
            {
                if (!lyr->getOutput(j)->setDynamicRange(
                        -PerTensorDynamicRangeMap.at(tName), PerTensorDynamicRangeMap.at(tName)))
                {
                    return false;
                }
            }
            // special operation for this sample
            // Convolution's output name is (Unnamed Layer*) [Convolution]_output.
            // Dynamic ranges of these convolutions should be the same with relu's output.
            else
            {
                if(i + 1 < network->getNbLayers())
                {
                    std::string nextTensorName = network->getLayer(i + 1)->getOutput(0)->getName();
                    if (PerTensorDynamicRangeMap.find(nextTensorName) != PerTensorDynamicRangeMap.end())
                    {
                        if (!lyr->getOutput(j)->setDynamicRange(
                                -PerTensorDynamicRangeMap.at(nextTensorName), PerTensorDynamicRangeMap.at(nextTensorName)))
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleFasterRCNN::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;

    // Available images
    const std::vector<std::string> imageList = {"000456.ppm", "000542.ppm", "001150.ppm", "001763.ppm", "004545.ppm"};
    mPPMs.resize(batchSize);
    ASSERT(mPPMs.size() <= imageList.size());

    // Fill im_info buffer
    float* hostImInfoBuffer = static_cast<float*>(buffers.getHostBuffer("im_info"));
    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(imageList[i], mParams.dataDirs), mPPMs[i]);
        hostImInfoBuffer[i * 3] = float(mPPMs[i].h);     // Number of rows
        hostImInfoBuffer[i * 3 + 1] = float(mPPMs[i].w); // Number of columns
        hostImInfoBuffer[i * 3 + 2] = 1;                 // Image scale
    }

    // Fill data buffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("data"));
    // Pixel mean used by the Faster R-CNN's author
    const float pixelMean[3]{102.9801f, 115.9465f, 122.7717f}; // Also in BGR order
    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
                hostDataBuffer[i * volImg + c * volChl + j] = float(mPPMs[i].buffer[j * inputC + 2 - c]) - pixelMean[c];
        }
    }

    return true;
}

//!
//! \brief Filters output detections and handles post-processing of bounding boxes, verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleFasterRCNN::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int batchSize = mParams.batchSize;
    const int nmsMaxOut = mParams.nmsMaxOut;
    const int outputClsSize = mParams.outputClsSize;
    const int outputBBoxSize = mParams.outputClsSize * 4;

    const float* imInfo = static_cast<const float*>(buffers.getHostBuffer("im_info"));
    const float* deltas = static_cast<const float*>(buffers.getHostBuffer("bbox_pred"));
    const float* clsProbs = static_cast<const float*>(buffers.getHostBuffer("cls_prob"));
    float* rois = static_cast<float*>(buffers.getHostBuffer("rois"));

    // Unscale back to raw image space
    for (int i = 0; i < batchSize; ++i)
    {
        for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
        {
            rois[i * nmsMaxOut * 4 + j] /= imInfo[i * 3 + 2];
        }
    }

    std::vector<float> predBBoxes(batchSize * nmsMaxOut * outputBBoxSize, 0);
    bboxTransformInvAndClip(rois, deltas, predBBoxes.data(), imInfo, batchSize, nmsMaxOut, outputClsSize);

    const float nmsThreshold = 0.3f;
    const float score_threshold = 0.8f;
    const std::vector<std::string> classes{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"};

    // The sample passes if there is at least one detection for each item in the batch
    bool pass = true;

    for (int i = 0; i < batchSize; ++i)
    {
        float* bbox = predBBoxes.data() + i * nmsMaxOut * outputBBoxSize;
        const float* scores = clsProbs + i * nmsMaxOut * outputClsSize;
        int numDetections = 0;
        for (int c = 1; c < outputClsSize; ++c) // Skip the background
        {
            std::vector<std::pair<float, int>> scoreIndex;
            for (int r = 0; r < nmsMaxOut; ++r)
            {
                if (scores[r * outputClsSize + c] > score_threshold)
                {
                    scoreIndex.push_back(std::make_pair(scores[r * outputClsSize + c], r));
                    std::stable_sort(scoreIndex.begin(), scoreIndex.end(),
                        [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
                            return pair1.first > pair2.first;
                        });
                }
            }

            // Apply NMS algorithm
            const std::vector<int> indices = nonMaximumSuppression(scoreIndex, bbox, c, outputClsSize, nmsThreshold);

            numDetections += static_cast<int>(indices.size());

            // Show results
            for (unsigned k = 0; k < indices.size(); ++k)
            {
                const int idx = indices[k];
                const std::string storeName
                    = classes[c] + "-" + std::to_string(scores[idx * outputClsSize + c]) + ".ppm";
                sample::gLogInfo << "Detected " << classes[c] << " in " << mPPMs[i].fileName << " with confidence "
                                 << scores[idx * outputClsSize + c] * 100.0f << "% "
                                 << " (Result stored in " << storeName << ")." << std::endl;

                const samplesCommon::BBox b{bbox[idx * outputBBoxSize + c * 4], bbox[idx * outputBBoxSize + c * 4 + 1],
                    bbox[idx * outputBBoxSize + c * 4 + 2], bbox[idx * outputBBoxSize + c * 4 + 3]};
                writePPMFileWithBBox(storeName, mPPMs[i], b);
            }
        }
        pass &= numDetections >= 1;
    }
    return pass;
}

//!
//! \brief Performs inverse bounding box transform
//!
void SampleFasterRCNN::bboxTransformInvAndClip(const float* rois, const float* deltas, float* predBBoxes,
    const float* imInfo, const int N, const int nmsMaxOut, const int numCls)
{
    for (int i = 0; i < N * nmsMaxOut; ++i)
    {
        float width = rois[i * 4 + 2] - rois[i * 4] + 1;
        float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
        float ctr_x = rois[i * 4] + 0.5f * width;
        float ctr_y = rois[i * 4 + 1] + 0.5f * height;
        const float* imInfo_offset = imInfo + i / nmsMaxOut * 3;
        for (int j = 0; j < numCls; ++j)
        {
            float dx = deltas[i * numCls * 4 + j * 4];
            float dy = deltas[i * numCls * 4 + j * 4 + 1];
            float dw = deltas[i * numCls * 4 + j * 4 + 2];
            float dh = deltas[i * numCls * 4 + j * 4 + 3];
            float pred_ctr_x = dx * width + ctr_x;
            float pred_ctr_y = dy * height + ctr_y;
            float pred_w = exp(dw) * width;
            float pred_h = exp(dh) * height;
            predBBoxes[i * numCls * 4 + j * 4]
                = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 1]
                = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 2]
                = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 3]
                = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
        }
    }
}

//!
//! \brief Performs non maximum suppression on final bounding boxes
//!
std::vector<int> SampleFasterRCNN::nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
    const int classNum, const int numClasses, const float nmsThreshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto i : scoreIndex)
    {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(
                    &bbox[(idx * numClasses + classNum) * 4], &bbox[(kept_idx * numClasses + classNum) * 4]);
                keep = overlap <= nmsThreshold;
            }
            else
            {
                break;
            }
        }
        if (keep)
        {
            indices.push_back(idx);
        }
    }
    return indices;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleFasterRCNNParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleFasterRCNNParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/faster-rcnn/");
        params.dataDirs.push_back("data/samples/faster-rcnn/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.prototxtFileName = "faster_rcnn_test_iplugin.prototxt";
    params.weightsFileName = "VGG16_faster_rcnn_final.caffemodel";
    params.inputTensorNames.push_back("data");
    params.inputTensorNames.push_back("im_info");
    params.batchSize = 5;
    params.outputTensorNames.push_back("bbox_pred");
    params.outputTensorNames.push_back("cls_prob");
    params.outputTensorNames.push_back("rois");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.dynamicRangeFileName = "tensor_range.txt";

    params.outputClsSize = 21;
    params.nmsMaxOut
        = 300; // This value needs to be changed as per the nmsMaxOut value set in RPROI plugin parameters in prototxt

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_fasterRCNN [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/faster-rcnn/ and data/faster-rcnn/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8  Enable int8 precision, in addition to fp32 (default = disabled)" << std::endl;
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

    initLibNvInferPlugins(&sample::gLogger, "");

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleFasterRCNN sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for FasterRCNN" << std::endl;

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
