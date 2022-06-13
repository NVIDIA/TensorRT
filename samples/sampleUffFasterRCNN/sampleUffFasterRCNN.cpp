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
//! sampleFasterRCNN_uff.cpp
//! This file contains the implementation of the Uff FasterRCNN sample. It creates the network using
//! the FasterRCNN UFF model.
//! It can be run with the following command line:
//! Command: ./sample_uff_fasterRCNN [-h]
//!

#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "frcnnUtils.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

using namespace samplesCommon;
using samplesCommon::SampleUniquePtr;

//! \brief Define the PPM objects as global variable.
//!
std::vector<vPPM> ppms;

//! \brief The name of this sample.
//!
const std::string gSampleName = "TensorRT.sample_uff_fasterRCNN";

//! \class
//!
//! \brief Define the parameters for this sample.
//!
struct SampleUffFasterRcnnParams : public samplesCommon::SampleParams
{
    std::string uffFileName; //!< The file name of the UFF model to use
    std::string inputNodeName;
    std::string outputClsName;
    std::string outputRegName;
    std::string outputProposalName;

    std::vector<std::string> inputImages;
    int inputChannels;
    int inputHeight;
    int inputWidth;
    int outputClassSize;
    int outputBboxSize;
    float nmsIouThresholdClassifier;
    float visualizeThreshold;
    std::vector<float> classifierRegressorStd;
    std::vector<std::string> classNames;

    int postNmsTopN;
    int calBatchSize;
    int nbCalBatches;

    int repeat;
    bool profile;

    std::string saveEngine;
    std::string loadEngine;
};

//! \class
//!
//! \brief The class that defines the overall workflow of this sample.
//!
class SampleUffFasterRcnn
{
public:
    SampleUffFasterRcnn(const SampleUffFasterRcnnParams& params)
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
    SampleUffFasterRcnnParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an UFF model for SSD and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvuffparser::IUffParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Helper function to do post-processing(apply delta to ROIs).
    //!
    void batch_inverse_transform_classifier(const float* roi_after_nms, int roi_num_per_img,
        const float* classifier_cls, const float* classifier_regr, std::vector<float>& pred_boxes,
        std::vector<int>& pred_cls_ids, std::vector<float>& pred_probs, std::vector<int>& box_num_per_img, int N);

    //!
    //! \brief NMS helper function in post-processing.
    //!
    std::vector<int> nms_classifier(std::vector<float>& boxes_per_cls, std::vector<float>& probs_per_cls,
        float NMS_OVERLAP_THRESHOLD, int NMS_MAX_BOXES);

    //!
    //! \brief Helper function to dump bbox-overlayed images as PPM files.
    //!
    void visualize_boxes(int img_num, int class_num, std::vector<float>& pred_boxes, std::vector<float>& pred_probs,
        std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img, std::vector<vPPM>& ppms);
};

bool SampleUffFasterRcnn::build()
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    if (mParams.loadEngine.size() > 0)
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.loadEngine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        if (mParams.dlaCore >= 0)
        {
            infer->setDLACore(mParams.dlaCore);
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());

        infer->destroy();
        sample::gLogInfo << "TRT Engine loaded from: " << mParams.loadEngine << std::endl;
        if (!mEngine)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

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

    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, parser);

    if (!constructed)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 3);
    ASSERT(network->getNbOutputs() == 3);
    return true;
}

bool SampleUffFasterRcnn::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvuffparser::IUffParser>& parser)
{
    SampleUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};

    parser->registerInput(mParams.inputNodeName.c_str(),
        Dims3(mParams.inputChannels, mParams.inputHeight, mParams.inputWidth), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputRegName.c_str());
    parser->registerOutput(mParams.outputClsName.c_str());
    parser->registerOutput(mParams.outputProposalName.c_str());
    auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);

    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);

    if (mParams.dlaCore >= 0)
    {
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(mParams.dlaCore);
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
    }

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
        const int imageC = 3;
        const int imageH = mParams.inputHeight;
        const int imageW = mParams.inputWidth;
        nvinfer1::Dims4 imageDims{mParams.calBatchSize, imageC, imageH, imageW};
        // To prevent compiler initialization warning with some versions of gcc
        for (int i = imageDims.nbDims; i < Dims::MAX_DIMS; ++i)
        {
            imageDims.d[i] = 0;
        }
        BatchStream calibrationStream(
            mParams.calBatchSize, mParams.nbCalBatches, imageDims, listFileName, mParams.dataDirs);
        calibrator.reset(
            new Int8EntropyCalibrator2(calibrationStream, 0, "UffFasterRcnn", mParams.inputNodeName.c_str()));
        config->setFlag(BuilderFlag::kINT8);
        // Fallback to FP16 if there is no INT8 kernels.
        config->setFlag(BuilderFlag::kFP16);
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

    if (mParams.saveEngine.size() > 0)
    {
        std::ofstream p(mParams.saveEngine, std::ios::binary);
        if (!p)
        {
            return false;
        }
        nvinfer1::IHostMemory* ptr = mEngine->serialize();
        ASSERT(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << mParams.saveEngine << std::endl;
    }

    return true;
}

bool SampleUffFasterRcnn::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    SimpleProfiler profiler("FasterRCNN performance");

    if (mParams.profile)
    {
        context->setProfiler(&profiler);
    }

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
    bool status{true};

    for (int i = 0; i < mParams.repeat; ++i)
    {
        status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
        if (!status)
        {
            return false;
        }
    }

    if (mParams.profile)
    {
        std::cout << profiler;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return status;
}

bool SampleUffFasterRcnn::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

bool SampleUffFasterRcnn::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mParams.inputChannels;
    const int inputH = mParams.inputHeight;
    const int inputW = mParams.inputWidth;
    const int batchSize = mParams.batchSize;
    std::vector<std::string> imageList = mParams.inputImages;
    ppms.resize(batchSize);
    ASSERT(ppms.size() <= imageList.size());

    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(imageList[i], ppms[i], mParams.dataDirs);
        // resize to input dimensions.
        resizePPM(ppms[i], inputW, inputH);
    }

    // subtract image channel mean
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputNodeName));
    float pixelMean[3]{103.939, 116.779, 123.68};

    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j] = float(ppms[i].buffer[j * inputC + 2 - c]) - pixelMean[c];
            }
        }
    }

    return true;
}

bool SampleUffFasterRcnn::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int batchSize = mParams.batchSize;
    const int outputClassSize = mParams.outputClassSize;
    std::vector<float> classifierRegressorStd;
    std::vector<std::string> classNames;
    const float* out_class = static_cast<const float*>(buffers.getHostBuffer(mParams.outputClsName));
    const float* out_reg = static_cast<const float*>(buffers.getHostBuffer(mParams.outputRegName));
    const float* out_proposal = static_cast<const float*>(buffers.getHostBuffer(mParams.outputProposalName));
    // host memory for outputs
    std::vector<float> pred_boxes;
    std::vector<int> pred_cls_ids;
    std::vector<float> pred_probs;
    std::vector<int> box_num_per_img;

    int post_nms_top_n = mParams.postNmsTopN;
    // post processing for stage 2.
    batch_inverse_transform_classifier(out_proposal, post_nms_top_n, out_class, out_reg, pred_boxes, pred_cls_ids,
        pred_probs, box_num_per_img, batchSize);
    visualize_boxes(batchSize, outputClassSize, pred_boxes, pred_probs, pred_cls_ids, box_num_per_img, ppms);
    return true;
}

SampleUffFasterRcnnParams initializeSampleParams(const FrcnnArgs& args)
{
    SampleUffFasterRcnnParams params;

    if (args.dataDirs.empty())
    {
        // Use default directories if user hasn't provided directory paths
        params.dataDirs.push_back("data/faster-rcnn/");
        params.dataDirs.push_back("data/samples/faster-rcnn/");
    }
    else
    {
        // Use the data directory provided by the user
        params.dataDirs = args.dataDirs;
        params.dataDirs.push_back("data/faster-rcnn/");
        params.dataDirs.push_back("data/samples/faster-rcnn/");
    }
    params.batchSize = static_cast<int>(args.inputImages.size());
    params.inputImages = args.inputImages;
    params.uffFileName = "faster_rcnn.uff";
    params.inputNodeName = "input_1";
    params.outputClsName = "dense_class/Softmax";
    params.outputRegName = "dense_regress/BiasAdd";
    params.outputProposalName = "proposal";
    params.classNames.push_back("Automobile");
    params.classNames.push_back("Bicycle");
    params.classNames.push_back("Person");
    params.classNames.push_back("Roadsign");
    params.classNames.push_back("background");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.repeat = args.repeat;
    params.profile = args.profile;
    params.inputChannels = 3;
    params.inputHeight = args.inputHeight;
    params.inputWidth = args.inputWidth;
    params.nmsIouThresholdClassifier = 0.3f;
    params.visualizeThreshold = 0.6f;
    params.classifierRegressorStd.push_back(10.0f);
    params.classifierRegressorStd.push_back(10.0f);
    params.classifierRegressorStd.push_back(5.0f);
    params.classifierRegressorStd.push_back(5.0f);
    params.outputClassSize = params.classNames.size();
    params.outputBboxSize = (params.outputClassSize - 1) * 4;
    params.postNmsTopN = 300;
    params.calBatchSize = 4;
    params.nbCalBatches = 1;

    params.saveEngine = args.saveEngine;
    params.loadEngine = args.loadEngine;

    return params;
}

void printHelpInfo()
{
    std::cout << "Usage: ./sample_uff_fasterRCNN [OPTIONS] -H [HEIGHT] -W [WIDTH] -I [INPUT IMAGES]" << std::endl;
    std::cout << "=== Required Arguments ===" << std::endl;
    std::cout << "--inputHeight[-H]       Specify the input height of the model." << std::endl;
    std::cout << "--inputWidth[-W]        Specify the input width of the model." << std::endl;
    std::cout << "--inputImages[-I]       Comma-separated list of input images to use for inference." << std::endl;
    std::cout << "=== Optional Arguments ===" << std::endl;
    std::cout << "--help[-h]              Display help information and exit" << std::endl;
    std::cout << "--datadir[-d]           Specify path to a data directory, overriding "
                 "the default. This option can be repeated to add multiple directories."
                 " If the option is unspecified, the default is to search"
                 " data/faster-rcnn/ and data/samples/faster-rcnn/."
              << std::endl;
    std::cout << "--useDLACore[-u]        Specify a DLA engine for layers that support DLA. "
                 "Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--fp16[-f]              Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8[-i]              Specify to run in int8 mode." << std::endl;
    std::cout << "--profile[-p]           Whether to do per-layer profiling." << std::endl;
    std::cout << "--repeat[-r]            Specify the repeat number to execute the TRT context,"
                 " used to smooth the profiling time."
              << std::endl;
    std::cout << "--saveEngine[-s]        Path to save engine." << std::endl;
    std::cout << "--loadEngine[-l]        Path to load engine." << std::endl;
}

//! \brief Define the function to apply delta to ROIs
//!
void SampleUffFasterRcnn::batch_inverse_transform_classifier(const float* roi_after_nms, int roi_num_per_img,
    const float* classifier_cls, const float* classifier_regr, std::vector<float>& pred_boxes,
    std::vector<int>& pred_cls_ids, std::vector<float>& pred_probs, std::vector<int>& box_num_per_img, int N)
{
    auto max_index = [](const float* start, const float* end) -> int {
        float max_val = start[0];
        int max_pos = 0;

        for (int i = 1; start + i < end; ++i)
        {
            if (start[i] > max_val)
            {
                max_val = start[i];
                max_pos = i;
            }
        }

        return max_pos;
    };
    int box_num;

    for (int n = 0; n < N; ++n)
    {
        box_num = 0;

        for (int i = 0; i < roi_num_per_img; ++i)
        {
            auto max_idx = max_index(
                classifier_cls + n * roi_num_per_img * mParams.outputClassSize + i * mParams.outputClassSize,
                classifier_cls + n * roi_num_per_img * mParams.outputClassSize + i * mParams.outputClassSize
                    + mParams.outputClassSize);

            if (max_idx == (mParams.outputClassSize - 1)
                || classifier_cls[n * roi_num_per_img * mParams.outputClassSize + max_idx + i * mParams.outputClassSize]
                    < mParams.visualizeThreshold)
            {
                continue;
            }

            // inverse transform
            float tx, ty, tw, th;
            tx = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize
                     + max_idx * 4]
                / mParams.classifierRegressorStd[0];
            ty = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize + max_idx * 4
                     + 1]
                / mParams.classifierRegressorStd[1];
            tw = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize + max_idx * 4
                     + 2]
                / mParams.classifierRegressorStd[2];
            th = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize + max_idx * 4
                     + 3]
                / mParams.classifierRegressorStd[3];
            float y = roi_after_nms[n * roi_num_per_img * 4 + 4 * i] * static_cast<float>(mParams.inputHeight);
            float x = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 1] * static_cast<float>(mParams.inputWidth);
            float ymax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 2] * static_cast<float>(mParams.inputHeight);
            float xmax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 3] * static_cast<float>(mParams.inputWidth);
            float w = xmax - x;
            float h = ymax - y;
            float cx = x + w / 2.0f;
            float cy = y + h / 2.0f;
            float cx1 = tx * w + cx;
            float cy1 = ty * h + cy;
            float w1 = std::round(std::exp(static_cast<double>(tw)) * w * 0.5f) * 2.0f;
            float h1 = std::round(std::exp(static_cast<double>(th)) * h * 0.5f) * 2.0f;
            float x1 = std::round((cx1 - w1 / 2.0f) * 0.5f) * 2.0f;
            float y1 = std::round((cy1 - h1 / 2.0f) * 0.5f) * 2.0f;
            auto clip
                = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
            float x2 = x1 + w1;
            float y2 = y1 + h1;
            x1 = clip(x1, 0.0f, mParams.inputWidth - 1.0f);
            y1 = clip(y1, 0.0f, mParams.inputHeight - 1.0f);
            x2 = clip(x2, 0.0f, mParams.inputWidth - 1.0f);
            y2 = clip(y2, 0.0f, mParams.inputHeight - 1.0f);

            if (x2 > x1 && y2 > y1)
            {
                pred_boxes.push_back(x1);
                pred_boxes.push_back(y1);
                pred_boxes.push_back(x2);
                pred_boxes.push_back(y2);
                pred_probs.push_back(classifier_cls[n * roi_num_per_img * mParams.outputClassSize + max_idx
                    + i * mParams.outputClassSize]);
                pred_cls_ids.push_back(max_idx);
                ++box_num;
            }
        }

        box_num_per_img.push_back(box_num);
    }
}

//! \brief NMS on CPU in post-processing of classifier outputs.
//!
std::vector<int> SampleUffFasterRcnn::nms_classifier(std::vector<float>& boxes_per_cls,
    std::vector<float>& probs_per_cls, float NMS_OVERLAP_THRESHOLD, int NMS_MAX_BOXES)
{
    int num_boxes = boxes_per_cls.size() / 4;
    std::vector<std::pair<float, int>> score_index;

    for (int i = 0; i < num_boxes; ++i)
    {
        score_index.push_back(std::make_pair(probs_per_cls[i], i));
    }

    std::stable_sort(score_index.begin(), score_index.end(),
        [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
            return pair1.first > pair2.first;
        });
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

    for (auto i : score_index)
    {
        const int idx = i.second;
        bool keep = true;

        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(&boxes_per_cls[idx * 4], &boxes_per_cls[kept_idx * 4]);
                keep = overlap <= NMS_OVERLAP_THRESHOLD;
            }
            else
            {
                break;
            }
        }

        if (indices.size() >= static_cast<unsigned>(NMS_MAX_BOXES))
        {
            break;
        }

        if (keep)
        {
            indices.push_back(idx);
        }
    }

    return indices;
}

//! \brief Dump the detection results(bboxes) as PPM images, overlayed on original image.
//!
void SampleUffFasterRcnn::visualize_boxes(int img_num, int class_num, std::vector<float>& pred_boxes,
    std::vector<float>& pred_probs, std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img,
    std::vector<vPPM>& ppms)
{
    int box_start_idx = 0;
    std::vector<float> boxes_per_cls;
    std::vector<float> probs_per_cls;
    std::vector<BBox> det_per_img;

    for (int i = 0; i < img_num; ++i)
    {
        det_per_img.clear();

        for (int c = 0; c < (class_num - 1); ++c)
        { // skip the background
            boxes_per_cls.clear();
            probs_per_cls.clear();

            for (int k = box_start_idx; k < box_start_idx + box_num_per_img[i]; ++k)
            {
                if (pred_cls_ids[k] == c)
                {
                    boxes_per_cls.push_back(pred_boxes[4 * k]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 1]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 2]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 3]);
                    probs_per_cls.push_back(pred_probs[k]);
                }
            }

            // apply NMS algorithm per class
            auto indices_after_nms
                = nms_classifier(boxes_per_cls, probs_per_cls, mParams.nmsIouThresholdClassifier, mParams.postNmsTopN);

            // Show results
            for (unsigned k = 0; k < indices_after_nms.size(); ++k)
            {
                int idx = indices_after_nms[k];
                std::cout << "Detected " << mParams.classNames[c] << " in " << ppms[i].fileName << " with confidence "
                          << probs_per_cls[idx] * 100.0f << "% " << std::endl;
                BBox b{boxes_per_cls[idx * 4], boxes_per_cls[idx * 4 + 1], boxes_per_cls[idx * 4 + 2],
                    boxes_per_cls[idx * 4 + 3]};
                det_per_img.push_back(b);
            }
        }

        box_start_idx += box_num_per_img[i];
        writePPMFileWithBBox(ppms[i].fileName + "_det.ppm", ppms[i], det_per_img);
    }
}

int main(int argc, char** argv)
{
    FrcnnArgs args;
    bool argsOK = parseFrcnnArgs(args, argc, argv);

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

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    sample::gLogger.reportTestStart(sampleTest);
    SampleUffFasterRcnn sample(initializeSampleParams(args));
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
