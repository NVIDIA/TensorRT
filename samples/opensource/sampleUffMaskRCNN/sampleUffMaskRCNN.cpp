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

#ifndef _MSC_VER
#include <unistd.h>
#include <sys/time.h>
#endif

#include <assert.h>
#include <chrono>
#include <ctime>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvUffParser.h"

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

// max
#include <algorithm>

// MaskRCNN Parameter
#include "mrcnn_config.h"

const std::string gSampleName = "TensorRT.sample_maskrcnn";

namespace MaskRCNNUtils
{
struct RawDetection
{
    float y1, x1, y2, x2, class_id, score;
};

struct Mask
{
    float raw[MaskRCNNConfig::MASK_POOL_SIZE * 2 * MaskRCNNConfig::MASK_POOL_SIZE * 2];
};

struct BBoxInfo
{
    samplesCommon::BBox box;
    int label = -1;
    float prob = 0.0f;

    Mask* mask = nullptr;
};

template <typename T>
struct PPM
{
    std::string magic, fileName;
    int h, w, max;
    std::vector<T> buffer;
};

void readPPMFile(const std::string& filename, PPM<uint8_t>& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(filename, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open. ");
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);

    ppm.buffer.resize(ppm.w * ppm.h * 3, 0);

    infile.read(reinterpret_cast<char*>(ppm.buffer.data()), ppm.w * ppm.h * 3);
}

void writePPMFile(const std::string& filename, PPM<uint8_t>& ppm)
{
    std::ofstream outfile("./" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6"
            << "\n"
            << ppm.w << " " << ppm.h << "\n"
            << ppm.max << "\n";

    outfile.write(reinterpret_cast<char*>(ppm.buffer.data()), ppm.w * ppm.h * 3);
}

template <typename T>
void resizePPM(const PPM<T>& src, PPM<T>& dst, int target_height, int target_width, int channel)
{
    auto clip = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
    int original_height = src.h;
    int original_width = src.w;
    assert(dst.h == target_height);
    assert(dst.w == target_width);
    float ratio_h = static_cast<float>(original_height - 1.0f) / static_cast<float>(target_height - 1.0f);
    float ratio_w = static_cast<float>(original_width - 1.0f) / static_cast<float>(target_width - 1.0f);

    int dst_idx = 0;
    for (int y = 0; y < target_height; ++y)
    {
        for (int x = 0; x < target_width; ++x)
        {
            float x0 = static_cast<float>(x) * ratio_w;
            float y0 = static_cast<float>(y) * ratio_h;
            int left = static_cast<int>(clip(std::floor(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
            int top = static_cast<int>(clip(std::floor(y0), 0.0f, static_cast<float>(original_height - 1.0f)));
            int right = static_cast<int>(clip(std::ceil(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
            int bottom = static_cast<int>(clip(std::ceil(y0), 0.0f, static_cast<float>(original_height - 1.0f)));

            for (int c = 0; c < channel; ++c)
            {
                // H, W, C ordering
                T left_top_val = src.buffer[top * (original_width * channel) + left * (channel) + c];
                T right_top_val = src.buffer[top * (original_width * channel) + right * (channel) + c];
                T left_bottom_val = src.buffer[bottom * (original_width * channel) + left * (channel) + c];
                T right_bottom_val = src.buffer[bottom * (original_width * channel) + right * (channel) + c];
                float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
                float bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
                float lerp = clip(std::round(top_lerp + (bottom_lerp - top_lerp) * (y0 - top)), 0.0f, 255.0f);
                dst.buffer[dst_idx] = (static_cast<T>(lerp));
                dst_idx++;
            }
        }
    }
}

void padPPM(const PPM<uint8_t>& src, PPM<uint8_t>& dst, int top, int bottom, int left, int right)
{
    assert(dst.h == (src.h + top + bottom));
    assert(dst.w == (src.w + left + right));

    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < src.w; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                dst.buffer[(top + y) * dst.w * 3 + (left + x) * 3 + c] = src.buffer[y * src.w * 3 + x * 3 + c];
            }
        }
    }
}

void preprocessPPM(PPM<uint8_t>& src, PPM<uint8_t>& dst, int target_h, int target_w)
{
    assert(target_h == target_w);
    int input_dim = target_h;
    // padding the input img to model's input_size:
    const int image_dim = std::max(src.h, src.w);
    int resize_h = src.h * input_dim / image_dim;
    int resize_w = src.w * input_dim / image_dim;
    assert(resize_h == input_dim || resize_w == input_dim);

    int y_offset = (input_dim - resize_h) / 2;
    int x_offset = (input_dim - resize_w) / 2;

    // resize
    PPM<uint8_t> resized_ppm;
    resized_ppm.h = resize_h;
    resized_ppm.w = resize_w;
    resized_ppm.max = src.max;
    resized_ppm.buffer.resize(resize_h * resize_w * 3, 0);
    resizePPM<uint8_t>(src, resized_ppm, resize_h, resize_w, 3);

    // pad
    dst.h = target_h;
    dst.w = target_w;
    dst.max = src.max;
    dst.buffer.resize(dst.h * dst.w * 3, 0);
    padPPM(resized_ppm, dst, y_offset, input_dim - resize_h - y_offset, x_offset, input_dim - resize_w - x_offset);
}

PPM<uint8_t> resizeMask(const BBoxInfo& box, const float mask_threshold)
{
    PPM<uint8_t> result;
    if (!box.mask)
    {
        assert(result.buffer.size() == 0);
        return result;
    }

    const int h = box.box.y2 - box.box.y1;
    const int w = box.box.x2 - box.box.x1;

    PPM<float> raw_mask;
    raw_mask.h = MaskRCNNConfig::MASK_POOL_SIZE * 2;
    raw_mask.w = MaskRCNNConfig::MASK_POOL_SIZE * 2;
    raw_mask.buffer.resize(raw_mask.h * raw_mask.w, 0);
    for (int i = 0; i < raw_mask.h * raw_mask.w; i++)
        raw_mask.buffer[i] = box.mask->raw[i];

    PPM<float> resized_mask;
    resized_mask.h = h;
    resized_mask.w = w;
    resized_mask.buffer.resize(h * w, 0);
    resizePPM<float>(raw_mask, resized_mask, h, w, 1);

    result.h = h;
    result.w = w;
    result.buffer.resize(result.h * result.w, 0);
    for (int i = 0; i < h * w; i++)
    {
        if (resized_mask.buffer[i] > mask_threshold)
        {
            result.buffer[i] = 1;
        }
    }

    return result;
}

void maskPPM(
    PPM<uint8_t>& image, const PPM<uint8_t>& mask, const int start_x, const int start_y, const std::vector<int>& color)
{

    float alpha = 0.6f;

    for (int y = 0; y < mask.h; ++y)
    {
        for (int x = 0; x < mask.w; ++x)
        {
            uint8_t mask_pixel = mask.buffer[y * mask.w + x];
            if (mask_pixel == 1)
            {
                assert(0 <= start_y + y && start_y + y < image.h);
                assert(0 <= start_x + x && start_x + x < image.w);

                int cur_y = start_y + y;
                int cur_x = start_x + x;

                float p_r = static_cast<float>(image.buffer[(cur_y * image.w + cur_x) * 3]);
                float p_g = static_cast<float>(image.buffer[(cur_y * image.w + cur_x) * 3 + 1]);
                float p_b = static_cast<float>(image.buffer[(cur_y * image.w + cur_x) * 3 + 2]);

                image.buffer[(cur_y * image.w + cur_x) * 3]
                    = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_r * (1 - alpha) + color[0] * alpha)));
                image.buffer[(cur_y * image.w + cur_x) * 3 + 1]
                    = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_g * (1 - alpha) + color[1] * alpha)));
                image.buffer[(cur_y * image.w + cur_x) * 3 + 2]
                    = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_b * (1 - alpha) + color[2] * alpha)));
            }
            else
                assert(mask_pixel == 0);
        }
    }
}
void addBBoxPPM(PPM<uint8_t>& ppm, const BBoxInfo& box, const PPM<uint8_t>& resized_mask)
{
    const int x1 = box.box.x1;
    const int y1 = box.box.y1;
    const int x2 = box.box.x2;
    const int y2 = box.box.y2;
    std::vector<int> color = {rand() % 256, rand() % 256, rand() % 256};

    for (int x = x1; x <= x2; x++)
    {
        // bbox top border
        ppm.buffer[(y1 * ppm.w + x) * 3] = color[0];
        ppm.buffer[(y1 * ppm.w + x) * 3 + 1] = color[1];
        ppm.buffer[(y1 * ppm.w + x) * 3 + 2] = color[2];
        // bbox bottom border
        ppm.buffer[(y2 * ppm.w + x) * 3] = color[0];
        ppm.buffer[(y2 * ppm.w + x) * 3 + 1] = color[1];
        ppm.buffer[(y2 * ppm.w + x) * 3 + 2] = color[2];
    }

    for (int y = y1; y <= y2; y++)
    {
        // bbox left border
        ppm.buffer[(y * ppm.w + x1) * 3] = color[0];
        ppm.buffer[(y * ppm.w + x1) * 3 + 1] = color[1];
        ppm.buffer[(y * ppm.w + x1) * 3 + 2] = color[2];
        // bbox right border
        ppm.buffer[(y * ppm.w + x2) * 3] = color[0];
        ppm.buffer[(y * ppm.w + x2) * 3 + 1] = color[1];
        ppm.buffer[(y * ppm.w + x2) * 3 + 2] = color[2];
    }

    if (resized_mask.buffer.size() != 0)
    {
        maskPPM(ppm, resized_mask, x1, y1, color);
    }
}
} // namespace MaskRCNNUtils

struct SampleMaskRCNNParams : public samplesCommon::SampleParams
{
    std::string uffFileName;
    float maskThreshold;
};

class SampleMaskRCNN
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMaskRCNN(const SampleMaskRCNNParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
        srand((int) time(0));
    }

    bool build();

    bool infer();

    bool teardown();

private:
    SampleMaskRCNNParams mParams;

    nvinfer1::Dims mInputDims;

    // original images
    std::vector<MaskRCNNUtils::PPM<uint8_t>> mOriginalPPMs;

    // processed images (resize + pad)
    std::vector<MaskRCNNUtils::PPM<uint8_t>> mPPMs;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvuffparser::IUffParser>& parser);

    bool processInput(const samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    vector<MaskRCNNUtils::BBoxInfo> decodeOutput(const int imageIdx, void* detectionsHost, void* masksHost);
};

bool SampleMaskRCNN::build()
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

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 2);

    return true;
}

bool SampleMaskRCNN::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvuffparser::IUffParser>& parser)
{
    parser->registerInput(
        mParams.inputTensorNames[0].c_str(), MaskRCNNConfig::IMAGE_SHAPE, nvuffparser::UffInputOrder::kNCHW);
    for (size_t i = 0; i < mParams.outputTensorNames.size(); i++)
        parser->registerOutput(mParams.outputTensorNames[i].c_str());

    auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(2_GiB);
    builder->setFp16Mode(mParams.fp16);

    // Only for speed test
    if (mParams.int8)
    {
        samplesCommon::setAllTensorScales(network.get());
        builder->setInt8Mode(true);
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

bool SampleMaskRCNN::infer()
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

    auto tStart = std::chrono::high_resolution_clock::now();
    bool status;
    for (int i = 0; i < 10; i++)
    {
        status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    gLogInfo << "Run for 10 times with Batch Size " << mParams.batchSize << std::endl;
    gLogInfo << "Average inference time is " << (totalHost / 10) / mParams.batchSize << " ms/frame" << std::endl;

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

bool SampleMaskRCNN::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

bool SampleMaskRCNN::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;

    // Available images
    std::vector<std::string> imageListCandidates = {"001763.ppm", "004545.ppm"};
    std::vector<std::string> imageList;
    for (int i = 0; i < batchSize; i++)
    {
        imageList.push_back(imageListCandidates[i % 2]);
    }

    mPPMs.resize(batchSize);
    mOriginalPPMs.resize(batchSize);
    assert(mPPMs.size() <= imageList.size());
    for (int i = 0; i < batchSize; ++i)
    {
        MaskRCNNUtils::readPPMFile(locateFile(imageList[i], mParams.dataDirs), mOriginalPPMs[i]);
        MaskRCNNUtils::preprocessPPM(mOriginalPPMs[i], mPPMs[i], inputH, inputW);
    }

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float pixelMean[3]{123.7, 116.8, 103.9};
    // Host memory for input buffer
    for (int i = 0, volImg = inputC * inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in RGB order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j] = float(mPPMs[i].buffer[j * inputC + c]) - pixelMean[c];
            }
        }
    }

    return true;
}

vector<MaskRCNNUtils::BBoxInfo> SampleMaskRCNN::decodeOutput(const int imageIdx, void* detectionsHost, void* masksHost)
{
    int input_dim_h = MaskRCNNConfig::IMAGE_SHAPE.d[1], input_dim_w = MaskRCNNConfig::IMAGE_SHAPE.d[2];
    assert(input_dim_h == input_dim_w);
    int image_height = mOriginalPPMs[imageIdx].h;
    int image_width = mOriginalPPMs[imageIdx].w;
    // resize the DsImage with scale
    const int image_dim = std::max(image_height, image_width);
    int resizeH = (int) image_height * input_dim_h / (float) image_dim;
    int resizeW = (int) image_width * input_dim_w / (float) image_dim;
    // keep accurary from (float) to (int), then to float
    float window_x = (1.0f - (float) resizeW / input_dim_w) / 2.0f;
    float window_y = (1.0f - (float) resizeH / input_dim_h) / 2.0f;
    float window_width = (float) resizeW / input_dim_w;
    float window_height = (float) resizeH / input_dim_h;

    float final_ratio_x = (float) image_width / window_width;
    float final_ratio_y = (float) image_height / window_height;

    std::vector<MaskRCNNUtils::BBoxInfo> binfo;

    int detectionOffset = samplesCommon::volume(MaskRCNNConfig::MODEL_DETECTION_SHAPE); // (100,6)
    int maskOffset = samplesCommon::volume(MaskRCNNConfig::MODEL_MASK_SHAPE);           // (100, 81, 28, 28)

    MaskRCNNUtils::RawDetection* detections
        = reinterpret_cast<MaskRCNNUtils::RawDetection*>((float*) detectionsHost + imageIdx * detectionOffset);
    MaskRCNNUtils::Mask* masks = reinterpret_cast<MaskRCNNUtils::Mask*>((float*) masksHost + imageIdx * maskOffset);
    for (int det_id = 0; det_id < MaskRCNNConfig::DETECTION_MAX_INSTANCES; det_id++)
    {
        MaskRCNNUtils::RawDetection cur_det = detections[det_id];
        int label = (int) cur_det.class_id;
        if (label <= 0)
            continue;

        MaskRCNNUtils::BBoxInfo det;
        det.label = label;
        det.prob = cur_det.score;

        det.box.x1 = std::min(std::max((cur_det.x1 - window_x) * final_ratio_x, 0.0f), (float) image_width);
        det.box.y1 = std::min(std::max((cur_det.y1 - window_y) * final_ratio_y, 0.0f), (float) image_height);
        det.box.x2 = std::min(std::max((cur_det.x2 - window_x) * final_ratio_x, 0.0f), (float) image_width);
        det.box.y2 = std::min(std::max((cur_det.y2 - window_y) * final_ratio_y, 0.0f), (float) image_height);

        if (det.box.x2 <= det.box.x1 || det.box.y2 <= det.box.y1)
            continue;

        det.mask = masks + det_id * MaskRCNNConfig::NUM_CLASSES + label;

        binfo.push_back(det);
    }

    return binfo;
}

bool SampleMaskRCNN::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    void* detectionsHost = buffers.getHostBuffer(mParams.outputTensorNames[0]);
    void* masksHost = buffers.getHostBuffer(mParams.outputTensorNames[1]);

    bool pass = true;

    for (int p = 0; p < mParams.batchSize; ++p)
    {
        vector<MaskRCNNUtils::BBoxInfo> binfo = decodeOutput(p, detectionsHost, masksHost);
        for (size_t roi_id = 0; roi_id < binfo.size(); roi_id++)
        {
            const auto resized_mask = MaskRCNNUtils::resizeMask(binfo[roi_id], mParams.maskThreshold); // mask threshold
            MaskRCNNUtils::addBBoxPPM(mOriginalPPMs[p], binfo[roi_id], resized_mask);

            gLogInfo << "Detected " << MaskRCNNConfig::CLASS_NAMES[binfo[roi_id].label] << " in"
                     << mOriginalPPMs[p].fileName << " with confidence " << binfo[roi_id].prob * 100.f
                     << " and coordinates (" << binfo[roi_id].box.x1 << ", " << binfo[roi_id].box.y1 << ", "
                     << binfo[roi_id].box.x2 << ", " << binfo[roi_id].box.y2 << ")" << std::endl;
        }
        gLogInfo << "The results are stored in current directory: " << std::to_string(p) + ".ppm" << std::endl;
        MaskRCNNUtils::writePPMFile(std::to_string(p) + ".ppm", mOriginalPPMs[p]);
    }

    return pass;
}

SampleMaskRCNNParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleMaskRCNNParams params;
    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("data/maskrcnn/");
        params.dataDirs.push_back("data/maskrcnn/images/");
        params.dataDirs.push_back("data/samples/maskrcnn/");
        params.dataDirs.push_back("data/samples/maskrcnn/images/");
    }
    else
    {
        params.dataDirs = args.dataDirs;
    }

    params.inputTensorNames.push_back(MaskRCNNConfig::MODEL_INPUT);
    params.batchSize = args.batch;
    params.outputTensorNames.push_back(MaskRCNNConfig::MODEL_OUTPUTS[0]);
    params.outputTensorNames.push_back(MaskRCNNConfig::MODEL_OUTPUTS[1]);
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.uffFileName = MaskRCNNConfig::MODEL_NAME;
    params.maskThreshold = MaskRCNNConfig::MASK_THRESHOLD;

    return params;
}

void printHelpInfo()
{
    std::cout << "Usage: ./sample_maskRCNN [-h or --help] [-d or --datadir=<path to data directory>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/maskrcnn/ and data/maskrcnn/"
              << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--batch         Specify inference batch size." << std::endl;
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

    SampleMaskRCNN sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for Mask RCNN" << std::endl;

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
