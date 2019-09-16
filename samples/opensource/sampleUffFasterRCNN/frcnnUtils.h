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

#ifndef FRCNN_UTILS_H
#define FRCNN_UTILS_H

#include "NvInfer.h"
#include "argsParser.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

using namespace samplesCommon;

//! \brief Split a string at a delimiter(defaults to comma).
//!
void splitStr(const char* s, std::vector<std::string>& ret, char del = ',')
{
    int idx = 0;
    while (std::string::npos != std::string(s + idx).find(std::string(1, del)))
    {
        auto s_tmp = std::string(s + idx).substr(0, std::string(s + idx).find(std::string(1, del)));
        ret.push_back(s_tmp);
        idx += std::string(s + idx).find(std::string(1, del)) + 1;
    }
    if (s[idx] != 0)
    {
        ret.push_back(std::string(s + idx));
    }
}

//! \class
//!
//! \brief The command line arguments for this sample.
//!
struct FrcnnArgs
{
    bool runInInt8{false};
    bool runInFp16{false};
    bool help{false};
    int useDLACore{-1};
    std::vector<std::string> dataDirs;
    int inputHeight;
    int inputWidth;
    int repeat{1};
    bool profile{false};
    int batchSize{1};
    std::vector<std::string> inputImages;
    std::string saveEngine{""};
    std::string loadEngine{""};
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
bool parseFrcnnArgs(FrcnnArgs& args, int argc, char* argv[])
{
    while (1)
    {
        int arg;
        static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"datadir", required_argument, 0, 'd'},
            {"int8", no_argument, 0, 'i'}, {"fp16", no_argument, 0, 'f'}, {"useDLACore", required_argument, 0, 'u'},
            {"inputHeight", required_argument, 0, 'H'}, {"inputWidth", required_argument, 0, 'w'},
            {"repeat", required_argument, 0, 'r'}, {"profile", no_argument, 0, 'p'},
            {"batchSize", required_argument, 0, 'b'}, {"inputImages", required_argument, 0, 'I'},
            {"saveEngine", required_argument, 0, 's'}, {"loadEngine", required_argument, 0, 'l'},
            {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:ifuH:W:r:pB:I:s:l:", long_options, &option_index);

        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h': args.help = true; return true;

        case 'd':
            if (optarg)
            {
                args.dataDirs.push_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }

            break;

        case 'i': args.runInInt8 = true; break;

        case 'f': args.runInFp16 = true; break;

        case 'u':
            if (optarg)
            {
                args.useDLACore = std::stoi(optarg);
            }

            break;

        case 'H': args.inputHeight = std::stoi(optarg); break;

        case 'W': args.inputWidth = std::stoi(optarg); break;

        case 'r': args.repeat = std::stoi(optarg); break;

        case 'p': args.profile = true; break;

        case 'B': args.batchSize = std::stoi(optarg); break;

        case 'I': splitStr(optarg, args.inputImages); break;
        case 's': args.saveEngine = optarg; break;
        case 'l': args.loadEngine = optarg; break;
        default: return false;
        }
    }

    return true;
}

//! \brief resize PPM on-the-fly so that user can specify input dimensions as commandline args.
//!
void resizePPM(vPPM& ppm, int target_width, int target_height)
{
    auto clip = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
    int original_height = ppm.h;
    int original_width = ppm.w;
    ppm.h = target_height;
    ppm.w = target_width;
    float ratio_h = static_cast<float>(original_height - 1.0f) / static_cast<float>(target_height - 1.0f);
    float ratio_w = static_cast<float>(original_width - 1.0f) / static_cast<float>(target_width - 1.0f);
    std::vector<uint8_t> tmp_buf;

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

            for (int c = 0; c < 3; ++c)
            {
                // H, W, C ordering
                uint8_t left_top_val = ppm.buffer[top * (original_width * 3) + left * (3) + c];
                uint8_t right_top_val = ppm.buffer[top * (original_width * 3) + right * (3) + c];
                uint8_t left_bottom_val = ppm.buffer[bottom * (original_width * 3) + left * (3) + c];
                uint8_t right_bottom_val = ppm.buffer[bottom * (original_width * 3) + right * (3) + c];
                float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
                float bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
                float lerp = clip(std::round(top_lerp + (bottom_lerp - top_lerp) * (y0 - top)), 0.0f, 255.0f);
                tmp_buf.push_back(static_cast<uint8_t>(lerp));
            }
        }
    }

    ppm.buffer = tmp_buf;
}

//! \class BatchStream
//!
//! \brief Custom BatchStream class for Faster-RCNN because we use variable input dimensions and different image
//! preprocessing.
//!
class BatchStream
{
public:
    BatchStream(
        int batchSize, int maxBatches, std::string prefix, std::string suffix, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mPrefix(prefix)
        , mSuffix(suffix)
        , mDataDir(directories)
    {
        FILE* file = fopen(locateFile(mPrefix + std::string("0") + mSuffix, mDataDir).c_str(), "rb");
        assert(file != nullptr);
        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  // The number of dimensions.
        mDims.d[0] = d[0]; // Batch Size
        mDims.d[1] = d[1]; // Channels
        mDims.d[2] = d[2]; // Height
        mDims.d[3] = d[3]; // Width
        assert(mDims.d[0] > 0 && mDims.d[1] > 0 && mDims.d[2] > 0 && mDims.d[3] > 0);
        fclose(file);
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    BatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories)
        : BatchStream(batchSize, maxBatches, prefix, ".batch", directories)
    {
    }

    BatchStream(
        int batchSize, int maxBatches, nvinfer1::Dims dims, std::string listFile, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mDims(dims)
        , mListFile(listFile)
        , mDataDir(directories)
    {
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next()
    {
        if (mBatchCount == mMaxBatches)
        {
            return false;
        }

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);

            if (mFileBatchPos == mDims.d[0] && !update())
            {
                return false;
            }

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(
                getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }

        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;

        for (int i = 0; i < skipCount; i++)
        {
            next();
        }

        mBatchCount = x;
    }

    float* getBatch()
    {
        return mBatch.data();
    }

    float* getLabels()
    {
        return mLabels.data();
    }

    int getBatchesRead() const
    {
        return mBatchCount;
    }

    int getBatchSize() const
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const
    {
        return mDims;
    }

private:
    float* getFileBatch()
    {
        return mFileBatch.data();
    }

    float* getFileLabels()
    {
        return mFileLabels.data();
    }

    bool update()
    {
        if (mListFile.empty())
        {
            std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + mSuffix, mDataDir);
            FILE* file = fopen(inputFileName.c_str(), "rb");

            if (!file)
            {
                return false;
            }

            int d[4];
            size_t readSize = fread(d, sizeof(int), 4, file);
            assert(readSize == 4);
            assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
            size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
            assert(readInputCount == size_t(mDims.d[0] * mImageSize));
            size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.d[0], file);
            assert(readLabelCount == 0 || readLabelCount == size_t(mDims.d[0]));
            fclose(file);
        }
        else
        {
            std::vector<std::string> fNames;
            std::ifstream file(locateFile(mListFile, mDataDir), std::ios::binary);

            if (!file)
            {
                return false;
            }

            gLogInfo << "Batch #" << mFileCount << std::endl;
            file.seekg(((mBatchCount * mBatchSize)) * 7);

            for (int i = 1; i <= mBatchSize; i++)
            {
                std::string sName;
                std::getline(file, sName);
                sName = sName + ".ppm";
                gLogInfo << "Calibrating with file " << sName << std::endl;
                fNames.emplace_back(sName);
            }

            mFileCount++;
            std::vector<vPPM> ppms(fNames.size());

            for (uint32_t i = 0; i < fNames.size(); ++i)
            {
                readPPMFile(fNames[i], ppms[i], mDataDir);
            }

            std::vector<float> data(samplesCommon::volume(mDims));
            // Normalize input data
            float pixelMean[3]{103.939, 116.779, 123.68};

            for (int i = 0, volImg = mDims.d[1] * mDims.d[2] * mDims.d[3]; i < mBatchSize; ++i)
            {
                for (int c = 0; c < mDims.d[1]; ++c)
                {
                    for (unsigned j = 0, volChl = mDims.d[2] * mDims.d[3]; j < volChl; ++j)
                    {
                        data[i * volImg + c * volChl + j]
                            = float(ppms[i].buffer[j * mDims.d[1] + 2 - c]) - pixelMean[c];
                    }
                }
            }

            std::copy_n(data.data(), mDims.d[0] * mImageSize, getFileBatch());
        }

        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    std::vector<float> mBatch;         //!< Data for the batch
    std::vector<float> mLabels;        //!< Labels for the batch
    std::vector<float> mFileBatch;     //!< List of image files
    std::vector<float> mFileLabels;    //!< List of label files
    std::string mPrefix;               //!< Batch file name prefix
    std::string mSuffix;               //!< Batch file name suffix
    nvinfer1::Dims mDims;              //!< Input dimensions
    std::string mListFile;             //!< File name of the list of image names
    std::vector<std::string> mDataDir; //!< Directories where the files can be found
};

//! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(
        BatchStream& stream, int firstBatch, std::string networkName, const char* inputBlobName, bool readCache = true)
        : mStream(stream)
        , mCalibrationTableName("CalibrationTable" + networkName)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = samplesCommon::volume(dims);
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~EntropyCalibratorImpl()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const
    {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        if (!mStream.next())
        {
            return false;
        }

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;

        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    BatchStream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(
        BatchStream& stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
    }

    int getBatchSize() const override
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) override
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl mImpl;
};

#endif
