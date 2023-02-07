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

#ifndef SampleConfig_H
#define SampleConfig_H

#include <cstring>
#include <iostream>
#include <string>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
class SampleConfig : public nvonnxparser::IOnnxConfig
{
public:
    enum class InputDataFormat : int
    {
        kASCII = 0,
        kPPM = 1
    };

private:
    std::string mModelFilename;
    std::string mEngineFilename;
    std::string mTextFilename;
    std::string mFullTextFilename;
    std::string mImageFilename;
    std::string mReferenceFilename;
    std::string mOutputFilename;
    std::string mCalibrationFilename;
    std::string mTimingCacheFilename;
    int64_t mLabel{-1};
    int64_t mMaxBatchSize{32};
    int64_t mCalibBatchSize{0};
    int64_t mMaxNCalibBatch{0};
    int64_t mFirstCalibBatch{0};
    int64_t mUseDLACore{-1};
    nvinfer1::DataType mModelDtype{nvinfer1::DataType::kFLOAT};
    bool mTF32{true};
    Verbosity mVerbosity{static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)};
    bool mPrintLayercInfo{false};
    bool mDebugBuilder{false};
    InputDataFormat mInputDataFormat{InputDataFormat::kASCII};
    uint64_t mTopK{0};
    float mFailurePercentage{-1.0f};
    float mTolerance{0.0f};
    float mAbsTolerance{1e-5f};

public:
    SampleConfig()
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << " SampleConfig::ctor(): " << this << "\t" << std::endl;
        }
#endif
    }

protected:
    ~SampleConfig()
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << "SampleConfig::dtor(): " << this << std::endl;
        }
#endif
    }

public:
    void setModelDtype(const nvinfer1::DataType mdt) noexcept
    {
        mModelDtype = mdt;
    }

    nvinfer1::DataType getModelDtype() const noexcept
    {
        return mModelDtype;
    }

    bool getTF32() const noexcept
    {
        return mTF32;
    }

    void setTF32(bool enabled) noexcept
    {
        mTF32 = enabled;
    }

    const char* getModelFileName() const noexcept
    {
        return mModelFilename.c_str();
    }

    void setModelFileName(const char* onnxFilename) noexcept
    {
        mModelFilename = std::string(onnxFilename);
    }
    Verbosity getVerbosityLevel() const noexcept
    {
        return mVerbosity;
    }
    void addVerbosity() noexcept
    {
        ++mVerbosity;
    }
    void reduceVerbosity() noexcept
    {
        --mVerbosity;
    }
    virtual void setVerbosityLevel(Verbosity v) noexcept
    {
        mVerbosity = v;
    }
    const char* getEngineFileName() const noexcept
    {
        return mEngineFilename.c_str();
    }
    void setEngineFileName(const char* engineFilename) noexcept
    {
        mEngineFilename = std::string(engineFilename);
    }
    const char* getTextFileName() const noexcept
    {
        return mTextFilename.c_str();
    }
    void setTextFileName(const char* textFilename) noexcept
    {
        mTextFilename = std::string(textFilename);
    }
    const char* getFullTextFileName() const noexcept
    {
        return mFullTextFilename.c_str();
    }
    void setFullTextFileName(const char* fullTextFilename) noexcept
    {
        mFullTextFilename = std::string(fullTextFilename);
    }
    void setLabel(int64_t label) noexcept
    {
        mLabel = label;
    } //!<  set the Label

    int64_t getLabel() const noexcept
    {
        return mLabel;
    } //!<  get the Label

    bool getPrintLayerInfo() const noexcept
    {
        return mPrintLayercInfo;
    }

    void setPrintLayerInfo(bool b) noexcept
    {
        mPrintLayercInfo = b;
    } //!< get the boolean variable corresponding to the Layer Info, see getPrintLayerInfo()

    void setMaxBatchSize(int64_t maxBatchSize) noexcept
    {
        mMaxBatchSize = maxBatchSize;
    } //!<  set the Max Batch Size
    int64_t getMaxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    } //!<  get the Max Batch Size

    void setCalibBatchSize(int64_t CalibBatchSize) noexcept
    {
        mCalibBatchSize = CalibBatchSize;
    } //!<  set the calibration batch size
    int64_t getCalibBatchSize() const noexcept
    {
        return mCalibBatchSize;
    } //!<  get calibration batch size

    void setMaxNCalibBatch(int64_t MaxNCalibBatch) noexcept
    {
        mMaxNCalibBatch = MaxNCalibBatch;
    } //!<  set Max Number of Calibration Batches
    int64_t getMaxNCalibBatch() const noexcept
    {
        return mMaxNCalibBatch;
    } //!<  get the Max Number of Calibration Batches

    void setFirstCalibBatch(int64_t FirstCalibBatch) noexcept
    {
        mFirstCalibBatch = FirstCalibBatch;
    } //!<  set the first calibration batch
    int64_t getFirstCalibBatch() const noexcept
    {
        return mFirstCalibBatch;
    } //!<  get the first calibration batch

    void setUseDLACore(int64_t UseDLACore) noexcept
    {
        mUseDLACore = UseDLACore;
    } //!<  set the DLA core to use
    int64_t getUseDLACore() const noexcept
    {
        return mUseDLACore;
    } //!<  get the DLA core to use

    void setDebugBuilder() noexcept
    {
        mDebugBuilder = true;
    } //!<  enable the Debug info, while building the engine.
    bool getDebugBuilder() const noexcept
    {
        return mDebugBuilder;
    } //!<  get the boolean variable, corresponding to the debug builder

    const char* getImageFileName() const noexcept //!<  set Image file name (PPM or ASCII)
    {
        return mImageFilename.c_str();
    }
    void setImageFileName(const char* imageFilename) noexcept //!< get the Image file name
    {
        mImageFilename = std::string(imageFilename);
    }
    const char* getReferenceFileName() const noexcept
    {
        return mReferenceFilename.c_str();
    }
    void setReferenceFileName(const char* referenceFilename) noexcept //!<  set reference file name
    {
        mReferenceFilename = std::string(referenceFilename);
    }

    void setInputDataFormat(InputDataFormat idt) noexcept
    {
        mInputDataFormat = idt;
    } //!<  specifies expected data format of the image file (PPM or ASCII)
    InputDataFormat getInputDataFormat() const noexcept
    {
        return mInputDataFormat;
    } //!<  returns the expected data format of the image file.

    const char* getOutputFileName() const noexcept //!<  specifies the file to save the results
    {
        return mOutputFilename.c_str();
    }
    void setOutputFileName(const char* outputFilename) noexcept //!<  get the output file name
    {
        mOutputFilename = std::string(outputFilename);
    }

    const char* getCalibrationFileName() const noexcept
    {
        return mCalibrationFilename.c_str();
    } //!<  specifies the file containing the list of image files for int8 calibration
    void setCalibrationFileName(const char* calibrationFilename) noexcept //!<  get the int 8 calibration list file name
    {
        mCalibrationFilename = std::string(calibrationFilename);
    }

    uint64_t getTopK() const noexcept
    {
        return mTopK;
    }
    void setTopK(uint64_t topK) noexcept
    {
        mTopK = topK;
    } //!<  If this options is specified, return the K top probabilities.

    float getFailurePercentage() const noexcept
    {
        return mFailurePercentage;
    }

    void setFailurePercentage(float f) noexcept
    {
        mFailurePercentage = f;
    }

    float getAbsoluteTolerance() const noexcept
    {
        return mAbsTolerance;
    }

    void setAbsoluteTolerance(float a) noexcept
    {
        mAbsTolerance = a;
    }

    float getTolerance() const noexcept
    {
        return mTolerance;
    }

    void setTolerance(float t) noexcept
    {
        mTolerance = t;
    }

    const char* getTimingCacheFilename() const noexcept
    {
        return mTimingCacheFilename.c_str();
    }
    
    void setTimingCacheFileName(const char* timingCacheFilename) noexcept
    {
        mTimingCacheFilename = std::string(timingCacheFilename);
    }

    bool isDebug() const noexcept
    {
#if ONNX_DEBUG
        return (std::getenv("ONNX_DEBUG") ? true : false);
#else
        return false;
#endif
    }

    void destroy() noexcept
    {
        delete this;
    }

}; // class SampleConfig

#endif
