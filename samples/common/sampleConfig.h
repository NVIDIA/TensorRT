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
    int64_t mMaxBatchSize{32};
    int64_t mMaxWorkspaceSize{1 * 1024 * 1024 * 1024};
    int64_t mCalibBatchSize{0};
    int64_t mMaxNCalibBatch{0};
    int64_t mFirstCalibBatch{0};
    int64_t mUseDLACore{-1};
    nvinfer1::DataType mModelDtype{nvinfer1::DataType::kFLOAT};
    Verbosity mVerbosity{static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)};
    bool mPrintLayercInfo{false};
    bool mDebugBuilder{false};
    InputDataFormat mInputDataFormat{InputDataFormat::kASCII};
    uint64_t mTopK{0};
    float mFailurePercentage{-1.0f};

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
    void setModelDtype(const nvinfer1::DataType mdt)
    {
        mModelDtype = mdt;
    }

    nvinfer1::DataType getModelDtype() const
    {
        return mModelDtype;
    }

    const char* getModelFileName() const
    {
        return mModelFilename.c_str();
    }

    void setModelFileName(const char* onnxFilename)
    {
        mModelFilename = string(onnxFilename);
    }
    Verbosity getVerbosityLevel() const
    {
        return mVerbosity;
    }
    void addVerbosity()
    {
        ++mVerbosity;
    }
    void reduceVerbosity()
    {
        --mVerbosity;
    }
    virtual void setVerbosityLevel(Verbosity v)
    {
        mVerbosity = v;
    }
    const char* getEngineFileName() const
    {
        return mEngineFilename.c_str();
    }
    void setEngineFileName(const char* engineFilename)
    {
        mEngineFilename = string(engineFilename);
    }
    const char* getTextFileName() const
    {
        return mTextFilename.c_str();
    }
    void setTextFileName(const char* textFilename)
    {
        mTextFilename = string(textFilename);
    }
    const char* getFullTextFileName() const
    {
        return mFullTextFilename.c_str();
    }
    void setFullTextFileName(const char* fullTextFilename)
    {
        mFullTextFilename = string(fullTextFilename);
    }
    bool getPrintLayerInfo() const
    {
        return mPrintLayercInfo;
    }
    void setPrintLayerInfo(bool b)
    {
        mPrintLayercInfo = b;
    } //!< get the boolean variable corresponding to the Layer Info, see getPrintLayerInfo()

    void setMaxBatchSize(int64_t maxBatchSize)
    {
        mMaxBatchSize = maxBatchSize;
    } //!<  set the Max Batch Size
    int64_t getMaxBatchSize() const
    {
        return mMaxBatchSize;
    } //!<  get the Max Batch Size

    void setMaxWorkSpaceSize(int64_t maxWorkSpaceSize)
    {
        mMaxWorkspaceSize = maxWorkSpaceSize;
    } //!<  set the Max Work Space size
    int64_t getMaxWorkSpaceSize() const
    {
        return mMaxWorkspaceSize;
    } //!<  get the Max Work Space size

    void setCalibBatchSize(int64_t CalibBatchSize)
    {
        mCalibBatchSize = CalibBatchSize;
    } //!<  set the calibration batch size
    int64_t getCalibBatchSize() const
    {
        return mCalibBatchSize;
    } //!<  get calibration batch size

    void setMaxNCalibBatch(int64_t MaxNCalibBatch)
    {
        mMaxNCalibBatch = MaxNCalibBatch;
    } //!<  set Max Number of Calibration Batches
    int64_t getMaxNCalibBatch() const
    {
        return mMaxNCalibBatch;
    } //!<  get the Max Number of Calibration Batches

    void setFirstCalibBatch(int64_t FirstCalibBatch)
    {
        mFirstCalibBatch = FirstCalibBatch;
    } //!<  set the first calibration batch
    int64_t getFirstCalibBatch() const
    {
        return mFirstCalibBatch;
    } //!<  get the first calibration batch

    void setUseDLACore(int64_t UseDLACore)
    {
        mUseDLACore = UseDLACore;
    } //!<  set the DLA core to use
    int64_t getUseDLACore() const
    {
        return mUseDLACore;
    } //!<  get the DLA core to use

    void setDebugBuilder()
    {
        mDebugBuilder = true;
    } //!<  enable the Debug info, while building the engine.
    bool getDebugBuilder() const
    {
        return mDebugBuilder;
    } //!<  get the boolean variable, corresponding to the debug builder

    const char* getImageFileName() const //!<  set Image file name (PPM or ASCII)
    {
        return mImageFilename.c_str();
    }
    void setImageFileName(const char* imageFilename) //!< get the Image file name
    {
        mImageFilename = string(imageFilename);
    }
    const char* getReferenceFileName() const
    {
        return mReferenceFilename.c_str();
    }
    void setReferenceFileName(const char* referenceFilename) //!<  set reference file name
    {
        mReferenceFilename = string(referenceFilename);
    }

    void setInputDataFormat(InputDataFormat idt)
    {
        mInputDataFormat = idt;
    } //!<  specifies expected data format of the image file (PPM or ASCII)
    InputDataFormat getInputDataFormat() const
    {
        return mInputDataFormat;
    } //!<  returns the expected data format of the image file.

    const char* getOutputFileName() const //!<  specifies the file to save the results
    {
        return mOutputFilename.c_str();
    }
    void setOutputFileName(const char* outputFilename) //!<  get the output file name
    {
        mOutputFilename = string(outputFilename);
    }

    const char* getCalibrationFileName() const
    {
        return mCalibrationFilename.c_str();
    } //!<  specifies the file containing the list of image files for int8 calibration
    void setCalibrationFileName(const char* calibrationFilename) //!<  get the int 8 calibration list file name
    {
        mCalibrationFilename = string(calibrationFilename);
    }

    uint64_t getTopK() const
    {
        return mTopK;
    }
    void setTopK(uint64_t topK)
    {
        mTopK = topK;
    } //!<  If this options is specified, return the K top probabilities.

    float getFailurePercentage() const
    {
        return mFailurePercentage;
    }

    void setFailurePercentage(float f)
    {
        mFailurePercentage = f;
    }

    bool isDebug() const
    {
#if ONNX_DEBUG
        return (std::getenv("ONNX_DEBUG") ? true : false);
#else
        return false;
#endif
    }

    void destroy()
    {
        delete this;
    }

}; // class SampleConfig

#endif
