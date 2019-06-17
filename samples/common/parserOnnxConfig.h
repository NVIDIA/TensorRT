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

#ifndef PARSER_ONNX_CONFIG_H
#define PARSER_ONNX_CONFIG_H

#include <cstring>
#include <iostream>
#include <string>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

#define ONNX_DEBUG 1

/**
 * \class ParserOnnxConfig
 * \brief Configuration Manager Class Concrete Implementation
 *
 * \note:
 *
 */

using namespace std;

class ParserOnnxConfig : public nvonnxparser::IOnnxConfig
{

protected:
    string mModelFilename{};
    string mTextFilename{};
    string mFullTextFilename{};
    nvinfer1::DataType mModelDtype;
    nvonnxparser::IOnnxConfig::Verbosity mVerbosity;
    bool mPrintLayercInfo;

public:
    ParserOnnxConfig()
        : mModelDtype(nvinfer1::DataType::kFLOAT)
        , mVerbosity(static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))
        , mPrintLayercInfo(false)
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << " ParserOnnxConfig::ctor(): " << this << "\t" << std::endl;
        }
#endif
    }

protected:
    ~ParserOnnxConfig()
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << "ParserOnnxConfig::dtor(): " << this << std::endl;
        }
#endif
    }

public:
    virtual void setModelDtype(const nvinfer1::DataType modelDtype)
    {
        mModelDtype = modelDtype;
    }

    virtual nvinfer1::DataType getModelDtype() const
    {
        return mModelDtype;
    }

    virtual const char* getModelFileName() const
    {
        return mModelFilename.c_str();
    }
    virtual void setModelFileName(const char* onnxFilename)
    {
        mModelFilename = string(onnxFilename);
    }
    virtual nvonnxparser::IOnnxConfig::Verbosity getVerbosityLevel() const
    {
        return mVerbosity;
    }
    virtual void addVerbosity()
    {
        ++mVerbosity;
    }
    virtual void reduceVerbosity()
    {
        --mVerbosity;
    }
    virtual void setVerbosityLevel(nvonnxparser::IOnnxConfig::Verbosity verbosity)
    {
        mVerbosity = verbosity;
    }

    virtual const char* getTextFileName() const
    {
        return mTextFilename.c_str();
    }
    virtual void setTextFileName(const char* textFilename)
    {
        mTextFilename = string(textFilename);
    }
    virtual const char* getFullTextFileName() const
    {
        return mFullTextFilename.c_str();
    }
    virtual void setFullTextFileName(const char* fullTextFilename)
    {
        mFullTextFilename = string(fullTextFilename);
    }
    virtual bool getPrintLayerInfo() const
    {
        return mPrintLayercInfo;
    }
    virtual void setPrintLayerInfo(bool src)
    {
        mPrintLayercInfo = src;
    } //!< get the boolean variable corresponding to the Layer Info, see getPrintLayerInfo()

    virtual bool isDebug() const
    {
#if ONNX_DEBUG
        return (std::getenv("ONNX_DEBUG") ? true : false);
#else
        return false;
#endif
    }

    virtual void destroy()
    {
        delete this;
    }

}; // class ParserOnnxConfig

#endif
