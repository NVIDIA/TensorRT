/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class ParserOnnxConfig : public nvonnxparser::IOnnxConfig
{

protected:
    std::string mModelFilename{};
    std::string mTextFilename{};
    std::string mFullTextFilename{};
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
    ~ParserOnnxConfig() override
    {
#ifdef ONNX_DEBUG
        if (isDebug())
        {
            std::cout << "ParserOnnxConfig::dtor(): " << this << std::endl;
        }
#endif
    }

public:
    void setModelDtype(const nvinfer1::DataType modelDtype) noexcept override
    {
        mModelDtype = modelDtype;
    }

    nvinfer1::DataType getModelDtype() const noexcept override
    {
        return mModelDtype;
    }

    const char* getModelFileName() const noexcept override
    {
        return mModelFilename.c_str();
    }
    void setModelFileName(const char* onnxFilename) noexcept override
    {
        mModelFilename = std::string(onnxFilename);
    }
    nvonnxparser::IOnnxConfig::Verbosity getVerbosityLevel() const noexcept override
    {
        return mVerbosity;
    }
    void addVerbosity() noexcept override
    {
        ++mVerbosity;
    }
    void reduceVerbosity() noexcept override
    {
        --mVerbosity;
    }
    void setVerbosityLevel(nvonnxparser::IOnnxConfig::Verbosity verbosity) noexcept override
    {
        mVerbosity = verbosity;
    }

    const char* getTextFileName() const noexcept override
    {
        return mTextFilename.c_str();
    }
    void setTextFileName(const char* textFilename) noexcept override
    {
        mTextFilename = std::string(textFilename);
    }
    const char* getFullTextFileName() const noexcept override
    {
        return mFullTextFilename.c_str();
    }
    void setFullTextFileName(const char* fullTextFilename) noexcept override
    {
        mFullTextFilename = std::string(fullTextFilename);
    }
    bool getPrintLayerInfo() const noexcept override
    {
        return mPrintLayercInfo;
    }
    void setPrintLayerInfo(bool src) noexcept override
    {
        mPrintLayercInfo = src;
    } //!< get the boolean variable corresponding to the Layer Info, see getPrintLayerInfo()

    virtual bool isDebug() const noexcept
    {
#if ONNX_DEBUG
        return (std::getenv("ONNX_DEBUG") ? true : false);
#else
        return false;
#endif
    }

    void destroy() noexcept override
    {
        delete this;
    }

}; // class ParserOnnxConfig

#endif
