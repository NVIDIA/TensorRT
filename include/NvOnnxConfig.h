/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef NV_OnnxConfig_H
#define NV_OnnxConfig_H

#include "NvInfer.h"

namespace nvonnxparser
{

//!
//! \mainpage
//!
//! This is the API documentation for the Configuration Manager for Open Neural Network Exchange (ONNX) Parser for Nvidia TensorRT Inference Engine.
//! It provides information on individual functions, classes
//! and methods. Use the index on the left to navigate the documentation.
//!
//! Please see the accompanying user guide and samples for higher-level information and general advice on using ONNX Parser and TensorRT.
//!

//!
//! \file NvOnnxConfig.h
//!
//! This is the API file for the Configuration Manager for ONNX Parser for Nvidia TensorRT.
//!

//!
//! \class IOnnxConfig
//! \brief Configuration Manager Class.
//!
class IOnnxConfig
{
protected:
    virtual ~IOnnxConfig() {}

public:
    //!
    //! \typedef Verbosity
    //! \brief Defines Verbosity level.
    //!
    typedef int Verbosity;

    //!
    //! \brief Set the Model Data Type.
    //!
    //! Sets the Model DataType, one of the following: float -d 32 (default), half precision -d 16, and int8 -d 8 data types.
    //!
    //! \see getModelDtype()
    //!
    virtual void setModelDtype(const nvinfer1::DataType) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the Model Data Type.
    //!
    //! \return DataType nvinfer1::DataType
    //!
    //! \see setModelDtype() and #DataType
    //!
    virtual nvinfer1::DataType getModelDtype() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the Model FileName.
    //!
    //! \return Return the Model Filename, as a pointer to a NULL-terminated character sequence.
    //!
    //! \see setModelFileName()
    //!
    virtual const char* getModelFileName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the Model File Name.
    //!
    //! The Model File name contains the Network Description in ONNX pb format.
    //!
    //! This method copies the name string.
    //!
    //! \param onnxFilename The name.
    //!
    //! \see getModelFileName()
    //!
    virtual void setModelFileName(const char* onnxFilename) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the Verbosity Level.
    //!
    //! \return The Verbosity Level.
    //!
    //! \see addVerbosity(), reduceVerbosity()
    //!
    virtual Verbosity getVerbosityLevel() const TRTNOEXCEPT = 0;

    //!
    //! \brief Increase the Verbosity Level.
    //!
    //! \return The Verbosity Level.
    //!
    //! \see addVerbosity(), reduceVerbosity(), setVerbosity(Verbosity)
    //!
    virtual void addVerbosity() TRTNOEXCEPT = 0;               //!< Increase verbosity Level.
    virtual void reduceVerbosity() TRTNOEXCEPT = 0;            //!< Decrease verbosity Level.
    virtual void setVerbosityLevel(Verbosity) TRTNOEXCEPT = 0; //!< Set to specific verbosity Level.

    //!
    //! \brief Returns the File Name of the Network Description as a Text File.
    //!
    //! \return Return the name of the file containing the network description converted to a plain text, used for debugging purposes.
    //!
    //! \see setTextFilename()
    //!
    virtual const char* getTextFileName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the File Name of the Network Description as a Text File.
    //!
    //! This API allows setting a file name for the network description in plain text, equivalent of the ONNX protobuf.
    //!
    //! This method copies the name string.
    //!
    //! \param textFileName Name of the file.
    //!
    //! \see getTextFilename()
    //!
    virtual void setTextFileName(const char* textFileName) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the File Name of the Network Description as a Text File, including the weights.
    //!
    //! \return Return the name of the file containing the network description converted to a plain text, used for debugging purposes.
    //!
    //! \see setFullTextFilename()
    //!
    virtual const char* getFullTextFileName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the File Name of the Network Description as a Text File, including the weights.
    //!
    //! This API allows setting a file name for the network description in plain text, equivalent of the ONNX protobuf.
    //!
    //! This method copies the name string.
    //!
    //! \param fullTextFileName Name of the file.
    //!
    //! \see getFullTextFilename()
    //!
    virtual void setFullTextFileName(const char* fullTextFileName) TRTNOEXCEPT = 0;

    //!
    //! \brief Get whether the layer information will be printed.
    //!
    //! \return Returns whether the layer information will be printed.
    //!
    //! \see setPrintLayerInfo()
    //!
    virtual bool getPrintLayerInfo() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether the layer information will be printed.
    //!
    //! \see getPrintLayerInfo()
    //!
    virtual void setPrintLayerInfo(bool) TRTNOEXCEPT = 0;

    //!
    //! \brief Destroy IOnnxConfig object.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

}; // class IOnnxConfig

TENSORRTAPI IOnnxConfig* createONNXConfig();

} // namespace nvonnxparser

#endif
