/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
public:
    virtual ~IOnnxConfig() noexcept = default;
    //!
    //! \typedef Verbosity
    //! \brief Defines Verbosity level.
    //!
    typedef int32_t Verbosity;

    //!
    //! \brief Set the Model Data Type.
    //!
    //! Sets the Model DataType, one of the following: float -d 32 (default), half precision -d 16, and int8 -d 8 data
    //! types.
    //!
    //! \see getModelDtype()
    //!
    virtual void setModelDtype(const nvinfer1::DataType) noexcept = 0;

    //!
    //! \brief Get the Model Data Type.
    //!
    //! \return DataType nvinfer1::DataType
    //!
    //! \see setModelDtype() and #DataType
    //!
    virtual nvinfer1::DataType getModelDtype() const noexcept = 0;

    //!
    //! \brief Get the Model FileName.
    //!
    //! \return Return the Model Filename, as a null-terminated C-style string.
    //!
    //! \see setModelFileName()
    //!
    virtual char const* getModelFileName() const noexcept = 0;

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
    virtual void setModelFileName(char const* onnxFilename) noexcept = 0;

    //!
    //! \brief Get the Verbosity Level.
    //!
    //! \return The Verbosity Level.
    //!
    //! \see addVerbosity(), reduceVerbosity()
    //!
    virtual Verbosity getVerbosityLevel() const noexcept = 0;

    //!
    //! \brief Increase the Verbosity Level.
    //!
    //! \return The Verbosity Level.
    //!
    //! \see reduceVerbosity(), setVerbosity(Verbosity)
    //!
    virtual void addVerbosity() noexcept = 0;

    //!
    //! \brief Reduce the Verbosity Level.
    //!
    //! \see addVerbosity(), setVerbosity(Verbosity)
    //!
    virtual void reduceVerbosity() noexcept = 0;

    //!
    //! \brief Set to specific verbosity Level.
    //!
    //! \see addVerbosity(), reduceVerbosity()
    //!
    virtual void setVerbosityLevel(Verbosity) noexcept = 0;

    //!
    //! \brief Returns the File Name of the Network Description as a Text File.
    //!
    //! \return Return the name of the file containing the network description converted to a plain text, used for
    //! debugging purposes.
    //!
    //! \see setTextFilename()
    //!
    virtual char const* getTextFileName() const noexcept = 0;

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
    virtual void setTextFileName(char const* textFileName) noexcept = 0;

    //!
    //! \brief Get the File Name of the Network Description as a Text File, including the weights.
    //!
    //! \return Return the name of the file containing the network description converted to a plain text, used for
    //! debugging purposes.
    //!
    //! \see setFullTextFilename()
    //!
    virtual char const* getFullTextFileName() const noexcept = 0;

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
    virtual void setFullTextFileName(char const* fullTextFileName) noexcept = 0;

    //!
    //! \brief Get whether the layer information will be printed.
    //!
    //! \return Returns whether the layer information will be printed.
    //!
    //! \see setPrintLayerInfo()
    //!
    virtual bool getPrintLayerInfo() const noexcept = 0;

    //!
    //! \brief Set whether the layer information will be printed.
    //!
    //! \see getPrintLayerInfo()
    //!
    virtual void setPrintLayerInfo(bool) noexcept = 0;

    //!
    //! \brief Destroy IOnnxConfig object.
    //!
    //! \deprecated Use `delete` instead. Deprecated in TRT 8.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED virtual void destroy() noexcept = 0;

}; // class IOnnxConfig

TENSORRTAPI IOnnxConfig* createONNXConfig();

} // namespace nvonnxparser

#endif
