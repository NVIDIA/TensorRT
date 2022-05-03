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

#ifndef NV_CAFFE_PARSER_H
#define NV_CAFFE_PARSER_H

#include "NvInfer.h"

//!
//! \file NvCaffeParser.h
//!
//! This is the API for the Caffe Parser
//!

//!
//! \namespace nvcaffeparser1
//!
//! \brief The TensorRT Caffe parser API namespace.
//!
namespace nvcaffeparser1
{

//!
//! \class IBlobNameToTensor
//!
//! \brief Object used to store and query Tensors after they have been extracted from a Caffe model using the ICaffeParser.
//!
//! \note The lifetime of IBlobNameToTensor is the same as the lifetime of its parent ICaffeParser.
//!
//! \see nvcaffeparser1::ICaffeParser
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IBlobNameToTensor
{
public:
    //! \brief Given a blob name, returns a pointer to a ITensor object.
    //!
    //! \param name Caffe blob name for which the user wants the corresponding ITensor.
    //!
    //! \return ITensor* corresponding to the queried name. If no such ITensor exists, then nullptr is returned.
    //!
    virtual nvinfer1::ITensor* find(char const* name) const noexcept = 0;

protected:
    virtual ~IBlobNameToTensor() {}
};

//!
//! \class IBinaryProtoBlob
//!
//! \brief Object used to store and query data extracted from a binaryproto file using the ICaffeParser.
//!
//! \see nvcaffeparser1::ICaffeParser
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IBinaryProtoBlob
{
public:
    virtual void const* getData() noexcept = 0;
    virtual nvinfer1::Dims4 getDimensions() noexcept = 0;
    virtual nvinfer1::DataType getDataType() noexcept = 0;
    //!
    //! \deprecated Use `delete` instead. Deprecated in TensorRT 8.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED virtual void destroy() noexcept = 0;
    virtual ~IBinaryProtoBlob() noexcept = default;
};

//!
//! \class IPluginFactoryV2
//!
//! \brief Plugin factory used to configure plugins.
//!
class IPluginFactoryV2
{
public:
    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPluginV2.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPluginV2(char const* layerName) noexcept = 0;

    //!
    //! \brief Creates a plugin.
    //!
    //! \param layerName Name of layer associated with the plugin.
    //! \param weights Weights used for the layer.
    //! \param nbWeights Number of weights.
    //! \param libNamespace Library Namespace associated with the plugin object
    //!
    virtual nvinfer1::IPluginV2* createPlugin(char const* layerName, nvinfer1::Weights const* weights,
        int32_t nbWeights, char const* libNamespace = "") noexcept = 0;

    virtual ~IPluginFactoryV2() noexcept = default;
};
//!
//! \class ICaffeParser
//!
//! \brief Class used for parsing Caffe models.
//!
//! Allows users to export models trained using Caffe to TRT.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICaffeParser
{
public:
    //!
    //! \brief Parse a prototxt file and a binaryproto Caffe model to extract
    //!   network definition and weights associated with the network, respectively.
    //!
    //! \param deploy The plain text, prototxt file used to define the network definition.
    //! \param model The binaryproto Caffe model that contains the weights associated with the network.
    //! \param network Network in which the CaffeParser will fill the layers.
    //! \param weightType The type to which the weights will transformed.
    //!
    //! \return A pointer to an IBlobNameToTensor object that contains the extracted data.
    //!
    //! \see nvcaffeparser1::IBlobNameToTensor
    //!
    virtual IBlobNameToTensor const* parse(char const* deploy, char const* model, nvinfer1::INetworkDefinition& network,
        nvinfer1::DataType weightType) noexcept = 0;

    //!
    //! \brief Parse a deploy prototxt and a binaryproto Caffe model from memory buffers to extract
    //!   network definition and weights associated with the network, respectively.
    //!
    //! \param deployBuffer The plain text deploy prototxt used to define the network definition.
    //! \param deployLength The length of the deploy buffer.
    //! \param modelBuffer The binaryproto Caffe memory buffer that contains the weights associated with the network.
    //! \param modelLength The length of the model buffer.
    //! \param network Network in which the CaffeParser will fill the layers.
    //! \param weightType The type to which the weights will transformed.
    //!
    //! \return A pointer to an IBlobNameToTensor object that contains the extracted data.
    //!
    //! \see nvcaffeparser1::IBlobNameToTensor
    //!
    virtual IBlobNameToTensor const* parseBuffers(uint8_t const* deployBuffer, std::size_t deployLength,
        uint8_t const* modelBuffer, std::size_t modelLength, nvinfer1::INetworkDefinition& network,
        nvinfer1::DataType weightType) noexcept = 0;

    //!
    //! \brief Parse and extract data stored in binaryproto file.
    //!
    //! The binaryproto file contains data stored in a binary blob. parseBinaryProto() converts it
    //! to an IBinaryProtoBlob object which gives the user access to the data and meta-data about data.
    //!
    //! \param fileName Path to file containing binary proto.
    //!
    //! \return A pointer to an IBinaryProtoBlob object that contains the extracted data.
    //!
    //! \see nvcaffeparser1::IBinaryProtoBlob
    //!
    virtual IBinaryProtoBlob* parseBinaryProto(char const* fileName) noexcept = 0;

    //!
    //! \brief Set buffer size for the parsing and storage of the learned model.
    //!
    //! \param size The size of the buffer specified as the number of bytes.
    //!
    //! \note  Default size is 2^30 bytes.
    //!
    virtual void setProtobufBufferSize(size_t size) noexcept = 0;

    //!
    //! \brief Destroy this ICaffeParser object.
    //!
    //! \deprecated Use `delete` instead. Deprecated in TensorRT 8.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED virtual void destroy() noexcept = 0;

    //!
    //! \brief Set the IPluginFactoryV2 used to create the user defined pluginV2 objects.
    //!
    //! \param factory Pointer to an instance of the user implementation of IPluginFactoryV2.
    //!
    virtual void setPluginFactoryV2(IPluginFactoryV2* factory) noexcept = 0;

    //!
    //! \brief Set the namespace used to lookup and create plugins in the network.
    //!
    virtual void setPluginNamespace(char const* libNamespace) noexcept = 0;

    virtual ~ICaffeParser() noexcept = default;

public:
    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //!
    //! \see getErrorRecorder()
    //!
    virtual void setErrorRecorder(nvinfer1::IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A
    //! nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    virtual nvinfer1::IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

//!
//! \brief Creates a ICaffeParser object.
//!
//! \return A pointer to the ICaffeParser object is returned.
//!
//! \see nvcaffeparser1::ICaffeParser
//!
//! \deprecated ICaffeParser will be removed in TensorRT 9.0. Plan to migrate your workflow to
//! use nvonnxparser::IParser for deployment.
//!
TENSORRTAPI ICaffeParser* createCaffeParser() noexcept;

//!
//! \brief Shuts down protocol buffers library.
//!
//! \note No part of the protocol buffers library can be used after this function is called.
//!
TENSORRTAPI void shutdownProtobufLibrary() noexcept;
} // namespace nvcaffeparser1

//!
//! Internal C entry point for creating ICaffeParser.
//! @private
//!
extern "C" TENSORRTAPI void* createNvCaffeParser_INTERNAL() noexcept;
#endif
