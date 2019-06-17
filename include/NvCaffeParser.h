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

#ifndef NV_CAFFE_PARSER_H
#define NV_CAFFE_PARSER_H

#include "NvInfer.h"

namespace ditcaffe
{
class NetParameter;
}

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
    virtual nvinfer1::ITensor* find(const char* name) const = 0;

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
    virtual const void* getData() = 0;
    virtual nvinfer1::DimsNCHW getDimensions() = 0;
    virtual nvinfer1::DataType getDataType() = 0;
    virtual void destroy() = 0;

protected:
    virtual ~IBinaryProtoBlob() {}
};

//!
//! \class IPluginFactory
//!
//! \brief Plugin factory used to configure plugins.
//!
class IPluginFactory
{
public:
    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPlugin.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPlugin(const char* layerName) = 0;

    //!
    //! \brief Creates a plugin.
    //!
    //! \param layerName Name of layer associated with the plugin.
    //! \param weights Weights used for the layer.
    //! \param nbWeights Number of weights.
    //!
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) = 0;
};

//!
//! \class IPluginFactoryExt
//!
//! \brief Plugin factory used to configure plugins with added support for TRT versioning.
//!
class IPluginFactoryExt : public IPluginFactory
{
public:
    virtual int getVersion() const
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPluginExt.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPluginExt(const char* layerName) = 0;
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
    virtual bool isPluginV2(const char* layerName) = 0;

    //!
    //! \brief Creates a plugin.
    //!
    //! \param layerName Name of layer associated with the plugin.
    //! \param weights Weights used for the layer.
    //! \param nbWeights Number of weights.
    //! \param libNamespace Library Namespace associated with the plugin object
    //!
    virtual nvinfer1::IPluginV2* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const char* libNamespace = "") = 0;
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
    //!   network configuration and weights associated with the network, respectively.
    //!
    //! \param deploy The plain text, prototxt file used to define the network configuration.
    //! \param model The binaryproto Caffe model that contains the weights associated with the network.
    //! \param network Network in which the CaffeParser will fill the layers.
    //! \param weightType The type to which the weights will transformed.
    //!
    //! \return A pointer to an IBlobNameToTensor object that contains the extracted data.
    //!
    //! \see nvcaffeparser1::IBlobNameToTensor
    //!
    virtual const IBlobNameToTensor* parse(const char* deploy,
                                           const char* model,
                                           nvinfer1::INetworkDefinition& network,
                                           nvinfer1::DataType weightType)
        = 0;

    //!
    //! \brief Parse a deploy prototxt a binaryproto Caffe model from memory buffers to extract
    //!   network configuration and weights associated with the network, respectively.
    //!
    //! \param deployBuffer The plain text deploy prototxt used to define the network configuration.
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
    virtual const IBlobNameToTensor* parseBuffers(const char* deployBuffer,
                                                  std::size_t deployLength,
                                                  const char* modelBuffer,
                                                  std::size_t modelLength,
                                                  nvinfer1::INetworkDefinition& network,
                                                  nvinfer1::DataType weightType) = 0;

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
    virtual IBinaryProtoBlob* parseBinaryProto(const char* fileName) = 0;

    //!
    //! \brief Set buffer size for the parsing and storage of the learned model.
    //!
    //! \param size The size of the buffer specified as the number of bytes.
    //!
    //! \note  Default size is 2^30 bytes.
    //!
    virtual void setProtobufBufferSize(size_t size) = 0;

    //!
    //! \brief Set the IPluginFactory used to create the user defined plugins.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactory.
    //!
    virtual void setPluginFactory(IPluginFactory* factory) = 0;

    //!
    //! \brief Set the IPluginFactoryExt used to create the user defined pluginExts.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactoryExt.
    //!
    virtual void setPluginFactoryExt(IPluginFactoryExt* factory) = 0;

    //!
    //! \brief Destroy this ICaffeParser object.
    //!
    virtual void destroy() = 0;

    //!
    //! \brief Set the IPluginFactoryV2 used to create the user defined pluginV2 objects.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactoryV2.
    //!
    virtual void setPluginFactoryV2(IPluginFactoryV2* factory) = 0;

    //!
    //! \brief Set the namespace used to lookup and create plugins in the network.
    //!
    virtual void setPluginNamespace(const char* libNamespace) = 0;

protected:
    virtual ~ICaffeParser() {}
};

//!
//! \brief Creates a ICaffeParser object.
//!
//! \return A pointer to the ICaffeParser object is returned.
//!
//! \see nvcaffeparser1::ICaffeParser
//!
TENSORRTAPI ICaffeParser* createCaffeParser();

//!
//! \brief Shuts down protocol buffers library.
//!
//! \note No part of the protocol buffers library can be used after this function is called.
//!
TENSORRTAPI void shutdownProtobufLibrary();
}

extern "C" TENSORRTAPI void* createNvCaffeParser_INTERNAL();
#endif
