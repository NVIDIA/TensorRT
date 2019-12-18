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

#ifndef NV_UFF_PARSER_H
#define NV_UFF_PARSER_H

#include "NvInfer.h"

// Current supported Universal Framework Format (UFF) version for the parser.
#define UFF_REQUIRED_VERSION_MAJOR 0
#define UFF_REQUIRED_VERSION_MINOR 6
#define UFF_REQUIRED_VERSION_PATCH 5

//!
//! \namespace nvuffparser
//!
//! \brief The TensorRT UFF parser API namespace.
//!
namespace nvuffparser
{

//!
//! \enum UffInputOrder
//! \brief The different possible supported input order.
//!
enum class UffInputOrder : int
{
    kNCHW = 0,  //!< NCHW order.
    kNHWC = 1,  //!< NHWC order.
    kNC = 2     //!< NC order.
};

//!
//! \enum FieldType
//! \brief The possible field types for custom layer.
//!

enum class FieldType : int
{
    kFLOAT = 0,     //!< FP32 field type.
    kINT32 = 1,     //!< INT32 field type.
    kCHAR = 2,      //!< char field type. String for length>1.
    kDIMS = 4,      //!< nvinfer1::Dims field type.
    kDATATYPE = 5,  //!< nvinfer1::DataType field type.
    kUNKNOWN = 6
};

//!
//! \class FieldMap
//!
//! \brief An array of field params used as a layer parameter for plugin layers.
//!
//! The node fields are passed by the parser to the API through the plugin
//! constructor. The implementation of the plugin should parse the contents of
//! the fieldMap as part of the plugin constructor
//!
class TENSORRTAPI FieldMap
{
public:
    const char* name;
    const void* data;
    FieldType type = FieldType::kUNKNOWN;
    int length = 1;

    FieldMap(const char* name, const void* data, const FieldType type, int length = 1) TRTNOEXCEPT;
};

struct FieldCollection
{
    int nbFields;
    const FieldMap* fields;
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
    virtual bool isPlugin(const char* layerName) TRTNOEXCEPT = 0;

    //!
    //! \brief Creates a plugin.
    //!
    //! \param layerName Name of layer associated with the plugin.
    //! \param weights Weights used for the layer.
    //! \param nbWeights Number of weights.
    //! \param fc A collection of FieldMaps used as layer parameters for different plugin layers.
    //!
    //! \see FieldCollection
    //!
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights,
                                            const FieldCollection fc) TRTNOEXCEPT = 0;

    virtual ~IPluginFactory() {}
};

//!
//! \class IPluginFactoryExt
//!
//! \brief Plugin factory used to configure plugins with added support for TRT versioning.
//!
class IPluginFactoryExt : public IPluginFactory
{
public:
    virtual int getVersion() const TRTNOEXCEPT
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPluginExt.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPluginExt(const char* layerName) TRTNOEXCEPT = 0;
};

//!
//! \class IUffParser
//!
//! \brief Class used for parsing models described using the UFF format.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IUffParser
{
public:
    //!
    //! \brief Register an input name of a UFF network with the associated Dimensions.
    //!
    //! \param inputName Input name.
    //! \param inputDims Input dimensions.
    //! \param inputOrder Input order on which the framework input was originally.
    //!
    virtual bool registerInput(const char* inputName, nvinfer1::Dims inputDims, UffInputOrder inputOrder) TRTNOEXCEPT = 0;

    //!
    //! \brief Register an output name of a UFF network.
    //!
    //! \param outputName Output name.
    //!
    virtual bool registerOutput(const char* outputName) TRTNOEXCEPT = 0;

    //!
    //! \brief Parse a UFF file.
    //!
    //! \param file File name of the UFF file.
    //! \param network Network in which the UFFParser will fill the layers.
    //! \param weightsType The type on which the weights will transformed in.
    //!
    virtual bool parse(const char* file,
                       nvinfer1::INetworkDefinition& network,
                       nvinfer1::DataType weightsType=nvinfer1::DataType::kFLOAT) TRTNOEXCEPT = 0;

    //!
    //! \brief Parse a UFF buffer, useful if the file already live in memory.
    //!
    //! \param buffer Buffer of the UFF file.
    //! \param size Size of buffer of the UFF file.
    //! \param network Network in which the UFFParser will fill the layers.
    //! \param weightsType The type on which the weights will transformed in.
    //!
    virtual bool parseBuffer(const char* buffer, std::size_t size,
                             nvinfer1::INetworkDefinition& network,
                             nvinfer1::DataType weightsType=nvinfer1::DataType::kFLOAT) TRTNOEXCEPT = 0;

    virtual void destroy() TRTNOEXCEPT = 0;

    //!
    //! \brief Return Version Major of the UFF.
    //!
    virtual int getUffRequiredVersionMajor() TRTNOEXCEPT = 0;

    //!
    //! \brief Return Version Minor of the UFF.
    //!
    virtual int getUffRequiredVersionMinor() TRTNOEXCEPT = 0;

    //!
    //! \brief Return Patch Version of the UFF.
    //!
    virtual int getUffRequiredVersionPatch() TRTNOEXCEPT = 0;

    //!
    //! \brief Set the IPluginFactory used to create the user defined plugins.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactory.
    //!
    virtual void setPluginFactory(IPluginFactory* factory) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the IPluginFactoryExt used to create the user defined pluginExts.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactoryExt.
    //!
    virtual void setPluginFactoryExt(IPluginFactoryExt* factory) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the namespace used to lookup and create plugins in the network.
    //!
    virtual void setPluginNamespace(const char* libNamespace) TRTNOEXCEPT = 0;

protected:
    virtual ~IUffParser() {}

public:
    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(nvinfer1::IErrorRecorder* recorder) TRTNOEXCEPT = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual nvinfer1::IErrorRecorder* getErrorRecorder() const TRTNOEXCEPT = 0;
};

//!
//! \brief Creates a IUffParser object.
//!
//! \return A pointer to the IUffParser object is returned.
//!
//! \see nvuffparser::IUffParser
//!
TENSORRTAPI IUffParser* createUffParser() TRTNOEXCEPT;

//!
//! \brief Shuts down protocol buffers library.
//!
//! \note No part of the protocol buffers library can be used after this function is called.
//!
TENSORRTAPI void shutdownProtobufLibrary(void) TRTNOEXCEPT;

} // namespace nvuffparser


//!
//! Internal C entry point for creating IUffParser
//! @private
//!
extern "C" TENSORRTAPI void* createNvUffParser_INTERNAL() TRTNOEXCEPT;

#endif /* !NV_UFF_PARSER_H */