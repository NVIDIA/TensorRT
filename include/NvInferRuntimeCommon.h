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

#ifndef NV_INFER_RUNTIME_COMMON_H
#define NV_INFER_RUNTIME_COMMON_H

#include <cstddef>
#include <cstdint>
#include "NvInferVersion.h"

#if __cplusplus >= 201103L
#define _TENSORRT_FINAL final
#define _TENSORRT_OVERRIDE override
#else
#define _TENSORRT_FINAL
#define _TENSORRT_OVERRIDE
#endif

//!< Items that are marked as deprecated will be removed in a future release.
#if __cplusplus >= 201402L
#define TRT_DEPRECATED [[deprecated]]
#if __GNUC__ < 6
#define TRT_DEPRECATED_ENUM
#else
#define TRT_DEPRECATED_ENUM TRT_DEPRECATED
#endif
#ifdef _MSC_VER
#define TRT_DEPRECATED_API __declspec(dllexport)
#else
#define TRT_DEPRECATED_API [[deprecated]] __attribute__((visibility("default")))
#endif
#else
#ifdef _MSC_VER
#define TRT_DEPRECATED
#define TRT_DEPRECATED_ENUM
#define TRT_DEPRECATED_API __declspec(dllexport)
#else
#define TRT_DEPRECATED __attribute__((deprecated))
#define TRT_DEPRECATED_ENUM
#define TRT_DEPRECATED_API __attribute__((deprecated, visibility("default")))
#endif
#endif

//!< Defines which symbols are exported
#ifdef TENSORRT_BUILD_LIB
#ifdef _MSC_VER
#define TENSORRTAPI __declspec(dllexport)
#else
#define TENSORRTAPI __attribute__((visibility("default")))
#endif
#else
#define TENSORRTAPI
#endif
#define TRTNOEXCEPT
//!
//! \file NvInferRuntimeCommon.h
//!
//! This is the top-level API file for TensorRT core runtime library.
//!

// forward declare some CUDA types to avoid an include dependency

struct cublasContext;
struct cudnnContext;

typedef struct CUstream_st* cudaStream_t; //!< Forward declaration of cudaStream_t.
typedef struct CUevent_st* cudaEvent_t;   //!< Forward declaration of cudaEvent_t.

static const int NV_TENSORRT_VERSION = (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSORRT_PATCH; // major, minor, patch

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{

class IErrorRecorder; //!< Forward declare IErrorRecorder for use in other interfaces.
class IGpuAllocator; //!< Forward declare IGpuAllocator for use in other interfaces.

//!
//! \enum ActivationType
//!
//! \brief Enumerates the types of activation to perform in an activation layer.
//!
enum class ActivationType : int
{
    kRELU = 0,             //!< Rectified linear activation.
    kSIGMOID = 1,          //!< Sigmoid activation.
    kTANH = 2,             //!< TanH activation.
    kLEAKY_RELU = 3,       //!< LeakyRelu activation: x>=0 ? x : alpha * x.
    kELU = 4,              //!< Elu activation: x>=0 ? x : alpha * (exp(x) - 1).
    kSELU = 5,             //!< Selu activation: x>0 ? beta * x : beta * (alpha*exp(x) - alpha)
    kSOFTSIGN = 6,         //!< Softsign activation: x / (1+|x|)
    kSOFTPLUS = 7,         //!< Parametric softplus activation: alpha*log(exp(beta*x)+1)
    kCLIP = 8,             //!< Clip activation: max(alpha, min(beta, x))
    kHARD_SIGMOID = 9,     //!< Hard sigmoid activation: max(0, min(1, alpha*x+beta))
    kSCALED_TANH = 10,     //!< Scaled tanh activation: alpha*tanh(beta*x)
    kTHRESHOLDED_RELU = 11 //!< Thresholded ReLU activation: x>alpha ? x : 0
};

template <typename T>
constexpr inline int EnumMax(); //!< Maximum number of elements in an enumeration type.

template <>
constexpr inline int EnumMax<ActivationType>()
{
    return 12;
} //!< Maximum number of elements in ActivationType enum. \see ActivationType

//!
//! \enum DataType
//! \brief The type of weights and tensors.
//!
enum class DataType : int
{
    kFLOAT = 0, //!< FP32 format.
    kHALF = 1,  //!< FP16 format.
    kINT8 = 2,  //!< quantized INT8 format.
    kINT32 = 3, //!< INT32 format.
    kBOOL = 4   //!< BOOL format.
};

template <>
constexpr inline int EnumMax<DataType>()
{
    return 5;
} //!< Maximum number of elements in DataType enum. \see DataType

//!
//! \enum DimensionType
//! \brief The type of data encoded across this dimension.
//!
enum class DimensionType : int
{
    kSPATIAL = 0, //!< Elements correspond to different spatial data.
    kCHANNEL = 1, //!< Elements correspond to different channels.
    kINDEX = 2,   //!< Elements correspond to different batch index.
    kSEQUENCE = 3 //!< Elements correspond to different sequence values.
};

template <>
constexpr inline int EnumMax<DimensionType>()
{
    return 4;
} //!< Maximum number of elements in DimensionType enum. \see DimensionType

//!
//! \class Dims
//! \brief Structure to define the dimensions of a tensor.
//!
//! \note: Currently the following formats are supported for layer inputs and outputs:
//! * zero or more index dimensions followed by one channel and two spatial dimensions (e.g. CHW)
//! * one time series dimension followed by one index dimension followed by one channel dimension (i.e. TNC)
//!
//! TensorRT can also return an invalid dims structure. This structure is represented by nbDims == -1
//! and d[i] == 0 for all d.
//!
class Dims
{
public:
    static const int MAX_DIMS = 8; //!< The maximum number of dimensions supported for a tensor.
    int nbDims;                    //!< The number of dimensions.
    int d[MAX_DIMS];               //!< The extent of each dimension.
    TRT_DEPRECATED DimensionType type[MAX_DIMS];  //!< The type of each dimension.
};

//!
//! \brief It is capable of representing one or more TensorFormat by binary OR
//! operations, e.g., 1U << TensorFormats::kCHW4 | 1U << TensorFormats::kCHW32.
//!
//! \see ITensor::getAllowedFormats(), ITensor::setAllowedFormats(),
//!
typedef uint32_t TensorFormats;

//!
//! \enum TensorFormat
//!
//! \brief Format of the input/output tensors.
//!
//! This enum is extended to be used by both plugins and reformat-free network
//! I/O tensors.
//!
//! \see IPluginExt::getPluginFormats(), safe::ICudaEngine::getBindingFormat()
//!
//! For more information about data formats, see the topic "Data Format Description" located in the
//! TensorRT Developer Guide (https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html).
//!
enum class TensorFormat : int
{
    //! Row major linear format.
    //! For a tensor with dimensions {N, C, H, W} or {numbers, channels,
    //! columns, rows}, the dimensional index corresponds to {3, 2, 1, 0}
    //! and thus the order is W minor.
    kLINEAR = 0,
    kNCHW TRT_DEPRECATED_ENUM = kLINEAR, //! <-- Deprecated, used for backward compatibility

    //! Two wide channel vectorized row major format. This format is bound to
    //! FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+1)/2][H][W][2], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/2][h][w][c%2].
    kCHW2 = 1,
    kNC2HW2 TRT_DEPRECATED_ENUM = kCHW2, //! <-- Deprecated, used for backward compatibility

    //! Eight channel format where C is padded to a multiple of 8. This format
    //! is bound to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, H, W, C},
    //! the memory layout is equivalent to the array with dimensions
    //! [N][H][W][(C+7)/8*8], with the tensor coordinates (n, h, w, c)
    //! mapping to array subscript [n][h][w][c].
    kHWC8 = 2,
    kNHWC8 TRT_DEPRECATED_ENUM = kHWC8, //! <-- Deprecated, used for backward compatibility

    //! Four wide channel vectorized row major format. This format is bound to
    //! INT8 or FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+3)/4][H][W][4], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/4][h][w][c%4].
    kCHW4 = 3,

    //! Sixteen wide channel vectorized row major format. This format is bound
    //! to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/16][h][w][c%16].
    kCHW16 = 4,

    //! Thirty-two wide channel vectorized row major format. This format is
    //! only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/32][h][w][c%32].
    kCHW32 = 5
};

//!
//! \brief PluginFormat is reserved for backward compatibility.
//!
//! \see IPluginExt::getPluginFormats()
//!
using PluginFormat = TensorFormat;

template <>
constexpr inline int EnumMax<TensorFormat>()
{
    return 6;
} //!< Maximum number of elements in TensorFormat enum. \see TensorFormat

//! \struct PluginTensorDesc
//!
//! \brief Fields that a plugin might see for an input or output.
//!
//! Scale is only valid when data type is DataType::kINT8. TensorRT will set
//! the value to -1.0f if it is invalid.
//!
//! \see IPluginV2IOExt::supportsFormat
//! \see IPluginV2IOExt::configurePlugin
//!
struct PluginTensorDesc
{
    Dims dims;
    DataType type;
    TensorFormat format;
    float scale;
};

//! \struct PluginVersion
//!
//! \brief Definition of plugin versions.
//!
//! Tag for plug-in versions.  Used in upper byte of getTensorRTVersion().
//!
enum class PluginVersion : uint8_t
{
    kV2 = 0,            //! IPluginV2
    kV2_EXT = 1,        //! IPluginV2Ext
    kV2_IOEXT = 2,      //! IPluginV2IOExt
    kV2_DYNAMICEXT = 3, //! IPluginV2DynamicExt
};

//! \class IPluginV2
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. When
//! combined with IPluginCreator it provides a mechanism to register plugins and
//! look up the Plugin Registry during de-serialization.
//!
//! \see IPluginCreator
//! \see IPluginRegistry
//!
class IPluginV2
{
public:
    //!
    //! \brief Return the API version with which this plugin was built.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with plugins.
    //!
    virtual int getTensorRTVersion() const TRTNOEXCEPT
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief Return the plugin type. Should match the plugin name returned by the corresponding plugin creator
    // \see IPluginCreator::getPluginName()
    //!
    virtual const char* getPluginType() const TRTNOEXCEPT = 0;

    //!
    //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator
    // \see IPluginCreator::getPluginVersion()
    //!
    virtual const char* getPluginVersion() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of outputs from the layer.
    //!
    //! \return The number of outputs.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
    //!
    virtual int getNbOutputs() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the dimension of an output tensor.
    //!
    //! \param index The index of the output tensor.
    //! \param inputs The input tensors.
    //! \param nbInputDims The number of input tensors.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
    //!
    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRTNOEXCEPT = 0;

    //!
    //! \brief Check format support.
    //!
    //! \param type DataType requested.
    //! \param format PluginFormat requested.
    //! \return true if the plugin supports the type-format combination.
    //!
    //! This function is called by the implementations of INetworkDefinition, IBuilder, and safe::ICudaEngine/ICudaEngine.
    //! In particular, it is called when creating an engine and when deserializing an engine.
    //!
    //! \warning for the format field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and PluginFormat::kCHW32
    //! will not be passed in, this is to keep backward compatibility with TensorRT 5.x series.  Use PluginV2IOExt
    //! or PluginV2DynamicExt for other PluginFormats.
    //!
    virtual bool supportsFormat(DataType type, PluginFormat format) const TRTNOEXCEPT = 0;

    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make algorithm choices on the basis
    //! of its weights, dimensions, and maximum batch size.
    //!
    //! \param inputDims The input tensor dimensions.
    //! \param nbInputs The number of inputs.
    //! \param outputDims The output tensor dimensions.
    //! \param nbOutputs The number of outputs.
    //! \param type The data type selected for the engine.
    //! \param format The format selected for the engine.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be 3-dimensional CHW dimensions).
    //!
    //! \warning for the format field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and PluginFormat::kCHW32
    //! will not be passed in, this is to keep backward compatibility with TensorRT 5.x series.  Use PluginV2IOExt
    //! or PluginV2DynamicExt for other PluginFormats.
    //!
    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Initialize the layer for execution. This is called when the engine is created.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    virtual int initialize() TRTNOEXCEPT = 0;

    //!
    //! \brief Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
    //! \see initialize()
    //!
    virtual void terminate() TRTNOEXCEPT = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called during engine startup, after initialize(). The workspace size returned should be sufficient for any
    //! batch size up to the maximum.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(int maxBatchSize) const TRTNOEXCEPT = 0;

    //!
    //! \brief Execute the layer.
    //!
    //! \param batchSize The number of inputs in the batch.
    //! \param inputs The memory for the input tensors.
    //! \param outputs The memory for the output tensors.
    //! \param workspace Workspace for execution.
    //! \param stream The stream in which to execute the kernels.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT = 0;

    //!
    //! \brief Find the size of the serialization buffer required.
    //!
    //! \return The size of the serialization buffer.
    //!
    virtual size_t getSerializationSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Serialize the layer.
    //!
    //! \param buffer A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by getSerializationSize.
    //!
    //! \see getSerializationSize()
    //!
    virtual void serialize(void* buffer) const TRTNOEXCEPT = 0;

    //!
    //! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with these parameters.
    //!
    virtual IPluginV2* clone() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
    //! objects from the same plugin library should have the same namespace.
    //!
    virtual void setPluginNamespace(const char* pluginNamespace) TRTNOEXCEPT = 0;

    //!
    //! \brief Return the namespace of the plugin object.
    //!
    virtual const char* getPluginNamespace() const TRTNOEXCEPT = 0;

protected:
    virtual ~IPluginV2() {}
};

//! \class IPluginV2Ext
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. This
//! interface provides additional capabilities to the IPluginV2 interface by
//! supporting different output data types and broadcast across batch.
//!
//! \see IPluginV2
//!
class IPluginV2Ext : public IPluginV2
{
public:
    //!
    //! \brief Return the DataType of the plugin output at the requested index.
    //! The default behavior should be to return the type of the first input, or DataType::kFLOAT if the layer has no inputs.
    //! The returned data type must have a format that is supported by the plugin.
    //! \see supportsFormat()
    //!
    virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRTNOEXCEPT = 0;

    //! \brief Return true if output tensor is broadcast across a batch.
    //!
    //! \param outputIndex The index of the output
    //! \param inputIsBroadcasted The ith element is true if the tensor for the ith input is broadcast across a batch.
    //! \param nbInputs The number of inputs
    //!
    //! The values in inputIsBroadcasted refer to broadcasting at the semantic level,
    //! i.e. are unaffected by whether method canBroadcastInputAcrossBatch requests
    //! physical replication of the values.
    //!
    virtual bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRTNOEXCEPT = 0;

    //! \brief Return true if plugin can use input that is broadcast across batch without replication.
    //!
    //! \param inputIndex Index of input that could be broadcast.
    //!
    //! For each input whose tensor is semantically broadcast across a batch,
    //! TensorRT calls this method before calling configurePlugin.
    //! If canBroadcastInputAcrossBatch returns true, TensorRT will not replicate the input tensor;
    //! i.e., there will be a single copy that the plugin should share across the batch.
    //! If it returns false, TensorRT will replicate the input tensor
    //! so that it appears like a non-broadcasted tensor.
    //!
    //! This method is called only for inputs that can be broadcast.
    //!
    virtual bool canBroadcastInputAcrossBatch(int inputIndex) const TRTNOEXCEPT = 0;

    //!
    //! \brief Configure the layer with input and output data types.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make algorithm choices on the basis
    //! of its weights, dimensions, data types and maximum batch size.
    //!
    //! \param inputDims The input tensor dimensions.
    //! \param nbInputs The number of inputs.
    //! \param outputDims The output tensor dimensions.
    //! \param nbOutputs The number of outputs.
    //! \param inputTypes The data types selected for the plugin inputs.
    //! \param outputTypes The data types selected for the plugin outputs.
    //! \param inputIsBroadcast True for each input that the plugin must broadcast across the batch.
    //! \param outputIsBroadcast True for each output that TensorRT will broadcast across the batch.
    //! \param floatFormat The format selected for the engine for the floating point inputs/outputs.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be 3-dimensional CHW dimensions).
    //! When inputIsBroadcast or outputIsBroadcast is true, the outermost batch size for that input or output should be treated as if it is one.
    //! \ref inputIsBroadcast[i] is true only if the input is semantically broadcast across the batch and \ref canBroadcastInputAcrossBatch(i) returned true.
    //! \ref outputIsBroadcast[i] is true only if \ref isOutputBroadcastAcrossBatch(i) returned true.
    //!
    //! \warning for the floatFormat field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and PluginFormat::kCHW32
    //! will not be passed in, this is to keep backward compatibility with TensorRT 5.x series.  Use PluginV2IOExt
    //! or PluginV2DynamicExt for other PluginFormats.
    //!

    virtual void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                                 int nbOutputs, const DataType* inputTypes, const DataType* outputTypes,
                                 const bool* inputIsBroadcast, const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) TRTNOEXCEPT = 0;

    virtual ~IPluginV2Ext() {}

    //!
    //! \brief Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    //!
    //! \param cudnn The cudnn context handle of the execution context
    //! \param cublas The cublas context handle of the execution context
    //! \param allocator The allocator used by the execution context
    //!
    //! This function is called automatically for each plugin when a new execution context is created.
    //! If the plugin needs per-context resource, it can be allocated here.
    //! The plugin can also get context-owned CUDNN and CUBLAS context here.
    //!
    virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) TRTNOEXCEPT {}

    //!
    //! \brief Detach the plugin object from its execution context.
    //!
    //! This function is called automatically for each plugin when a execution context is destroyed.
    //! If the plugin owns per-context resource, it can be released here.
    //!
    virtual void detachFromContext() TRTNOEXCEPT {}

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.
    //! If the source plugin is pre-configured with configurePlugin(), the returned object should also be pre-configured. The returned object should allow attachToContext() with a new execution context.
    //! Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object (e.g. via ref-counting) to avoid duplication.
    //!
    virtual IPluginV2Ext* clone() const _TENSORRT_OVERRIDE TRTNOEXCEPT = 0;

protected:
    //!
    //! \brief Return the API version with which this plugin was built. The
    //!  upper byte reserved by TensorRT and is used to differentiate this from IPlguinV2.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with plugins.
    //!
    int getTensorRTVersion() const _TENSORRT_OVERRIDE TRTNOEXCEPT
    {
        return (static_cast<int>(PluginVersion::kV2_EXT) << 24 | (NV_TENSORRT_VERSION & 0xFFFFFF));
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    void configureWithFormat(const Dims* /*inputDims*/, int /*nbInputs*/, const Dims* /*outputDims*/,
                             int /*nbOutputs*/, DataType /*type*/, PluginFormat /*format*/, int /*maxBatchSize*/) _TENSORRT_OVERRIDE TRTNOEXCEPT {}
};

//! \class IPluginV2IOExt
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. This interface provides additional
//! capabilities to the IPluginV2Ext interface by extending different I/O data types and tensor formats.
//!
//! \see IPluginV2Ext
//!
class IPluginV2IOExt : public IPluginV2Ext
{
public:
    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of I/O PluginTensorDesc and the maximum batch size.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInput Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutput Number of output tensors.
    //!
    virtual void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRTNOEXCEPT = 0;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
    //!
    //! For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
    //! Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs-1.
    //!
    //! TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
    //! by inOut[pos].format and inOut[pos].type. The override should return true if that format/datatype at inOut[pos]
    //! are supported by the plugin. If support is conditional on other input/output formats/datatypes, the plugin can
    //! make its result conditional on the formats/datatypes in inOut[0..pos-1], which will be set to values
    //! that the plugin supports. The override should not inspect inOut[pos+1..nbInputs+nbOutputs-1],
    //! which will have invalid values.  In other words, the decision for pos must be based on inOut[0..pos] only.
    //!
    //! Some examples:
    //!
    //! * A definition for a plugin that supports only FP16 NCHW:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR && inOut.type[pos] == DataType::kHALF;
    //!
    //! * A definition for a plugin that supports only FP16 NCHW for its two inputs,
    //!   and FP32 NCHW for its single output:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR && (inOut.type[pos] == pos < 2 ?  DataType::kHALF : DataType::kFLOAT);
    //!
    //! * A definition for a "polymorphic" plugin with two inputs and one output that supports
    //!   any format or type, but the inputs and output must have the same format and type:
    //!
    //!         return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
    //!
    //! Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
    //!
    virtual bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRTNOEXCEPT = 0;

protected:
    //!
    //! \brief Return the API version with which this plugin was built. The upper byte is reserved by TensorRT and is
    //! used to differentiate this from IPlguinV2 and IPluginV2Ext.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    TRT_DEPRECATED
    int getTensorRTVersion() const _TENSORRT_OVERRIDE
    {
        return (static_cast<int>(PluginVersion::kV2_IOEXT) << 24 | (NV_TENSORRT_VERSION & 0xFFFFFF));
    }

    //!
    //! \brief Deprecated interface inheriting from base class. Derived classes should not implement this. In a C++11
    //! API it would be override final.
    //!
    TRT_DEPRECATED
    void configureWithFormat(
        const Dims*, int, const Dims*, int, DataType, PluginFormat, int) _TENSORRT_OVERRIDE _TENSORRT_FINAL
    {
    }

    //!
    //! \brief Deprecated interface inheriting from base class. Derived classes should not implement this. In a C++11
    //! API it would be override final.
    //!
    TRT_DEPRECATED
    void configurePlugin(const Dims*, int, const Dims*, int, const DataType*, const DataType*, const bool*, const bool*,
        PluginFormat, int) _TENSORRT_OVERRIDE _TENSORRT_FINAL
    {
    }

    //!
    //! \brief Deprecated interface inheriting from base class. Derived classes should not implement this. In a C++11
    //! API it would be override final.
    //!
    TRT_DEPRECATED
    bool supportsFormat(DataType, PluginFormat) const _TENSORRT_OVERRIDE _TENSORRT_FINAL
    {
        return false;
    }
};

//!
//! \enum FieldType
//! \brief The possible field types for custom layer.
//!

enum class PluginFieldType : int
{
    kFLOAT16 = 0, //!< FP16 field type.
    kFLOAT32 = 1, //!< FP32 field type.
    kFLOAT64 = 2, //!< FP64 field type.
    kINT8 = 3,    //!< INT8 field type.
    kINT16 = 4,   //!< INT16 field type.
    kINT32 = 5,   //!< INT32 field type.
    kCHAR = 6,    //!< char field type.
    kDIMS = 7,    //!< nvinfer1::Dims field type.
    kUNKNOWN = 8
};

//!
//! \class PluginField
//!
//! \brief Structure containing plugin attribute field names and associated data
//! This information can be parsed to decode necessary plugin metadata
//!
//!
struct PluginField
{
    //!
    //! \brief Plugin field attribute name
    //!
    const char* name{nullptr};
    //!
    //! \brief Plugin field attribute data
    //!
    const void* data{nullptr};
    //!
    //! \brief Plugin field attribute type
    //! \see PluginFieldType
    //!
    PluginFieldType type{PluginFieldType::kUNKNOWN};
    //!
    //! \brief Number of data entries in the Plugin attribute
    //!
    int32_t length{0};

    PluginField(const char* name_ = nullptr, const void* data_ = nullptr, const PluginFieldType type_ = PluginFieldType::kUNKNOWN, int32_t length_ = 0)
        : name(name_)
        , data(data_)
        , type(type_)
        , length(length_)
    {
    }
};

struct PluginFieldCollection
{
    int nbFields;              //!< Number of PluginField entries
    const PluginField* fields; //!< Pointer to PluginField entries
};

//!
//! \class IPluginCreator
//!
//! \brief Plugin creator class for user implemented layers.
//!
//! \see IPlugin and IPluginFactory
//!

class IPluginCreator
{
public:
    //!
    //! \brief Return the version of the API the plugin creator was compiled with.
    //!
    virtual int getTensorRTVersion() const TRTNOEXCEPT { return NV_TENSORRT_VERSION; }

    //!
    //! \brief Return the plugin name.
    //!
    virtual const char* getPluginName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Return the plugin version.
    //!
    virtual const char* getPluginVersion() const TRTNOEXCEPT = 0;

    //!
    //! \brief Return a list of fields that needs to be passed to createPlugin.
    //! \see PluginFieldCollection
    //!
    virtual const PluginFieldCollection* getFieldNames() TRTNOEXCEPT = 0;

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
    virtual IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) TRTNOEXCEPT = 0;

    //!
    //! \brief Called during deserialization of plugin layer. Return a plugin object.
    //!
    virtual IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the namespace of the plugin creator based on the plugin
    //! library it belongs to. This can be set while registering the plugin creator.
    //!
    //! \see IPluginRegistry::registerCreator()
    //!
    virtual void setPluginNamespace(const char* pluginNamespace) TRTNOEXCEPT = 0;

    //!
    //! \brief Return the namespace of the plugin creator object.
    //!
    virtual const char* getPluginNamespace() const TRTNOEXCEPT = 0;

    virtual ~IPluginCreator() {}
};

//!
//! \class IPluginRegistry
//!
//! \brief Single registration point for all plugins in an application. It is
//! used to find plugin implementations during engine deserialization.
//! Internally, the plugin registry is considered to be a singleton so all
//! plugins in an application are part of the same global registry.
//! Note that the plugin registry is only supported for plugins of type
//! IPluginV2 and should also have a corresponding IPluginCreator implementation.
//!
//! \see IPluginV2 and IPluginCreator
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!

class IPluginRegistry
{
public:
    //!
    //! \brief Register a plugin creator. Returns false if one with same type
    //! is already registered.
    //!
    virtual bool registerCreator(IPluginCreator& creator, const char* pluginNamespace) noexcept = 0;

    //!
    //! \brief Return all the registered plugin creators and the number of
    //! registered plugin creators. Returns nullptr if none found.
    //!
    virtual IPluginCreator* const* getPluginCreatorList(int* numCreators) const noexcept = 0;

    //!
    //! \brief Return plugin creator based on plugin type, version and
    //! namespace associated with plugin during network creation.
    //!
    virtual IPluginCreator* getPluginCreator(const char* pluginType, const char* pluginVersion, const char* pluginNamespace = "") noexcept = 0;

protected:
    virtual ~IPluginRegistry() noexcept {}

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
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief set the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called, or an ErrorRecorder has not been
    //! inherited.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};


//!
//! \enum TensorLocation
//! \brief The location for tensor data storage, device or host.
//!
enum class TensorLocation : int
{
    kDEVICE = 0, //!< Data stored on device.
    kHOST = 1,   //!< Data stored on host.
};

template <>
constexpr inline int EnumMax<TensorLocation>()
{
    return 2;
} //!< Maximum number of elements in TensorLocation enum. \see TensorLocation

//!
//! \class IGpuAllocator
//!
//! \brief Application-implemented class for controlling allocation on the GPU.
//!
class IGpuAllocator
{
public:
    //!
    //! A callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory required.
    //! \param alignment The required alignment of memory. Alignment will zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! If an allocation request of size 0 is made, nullptr should be returned.
    //!
    //! If an allocation request cannot be satisfied, nullptr should be returned.
    //!
    virtual void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) TRTNOEXCEPT = 0;

    //!
    //! A callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //!
    virtual void free(void* memory) TRTNOEXCEPT = 0;

    //!
    //! Destructor declared virtual as general good practice for a class with virtual methods.
    //! TensorRT never calls the destructor for an IGpuAllocator defined by the application.
    //!
    virtual ~IGpuAllocator() {}
};

//!
//! \class ILogger
//!
//! \brief Application-implemented logging interface for the builder, engine and runtime.
//!
//! Note that although a logger is passed on creation to each instance of a IBuilder or safe::IRuntime interface, the logger is internally considered a singleton, and thus
//! multiple instances of safe::IRuntime and/or IBuilder must all use the same logger.
//!
class ILogger
{
public:
    //!
    //! \enum Severity
    //!
    //! The severity corresponding to a log message.
    //!
    enum class Severity : int
    {
        kINTERNAL_ERROR = 0, //!< An internal error has occurred. Execution is unrecoverable.
        kERROR = 1,          //!< An application error has occurred.
        kWARNING = 2,        //!< An application error has been discovered, but TensorRT has recovered or fallen back to a default.
        kINFO = 3,           //!< Informational messages with instructional information.
        kVERBOSE = 4,        //!< Verbose messages with debugging information.
    };

    //!
    //! A callback implemented by the application to handle logging messages;
    //!
    //! \param severity The severity of the message.
    //! \param msg The log message, null terminated.
    //!
    virtual void log(Severity severity, const char* msg) TRTNOEXCEPT = 0;

    virtual ~ILogger() {}
};

template <>
constexpr inline int EnumMax<ILogger::Severity>()
{
    return 5;
} //!< Maximum number of elements in ILogger::Severity enum. \see ILogger::Severity

//!
//! \enum ErrorCode
//!
//! \brief Error codes that can be returned by TensorRT during execution.
//!
enum class ErrorCode : int
{
    //!
    //! Execution completed successfully.
    //!
    kSUCCESS = 0,

    //!
    //! An error that does not fall into any other category. This error is included for forward compatibility
    //!
    kUNSPECIFIED_ERROR = 1,

    //!
    //! A non-recoverable TensorRT error occurred.
    //!
    kINTERNAL_ERROR = 2,

    //!
    //! An argument passed to the function is invalid in isolation.
    //! This is a violation of the API contract
    //!
    kINVALID_ARGUMENT = 3,

    //!
    //! An error occurred when comparing the state of an argument relative to other arguments. For example, the
    //! dimensions for concat differ between two tensors outside of the channel dimension. This error is triggered
    //! when an argument is correct in isolation, but not relative to other arguments. This is to help to distinguish
    //! from the simple errors from the more complex errors.
    //! This is a violation of the API contract.
    //!
    kINVALID_CONFIG = 4,

    //!
    //! An error occurred when performing an allocation of memory on the host or the device.
    //! A memory allocation error is normally fatal, but in the case where the application provided its own memory
    //! allocation routine, it is possible to increase the pool of available memory and resume execution.
    //!
    kFAILED_ALLOCATION = 5,

    //!
    //! One, or more, of the components that TensorRT relies on did not initialize correctly.
    //! This is a system setup issue.
    //!
    kFAILED_INITIALIZATION = 6,

    //!
    //! An error occurred during execution that caused TensorRT to end prematurely, either an asynchronous error or
    //! other execution errors reported by CUDA/DLA. In a dynamic system, the
    //! data can be thrown away and the next frame can be processed or execution can be retried.
    //! This is either an execution error or a memory error.
    //!
    kFAILED_EXECUTION = 7,

    //!
    //! An error occurred during execution that caused the data to become corrupted, but execution finished. Examples
    //! of this error are NaN squashing or integer overflow. In a dynamic system, the data can be thrown away and the
    //! next frame can be processed or execution can be retried.
    //! This is either a data corruption error, an input error, or a range error.
    //!
    kFAILED_COMPUTATION = 8,

    //!
    //! TensorRT was put into a bad state by incorrect sequence of function calls. An example of an invalid state is
    //! specifying a layer to be DLA only without GPU fallback, and that layer is not supported by DLA. This can occur
    //! in situations where a service is optimistically executing networks for multiple different configurations
    //! without checking proper error configurations, and instead throwing away bad configurations caught by TensorRT.
    //! This is a violation of the API contract, but can be recoverable.
    //!
    //! Example of a recovery:
    //! GPU fallback is disabled and conv layer with large filter(63x63) is specified to run on DLA. This will fail due
    //! to DLA not supporting the large kernel size. This can be recovered by either turning on GPU fallback
    //! or setting the layer to run on the GPU.
    //!
    kINVALID_STATE = 9,

    //!
    //! An error occurred due to the network not being supported on the device due to constraints of the hardware or
    //! system. An example is running a unsafe layer in a safety certified context, or a resource requirement for the
    //! current network is greater than the capabilities of the target device. The network is otherwise correct, but
    //! the network and hardware combination is problematic. This can be recoverable.
    //! Examples:
    //!  * Scratch space requests larger than available device memory and can be recovered by increasing allowed
    //!    workspace size.
    //!  * Tensor size exceeds the maximum element count and can be recovered by reducing the maximum batch size.
    //!
    kUNSUPPORTED_STATE = 10,

};

template <>
constexpr inline int EnumMax<ErrorCode>()
{
    return 11;
} //!< Maximum number of elements in ErrorCode enum. \see ErrorCode


//!
//! \class IErrorRecorder
//!
//! \brief Reference counted application-implemented error reporting interface for TensorRT objects.
//!
//! The error reporting mechanism is a user defined object that interacts with the internal state of the object
//! that it is assigned to in order to determine information about abnormalities in execution. The error recorder
//! gets both an error enum that is more descriptive than pass/fail and also a string description that gives more
//! detail on the exact failure modes. In the safety context, the error strings are all limited to 128 characters
//! in length.
//! The ErrorRecorder gets passed along to any class that is created from another class that has an ErrorRecorder
//! assigned to it. For example, assigning an ErrorRecorder to an IBuilder allows all INetwork's, ILayer's, and
//! ITensor's to use the same error recorder. For functions that have their own ErrorRecorder accessor functions.
//! This allows registering a different error recorder or de-registering of the error recorder for that specific
//! object.
//!
//! The ErrorRecorder object implementation must be thread safe if the same ErrorRecorder is passed to different
//! interface objects being executed in parallel in different threads. All locking and synchronization is
//! pushed to the interface implementation and TensorRT does not hold any synchronization primitives when accessing
//! the interface functions.
//!
class IErrorRecorder
{
public:
    //!
    //! A typedef of a c-style string for reporting error descriptions.
    //!
    using ErrorDesc = const char*;

    //!
    //! A typedef of a 32bit integer for reference counting.
    //!
    using RefCount = int32_t;

    virtual ~IErrorRecorder() noexcept {};

    // Public API’s used to retrieve information from the error recorder.

    //!
    //! \brief Return the number of errors
    //!
    //! Determines the number of errors that occurred between the current point in execution
    //! and the last time that the clear() was executed. Due to the possibility of asynchronous
    //! errors occuring, a TensorRT API can return correct results, but still register errors
    //! with the Error Recorder. The value of getNbErrors must monotonically increases until clear()
    //! is called.
    //!
    //! \return Returns the number of errors detected, or 0 if there are no errors.
    //!
    //! \see clear
    //!
    virtual int32_t getNbErrors() const noexcept = 0;

    //!
    //! \brief Returns the ErrorCode enumeration.
    //!
    //! \param errorIdx A 32bit integer that indexes into the error array.
    //!
    //! The errorIdx specifies what error code from 0 to getNbErrors()-1 that the application
    //! wants to analyze and return the error code enum.
    //!
    //! \return Returns the enum corresponding to errorIdx.
    //!
    //! \see getErrorDesc, ErrorCode
    //!
    virtual ErrorCode getErrorCode(int32_t errorIdx) const noexcept = 0;

    //!
    //! \brief Returns the c-style string description of the error.
    //!
    //! \param errorIdx A 32bit integer that indexes into the error array.
    //!
    //! For the error specified by the idx value, return the string description of the error. The
    //! error string is a c-style string that is zero delimited. In the safety context there is a
    //! constant length requirement to remove any dynamic memory allocations and the error message
    //! may be truncated. The format of the string is "<EnumAsStr> - <Description>".
    //!
    //! \return Returns a string representation of the error along with a description of the error.
    //!
    //! \see getErrorCode
    //!
    virtual ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept = 0;

    //!
    //! \brief Determine if the error stack has overflowed.
    //!
    //! In the case when the number of errors is large, this function is used to query if one or more
    //! errors have been dropped due to lack of storage capacity. This is especially important in the
    //! automotive safety case where the internal error handling mechanisms cannot allocate memory.
    //!
    //! \return true if errors have been dropped due to overflowing the error stack.
    //!
    virtual bool hasOverflowed() const noexcept = 0;

    //!
    //! \brief Clear the error stack on the error recorder.
    //!
    //! Removes all the tracked errors by the error recorder.  This function must guarantee that after
    //! this function is called, and as long as no error occurs, the next call to getNbErrors will return
    //! zero.
    //!
    //! \see getNbErrors
    //!
    virtual void clear() noexcept = 0;

    // API’s used by TensorRT to report Error information to the application.

    //!
    //! \brief report an error to the error recorder with the corresponding enum and description.
    //!
    //! \param val The error code enum that is being reported.
    //! \param desc The string description of the error.
    //!
    //! Report an error to the user that has a given value and human readable description. The function returns false
    //! if processing can continue, which implies that the reported error is not fatal. This does not guarantee that
    //! processing continues, but provides a hint to TensorRT.
    //!
    //! \return True if the error is determined to be fatal and processing of the current function must end.
    //!
    virtual bool reportError(ErrorCode val, ErrorDesc desc) noexcept = 0;

    //!
    //! \brief Increments the refcount for the current ErrorRecorder.
    //!
    //! Increments the reference count for the object by one and returns the current value.
    //! This reference count allows the application to know that an object inside of TensorRT has
    //! taken a reference to the ErrorRecorder. If the ErrorRecorder is released before the
    //! reference count hits zero, then behavior in TensorRT is undefined. It is strongly recommended
    //! that the increment is an atomic operation. TensorRT guarantees that each incRefCount called on
    //! an objects construction is paired with a decRefCount call when an object is destructed.
    //!
    //! \return The current reference counted value.
    //!
    virtual RefCount incRefCount() noexcept = 0;

    //!
    //! \brief Decrements the refcount for the current ErrorRecorder.
    //!
    //! Decrements the reference count for the object by one and returns the current value. It is undefined behavior
    //! to call decRefCount when RefCount is zero. If the ErrorRecorder is destroyed before the reference count
    //! hits zero, then behavior in TensorRT is undefined. It is strongly recommended that the decrement is an
    //! atomic operation. TensorRT guarantees that each decRefCount called when an object is destructed is
    //! paired with a incRefCount call when that object was constructed.
    //!
    //! \return The current reference counted value.
    //!
    virtual RefCount decRefCount() noexcept = 0;

}; // class IErrorRecorder

} // namespace nvinfer1

//!
//! Internal C entry point for creating safe::IRuntime.
//! @private
//!
extern "C" TENSORRTAPI void* createSafeInferRuntime_INTERNAL(void* logger, int version);

//!
//! \brief Return the logger object.
//!
extern "C" TENSORRTAPI nvinfer1::ILogger* getLogger();

//!
//! \brief Return the library version number.
//!
//! The format is as for TENSORRT_VERSION: (TENSORRT_MAJOR * 1000) + (TENSORRT_MINOR * 100) + TENSOR_PATCH.
//!
extern "C" TENSORRTAPI int getInferLibVersion();

//!
//! \brief Return the plugin registry
//!
extern "C" TENSORRTAPI nvinfer1::IPluginRegistry* getPluginRegistry();

namespace nvinfer1
{

//!
//! \brief Register the plugin creator to the registry
//! The static registry object will be instantiated when the plugin library is
//! loaded. This static object will register all creators available in the
//! library to the registry.
//!
template <typename T>
class PluginRegistrar
{
public:
    PluginRegistrar() { getPluginRegistry()->registerCreator(instance, ""); }
private:
    T instance{};
};

#define REGISTER_TENSORRT_PLUGIN(name) \
    static nvinfer1::PluginRegistrar<name> pluginRegistrar##name {}

} // namespace nvinfer1

#endif // NV_INFER_RUNTIME_COMMON_H
