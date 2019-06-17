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

#ifndef NV_INFER_H
#define NV_INFER_H

#include <cstddef>
#include <cstdint>

#define NV_TENSORRT_MAJOR 5 //!< TensorRT major version.
#define NV_TENSORRT_MINOR 1 //!< TensorRT minor version.
#define NV_TENSORRT_PATCH 5 //!< TensorRT patch version.
#define NV_TENSORRT_BUILD 0 //!< TensorRT build number.

#define NV_TENSORRT_SONAME_MAJOR 5 //!< Shared object library major version number.
#define NV_TENSORRT_SONAME_MINOR 1 //!< Shared object library minor version number.
#define NV_TENSORRT_SONAME_PATCH 5 //!< Shared object library patch version number.

#if __cplusplus > 201103L
#define _TENSORRT_FINAL final
#define _TENSORRT_OVERRIDE override
#else
#define _TENSORRT_FINAL
#define _TENSORRT_OVERRIDE
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

//!
//! \mainpage
//!
//! This is the API documentation for the NVIDIA TensorRT library. It provides information on individual functions, classes
//! and methods. Use the index on the left to navigate the documentation.
//!
//! Please see the accompanying user guide and samples for higher-level information and general advice on using TensorRT.
//! TensorRT Versioning follows Semantic Versioning Guidelines specified here: https://semver.org/
//!

//!
//! \file NvInfer.h
//!
//! This is the top-level API file for TensorRT.
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

template <typename T>
inline int EnumMax(); //!< Maximum number of elements in an enumeration type.

//!
//! \enum DataType
//! \brief The type of weights and tensors.
//!
enum class DataType : int
{
    kFLOAT = 0, //!< FP32 format.
    kHALF = 1,  //!< FP16 format.
    kINT8 = 2,  //!< quantized INT8 format.
    kINT32 = 3  //!< INT32 format.
};

template <>
inline int EnumMax<DataType>()
{
    return 4;
} //!< Maximum number of elements in DataType enum. \see DataType

//!
//! \enum DeviceType
//! \brief The device that this layer/network will execute on.
//!
//!
enum class DeviceType : int
{
    kGPU, //!< GPU Device
    kDLA, //!< DLA Core
};
template <>
inline int EnumMax<DeviceType>()
{
    return 2;
} //!< Maximum number of elements in DeviceType enum. \see DeviceType

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
inline int EnumMax<DimensionType>()
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
class Dims
{
public:
    static const int MAX_DIMS = 8; //!< The maximum number of dimensions supported for a tensor.
    int nbDims;                    //!< The number of dimensions.
    int d[MAX_DIMS];               //!< The extent of each dimension.
    DimensionType type[MAX_DIMS];  //!< The type of each dimension.
};

//!
//! \class Dims2
//! \brief Descriptor for two-dimensional data.
//!
class Dims2 : public Dims
{
public:
    //!
    //! \brief Construct an empty Dims2 object.
    //!
    Dims2()
    {
        nbDims = 2;
        d[0] = d[1] = 0;
    }

    //!
    //! \brief Construct a Dims2 from 2 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //!
    Dims2(int d0, int d1)
    {
        nbDims = 2;
        d[0] = d0;
        d[1] = d1;
    }
};

//!
//! \class DimsHW
//! \brief Descriptor for two-dimensional spatial data.
//!
class DimsHW : public Dims2
{
public:
    //!
    //! \brief Construct an empty DimsHW object.
    //!
    DimsHW()
        : Dims2()
    {
        type[0] = type[1] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Construct a DimsHW given height and width.
    //!
    //! \param Height the height of the data
    //! \param Width the width of the data
    //!
    DimsHW(int height, int width)
        : Dims2(height, width)
    {
        type[0] = type[1] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int& h() { return d[0]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int h() const { return d[0]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int& w() { return d[1]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int w() const { return d[1]; }
};

//!
//! \class Dims3
//! \brief Descriptor for three-dimensional data.
//!
class Dims3 : public Dims
{
public:
    //!
    //! \brief Construct an empty Dims3 object.
    //!
    Dims3()
    {
        nbDims = 3;
        d[0] = d[1] = d[2] = 0;
    }

    //!
    //! \brief Construct a Dims3 from 3 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //! \param d2 The third element.
    //!
    Dims3(int d0, int d1, int d2)
    {
        nbDims = 3;
        d[0] = d0;
        d[1] = d1;
        d[2] = d2;
    }
};

//!
//! \class DimsCHW
//! \brief Descriptor for data with one channel dimension and two spatial dimensions.
//!
class DimsCHW : public Dims3
{
public:
    //!
    //! \brief Construct an empty DimsCHW object.
    //!
    DimsCHW()
        : Dims3()
    {
        type[0] = DimensionType::kCHANNEL;
        type[1] = type[2] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Construct a DimsCHW given channel count, height and width.
    //!
    //! \param channels The channel count.
    //! \param height The height of the data.
    //! \param width The width of the data.
    //!
    DimsCHW(int channels, int height, int width)
        : Dims3(channels, height, width)
    {
        type[0] = DimensionType::kCHANNEL;
        type[1] = type[2] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int& c() { return d[0]; }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int c() const { return d[0]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int& h() { return d[1]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int h() const { return d[1]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int& w() { return d[2]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int w() const { return d[2]; }
};

//!
//! \class Dims4
//! \brief Descriptor for four-dimensional data.
//!
class Dims4 : public Dims
{
public:
    //!
    //! \brief Construct an empty Dims2 object.
    //!
    Dims4()
    {
        nbDims = 4;
        d[0] = d[1] = d[2] = d[3] = 0;
    }

    //!
    //! \brief Construct a Dims4 from 4 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //! \param d2 The third element.
    //! \param d3 The fourth element.
    //!
    Dims4(int d0, int d1, int d2, int d3)
    {
        nbDims = 4;
        d[0] = d0;
        d[1] = d1;
        d[2] = d2;
        d[3] = d3;
    }
};

//!
//! \class DimsNCHW
//! \brief Descriptor for data with one index dimension, one channel dimension and two spatial dimensions.
//!
class DimsNCHW : public Dims4
{
public:
    //!
    //! \brief Construct an empty DimsNCHW object.
    //!
    DimsNCHW()
        : Dims4()
    {
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = type[3] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Construct a DimsNCHW given batch size, channel count, height and width.
    //!
    //! \param batchSize The batch size (commonly denoted N).
    //! \param channels The channel count.
    //! \param height The height of the data.
    //! \param width The width of the data.
    //!
    DimsNCHW(int batchSize, int channels, int height, int width)
        : Dims4(batchSize, channels, height, width)
    {
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = type[3] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Get the index count.
    //!
    //! \return The index count.
    //!
    int& n() { return d[0]; }

    //!
    //! \brief Get the index count.
    //!
    //! \return The index count.
    //!
    int n() const { return d[0]; }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int& c() { return d[1]; }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int c() const { return d[1]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int& h() { return d[2]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int h() const { return d[2]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int& w() { return d[3]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int w() const { return d[3]; }
};

//!
//! \class Weights
//!
//! \brief An array of weights used as a layer parameter.
//!
//! The weights are held by reference until the engine has been built. Therefore the data referenced
//! by \p values field should be preserved until the build is complete.
//!
class Weights
{
public:
    DataType type;      //!< The type of the weights.
    const void* values; //!< The weight values, in a contiguous array.
    int64_t count;      //!< The number of weights in the array.
};

//!
//! \class IHostMemory
//!
//! \brief Class to handle library allocated memory that is accessible to the user.
//!
//! The memory allocated via the host memory object is owned by the library and will
//! be de-allocated when the destroy method is called.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IHostMemory
{
public:
    virtual void* data() const = 0;       //!< A pointer to the raw data that is owned by the library.
    virtual std::size_t size() const = 0; //!< The size in bytes of the data that was allocated.
    virtual DataType type() const = 0;    //!< The type of the memory that was allocated.
    virtual void destroy() = 0;           //!< Destroy the allocated memory.
protected:
    virtual ~IHostMemory() {}
};

//!
//! \enum LayerType
//!
//! \brief The type values of layer classes.
//!
//! \see ILayer::getType()
//!
enum class LayerType : int
{
    kCONVOLUTION = 0,      //!< Convolution layer.
    kFULLY_CONNECTED = 1,  //!< Fully connected layer.
    kACTIVATION = 2,       //!< Activation layer.
    kPOOLING = 3,          //!< Pooling layer.
    kLRN = 4,              //!< LRN layer.
    kSCALE = 5,            //!< Scale layer.
    kSOFTMAX = 6,          //!< SoftMax layer.
    kDECONVOLUTION = 7,    //!< Deconvolution layer.
    kCONCATENATION = 8,    //!< Concatenation layer.
    kELEMENTWISE = 9,      //!< Elementwise layer.
    kPLUGIN = 10,          //!< Plugin layer.
    kRNN = 11,             //!< RNN layer.
    kUNARY = 12,           //!< UnaryOp operation Layer.
    kPADDING = 13,         //!< Padding layer.
    kSHUFFLE = 14,         //!< Shuffle layer.
    kREDUCE = 15,          //!< Reduce layer.
    kTOPK = 16,            //!< TopK layer.
    kGATHER = 17,          //!< Gather layer.
    kMATRIX_MULTIPLY = 18, //!< Matrix multiply layer.
    kRAGGED_SOFTMAX = 19,  //!< Ragged softmax layer.
    kCONSTANT = 20,        //!< Constant layer.
    kRNN_V2 = 21,          //!< RNNv2 layer.
    kIDENTITY = 22,        //!< Identity layer.
    kPLUGIN_V2 = 23,       //!< PluginV2 layer.
    kSLICE = 24            //!< Slice layer.
};

template <>
inline int EnumMax<LayerType>()
{
    return 25;
} //!< Maximum number of elements in LayerType enum. \see LayerType

//!
//! \enum TensorLocation
//! \brief The location for tensor data storage, device or host.
//!
enum class TensorLocation : int
{
    kDEVICE = 0, //!< Data stored on device.
    kHOST = 1    //!< Data stored on host.
};

template <>
inline int EnumMax<TensorLocation>()
{
    return 2;
} //!< Maximum number of elements in TensorLocation enum. \see TensorLocation

//!
//! \class ITensor
//!
//! \brief A tensor in a network definition.
//!
//! to remove a tensor from a network definition, use INetworkDefinition::removeTensor()
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ITensor
{
public:
    //!
    //! \brief Set the tensor name.
    //!
    //! For a network input, the name is assigned by the application. For tensors which are layer outputs,
    //! a default name is assigned consisting of the layer name followed by the index of the output in brackets.
    //!
    //! This method copies the name string.
    //!
    //! \param name The name.
    //!
    //! \see getName()
    //!
    virtual void setName(const char* name) = 0;

    //!
    //! \brief Get the tensor name.
    //!
    //! \return The name, as a pointer to a NULL-terminated character sequence.
    //!
    //! \see setName()
    //!
    virtual const char* getName() const = 0;

    //!
    //! \brief Set the dimensions of a tensor.
    //!
    //! For a network input the name is assigned by the application. For a network output it is computed based on
    //! the layer parameters and the inputs to the layer. If a tensor size or a parameter is modified in the network,
    //! the dimensions of all dependent tensors will be recomputed.
    //!
    //! This call is only legal for network input tensors, since the dimensions of layer output tensors are inferred based on
    //! layer inputs and parameters.
    //!
    //! \param dimensions The dimensions of the tensor.
    //!
    //! \see getDimensions()
    //!
    virtual void setDimensions(Dims dimensions) = 0; // only valid for input tensors

    //!
    //! \brief Get the dimensions of a tensor.
    //!
    //! \return The dimensions of the tensor.
    //!
    //! \see setDimensions()
    //!
    virtual Dims getDimensions() const = 0;

    //!
    //! \brief Set the data type of a tensor.
    //!
    //! \param type The data type of the tensor.
    //!
    //! The type is unchanged if the type is
    //! invalid for the given tensor.
    //!
    //! If the tensor is a network input or output,
    //! then the tensor type cannot be DataType::kINT8.
    //!
    //! \see getType()
    //!
    virtual void setType(DataType type) = 0;

    //!
    //! \brief Get the data type of a tensor.
    //!
    //! \return The data type of the tensor.
    //!
    //! \see setType()
    //!
    virtual DataType getType() const = 0;

    //!
    //! \brief Set dynamic range for the tensor
    //!
    //! Currently, only symmetric ranges are supported.
    //! Therefore, the larger of the absolute values of the provided bounds is used.
    //!
    //! \return Whether the dynamic range was set successfully.
    //!
    //! Requires that min and max be finite, and min <= max.
    //!
    virtual bool setDynamicRange(float min, float max) = 0;

    //!
    //! \brief Get dynamic range for the tensor
    //!
    //! \return maximal absolute value of the dynamic range, -1.0f if no dynamic range is set.
    //!
    //! \deprecated This interface is superceded by getDynamicRangeMin and getDynamicRangeMax.
    //!
    virtual float getDynamicRange() const = 0;

    //!
    //! \brief Whether the tensor is a network input.
    //!
    virtual bool isNetworkInput() const = 0;

    //!
    //! \brief Whether the tensor is a network output.
    //!
    virtual bool isNetworkOutput() const = 0;

protected:
    virtual ~ITensor() {}

public:
    //!
    //! \brief Set whether to enable broadcast of tensor across the batch.
    //!
    //! When a tensor is broadcast across a batch, it has the same value for every member in the batch.
    //! Memory is only allocated once for the single member.
    //!
    //! This method is only valid for network input tensors, since the flags of layer output tensors are inferred based on
    //! layer inputs and parameters.
    //! If this state is modified for a tensor in the network, the states of all dependent tensors will be recomputed.
    //!
    //! \param broadcastAcrossBatch Whether to enable broadcast of tensor across the batch.
    //!
    //! \see getBroadcastAcrossBatch()
    //!
    virtual void setBroadcastAcrossBatch(bool broadcastAcrossBatch) = 0;

    //!
    //! \brief Check if tensor is broadcast across the batch.
    //!
    //! When a tensor is broadcast across a batch, it has the same value for every member in the batch.
    //! Memory is only allocated once for the single member.
    //!
    //! \return True if tensor is broadcast across the batch, false otherwise.
    //!
    //! \see setBroadcastAcrossBatch()
    //!
    virtual bool getBroadcastAcrossBatch() const = 0;

    //!
    //! \brief Get the storage location of a tensor.
    //! \return The location of tensor data.
    //! \see setLocation()
    //!
    virtual TensorLocation getLocation() const = 0;

    //!
    //! \brief Set the storage location of a tensor
    //! \param location the location of tensor data
    //!
    //! Only input tensors for storing sequence lengths for RNNv2 are supported.
    //! Using host storage for layers that do not support it will generate
    //! errors at build time.
    //!
    //! \see getLocation()
    //!
    virtual void setLocation(TensorLocation location) = 0;

    //!
    //! \brief Query whether dynamic range is set.
    //!
    //! \return True if dynamic range is set, false otherwise.
    //!
    virtual bool dynamicRangeIsSet() const = 0;

    //!
    //! \brief Undo effect of setDynamicRange.
    //!
    virtual void resetDynamicRange() = 0;

    //!
    //! \brief Get minimum of dynamic range.
    //!
    //! \return Minimum of dynamic range, or quiet NaN if range was not set.
    //!
    virtual float getDynamicRangeMin() const = 0;

    //!
    //! \brief Get maximum of dynamic range.
    //!
    //! \return Maximum of dynamic range, or quiet NaN if range was not set.
    //!
    virtual float getDynamicRangeMax() const = 0;
};

//!
//! \class ILayer
//!
//! \brief Base class for all layer classes in a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ILayer
{
public:
    //!
    //! \brief Return the type of a layer.
    //!
    //! \see LayerType
    //!
    virtual LayerType getType() const = 0;

    //!
    //! \brief Set the name of a layer.
    //!
    //! This method copies the name string.
    //!
    //! \see getName()
    //!
    virtual void setName(const char* name) = 0;

    //!
    //! \brief Return the name of a layer.
    //!

    //! \see setName()
    //!
    virtual const char* getName() const = 0;

    //!
    //! \brief Get the number of inputs of a layer.
    //!
    virtual int getNbInputs() const = 0;

    //!
    //! \brief Get the layer input corresponding to the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range or the tensor is optional(\ref IRNNLayer and \ref IRNNv2Layer).
    //!
    virtual ITensor* getInput(int index) const = 0;

    //!
    //! \brief Get the number of outputs of a layer.
    //!
    virtual int getNbOutputs() const = 0;

    //!
    //! \brief Get the layer output corresponding to the given index.
    //!
    //! \return The indexed output tensor, or nullptr if the index is out of range or the tensor is optional(\ref IRNNLayer and \ref IRNNv2Layer).
    //!
    virtual ITensor* getOutput(int index) const = 0;

    //!
    //! \brief replace an input of this layer with a specific tensor
    //!
    //! Note that this method cannot change the number of inputs to a layer.  The index argument must be less
    //! than the value of getNbInputs()
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    virtual void setInput(int index, ITensor& tensor) = 0;

    //!
    //! \brief Set the computational precision of this layer
    //!
    //! setting the precision forces TensorRT to choose implementations which run at this precision. If precision is not set,
    //! TensorRT will select the computational precision based on performance considerations and the flags specified to the builder.
    //!
    //! \param precision the computational precision.
    //!
    //! \see getPrecision() precisionIsSet() resetPrecision()

    virtual void setPrecision(DataType dataType) = 0;

    //!
    //! \brief get the computational precision of this layer
    //!
    //! \return the computational precision
    //!
    //! \see setPrecision() precisionIsSet() resetPrecision()

    virtual DataType getPrecision() const = 0;

    //!
    //! \brief whether the computational precision has been set for this layer
    //!
    //! \return whether the computational precision has been explicitly set
    //!
    //! \see setPrecision() getPrecision() resetPrecision()

    virtual bool precisionIsSet() const = 0;

    //!
    //! \brief reset the computational precision for this layer
    //!
    //! \see setPrecision() getPrecision() precisionIsSet()

    virtual void resetPrecision() = 0;

    //!
    //! \brief Set the output type of this layer
    //!
    //! setting the output type constrains TensorRT to choose implementations which generate output data with the given type.
    //! If it is not set, TensorRT will select the implementation based on performance considerations and the flags specified to the builder.
    //!
    //! \param index the index of the output to set
    //! \param dataType the type of the output
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()

    virtual void setOutputType(int index, DataType dataType) = 0;

    //!
    //! \brief get the output type of this layer
    //!
    //! \param index the index of the output
    //! \return the output precision. If no precision has been set, DataType::kFLOAT will be returned,
    //!         unless the output type is inherently DataType::kINT32.
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()

    virtual DataType getOutputType(int index) const = 0;

    //!
    //! \brief whether the output type has been set for this layer
    //!
    //! \param index the index of the output
    //! \return whether the output type has been explicitly set
    //!
    //! \see setOutputType() getOutputType() resetOutputType()

    virtual bool outputTypeIsSet(int index) const = 0;

    //!
    //! \brief reset the output type for this layer
    //!
    //! \param index the index of the output
    //!
    //! \see setOutputType() getOutputType() outputTypeIsSet()

    virtual void resetOutputType(int index) = 0;

protected:
    virtual ~ILayer() {}
};

//!
//! \enum PaddingMode
//!
//! \brief Enumerates the modes of padding to perform in convolution, deconvolution and pooling layer,
//! padding mode gets precedence if setPaddingMode() and setPrePadding() are also used.
//!
//! kEXPLICIT* padding is to use explicit padding.
//! kSAME* padding is to implicitly calculate padding to keep output dim to be the "same" with input dim. For convolution and pooling,
//! output dim is ceil(input dim, stride), for deconvolution it is inverse, then use the output dim to calculate padding size.
//! kCAFFE* padding is symmetric padding.
//!
enum class PaddingMode : int
{
    kEXPLICIT_ROUND_DOWN = 0, //!< Use explicit padding, rounding output size down.
    kEXPLICIT_ROUND_UP = 1,   //!< Use explicit padding, rounding output size up.
    kSAME_UPPER = 2,          //!< Use SAME padding with prePadding <= postPadding.
    kSAME_LOWER = 3,          //!< Use SAME padding, with prePadding >= postPadding.
    kCAFFE_ROUND_DOWN = 4,    //!< Use CAFFE padding, rounding output size down.
    kCAFFE_ROUND_UP = 5       //!< Use CAFFE padding, rounding output size up.
};

template <>
inline int EnumMax<PaddingMode>()
{
    return 6;
} //!< Maximum number of elements in PaddingMode enum. \see PaddingMode

//!
//! \class IConvolutionLayer
//!
//! \brief A convolution layer in a network definition.
//!
//! This layer performs a correlation operation between 3-dimensional filter with a 4-dimensional tensor to produce another 4-dimensional tensor.
//!
//! The HW output size of the convolution is set according to the \p INetworkCustomDimensions set in INetworkDefinition::setCustomConvolutionDimensions().
//!
//! An optional bias argument is supported, which adds a per-channel constant to each value in the output.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConvolutionLayer : public ILayer
{
public:
    //!
    //! \brief Set the HW kernel size of the convolution.
    //!
    //! If executing this layer on DLA, both height and width of kernel size must be in the range [1,16].
    //!
    //! \see getKernelSize()
    //!
    virtual void setKernelSize(DimsHW kernelSize) = 0;

    //!
    //! \brief Get the HW kernel size of the convolution.
    //!
    //! \see setKernelSize()
    //!
    virtual DimsHW getKernelSize() const = 0;

    //!
    //! \brief Set the number of output maps for the convolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    virtual void setNbOutputMaps(int nbOutputMaps) = 0;

    //!
    //! \brief Get the number of output maps for the convolution.
    //!
    //! \see setNbOutputMaps()
    //!
    virtual int getNbOutputMaps() const = 0;

    //!
    //! \brief Get the stride of the convolution.
    //!
    //! Default: (1,1)
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,8].
    //!
    //! \see setStride()
    //!
    virtual void setStride(DimsHW stride) = 0;

    //!
    //! \brief Get the stride of the convolution.
    //!
    virtual DimsHW getStride() const = 0;

    //!
    //! \brief Set the padding of the convolution.
    //!
    //! The input will be zero-padded by this number of elements in the height and width directions. Padding is symmetric.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPadding(DimsHW padding) = 0;

    //!
    //! \brief Get the padding of the convolution. If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPadding()
    //!
    virtual DimsHW getPadding() const = 0;

    //!
    //! \brief Set the number of groups for a convolution.
    //!
    //! The input tensor channels are  divided into \p nbGroups groups, and a convolution is executed for each group, using a filter per group. The results of the group
    //! convolutions are concatenated to form the output.
    //!
    //! \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output.
    //!
    //! Default: 1
    //!
    //! \see getNbGroups()
    //!
    virtual void setNbGroups(int nbGroups) = 0;

    //!
    //! \brief Set the number of groups for a convolution.
    //!
    //! \see setNbGroups()
    //!
    virtual int getNbGroups() const = 0;

    //!
    //! \brief Set the kernel weights for the convolution.
    //!
    //! The weights are specified as a contiguous array in \p GKCRS order, where \p G is the number of groups, \p K the number of output feature maps, \p C the number of
    //! input channels, and \p R and \p S are the height and width of the filter.
    //!
    //! \see getKernelWeights()
    //!
    virtual void setKernelWeights(Weights weights) = 0;

    //!
    //! \brief Get the kernel weights for the convolution.
    //!
    //! \see setKernelWeights()
    //!
    virtual Weights getKernelWeights() const = 0;

    //!
    //! \brief Set the bias weights for the convolution.
    //!
    //! Bias is optional. To omit bias, set the count value of the weights structure to zero.
    //!
    //! The bias is applied per-channel, so the number of weights (if non-zero) must be equal to the number of output feature maps.
    //!
    //! \see getBiasWeights()
    //!
    virtual void setBiasWeights(Weights weights) = 0;

    //!
    //! \brief Get the bias weights for the convolution.
    //!
    //! \see setBiasWeights()
    //!
    virtual Weights getBiasWeights() const = 0;

    //!
    //! \brief Set the dilation for a convolution.
    //!
    //! Default: (1,1)
    //!
    //! \see getDilation()
    //!
    virtual void setDilation(DimsHW dims) = 0;

    //!
    //! \brief Get the dilation for a convolution.
    //!
    //! \see setDilation()
    //!
    virtual DimsHW getDilation() const = 0;

protected:
    virtual ~IConvolutionLayer() {}

public:
    //!
    //! \brief Set the pre-padding.
    //!
    //! The start of input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPrePadding()
    //!
    virtual void setPrePadding(Dims padding) = 0;

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    virtual Dims getPrePadding() const = 0;

    //!
    //! \brief Set the post-padding.
    //!
    //! The end of the input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPostPadding()
    //!
    virtual void setPostPadding(Dims padding) = 0;

    //!
    //! \brief Get the post-padding.
    //!
    //! \see setPostPadding()
    //!
    virtual Dims getPostPadding() const = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode gets precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    virtual void setPaddingMode(PaddingMode paddingMode) = 0;

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    virtual PaddingMode getPaddingMode() const = 0;
};

//! \class IFullyConnectedLayer
//!
//! \brief A fully connected layer in a network definition.
//! This layer expects an input tensor of three or more non-batch dimensions.  The input is automatically
//! reshaped into an `MxV` tensor `X`, where `V` is a product of the last three dimensions and `M`
//! is a product of the remaining dimensions (where the product over 0 dimensions is defined as 1).  For example:
//!
//! - If the input tensor has shape `{C, H, W}`, then the tensor is reshaped into `{1, C*H*W}`.
//! - If the input tensor has shape `{P, C, H, W}`, then the tensor is reshaped into `{P, C*H*W}`.
//!
//! The layer then performs the following operation:
//!
//! ~~~
//! Y := matmul(X, W^T) + bias
//! ~~~
//!
//! Where `X` is the `MxV` tensor defined above, `W` is the `KxV` weight tensor
//! of the layer, and `bias` is a row vector size `K` that is broadcasted to
//! `MxK`.  `K` is the number of output channels, and configurable via
//! setNbOutputChannels().  If `bias` is not specified, it is implicitly `0`.
//!
//! The `MxK` result `Y` is then reshaped such that the last three dimensions are `{K, 1, 1}` and
//! the remaining dimensions match the dimensions of the input tensor. For example:
//!
//! - If the input tensor has shape `{C, H, W}`, then the output tensor will have shape `{K, 1, 1}`.
//! - If the input tensor has shape `{P, C, H, W}`, then the output tensor will have shape `{P, K, 1, 1}`.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IFullyConnectedLayer : public ILayer
{
public:
    //!
    //! \brief Set the number of output channels `K` from the fully connected layer.
    //!
    //! If executing this layer on DLA, number of output channels must in the range [1,8192].
    //!
    //! \see getNbOutputChannels()
    //!
    virtual void setNbOutputChannels(int nbOutputs) = 0;

    //!
    //! \brief Get the number of output channels `K` from the fully connected layer.
    //!
    //! \see setNbOutputChannels()
    //!
    virtual int getNbOutputChannels() const = 0;

    //!
    //! \brief Set the kernel weights, given as a `KxC` matrix in row-major order.
    //!
    //! \see getKernelWeights()
    //!
    virtual void setKernelWeights(Weights weights) = 0;

    //!
    //! \brief Get the kernel weights.
    //!
    //! \see setKernelWeights()
    //!
    virtual Weights getKernelWeights() const = 0;

    //!
    //! \brief Set the bias weights.
    //!
    //! Bias is optional. To omit bias, set the count value in the weights structure to zero.
    //!
    //! \see getBiasWeightsWeights()
    //!
    virtual void setBiasWeights(Weights weights) = 0;

    //!
    //! \brief Get the bias weights.
    //!
    //! \see setBiasWeightsWeights()
    //!
    virtual Weights getBiasWeights() const = 0;

protected:
    virtual ~IFullyConnectedLayer() {}
};

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
    kTHRESHOLDED_RELU = 11 //!< Thresholded ReLU activation: x>alpha : x : 0
};

template <>
inline int EnumMax<ActivationType>()
{
    return 12;
} //!< Maximum number of elements in ActivationType enum. \see ActivationType

//!
//! \class IActivationLayer
//!
//! \brief An Activation layer in a network definition.
//!
//! This layer applies a per-element activation function to its input.
//!
//! The output has the same shape as the input.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IActivationLayer : public ILayer
{
public:
    //!
    //! \brief Set the type of activation to be performed.
    //!
    //! \see getActivationType(), ActivationType
    //!
    virtual void setActivationType(ActivationType type) = 0;

    //!
    //! \brief Get the type of activation to be performed.
    //!
    //! \see setActivationType(), ActivationType
    //!
    virtual ActivationType getActivationType() const = 0;

protected:
    virtual ~IActivationLayer() {}
public:
    //!
    //! \brief Set the alpha parameter (must be finite).
    //!
    //! This parameter is used by the following activations:
    //! LeakyRelu, Elu, Selu, Softplus, Clip, HardSigmoid, ScaledTanh,
    //! ThresholdedRelu.
    //!
    //! It is ignored by the other activations.
    //!
    //! \see getAlpha(), setBeta()
    virtual void setAlpha(float alpha) = 0;

    //!
    //! \brief Set the beta parameter (must be finite).
    //!
    //! This parameter is used by the following activations:
    //! Selu, Softplus, Clip, HardSigmoid, ScaledTanh.
    //!
    //! It is ignored by the other activations.
    //!
    //! \see getBeta(), setAlpha()
    virtual void setBeta(float beta) = 0;

    //!
    //! \brief Get the alpha parameter.
    //!
    //! \see getBeta(), setAlpha()
    virtual float getAlpha() const = 0;

    //!
    //! \brief Get the beta parameter.
    //!
    //! \see getAlpha(), setBeta()
    virtual float getBeta() const = 0;
};

//!
//! \enum PoolingType
//!
//! \brief The type of pooling to perform in a pooling layer.
//!
enum class PoolingType : int
{
    kMAX = 0,              // Maximum over elements
    kAVERAGE = 1,          // Average over elements. If the tensor is padded, the count includes the padding
    kMAX_AVERAGE_BLEND = 2 // Blending between the max pooling and average pooling: (1-blendFactor)*maxPool + blendFactor*avgPool
};

template <>
inline int EnumMax<PoolingType>()
{
    return 3;
} //!< Maximum number of elements in PoolingType enum. \see PoolingType

//! \class IPoolingLayer
//!
//! \brief A Pooling layer in a network definition.
//!
//! The layer applies a reduction operation within a window over the input.
//!
//! The output size is determined from the input size using the formula set by INetworkDefinition::setCustomPoolingDimensions().
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPoolingLayer : public ILayer
{
public:
    //!
    //! \brief Set the type of activation to be performed.
    //!
    //! DLA only supports kMAX and kAVERAGE.
    //!
    //! \see getPoolingType(), PoolingType
    //!
    virtual void setPoolingType(PoolingType type) = 0;

    //!
    //! \brief Get the type of activation to be performed.
    //!
    //! \see setPoolingType(), PoolingType
    //!
    virtual PoolingType getPoolingType() const = 0;

    //!
    //! \brief Set the window size for pooling.
    //!
    //! If executing this layer on DLA, both height and width of window size must be in the range [1,8].
    //!
    //! \see getWindowSize()
    //!
    virtual void setWindowSize(DimsHW windowSize) = 0;

    //!
    //! \brief Get the window size for pooling.
    //!
    //! \see setWindowSize()
    //!
    virtual DimsHW getWindowSize() const = 0;

    //!
    //! \brief Set the stride for pooling.
    //!
    //! Default: 1
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,16].
    //!
    //! \see getStride()
    //!
    virtual void setStride(DimsHW stride) = 0;

    //!
    //! \brief Get the stride for pooling.
    //!
    //! \see setStride()
    //!
    virtual DimsHW getStride() const = 0;

    //!
    //! \brief Set the padding for pooling.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,7].
    //!
    //! \see getStride()
    //!
    virtual void setPadding(DimsHW padding) = 0;

    //!
    //! \brief Get the padding for pooling.
    //!
    //! Default: 0
    //!
    //! \see getStride()
    //!
    virtual DimsHW getPadding() const = 0;

    //!
    //! \brief Set the blending factor for the max_average_blend mode: max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
    //! blendFactor is a user value in [0,1] with the default value of 0.0
    //! This value only applies for the kMAX_AVERAGE_BLEND mode.
    //!
    //! \see getBlendFactor()
    //!
    virtual void setBlendFactor(float blendFactor) = 0;

    //!
    //! \brief Get the blending factor for the max_average_blend mode: max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
    //! blendFactor is a user value in [0,1] with the default value of 0.0
    //! In modes other than kMAX_AVERAGE_BLEND, blendFactor is ignored.
    //!
    //! \see setBlendFactor()
    //!
    virtual float getBlendFactor() const = 0;

    //!
    //! \brief Set whether average pooling uses as a denominator the overlap area between the window and the unpadded input.
    //! If this is not set, the denominator is the overlap between the pooling window and the padded input.
    //!
    //! Default: true
    //!
    //! \see getAverageCountExcludesPadding()
    //!
    virtual void setAverageCountExcludesPadding(bool exclusive) = 0;

    //!
    //! \brief Get whether exclusive pooling uses as a denominator the overlap area betwen the window and the unpadded input.
    //!
    //! \see setAverageCountExcludesPadding()
    //!
    virtual bool getAverageCountExcludesPadding() const = 0;

protected:
    virtual ~IPoolingLayer() {}

public:
    //!
    //! \brief Set the pre-padding.
    //!
    //! The start of input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPrePadding(Dims padding) = 0;

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    virtual Dims getPrePadding() const = 0;

    //!
    //! \brief Set the post-padding.
    //!
    //! The end of the input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPostPadding(Dims padding) = 0;

    //!
    //! \brief Get the padding.
    //!
    //! \see setPadding()
    //!
    virtual Dims getPostPadding() const = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode gets precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    virtual void setPaddingMode(PaddingMode paddingMode) = 0;

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    virtual PaddingMode getPaddingMode() const = 0;
};

//!
//! \class ILRNLayer
//!
//! \brief A LRN layer in a network definition.
//!
//! The output size is the same as the input size.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ILRNLayer : public ILayer
{
public:
    //!
    //! \brief Set the LRN window size.
    //!
    //! The window size must be odd and in the range of [1, 15].
    //! \see setWindowStride()
    //!
    virtual void setWindowSize(int windowSize) = 0;

    //!
    //! \brief Get the LRN window size.
    //!
    //! \see getWindowStride()
    //!
    virtual int getWindowSize() const = 0;

    //!
    //! \brief Set the LRN alpha value.
    //!
    //! The valid range is [-1e20, 1e20].
    //! \see getAlpha()
    //!
    virtual void setAlpha(float alpha) = 0;

    //!
    //! \brief Get the LRN alpha value.
    //!
    //! \see setAlpha()
    //!
    virtual float getAlpha() const = 0;

    //!
    //! \brief Set the LRN beta value.
    //!
    //! The valid range is [0.01, 1e5f].
    //! \see getBeta()
    //!
    virtual void setBeta(float beta) = 0;

    //!
    //! \brief Get the LRN beta value.
    //!
    //! \see setBeta()
    //!
    virtual float getBeta() const = 0;

    //!
    //! \brief Set the LRN K value.
    //!
    //! The valid range is [1e-5, 1e10].
    //! \see getK()
    //!
    virtual void setK(float k) = 0;

    //!
    //! \brief Get the LRN K value.
    //!
    //! \see setK()
    //!
    virtual float getK() const = 0;

protected:
    virtual ~ILRNLayer() {}
};

//!
//! \brief Controls how shift, scale and power are applied in a Scale layer.
//!
//! \see IScaleLayer
//!
enum class ScaleMode : int
{
    kUNIFORM = 0,    //!< Identical coefficients across all elements of the tensor.
    kCHANNEL = 1,    //!< Per-channel coefficients. The channel dimension is assumed to be the third to last dimension
    kELEMENTWISE = 2 //!< Elementwise coefficients.
};

template <>
inline int EnumMax<ScaleMode>()
{
    return 3;
} //!< Maximum number of elements in ScaleMode enum. \see ScaleMode

//!
//! \class IScaleLayer
//!
//! \brief A Scale layer in a network definition.
//!
//! This layer applies a per-element computation to its input:
//!
//! \p output = (\p input* \p scale + \p shift)^ \p power
//!
//! The coefficients can be applied on a per-tensor, per-channel, or per-element basis.
//!
//! \note If the number of weights is 0, then a default value is used for shift, power, and scale.
//!       The default shift is 0, the default power is 1, and the default scale is 1.
//!
//! The output size is the same as the input size.
//!
//! \note The input tensor for this layer is required to have a minimum of 3 dimensions.
//!
//! \see ScaleMode
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IScaleLayer : public ILayer
{
public:
    //!
    //! \brief Set the scale mode.
    //!
    //! \see getMode()
    //!
    virtual void setMode(ScaleMode mode) = 0;

    //!
    //! \brief Set the scale mode.
    //!
    //! \see setMode()
    //!
    virtual ScaleMode getMode() const = 0;

    //!
    //! \brief Set the shift value.
    //!
    //! \see getShift()
    //!
    virtual void setShift(Weights shift) = 0;

    //!
    //! \brief Get the shift value.
    //!
    //! \see setShift()
    //!
    virtual Weights getShift() const = 0;

    //!
    //! \brief Set the scale value.
    //!
    //! \see getScale()
    //!
    virtual void setScale(Weights scale) = 0;

    //!
    //! \brief Get the scale value.
    //!
    //! \see setScale()
    //!
    virtual Weights getScale() const = 0;

    //!
    //! \brief Set the power value.
    //!
    //! \see getPower()
    //!
    virtual void setPower(Weights power) = 0;

    //!
    //! \brief Get the power value.
    //!
    //! \see setPower()
    //!
    virtual Weights getPower() const = 0;

protected:
    virtual ~IScaleLayer() {}
};

//!
//! \class ISoftMaxLayer
//!
//! \brief A Softmax layer in a network definition.
//!
//! This layer applies a per-channel softmax to its input.
//!
//! The output size is the same as the input size.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISoftMaxLayer : public ILayer
{
protected:
    virtual ~ISoftMaxLayer() {}
public:
    //!
    //! \brief Set the axis along which softmax is computed. Currently, only one axis can be set.
    //!
    //! The axis is specified by setting the bit corresponding to the axis, after excluding the batch dimension, to 1.
    //! Let's say we have an NCHW tensor as input (three non-batch dimensions).
    //! Bit 0 corresponds to the C dimension boolean.
    //! Bit 1 corresponds to the H dimension boolean.
    //! Bit 2 corresponds to the W dimension boolean.
    //! For example, to perform softmax on axis R of a NPQRCHW input, set bit 2.
    //!
    //! By default, softmax is performed on the axis which is the number of non-batch axes minus three. It is 0 if there are fewer than 3 non-batch axes.
    //! For example, if the input is NCHW, the default axis is C. If the input is NHW, then the default axis is H.
    //!
    //! \param axes The axis along which softmax is computed.
    //!
    virtual void setAxes(uint32_t axes) = 0;

    //!
    //! \brief Get the axis along which softmax occurs.
    //!
    //! \see setAxes()
    //!
    virtual uint32_t getAxes() const = 0;
};

//!
//! \class IConcatenationLayer
//!
//! \brief A concatenation layer in a network definition.
//!
//! The output channel size is the sum of the channel sizes of the inputs.
//! The other output sizes are the same as the other input sizes,
//! which must all match.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConcatenationLayer : public ILayer
{
protected:
    virtual ~IConcatenationLayer() {}

public:
    //!
    //! \brief Set the axis along which concatenation occurs.
    //!
    //! 0 is the major axis (excluding the batch dimension). The default is the number of non-batch axes in the tensor minus three (e.g.
    //! for an NCHW input it would be 0), or 0 if there are fewer than 3 non-batch axes.
    //!
    //! \param axis The axis along which concatenation occurs.
    //!
    virtual void setAxis(int axis) = 0;

    //!
    //! \brief Get the axis along which concatenation occurs.
    //!
    //! \see setAxis()
    //!
    virtual int getAxis() const = 0;
};

//!
//! \class IDeconvolutionLayer
//!
//! \brief A deconvolution layer in a network definition.
//!
//! The output size is defined using the formula set by INetworkDefinition::setDeconvolutionOutputDimensionsFormula().
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IDeconvolutionLayer : public ILayer
{
public:
    //!
    //! \brief Set the HW kernel size of the convolution.
    //!
    //! If executing this layer on DLA, both height and width of kernel size must be in the range [1,16].
    //!
    //! \see getKernelSize()
    //!
    virtual void setKernelSize(DimsHW kernelSize) = 0;

    //!
    //! \brief Get the HW kernel size of the deconvolution.
    //!
    //! \see setKernelSize()
    //!
    virtual DimsHW getKernelSize() const = 0;

    //!
    //! \brief Set the number of output feature maps for the deconvolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    virtual void setNbOutputMaps(int nbOutputMaps) = 0;

    //!
    //! \brief Get the number of output feature maps for the deconvolution.
    //!
    //! \see setNbOutputMaps()
    //!
    virtual int getNbOutputMaps() const = 0;

    //!
    //! \brief Get the stride of the deconvolution.
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,8].
    //!
    //! \see setStride()
    //!
    virtual void setStride(DimsHW stride) = 0;

    //!
    //! \brief Get the stride of the deconvolution.
    //!
    //! Default: (1,1)
    //!
    virtual DimsHW getStride() const = 0;

    //!
    //! \brief Set the padding of the deconvolution.
    //!
    //! The output will be trimmed by this number of elements on each side in the height and width directions. In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Padding is symmetric, and negative padding is not supported.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPadding(DimsHW padding) = 0;

    //!
    //! \brief Get the padding of the deconvolution.
    //!
    //! \see setPadding()
    //!
    virtual DimsHW getPadding() const = 0; // padding defaults to 0

    //!
    //! \brief Set the number of groups for a deconvolution.
    //!
    //! The input tensor channels are divided into \p nbGroups groups, and a deconvolution is executed for each group, using a filter per group. The results of the group
    //! convolutions are concatenated to form the output.
    //!
    //! \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output.
    //!
    //! Default: 1
    //!
    //! \see getNbGroups()
    //!
    virtual void setNbGroups(int nbGroups) = 0;

    //!
    //! \brief Get the number of groups for a deconvolution.
    //!
    //! \see setNbGroups()
    //!
    virtual int getNbGroups() const = 0;

    //!
    //! \brief Set the kernel weights for the deconvolution.
    //!
    //! The weights are specified as a contiguous array in \p CKRS order, where \p C the number of
    //! input channels, \p K the number of output feature maps, and \p R and \p S are the height and width of the filter.
    //!
    //! \see getWeights()
    //!
    virtual void setKernelWeights(Weights weights) = 0;

    //!
    //! \brief Get the kernel weights for the deconvolution.
    //!
    //! \see setNbGroups()
    //!
    virtual Weights getKernelWeights() const = 0;

    //!
    //! \brief Set the bias weights for the deconvolution.
    //!
    //! Bias is optional. To omit bias, set the count value of the weights structure to zero.
    //!
    //! The bias is applied per-feature-map, so the number of weights (if non-zero) must be equal to the number of output feature maps.
    //!
    //! \see getBiasWeights()
    //!
    virtual void setBiasWeights(Weights weights) = 0;

    //!
    //! \brief Get the bias weights for the deconvolution.
    //!
    //! \see getBiasWeights()
    //!
    virtual Weights getBiasWeights() const = 0;

protected:
    virtual ~IDeconvolutionLayer() {}

public:
    //!
    //! \brief Set the pre-padding.
    //!
    //! The start of input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPrePadding(Dims padding) = 0;

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    virtual Dims getPrePadding() const = 0;

    //!
    //! \brief Set the post-padding.
    //!
    //! The end of the input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPostPadding(Dims padding) = 0;

    //!
    //! \brief Get the padding.
    //!
    //! \see setPadding()
    //!
    virtual Dims getPostPadding() const = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode gets precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    virtual void setPaddingMode(PaddingMode paddingMode) = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    virtual PaddingMode getPaddingMode() const = 0;
};

//!
//! \enum ElementWiseOperation
//!
//! \brief Enumerates the binary operations that may be performed by an ElementWise layer.
//!
//! \see IElementWiseLayer
//!
enum class ElementWiseOperation : int
{
    kSUM = 0,  //!< Sum of the two elements.
    kPROD = 1, //!< Product of the two elements.
    kMAX = 2,  //!< Maximum of the two elements.
    kMIN = 3,  //!< Minimum of the two elements.
    kSUB = 4,  //!< Substract the second element from the first.
    kDIV = 5,  //!< Divide the first element by the second.
    kPOW = 6   //!< The first element to the power of the second element.
};

template <>
inline int EnumMax<ElementWiseOperation>()
{
    return 7;
} //!< Maximum number of elements in ElementWiseOperation enum. \see ElementWiseOperation

//!
//! \class IElementWiseLayer
//!
//! \brief A elementwise layer in a network definition.
//!
//! This layer applies a per-element binary operation between corresponding elements of two tensors.
//!
//! The input dimensions of the two input tensors must be equal, and the output tensor is the same size as each input.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IElementWiseLayer : public ILayer
{
public:
    //!
    //! \brief Set the binary operation for the layer.
    //!
    //! DLA supports only kSUM, kPROD, kMAX and kMIN.
    //!
    //! \see getOperation(), ElementWiseOperation
    //!
    //! \see getBiasWeights()
    //!
    virtual void setOperation(ElementWiseOperation type) = 0;

    //!
    //! \brief Get the binary operation for the layer.
    //!
    //! \see setOperation(), ElementWiseOperation
    //!
    //! \see setBiasWeights()
    //!
    virtual ElementWiseOperation getOperation() const = 0;

protected:
    virtual ~IElementWiseLayer() {}
};

//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IGatherLayer : public ILayer
{
public:
    //!
    //! \brief Set the non-batch dimension axis to gather on.
    //!  The axis must be less than the number of non-batch dimensions in the data input.
    //!
    //! \see getGatherAxis()
    //!
    virtual void setGatherAxis(int axis) = 0;

    //!
    //! \brief Get the non-batch dimension axis to gather on.
    //!
    //! \see setGatherAxis()
    //!
    virtual int getGatherAxis() const = 0;

protected:
    virtual ~IGatherLayer() {}
};

//!
//! \enum RNNOperation
//!
//! \brief Enumerates the RNN operations that may be performed by an RNN layer.
//!
//! __Equation definitions__
//!
//! In the equations below, we use the following naming convention:
//!
//! ~~~
//! t := current time step
//!
//! i := input gate
//! o := output gate
//! f := forget gate
//! z := update gate
//! r := reset gate
//! c := cell gate
//! h := hidden gate
//!
//! g[t] denotes the output of gate g at timestep t, e.g.
//! f[t] is the output of the forget gate f.
//!
//! X[t] := input tensor for timestep t
//! C[t] := cell state for timestep t
//! H[t] := hidden state for timestep t
//!
//! W[g] := W (input) parameter weight matrix for gate g
//! R[g] := U (recurrent) parameter weight matrix for gate g
//! Wb[g] := W (input) parameter bias vector for gate g
//! Rb[g] := U (recurrent) parameter bias vector for gate g
//!
//! Unless otherwise specified, all operations apply pointwise
//! to elements of each operand tensor.
//!
//! ReLU(X) := max(X, 0)
//! tanh(X) := hyperbolic tangent of X
//! sigmoid(X) := 1 / (1 + exp(-X))
//! exp(X) := e^X
//!
//! A.B denotes matrix multiplication of A and B.
//! A*B denotes pointwise multiplication of A and B.
//! ~~~
//!
//! __Equations__
//!
//! Depending on the value of RNNOperation chosen, each sub-layer of the RNN
//! layer will perform one of the following operations:
//!
//! ~~~
//! ::kRELU
//!
//!   H[t] := ReLU(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])
//!
//! ::kTANH
//!
//!   H[t] := tanh(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])
//!
//! ::kLSTM
//!
//!   i[t] := sigmoid(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])
//!   f[t] := sigmoid(W[f].X[t] + R[f].H[t-1] + Wb[f] + Rb[f])
//!   o[t] := sigmoid(W[o].X[t] + R[o].H[t-1] + Wb[o] + Rb[o])
//!   c[t] :=    tanh(W[c].X[t] + R[c].H[t-1] + Wb[c] + Rb[c])
//!
//!   C[t] := f[t]*C[t-1] + i[t]*c[t]
//!   H[t] := o[t]*tanh(C[t])
//!
//! ::kGRU
//!
//!   z[t] := sigmoid(W[z].X[t] + R[z].H[t-1] + Wb[z] + Rb[z])
//!   r[t] := sigmoid(W[r].X[t] + R[r].H[t-1] + Wb[r] + Rb[r])
//!   h[t] := tanh(W[h].X[t] + r[t]*(R[h].H[t-1] + Rb[h]) + Wb[h])
//!
//!   H[t] := (1 - z[t])*h[t] + z[t]*H[t-1]
//! ~~~
//!
//! \see IRNNLayer, IRNNv2Layer
//!
enum class RNNOperation : int
{
    kRELU = 0, //!< Single gate RNN w/ ReLU activation function.
    kTANH = 1, //!< Single gate RNN w/ TANH activation function.
    kLSTM = 2, //!< Four-gate LSTM network w/o peephole connections.
    kGRU = 3   //!< Three-gate network consisting of Gated Recurrent Units.
};

template <>
inline int EnumMax<RNNOperation>()
{
    return 4;
} //!< Maximum number of elements in RNNOperation enum. \see RNNOperation

//!
//! \enum RNNDirection
//!
//! \brief Enumerates the RNN direction that may be performed by an RNN layer.
//!
//! \see IRNNLayer, IRNNv2Layer
//!
enum class RNNDirection : int
{
    kUNIDIRECTION = 0, //!< Network iterations from first input to last input.
    kBIDIRECTION = 1   //!< Network iterates from first to last and vice versa and outputs concatenated.
};

template <>
inline int EnumMax<RNNDirection>()
{
    return 2;
} //!< Maximum number of elements in RNNDirection enum. \see RNNDirection

//!
//! \enum RNNInputMode
//!
//! \brief Enumerates the RNN input modes that may occur with an RNN layer.
//!
//! If the RNN is configured with RNNInputMode::kLINEAR, then for each gate `g` in the first layer of the RNN,
//! the input vector `X[t]` (length `E`) is left-multiplied by the gate's corresponding weight matrix `W[g]`
//! (dimensions `HxE`) as usual, before being used to compute the gate output as described by \ref RNNOperation.
//!
//! If the RNN is configured with RNNInputMode::kSKIP, then this initial matrix multiplication is "skipped"
//! and `W[g]` is conceptually an identity matrix.  In this case, the input vector `X[t]` must have length `H`
//! (the size of the hidden state).
//!
//! \see IRNNLayer, IRNNv2Layer
//!
enum class RNNInputMode : int
{
    kLINEAR = 0, //!< Perform the normal matrix multiplication in the first recurrent layer.
    kSKIP = 1    //!< No operation is performed on the first recurrent layer.
};

template <>
inline int EnumMax<RNNInputMode>()
{
    return 2;
} //!< Maximum number of elements in RNNInputMode enum. \see RNNInputMode

//!
//! \class IRNNLayer
//!
//! \brief A RNN layer in a network definition.
//!
//! This layer applies an RNN operation on the inputs.
//!
//! \deprecated This interface is superseded by IRNNv2Layer.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRNNLayer : public ILayer
{
public:
    //!
    //! \brief Get the number of layers in the RNN.
    //!
    //! \return The number of layers in the RNN.
    //!
    virtual unsigned getLayerCount() const = 0;

    //!
    //! \brief Get the size of the hidden layers.
    //!
    //! The hidden size is the value of hiddenSize parameter passed into addRNN().
    //!
    //! \return The internal hidden layer size for the RNN.
    //! \see getDirection(), addRNN()
    //!
    virtual std::size_t getHiddenSize() const = 0;

    //!
    //! \brief Get the sequence length.
    //!
    //! The sequence length is the maximum number of time steps passed into the addRNN() function.
    //! This is also the maximum number of input tensors that the RNN can process at once.
    //!
    //! \return the maximum number of time steps that can be executed by a single call RNN layer.
    //!
    virtual int getSeqLength() const = 0;

    //!
    //! \brief Set the operation of the RNN layer.
    //!
    //! \see getOperation(), RNNOperation
    //!
    virtual void setOperation(RNNOperation op) = 0;

    //!
    //! \brief Get the operation of the RNN layer.
    //!
    //! \see setOperation(), RNNOperation
    //!
    virtual RNNOperation getOperation() const = 0;

    //!
    //! \brief Set the operation of the RNN layer.
    //!
    //! \see getInputMode(), RNNInputMode
    //!
    virtual void setInputMode(RNNInputMode op) = 0;

    //!
    //! \brief Get the operation of the RNN layer.
    //!
    //! \see setInputMode(), RNNInputMode
    //!
    virtual RNNInputMode getInputMode() const = 0;

    //!
    //! \brief Set the direction of the RNN layer.
    //!
    //! The direction determines if the RNN is run
    //! as a unidirectional(left to right) or
    //! bidirectional(left to right and right to left).
    //! In the ::kBIDIRECTION case the
    //! output is concatenated together, resulting
    //! in output size of 2x getHiddenSize().
    //! \see getDirection(), RNNDirection
    //!
    virtual void setDirection(RNNDirection op) = 0;

    //!
    //! \brief Get the direction of the RNN layer.
    //!
    //! \see setDirection(), RNNDirection
    //!
    virtual RNNDirection getDirection() const = 0;

    //!
    //! \param weights The weight structure holding the weight parameters.
    //!
    //! \brief Set the weight parameters for the RNN.
    //!
    //! The trained weights for the weight parameter matrices of the RNN.
    //! The #DataType for this structure must be ::kFLOAT or ::kHALF, and must be the same
    //! datatype as the input tensor.
    //!
    //! The layout of the weight structure depends on the #RNNOperation, #RNNInputMode, and
    //! #RNNDirection of the layer.  The array specified by `weights.values` contains a sequence of
    //! parameter matrices, where each parameter matrix is linearly appended after the previous
    //! without padding; e.g., if parameter matrix 0 and 1 have M and N elements respectively, then
    //! the layout of `weights.values` in memory looks like:
    //!
    //! ~~~
    //! index | 0 1 2 3 4 ...  M-2 M-1 | M M+1  ... M+N-2 M+N-1 | M+N M+N+1 M+N+2 ...    | ...
    //! data  |-- parameter matrix 0 --|-- parameter matrix 1 --|-- parameter matrix 2 --| ...
    //! ~~~
    //!
    //! The following sections describe \ref setRNNWeightsOrder "the order of weight matrices" and
    //! \ref setRNNWeightsLayout "the layout of elements within a weight matrix".
    //!
    //! \section setRNNWeightsOrder Order of weight matrices
    //!
    //! The parameter matrices are ordered as described below:
    //!
    //! ~~~
    //!    Let G(op, l) be defined to be a function that produces lists of parameter names, as follows:
    //!
    //!         G(::kRELU, l) := [ Wl[i], Rl[i] ]
    //!         G(::kTANH, l) := [ Wl[i], Rl[i] ]
    //!         G(::kLSTM, l) := [ Wl[f], Wl[i], Wl[c], Wl[o], Rl[f], Rl[i], Rl[c], Rl[o] ]
    //!         G(::kGRU, l)  := [ Wl[z], Wl[r], Wl[h], Rl[z], Rl[r], Rl[h] ]
    //!
    //!    where Wl[g] and Rl[g] are the names of the input and recurrent
    //!    input weight matrices for gate g, layer index l.
    //!
    //!    See RNNOperation for an overview of the naming convention used for gates.
    //!
    //!    If getDirection() == ::kUNIDIRECTION, then l identifies the stacked layer of the
    //!    RNN, with l=0 being the first recurrent layer and l=L-1 being the last recurrent layer.
    //!
    //!    If getDirection() == ::kBIDIRECTION, then (l % 2) identifies the direction of the
    //!    recurrent layer (forward if 0, or backward if 1), and (l / 2) identifies the position
    //!    of the recurrent layer within the (forward or backward) stack.
    //!
    //!    Let op := getOperation(),
    //!        L  := { ::kUNIDIRECTION => getLayerCount()
    //!              { ::kBIDIRECTION => (2 * getLayerCount())
    //!
    //!    Then the ordering of parameter matrices is the list produced by concatenating
    //!    G(op, 0), G(op, 1), G(op, 2), ..., G(op, L-1).
    //! ~~~
    //!
    //! For example:
    //!
    //!    - an RNN with `getLayerCount() == 3`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ W0[i], R0[i], W1[i], R1[i], W2[i], R2[i] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kGRU` has the following order:
    //!
    //!      `[ W0[z], W0[r], W0[h], R0[z], R0[r], R0[h], W1[z], W1[r], W1[h], R1[z], R1[r], R1[h] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kBIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ W0_fw[i], R0_fw[i], W0_bw[i], R0_bw[i], W1_fw[i], R1_fw[i], W1_bw[i], R1_bw[i] ]`
    //!
    //!      (fw = "forward", bw = "backward")
    //!
    //! \section setRNNWeightsLayout Layout of elements within a weight matrix
    //!
    //! Each parameter matrix is row-major in memory, and has the following dimensions:
    //!
    //! ~~~
    //!     Let K := { ::kUNIDIRECTION => 1
    //!              { ::kBIDIRECTION => 2
    //!         l := layer index (as described above)
    //!         H := getHiddenSize()
    //!         E := getDataLength() (the embedding length)
    //!         isW := true if the matrix is an input (W) matrix, and false if
    //!                the matrix is a recurrent input (R) matrix.
    //!
    //!    if isW:
    //!       if l < K and ::kSKIP:
    //!          (numRows, numCols) := (0, 0) # input matrix is skipped
    //!       elif l < K and ::kLINEAR:
    //!          (numRows, numCols) := (H, E) # input matrix acts on input data size E
    //!       elif l >= K:
    //!          (numRows, numCols) := (H, K * H) # input matrix acts on previous hidden state
    //!    else: # not isW
    //!       (numRows, numCols) := (H, H)
    //! ~~~
    //!
    //! In other words, the input weights of the first layer of the RNN (if
    //! not skipped) transform a `getDataLength()`-size column
    //! vector into a `getHiddenSize()`-size column vector.  The input
    //! weights of subsequent layers transform a `K*getHiddenSize()`-size
    //! column vector into a `getHiddenSize()`-size column vector.  `K=2` in
    //! the bidirectional case to account for the full hidden state being
    //! the concatenation of the forward and backward RNN hidden states.
    //!
    //! The recurrent weight matrices for all layers all have shape `(H, H)`,
    //! both in the unidirectional and bidirectional cases.  (In the
    //! bidirectional case, each recurrent weight matrix for the (forward or
    //! backward) RNN cell operates on the previous (forward or
    //! backward) RNN cell's hidden state, which is size `H`).
    //!
    //! \see getWeights(), #RNNOperation
    //!
    virtual void setWeights(Weights weights) = 0;

    //!
    //! \brief Get the W weights for the RNN.
    //!
    //! \see setWeights()
    //!
    virtual Weights getWeights() const = 0;

    //!
    //! \param bias The weight structure holding the bias parameters.
    //!
    //! \brief Set the bias parameters for the RNN.
    //!
    //! The trained weights for the bias parameter vectors of the RNN.
    //! The #DataType for this structure must be ::kFLOAT or ::kHALF, and must be the same
    //! datatype as the input tensor.
    //!
    //! The layout of the weight structure depends on the #RNNOperation, #RNNInputMode, and
    //! #RNNDirection of the layer.  The array specified by `weights.values` contains a sequence of
    //! bias vectors, where each bias vector is linearly appended after the previous
    //! without padding; e.g., if bias vector 0 and 1 have M and N elements respectively, then
    //! the layout of `weights.values` in memory looks like:
    //!
    //! ~~~
    //! index | 0 1 2 3 4 ...  M-2 M-1 | M M+1  ... M+N-2 M+N-1 | M+N M+N+1 M+N+2 ...   | ...
    //! data  |--   bias vector 0    --|--   bias vector 1    --|--   bias vector 2   --| ...
    //! ~~~
    //!
    //! The ordering of bias vectors is similar to the \ref setRNNWeightsOrder "ordering of weight matrices"
    //! as described in setWeights().  To determine the order of bias vectors for a given RNN configuration,
    //! determine the ordered list of weight matrices `[ W0, W1, ..., Wn ]`.  Then replace each weight matrix
    //! with its corresponding bias vector, i.e. apply the following transform (for layer `l`, gate `g`):
    //!
    //! - `Wl[g]` becomes `Wbl[g]`
    //! - `Rl[g]` becomes `Rbl[g]`
    //!
    //! For example:
    //!
    //!    - an RNN with `getLayerCount() == 3`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ Wb0[i], Rb0[i], Wb1[i], Rb1[i], Wb2[i], Rb2[i] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kGRU` has the following order:
    //!
    //!      `[ Wb0[z], Wb0[r], Wb0[h], Rb0[z], Rb0[r], Rb0[h], Wb1[z], Wb1[r], Wb1[h], Rb1[z], Rb1[r], Rb1[h] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kBIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ Wb0_fw[i], Rb0_fw[i], Wb0_bw[i], Rb0_bw[i], Wb1_fw[i], Rb1_fw[i], Wb1_bw[i], Rb1_bw[i] ]`
    //!
    //!      (fw = "forward", bw = "backward")
    //!
    //! Each bias vector has a fixed size, getHiddenSize().
    //!
    //! \see getBias(), #RNNOperation
    //!
    virtual void setBias(Weights bias) = 0;

    //!
    //! \brief Get the bias parameter vector for the RNN.
    //!
    //! \see setBias()
    //!
    virtual Weights getBias() const = 0;

    //!
    //! \brief Get the length of the data being processed by the RNN for use in computing
    //! other values.
    //!
    //! \see setHiddenState(), setCellState()
    //!
    virtual int getDataLength() const = 0;

    //!
    //! \param hidden The initial hidden state of the RNN.
    //!
    //! \brief Set the initial hidden state of the RNN with the provided \p hidden ITensor.
    //!
    //! The layout for \p hidden is a linear layout of a 3D matrix:
    //!  - C - The number of layers in the RNN, it must match getLayerCount().
    //!  - H - The number of mini-batches for each time sequence.
    //!  - W - The size of the per layer hidden states, it must match getHiddenSize().
    //!
    //! If getDirection() is ::kBIDIRECTION, the amount of space required is doubled and C is equal to getLayerCount() * 2.
    //!
    //! If hidden is not specified, then the initial hidden state is set to zero.
    //!
    //! \see getHiddenState()
    //!
    virtual void setHiddenState(ITensor& hidden) = 0;

    //!
    //! \brief Get the initial hidden state of the RNN.
    //!
    //! \return nullptr if no initial hidden tensor was specified, the initial hidden data otherwise.
    //!
    virtual ITensor* getHiddenState() const = 0;

    //!
    //! \param cell The initial cell state of the RNN.
    //!
    //! \brief Set the initial cell state of the RNN with the provided \p cell ITensor.
    //!
    //! The layout for \p cell is a linear layout of a 3D matrix:
    //!  - C - The number of layers in the RNN, it must match getLayerCount().
    //!  - H - The number of mini-batches for each time sequence.
    //!  - W - The size of the per layer hidden states, it must match getHiddenSize().
    //!
    //! If \p cell is not specified, then the initial cell state is set to zero.
    //!
    //! If getDirection() is ::kBIDIRECTION, the amount of space required is doubled and C is equal to getLayerCount() * 2.
    //!
    //! The cell state only affects LSTM RNN's.
    //!
    //! \see getCellState()
    //!
    virtual void setCellState(ITensor& cell) = 0;

    //!
    //! \brief Get the initial cell state of the RNN.
    //!
    //! \return nullptr if no initial cell tensor was specified, the initial cell data otherwise.
    //!
    virtual ITensor* getCellState() const = 0;

protected:
    virtual ~IRNNLayer() {}
};

//!
//! \enum RNNGateType
//!
//! \brief Identifies an individual gate within an RNN cell.
//!
//! \see RNNOperation
//!
enum class RNNGateType : int
{
    kINPUT = 0,  //!< Input gate  (i).
    kOUTPUT = 1, //!< Output gate (o).
    kFORGET = 2, //!< Forget gate (f).
    kUPDATE = 3, //!< Update gate (z).
    kRESET = 4,  //!< Reset gate  (r).
    kCELL = 5,   //!< Cell gate   (c).
    kHIDDEN = 6  //!< Hidden gate (h).
};

template <>
inline int EnumMax<RNNGateType>()
{
    return 7;
} //!< Maximum number of elements in RNNGateType enum. \see RNNGateType

//!
//! \class IRNNv2Layer
//!
//! \brief An RNN layer in a network definition, version 2.
//!
//! This layer supersedes IRNNLayer.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRNNv2Layer : public ILayer
{
public:
    virtual int32_t getLayerCount() const = 0;   //< Get the layer count of the RNN
    virtual int32_t getHiddenSize() const = 0;   //< Get the hidden size of the RNN
    virtual int32_t getMaxSeqLength() const = 0; //< Get the maximum sequence length of the RNN
    virtual int32_t getDataLength() const = 0;   //< Get the maximum data length of the RNN

    //!
    //! \brief Specify individual sequence lengths in the batch with the ITensor pointed to by
    //! \p seqLengths.
    //!
    //! The \p seqLengths ITensor should be a {N1, ..., Np} tensor, where N1..Np are the index dimensions
    //! of the input tensor to the RNN.
    //!
    //! If this is not specified, then the RNN layer assumes all sequences are size getMaxSeqLength().
    //!
    //! All sequence lengths in \p seqLengths should be in the range [1, getMaxSeqLength()].  Zero-length
    //! sequences are not supported.
    //!
    //! This tensor must be of type DataType::kINT32.
    //!
    virtual void setSequenceLengths(ITensor& seqLengths) = 0;

    //!
    //! \brief Get the sequence lengths specified for the RNN.
    //!
    //! \return nullptr if no sequence lengths were specified, the sequence length data otherwise.
    //!
    //! \see setSequenceLengths()
    //!
    virtual ITensor* getSequenceLengths() const = 0;

    //!
    //! \brief Set the operation of the RNN layer.
    //! \see getOperation(), RNNOperation
    //!
    virtual void setOperation(RNNOperation op) = 0;

    //!
    //! \brief Get the operation of the RNN layer.
    //! \see setOperation(), RNNOperation
    //!
    virtual RNNOperation getOperation() const = 0;

    //!
    //! \brief Set the input mode of the RNN layer.
    //! \see getInputMode(), RNNInputMode
    //!
    virtual void setInputMode(RNNInputMode op) = 0;

    //!
    //! \brief Get the input mode of the RNN layer.
    //! \see setInputMode(), RNNInputMode
    //!
    virtual RNNInputMode getInputMode() const = 0;

    //!
    //! \brief Set the direction of the RNN layer.
    //! \see getDirection(), RNNDirection
    //!
    virtual void setDirection(RNNDirection op) = 0;

    //!
    //! \brief Get the direction of the RNN layer.
    //! \see setDirection(), RNNDirection
    //!
    virtual RNNDirection getDirection() const = 0;

    //!
    //! \brief Set the weight parameters for an individual gate in the RNN.
    //!
    //! \param layerIndex The index of the layer that contains this gate.  See the section
    //!        \ref setRNNWeightsOrder "Order of weight matrices" in IRNNLayer::setWeights()
    //!        for a description of the layer index.
    //! \param gate The name of the gate within the RNN layer.  The gate name must correspond
    //!        to one of the gates used by this layer's #RNNOperation.
    //! \param isW True if the weight parameters are for the input matrix W[g]
    //!        and false if they are for the recurrent input matrix R[g].  See
    //!        #RNNOperation for equations showing how these matrices are used
    //!        in the RNN gate.
    //! \param weights The weight structure holding the weight parameters, which are stored
    //!        as a row-major 2D matrix.  See \ref setRNNWeightsLayout "the layout of elements within a weight matrix"
    //!        in IRNNLayer::setWeights() for documentation on the expected
    //!        dimensions of this matrix.
    //!
    virtual void setWeightsForGate(int layerIndex, RNNGateType gate, bool isW, Weights weights) = 0;

    //!
    //! \brief Get the weight parameters for an individual gate in the RNN.
    //! \see setWeightsForGate()
    //!
    virtual Weights getWeightsForGate(int layerIndex, RNNGateType gate, bool isW) const = 0;

    //!
    //! \brief Set the bias parameters for an individual gate in the RNN.
    //!
    //! \param layerIndex The index of the layer that contains this gate.  See the section
    //!        \ref setRNNWeightsOrder "Order of weight matrices" in IRNNLayer::setWeights()
    //!        for a description of the layer index.
    //! \param gate The name of the gate within the RNN layer.  The gate name must correspond
    //!        to one of the gates used by this layer's #RNNOperation.
    //! \param isW True if the bias parameters are for the input bias Wb[g]
    //!        and false if they are for the recurrent input bias Rb[g].  See
    //!        #RNNOperation for equations showing how these bias vectors are used
    //!        in the RNN gate.
    //! \param bias The weight structure holding the bias parameters, which should be an
    //!        array of size getHiddenSize().
    //!
    virtual void setBiasForGate(int layerIndex, RNNGateType gate, bool isW, Weights bias) = 0;

    //!
    //! \brief Get the bias parameters for an individual gate in the RNN.
    //! \see setBiasForGate()
    //!
    virtual Weights getBiasForGate(int layerIndex, RNNGateType gate, bool isW) const = 0;

    //!
    //! \brief Set the initial hidden state of the RNN with the provided \p hidden ITensor.
    //!
    //! The \p hidden ITensor should have the dimensions `{N1, ..., Np, L, H}`, where:
    //!
    //!  - `N1..Np` are the index dimensions specified by the input tensor
    //!  - `L` is the number of layers in the RNN, equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2x getLayerCount() if getDirection is ::kBIDIRECTION. In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L= 2*l + 1`.
    //!  - `H` is the hidden state for each layer, equal to getHiddenSize().
    //!
    virtual void setHiddenState(ITensor& hidden) = 0;

    //!
    //! \brief Get the initial hidden state of the RNN.
    //! \see setHiddenState()
    //!
    virtual ITensor* getHiddenState() const = 0;

    //!
    //! \brief Set the initial cell state of the LSTM with the provided \p cell ITensor.
    //!
    //! The \p cell ITensor should have the dimensions `{N1, ..., Np, L, H}`, where:
    //!
    //!  - `N1..Np` are the index dimensions specified by the input tensor
    //!  - `L` is the number of layers in the RNN, equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2x getLayerCount() if getDirection is ::kBIDIRECTION. In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L= 2*l + 1`.
    //!  - `H` is the hidden state for each layer, equal to getHiddenSize().
    //!
    //! It is an error to call setCellState() on an RNN layer that is not configured with RNNOperation::kLSTM.
    //!
    virtual void setCellState(ITensor& cell) = 0;

    //!
    //! \brief Get the initial cell state of the RNN.
    //! \see setCellState()
    //!
    virtual ITensor* getCellState() const = 0;

protected:
    virtual ~IRNNv2Layer() {}
};

//!
//! \class IOutputDimensionsFormula
//!
//! \brief Application-implemented interface to compute layer output sizes.
//!
class IOutputDimensionsFormula
{
public:
    //!
    //! \brief Application-implemented interface to compute the HW output dimensions of a layer from the layer input and parameters.
    //!
    //! \param inputDims The input dimensions of the layer.
    //! \param kernelSize The kernel size (or window size, for a pooling layer) parameter of the layer operation.
    //! \param stride The stride parameter for the layer.
    //! \param padding The padding parameter of the layer.
    //! \param dilation The dilation parameter of the layer (only applicable to convolutions).
    //! \param layerName The name of the layer.
    //!
    //! \return The output size of the layer
    //!
    //! Note that for dilated convolutions, the dilation is applied to the kernel size before this routine is called.
    //!
    virtual DimsHW compute(DimsHW inputDims, DimsHW kernelSize, DimsHW stride, DimsHW padding, DimsHW dilation, const char* layerName) const = 0;

    virtual ~IOutputDimensionsFormula() {}
};

//!
//! \enum PluginFormatType
//!
//! \brief Format of the input/output tensors.
//!
//! \see IPluginExt::getPluginFormats()
//!
//! For more information about data formats, see the topic "Data Format Description" located in the
//! TensorRT Developer Guide (https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html).
//!
enum class PluginFormat : uint8_t
{
    //! NCHW.
    kNCHW = 0,

    //! NCHW with 2-element packed channels.  For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions [N][(C+1)/2][H][W][2],
    //! with the tensor coordinates (n,c,h,w) mapping to array subscript [n][c/2][h][w][c%2].
    kNC2HW2 = 1,

    //! NHWC where C must be a multiple of 8.
    kNHWC8 = 2
};

template <>
inline int EnumMax<PluginFormat>()
{
    return 3;
} //!< Maximum number of elements in PluginFormat enum. \see PluginFormat

//! \class IPlugin
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. Each plugin is owned by the application, and its lifetime
//! must span any use of it by TensorRT
//!
class IPlugin
{
public:
    //!
    //! \brief Get the number of outputs from the layer.
    //!
    //! \return The number of outputs.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
    //!
    virtual int getNbOutputs() const = 0;

    //!
    //! \brief Get the dimension of an output tensor.
    //!
    //! \param index The index of the output tensor.
    //! \param inputs The input tensors.
    //! \param nbInputDims The number of input tensors.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
    //!
    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) = 0;

    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make algorithm choices on the basis
    //! of its weights, dimensions, and maximum batch size. The type is assumed to be FP32 and format NCHW.
    //!
    //! \param inputDims The input tensor dimensions.
    //! \param nbInputs The number of inputs.
    //! \param outputDims The output tensor dimensions.
    //! \param nbOutputs The number of outputs.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be 3-dimensional CHW dimensions).
    //!
    //! This method is not called for PluginExt classes; configureWithFormat is called instead.
    //!
    virtual void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) = 0;

    //!
    //! \brief Initialize the layer for execution. This is called when the engine is created.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    virtual int initialize() = 0;

    //!
    //! \brief Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
    //! \see initialize()
    //!
    virtual void terminate() = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called during engine startup, after initialize(). The workspace size returned should be sufficient for any
    //! batch size up to the maximum.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(int maxBatchSize) const = 0;

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
    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) = 0;

    //!
    //! \brief Find the size of the serialization buffer required.
    //!
    //! \return The size of the serialization buffer.
    //!
    virtual size_t getSerializationSize() = 0;

    //!
    //! \brief Serialize the layer.
    //!
    //! \param buffer A pointer to a buffer of size at least that returned by getSerializationSize().
    //!
    //! \see getSerializationSize()
    //!
    virtual void serialize(void* buffer) = 0;

    virtual ~IPlugin() {}
};

//!
//! \class IPluginExt
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. Each plugin is owned by the application, and its lifetime
//! must span any use of it by TensorRT.
//!
class IPluginExt : public IPlugin
{
public:
    //!
    //! \brief Return the API version with which this plugin was built.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with plugins.
    //!
    virtual int getTensorRTVersion() const
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief Check format support.
    //!
    //! \param type DataType requested.
    //! \param format PluginFormat requested.
    //! \return true if the plugin supports the type-format combination.
    //!
    //! This function is called by the implementations of INetworkDefinition, IBuilder, and ICudaEngine.
    //! In particular, it is called when creating an engine and when deserializing an engine.
    //!
    virtual bool supportsFormat(DataType type, PluginFormat format) const = 0;

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
    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) = 0;

    virtual ~IPluginExt() {}

protected:
    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    void configure(const Dims* /*inputDims*/, int /*nbInputs*/, const Dims* /*outputDims*/, int /*nbOutputs*/, int /*maxBatchSize*/) _TENSORRT_FINAL {}
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
    virtual int getTensorRTVersion() const
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief Return the plugin type. Should match the plugin name returned by the corresponding plugin creator
    // \see IPluginCreator::getPluginName()
    //!
    virtual const char* getPluginType() const = 0;

    //!
    //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator
    // \see IPluginCreator::getPluginVersion()
    //!
    virtual const char* getPluginVersion() const = 0;

    //!
    //! \brief Get the number of outputs from the layer.
    //!
    //! \return The number of outputs.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
    //!
    virtual int getNbOutputs() const = 0;

    //!
    //! \brief Get the dimension of an output tensor.
    //!
    //! \param index The index of the output tensor.
    //! \param inputs The input tensors.
    //! \param nbInputDims The number of input tensors.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
    //!
    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) = 0;

    //!
    //! \brief Check format support.
    //!
    //! \param type DataType requested.
    //! \param format PluginFormat requested.
    //! \return true if the plugin supports the type-format combination.
    //!
    //! This function is called by the implementations of INetworkDefinition, IBuilder, and ICudaEngine.
    //! In particular, it is called when creating an engine and when deserializing an engine.
    //!
    virtual bool supportsFormat(DataType type, PluginFormat format) const = 0;

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
    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) = 0;

    //!
    //! \brief Initialize the layer for execution. This is called when the engine is created.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    virtual int initialize() = 0;

    //!
    //! \brief Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
    //! \see initialize()
    //!
    virtual void terminate() = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called during engine startup, after initialize(). The workspace size returned should be sufficient for any
    //! batch size up to the maximum.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(int maxBatchSize) const = 0;

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
    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) = 0;

    //!
    //! \brief Find the size of the serialization buffer required.
    //!
    //! \return The size of the serialization buffer.
    //!
    virtual size_t getSerializationSize() const = 0;

    //!
    //! \brief Serialize the layer.
    //!
    //! \param buffer A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by getSerializationSize.
    //!
    //! \see getSerializationSize()
    //!
    virtual void serialize(void* buffer) const = 0;

    //!
    //! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
    //!
    virtual void destroy() = 0;

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with these parameters.
    //!
    virtual IPluginV2* clone() const = 0;

    //!
    //! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
    //! objects from the same plugin library should have the same namespace.
    //!
    virtual void setPluginNamespace(const char* pluginNamespace) = 0;

    //!
    //! \brief Return the namespace of the plugin object.
    //!
    virtual const char* getPluginNamespace() const = 0;

protected:
    virtual ~IPluginV2() {}
};

class IGpuAllocator;

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
    virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const = 0;

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
    virtual bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const = 0;

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
    virtual bool canBroadcastInputAcrossBatch(int inputIndex) const = 0;

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
    //! \param floatFormat The format selected for the engine for the floating point
    //!  inputs/outputs.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be 3-dimensional CHW dimensions).
    //! When inputIsBroadcast or outputIsBroadcast is true, the outermost batch size for that input or output should be treated as if it is one.
    //! \ref inputIsBroadcast[i] is true only if the input is semantically broadcast across the batch and \ref canBroadcastInputAcrossBatch(i) returned true.
    //! \ref outputIsBroadcast[i] is true only if \ref isOutputBroadcastAcrossBatch(i) returned true.

    virtual void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                                 int nbOutputs, const DataType* inputTypes, const DataType* outputTypes,
                                 const bool* inputIsBroadcast, const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
        = 0;

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
    virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) {}

    //!
    //! \brief Detach the plugin object from its execution context.
    //!
    //! This function is called automatically for each plugin when a execution context is destroyed.
    //! If the plugin owns per-context resource, it can be released here.
    //!
    virtual void detachFromContext() {}

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.
    //! If the source plugin is pre-configured with configurePlugin(), the returned object should also be pre-configured. The returned object should allow attachToContext() with a new execution context.
    //! Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object (e.g. via ref-counting) to avoid duplication.
    //!
    virtual IPluginV2Ext* clone() const _TENSORRT_OVERRIDE = 0;

protected:
    //!
    //! \brief Return the API version with which this plugin was built. The
    //!  upper byte reserved by TensorRT and is used to differentiate this from IPlguinV2.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with plugins.
    //!
    int getTensorRTVersion() const _TENSORRT_OVERRIDE
    {
        return (0x01000000 | (NV_TENSORRT_VERSION & 0xFFFFFF));
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    void configureWithFormat(const Dims* /*inputDims*/, int /*nbInputs*/, const Dims* /*outputDims*/,
                             int /*nbOutputs*/, DataType /*type*/, PluginFormat /*format*/, int /*maxBatchSize*/) _TENSORRT_OVERRIDE _TENSORRT_FINAL {}
};

//!
//! \class IPluginLayer
//!
//! \brief Layer type for plugins.
//!
//! \see IPluginExt
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPluginLayer : public ILayer
{
public:
    //!
    //! \brief Get the plugin for the layer.
    //!
    //! \see IPluginExt
    //!
    virtual IPlugin& getPlugin() = 0;

protected:
    virtual ~IPluginLayer() {}
};

//!
//! \class IPluginV2Layer
//!
//! \brief Layer type for pluginV2
//!
//! \see IPluginV2
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPluginV2Layer : public ILayer
{
public:
    //!
    //! \brief Get the plugin for the layer.
    //!
    //! \see IPluginV2
    //!
    virtual IPluginV2& getPlugin() = 0;

protected:
    virtual ~IPluginV2Layer() {}
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
    const char* name;
    //!
    //! \brief Plugin field attribute data
    //!
    const void* data;
    //!
    //! \brief Plugin field attribute type
    //! \see PluginFieldType
    //!
    PluginFieldType type;
    //!
    //! \brief Number of data entries in the Plugin attribute
    //!
    int length;

    PluginField(const char* name_ = nullptr, const void* data_ = nullptr, const PluginFieldType type_ = PluginFieldType::kUNKNOWN, int length_ = 0)
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
    virtual int getTensorRTVersion() const { return NV_TENSORRT_VERSION; }

    //!
    //! \brief Return the plugin name.
    //!
    virtual const char* getPluginName() const = 0;

    //!
    //! \brief Return the plugin version.
    //!
    virtual const char* getPluginVersion() const = 0;

    //!
    //! \brief Return a list of fields that needs to be passed to createPlugin.
    //! \see PluginFieldCollection
    //!
    virtual const PluginFieldCollection* getFieldNames() = 0;

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
    virtual IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) = 0;

    //!
    //! \brief Called during deserialization of plugin layer. Return a plugin object.
    //!
    virtual IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) = 0;

    //!
    //! \brief Set the namespace of the plugin creator based on the plugin
    //! library it belongs to. This can be set while registering the plugin creator.
    //!
    //! \see IPluginRegistry::registerCreator()
    //!
    virtual void setPluginNamespace(const char* pluginNamespace) = 0;

    //!
    //! \brief Return the namespace of the plugin creator object.
    //!
    virtual const char* getPluginNamespace() const = 0;

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
    virtual bool registerCreator(IPluginCreator& creator, const char* pluginNamespace) = 0;

    //!
    //! \brief Return all the registered plugin creators and the number of
    //! registered plugin creators. Returns nullptr if none found.
    //!
    virtual IPluginCreator* const* getPluginCreatorList(int* numCreators) const = 0;

    //!
    //! \brief Return plugin creator based on plugin type, version and
    //! namespace associated with plugin during network creation.
    //!
    virtual IPluginCreator* getPluginCreator(const char* pluginType, const char* pluginVersion, const char* pluginNamespace = "") = 0;

protected:
    virtual ~IPluginRegistry() {}
};

//!
//! \enum UnaryOperation
//!
//! \brief Enumerates the unary operations that may be performed by a Unary layer.
//!
//! \see IUnaryLayer
//!
enum class UnaryOperation : int
{
    kEXP = 0,    //!< Exponentiation.
    kLOG = 1,    //!< Log (base e).
    kSQRT = 2,   //!< Square root.
    kRECIP = 3,  //!< Reciprocal.
    kABS = 4,    //!< Absolute value.
    kNEG = 5,    //!< Negation.
    kSIN = 6,    //!< Sine.
    kCOS = 7,    //!< Cosine.
    kTAN = 8,    //!< Tangent.
    kSINH = 9,   //!< Hyperbolic sine.
    kCOSH = 10,  //!< Hyperbolic cosine.
    kASIN = 11,  //!< Inverse sine.
    kACOS = 12,  //!< Inverse cosine.
    kATAN = 13,  //!< Inverse tangent.
    kASINH = 14, //!< Inverse hyperbolic sine.
    kACOSH = 15, //!< Inverse hyperbolic cosine.
    kATANH = 16, //!< Inverse hyperbolic tangent.
    kCEIL = 17,  //!< Ceiling.
    kFLOOR = 18  //!< Floor.
};

template <>
inline int EnumMax<UnaryOperation>()
{
    return 19;
} //!< Maximum number of elements in UnaryOperation enum. \see UnaryOperation

//!
//! \class IUnaryLayer
//!
//! \brief Layer that represents an unary operation.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IUnaryLayer : public ILayer
{
public:
    //!
    //! \brief Set the unary operation for the layer.
    //!
    //! \see getOperation(), UnaryOperation
    //!
    virtual void setOperation(UnaryOperation op) = 0;

    //!
    //! \brief Get the unary operation for the layer.
    //!
    //! \see setOperation(), UnaryOperation
    //!
    virtual UnaryOperation getOperation() const = 0;

protected:
    virtual ~IUnaryLayer() {}
};

//!
//! \enum ReduceOperation
//!
//! \brief Enumerates the reduce operations that may be performed by a Reduce layer.
//!
enum class ReduceOperation : int
{
    kSUM = 0,
    kPROD = 1,
    kMAX = 2,
    kMIN = 3,
    kAVG = 4
};

template <>
inline int EnumMax<ReduceOperation>()
{
    return 5;
} //!< Maximum number of elements in ReduceOperation enum. \see ReduceOperation

//!
//! \class IReduceLayer
//!
//! \brief Layer that represents a reduction operator.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IReduceLayer : public ILayer
{
public:
    //!
    //! \brief Set the reduce operation for the layer.
    //!
    //! \see getOperation(), ReduceOperation
    //!
    virtual void setOperation(ReduceOperation op) = 0;

    //!
    //! \brief Get the reduce operation for the layer.
    //!
    //! \see setOperation(), ReduceOperation
    //!
    virtual ReduceOperation getOperation() const = 0;

    //!
    //! \brief Set the axes over which to reduce.
    //!
    //! \see getReduceAxes
    //!
    virtual void setReduceAxes(uint32_t reduceAxes) = 0;

    //!
    //! \brief Get the axes over which to reduce for the layer.
    //!
    //! \see setReduceAxes
    //!
    virtual uint32_t getReduceAxes() const = 0;

    //!
    //! \brief Set the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    //! \see getKeepDimensions
    //!
    virtual void setKeepDimensions(bool keepDimensions) = 0;

    //!
    //! \brief Get the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    //! \see setKeepDimensions
    //!
    virtual bool getKeepDimensions() const = 0;

protected:
    virtual ~IReduceLayer() {}
};

//!
//! \class IPaddingLayer
//!
//! \brief Layer that represents a padding operation.
//!
//! The padding layer adds zero-padding at the start and end of the input tensor. It only supports padding along the two innermost dimensions.
//! Applying negative padding results in cropping of the input.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPaddingLayer : public ILayer
{
public:
    //!
    //! \brief Set the padding that is applied at the start of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount
    //!
    //! \see getPrePadding
    //!
    virtual void setPrePadding(DimsHW padding) = 0;

    //!
    //! \brief Set the padding that is applied at the start of the tensor.
    //!
    //! \see setPrePadding
    //!
    virtual DimsHW getPrePadding() const = 0;

    //!
    //! \brief Set the padding that is applied at the end of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount
    //!
    //! \see getPostPadding
    //!
    virtual void setPostPadding(DimsHW padding) = 0;

    //!
    //! \brief Set the padding that is applied at the end of the tensor.
    //!
    //! \see setPostPadding
    //!
    virtual DimsHW getPostPadding() const = 0;

protected:
    virtual ~IPaddingLayer() {}
};

struct Permutation
{
    //!
    //! The elements of the permutation.
    //! The permutation is applied as outputDimensionIndex = permutation.order[inputDimensionIndex], so to
    //! permute from CHW order to HWC order, the required permutation is [1, 2, 0], and to permute
    //! from HWC to CHW, the required permutation is [2, 0, 1].
    //!
    int order[Dims::MAX_DIMS];
};

//! \class IShuffleLayer
//!
//! \brief Layer type for shuffling data.
//!
//! This class shuffles data by applying in sequence: a transpose operation, a reshape operation
//! and a second transpose operation. The dimension types of the output are those of the reshape dimension.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IShuffleLayer : public ILayer
{
public:
    //!
    //! \brief Set the permutation applied by the first transpose operation.
    //!
    //! \param permutation The dimension permutation applied before the reshape.
    //!
    //! The default is the identity permutation.
    //!
    //! \see getFirstTranspose
    //!
    virtual void setFirstTranspose(Permutation permutation) = 0;

    //!
    //! \brief Get the permutation applied by the first transpose operation.
    //!
    //! \return The dimension permutation applied before the reshape.
    //!
    //! \see setFirstTranspose
    //!
    virtual Permutation getFirstTranspose() const = 0;

    //!
    //! \brief Set the reshaped dimensions.
    //!
    //! \param dimensions The reshaped dimensions.
    //!
    //! Two special values can be used as dimensions.
    //!
    //! Value 0 copies the corresponding dimension from input. This special value
    //! can be used more than once in the dimensions. If number of reshape
    //! dimensions is less than input, 0s are resolved by aligning the most
    //! significant dimensions of input.
    //!
    //! Value -1 infers that particular dimension by looking at input and rest
    //! of the reshape dimensions. Note that only a maximum of one dimension is
    //! permitted to be specified as -1.
    //!
    //! The product of the new dimensions must be equal to the product of the old.
    //!
    virtual void setReshapeDimensions(Dims dimensions) = 0;

    //!
    //! \brief Get the reshaped dimensions.
    //!
    //! \return The reshaped dimensions.
    //!
    virtual Dims getReshapeDimensions() const = 0;

    //!
    //! \brief Set the permutation applied by the second transpose operation.
    //!
    //! \param permutation The dimension permutation applied after the reshape.
    //!
    //! The default is the identity permutation.
    //!
    //! The permutation is applied as outputDimensionIndex = permutation.order[inputDimensionIndex], so to
    //! permute from CHW order to HWC order, the required permutation is [1, 2, 0].
    //!
    //! \see getSecondTranspose
    //!
    virtual void setSecondTranspose(Permutation permutation) = 0;

    //!
    //! \brief Get the permutation applied by the second transpose operation.
    //!
    //! \return The dimension permutation applied after the reshape.
    //!
    //! \see setSecondTranspose
    //!
    virtual Permutation getSecondTranspose() const = 0;

protected:
    virtual ~IShuffleLayer() {}
};

//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISliceLayer : public ILayer
{
public:
    //!
    //! \brief Set the start offset
    //!
    //! \param start The start offset
    //!
    //! \see getStart
    //!
    virtual void setStart(Dims start) = 0;

    //!
    //! \brief Get the start offset
    //!
    //! \return The start Offset
    //!
    //! \see setStart
    virtual Dims getStart() const = 0;

    //!
    //! \brief Set the output dimension
    //!
    //! \param size The output dimension
    //!
    //! \see getSize
    virtual void setSize(Dims size) = 0;

    //!
    //! \brief Get the output dimension
    //!
    //! \return The output dimesion
    //!
    //! \see setSize
    virtual Dims getSize() const = 0;

    //!
    //! \brief Set the slicing stride
    //!
    //! \param stride The slicing stride
    //!
    //! \see getStride
    virtual void setStride(Dims stride) = 0;

    //!
    //! \brief Get the slicing stride
    //!
    //! \return The slicing stride
    //!
    //! \see setStride
    virtual Dims getStride() const = 0;

protected:
    virtual ~ISliceLayer() {}
};

//!
//! \enum TopKOperation
//!
//! \brief Enumerates the operations that may be performed by a TopK layer.
//!
enum class TopKOperation : int
{
    kMAX = 0, //!< Maximum of the elements.
    kMIN = 1, //!< Minimum of the elements.
};

template <>
inline int EnumMax<TopKOperation>()
{
    return 2;
} //!< Maximum number of elements in TopKOperation enum. \see TopKOperation

//!
//! \class ITopKLayer
//!
//! \brief Layer that represents a TopK reduction.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ITopKLayer : public ILayer
{
public:
    //!
    //! \brief Set the operation for the layer.
    //!
    //! \see getOperation(), TopKOperation
    //!
    virtual void setOperation(TopKOperation op) = 0;

    //!
    //! \brief Get the operation for the layer.
    //!
    //! \see setOperation(), TopKOperation
    //!
    virtual TopKOperation getOperation() const = 0;

    //!
    //! \brief Set the k value for the layer.
    //!
    //! Currently only values up to 25 are supported.
    //!
    //! \see getK()
    //!
    virtual void setK(int k) = 0;

    //!
    //! \brief Get the k value for the layer.
    //!
    //! \see setK()
    //!
    virtual int getK() const = 0;

    //!
    //! \brief Set which axes to reduce for the layer.
    //!
    //! \see getReduceAxes()
    //!
    virtual void setReduceAxes(uint32_t reduceAxes) = 0;

    //!
    //! \brief Get the axes to reduce for the layer.
    //!
    //! \see setReduceAxes()
    //!
    virtual uint32_t getReduceAxes() const = 0;

protected:
    virtual ~ITopKLayer() {}
};

//!
//! \enum MatrixOperation
//!
//! \brief Enumerates the operations that may be performed on a tensor
//!        by IMatrixMultiplyLayer before multiplication.
//!
enum class MatrixOperation : int
{
    //! Treat x as a matrix if it has two dimensions, or as a collection of
    //! matrices if x has more than two dimensions, where the last two dimensions
    //! are the matrix dimensions.  x must have at least two dimensions.
    kNONE,

    //! Like kNONE, but transpose the matrix dimensions.
    kTRANSPOSE,

    //! Treat x as a vector if it has one dimension, or as a collection of
    //! vectors if x has more than one dimension.  x must have at least one dimension.
    kVECTOR
};

template <>
inline int EnumMax<MatrixOperation>()
{
    return 3;
} //!< Maximum number of elements in MatrixOperation enum. \see DataType

//!
//! \class IMatrixMultiplyLayer
//!
//! \brief Layer that represents a Matrix Multiplication.
//!
//! Let A be op(getInput(0)) and B be op(getInput(1)) where
//! op(x) denotes the corresponding MatrixOperation.
//!
//! When A and B are matrices or vectors, computes the inner product A * B:
//!
//!     matrix * matrix -> matrix
//!     matrix * vector -> vector
//!     vector * matrix -> vector
//!     vector * vector -> scalar
//!
//! Inputs of higher rank are treated as collections of matrices or vectors.
//! The output will be a corresponding collection of matrices, vectors, or scalars.
//!
//! For a dimension that is not one of the matrix or vector dimensions:
//! If the dimension is 1 for one of the tensors but not the other tensor,
//! the former tensor is broadcast along that dimension to match the dimension of the latter tensor.
//! The number of these extra dimensions for A and B must match.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IMatrixMultiplyLayer : public ILayer
{
public:
    //!
    //! \brief Set the operation for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \param op New operation.
    //! \see getTranspose()
    //!
    virtual void setOperation(int index, MatrixOperation op) = 0;

    //!
    //! \brief Get the operation for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \see setTranspose()
    //!
    virtual MatrixOperation getOperation(int index) const = 0;

    //!
    //! \brief Set the transpose flag for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \param val New transpose flag.
    //! \see getTranspose()
    //!
    //! \deprecated setTranspose is superseded by setOperation.
    //!
    virtual void setTranspose(int index, bool val) = 0;

    //!
    //! \brief Get the transpose flag for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \see setTranspose()
    //!
    //! \deprecated getTranspose is superseded by getOperation.
    //!
    virtual bool getTranspose(int index) const = 0;

protected:
    virtual ~IMatrixMultiplyLayer() {}
};

//!
//! \class IRaggedSoftMaxLayer
//!
//! \brief A RaggedSoftmax layer in a network definition.
//!
//! This layer takes a ZxS input tensor and an additional Zx1 bounds tensor
//! holding the lengths of the Z sequences.
//!
//! This layer computes a softmax across each of the Z sequences.
//!
//! The output tensor is of the same size as the input tensor.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRaggedSoftMaxLayer : public ILayer
{
protected:
    virtual ~IRaggedSoftMaxLayer() {}
};

//! \class IIdentityLayer
//!
//! \brief A layer that represents the identity function.
//!
//! If tensor precision is being explicitly specified, it can be used to transform from one precision to another.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IIdentityLayer : public ILayer
{
protected:
    virtual ~IIdentityLayer() {}
};

//! \class IConstantLayer
//!
//! \brief Layer that represents a constant value.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConstantLayer : public ILayer
{
public:
    //!
    //! \brief Set the weights for the layer.
    //!
    //! If weights.type is DataType::kINT32, the output is a tensor of 32-bit indices.
    //! Otherwise the output is a tensor of real values, and the output type will be
    //! FP32, FP16, or quantized INT8 following TensorRT's normal precision rules.
    //!
    //! \see getWeights()
    //!
    virtual void setWeights(Weights weights) = 0;

    //!
    //! \brief Get the weights for the layer.
    //!
    //! \see setWeights
    //!
    virtual Weights getWeights() const = 0;

    //!
    //! \brief Set the dimensions for the layer.
    //!
    //! \param dimensions The dimensions of the layer
    //!
    //! @see setDimensions
    //!
    virtual void setDimensions(Dims dimensions) = 0;

    //!
    //! \brief Get the dimensions for the layer.
    //!
    //! \return the dimensions for the layer
    //!
    //! @see getDimensions
    //!
    virtual Dims getDimensions() const = 0;

protected:
    virtual ~IConstantLayer() {}
};

//!
//! \class INetworkDefinition
//!
//! \brief A network definition for input to the builder.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class INetworkDefinition
{
public:
    //!
    //! \brief Add an input tensor to the network.
    //!
    //! The name of the input tensor is used to find the index into the buffer array for an engine built from the network.
    //!
    //! \param name The name of the tensor.
    //! \param type The type of the data held in the tensor.
    //! \param dimensions The dimensions of the tensor.
    //!
    //! Only DataType::kFLOAT, DataType::kHALF and DataType::kINT32 are valid input tensor types.
    //! The volume of the dimensions, including the maximum batch size, must be less than 2^30 elements.
    //!
    //! \see ITensor
    //!
    //! \return The new tensor or nullptr if there is an error.
    //!
    virtual ITensor* addInput(const char* name, DataType type, Dims dimensions) = 0;

    //!
    //! \brief Mark a tensor as a network output.
    //!
    //! \param tensor The tensor to mark as an output tensor.
    //!
    virtual void markOutput(ITensor& tensor) = 0;

    //!
    //! \brief Add a convolution layer to the network.
    //!
    //! \param input The input tensor to the convolution.
    //! \param nbOutputMaps The number of output feature maps for the convolution.
    //! \param kernelSize The HW-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The optional bias weights for the convolution.
    //!
    //! \see IConvolutionLayer
    //!
    //! \return The new convolution layer, or nullptr if it could not be created.
    //!
    virtual IConvolutionLayer* addConvolution(ITensor& input, int nbOutputMaps, DimsHW kernelSize, Weights kernelWeights, Weights biasWeights) = 0;

    //!
    //! \brief Add a fully connected layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputs The number of outputs of the layer.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The optional bias weights for the convolution.
    //!
    //! \see IFullyConnectedLayer
    //!
    //! \return The new fully connected layer, or nullptr if it could not be created.
    //!
    virtual IFullyConnectedLayer* addFullyConnected(ITensor& input, int nbOutputs, Weights kernelWeights, Weights biasWeights) = 0;

    //!
    //! \brief Add an activation layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of activation function to apply.
    //!
    //! Note that the setAlpha() and setBeta() methods must be used on the
    //! output for activations that require these parameters.
    //!
    //! \see IActivationLayer ActivationType
    //!
    //! \return The new activation layer, or nullptr if it could not be created.
    //!
    virtual IActivationLayer* addActivation(ITensor& input, ActivationType type) = 0;

    //!
    //! \brief Add a pooling layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of pooling to apply.
    //! \param windowSize The size of the pooling window.
    //!
    //! \see IPoolingLayer PoolingType
    //!
    //! \return The new pooling layer, or nullptr if it could not be created.
    //!
    virtual IPoolingLayer* addPooling(ITensor& input, PoolingType type, DimsHW windowSize) = 0;

    //!
    //! \brief Add a LRN layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param window The size of the window.
    //! \param alpha The alpha value for the LRN computation.
    //! \param beta The beta value for the LRN computation.
    //! \param k The k value for the LRN computation.
    //!
    //! \see ILRNLayer
    //!
    //! \return The new LRN layer, or nullptr if it could not be created.
    //!
    virtual ILRNLayer* addLRN(ITensor& input, int window, float alpha, float beta, float k) = 0;

    //!
    //! \brief Add a Scale layer to the network.
    //!
    //! \param input The input tensor to The layer. This tensor is required to have a minimum of 3 dimensions.
    //! \param mode The scaling mode.
    //! \param shift The shift value.
    //! \param scale The scale value.
    //! \param power The power value.
    //!
    //! If the weights are available, then the size of weights are dependent on the on the ScaleMode.
    //! For ::kUNIFORM, the number of weights is equal to 1.
    //! For ::kCHANNEL, the number of weights is equal to the channel dimension.
    //! For ::kELEMENTWISE, the number of weights is equal to the volume of the input.
    //!
    //! \see IScaleLayer
    //!
    //! \return The new Scale layer, or nullptr if it could not be created.
    //!
    virtual IScaleLayer* addScale(ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) = 0;

    //!
    //! \brief Add a SoftMax layer to the network.
    //!
    //! \see ISoftMaxLayer
    //!
    //! \return The new SoftMax layer, or nullptr if it could not be created.
    //!
    virtual ISoftMaxLayer* addSoftMax(ITensor& input) = 0;

    //!
    //! \brief Add a concatenation layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //!
    //! \see IConcatenationLayer
    //!
    //! \return The new concatenation layer, or nullptr if it could not be created.
    //!
    //! \warning All tensors must have the same dimensions for all dimensions except for channel.
    //!
    virtual IConcatenationLayer* addConcatenation(ITensor* const* inputs, int nbInputs) = 0;

    //!
    //! \brief Add a deconvolution layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputMaps The number of output feature maps.
    //! \param kernelSize The HW-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The optional bias weights for the convolution.
    //!
    //! \see IDeconvolutionLayer
    //!
    //! \return The new deconvolution layer, or nullptr if it could not be created.
    //!
    virtual IDeconvolutionLayer* addDeconvolution(ITensor& input, int nbOutputMaps, DimsHW kernelSize, Weights kernelWeights, Weights biasWeights) = 0;

    //!
    //! \brief Add an elementwise layer to the network.
    //!
    //! \param input1 The first input tensor to the layer.
    //! \param input2 The second input tensor to the layer.
    //! \param op The binary operation that the layer applies.
    //!
    //! The input tensors must have the same number of dimensions.
    //! For each dimension, their lengths must match, or one of them must be one.
    //! In the latter case, the tensor is broadcast along that axis.
    //!
    //! The output tensor has the same number of dimensions as the inputs.
    //! For each dimension, its length is the maximum of the lengths of the
    //! corresponding input dimension.
    //!
    //! \see IElementWiseLayer
    //!
    //! \return The new elementwise layer, or nullptr if it could not be created.
    //!
    virtual IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) = 0;

    //!
    //! \brief Add an \p layerCount deep RNN layer to the network with a
    //! sequence length of \p maxSeqLen and \p hiddenSize internal state per
    //! layer.
    //!
    //! \param inputs The input tensor to the layer.
    //! \param layerCount The number of layers in the RNN.
    //! \param hiddenSize The size of the internal hidden state for each layer.
    //! \param maxSeqLen The maximum length of the time sequence.
    //! \param op The type of RNN to execute.
    //! \param mode The input mode for the RNN.
    //! \param dir The direction to run the RNN.
    //! \param weights The weights for the weight matrix parameters of the RNN.
    //! \param bias The weights for the bias vectors parameters of the RNN.
    //!
    //! The input tensors must be of the type DataType::kFLOAT or DataType::kHALF.
    //!
    //! See IRNNLayer::setWeights() and IRNNLayer::setBias() for details on the required input
    //! format for \p weights and \p bias.
    //!
    //! The layout for the \p input tensor should be `{1, S_max, N, E}`, where:
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `N` is the batch size
    //!   - `E` specifies the embedding length (unless ::kSKIP is set, in which case it should match
    //!     getHiddenSize()).
    //!
    //! The first output tensor is the output of the final RNN layer across all timesteps, with dimensions
    //! `{S_max, N, H}`:
    //!
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `N` is the batch size
    //!   - `H` is an output hidden state (equal to getHiddenSize() or 2x getHiddenSize())
    //!
    //! The second tensor is the final hidden state of the RNN across all layers, and if the RNN
    //! is an LSTM (i.e. getOperation() is ::kLSTM), then the third tensor is the final cell
    //! state of the RNN across all layers.  Both the second and third output tensors have dimensions
    //! `{L, N, H}`:
    //!
    //!  - `L` is equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2*getLayerCount() if getDirection is ::kBIDIRECTION.  In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L = 2*l + 1`.
    //!  - `N` is the batch size
    //!  - `H` is getHiddenSize().
    //!
    //! Note that in bidirectional RNNs, the full "hidden state" for a layer `l`
    //! is the concatenation of its forward hidden state and its backward hidden
    //! state, and its size is 2*H.
    //!
    //! \deprecated IRNNLayer is superseded by IRNNv2Layer. Use addRNNv2() instead.
    //!
    //! \see IRNNLayer
    //!
    //! \return The new RNN layer, or nullptr if it could not be created.
    //!
    virtual IRNNLayer* addRNN(ITensor& inputs, int layerCount, std::size_t hiddenSize, int maxSeqLen, RNNOperation op, RNNInputMode mode, RNNDirection dir, Weights weights, Weights bias) = 0;

    //!
    //! \brief Add a plugin layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginLayer
    //!
    //! \return the new plugin layer, or nullptr if it could not be created.
    //!
    virtual IPluginLayer* addPlugin(ITensor* const* inputs, int nbInputs, IPlugin& plugin) = 0;

    //!
    //! \brief Add a unary layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The operation to apply.
    //!
    //! \see IUnaryLayer
    //!
    //! \return The new unary layer, or nullptr if it could not be created
    //!
    virtual IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) = 0;

    //! \brief Add a padding layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param prePadding The padding to apply to the start of the tensor.
    //! \param postPadding The padding to apply to the end of the tensor.
    //!
    //! \see IPaddingLayer
    //!
    //! \return The new padding layer, or nullptr if it could not be created.
    //!
    virtual IPaddingLayer* addPadding(ITensor& input, DimsHW prePadding, DimsHW postPadding) = 0;

    //!
    //! \brief Add a shuffle layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShuffleLayer
    //!
    //! \return The new shuffle layer, or nullptr if it could not be created.
    //!
    virtual IShuffleLayer* addShuffle(ITensor& input) = 0;

    //!
    //! \brief Set the pooling output dimensions formula.
    //!
    //! \param formula The formula from computing the pooling output dimensions. If null is passed, the default formula is used.
    //!
    //! The default formula in each dimension is (inputDim + padding * 2 - kernelSize) / stride + 1.
    //!
    //! \see IOutputDimensionsFormula getPoolingOutputDimensionsFormula()
    //!
    virtual void setPoolingOutputDimensionsFormula(IOutputDimensionsFormula* formula) = 0;

    //!
    //! \brief Get the pooling output dimensions formula.
    //!
    //! \return The formula from computing the pooling output dimensions.
    //!
    //! \see IOutputDimensionsFormula setPoolingOutputDimensionsFormula()
    //!
    virtual IOutputDimensionsFormula& getPoolingOutputDimensionsFormula() const = 0;

    //!
    //! \brief Set the convolution output dimensions formula.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \param formula The formula from computing the convolution output dimensions. If null is passed, the default formula is used.
    //!
    //! The default formula in each dimension is (inputDim + padding * 2 - kernelSize) / stride + 1.
    //!
    //! \see IOutputDimensionsFormula getConvolutionOutputDimensionsFormula()
    //!
    virtual void setConvolutionOutputDimensionsFormula(IOutputDimensionsFormula* formula) = 0;

    //!
    //! \brief Get the convolution output dimensions formula.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \return The formula from computing the convolution output dimensions.
    //!
    //! \see IOutputDimensionsFormula setConvolutionOutputDimensionsFormula()
    //!
    virtual IOutputDimensionsFormula& getConvolutionOutputDimensionsFormula() const = 0;

    //!
    //! \brief Set the deconvolution output dimensions formula.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \param formula The formula from computing the deconvolution output dimensions. If null is passed, the default formula is used.
    //!
    //! The default formula in each dimension is (inputDim - 1) * stride + kernelSize - 2 * padding.
    //!
    //! \see IOutputDimensionsFormula getDevonvolutionOutputDimensionsFormula()
    //!
    virtual void setDeconvolutionOutputDimensionsFormula(IOutputDimensionsFormula* formula) = 0;

    //!
    //! \brief Get the deconvolution output dimensions formula.
    //!
    //! \return The formula from computing the deconvolution output dimensions.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \see IOutputDimensionsFormula setDeconvolutionOutputDimensionsFormula()
    //!
    virtual IOutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const = 0;

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! \return The number of layers in the network.
    //!
    //! \see getLayer()
    //!
    virtual int getNbLayers() const = 0;

    //!
    //! \brief Get the layer specified by the given index.
    //!
    //! \param index The index of the layer.
    //!
    //! \return The layer, or nullptr if the index is out of range.
    //!
    //! \see getNbLayers()
    //!
    virtual ILayer* getLayer(int index) const = 0;

    //!
    //! \brief Get the number of inputs in the network.
    //!
    //! \return The number of inputs in the network.
    //!
    //! \see getInput()
    //!
    virtual int getNbInputs() const = 0;

    //!
    //! \brief Get the input tensor specified by the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range.
    //!
    //! \see getNbInputs()
    //!
    virtual ITensor* getInput(int index) const = 0; // adding inputs invalidates indexing here

    //!
    //! \brief Get the number of outputs in the network.
    //!
    //! \return The number of outputs in the network.
    //!
    //! \see getOutput()
    //!
    virtual int getNbOutputs() const = 0;

    //!
    //! \brief Get the output tensor specified by the given index.
    //!
    //! \param index The index of the output tensor.
    //!
    //! \return The output tensor, or nullptr if the index is out of range.
    //!
    //! \see getNbOutputs()
    //!
    virtual ITensor* getOutput(int index) const = 0; // adding outputs invalidates indexing here

    //!
    //! \brief Destroy this INetworkDefinition object.
    //!
    virtual void destroy() = 0;

protected:
    virtual ~INetworkDefinition() {}

public:
    //!
    //! \brief Add a reduce layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The reduction operation to perform.
    //! \param reduceAxes The reduction dimensions.
    //!        Bit 0 of the uint32_t type corresponds to the non-batch dimension 0 boolean and so on.
    //!        If a bit is set, then the corresponding dimension will be reduced.
    //!        Let's say we have an NCHW tensor as input (three non-batch dimensions).
    //!        Bit 0 corresponds to the C dimension boolean.
    //!        Bit 1 corresponds to the H dimension boolean.
    //!        Bit 2 corresponds to the W dimension boolean.
    //!        Note that reduction is not permitted over the batch size dimension.
    //! \param keepDimensions The boolean that specifies whether or not to keep the reduced dimensions in the output of the layer.
    //!
    //! \see IReduceLayer
    //!
    //! \return The new reduce layer, or nullptr if it could not be created.
    //!
    virtual IReduceLayer* addReduce(ITensor& input, ReduceOperation operation, uint32_t reduceAxes, bool keepDimensions) = 0;

    //!
    //! \brief Add a TopK layer to the network.
    //!
    //! The TopK layer has two outputs of the same dimensions. The first contains data values,
    //! the second contains index positions for the values. Output values are sorted, largest first
    //! for operation kMAX and smallest first for operation kMIN.
    //!
    //! Currently only values of K up to 1024 are supported.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \param op Operation to perform.
    //!
    //! \param k Number of elements to keep.
    //!
    //! \param reduceAxes The reduction dimensions.
    //!        Bit 0 of the uint32_t type corresponds to the non-batch dimension 0 boolean and so on.
    //!        If a bit is set, then the corresponding dimension will be reduced.
    //!        Let's say we have an NCHW tensor as input (three non-batch dimensions).
    //!        Bit 0 corresponds to the C dimension boolean.
    //!        Bit 1 corresponds to the H dimension boolean.
    //!        Bit 2 corresponds to the W dimension boolean.
    //!        Note that TopK reduction is currently only permitted over one dimension.
    //!
    //! \see ITopKLayer
    //!
    //! \return The new TopK layer, or nullptr if it could not be created.
    //!
    virtual ITopKLayer* addTopK(ITensor& input, TopKOperation op, int k, uint32_t reduceAxes) = 0;

    //!
    //! \brief Add a gather layer to the network.
    //!
    //! \param data The tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param axis The non-batch dimension axis in the data tensor to gather on.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it could not be created.
    //!
    virtual IGatherLayer* addGather(ITensor& data, ITensor& indices, int axis) = 0;

    //!
    //! \brief Add a RaggedSoftMax layer to the network.
    //!
    //! \param input The ZxS input tensor.
    //! \param bounds The Zx1 bounds tensor.
    //!
    //! \see IRaggedSoftMaxLayer
    //!
    //! \return The new RaggedSoftMax layer, or nullptr if it could not be created.
    //!
    virtual IRaggedSoftMaxLayer* addRaggedSoftMax(ITensor& input, ITensor& bounds) = 0;

    //!
    //! \brief Add a MatrixMultiply layer to the network.
    //!
    //! \param input0 The first input tensor (commonly A).
    //! \param op0 The operation to apply to input0.
    //! \param input1 The second input tensor (commonly B).
    //! \param op1 The operation to apply to input1.
    //!
    //! \see IMatrixMultiplyLayer
    //!
    //! \return The new matrix multiply layer, or nullptr if it could not be created.
    //!
    virtual IMatrixMultiplyLayer* addMatrixMultiply(ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) = 0;

    //!
    //! \brief Add a MatrixMultiply layer to the network.
    //!
    //! \param input0 The first input tensor (commonly A).
    //! \param transpose0 If true, op(input0)=transpose(input0), else op(input0)=input0.
    //! \param input1 The second input tensor (commonly B).
    //! \param transpose1 If true, op(input1)=transpose(input1), else op(input1)=input1.
    //!
    //! \see IMatrixMultiplyLayer
    //!
    //! \return The new matrix multiply layer, or nullptr if it could not be created.
    //!
    //! \deprecated This interface is superseded by the overload that replaces bool with MatrixOperation.
    //!
    virtual IMatrixMultiplyLayer* addMatrixMultiply(ITensor& input0, bool transpose0, ITensor& input1, bool transpose1) = 0;

    //!
    //! \brief Add a constant layer to the network.
    //!
    //! \param dimensions The dimensions of the constant.
    //! \param weights The constant value, represented as weights.
    //!
    //! \see IConstantLayer
    //!
    //! \return The new constant layer, or nullptr if it could not be created.
    //!
    //! If weights.type is DataType::kINT32, the output is a tensor of 32-bit indices.
    //! Otherwise the output is a tensor of real values, and the output type will be
    //! FP32, FP16, or quantized INT8 following TensorRT's normal precision rules.
    //!
    virtual IConstantLayer* addConstant(Dims dimensions, Weights weights) = 0;

    //!
    //! \brief Add an \p layerCount deep RNN layer to the network with \p hiddenSize internal states that can
    //! take a batch with fixed or variable sequence lengths.
    //!
    //! \param input The input tensor to the layer (see below).
    //! \param layerCount The number of layers in the RNN.
    //! \param hiddenSize Size of the internal hidden state for each layer.
    //! \param maxSeqLen Maximum sequence length for the input.
    //! \param op The type of RNN to execute.
    //!
    //! By default, the layer is configured with RNNDirection::kUNIDIRECTION and RNNInputMode::kLINEAR.
    //! To change these settings, use IRNNv2Layer::setDirection() and IRNNv2Layer::setInputMode().
    //!
    //! %Weights and biases for the added layer should be set using
    //! IRNNv2Layer::setWeightsForGate() and IRNNv2Layer::setBiasForGate() prior
    //! to building an engine using this network.
    //!
    //! The input tensors must be of the type DataType::kFLOAT or DataType::kHALF.
    //! The layout of the weights is row major and must be the same datatype as the input tensor.
    //! \p weights contain 8 matrices and \p bias contains 8 vectors.
    //!
    //! See IRNNv2Layer::setWeightsForGate() and IRNNv2Layer::setBiasForGate() for details on the required input
    //! format for \p weights and \p bias.
    //!
    //! The \p input ITensor should contain zero or more index dimensions `{N1, ..., Np}`, followed by
    //! two dimensions, defined as follows:
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `E` specifies the embedding length (unless ::kSKIP is set, in which case it should match
    //!     getHiddenSize()).
    //!
    //! By default, all sequences in the input are assumed to be size \p maxSeqLen.  To provide explicit sequence
    //! lengths for each input sequence in the batch, use IRNNv2Layer::setSequenceLengths().
    //!
    //! The RNN layer outputs up to three tensors.
    //!
    //! The first output tensor is the output of the final RNN layer across all timesteps, with dimensions
    //! `{N1, ..., Np, S_max, H}`:
    //!
    //!   - `N1..Np` are the index dimensions specified by the input tensor
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `H` is an output hidden state (equal to getHiddenSize() or 2x getHiddenSize())
    //!
    //! The second tensor is the final hidden state of the RNN across all layers, and if the RNN
    //! is an LSTM (i.e. getOperation() is ::kLSTM), then the third tensor is the final cell state
    //! of the RNN across all layers.  Both the second and third output tensors have dimensions
    //! `{N1, ..., Np, L, H}`:
    //!
    //!  - `N1..Np` are the index dimensions specified by the input tensor
    //!  - `L` is the number of layers in the RNN, equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2x getLayerCount() if getDirection is ::kBIDIRECTION. In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L= 2*l + 1`.
    //!  - `H` is the hidden state for each layer, equal to getHiddenSize().
    //!
    //! \see IRNNv2Layer
    //!
    //! \return The new RNN layer, or nullptr if it could not be created.
    //!
    virtual IRNNv2Layer* addRNNv2(ITensor& input, int32_t layerCount, int32_t hiddenSize, int32_t maxSeqLen, RNNOperation op) = 0;

    //!
    //! \brief Add a plugin layer to the network using an IPluginExt interface.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginLayer
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    virtual IPluginLayer* addPluginExt(ITensor* const* inputs, int nbInputs, IPluginExt& plugin) = 0;

    //!
    //! \brief Add an identity layer.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IIdentityLayer
    //!
    //! \return The new identity layer, or nullptr if it could not be created.
    //!
    virtual IIdentityLayer* addIdentity(ITensor& input) = 0;

    //!
    //! \brief remove a tensor from the network definition.
    //!
    //! \param tensor the tensor to remove
    //!
    //! It is illegal to remove a tensor that is the input or output of a layer.
    //! if this method is called with such a tensor, a warning will be emitted on the log
    //! and the call will be ignored. Its intended use is to remove detached tensors after
    //! e.g. concatenating two networks with Layer::setInput().
    //!
    virtual void removeTensor(ITensor& tensor) = 0;

    //!
    //! \brief unmark a tensor as a network output.
    //!
    //! \param tensor The tensor to unmark as an output tensor.
    //!
    //! see markOutput()
    //!
    virtual void unmarkOutput(ITensor& tensor) = 0;

    //!
    //! \brief Add a plugin layer to the network using the IPluginV2 interface.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginV2Layer
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    virtual IPluginV2Layer* addPluginV2(ITensor* const* inputs, int nbInputs, IPluginV2& plugin) = 0;

    //!
    //! \brief Add a slice layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param start The start offset
    //! \param size The output dimension
    //! \param stride The slicing stride
    //!
    //! Positive, negative, zero stride values, and combinations of them in different dimensions are allowed.
    //!
    //! \see ISliceLayer
    //!
    //! \return The new slice layer, or nullptr if it could not be created.
    //!
    virtual ISliceLayer* addSlice(ITensor& input, Dims start, Dims size, Dims stride) = 0;
};

//!
//! \class IProfiler
//!
//! \brief Application-implemented interface for profiling.
//!
//! When this class is added to an execution context, the profiler will be called once per layer for each invocation of execute().
//! Note that enqueue() does not currently support profiling.
//!
//! The profiler will only be called after execution is complete. It has a small impact on execution time.
//!
class IProfiler
{
public:
    //!
    //! \brief Layer time reporting callback.
    //!
    //! \param layerName The name of the layer, set when constructing the network definition.
    //! \param ms The time in milliseconds to execute the layer.
    //!
    virtual void reportLayerTime(const char* layerName, float ms) = 0;

    virtual ~IProfiler() {}
};

class ICudaEngine;

//!
//! \class IExecutionContext
//!
//! \brief Context for executing inference using an engine.
//!
//! Multiple execution contexts may exist for one ICudaEngine instance, allowing the same
//! engine to be used for the execution of multiple batches simultaneously.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IExecutionContext
{
public:
    //!
    //! \brief Synchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be queried using ICudaEngine::getBindingIndex()
    //! \param batchSize The batch size. This is at most the value supplied when the engine was built.
    //! \param bindings An array of pointers to input and output buffers for the network.
    //!
    //! \return True if execution succeeded.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    virtual bool execute(int batchSize, void** bindings) = 0;

    //!
    //! \brief Asynchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be queried using ICudaEngine::getBindingIndex()
    //! \param batchSize The batch size. This is at most the value supplied when the engine was built.
    //! \param bindings An array of pointers to input and output buffers for the network.
    //! \param stream A cuda stream on which the inference kernels will be enqueued
    //! \param inputConsumed An optional event which will be signaled when the input buffers can be refilled with new data
    //!
    //! \return True if the kernels were enqueued successfully.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    virtual bool enqueue(int batchSize, void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) = 0;

    //!
    //! \brief Set the debug sync flag.
    //!
    //! If this flag is set to true, the engine will log the successful execution for each kernel during execute(). It has no effect when using enqueue().
    //!
    //! \see getDebugSync()
    //!
    virtual void setDebugSync(bool sync) = 0;

    //!
    //! \brief Get the debug sync flag.
    //!
    //! \see setDebugSync()
    //!
    virtual bool getDebugSync() const = 0;

    //!
    //! \brief Set the profiler.
    //!
    //! \see IProfiler getProfiler()
    //!
    virtual void setProfiler(IProfiler*) = 0;

    //!
    //! \brief Get the profiler.
    //!
    //! \see IProfiler setProfiler()
    //!
    virtual IProfiler* getProfiler() const = 0;

    //!
    //! \brief Get the associated engine.
    //!
    //! \see ICudaEngine
    //!
    virtual const ICudaEngine& getEngine() const = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() = 0;

protected:
    virtual ~IExecutionContext() {}

public:
    //!
    //! \brief Set the name of the execution context.
    //!
    //! This method copies the name string.
    //!
    //! \see getName()
    //!
    virtual void setName(const char* name) = 0;

    //!
    //! \brief Return the name of the execution context.
    //!
    //! \see setName()
    //!
    virtual const char* getName() const = 0;

    //!
    //! \brief set the device memory for use by this execution context.
    //!
    //! The memory must be aligned with cuda memory alignment property (using cudaGetDeviceProperties()), and its size must be at least that
    //! returned by getDeviceMemorySize(). If using enqueue() to run the network, The memory is in
    //! use from the invocation of enqueue() until network execution is complete. If using execute(),
    //! it is in use until execute() returns. Releasing or otherwise using the memory for other
    //! purposes during this time will result in undefined behavior.
    //!
    //! \see ICudaEngine::getDeviceMemorySize() ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    virtual void setDeviceMemory(void* memory) = 0;
};

//!
//! \class ICudaEngine
//!
//! \brief An engine for executing inference on a built network.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICudaEngine
{
public:
    //!
    //! \brief Get the number of binding indices.
    //!
    //! \see getBindingIndex();
    //!
    virtual int getNbBindings() const = 0;

    //!
    //! \brief Retrieve the binding index for a named tensor.
    //!
    //! IExecutionContext::enqueue() and IExecutionContext::execute() require an array of buffers.
    //!
    //! Engine bindings map from tensor names to indices in this array.
    //! Binding indices are assigned at engine build time, and take values in the range [0 ... n-1] where n is the total number of inputs and outputs.
    //!
    //! \param name The tensor name.
    //! \return The binding index for the named tensor, or -1 if the name is not found.
    //!
    //! see getNbBindings() getBindingIndex()
    //!
    virtual int getBindingIndex(const char* name) const = 0;

    //!
    //! \brief Retrieve the name corresponding to a binding index.
    //!
    //! This is the reverse mapping to that provided by getBindingIndex().
    //!
    //! \param bindingIndex The binding index.
    //! \return The name corresponding to the index, or nullptr if the index is out of range.
    //!
    //! \see getBindingIndex()
    //!
    virtual const char* getBindingName(int bindingIndex) const = 0;

    //!
    //! \brief Determine whether a binding is an input binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return True if the index corresponds to an input binding and the index is in range.
    //!
    //! \see getBindingIndex()
    //!
    virtual bool bindingIsInput(int bindingIndex) const = 0;

    //!
    //! \brief Get the dimensions of a binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return The dimensions of the binding if the index is in range, otherwise (0,0,0).
    //!
    //! \see getBindingIndex()
    //!
    virtual Dims getBindingDimensions(int bindingIndex) const = 0;

    //!
    //! \brief Determine the required data type for a buffer from its binding index.
    //!
    //! \param bindingIndex The binding index.
    //! \return The type of the data in the buffer.
    //!
    //! \see getBindingIndex()
    //!
    virtual DataType getBindingDataType(int bindingIndex) const = 0;

    //!
    //! \brief Get the maximum batch size which can be used for inference.
    //!
    //! \return The maximum batch size for this engine.
    //!
    virtual int getMaxBatchSize() const = 0;

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! The number of layers in the network is not necessarily the number in the original network definition, as layers may be combined or eliminated as the engine is
    //! optimized. This value can be useful when building per-layer tables, such as when aggregating profiling data over a number of executions.
    //!
    //! \return The number of layers in the network.
    //!
    virtual int getNbLayers() const = 0;

    //!
    //! \brief Get the amount of workspace the engine uses.
    //!
    //! The workspace size will be no greater than the value provided to the builder when the engine was built, and will typically be smaller.
    //! Workspace will be allocated for each execution context.
    //!
    virtual std::size_t getWorkspaceSize() const = 0;

    //!
    //! \brief Serialize the network to a stream.
    //!
    //! \return A IHostMemory object that contains the serialized engine.
    //!
    //! The network may be deserialized with IRuntime::deserializeCudaEngine()
    //!
    //! \see IRuntime::deserializeCudaEngine()
    //!
    virtual IHostMemory* serialize() const = 0;

    //!
    //! \brief Create an execution context.
    //!
    //! \see IExecutionContext.
    //!
    virtual IExecutionContext* createExecutionContext() = 0;

    //!
    //! \brief Destroy this object;
    //!
    virtual void destroy() = 0;

    //!
    //! \brief Get location of binding
    //!
    //! This lets you know whether the binding should be a pointer to device or host memory.
    //!
    //! \see ITensor::setLocation() ITensor::getLocation()
    //!
    //! \param bindingIndex The binding index.
    //! \return The location of the bound tensor with given index.
    //!
    virtual TensorLocation getLocation(int bindingIndex) const = 0;

protected:
    virtual ~ICudaEngine() {}

public:
    //!
    //! \brief create an execution context without any device memory allocated
    //!
    //! The memory for execution of this device context must be supplied by the application.
    //!
    //! \see getDeviceMemorySize() IExecutionContext::setDeviceMemory()
    //!
    virtual IExecutionContext* createExecutionContextWithoutDeviceMemory() = 0;

    //!
    //! \brief Return the amount of device memory required by an execution context.
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    virtual size_t getDeviceMemorySize() const = 0;

    //!
    //! \brief Return true if engine can be refit.
    //!
    //! \see nvinfer1::createInferRefitter()
    //!
    virtual bool isRefittable() const = 0;
};

//!
//! enum CalibrationAlgoType
//!
//! \brief Version of calibration algorithm to use.
//!
enum class CalibrationAlgoType : int
{
    kLEGACY_CALIBRATION = 0,
    kENTROPY_CALIBRATION = 1,
    kENTROPY_CALIBRATION_2 = 2
};

template <>
inline int EnumMax<CalibrationAlgoType>()
{
    return 3;
} //!< Maximum number of elements in CalibrationAlgoType enum. \see DataType

//!
//! \class IInt8Calibrator
//!
//! \brief Application-implemented interface for calibration.
//!
//! Calibration is a step performed by the builder when deciding suitable scale factors for 8-bit inference.
//!
//! It must also provide a method for retrieving representative images which the calibration process can use to examine
//! the distribution of activations. It may optionally implement a method for caching the calibration result for reuse
//! on subsequent runs.
//!
class IInt8Calibrator
{
public:
    //!
    //! \brief Get the batch size used for calibration batches.
    //!
    //! \return The batch size.
    //!
    virtual int getBatchSize() const = 0;

    //!
    //! \brief Get a batch of input for calibration.
    //!
    //! The batch size of the input must match the batch size returned by getBatchSize().
    //!
    //! \param bindings An array of pointers to device memory that must be updated to point to device memory containing each network input data.
    //! \param names The names of the network input for each pointer in the binding array.
    //! \param nbBindings The number of pointers in the bindings array.
    //! \return False if there are no more batches for calibration.
    //!
    //!
    //! \see getBatchSize()
    //!
    virtual bool getBatch(void* bindings[], const char* names[], int nbBindings) = 0; // get a pointer to the input batch

    //!
    //! \brief Load a calibration cache.
    //!
    //! Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
    //! of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
    //! these do not batch the settings of the current calibrator. However, the network should also be recalibrated if its structure
    //! changes, or the input data set changes, and it is the responsibility of the application to ensure this.
    //!
    //! \param length The length of the cached data, that should be set by the called function. If there is no data, this should be zero.
    //!
    //! \return A pointer to the cache, or nullptr if there is no data.
    //!
    virtual const void* readCalibrationCache(std::size_t& length) = 0;

    //!
    //! \brief Save a calibration cache.
    //!
    //! \param ptr A pointer to the data to cache.
    //! \param length The length in bytes of the data to cache.
    //!
    //! \see readCalibrationCache()
    //!
    virtual void writeCalibrationCache(const void* ptr, std::size_t length) = 0;

    //!
    //! \brief Get the algorithm used by this calibrator.
    //!
    //! \return The algorithm used by the calibrator.
    //!
    virtual CalibrationAlgoType getAlgorithm() = 0;

    virtual ~IInt8Calibrator() {}
};

//!
//! Entropy calibrator. This is the Legacy Entropy calibrator. It is less complicated than the legacy calibrator and produces better results.
//!
class IInt8EntropyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the entropy calibrator.
    //!
    virtual CalibrationAlgoType getAlgorithm() { return CalibrationAlgoType::kENTROPY_CALIBRATION; }

    virtual ~IInt8EntropyCalibrator() {}
};

//!
//! Entropy calibrator 2. This is the preferred calibrator. This is the required calibrator for DLA, as it supports per activation tensor scaling.
//!
class IInt8EntropyCalibrator2 : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the entropy calibrator 2.
    //!
    CalibrationAlgoType getAlgorithm() override { return CalibrationAlgoType::kENTROPY_CALIBRATION_2; }

    virtual ~IInt8EntropyCalibrator2() {}
};

//!
//! Legacy calibrator for compatibility with 2.0 EA. Will be removed in 2.2.
//! \deprecated
//!
class IInt8LegacyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the legacy calibrator.
    //!
    virtual CalibrationAlgoType getAlgorithm() { return CalibrationAlgoType::kLEGACY_CALIBRATION; }

    //!
    //! \brief The quantile (between 0 and 1) that will be used to select the region maximum when the quantile method is in use.
    //!
    //! See the user guide for more details on how the quantile is used.
    //!
    virtual double getQuantile() const = 0;

    //!
    //! \brief The fraction (between 0 and 1) of the maximum used to define the regression cutoff when using regression to determine the region maximum.
    //!
    //! See the user guide for more details on how the regression cutoff is used
    //!
    virtual double getRegressionCutoff() const = 0;

    //!
    //! \brief Load a histogram.
    //!
    //! Histogram generation is potentially expensive, so it can be useful to generate the histograms once, then use them when exploring
    //! the space of calibrations. The histograms should be regenerated if the network structure
    //! changes, or the input data set changes, and it is the responsibility of the application to ensure this.
    //!
    //! \param length The length of the cached data, that should be set by the called function. If there is no data, this should be zero.
    //!
    //! \return A pointer to the cache, or nullptr if there is no data.
    //!
    virtual const void* readHistogramCache(std::size_t& length) = 0;

    //!
    //! \brief Save a histogram cache.
    //!
    //! \param ptr A pointer to the data to cache.
    //! \param length The length in bytes of the data to cache.
    //!
    //! \see readHistogramCache()
    //!
    virtual void writeHistogramCache(const void* ptr, std::size_t length) = 0;

    virtual ~IInt8LegacyCalibrator() {}
};

//!
//! \enum EngineCapability
//!
//! \brief List of supported engine capability flows.
//!
//! \note at present, kSAFE_DLA flow doesn't strictly limit execution to DLA/PVA devices - it simply
//! restricts the engine capabilities to DLA/PVA support levels anticipated in future releases.
//!
enum class EngineCapability
{
    kDEFAULT = 0,   //!< Full capability, TensorRT mode without any restrictions.
    kSAFE_GPU = 1,  //!< Safety restricted capability, TensorRT flow that can only run on GPU devices.
    kSAFE_DLA = 2,  //!< Safety restricted capability, TensorRT flow that can only run on DLA/PVA devices.
};

template <>
inline int EnumMax<EngineCapability>()
{
    return 3;
} //!< Maximum number of elements in EngineCapability enum. \see EngineCapability

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
    virtual void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) = 0;

    //!
    //! A callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //!
    virtual void free(void* memory) = 0;

    //!
    //! Destructor declared virtual as general good practice for a class with virtual methods.
    //! TensorRT never calls the destructor for an IGpuAllocator defined by the application.
    //!
    virtual ~IGpuAllocator() {}
};

//!
//! \class IBuilder
//!
//! \brief Builds an engine from a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IBuilder
{
public:
    //!
    //! \brief Create a network definition object.
    //!
    //! \see INetworkDefinition
    //!
    virtual nvinfer1::INetworkDefinition* createNetwork() = 0;

    //!
    //! \brief Set the maximum batch size.
    //!
    //! \param batchSize The maximum batch size which can be used at execution time, and also the batch size for which the engine will be optimized.
    //!
    //! \see getMaxBatchSize()
    //!
    virtual void setMaxBatchSize(int batchSize) = 0;

    //!
    //! \brief Get the maximum batch size.
    //!
    //! \return The maximum batch size.
    //!
    //! \see setMaxBatchSize()
    //! \see getMaxDLABatchSize()
    //!
    virtual int getMaxBatchSize() const = 0;

    //!
    //! \brief Set the maximum workspace size.
    //!
    //! \param workspaceSize The maximum GPU temporary memory which the engine can use at execution time.
    //!
    //! \see getMaxWorkspaceSize()
    //!
    virtual void setMaxWorkspaceSize(std::size_t workspaceSize) = 0;

    //!
    //! \brief Get the maximum workspace size.
    //!
    //! \return The maximum workspace size.
    //!
    //! \see setMaxWorkspaceSize()
    //!
    virtual std::size_t getMaxWorkspaceSize() const = 0;

    //!
    //! \brief Set whether half2 mode is used.
    //!
    //! half2 mode is a paired-image mode that is significantly faster for batch sizes greater than one on platforms with fp16 support.
    //!
    //! \param mode Whether half2 mode is used.
    //!
    //! \see getHalf2Mode()
    //!
    //! \deprecated This function is superseded by setFp16Mode.
    //!
    virtual void setHalf2Mode(bool mode) = 0;

    //!
    //! \brief Query whether half2 mode is used.
    //!
    //! \see setHalf2Mode()
    //!
    //! \deprecated This function is superseded by getFp16Mode.
    //!
    virtual bool getHalf2Mode() const = 0;

    //!
    //! \brief Set whether the builder should use debug synchronization.
    //!
    //! If this flag is true, the builder will synchronize after timing each layer, and report the layer name. It can be useful when diagnosing issues at build time.
    //!
    virtual void setDebugSync(bool sync) = 0;

    //!
    //! \brief Query whether the builder will use debug synchronization.
    //!
    //! \see setDebugSync()
    //!
    virtual bool getDebugSync() const = 0;

    //!
    //! \brief Set the number of minimization iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations
    //! used in minimization.
    //!
    //! \see getMinFindIterations()
    //!
    virtual void setMinFindIterations(int minFind) = 0;

    //!
    //! \brief Query the number of minimization iterations.
    //!
    //! \see setMinFindIterations()
    //!
    virtual int getMinFindIterations() const = 0;

    //!
    //! \brief Set the number of averaging iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations
    //! used in averaging.
    //!
    //! \see getAverageFindIterations()
    //!
    virtual void setAverageFindIterations(int avgFind) = 0;

    //!
    //! \brief Query the number of averaging iterations.
    //!
    //! \see setAverageFindIterations()
    //!
    virtual int getAverageFindIterations() const = 0;

    //!
    //! \brief Build a CUDA engine from a network definition.
    //!
    //! \see INetworkDefinition ICudaEngine
    //!
    virtual nvinfer1::ICudaEngine* buildCudaEngine(nvinfer1::INetworkDefinition& network) = 0;

    //!
    //! \brief Determine whether the platform has fast native fp16.
    //!
    virtual bool platformHasFastFp16() const = 0;

    //!
    //! \brief Determine whether the platform has fast native int8.
    //!
    virtual bool platformHasFastInt8() const = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() = 0;

    //!
    //! \brief Set the maximum value for a region.
    //!
    //! Used for INT8 mode compression.
    //!
    virtual void setInt8Mode(bool mode) = 0;

    //!
    //! \brief Query whether Int8 mode is used.
    //!
    //! \see setInt8Mode()
    //!
    virtual bool getInt8Mode() const = 0;

    //!
    //! \brief Set Int8 Calibration interface.
    //!
    virtual void setInt8Calibrator(IInt8Calibrator* calibrator) = 0;

    //!
    //! \brief Set the device that this layer must execute on.
    //! \param DeviceType that this layer must execute on.
    //! If DeviceType is not set or is reset, TensorRT will use the default DeviceType set in the builder.
    //!
    //! \note The device type for a layer must be compatible with the safety flow (if specified).
    //! For example a layer cannot be marked for DLA execution while the builder is configured for kSAFE_GPU.
    //!
    //! \see getDeviceType()
    //!
    virtual void setDeviceType(ILayer* layer, DeviceType deviceType) = 0;

    //!
    //! \brief Get the device that this layer executes on.
    //! \return Returns DeviceType of the layer.
    //!
    virtual DeviceType getDeviceType(const ILayer* layer) const = 0;

    //!
    //! \brief whether the DeviceType has been explicitly set for this layer
    //! \return whether the DeviceType has been explicitly set
    //! \see setDeviceType() getDeviceType() resetDeviceType()
    //!
    virtual bool isDeviceTypeSet(const ILayer* layer) const = 0;

    //!
    //! \brief reset the DeviceType for this layer
    //!
    //! \see setDeviceType() getDeviceType() isDeviceTypeSet()
    //!
    virtual void resetDeviceType(ILayer* layer) = 0;

    //!
    //! \brief Checks if a layer can run on DLA.
    //! \return status true if the layer can on DLA else returns false.
    //!
    virtual bool canRunOnDLA(const ILayer* layer) const = 0;

    //!
    //! \brief Sets the default DeviceType to be used by the builder. It ensures that all the layers that can run on this device will run on it, unless setDeviceType is used to override the default DeviceType for a layer.
    //! \see getDefaultDeviceType()
    //!
    virtual void setDefaultDeviceType(DeviceType deviceType) = 0;

    //!
    //! \brief Get the default DeviceType which was set by setDefaultDeviceType.
    //!
    virtual DeviceType getDefaultDeviceType() const = 0;

    //!
    //! \brief Get the maximum batch size DLA can support.
    //! For any tensor the total volume of index dimensions combined(dimensions other than CHW) with the requested batch size should not exceed the value returned by this function.
    //!
    virtual int getMaxDLABatchSize() const = 0;

    //!
    //! \brief Sets the builder to use GPU if a layer that was supposed to run on DLA can not run on DLA.
    //! \param Allows fallback if setFallBackMode is true else disables fallback option.
    //!
    //! \note GPU fallback may only be specified for non-safety modes. \see EngineCapability
    //! Simultaneously enabling GPU fallback and safety-restricted modes is disallowed.
    //!
    virtual void allowGPUFallback(bool setFallBackMode) = 0;

    //!
    //! \brief Returns number of DLA hardware cores accessible.
    //!
    virtual int getNbDLACores() const = 0;

    //!
    //! \brief Set the DLA core that the engine must execute on.
    //! \param dlaCore The DLA core to execute the engine on (0 to N-1, where N is the maximum number of DLA cores present on the device). Default value is 0.
    //! DLA Core is not a property of the engine that is preserved by serialization: when the engine is deserialized it will be associated with the DLA core which is configured for the runtime.
    //! \see IRuntime::setDLACore() getDLACore()
    //!
    virtual void setDLACore(int dlaCore) = 0;

    //!
    //! \brief Get the DLA core that the engine executes on.
    //! \return If setDLACore is called, returns DLA core from 0 to N-1, else returns 0.
    //!
    virtual int getDLACore() const = 0;

    //!
    //! \brief Resets the builder state
    //!
    virtual void reset(nvinfer1::INetworkDefinition& network) = 0;

protected:
    virtual ~IBuilder() {}

public:
    //!
    //! \brief Set the GPU allocator.
    //! \param allocator Set the GPU allocator to be used by the builder. All GPU memory acquired will use this allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! \note This allocator will be passed to any engines created via the builder; thus the lifetime of the allocator must span the lifetime of those engines as
    //! well as that of the builder. If nullptr is passed, the default allocator will be used.
    //!
    virtual void setGpuAllocator(IGpuAllocator* allocator) = 0;

    //!
    //! \brief Set whether or not 16-bit kernels are permitted.
    //!
    //! During engine build fp16 kernels will also be tried when this mode is enabled.
    //!
    //! \param mode Whether 16-bit kernels are permitted.
    //!
    //! \see getFp16Mode()
    //!
    virtual void setFp16Mode(bool mode) = 0;

    //!
    //! \brief Query whether 16-bit kernels are permitted.
    //!
    //! \see setFp16Mode()
    //!
    virtual bool getFp16Mode() const = 0;

    //!
    //! \brief Set whether or not type constraints are strict.
    //!
    //! When strict type constraints are in use, TensorRT will always choose a layer implementation that conforms to the type constraints
    //! specified, if one exists. If this flag is not set, a higher-precision implementation may be chosen if it results in higher performance.
    //!
    //! If no conformant layer exists, TensorRT will choose a non-conformant layer if available regardless of the setting of this flag.
    //!
    //! See the developer guide for the definition of strictness.
    //!
    //! \param mode Whether type constraints are strict
    //!
    //! \see getStrictTypeConstraints()
    //!
    virtual void setStrictTypeConstraints(bool mode) = 0;

    //!
    //! \brief Query whether or not type constraints are strict.
    //!
    //! \see setStrictTypeConstraints()
    //!
    virtual bool getStrictTypeConstraints() const = 0;

    //!
    //! Set whether engines will be refittable.
    //!
    virtual void setRefittable(bool canRefit) = 0;

    //!
    //! \brief Query whether or not engines will be refittable.
    //!
    //! \see getRefittable()
    //!
    virtual bool getRefittable() const = 0;

    //!
    //! \brief Configure the builder to target specified EngineCapability flow.
    //!
    virtual void setEngineCapability(EngineCapability capability) = 0;

    //!
    //! \brief Query EngineCapability flow configured for the builder.
    //!
    //! \see setEngineCapability()
    //!
    virtual EngineCapability getEngineCapability() const = 0;
};

//!
//! \enum WeightsRole
//! \brief How a layer uses particular Weights.
//!
//! The power weights of an IScaleLayer are omitted.  Refitting those is not supported.
//!
enum class WeightsRole : int
{
    kKERNEL = 0,   //!< kernel for IConvolutionLayer, IDeconvolutionLayer, or IFullyConnectedLayer
    kBIAS = 1,     //!< bias for IConvolutionLayer, IDeconvolutionLayer, or IFullyConnectedLayer
    kSHIFT = 2,    //!< shift part of IScaleLayer
    kSCALE = 3,    //!< scale part of IScaleLayer
    kCONSTANT = 4, //!< weights for IConstantLayer
};

template <>
inline int EnumMax<WeightsRole>()
{
    return 5;
} //!< Maximum number of elements in WeightsRole enum. \see WeightsRole

//!
//! \class IRefitter
//!
//! \brief Updates weights in an engine.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRefitter
{
public:
    //!
    //! \brief Specify new weights for a layer of given name.
    //! Returns true on success, or false if new weights are rejected.
    //! Possible reasons for rejection are:
    //!
    //! * There is no such layer by that name.
    //! * The layer does not have weights with the specified role.
    //! * The number of weights is inconsistent with the layer’s original specification.
    //!
    //! Modifying the weights before method refit() completes will result in undefined behavior.
    virtual bool setWeights(const char* layerName,
                            WeightsRole role, Weights weights)
        = 0;

    //!
    //! \brief Updates associated engine.  Return true if successful.
    //!
    //! Failure occurs if getMissing() != 0 before the call.
    //!
    virtual bool refitCudaEngine() = 0;

    //!
    //! \brief Get description of missing weights.
    //!
    //! For example, if some Weights have been set, but the engine was optimized
    //! in a way that combines weights, any unsupplied Weights in the combination
    //! are considered missing.
    //!
    //! \param size The number of items that can be safely written to a non-null layerNames or roles.
    //! \param layerNames Where to write the layer names.
    //! \param roles Where to write the weights roles.
    //!
    //! \return The number of missing Weights.
    //!
    //! If layerNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refitted, and becomes invalid when the engine is destroyed.
    //!
    virtual int getMissing(int size, const char** layerNames, WeightsRole* roles) = 0;

    //!
    //! \brief Get description of all weights that could be refit.
    //!
    //! \param size The number of items that can be safely written to a non-null layerNames or roles.
    //! \param layerNames Where to write the layer names.
    //! \param roles Where to write the weights roles.
    //!
    //! \return The number of Weights that could be refit.
    //!
    //! If layerNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refitted, and becomes invalid when the engine is destroyed.
    //!
    virtual int getAll(int size, const char** layerNames, WeightsRole* roles) = 0;

    virtual void destroy() = 0;

protected:
    virtual ~IRefitter() {}
};

//!
//! \class IPluginFactory
//!
//! \brief Plugin factory for deserialization.
//!
//! This Interface is guaranteed not to change for the same major version of TensorRT.
class IPluginFactory
{
public:
    //!
    //! \brief Create a plugin from serialized data.
    //!
    //! Responsibility of destroying this plugin lies with the application.
    //! It can be done anytime after consumers of this plugin are destroyed.
    //!
    //! \param layerName The name of the layer.
    //! \param serialData The serialized data.
    //! \param serialLength The length of the serialized data.
    //!
    //! \return The plugin.
    //!
    //! \see IPlugin::serialize()
    //!
    virtual IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) = 0;
};

//!
//! \class IRuntime
//!
//! \brief Allows a serialized engine to be deserialized.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRuntime
{
public:
    //!
    //! \brief Deserialize an engine from a stream.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory.
    //! \param pluginFactory The plugin factory, if any plugins are used by the network, otherwise nullptr.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    virtual nvinfer1::ICudaEngine* deserializeCudaEngine(const void* blob, std::size_t size, IPluginFactory* pluginFactory) = 0;

    //!
    //! \brief Set the DLA core that the deserialized engine must execute on.
    //! \param dlaCore The DLA core to execute the engine on (0 to N-1, where N is the maximum number of DLA's present on the device). Default value is 0.
    //! \see getDLACore()
    //!
    virtual void setDLACore(int dlaCore) = 0;

    //!
    //! \brief Get the DLA core that the engine executes on.
    //! \return If setDLACore is called, returns DLA core from 0 to N-1, else returns 0.
    //!
    virtual int getDLACore() const = 0;

    //!
    //! \brief Returns number of DLA hardware cores accessible.
    //!
    virtual int getNbDLACores() const = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() = 0;

protected:
    virtual ~IRuntime() {}

public:
    //!
    //! \brief Set the GPU allocator.
    //! \param allocator Set the GPU allocator to be used by the runtime. All GPU memory acquired will use this allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! If nullptr is passed, the default allocator will be used.
    //!
    virtual void setGpuAllocator(IGpuAllocator* allocator) = 0;
};

//!
//! \class ILogger
//!
//! \brief Application-implemented logging interface for the builder, engine and runtime.
//!
//! Note that although a logger is passed on creation to each instance of a IBuilder or IRuntime interface, the logger is internally considered a singleton, and thus
//! multiple instances of IRuntime and/or IBuilder must all use the same logger.
//!
class ILogger
{
public:
    //!
    //! \enum Severity
    //!
    //! The severity corresponding to a log message.
    //!
    enum class Severity
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
    virtual void log(Severity severity, const char* msg) = 0;

    virtual ~ILogger() {}
};

template <>
inline int EnumMax<ILogger::Severity>()
{
    return 5;
} //!< Maximum number of elements in ILogger::Severity enum. \see ILogger::Severity

} // namespace nvinfer1

extern "C" TENSORRTAPI void* createInferBuilder_INTERNAL(void* logger, int version);                //!< Internal C entry point for creating IBuilder.
extern "C" TENSORRTAPI void* createInferRefitter_INTERNAL(void* engine, void* logger, int version); //!< Internal C entry point for creating IRefitter.
extern "C" TENSORRTAPI void* createInferRuntime_INTERNAL(void* logger, int version);                //!< Internal C entry point for creating IRuntime.

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
//! \brief Create an instance of an IBuilder class.
//!
//! This class is the logging class for the builder.
//!
namespace // unnamed namespace in case the compiler doesn't inline these
{
inline IBuilder* createInferBuilder(ILogger& logger)
{
    return static_cast<IBuilder*>(createInferBuilder_INTERNAL(&logger, NV_TENSORRT_VERSION));
}

//!
//! \brief Create an instance of an IRefitter class.
//!
//! This class is the logging class for the refitter.
//!
inline IRefitter* createInferRefitter(ICudaEngine& engine, ILogger& logger)
{
    return static_cast<IRefitter*>(createInferRefitter_INTERNAL(&engine, &logger, NV_TENSORRT_VERSION));
}

//!
//! \brief Create an instance of an IRuntime class.
//!
//! This class is the logging class for the runtime.
//!
inline IRuntime* createInferRuntime(ILogger& logger)
{
    return static_cast<IRuntime*>(createInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}
}

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
}

#endif
