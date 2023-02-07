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

#ifndef NV_INFER_H
#define NV_INFER_H

#include "NvInferLegacyDims.h"
#include "NvInferRuntime.h"

//!
//! \mainpage
//!
//! This is the API documentation for the NVIDIA TensorRT library. It provides information on individual
//! functions, classes and methods. Use the index on the left to navigate the documentation.
//!
//! Please see the accompanying user guide and samples for higher-level information and general advice on
//! using TensorRT.
//
//! TensorRT Versioning follows Semantic Versioning Guidelines specified here: https://semver.org/
//!

//!
//! \file NvInfer.h
//!
//! This is the top-level API file for TensorRT.
//!

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{

//!
//! \enum LayerType
//!
//! \brief The type values of layer classes.
//!
//! \see ILayer::getType()
//!
enum class LayerType : int32_t
{
    kCONVOLUTION = 0,         //!< Convolution layer.
    kFULLY_CONNECTED = 1,     //!< Fully connected layer.
    kACTIVATION = 2,          //!< Activation layer.
    kPOOLING = 3,             //!< Pooling layer.
    kLRN = 4,                 //!< LRN layer.
    kSCALE = 5,               //!< Scale layer.
    kSOFTMAX = 6,             //!< SoftMax layer.
    kDECONVOLUTION = 7,       //!< Deconvolution layer.
    kCONCATENATION = 8,       //!< Concatenation layer.
    kELEMENTWISE = 9,         //!< Elementwise layer.
    kPLUGIN = 10,             //!< Plugin layer.
    kUNARY = 11,              //!< UnaryOp operation Layer.
    kPADDING = 12,            //!< Padding layer.
    kSHUFFLE = 13,            //!< Shuffle layer.
    kREDUCE = 14,             //!< Reduce layer.
    kTOPK = 15,               //!< TopK layer.
    kGATHER = 16,             //!< Gather layer.
    kMATRIX_MULTIPLY = 17,    //!< Matrix multiply layer.
    kRAGGED_SOFTMAX = 18,     //!< Ragged softmax layer.
    kCONSTANT = 19,           //!< Constant layer.
    kRNN_V2 = 20,             //!< RNNv2 layer.
    kIDENTITY = 21,           //!< Identity layer.
    kPLUGIN_V2 = 22,          //!< PluginV2 layer.
    kSLICE = 23,              //!< Slice layer.
    kSHAPE = 24,              //!< Shape layer.
    kPARAMETRIC_RELU = 25,    //!< Parametric ReLU layer.
    kRESIZE = 26,             //!< Resize Layer.
    kTRIP_LIMIT = 27,         //!< Loop Trip limit layer
    kRECURRENCE = 28,         //!< Loop Recurrence layer
    kITERATOR = 29,           //!< Loop Iterator layer
    kLOOP_OUTPUT = 30,        //!< Loop output layer
    kSELECT = 31,             //!< Select layer.
    kFILL = 32,               //!< Fill layer
    kQUANTIZE = 33,           //!< Quantize layer
    kDEQUANTIZE = 34,         //!< Dequantize layer
    kCONDITION = 35,          //!< Condition layer
    kCONDITIONAL_INPUT = 36,  //!< Conditional Input layer
    kCONDITIONAL_OUTPUT = 37, //!< Conditional Output layer
    kSCATTER = 38,            //!< Scatter layer
    kEINSUM = 39,             //!< Einsum layer
    kASSERTION = 40,          //!< Assertion layer
    kONE_HOT = 41,            //!< OneHot layer
    kNON_ZERO = 42,           //!< NonZero layer
    kGRID_SAMPLE = 43,        //!< Grid sample layer
    kNMS = 44,                //!< NMS layer
};

//!
//! Maximum number of elements in LayerType enum.
//!
//! \see LayerType
//!
template <>
constexpr inline int32_t EnumMax<LayerType>() noexcept
{
    return 45;
}

//!
//! \brief It is capable of representing one or more TensorFormat by binary OR
//! operations, e.g., 1U << TensorFormat::kCHW4 | 1U << TensorFormat::kCHW32.
//!
//! \see ITensor::getAllowedFormats(), ITensor::setAllowedFormats(),
//!
using TensorFormats = uint32_t;

//!
//! \enum ActivationType
//!
//! \brief Enumerates the types of activation to perform in an activation layer.
//!
enum class ActivationType : int32_t
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

namespace impl
{
//!
//! Maximum number of elements in ActivationType enum.
//!
//! \see ActivationType
//!
template <>
struct EnumMaxImpl<ActivationType>
{
    static constexpr int32_t kVALUE = 12;
};
} // namespace impl

//!
//! \class ITensor
//!
//! \brief A tensor in a network definition.
//!
//! To remove a tensor from a network definition, use INetworkDefinition::removeTensor().
//!
//! When using the DLA, the cumulative size of all Tensors that are not marked as Network Input or Output tensors,
//! must be less than 1GB in size to fit into a single subgraph. If the build option kGPU_FALLBACK is specified, then
//! multiple subgraphs can be created, with each subgraph limited to less than 1GB of internal tensors data.
//!
//! \warning The volume of the tensor must be less than 2^31 elements. If the tensor is a shape tensor,
//! its volume must not exceed 64.
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and
//! ABI.
//!
class ITensor : public INoCopy
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
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        mImpl->setName(name);
    }

    //!
    //! \brief Get the tensor name.
    //!
    //! \return The name as a null-terminated C-style string.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Set the dimensions of a tensor.
    //!
    //! For a network input, the dimensions are assigned by the application. For a network output, the dimensions are
    //! computed based on the layer parameters and the inputs to the layer. If a tensor size or a parameter is modified
    //! in the network, the dimensions of all dependent tensors will be recomputed.
    //!
    //! This call is only legal for network input tensors, since the dimensions of layer output tensors are inferred
    //! based on layer inputs and parameters. The volume must be less than 2^31 elements.
    //!
    //! \param dimensions The dimensions of the tensor.
    //!
    //! \see getDimensions()
    //!
    void setDimensions(Dims dimensions) noexcept
    {
        mImpl->setDimensions(dimensions);
    }

    //!
    //! \brief Get the dimensions of a tensor.
    //!
    //! \return The dimensions of the tensor.
    //!
    //! \warning getDimensions() returns a -1 for dimensions that are derived from a wildcard dimension.
    //! \see setDimensions()
    //!
    Dims getDimensions() const noexcept
    {
        return mImpl->getDimensions();
    }

    //!
    //! \brief Set the data type of a tensor.
    //!
    //! \param type The data type of the tensor.
    //!
    //! The type is unchanged if the tensor is not a network input tensor, or marked as an output tensor or shape
    //! output tensor.
    //!
    //! \see getType()
    //!
    void setType(DataType type) noexcept
    {
        mImpl->setType(type);
    }

    //!
    //! \brief Get the data type of a tensor.
    //!
    //! \return The data type of the tensor.
    //!
    //! \see setType()
    //!
    DataType getType() const noexcept
    {
        return mImpl->getType();
    }

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
    bool setDynamicRange(float min, float max) noexcept
    {
        return mImpl->setDynamicRange(min, max);
    }

    //!
    //! \brief Whether the tensor is a network input.
    //!
    bool isNetworkInput() const noexcept
    {
        return mImpl->isNetworkInput();
    }

    //!
    //! \brief Whether the tensor is a network output.
    //!
    bool isNetworkOutput() const noexcept
    {
        return mImpl->isNetworkOutput();
    }

    //!
    //! \brief Set whether to enable broadcast of tensor across the batch.
    //!
    //! When a tensor is broadcast across a batch, it has the same value for every member in the batch.
    //! Memory is only allocated once for the single member.
    //!
    //! This method is only valid for network input tensors, since the flags of layer output tensors are inferred based
    //! on layer inputs and parameters.
    //! If this state is modified for a tensor in the network, the states of all dependent tensors will be recomputed.
    //! If the tensor is for an explicit batch network, then this function does nothing.
    //!
    //! \warning The broadcast flag is ignored when using explicit batch network mode.
    //!
    //! \param broadcastAcrossBatch Whether to enable broadcast of tensor across the batch.
    //!
    //! \see getBroadcastAcrossBatch()
    //!
    void setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept
    {
        mImpl->setBroadcastAcrossBatch(broadcastAcrossBatch);
    }

    //!
    //! \brief Check if tensor is broadcast across the batch.
    //!
    //! When a tensor is broadcast across a batch, it has the same value for every member in the batch.
    //! Memory is only allocated once for the single member. If the network is in explicit batch mode,
    //! this function returns true if the leading dimension is 1.
    //!
    //! \return True if tensor is broadcast across the batch, false otherwise.
    //!
    //! \see setBroadcastAcrossBatch()
    //!
    bool getBroadcastAcrossBatch() const noexcept
    {
        return mImpl->getBroadcastAcrossBatch();
    }

    //!
    //! \brief Get the storage location of a tensor.
    //! \return The location of tensor data.
    //! \see setLocation()
    //!
    TensorLocation getLocation() const noexcept
    {
        return mImpl->getLocation();
    }

    //!
    //! \brief Set the storage location of a tensor
    //! \param location the location of tensor data
    //!
    //! Only network input tensors for storing sequence lengths for RNNv2 are supported.
    //! Using host storage for layers that do not support it will generate
    //! errors at build time.
    //!
    //! \see getLocation()
    //!
    void setLocation(TensorLocation location) noexcept
    {
        mImpl->setLocation(location);
    }

    //!
    //! \brief Query whether dynamic range is set.
    //!
    //! \return True if dynamic range is set, false otherwise.
    //!
    bool dynamicRangeIsSet() const noexcept
    {
        return mImpl->dynamicRangeIsSet();
    }

    //!
    //! \brief Undo effect of setDynamicRange.
    //!
    void resetDynamicRange() noexcept
    {
        mImpl->resetDynamicRange();
    }

    //!
    //! \brief Get minimum of dynamic range.
    //!
    //! \return Minimum of dynamic range, or quiet NaN if range was not set.
    //!
    float getDynamicRangeMin() const noexcept
    {
        return mImpl->getDynamicRangeMin();
    }

    //!
    //! \brief Get maximum of dynamic range.
    //!
    //! \return Maximum of dynamic range, or quiet NaN if range was not set.
    //!
    float getDynamicRangeMax() const noexcept
    {
        return mImpl->getDynamicRangeMax();
    }

    //!
    //! \brief Set allowed formats for this tensor. By default all formats are allowed.
    //!        Shape tensors (for which isShapeTensor() returns true) may only have row major linear format.
    //!
    //! When running network on DLA and the build option kGPU_FALLBACK is not specified, if DLA format(kCHW4 with Int8,
    //! kCHW4 with FP16, kCHW16 with FP16, kCHW32 with Int8) is set, the input format is treated as native DLA format with
    //! line stride requirement. Input/output binding with these format should have correct layout during
    //! inference.
    //!
    //! \param formats A bitmask of TensorFormat values that are supported for this tensor.
    //!
    //! \see ITensor::getAllowedFormats()
    //! \see TensorFormats
    //!
    void setAllowedFormats(TensorFormats formats) noexcept
    {
        mImpl->setAllowedFormats(formats);
    }

    //!
    //! \brief Get a bitmask of TensorFormat values that the tensor supports.
    //!        For a shape tensor, only row major linear format is allowed.
    //!
    //! \return The value specified by setAllowedFormats or all possible formats.
    //!
    //! \see ITensor::setAllowedFormats()
    //!
    TensorFormats getAllowedFormats() const noexcept
    {
        return mImpl->getAllowedFormats();
    }

    //!
    //! \brief Whether the tensor is a shape tensor.
    //!
    //! A shape tensor is a tensor that is related to shape calculations.
    //! It must have type Int32, Bool, or Float, and its shape must be determinable at build time.
    //! Furthermore, it must be needed as a shape tensor, either marked as a network shape
    //! output via markOutputForShapes(), or as a layer input that is required to be a shape
    //! tensor, such as the second input to IShuffleLayer. Some layers are "polymorphic" in
    //! this respect. For example, the inputs to IElementWiseLayer must be shape tensors
    //! if the output is a shape tensor.
    //!
    //! The TensorRT Developer Guide give the formal rules for what tensors are shape tensors.
    //!
    //! The result of isShapeTensor() is reliable only when network construction is complete.
    //! For example, if a partially built network sums two tensors T1 and T2 to create
    //! tensor T3, and none are yet needed as shape tensors, isShapeTensor() returns false
    //! for all three tensors.  Setting the second input of IShuffleLayer to be T3 would
    //! cause all three tensors to be shape tensors, because IShuffleLayer requires that its
    //! second optional input be a shape tensor, and IElementWiseLayer is "polymorphic".
    //!
    //! If a tensor is a shape tensor and becomes an engine input or output,
    //! then ICudaEngine::isShapeBinding will be true for that tensor.
    //! Such a shape tensor must have type Int32.
    //!
    //! It is possible for a tensor to be both a shape tensor and an execution tensor.
    //!
    //! \return True if tensor is a shape tensor, false otherwise.
    //!
    //! \see INetworkDefinition::markOutputForShapes(), ICudaEngine::isShapeBinding()
    //!
    bool isShapeTensor() const noexcept
    {
        return mImpl->isShapeTensor();
    }

    //!
    //! \brief Whether the tensor is an execution tensor.
    //!
    //! Tensors are usually execution tensors.  The exceptions are tensors used
    //! solely for shape calculations or whose contents not needed to compute the outputs.
    //!
    //! The result of isExecutionTensor() is reliable only when network construction is complete.
    //! For example, if a partially built network has no path from a tensor to a network output,
    //! isExecutionTensor() returns false. Completing the path would cause it to become true.
    //!
    //! If a tensor is an execution tensor and becomes an engine input or output,
    //! then ICudaEngine::isExecutionBinding will be true for that tensor.
    //!
    //! A tensor with isShapeTensor() == false and isExecutionTensor() == false
    //! can still show up as an input to the engine if its dimensions are required.
    //! In that case, only its dimensions need to be set at runtime and a nullptr
    //! can be passed instead of a pointer to its contents.
    //!
    bool isExecutionTensor() const noexcept
    {
        return mImpl->isExecutionTensor();
    }

    //!
    //! \brief Name a dimension of an input tensor.
    //!
    //! Associate a runtime dimension of an input tensor with a symbolic name.
    //! Dimensions with the same non-empty name must be equal at runtime.
    //! Knowing this equality for runtime dimensions may help the TensorRT optimizer.
    //! Both runtime and build-time dimensions can be named.
    //!
    //! For example, setDimensionName(0, "n") associates the symbolic name "n" with the leading dimension.
    //!
    //! This method copies the name string.
    //! If the function is called again, with the same index, it will overwrite the previous name.
    //! If nullptr is passed as name, it will clear the name of the dimension.
    //!
    //! \param index index of the dimension
    //! \param name of the dimension, as a pointer to a null-terminated character sequence.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getDimensionName()
    //!
    void setDimensionName(int32_t index, char const* name) noexcept
    {
        mImpl->setDimensionName(index, name);
    }

    //!
    //! \brief Get the name of an input dimension.
    //!
    //! \param index index of the dimension
    //!
    //! \return The name of the input dimension, or nullptr if the dimension has no name.
    //!         The name is a pointer to a null-terminated character sequence.
    //!
    //! \see setDimensionName()
    //!
    char const* getDimensionName(int32_t index) const noexcept
    {
        return mImpl->getDimensionName(index);
    }

protected:
    apiv::VTensor* mImpl;
    virtual ~ITensor() noexcept = default;
};

//!
//! \class ILayer
//!
//! \brief Base class for all layer classes in a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ILayer : public INoCopy
{
public:
    //!
    //! \brief Return the type of a layer.
    //!
    //! \see LayerType
    //!
    LayerType getType() const noexcept
    {
        return mLayer->getType();
    }

    //!
    //! \brief Set the name of a layer.
    //!
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        mLayer->setName(name);
    }

    //!
    //! \brief Return the name of a layer.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        return mLayer->getName();
    }

    //!
    //! \brief Get the number of inputs of a layer.
    //!
    int32_t getNbInputs() const noexcept
    {
        return mLayer->getNbInputs();
    }

    //!
    //! \brief Get the layer input corresponding to the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range or the tensor is optional
    //! (\ref ISliceLayer and \ref IRNNv2Layer).
    //!
    ITensor* getInput(int32_t index) const noexcept
    {
        return mLayer->getInput(index);
    }

    //!
    //! \brief Get the number of outputs of a layer.
    //!
    int32_t getNbOutputs() const noexcept
    {
        return mLayer->getNbOutputs();
    }

    //!
    //! \brief Get the layer output corresponding to the given index.
    //!
    //! \return The indexed output tensor, or nullptr if the index is out of range or the tensor is optional
    //! (\ref IRNNv2Layer).
    //!
    ITensor* getOutput(int32_t index) const noexcept
    {
        return mLayer->getOutput(index);
    }

    //!
    //! \brief Replace an input of this layer with a specific tensor.
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Except for IFillLayer, ILoopOutputLayer, INMSLayer, IResizeLayer, IShuffleLayer, and ISliceLayer,
    //! this method cannot change the number of inputs to a layer. The index argument must be
    //! less than the value of getNbInputs().
    //!
    //! See comments for overloads of setInput() for layers with special behavior.
    //!
    void setInput(int32_t index, ITensor& tensor) noexcept
    {
        return mLayer->setInput(index, tensor);
    }

    //!
    //! \brief Set the computational precision of this layer
    //!
    //! Setting the precision allows TensorRT to choose an implementation which run at this computational precision.
    //! TensorRT could still choose a non-conforming fastest implementation that ignores the requested precision.
    //! To force choosing an implementation with the requested precision, set exactly one of the following flags,
    //! which differ in what happens if no such implementation exists:
    //!
    //! * BuilderFlag::kOBEY_PRECISION_CONSTRAINTS - build fails with an error message.
    //!
    //! * BuilderFlag::kPREFER_PRECISION_CONSTRAINTS - TensorRT falls back to an
    //!   implementation without the requested precision.
    //!
    //! If precision is not set, or falling back, TensorRT will select the layer computational precision
    //! and layer input type based on global performance considerations and the flags specified to the builder.
    //!
    //! For a IIdentityLayer: If it casts to/from float/half/int8/uint8, the precision must be one of those types,
    //! otherwise it must be either the input or output type.
    //!
    //! \param dataType the computational precision.
    //!
    //! \see getPrecision() precisionIsSet() resetPrecision()
    //!
    void setPrecision(DataType dataType) noexcept
    {
        mLayer->setPrecision(dataType);
    }

    //!
    //! \brief get the computational precision of this layer
    //!
    //! \return the computational precision
    //!
    //! \see setPrecision() precisionIsSet() resetPrecision()
    //!
    DataType getPrecision() const noexcept
    {
        return mLayer->getPrecision();
    }

    //!
    //! \brief whether the computational precision has been set for this layer
    //!
    //! \return whether the computational precision has been explicitly set
    //!
    //! \see setPrecision() getPrecision() resetPrecision()
    //!
    bool precisionIsSet() const noexcept
    {
        return mLayer->precisionIsSet();
    }

    //!
    //! \brief reset the computational precision for this layer
    //!
    //! \see setPrecision() getPrecision() precisionIsSet()
    //!
    void resetPrecision() noexcept
    {
        mLayer->resetPrecision();
    }

    //!
    //! \brief Set the output type of this layer
    //!
    //! Setting the output type constrains TensorRT to choose implementations which generate output data with the
    //! given type. If it is not set, TensorRT will select output type based on layer computational precision. TensorRT
    //! could still choose non-conforming output type based on fastest implementation. To force choosing the requested
    //! output type, set exactly one of the following flags, which differ in what happens if no such implementation exists:
    //!
    //! * BuilderFlag::kOBEY_PRECISION_CONSTRAINTS - build fails with an error message.
    //!
    //! * BuilderFlag::kPREFER_PRECISION_CONSTRAINTS - TensorRT falls back to an
    //!   implementation with a non-conforming output type.
    //!
    //! In case layer precision is not specified, or falling back, the output type depends on the
    //! chosen implementation, based on performance considerations and the flags specified to the builder.
    //!
    //! This method cannot be used to set the data type of the second output tensor of the TopK layer. The data type of
    //! the second output tensor of the topK layer is always Int32. Also the output type of all layers that are shape
    //! operations must be DataType::kINT32, and all attempts to set the output type to some other data type will be
    //! ignored except for issuing an error message.
    //!
    //! Note that the layer output type is generally not identical to the data type of the output tensor, as TensorRT
    //! may insert implicit reformatting operations to convert the former to the latter. Calling layer->setOutputType(i,
    //! type) has no effect on the data type of the i-th output tensor of layer, and users need to call
    //! layer->getOutput(i)->setType(type) to change the tensor data type. This is particularly relevant if the tensor
    //! is marked as a network output, since only setType() [but not setOutputType()] will affect the data
    //! representation in the corresponding output binding.
    //!
    //! \param index the index of the output to set
    //! \param dataType the type of the output
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()
    //!
    void setOutputType(int32_t index, DataType dataType) noexcept
    {
        mLayer->setOutputType(index, dataType);
    }

    //!
    //! \brief get the output type of this layer
    //!
    //! \param index the index of the output
    //! \return the output precision. If no precision has been set, DataType::kFLOAT will be returned,
    //!         unless the output type is inherently DataType::kINT32.
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()
    //!
    DataType getOutputType(int32_t index) const noexcept
    {
        return mLayer->getOutputType(index);
    }

    //!
    //! \brief whether the output type has been set for this layer
    //!
    //! \param index the index of the output
    //! \return whether the output type has been explicitly set
    //!
    //! \see setOutputType() getOutputType() resetOutputType()
    //!
    bool outputTypeIsSet(int32_t index) const noexcept
    {
        return mLayer->outputTypeIsSet(index);
    }

    //!
    //! \brief reset the output type for this layer
    //!
    //! \param index the index of the output
    //!
    //! \see setOutputType() getOutputType() outputTypeIsSet()
    //!
    void resetOutputType(int32_t index) noexcept
    {
        return mLayer->resetOutputType(index);
    }

protected:
    virtual ~ILayer() noexcept = default;
    apiv::VLayer* mLayer;
};

//!
//! \enum PaddingMode
//!
//! \brief Enumerates the modes of padding to perform in convolution, deconvolution and pooling layer,
//! padding mode takes precedence if setPaddingMode() and setPrePadding() are also used.
//!
//! There are three padding styles, EXPLICIT, SAME, and CAFFE, with each style having two variants.
//! The EXPLICIT and CAFFE styles determine if the final sampling location is used or not.
//! The SAME style determine if the asymmetry in the padding is on the pre or post padding.
//!
//! \code
//! Shorthand:
//!     I = dimensions of input image.
//!     B = prePadding, before the image data. For deconvolution, prePadding is set before output.
//!     A = postPadding, after the image data. For deconvolution, postPadding is set after output.
//!     P = delta between input and output
//!     S = stride
//!     F = filter
//!     O = output
//!     D = dilation
//!     M = I + B + A ; The image data plus any padding
//!     DK = 1 + D * (F - 1)
//! \endcode
//!
//! Formulas for Convolution:
//!     - EXPLICIT_ROUND_DOWN:
//! \code
//!         O = floor((M - DK) / S) + 1
//! \endcode
//!     - CAFFE_ROUND_DOWN:
//! \code
//!         O = floor((I + B * 2 - DK) / S) + 1
//! \endcode
//!     - EXPLICIT_ROUND_UP:
//! \code
//!         O = ceil((M - DK) / S) + 1
//! \endcode
//!     - CAFFE_ROUND_UP:
//! \code
//!         O = ceil((I + B * 2 - DK) / S) + 1
//! \endcode
//!     - SAME_UPPER:
//! \code
//!         O = ceil(I / S)
//!         P = floor((I - 1) / S) * S + DK - I;
//!         B = floor(P / 2)
//!         A = P - B
//! \endcode
//!     - SAME_LOWER:
//! \code
//!         O = ceil(I / S)
//!         P = floor((I - 1) / S) * S + DK - I;
//!         A = floor(P / 2)
//!         B = P - A
//! \endcode
//!
//! Formulas for Deconvolution:
//!     - EXPLICIT_ROUND_DOWN:
//!     - CAFFE_ROUND_DOWN:
//!     - EXPLICIT_ROUND_UP:
//!     - CAFFE_ROUND_UP:
//! \code
//!         O = (I - 1) * S + DK - (B + A)
//! \endcode
//!     - SAME_UPPER:
//! \code
//!         O = min(I * S, (I - 1) * S + DK)
//!         P = max(DK - S, 0)
//!         B = floor(P / 2)
//!         A = P - B
//! \endcode
//!     - SAME_LOWER:
//! \code
//!         O = min(I * S, (I - 1) * S + DK)
//!         P = max(DK - S, 0)
//!         A = floor(P / 2)
//!         B = P - A
//! \endcode
//!
//! Formulas for Pooling:
//!     - EXPLICIT_ROUND_DOWN:
//! \code
//!         O = floor((M - F) / S) + 1
//! \endcode
//!     - EXPLICIT_ROUND_UP:
//! \code
//!         O = ceil((M - F) / S) + 1
//! \endcode
//!     - SAME_UPPER:
//! \code
//!         O = ceil(I / S)
//!         P = floor((I - 1) / S) * S + F - I;
//!         B = floor(P / 2)
//!         A = P - B
//! \endcode
//!     - SAME_LOWER:
//! \code
//!         O = ceil(I / S)
//!         P = floor((I - 1) / S) * S + F - I;
//!         A = floor(P / 2)
//!         B = P - A
//! \endcode
//!     - CAFFE_ROUND_DOWN:
//! \code
//!         EXPLICIT_ROUND_DOWN - ((EXPLICIT_ROUND_DOWN - 1) * S >= I + B)
//! \endcode
//!     - CAFFE_ROUND_UP:
//! \code
//!         EXPLICIT_ROUND_UP - ((EXPLICIT_ROUND_UP - 1) * S >= I + B)
//! \endcode
//!
//! Pooling Example 1:
//! \code
//!     Given I = {6, 6}, B = {3, 3}, A = {2, 2}, S = {2, 2}, F = {3, 3}. What is O?
//!     (B, A can be calculated for SAME_UPPER and SAME_LOWER mode)
//! \endcode
//!
//! - EXPLICIT_ROUND_DOWN:
//! \code
//!     Computation:
//!         M = {6, 6} + {3, 3} + {2, 2} ==> {11, 11}
//!         O ==> floor((M - F) / S) + 1
//!           ==> floor(({11, 11} - {3, 3}) / {2, 2}) + {1, 1}
//!           ==> floor({8, 8} / {2, 2}) + {1, 1}
//!           ==> {5, 5}
//! \endcode
//! - EXPLICIT_ROUND_UP:
//! \code
//!     Computation:
//!         M = {6, 6} + {3, 3} + {2, 2} ==> {11, 11}
//!         O ==> ceil((M - F) / S) + 1
//!           ==> ceil(({11, 11} - {3, 3}) / {2, 2}) + {1, 1}
//!           ==> ceil({8, 8} / {2, 2}) + {1, 1}
//!           ==> {5, 5}
//! \endcode
//!     The sample points are {0, 2, 4, 6, 8} in each dimension.
//!
//! - SAME_UPPER:
//! \code
//!     Computation:
//!         I = {6, 6}
//!         S = {2, 2}
//!         O = ceil(I / S) = {3, 3}
//!         P = floor((I - 1) / S) * S + F - I
//!             ==> floor(({6, 6} - {1, 1}) / {2, 2}) * {2, 2} + {3, 3} - {6, 6}
//!             ==> {4, 4} + {3, 3} - {6, 6}
//!             ==> {1, 1}
//!         B = floor({1, 1} / {2, 2})
//!             ==> {0, 0}
//!         A = {1, 1} - {0, 0}
//!             ==> {1, 1}
//! \endcode
//! - SAME_LOWER:
//! \code
//!     Computation:
//!         I = {6, 6}
//!         S = {2, 2}
//!         O = ceil(I / S) = {3, 3}
//!         P = floor((I - 1) / S) * S + F - I
//!           ==> {1, 1}
//!         A = floor({1, 1} / {2, 2})
//!           ==> {0, 0}
//!         B = {1, 1} - {0, 0}
//!           ==> {1, 1}
//! \endcode
//!     The sample pointers are {0, 2, 4} in each dimension.
//!     SAMPLE_UPPER has {O0, O1, O2, pad} in output in each dimension.
//!     SAMPLE_LOWER has {pad, O0, O1, O2} in output in each dimension.
//!
//! Pooling Example 2:
//! \code
//!     Given I = {6, 6}, B = {3, 3}, A = {3, 3}, S = {2, 2}, F = {3, 3}. What is O?
//! \endcode
//!
//! - CAFFE_ROUND_DOWN:
//! \code
//!     Computation:
//!         M = {6, 6} + {3, 3} + {3, 3} ==> {12, 12}
//!         EXPLICIT_ROUND_DOWN ==> floor((M - F) / S) + 1
//!                             ==> floor(({12, 12} - {3, 3}) / {2, 2}) + {1, 1}
//!                             ==> {5, 5}
//!         DIFF = (((EXPLICIT_ROUND_DOWN - 1) * S >= I + B) ? {1, 1} : {0, 0})
//!           ==> ({5, 5} - {1, 1}) * {2, 2} >= {6, 6} + {3, 3} ? {1, 1} : {0,0}
//!           ==> {0, 0}
//!         O ==> EXPLICIT_ROUND_DOWN - DIFF
//!           ==> {5, 5} - {0, 0}
//!           ==> {5, 5}
//! \endcode
//! - CAFFE_ROUND_UP:
//! \code
//!     Computation:
//!         M = {6, 6} + {3, 3} + {3, 3} ==> {12, 12}
//!         EXPLICIT_ROUND_UP ==> ceil((M - F) / S) + 1
//!                           ==> ceil(({12, 12} - {3, 3}) / {2, 2}) + {1, 1}
//!                           ==> {6, 6}
//!         DIFF = (((EXPLICIT_ROUND_UP - 1) * S >= I + B) ? {1, 1} : {0, 0})
//!           ==> ({6, 6} - {1, 1}) * {2, 2} >= {6, 6} + {3, 3} ? {1, 1} : {0,0}
//!           ==> {1, 1}
//!         O ==> EXPLICIT_ROUND_UP - DIFF
//!           ==> {6, 6} - {1, 1}
//!           ==> {5, 5}
//! \endcode
//!
//! The sample points are {0, 2, 4, 6, 8} in each dimension. <br>
//! CAFFE_ROUND_DOWN and CAFFE_ROUND_UP have two restrictions each on usage with pooling operations.
//! This will cause getDimensions to return an empty dimension and also to reject the network
//! at validation time. <br>
//! For more information on original reference code, see
//! https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp
//!
//! - Restriction 1:
//! \code
//!     CAFFE_ROUND_DOWN: B >= F is an error if (B - S) < F
//!     CAFFE_ROUND_UP: (B + S) >= (F + 1) is an error if B < (F + 1)
//! \endcode
//!
//! - Restriction 2:
//! \code
//!     CAFFE_ROUND_DOWN: (B - S) >= F is an error if B >= F
//!     CAFFE_ROUND_UP: B >= (F + 1) is an error if (B + S) >= (F + 1)
//! \endcode
//!
enum class PaddingMode : int32_t
{
    kEXPLICIT_ROUND_DOWN = 0, //!< Use explicit padding, rounding output size down.
    kEXPLICIT_ROUND_UP = 1,   //!< Use explicit padding, rounding output size up.
    kSAME_UPPER = 2,          //!< Use SAME padding, with prePadding <= postPadding.
    kSAME_LOWER = 3,          //!< Use SAME padding, with prePadding >= postPadding.
    kCAFFE_ROUND_DOWN = 4,    //!< Use CAFFE padding, rounding output size down, uses prePadding value.
    kCAFFE_ROUND_UP = 5       //!< Use CAFFE padding, rounding output size up, uses prePadding value.
};

namespace impl
{
//!
//! Maximum number of elements in PaddingMode enum.
//!
//! \see PaddingMode
//!
template <>
struct EnumMaxImpl<PaddingMode>
{
    static constexpr int32_t kVALUE = 6;
};
} // namespace impl

//!
//! \class IConvolutionLayer
//!
//! \brief A convolution layer in a network definition.
//!
//! This layer performs a correlation operation between 3-dimensional filter with a 4-dimensional tensor to produce
//! another 4-dimensional tensor.
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
    //! If executing this layer on DLA, both height and width of kernel size must be in the range [1,32].
    //!
    //! \see getKernelSize()
    //!
    //! \deprecated Superseded by setKernelSizeNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setKernelSize(DimsHW kernelSize) noexcept
    {
        mImpl->setKernelSize(kernelSize);
    }

    //!
    //! \brief Get the HW kernel size of the convolution.
    //!
    //! \see setKernelSize()
    //!
    //! \deprecated Superseded by getKernelSizeNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getKernelSize() const noexcept
    {
        return mImpl->getKernelSize();
    }

    //!
    //! \brief Set the number of output maps for the convolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    void setNbOutputMaps(int32_t nbOutputMaps) noexcept
    {
        mImpl->setNbOutputMaps(nbOutputMaps);
    }

    //!
    //! \brief Get the number of output maps for the convolution.
    //!
    //! \see setNbOutputMaps()
    //!
    int32_t getNbOutputMaps() const noexcept
    {
        return mImpl->getNbOutputMaps();
    }

    //!
    //! \brief Get the stride of the convolution.
    //!
    //! Default: (1,1)
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,8].
    //!
    //! \see getStride()
    //!
    //! \deprecated Superseded by setStrideNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setStride(DimsHW stride) noexcept
    {
        mImpl->setStride(stride);
    }

    //!
    //! \brief Get the stride of the convolution.
    //!
    //! \deprecated Superseded by getStrideNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getStride() const noexcept
    {
        return mImpl->getStride();
    }

    //!
    //! \brief Set the padding of the convolution.
    //!
    //! The input will be zero-padded by this number of elements in the height and width directions.
    //! Padding is symmetric.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,31],
    //! and the padding size must be less than the kernel size.
    //!
    //! \see getPadding()
    //!
    //! \deprecated Superseded by setPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setPadding(DimsHW padding) noexcept
    {
        return mImpl->setPadding(padding);
    }

    //!
    //! \brief Get the padding of the convolution. If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPadding()
    //!
    //! \deprecated Superseded by getPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getPadding() const noexcept
    {
        return mImpl->getPadding();
    }

    //!
    //! \brief Set the number of groups for a convolution.
    //!
    //! The input tensor channels are  divided into \p nbGroups groups, and a convolution is executed for each group,
    //! using a filter per group. The results of the group convolutions are concatenated to form the output.
    //!
    //! \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group
    //! count) must be a multiple of 4 for both input and output.
    //!
    //! Default: 1
    //!
    //! If executing this layer on DLA, the max number of groups is 8192.
    //!
    //! \see getNbGroups()
    //!
    void setNbGroups(int32_t nbGroups) noexcept
    {
        mImpl->setNbGroups(nbGroups);
    }

    //!
    //! \brief Get the number of groups of the convolution.
    //!
    //! \see setNbGroups()
    //!
    int32_t getNbGroups() const noexcept
    {
        return mImpl->getNbGroups();
    }

    //!
    //! \brief Set the kernel weights for the convolution.
    //!
    //! The weights are specified as a contiguous array in \p GKCRS order, where \p G is the number of groups, \p K
    //! the number of output feature maps, \p C the number of input channels, and \p R and \p S are the height and
    //! width of the filter.
    //!
    //! \see getKernelWeights()
    //!
    void setKernelWeights(Weights weights) noexcept
    {
        mImpl->setKernelWeights(weights);
    }

    //!
    //! \brief Get the kernel weights of the convolution.
    //!
    //! \see setKernelWeights()
    //!
    Weights getKernelWeights() const noexcept
    {
        return mImpl->getKernelWeights();
    }

    //!
    //! \brief Set the bias weights for the convolution.
    //!
    //! Bias is optional. To omit bias, set the count value of the weights structure to zero.
    //!
    //! The bias is applied per-channel, so the number of weights (if non-zero) must be equal to the number of output
    //! feature maps.
    //!
    //! \see getBiasWeights()
    //!
    void setBiasWeights(Weights weights) noexcept
    {
        mImpl->setBiasWeights(weights);
    }

    //!
    //! \brief Get the bias weights for the convolution.
    //!
    //! \see setBiasWeights()
    //!
    Weights getBiasWeights() const noexcept
    {
        return mImpl->getBiasWeights();
    }

    //!
    //! \brief Set the dilation for a convolution.
    //!
    //! Default: (1,1)
    //!
    //! If executing this layer on DLA, both height and width must be in the range [1,32].
    //!
    //! \see getDilation()
    //!
    //! \deprecated Superseded by setDilationNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setDilation(DimsHW dilation) noexcept
    {
        return mImpl->setDilation(dilation);
    }

    //!
    //! \brief Get the dilation for a convolution.
    //!
    //! \see setDilation()
    //!
    //! \deprecated Superseded by getDilationNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getDilation() const noexcept
    {
        return mImpl->getDilation();
    }

    //!
    //! \brief Set the multi-dimension pre-padding of the convolution.
    //!
    //! The start of the input will be zero-padded by this number of elements in each dimension.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range
    //! [0,31], and the padding must be less than the kernel size.
    //!
    //! \see getPrePadding()
    //!
    void setPrePadding(Dims padding) noexcept
    {
        mImpl->setPrePadding(padding);
    }

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    Dims getPrePadding() const noexcept
    {
        return mImpl->getPrePadding();
    }

    //!
    //! \brief Set the multi-dimension post-padding of the convolution.
    //!
    //! The end of the input will be zero-padded by this number of elements in each dimension.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range
    //! [0,31], and the padding must be less than the kernel size.
    //!
    //! \see getPostPadding()
    //!
    void setPostPadding(Dims padding) noexcept
    {
        mImpl->setPostPadding(padding);
    }

    //!
    //! \brief Get the post-padding.
    //!
    //! \see setPostPadding()
    //!
    Dims getPostPadding() const noexcept
    {
        return mImpl->getPostPadding();
    }

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    //!
    void setPaddingMode(PaddingMode paddingMode) noexcept
    {
        mImpl->setPaddingMode(paddingMode);
    }

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    //!
    PaddingMode getPaddingMode() const noexcept
    {
        return mImpl->getPaddingMode();
    }

    //!
    //! \brief Set the multi-dimension kernel size of the convolution.
    //!
    //! If executing this layer on DLA, only support 2D kernel size, both height and width of kernel size must be in the
    //! range [1,32].
    //!
    //! \see getKernelSizeNd()
    //!
    void setKernelSizeNd(Dims kernelSize) noexcept
    {
        mImpl->setKernelSizeNd(kernelSize);
    }

    //!
    //! \brief Get the multi-dimension kernel size of the convolution.
    //!
    //! \see setKernelSizeNd()
    //!
    Dims getKernelSizeNd() const noexcept
    {
        return mImpl->getKernelSizeNd();
    }

    //!
    //! \brief Set the multi-dimension stride of the convolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! If executing this layer on DLA, only support 2D stride, both height and width of stride must be in the range
    //! [1,8].
    //!
    //! \see getStrideNd() setStride() getStride()
    //!
    void setStrideNd(Dims stride) noexcept
    {
        mImpl->setStrideNd(stride);
    }

    //!
    //! \brief Get the multi-dimension stride of the convolution.
    //!
    //! \see setStrideNd()
    //!
    Dims getStrideNd() const noexcept
    {
        return mImpl->getStrideNd();
    }

    //!
    //! \brief Set the multi-dimension padding of the convolution.
    //!
    //! The input will be zero-padded by this number of elements in each dimension.
    //! Padding is symmetric.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range
    //! [0,31], and the padding must be less than the kernel size.
    //!
    //! \see getPaddingNd() setPadding() getPadding()
    //!
    void setPaddingNd(Dims padding) noexcept
    {
        mImpl->setPaddingNd(padding);
    }

    //!
    //! \brief Get the multi-dimension padding of the convolution.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    Dims getPaddingNd() const noexcept
    {
        return mImpl->getPaddingNd();
    }

    //!
    //! \brief Set the multi-dimension dilation of the convolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width must be in the range [1,32].
    //!
    //! \see getDilation()
    //!
    void setDilationNd(Dims dilation) noexcept
    {
        mImpl->setDilationNd(dilation);
    }

    //!
    //! \brief Get the multi-dimension dilation of the convolution.
    //!
    //! \see setDilation()
    //!
    Dims getDilationNd() const noexcept
    {
        return mImpl->getDilationNd();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! The indices are as follows:
    //!
    //! Input 0 is the input activation tensor.
    //! Input 1 is the kernel tensor. If used, the kernel weights parameter must be set to empty weights.
    //! Input 2 is the bias tensor. If used, the bias parameter must be set to empty weights.
    //! \see getKernelWeights(), setKernelWeights(), getBiasWeights(), setBiasWeights()
    //!
    using ILayer::setInput;

protected:
    virtual ~IConvolutionLayer() noexcept = default;
    apiv::VConvolutionLayer* mImpl;
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
//! \deprecated Deprecated in TensorRT 8.4. Superseded by IMatrixMultiplyLayer.
//!
class TRT_DEPRECATED IFullyConnectedLayer : public ILayer
{
public:
    //!
    //! \brief Set the number of output channels `K` from the fully connected layer.
    //!
    //! If executing this layer on DLA, number of output channels must in the range [1,8192].
    //!
    //! \see getNbOutputChannels()
    //!
    void setNbOutputChannels(int32_t nbOutputs) noexcept
    {
        mImpl->setNbOutputChannels(nbOutputs);
    }

    //!
    //! \brief Get the number of output channels `K` from the fully connected layer.
    //!
    //! \see setNbOutputChannels()
    //!
    int32_t getNbOutputChannels() const noexcept
    {
        return mImpl->getNbOutputChannels();
    }

    //!
    //! \brief Set the kernel weights, given as a `KxC` matrix in row-major order.
    //!
    //! \see getKernelWeights()
    //!
    void setKernelWeights(Weights weights) noexcept
    {
        mImpl->setKernelWeights(weights);
    }

    //!
    //! \brief Get the kernel weights.
    //!
    //! \see setKernelWeights()
    //!
    Weights getKernelWeights() const noexcept
    {
        return mImpl->getKernelWeights();
    }

    //!
    //! \brief Set the bias weights.
    //!
    //! Bias is optional. To omit bias, set the count value in the weights structure to zero.
    //!
    //! \see getBiasWeightsWeights()
    //!
    void setBiasWeights(Weights weights) noexcept
    {
        mImpl->setBiasWeights(weights);
    }

    //!
    //! \brief Get the bias weights.
    //!
    //! \see setBiasWeightsWeights()
    //!
    Weights getBiasWeights() const noexcept
    {
        return mImpl->getBiasWeights();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Only index 0 (data input) is valid, unless explicit-quantization mode is enabled.
    //! In explicit-quantization mode, input with index 1 is the kernel-weights tensor, if present.
    //! The kernel-weights tensor must be a build-time constant (computable at build-time via constant-folding)
    //! and an output of a dequantize layer.
    //! If input index 1 is used then the kernel-weights parameter must be set to empty Weights.
    //!
    //! \see getKernelWeights(), setKernelWeights()
    //!
    //! The indices are as follows:
    //!
    //! - 0: The input activation tensor.
    //! - 1: The kernel weights tensor (a constant tensor).
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    using ILayer::setInput;

protected:
    virtual ~IFullyConnectedLayer() noexcept = default;
    apiv::VFullyConnectedLayer* mImpl;
};

//!
//! \class IActivationLayer
//!
//! \brief An Activation layer in a network definition.
//!
//! This layer applies a per-element activation function to its input.
//!
//! The output has the same shape as the input.
//!
//! The input is a shape tensor if the output is a shape tensor.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IActivationLayer : public ILayer
{
public:
    //!
    //! \brief Set the type of activation to be performed.
    //!
    //! On the DLA, the valid activation types are kRELU, kSIGMOID, kTANH, and kCLIP.
    //!
    //! \see getActivationType(), ActivationType
    //!
    void setActivationType(ActivationType type) noexcept
    {
        mImpl->setActivationType(type);
    }

    //!
    //! \brief Get the type of activation to be performed.
    //!
    //! \see setActivationType(), ActivationType
    //!
    ActivationType getActivationType() const noexcept
    {
        return mImpl->getActivationType();
    }

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
    void setAlpha(float alpha) noexcept
    {
        mImpl->setAlpha(alpha);
    }

    //!
    //! \brief Set the beta parameter (must be finite).
    //!
    //! This parameter is used by the following activations:
    //! Selu, Softplus, Clip, HardSigmoid, ScaledTanh.
    //!
    //! It is ignored by the other activations.
    //!
    //! \see getBeta(), setAlpha()
    void setBeta(float beta) noexcept
    {
        mImpl->setBeta(beta);
    }

    //!
    //! \brief Get the alpha parameter.
    //!
    //! \see getBeta(), setAlpha()
    float getAlpha() const noexcept
    {
        return mImpl->getAlpha();
    }

    //!
    //! \brief Get the beta parameter.
    //!
    //! \see getAlpha(), setBeta()
    float getBeta() const noexcept
    {
        return mImpl->getBeta();
    }

protected:
    virtual ~IActivationLayer() noexcept = default;
    apiv::VActivationLayer* mImpl;
};

//!
//! \enum PoolingType
//!
//! \brief The type of pooling to perform in a pooling layer.
//!
enum class PoolingType : int32_t
{
    kMAX = 0,              // Maximum over elements
    kAVERAGE = 1,          // Average over elements. If the tensor is padded, the count includes the padding
    kMAX_AVERAGE_BLEND = 2 // Blending between max and average pooling: (1-blendFactor)*maxPool + blendFactor*avgPool
};

namespace impl
{
//!
//! Maximum number of elements in PoolingType enum.
//!
//! \see PoolingType
//!
template <>
struct EnumMaxImpl<PoolingType>
{
    static constexpr int32_t kVALUE = 3;
};
} // namespace impl

//! \class IPoolingLayer
//!
//! \brief A Pooling layer in a network definition.
//!
//! The layer applies a reduction operation within a window over the input.
//!
//! \warning When running pooling layer with DeviceType::kDLA in Int8 mode, the dynamic ranges
//! for input and output tensors must be equal.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPoolingLayer : public ILayer
{
public:
    //!
    //! \brief Set the type of activation to be performed.
    //!
    //! DLA only supports kMAX and kAVERAGE pooling types.
    //!
    //! \see getPoolingType(), PoolingType
    //!
    void setPoolingType(PoolingType type) noexcept
    {
        mImpl->setPoolingType(type);
    }

    //!
    //! \brief Get the type of activation to be performed.
    //!
    //! \see setPoolingType(), PoolingType
    //!
    PoolingType getPoolingType() const noexcept
    {
        return mImpl->getPoolingType();
    }

    //!
    //! \brief Set the window size for pooling.
    //!
    //! If executing this layer on DLA, both height and width of window size must be in the range [1,8].
    //!
    //! \see getWindowSize()
    //!
    //! \deprecated Superseded by setWindowSizeNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setWindowSize(DimsHW windowSize) noexcept
    {
        mImpl->setWindowSize(windowSize);
    }

    //!
    //! \brief Get the window size for pooling.
    //!
    //! \see setWindowSize()
    //!
    //! \deprecated Superseded by getWindowSizeNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getWindowSize() const noexcept
    {
        return mImpl->getWindowSize();
    }

    //!
    //! \brief Set the stride for pooling.
    //!
    //! Default: 1
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,16].
    //!
    //! \see getStride()
    //!
    //! \deprecated Superseded by setStrideNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setStride(DimsHW stride) noexcept
    {
        mImpl->setStride(stride);
    }

    //!
    //! \brief Get the stride for pooling.
    //!
    //! \see setStride()
    //!
    //! \deprecated Superseded by getStrideNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getStride() const noexcept
    {
        return mImpl->getStride();
    }

    //!
    //! \brief Set the padding for pooling.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,7].
    //!
    //! \see getPadding()
    //!
    //! \deprecated Superseded by setPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setPadding(DimsHW padding) noexcept
    {
        mImpl->setPadding(padding);
    }

    //!
    //! \brief Get the padding for pooling.
    //!
    //! Default: 0
    //!
    //! \see setPadding()
    //!
    //! \deprecated Superseded by getPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getPadding() const noexcept
    {
        return mImpl->getPadding();
    }

    //!
    //! \brief Set the blending factor for the max_average_blend mode:
    //! max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
    //! blendFactor is a user value in [0,1] with the default value of 0.0
    //! This value only applies for the kMAX_AVERAGE_BLEND mode.
    //!
    //! Since DLA does not support kMAX_AVERAGE_BLEND, blendFactor is ignored on the DLA.
    //!
    //! \see getBlendFactor()
    //!
    void setBlendFactor(float blendFactor) noexcept
    {
        mImpl->setBlendFactor(blendFactor);
    }

    //!
    //! \brief Get the blending factor for the max_average_blend mode:
    //! max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
    //! blendFactor is a user value in [0,1] with the default value of 0.0
    //! In modes other than kMAX_AVERAGE_BLEND, blendFactor is ignored.
    //!
    //! \see setBlendFactor()
    //!
    float getBlendFactor() const noexcept
    {
        return mImpl->getBlendFactor();
    }

    //!
    //! \brief Set whether average pooling uses as a denominator the overlap area between the window
    //! and the unpadded input.
    //! If this is not set, the denominator is the overlap between the pooling window and the padded input.
    //!
    //! Default: true
    //!
    //! \note On Xavier, DLA supports only inclusive padding and this must be explicitly
    //! set to false.
    //!
    //! \see getAverageCountExcludesPadding()
    //!
    void setAverageCountExcludesPadding(bool exclusive) noexcept
    {
        mImpl->setAverageCountExcludesPadding(exclusive);
    }

    //!
    //! \brief Get whether average pooling uses as a denominator the overlap area between the window
    //! and the unpadded input.
    //!
    //! \see setAverageCountExcludesPadding()
    //!
    bool getAverageCountExcludesPadding() const noexcept
    {
        return mImpl->getAverageCountExcludesPadding();
    }

    //!
    //! \brief Set the multi-dimension pre-padding for pooling.
    //!
    //! The start of the input will be padded by this number of elements in each dimension.
    //! Padding value depends on pooling type, -inf is used for max pooling and zero padding for average pooling.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range
    //! [0,7].
    //!
    //! \see getPrePadding()
    //!
    void setPrePadding(Dims padding) noexcept
    {
        mImpl->setPrePadding(padding);
    }

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    Dims getPrePadding() const noexcept
    {
        return mImpl->getPrePadding();
    }

    //!
    //! \brief Set the multi-dimension post-padding for pooling.
    //!
    //! The end of the input will be padded by this number of elements in each dimension.
    //! Padding value depends on pooling type, -inf is used for max pooling and zero padding for average pooling.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range
    //! [0,7].
    //!
    //! \see getPostPadding()
    //!
    void setPostPadding(Dims padding) noexcept
    {
        mImpl->setPostPadding(padding);
    }

    //!
    //! \brief Get the padding.
    //!
    //! \see setPostPadding()
    //!
    Dims getPostPadding() const noexcept
    {
        return mImpl->getPostPadding();
    }

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    void setPaddingMode(PaddingMode paddingMode) noexcept
    {
        mImpl->setPaddingMode(paddingMode);
    }

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    PaddingMode getPaddingMode() const noexcept
    {
        return mImpl->getPaddingMode();
    }

    //!
    //! \brief Set the multi-dimension window size for pooling.
    //!
    //! If executing this layer on DLA, only support 2D window size, both height and width of window size must be in the
    //! range [1,8].
    //!
    //! \see getWindowSizeNd() setWindowSize() getWindowSize()
    //!
    void setWindowSizeNd(Dims windowSize) noexcept
    {
        mImpl->setWindowSizeNd(windowSize);
    }

    //!
    //! \brief Get the multi-dimension window size for pooling.
    //!
    //! \see setWindowSizeNd()
    //!
    Dims getWindowSizeNd() const noexcept
    {
        return mImpl->getWindowSizeNd();
    }

    //!
    //! \brief Set the multi-dimension stride for pooling.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! If executing this layer on DLA, only support 2D stride, both height and width of stride must be in the range
    //! [1,16].
    //!
    //! \see getStrideNd() setStride() getStride()
    //!
    void setStrideNd(Dims stride) noexcept
    {
        mImpl->setStrideNd(stride);
    }

    //!
    //! \brief Get the multi-dimension stride for pooling.
    //!
    //! \see setStrideNd()
    //!
    Dims getStrideNd() const noexcept
    {
        return mImpl->getStrideNd();
    }

    //!
    //! \brief Set the multi-dimension padding for pooling.
    //!
    //! The input will be padded by this number of elements in each dimension.
    //! Padding is symmetric.
    //! Padding value depends on pooling type, -inf is used for max pooling and zero padding for average pooling.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range
    //! [0,7].
    //!
    //! \see getPaddingNd() setPadding() getPadding()
    //!
    void setPaddingNd(Dims padding) noexcept
    {
        mImpl->setPaddingNd(padding);
    }

    //!
    //! \brief Get the multi-dimension padding for pooling.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    Dims getPaddingNd() const noexcept
    {
        return mImpl->getPaddingNd();
    }

protected:
    virtual ~IPoolingLayer() noexcept = default;
    apiv::VPoolingLayer* mImpl;
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
    //!
    //! If executing this layer on the DLA, only values in the set, [3, 5, 7, 9], are valid.
    //!
    //! \see setWindowStride()
    //!
    void setWindowSize(int32_t windowSize) noexcept
    {
        mImpl->setWindowSize(windowSize);
    }

    //!
    //! \brief Get the LRN window size.
    //!
    //! \see getWindowStride()
    //!
    int32_t getWindowSize() const noexcept
    {
        return mImpl->getWindowSize();
    }

    //!
    //! \brief Set the LRN alpha value.
    //!
    //! The valid range is [-1e20, 1e20].
    //! \see getAlpha()
    //!
    void setAlpha(float alpha) noexcept
    {
        mImpl->setAlpha(alpha);
    }

    //!
    //! \brief Get the LRN alpha value.
    //!
    //! \see setAlpha()
    //!
    float getAlpha() const noexcept
    {
        return mImpl->getAlpha();
    }

    //!
    //! \brief Set the LRN beta value.
    //!
    //! The valid range is [0.01, 1e5f].
    //! \see getBeta()
    //!
    void setBeta(float beta) noexcept
    {
        mImpl->setBeta(beta);
    }

    //!
    //! \brief Get the LRN beta value.
    //!
    //! \see setBeta()
    //!
    float getBeta() const noexcept
    {
        return mImpl->getBeta();
    }

    //!
    //! \brief Set the LRN K value.
    //!
    //! The valid range is [1e-5, 1e10].
    //! \see getK()
    //!
    void setK(float k) noexcept
    {
        mImpl->setK(k);
    }

    //!
    //! \brief Get the LRN K value.
    //!
    //! \see setK()
    //!
    float getK() const noexcept
    {
        return mImpl->getK();
    }

protected:
    virtual ~ILRNLayer() noexcept = default;
    apiv::VLRNLayer* mImpl;
};

//!
//! \brief Controls how shift, scale and power are applied in a Scale layer.
//!
//! \see IScaleLayer
//!
enum class ScaleMode : int32_t
{
    kUNIFORM = 0,    //!< Identical coefficients across all elements of the tensor.
    kCHANNEL = 1,    //!< Per-channel coefficients.
    kELEMENTWISE = 2 //!< Elementwise coefficients.
};

//!
//! Maximum number of elements in ScaleMode enum.
//!
//! \see ScaleMode
//!
template <>
constexpr inline int32_t EnumMax<ScaleMode>() noexcept
{
    return 3;
}

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
//! \note The input tensor for this layer is required to have a minimum of 3 dimensions in implicit batch mode
//!       and a minimum of 4 dimensions in explicit batch mode.
//!
//! A scale layer may be used as an INT8 quantization node in a graph, if the output is constrained to INT8 and
//! the input to FP32. Quantization rounds ties to even, and clamps to [-128, 127].
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
    void setMode(ScaleMode mode) noexcept
    {
        mImpl->setMode(mode);
    }

    //!
    //! \brief Get the scale mode.
    //!
    //! \see setMode()
    //!
    ScaleMode getMode() const noexcept
    {
        return mImpl->getMode();
    }

    //!
    //! \brief Set the shift value.
    //!
    //! \see getShift()
    //!
    void setShift(Weights shift) noexcept
    {
        mImpl->setShift(shift);
    }

    //!
    //! \brief Get the shift value.
    //!
    //! \see setShift()
    //!
    Weights getShift() const noexcept
    {
        return mImpl->getShift();
    }

    //!
    //! \brief Set the scale value.
    //!
    //! \see getScale()
    //!
    void setScale(Weights scale) noexcept
    {
        mImpl->setScale(scale);
    }

    //!
    //! \brief Get the scale value.
    //!
    //! \see setScale()
    //!
    Weights getScale() const noexcept
    {
        return mImpl->getScale();
    }

    //!
    //! \brief Set the power value.
    //!
    //! \see getPower()
    //!
    void setPower(Weights power) noexcept
    {
        mImpl->setPower(power);
    }

    //!
    //! \brief Get the power value.
    //!
    //! \see setPower()
    //!
    Weights getPower() const noexcept
    {
        return mImpl->getPower();
    }

    //!
    //! \brief Get the channel axis.
    //!
    //! \return channelAxis parameter passed to addScaleNd() or set by setChannelAxis()
    //!
    //! The value is the index of the channel axis in the input tensor's dimensions.
    //! Scaling happens along the channel axis when ScaleMode::kCHANNEL is enabled.
    //!
    //! \see addScaleNd()
    //!
    int32_t getChannelAxis() const noexcept
    {
        return mImpl->getChannelAxis();
    }

    //!
    //! \brief Set the channel axis.
    //!
    //! The value is the index of the channel axis in the input tensor's dimensions.
    //!
    //! For ScaleMode::kCHANNEL, there can be distinct scale, shift, and power weights for each channel coordinate.
    //! For ScaleMode::kELEMENTWISE, there can be distinct scale, shift, and power weights for each combination of
    //! coordinates from the channel axis and axes after it.
    //!
    //! For example, suppose the input tensor has dimensions [10,20,30,40] and the channel axis is 1.
    //! Let [n,c,h,w] denote an input coordinate.
    //! For ScaleMode::kCHANNEL, the scale, shift, and power weights are indexed by c.
    //! For ScaleMode::kELEMENTWISE, the scale, shift, and power weights are indexed by [c,h,w].
    //!
    //! \see addScaleNd()
    //!
    void setChannelAxis(int32_t channelAxis) noexcept
    {
        mImpl->setChannelAxis(channelAxis);
    }

protected:
    virtual ~IScaleLayer() noexcept = default;
    apiv::VScaleLayer* mImpl;
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
//! On Xavier, this layer is not supported on DLA.
//! Otherwise, the following constraints must be satisfied to execute this layer on DLA:
//! * Axis must be one of the channel or spatial dimensions.
//! * There are two classes of supported input sizes:
//!     1. Non-axis, non-batch dimensions are all 1 and the axis dimension is at most 8192.
//!        This is the recommended case for using softmax since it is the most accurate.
//!     2. At least one non-axis, non-batch dimension greater than 1 and the axis dimension is at most 1024.
//!        Note that in this case, there may be some approximation error as the axis dimension size approaches
//!        the upper bound. See the TensorRT Developer Guide for more details on the approximation error.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISoftMaxLayer : public ILayer
{
public:
    //!
    //! \brief Set the axis along which softmax is computed. Currently, only one axis can be set.
    //!
    //! The axis is specified by setting the bit corresponding to the axis to 1.
    //! For example, consider an NCHW tensor as input (three non-batch dimensions).
    //!
    //! In implicit mode :
    //! Bit 0 corresponds to the C dimension boolean.
    //! Bit 1 corresponds to the H dimension boolean.
    //! Bit 2 corresponds to the W dimension boolean.
    //! By default, softmax is performed on the axis which is the number of axes minus three. It is 0 if
    //! there are fewer than 3 non-batch axes. For example, if the input is NCHW, the default axis is C. If the input
    //! is NHW, then the default axis is H.
    //!
    //! In explicit mode :
    //! Bit 0 corresponds to the N dimension boolean.
    //! Bit 1 corresponds to the C dimension boolean.
    //! Bit 2 corresponds to the H dimension boolean.
    //! Bit 3 corresponds to the W dimension boolean.
    //! By default, softmax is performed on the axis which is the number of axes minus three. It is 0 if
    //! there are fewer than 3 axes. For example, if the input is NCHW, the default axis is C. If the input
    //! is NHW, then the default axis is N.
    //!
    //! For example, to perform softmax on axis R of a NPQRCHW input, set bit 2 with implicit batch mode,
    //! set bit 3 with explicit batch mode.
    //!
    //! \param axes The axis along which softmax is computed.
    //!        Here axes is a bitmap. For example, when doing softmax along axis 0, bit 0 is set to 1, axes = 1 << axis
    //!        = 1.
    //!
    void setAxes(uint32_t axes) noexcept
    {
        mImpl->setAxes(axes);
    }

    //!
    //! \brief Get the axis along which softmax occurs.
    //!
    //! \see setAxes()
    //!
    uint32_t getAxes() const noexcept
    {
        return mImpl->getAxes();
    }

protected:
    virtual ~ISoftMaxLayer() noexcept = default;
    apiv::VSoftMaxLayer* mImpl;
};

//!
//! \class IConcatenationLayer
//!
//! \brief A concatenation layer in a network definition.
//!
//! The output dimension along the concatenation axis is the sum of the corresponding input dimensions.
//! Every other output dimension is the same as the corresponding dimension of the inputs.
//!
//! \warning All tensors must have the same dimensions except along the concatenation axis.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConcatenationLayer : public ILayer
{
public:
    //!
    //! \brief Set the axis along which concatenation occurs.
    //!
    //! The default axis is the number of tensor dimensions minus three, or zero if the tensor has fewer than three
    //! dimensions. For example, for a tensor with dimensions NCHW, it is C.
    //! For implicit batch mode, the number of tensor dimensions does NOT include the implicit batch dimension.
    //!
    //! When running this layer on the DLA, the concatenation axis must be the third to last axis, e.g. C if tensor
    //! dimensions are NCHW.
    //!
    //! \param axis The axis along which concatenation occurs.
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

    //!
    //! \brief Get the axis along which concatenation occurs.
    //!
    //! \see setAxis()
    //!
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }

protected:
    virtual ~IConcatenationLayer() noexcept = default;
    apiv::VConcatenationLayer* mImpl;
};

//!
//! \class IDeconvolutionLayer
//!
//! \brief A deconvolution layer in a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IDeconvolutionLayer : public ILayer
{
public:
    //!
    //! \brief Set the HW kernel size of the convolution.
    //!
    //! If executing this layer on DLA, both height and width of kernel size must be in the range [1,32], or the
    //! combinations of [64, 96, 128] in one dimension and 1 in the other dimensions, i.e. [1x64] or [64x1] are valid,
    //! but not [64x64].
    //!
    //! \see getKernelSize()
    //!
    //! \deprecated Superseded by setKernelSizeNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setKernelSize(DimsHW kernelSize) noexcept
    {
        mImpl->setKernelSize(kernelSize);
    }

    //!
    //! \brief Get the HW kernel size of the deconvolution.
    //!
    //! \see setKernelSize()
    //!
    //! \deprecated Superseded by getKernelSizeNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getKernelSize() const noexcept
    {
        return mImpl->getKernelSize();
    }

    //!
    //! \brief Set the number of output feature maps for the deconvolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    void setNbOutputMaps(int32_t nbOutputMaps) noexcept
    {
        mImpl->setNbOutputMaps(nbOutputMaps);
    }

    //!
    //! \brief Get the number of output feature maps for the deconvolution.
    //!
    //! \see setNbOutputMaps()
    //!
    int32_t getNbOutputMaps() const noexcept
    {
        return mImpl->getNbOutputMaps();
    }

    //!
    //! \brief Set the stride of the deconvolution.
    //!
    //! If executing this layer on DLA, there is one restriction:
    //! 1) Stride height and width must be in the range [1,32] or the combinations of [64, 96, 128] in one
    //! dimension and 1 in the other dimensions, i.e. [1x64] or [64x1] are valid, but not [64x64].
    //!
    //! \see getStride()
    //!
    //! \deprecated Superseded by setStrideNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setStride(DimsHW stride) noexcept
    {
        mImpl->setStride(stride);
    }

    //!
    //! \brief Get the stride of the deconvolution.
    //!
    //! Default: (1,1)
    //!
    //! \deprecated Superseded by getStrideNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getStride() const noexcept
    {
        return mImpl->getStride();
    }

    //!
    //! \brief Set the padding of the deconvolution.
    //!
    //! The output will be trimmed by this number of elements on each side in the height and width directions.
    //! In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Padding is symmetric, and negative padding is not supported.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be 0.
    //!
    //! \see getPadding()
    //!
    //! \deprecated Superseded by setPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setPadding(DimsHW padding) noexcept
    {
        mImpl->setPadding(padding);
    }

    //!
    //! \brief Get the padding of the deconvolution.
    //!
    //! Default: (0, 0)
    //!
    //! \see setPadding()
    //!
    //! \deprecated Superseded by getPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getPadding() const noexcept
    {
        return mImpl->getPadding();
    }

    //!
    //! \brief Set the number of groups for a deconvolution.
    //!
    //! The input tensor channels are divided into \p nbGroups groups, and a deconvolution is executed for each group,
    //! using a filter per group. The results of the group convolutions are concatenated to form the output.
    //!
    //! If executing this layer on DLA, nbGroups must be one
    //!
    //! \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count)
    //! must be a multiple of 4 for both input and output.
    //!
    //! Default: 1
    //!
    //! \see getNbGroups()
    //!
    void setNbGroups(int32_t nbGroups) noexcept
    {
        mImpl->setNbGroups(nbGroups);
    }

    //!
    //! \brief Get the number of groups for a deconvolution.
    //!
    //! \see setNbGroups()
    //!
    int32_t getNbGroups() const noexcept
    {
        return mImpl->getNbGroups();
    }

    //!
    //! \brief Set the kernel weights for the deconvolution.
    //!
    //! The weights are specified as a contiguous array in \p CKRS order, where \p C the number of
    //! input channels, \p K the number of output feature maps, and \p R and \p S are the height and width
    //! of the filter.
    //!
    //! \see getWeights()
    //!
    void setKernelWeights(Weights weights) noexcept
    {
        mImpl->setKernelWeights(weights);
    }

    //!
    //! \brief Get the kernel weights for the deconvolution.
    //!
    //! \see setNbGroups()
    //!
    Weights getKernelWeights() const noexcept
    {
        return mImpl->getKernelWeights();
    }

    //!
    //! \brief Set the bias weights for the deconvolution.
    //!
    //! Bias is optional. To omit bias, set the count value of the weights structure to zero.
    //!
    //! The bias is applied per-feature-map, so the number of weights (if non-zero) must be equal to the number of
    //! output feature maps.
    //!
    //! \see getBiasWeights()
    //!
    void setBiasWeights(Weights weights) noexcept
    {
        mImpl->setBiasWeights(weights);
    }

    //!
    //! \brief Get the bias weights for the deconvolution.
    //!
    //! \see getBiasWeights()
    //!
    Weights getBiasWeights() const noexcept
    {
        return mImpl->getBiasWeights();
    }

    //!
    //! \brief Set the multi-dimension pre-padding of the deconvolution.
    //!
    //! The output will be trimmed by this number of elements on the start of every dimension.
    //! In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Negative padding is not supported.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, padding must be 0.
    //!
    //! \see getPrePadding()
    //!
    void setPrePadding(Dims padding) noexcept
    {
        mImpl->setPrePadding(padding);
    }

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    Dims getPrePadding() const noexcept
    {
        return mImpl->getPrePadding();
    }

    //!
    //! \brief Set the multi-dimension post-padding of the deconvolution.
    //!
    //! The output will be trimmed by this number of elements on the end of every dimension.
    //! In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Negative padding is not supported.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, padding must be 0.
    //!
    //! \see getPostPadding()
    //!
    void setPostPadding(Dims padding) noexcept
    {
        mImpl->setPostPadding(padding);
    }

    //!
    //! \brief Get the padding.
    //!
    //! \see setPostPadding()
    //!
    Dims getPostPadding() const noexcept
    {
        return mImpl->getPostPadding();
    }

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    //!
    void setPaddingMode(PaddingMode paddingMode) noexcept
    {
        mImpl->setPaddingMode(paddingMode);
    }

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    //!
    PaddingMode getPaddingMode() const noexcept
    {
        return mImpl->getPaddingMode();
    }

    //!
    //! \brief Set the multi-dimension kernel size of the deconvolution.
    //!
    //! If executing this layer on DLA, there are ttwo restrictions:
    //! 1) Only 2D Kernel is supported.
    //! 2) Kernel height and width must be in the range [1,32] or the combinations of [64, 96, 128] in one
    //! dimension and 1 in the other dimensions, i.e. [1x64] or [64x1] are valid, but not [64x64].
    //!
    //! \see getKernelSizeNd() setKernelSize() getKernelSize()
    //!
    void setKernelSizeNd(Dims kernelSize) noexcept
    {
        mImpl->setKernelSizeNd(kernelSize);
    }

    //!
    //! \brief Get the multi-dimension kernel size of the deconvolution.
    //!
    //! \see setKernelSizeNd()
    //!
    Dims getKernelSizeNd() const noexcept
    {
        return mImpl->getKernelSizeNd();
    }

    //!
    //! \brief Set the multi-dimension stride of the deconvolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! If executing this layer on DLA, there are two restrictions:
    //! 1) Only 2D Stride is supported.
    //! 2) Stride height and width must be in the range [1,32] or the combinations of [64, 96, 128] in one
    //! dimension and 1 in the other dimensions, i.e. [1x64] or [64x1] are valid, but not [64x64].
    //!
    //! \see getStrideNd() setStride() getStride()
    //!
    void setStrideNd(Dims stride) noexcept
    {
        mImpl->setStrideNd(stride);
    }

    //!
    //! \brief Get the multi-dimension stride of the deconvolution.
    //!
    //! \see setStrideNd()
    //!
    Dims getStrideNd() const noexcept
    {
        return mImpl->getStrideNd();
    }

    //!
    //! \brief Set the multi-dimension padding of the deconvolution.
    //!
    //! The output will be trimmed by this number of elements on both sides of every dimension.
    //! In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Padding is symmetric, and negative padding is not supported.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, padding must be 0.
    //!
    //! \see getPaddingNd() setPadding() getPadding()
    //!
    void setPaddingNd(Dims padding) noexcept
    {
        mImpl->setPaddingNd(padding);
    }

    //!
    //! \brief Get the multi-dimension padding of the deconvolution.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    Dims getPaddingNd() const noexcept
    {
        return mImpl->getPaddingNd();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Input 0 is the input activation tensor.
    //! Input 1 is the kernel tensor. If used, the kernel weights parameter must be set to empty weights.
    //! Input 2 is the bias tensor. If used, the bias parameter must be set to empty weights.
    //! \see getKernelWeights(), setKernelWeights(), getBiasWeights(), setBiasWeights()
    //!
    using ILayer::setInput;

    //! \brief Set the multi-dimension dilation of the deconvolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! \see getDilationNd()
    //!
    void setDilationNd(Dims dilation) noexcept
    {
        mImpl->setDilationNd(dilation);
    }

    //!
    //! \brief Get the multi-dimension dilation of the deconvolution.
    //!
    //! \see setDilationNd()
    //!
    Dims getDilationNd() const noexcept
    {
        return mImpl->getDilationNd();
    }

protected:
    virtual ~IDeconvolutionLayer() noexcept = default;
    apiv::VDeconvolutionLayer* mImpl;
};

//!
//! \enum ElementWiseOperation
//!
//! \brief Enumerates the binary operations that may be performed by an ElementWise layer.
//!
//! Operations kAND, kOR, and kXOR must have inputs of #DataType kBOOL.
//!
//! Operation kPOW must have inputs of #DataType kFLOAT, kHALF, or kINT8.
//!
//! All other operations must have inputs of #DataType kFLOAT, kHALF, kINT8, or kINT32.
//!
//! \see IElementWiseLayer
//!
enum class ElementWiseOperation : int32_t
{
    kSUM = 0,       //!< Sum of the two elements.
    kPROD = 1,      //!< Product of the two elements.
    kMAX = 2,       //!< Maximum of the two elements.
    kMIN = 3,       //!< Minimum of the two elements.
    kSUB = 4,       //!< Subtract the second element from the first.
    kDIV = 5,       //!< Divide the first element by the second.
    kPOW = 6,       //!< The first element to the power of the second element.
    kFLOOR_DIV = 7, //!< Floor division of the first element by the second.
    kAND = 8,       //!< Logical AND of two elements.
    kOR = 9,        //!< Logical OR of two elements.
    kXOR = 10,      //!< Logical XOR of two elements.
    kEQUAL = 11,    //!< Check if two elements are equal.
    kGREATER = 12,  //!< Check if element in first tensor is greater than corresponding element in second tensor.
    kLESS = 13      //!< Check if element in first tensor is less than corresponding element in second tensor.
};

namespace impl
{
//!
//! Maximum number of elements in ElementWiseOperation enum.
//!
//! \see ElementWiseOperation
//!
template <>
struct EnumMaxImpl<ElementWiseOperation>
{
    static constexpr int32_t kVALUE = 14;
};
} // namespace impl

//!
//! \class IElementWiseLayer
//!
//! \brief A elementwise layer in a network definition.
//!
//! This layer applies a per-element binary operation between corresponding elements of two tensors.
//!
//! The input tensors must have the same rank. For each dimension, their lengths must
//! match, or one of them must be one. In the latter case, the tensor is broadcast along that axis.
//!
//! The output tensor has the same rank as the inputs. For each output dimension,
//! its length is equal to the lengths of the corresponding input dimensions if they match,
//! otherwise it is equal to the length that is not one.
//!
//! \warning When running this layer on the DLA with Int8 data type, the dynamic ranges of two input tensors shall be
//! equal. If the dynamic ranges are generated using calibrator, the largest value shall be used.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IElementWiseLayer : public ILayer
{
public:
    //!
    //! \brief Set the binary operation for the layer.
    //!
    //! DLA supports only kSUM, kPROD, kMAX, kMIN, and kSUB.
    //!
    //! \see getOperation(), ElementWiseOperation
    //!
    //! \see getBiasWeights()
    //!
    void setOperation(ElementWiseOperation op) noexcept
    {
        return mImpl->setOperation(op);
    }

    //!
    //! \brief Get the binary operation for the layer.
    //!
    //! \see setOperation(), ElementWiseOperation
    //!
    //! \see setBiasWeights()
    //!
    ElementWiseOperation getOperation() const noexcept
    {
        return mImpl->getOperation();
    }

protected:
    apiv::VElementWiseLayer* mImpl;
    virtual ~IElementWiseLayer() noexcept = default;
};

//!
//! \brief Control form of IGatherLayer
//!
//! \see IGatherLayer
//!
enum class GatherMode : int32_t
{
    kDEFAULT = 0, //!< Similar to ONNX Gather
    kELEMENT = 1, //!< Similar to ONNX GatherElements
    kND = 2       //!< Similar to ONNX GatherND
};

//!
//! Maximum number of elements in GatherMode enum.
//!
//! \see GatherMode
//!
template <>
constexpr inline int32_t EnumMax<GatherMode>() noexcept
{
    return 3;
}

//!
//! \class IGatherLayer
//!
//! \brief A Gather layer in a network definition. Supports several kinds of gathering.
//!
//! The Gather layer has two input tensors, Data and Indices, and an output tensor Output.
//! Additionally, there are three parameters: mode, nbElementwiseDims, and axis that control
//! how the indices are interpreted.
//!
//! * Data is a tensor of rank r >= 1 that stores the values to be gathered in Output.
//! * Indices is a tensor of rank q that determines which locations in Data to gather.
//!     * GatherMode::kDEFAULT: q >= 0
//!     * GatherMode::kND:      q >= 1 and the last dimension of Indices must be a build time constant.
//!     * GatherMode::kELEMENT: q = r
//! * Output stores the gathered results. Its rank s depends on the mode:
//!     * GatherMode::kDEFAULT: s = q + r - 1 - nbElementwiseDims
//!     * GatherMode::kND:      s = q + r - indices.d[q-1] - 1 - nbElementwiseDims
//!     * GatherMode::kELEMENT: s = q = r.
//! The output can be a shape tensor only if the mode is GatherMode::kDEFAULT.
//!
//! The dimensions of the output likewise depends on the mode:
//!
//!     GatherMode::kDEFAULT:
//!
//!         First nbElementwiseDims of output are computed by applying broadcast rules to
//!         first nbElementwiseDims of indices and data. Note that nbElementwiseDims <= 1.
//!         Rest of dimensions are computed by copying dimensions of Data, and replacing
//!         the dimension for axis gatherAxis with the dimensions of indices.
//!
//!     GatherMode::kND:
//!         If indices.d[q-1] = r - nbElementwiseDims
//!             output.d = [indices.d[0], ... , indices.d[q-2]]
//!         Else if indices.d[q-1] < r - nbElementWiseDims
//!             output.d = [indices.d[0], ... , indices.d[q-1], data.d[nbElementwiseDims + indices.d[q-1] + q],
//!             data.d[r-1]]
//!         Else
//!             This is build time error
//!
//!     GatherMode::kELEMENT:
//!         The output dimensions match the dimensions of the indices tensor.
//!
//! The types of Data and Output must be the same, and Indices shall be DataType::kINT32.
//!
//! How the elements of Data are gathered depends on the mode:
//!
//!     GatherMode::kDEFAULT:
//!         Each index in indices is used to index Data along axis gatherAxis.
//!
//!     GatherMode::kND:
//!         Indices is a rank q integer tensor, best thought of as a rank (q-1) tensor of
//!         indices into data, where each element defines a slice of data
//!         The operation can be formulated as output[i_1, ..., i_{q-1}] = data[indices[i_1, ..., i_{q-1}]]
//!
//!     GatherMode::kELEMENT:
//!
//!         Here "axis" denotes the result of getGatherAxis().
//!         For each element X of indices:
//!             Let J denote a sequence for the subscripts of X
//!             Let K = sequence J with element [axis] replaced by X
//!             output[J] = data[K]
//!
//! The handling of nbElementWiseDims depends on the mode:
//!     * GatherMode::kDEFAULT: nbElementWiseDims <= 1. Broadcast is supported across the elementwise dimension if
//!     present.
//!     * GatherMode::kND:      0 <= nbElementWiseDims < rank(Data)-1. Broadcast is not supported across the elementwise
//!     dimensions.
//!     * GatherMode::kELEMENT: nbElementWiseDims = 0
//!
//! Notes:
//! * For modes GatherMode::kND and GatherMode::kELEMENT, the first nbElementWiseDims dimensions of data and index must
//! be equal. If not, an error will be reported at build time or run time.
//! * Only mode GatherMode::kDEFAULT supports an implicit batch dimensions or broadcast on the elementwise dimensions.
//! * If an axis of Data has dynamic length, using a negative index for it has undefined behavior.
//! * No DLA support
//! * Zero will be stored for OOB access
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IGatherLayer : public ILayer
{
public:
    //!
    //! \brief Set the axis used by GatherMode::kELEMENTS and GatherMode::kDEFAULT
    //! The axis must be less than the number of dimensions in the data input.
    //! The axis defaults to 0.
    //!
    //! \warning Undefined behavior when used with GatherMode::kND.
    //!
    //! \see getGatherAxis()
    //!
    void setGatherAxis(int32_t axis) noexcept
    {
        mImpl->setGatherAxis(axis);
    }

    //!
    //! \brief Get the axis to gather on.
    //! \warning Undefined behavior when used with GatherMode::kND.
    //!
    //! \see setGatherAxis()
    //!
    int32_t getGatherAxis() const noexcept
    {
        return mImpl->getGatherAxis();
    }

    //! \brief Set the number of leading dimensions of indices tensor to be handled elementwise.
    //! The gathering of indexing starts from the dimension of data[NbElementWiseDims:].
    //! The NbElementWiseDims must be less than the Rank of the data input.
    //! \param elementWiseDims number of dims to be handled as elementwise.
    //!
    //! Default: 0
    //!
    //! The value of nbElementWiseDims and GatherMode are checked during network validation:
    //!
    //! GatherMode::kDEFAULT: nbElementWiseDims must be 0 if there is an implicit batch dimension. It can be 0 or 1 if
    //! there is not an implicit batch dimension.
    //! GatherMode::kND: nbElementWiseDims can be between 0 and one less than rank(data).
    //! GatherMode::kELEMENT: nbElementWiseDims must be 0
    //!
    //! \see getNbElementWiseDims()
    //!
    void setNbElementWiseDims(int32_t elementWiseDims) noexcept
    {
        mImpl->setNbElementWiseDims(elementWiseDims);
    }

    //!
    //! \brief Get the number of leading dimensions of indices tensor to be handled elementwise.
    //!
    //! \see setNbElementWiseDims()
    //!
    int32_t getNbElementWiseDims() const noexcept
    {
        return mImpl->getNbElementWiseDims();
    }

    //!
    //! \brief Set the gather mode.
    //!
    //! \see getMode()
    //!
    void setMode(GatherMode mode) noexcept
    {
        mImpl->setMode(mode);
    }

    //!
    //! \brief Get the gather mode.
    //!
    //! \see setMode()
    //!
    GatherMode getMode() const noexcept
    {
        return mImpl->getMode();
    }

protected:
    apiv::VGatherLayer* mImpl;
    virtual ~IGatherLayer() noexcept = default;
};

//!
//! \enum RNNOperation
//!
//! \brief Enumerates the RNN operations that may be performed by an RNN layer.
//!
//! __Equation definitions__
//!
//! The equations below have the following naming convention:
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
//! \see IRNNv2Layer
//!
enum class RNNOperation : int32_t
{
    kRELU = 0, //!< Single gate RNN w/ ReLU activation function.
    kTANH = 1, //!< Single gate RNN w/ TANH activation function.
    kLSTM = 2, //!< Four-gate LSTM network w/o peephole connections.
    kGRU = 3   //!< Three-gate network consisting of Gated Recurrent Units.
};

//!
//! Maximum number of elements in RNNOperation enum.
//!
//! \see RNNOperation
//!
template <>
constexpr inline int32_t EnumMax<RNNOperation>() noexcept
{
    return 4;
}

//!
//! \enum RNNDirection
//!
//! \brief Enumerates the RNN direction that may be performed by an RNN layer.
//!
//! \see IRNNv2Layer
//!
enum class RNNDirection : int32_t
{
    kUNIDIRECTION = 0, //!< Network iterations from first input to last input.
    kBIDIRECTION = 1   //!< Network iterates from first to last and vice versa and outputs concatenated.
};

//!
//! Maximum number of elements in RNNDirection enum.
//!
//! \see RNNDirection
//!
template <>
constexpr inline int32_t EnumMax<RNNDirection>() noexcept
{
    return 2;
}

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
//! \see IRNNv2Layer
//!
enum class RNNInputMode : int32_t
{
    kLINEAR = 0, //!< Perform the normal matrix multiplication in the first recurrent layer.
    kSKIP = 1    //!< No operation is performed on the first recurrent layer.
};

//!
//! Maximum number of elements in RNNInputMode enum.
//!
//! \see RNNInputMode
//!
template <>
constexpr inline int32_t EnumMax<RNNInputMode>() noexcept
{
    return 2;
}

//!
//! \enum RNNGateType
//!
//! \brief Identifies an individual gate within an RNN cell.
//!
//! \see RNNOperation
//!
enum class RNNGateType : int32_t
{
    kINPUT = 0,  //!< Input gate  (i).
    kOUTPUT = 1, //!< Output gate (o).
    kFORGET = 2, //!< Forget gate (f).
    kUPDATE = 3, //!< Update gate (z).
    kRESET = 4,  //!< Reset gate  (r).
    kCELL = 5,   //!< Cell gate   (c).
    kHIDDEN = 6  //!< Hidden gate (h).
};

//!
//! Maximum number of elements in RNNGateType enum.
//!
//! \see RNNGateType
//!
template <>
constexpr inline int32_t EnumMax<RNNGateType>() noexcept
{
    return 7;
}

//!
//! \class IRNNv2Layer
//!
//! \brief An RNN layer in a network definition, version 2.
//!
//! This layer supersedes IRNNLayer.
//!
//! \deprecated Deprecated prior to TensorRT 8.0 and will be removed in 9.0. Superseded by
//! INetworkDefinition::addLoop().
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class TRT_DEPRECATED IRNNv2Layer : public ILayer
{
public:
    int32_t getLayerCount() const noexcept
    {
        return mImpl->getLayerCount();
    } //!< Get the layer count of the RNN.
    int32_t getHiddenSize() const noexcept
    {
        return mImpl->getHiddenSize();
    } //!< Get the hidden size of the RNN.
    int32_t getMaxSeqLength() const noexcept
    {
        return mImpl->getMaxSeqLength();
    } //!< Get the maximum sequence length of the RNN.
    int32_t getDataLength() const noexcept
    {
        return mImpl->getDataLength();
    } //!< Get the maximum data length of the RNN.

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
    void setSequenceLengths(ITensor& seqLengths) noexcept
    {
        return mImpl->setSequenceLengths(seqLengths);
    }

    //!
    //! \brief Get the sequence lengths specified for the RNN.
    //!
    //! \return nullptr if no sequence lengths were specified, the sequence length data otherwise.
    //!
    //! \see setSequenceLengths()
    //!
    ITensor* getSequenceLengths() const noexcept
    {
        return mImpl->getSequenceLengths();
    }

    //!
    //! \brief Set the operation of the RNN layer.
    //!
    //! \see getOperation(), RNNOperation
    //!
    void setOperation(RNNOperation op) noexcept
    {
        mImpl->setOperation(op);
    }

    //!
    //! \brief Get the operation of the RNN layer.
    //!
    //! \see setOperation(), RNNOperation
    //!
    RNNOperation getOperation() const noexcept
    {
        return mImpl->getOperation();
    }

    //!
    //! \brief Set the input mode of the RNN layer.
    //!
    //! \see getInputMode(), RNNInputMode
    //!
    void setInputMode(RNNInputMode op) noexcept
    {
        mImpl->setInputMode(op);
    }

    //!
    //! \brief Get the input mode of the RNN layer.
    //!
    //! \see setInputMode(), RNNInputMode
    //!
    RNNInputMode getInputMode() const noexcept
    {
        return mImpl->getInputMode();
    }

    //!
    //! \brief Set the direction of the RNN layer.
    //!
    //! The direction determines if the RNN is run as a unidirectional(left to right) or
    //! bidirectional(left to right and right to left).
    //! In the ::kBIDIRECTION case the output is concatenated together, resulting
    //! in output size of 2x getHiddenSize().
    //!
    //! \see getDirection(), RNNDirection
    //!
    void setDirection(RNNDirection op) noexcept
    {
        mImpl->setDirection(op);
    }

    //!
    //! \brief Get the direction of the RNN layer.
    //!
    //! \see setDirection(), RNNDirection
    //!
    RNNDirection getDirection() const noexcept
    {
        return mImpl->getDirection();
    }

    //!
    //! \brief Set the weight parameters for an individual gate in the RNN.
    //!
    //! The #DataType for this structure must be ::kFLOAT or ::kHALF, and must be the same
    //! datatype as the input tensor.
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
    //! \param layerIndex The index of the layer that contains this gate.  See the section
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
    void setWeightsForGate(int32_t layerIndex, RNNGateType gate, bool isW, Weights weights) noexcept
    {
        mImpl->setWeightsForGate(layerIndex, gate, isW, weights);
    }

    //!
    //! \brief Get the weight parameters for an individual gate in the RNN.
    //!
    //! \see setWeightsForGate()
    //!
    Weights getWeightsForGate(int32_t layerIndex, RNNGateType gate, bool isW) const noexcept
    {
        return mImpl->getWeightsForGate(layerIndex, gate, isW);
    }

    //!
    //! \brief Set the bias parameters for an individual gate in the RNN.
    //!
    //! The #DataType for this structure must be ::kFLOAT or ::kHALF, and must be the same
    //! datatype as the input tensor.
    //!
    //! Each bias vector has a fixed size, getHiddenSize().
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
    void setBiasForGate(int32_t layerIndex, RNNGateType gate, bool isW, Weights bias) noexcept
    {
        mImpl->setBiasForGate(layerIndex, gate, isW, bias);
    }

    //!
    //! \brief Get the bias parameters for an individual gate in the RNN.
    //!
    //! \see setBiasForGate()
    //!
    Weights getBiasForGate(int32_t layerIndex, RNNGateType gate, bool isW) const noexcept
    {
        return mImpl->getBiasForGate(layerIndex, gate, isW);
    }

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
    void setHiddenState(ITensor& hidden) noexcept
    {
        mImpl->setHiddenState(hidden);
    }

    //!
    //! \brief Get the initial hidden state of the RNN.
    //!
    //! \see setHiddenState()
    //!
    ITensor* getHiddenState() const noexcept
    {
        return mImpl->getHiddenState();
    }

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
    void setCellState(ITensor& cell) noexcept
    {
        mImpl->setCellState(cell);
    }

    //!
    //! \brief Get the initial cell state of the RNN.
    //!
    //! \see setCellState()
    //!
    ITensor* getCellState() const noexcept
    {
        return mImpl->getCellState();
    }

protected:
    apiv::VRNNv2Layer* mImpl;
    virtual ~IRNNv2Layer() noexcept = default;
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
    IPluginV2& getPlugin() noexcept
    {
        return mImpl->getPlugin();
    }

protected:
    apiv::VPluginV2Layer* mImpl;
    virtual ~IPluginV2Layer() noexcept = default;
};

//!
//! \enum UnaryOperation
//!
//! \brief Enumerates the unary operations that may be performed by a Unary layer.
//!
//! Operations kNOT must have inputs of #DataType kBOOL.
//!
//! Operation kSIGN must have inputs of #DataType kFLOAT, kHALF, kINT8, or kINT32.
//!
//! All other operations must have inputs of #DataType kFLOAT, kHALF, or kINT8.
//!
//! \see IUnaryLayer
//!
enum class UnaryOperation : int32_t
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
    kFLOOR = 18, //!< Floor.
    kERF = 19,   //!< Gauss error function.
    kNOT = 20,   //!< Logical NOT.
    kSIGN = 21,  //!< Sign, If input > 0, output 1; if input < 0, output -1; if input == 0, output 0.
    kROUND = 22  //!< Round to nearest even for float datatype.
};

//!
//! Maximum number of elements in UnaryOperation enum.
//!
//! \see UnaryOperation
//!
template <>
constexpr inline int32_t EnumMax<UnaryOperation>() noexcept
{
    return 23;
}

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
    //! When running this layer on DLA, only UnaryOperation::kABS is supported.
    //!
    //! \see getOperation(), UnaryOperation
    //!
    void setOperation(UnaryOperation op) noexcept
    {
        mImpl->setOperation(op);
    }

    //!
    //! \brief Get the unary operation for the layer.
    //!
    //! \see setOperation(), UnaryOperation
    //!
    UnaryOperation getOperation() const noexcept
    {
        return mImpl->getOperation();
    }

protected:
    apiv::VUnaryLayer* mImpl;
    virtual ~IUnaryLayer() noexcept = default;
};

//!
//! \enum ReduceOperation
//!
//! \brief Enumerates the reduce operations that may be performed by a Reduce layer.
//!
//! The table shows the result of reducing across an empty volume of a given type.
//!
//! Operation | kFLOAT and kHALF  | kINT32  | kINT8
//! --------- | ----------------- | ------- | -----
//! kSUM      | 0                 | 0       | 0
//! kPROD     | 1                 | 1       | 1
//! kMAX      | negative infinity | INT_MIN | -128
//! kMIN      | positive infinity | INT_MAX | 127
//! kAVG      | NaN               | 0       | -128
//!
//! The current version of TensorRT usually performs reduction for kINT8 via kFLOAT or kHALF.
//! The kINT8 values show the quantized representations of the floating-point values.
//!
enum class ReduceOperation : int32_t
{
    kSUM = 0,
    kPROD = 1,
    kMAX = 2,
    kMIN = 3,
    kAVG = 4
};

//!
//! Maximum number of elements in ReduceOperation enum.
//!
//! \see ReduceOperation
//!
template <>
constexpr inline int32_t EnumMax<ReduceOperation>() noexcept
{
    return 5;
}

//!
//! \class IReduceLayer
//!
//! \brief Layer that represents a reduction across a non-bool tensor.
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
    void setOperation(ReduceOperation op) noexcept
    {
        mImpl->setOperation(op);
    }

    //!
    //! \brief Get the reduce operation for the layer.
    //!
    //! \see setOperation(), ReduceOperation
    //!
    ReduceOperation getOperation() const noexcept
    {
        return mImpl->getOperation();
    }

    //!
    //! \brief Set the axes over which to reduce.
    //!
    //! \see getReduceAxes
    //!
    void setReduceAxes(uint32_t reduceAxes) noexcept
    {
        mImpl->setReduceAxes(reduceAxes);
    }

    //!
    //! \brief Get the axes over which to reduce for the layer.
    //!
    //! \see setReduceAxes
    //!
    uint32_t getReduceAxes() const noexcept
    {
        return mImpl->getReduceAxes();
    }

    //!
    //! \brief Set the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    //! \see getKeepDimensions
    //!
    void setKeepDimensions(bool keepDimensions) noexcept
    {
        mImpl->setKeepDimensions(keepDimensions);
    }

    //!
    //! \brief Get the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    //! \see setKeepDimensions
    //!
    bool getKeepDimensions() const noexcept
    {
        return mImpl->getKeepDimensions();
    }

protected:
    apiv::VReduceLayer* mImpl;
    virtual ~IReduceLayer() noexcept = default;
};

//!
//! \class IPaddingLayer
//!
//! \brief Layer that represents a padding operation.
//!
//! The padding layer adds zero-padding at the start and end of the input tensor. It only supports padding along the two
//! innermost dimensions. Applying negative padding results in cropping of the input.
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
    //! \deprecated Superseded by setPrePaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setPrePadding(DimsHW padding) noexcept
    {
        mImpl->setPrePadding(padding);
    }

    //!
    //! \brief Get the padding that is applied at the start of the tensor.
    //!
    //! \see setPrePadding
    //!
    //! \deprecated Superseded by getPrePaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getPrePadding() const noexcept
    {
        return mImpl->getPrePadding();
    }

    //!
    //! \brief Set the padding that is applied at the end of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount
    //!
    //! \see getPostPadding
    //!
    //! \deprecated Superseded by setPostPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED void setPostPadding(DimsHW padding) noexcept
    {
        mImpl->setPostPadding(padding);
    }

    //!
    //! \brief Get the padding that is applied at the end of the tensor.
    //!
    //! \see setPostPadding
    //!
    //! \deprecated Superseded by getPostPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED DimsHW getPostPadding() const noexcept
    {
        return mImpl->getPostPadding();
    }

    //!
    //! \brief Set the padding that is applied at the start of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount.
    //!
    //! \warning Only 2 dimensional padding is currently supported.
    //!
    //! \see getPrePaddingNd
    //!
    void setPrePaddingNd(Dims padding) noexcept
    {
        mImpl->setPrePaddingNd(padding);
    }

    //!
    //! \brief Get the padding that is applied at the start of the tensor.
    //!
    //! \warning Only 2 dimensional padding is currently supported.
    //!
    //! \see setPrePaddingNd
    //!
    Dims getPrePaddingNd() const noexcept
    {
        return mImpl->getPrePaddingNd();
    }

    //!
    //! \brief Set the padding that is applied at the end of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount
    //!
    //! \warning Only 2 dimensional padding is currently supported.
    //!
    //! \see getPostPaddingNd
    //!
    void setPostPaddingNd(Dims padding) noexcept
    {
        mImpl->setPostPaddingNd(padding);
    }

    //!
    //! \brief Get the padding that is applied at the end of the tensor.
    //!
    //! \warning Only 2 dimensional padding is currently supported.
    //!
    //! \see setPostPaddingNd
    //!
    Dims getPostPaddingNd() const noexcept
    {
        return mImpl->getPostPaddingNd();
    }

protected:
    apiv::VPaddingLayer* mImpl;
    virtual ~IPaddingLayer() noexcept = default;
};

struct Permutation
{
    //!
    //! The elements of the permutation.
    //! The permutation is applied as outputDimensionIndex = permutation.order[inputDimensionIndex], so to
    //! permute from CHW order to HWC order, the required permutation is [1, 2, 0], and to permute
    //! from HWC to CHW, the required permutation is [2, 0, 1].
    //!
    int32_t order[Dims::MAX_DIMS];
};

//! \class IShuffleLayer
//!
//! \brief Layer type for shuffling data.
//!
//! This layer shuffles data by applying in sequence: a transpose operation, a reshape operation
//! and a second transpose operation. The dimension types of the output are those of the reshape dimension.
//!
//! The layer has an optional second input.  If present, it must be a 1D Int32 shape tensor,
//! and the reshape dimensions are taken from it.
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
    void setFirstTranspose(Permutation permutation) noexcept
    {
        mImpl->setFirstTranspose(permutation);
    }

    //!
    //! \brief Get the permutation applied by the first transpose operation.
    //!
    //! \return The dimension permutation applied before the reshape.
    //!
    //! \see setFirstTranspose
    //!
    Permutation getFirstTranspose() const noexcept
    {
        return mImpl->getFirstTranspose();
    }

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
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    void setReshapeDimensions(Dims dimensions) noexcept
    {
        mImpl->setReshapeDimensions(dimensions);
    }

    //!
    //! \brief Get the reshaped dimensions.
    //!
    //! \return The reshaped dimensions.
    //!
    //! If a second input is present and non-null, or setReshapeDimensions has
    //! not yet been called, this function returns Dims with nbDims == -1.
    //!
    Dims getReshapeDimensions() const noexcept
    {
        return mImpl->getReshapeDimensions();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //
    //! Sets the input tensor for the given index. The index must be 0 for a static shuffle layer.
    //! A static shuffle layer is converted to a dynamic shuffle layer by calling setInput with an index 1.
    //! A dynamic shuffle layer cannot be converted back to a static shuffle layer.
    //!
    //! For a dynamic shuffle layer, the values 0 and 1 are valid.
    //! The indices in the dynamic case are as follows:
    //!
    //! - 0: Data or Shape tensor to be shuffled.
    //! - 1: The dimensions for the reshape operation, as a 1D Int32 shape tensor.
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    //!
    //! The reshape dimensions are treated identically to how they are treated if set statically
    //! via setReshapeDimensions. In particular, a -1 is treated as a wildcard even if dynamically
    //! supplied at runtime, and a 0 is treated as a placeholder if getZeroIsPlaceholder() = true,
    //! which is the default. If the placeholder interpretation of 0 is unwanted because the
    //! runtime dimension should be 0 when the reshape dimension is 0, be sure to call
    //! setZeroIsPlacholder(false) on the IShuffleLayer.
    //!
    //! \see setReshapeDimensions.
    //!
    using ILayer::setInput;

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
    void setSecondTranspose(Permutation permutation) noexcept
    {
        mImpl->setSecondTranspose(permutation);
    }

    //!
    //! \brief Get the permutation applied by the second transpose operation.
    //!
    //! \return The dimension permutation applied after the reshape.
    //!
    //! \see setSecondTranspose
    //!
    Permutation getSecondTranspose() const noexcept
    {
        return mImpl->getSecondTranspose();
    }

    //!
    //! \brief Set meaning of 0 in reshape dimensions.
    //!
    //! If true, then a 0 in the reshape dimensions denotes copying the corresponding
    //! dimension from the first input tensor.  If false, then a 0 in the reshape
    //! dimensions denotes a zero-length dimension.
    //!
    //! Default: true
    //!
    //! \see getZeroIsPlaceholder();
    //!
    void setZeroIsPlaceholder(bool zeroIsPlaceholder) noexcept
    {
        return mImpl->setZeroIsPlaceholder(zeroIsPlaceholder);
    }

    //!
    //! \brief Get meaning of 0 in reshape dimensions.
    //!
    //! \return true if 0 is placeholder for corresponding input dimension,
    //!         false if 0 denotes a zero-length dimension.
    //!
    //! \see setZeroIsPlaceholder
    //!
    bool getZeroIsPlaceholder() const noexcept
    {
        return mImpl->getZeroIsPlaceholder();
    }

protected:
    apiv::VShuffleLayer* mImpl;
    virtual ~IShuffleLayer() noexcept = default;
};

//!
//! \brief Controls how ISliceLayer and IGridSample handle out-of-bounds coordinates.
//!
//! \see ISliceLayer and IGridSample
//!
enum class SampleMode : int32_t
{
    kSTRICT_BOUNDS = 0,                            //!< Fail with error when the coordinates are out of bounds.
    kDEFAULT TRT_DEPRECATED_ENUM = kSTRICT_BOUNDS, //! \deprecated Use kSTRICT_BOUNDS.
    kWRAP = 1,                                     //!< Coordinates wrap around periodically.
    kCLAMP = 2,                                    //!< Out of bounds indices are clamped to bounds.
    kFILL = 3,                                     //!< Use fill input value when coordinates are out of bounds.
    kREFLECT = 4, //!< Coordinates reflect. The axis of reflection is the middle of the perimeter pixel and the
                  //!< reflections are repeated indefinitely within the padded regions. Repeats values for a single
                  //!< pixel and throws error for zero pixels.
};

//! \deprecated Deprecated in TensorRT 8.5. Superseded by SampleMode.
using SliceMode = SampleMode;

//!
//! Maximum number of elements in SampleMode enum.
//!
//! \see SampleMode
//!
template <>
constexpr inline int32_t EnumMax<SampleMode>() noexcept
{
    return 5;
}

//!
//! \brief Slices an input tensor into an output tensor based on the offset and strides.
//!
//! The slice layer has two variants, static and dynamic. Static slice specifies the start, size, and stride
//! dimensions at layer creation time via Dims and can use the get/set accessor functions of the ISliceLayer.
//! Dynamic slice specifies one or more of start, size or stride as ITensors, by using ILayer::setInput to add
//! a second, third, or fourth input respectively. The corresponding Dims are used if an input
//! is missing or null.
//!
//! An application can determine if the ISliceLayer has a dynamic output shape based on whether
//! the size input (third input) is present and non-null.
//!
//! The slice layer selects for each dimension a start location from within the input tensor, and
//! copies elements to the output tensor using the specified stride across the input tensor.
//! Start, size, and stride tensors must be 1D Int32 shape tensors if not specified via Dims.
//!
//! An example of using slice on a tensor:
//! input = {{0, 2, 4}, {1, 3, 5}}
//! start = {1, 0}
//! size = {1, 2}
//! stride = {1, 2}
//! output = {{1, 5}}
//!
//! When the sliceMode is kCLAMP or kREFLECT, for each input dimension, if its size is 0 then the corresponding output
//! dimension must be 0 too.
//!
//! A slice layer can produce a shape tensor if the following conditions are met:
//!
//! * start, size, and stride are build time constants, either as static Dims or as constant input tensors.
//! * The number of elements in the output tensor does not exceed 2*Dims::MAX_DIMS.
//!
//! The input tensor is a shape tensor if the output is a shape tensor.
//!
//! The following constraints must be satisfied to execute this layer on DLA:
//! * start, size, and stride are build time constants, either as static Dims or as constant input tensors.
//! * sliceMode is kDEFAULT.
//! * Strides are 1 for all dimensions.
//! * Slicing is not performed on the first dimension
//! * The input tensor has four dimensions
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISliceLayer : public ILayer
{
public:
    //!
    //! \brief Set the start offset that the slice layer uses to create the output slice.
    //!
    //! \param start The start offset to read data from the input tensor.
    //!
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getStart
    //!
    void setStart(Dims start) noexcept
    {
        mImpl->setStart(start);
    }

    //!
    //! \brief Get the start offset for the slice layer.
    //!
    //! \return The start offset, or an invalid Dims structure.
    //!
    //! If the second input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setStart
    //!
    Dims getStart() const noexcept
    {
        return mImpl->getStart();
    }

    //!
    //! \brief Set the dimensions of the output slice.
    //!
    //! \param size The dimensions of the output slice.
    //!
    //! If a third input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getSize
    //!
    void setSize(Dims size) noexcept
    {
        return mImpl->setSize(size);
    }

    //!
    //! \brief Get dimensions of the output slice.
    //!
    //! \return The output dimension, or an invalid Dims structure.
    //!
    //! If the third input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setSize
    //!
    Dims getSize() const noexcept
    {
        return mImpl->getSize();
    }

    //!
    //! \brief Set the stride for computing the output slice data.
    //!
    //! \param stride The dimensions of the stride to compute the values to store in the output slice.
    //!
    //! If a fourth input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getStride
    //!
    void setStride(Dims stride) noexcept
    {
        mImpl->setStride(stride);
    }

    //!
    //! \brief Get the stride for the output slice.
    //!
    //! \return The slicing stride, or an invalid Dims structure.
    //!
    //! If the fourth input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setStride
    //!
    Dims getStride() const noexcept
    {
        return mImpl->getStride();
    }

    //!
    //! \brief Set the slice mode.
    //!
    //! \see getMode()
    //!
    void setMode(SliceMode mode) noexcept
    {
        mImpl->setMode(mode);
    }

    //!
    //! \brief Get the slice mode.
    //!
    //! \see setMode()
    //!
    SliceMode getMode() const noexcept
    {
        return mImpl->getMode();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! For a slice layer, the values 0-4 are valid.
    //! The indices are as follows:
    //!
    //! - 0: Tensor to be sliced.
    //! - 1: The start tensor to begin slicing, as a 1D Int32 shape tensor.
    //! - 2: The size tensor of the resulting slice, as a 1D Int32 shape tensor.
    //! - 3: The stride of the slicing operation, as a 1D Int32 shape tensor.
    //! - 4: Value for the kFILL slice mode. The fill value data type should have the same
    //!      data phylum as input data type. And this input is disallowed for other modes.
    //!
    //! Using the corresponding setter resets the input to null.
    //!
    //! If this function is called with a value greater than 0, then the function getNbInputs() changes
    //! from returning 1 to index + 1.
    //!
    using ILayer::setInput;

protected:
    apiv::VSliceLayer* mImpl;
    virtual ~ISliceLayer() noexcept = default;
};

//! \class IShapeLayer
//!
//! \brief Layer type for getting shape of a tensor.
//!
//! This layer sets the output to a 1D tensor of type Int32 with the dimensions of the input tensor.
//!
//! For example, if the input is a four-dimensional tensor (of any type) with
//! dimensions [2,3,5,7], the output tensor is a one-dimensional Int32 tensor
//! of length 4 containing the sequence 2, 3, 5, 7.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IShapeLayer : public ILayer
{
protected:
    apiv::VShapeLayer* mImpl;
    virtual ~IShapeLayer() noexcept = default;
};

//!
//! \enum TopKOperation
//!
//! \brief Enumerates the operations that may be performed by a TopK layer.
//!
enum class TopKOperation : int32_t
{
    kMAX = 0, //!< Maximum of the elements.
    kMIN = 1, //!< Minimum of the elements.
};

//!
//! Maximum number of elements in TopKOperation enum.
//!
//! \see TopKOperation
//!
template <>
constexpr inline int32_t EnumMax<TopKOperation>() noexcept
{
    return 2;
}

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
    void setOperation(TopKOperation op) noexcept
    {
        mImpl->setOperation(op);
    }

    //!
    //! \brief Get the operation for the layer.
    //!
    //! \see setOperation(), TopKOperation
    //!
    TopKOperation getOperation() const noexcept
    {
        return mImpl->getOperation();
    }

    //!
    //! \brief Set the k value for the layer.
    //!
    //! Currently only values up to 3840 are supported.
    //!
    //! \see getK()
    //!
    void setK(int32_t k) noexcept
    {
        mImpl->setK(k);
    }

    //!
    //! \brief Get the k value for the layer.
    //!
    //! \see setK()
    //!
    int32_t getK() const noexcept
    {
        return mImpl->getK();
    }

    //!
    //! \brief Set which axes to reduce for the layer.
    //!
    //! \see getReduceAxes()
    //!
    void setReduceAxes(uint32_t reduceAxes) noexcept
    {
        mImpl->setReduceAxes(reduceAxes);
    }

    //!
    //! \brief Get the axes to reduce for the layer.
    //!
    //! \see setReduceAxes()
    //!
    uint32_t getReduceAxes() const noexcept
    {
        return mImpl->getReduceAxes();
    }

protected:
    apiv::VTopKLayer* mImpl;
    virtual ~ITopKLayer() noexcept = default;
};

//!
//! \enum MatrixOperation
//!
//! \brief Enumerates the operations that may be performed on a tensor
//!        by IMatrixMultiplyLayer before multiplication.
//!
enum class MatrixOperation : int32_t
{
    //! Treat x as a matrix if it has two dimensions, or as a collection of
    //! matrices if x has more than two dimensions, where the last two dimensions
    //! are the matrix dimensions. x must have at least two dimensions.
    kNONE,

    //! Like kNONE, but transpose the matrix dimensions.
    kTRANSPOSE,

    //! Treat x as a vector if it has one dimension, or as a collection of
    //! vectors if x has more than one dimension. x must have at least one dimension.
    //!
    //! The first input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is equivalent to a tensor
    //! with dimensions [M, 1, K] with MatrixOperation::kNONE, i.e. is treated as M row vectors of length K,
    //! or dimensions [M, K, 1] with MatrixOperation::kTRANSPOSE.
    //!
    //! The second input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is equivalent to a tensor
    //! with dimensions [M, K, 1] with MatrixOperation::kNONE, i.e. is treated as M column vectors of length K,
    //! or dimensions [M, 1, K] with MatrixOperation::kTRANSPOSE.
    kVECTOR
};

//!
//! Maximum number of elements in MatrixOperation enum.
//!
//! \see DataType
//!
template <>
constexpr inline int32_t EnumMax<MatrixOperation>() noexcept
{
    return 3;
}

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
    //! \see getOperation()
    //!
    void setOperation(int32_t index, MatrixOperation op) noexcept
    {
        mImpl->setOperation(index, op);
    }

    //!
    //! \brief Get the operation for an input tensor.
    //!
    //! \param index Input tensor number (0 or 1).
    //!
    //! \see setOperation()
    //!
    MatrixOperation getOperation(int32_t index) const noexcept
    {
        return mImpl->getOperation(index);
    }

protected:
    apiv::VMatrixMultiplyLayer* mImpl;
    virtual ~IMatrixMultiplyLayer() noexcept = default;
};

//! \class INonZero
//!
//! \brief A NonZero layer in a network.
//!
//! This layer gets the positions of elements that are non-zero in the input.
//! For boolean input, "non-zero" means "true". Semantics are similar to ONNX NonZero.
//!
//! The input may have type kFLOAT, kHALF, kINT32, or kBOOL.
//!
//! The output is a matrix of type kINT32.
//! For an input with dimensions [L1, L2, ..., Lm], the output has dimensions [m,n],
//! where n is the number of non-zero elements. I.e., each column denotes a m-D position.
//!
//! The columns are lexically ordered.
//! E.g., a column with [3,2,4,7] precedes a column with [3,2,5,6].
//!
//! Tip: "compress" can be implemented with INonZero+IShuffle+Gather.
//! For example, to compress a tensor x over axis k using mask vector v,
//! use nonzero(v) to compute the subscripts, shuffle with reshape dimensions = [-1]
//! to make the subscripts 1D, and then gather with the subscripts.
//!
class INonZeroLayer : public ILayer
{
protected:
    virtual ~INonZeroLayer() noexcept = default;
    apiv::VNonZeroLayer* mImpl;
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
    apiv::VRaggedSoftMaxLayer* mImpl;
    virtual ~IRaggedSoftMaxLayer() noexcept = default;
};

//! \class IIdentityLayer
//!
//! \brief A layer that represents the identity function.
//!
//! If the output type is explicitly specified via setOutputType, IIdentityLayer can be
//! used to convert from one type to another. Other than conversions between the same
//! type (kFLOAT -> kFLOAT for example), the only valid conversions are:
//!
//!     (kFLOAT | kHALF | kINT32 | kBOOL) -> (kFLOAT | kHALF | kINT32 | kBOOL)
//!
//!     (kFLOAT | kHALF) -> kUINT8
//!
//!     kUINT8 -> (kFLOAT | kHALF)
//!
//! Conversion also happens implicitly, without calling setOutputType, if the output
//! tensor is a network output.
//!
//! Two types are compatible if they are identical, or are both in {kFLOAT, kHALF}.
//! Implicit conversion between incompatible types, i.e. without using setOutputType,
//! is recognized as incorrect as of TensorRT 8.4, but is retained for API compatibility
//! within TensorRT 8.x releases. In a future major release the behavior will change
//! to record an error if the network output tensor type is incompatible with the layer
//! output type. E.g., implicit conversion from kFLOAT to kINT32 will not be allowed,
//! and instead such a conversion will require calling setOutputType(DataType::kINT32).
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IIdentityLayer : public ILayer
{
protected:
    apiv::VIdentityLayer* mImpl;
    virtual ~IIdentityLayer() noexcept = default;
};

//! \class IConstantLayer
//!
//! \brief Layer that represents a constant value.
//!
//! \note This layer does not support boolean types.
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
    //! Otherwise the output is a tensor of real values and the output type will be
    //! follow TensorRT's normal precision rules.
    //!
    //! \see getWeights()
    //!
    void setWeights(Weights weights) noexcept
    {
        mImpl->setWeights(weights);
    }

    //!
    //! \brief Get the weights for the layer.
    //!
    //! \see setWeights
    //!
    Weights getWeights() const noexcept
    {
        return mImpl->getWeights();
    }

    //!
    //! \brief Set the dimensions for the layer.
    //!
    //! \param dimensions The dimensions of the layer
    //!
    //! \see setDimensions
    //!
    void setDimensions(Dims dimensions) noexcept
    {
        mImpl->setDimensions(dimensions);
    }

    //!
    //! \brief Get the dimensions for the layer.
    //!
    //! \return the dimensions for the layer
    //!
    //! \see getDimensions
    //!
    Dims getDimensions() const noexcept
    {
        return mImpl->getDimensions();
    }

protected:
    apiv::VConstantLayer* mImpl;
    virtual ~IConstantLayer() noexcept = default;
};

//!
//! \class IParametricReLULayer
//!
//! \brief Layer that represents a parametric ReLU operation.
//!
//! When running this layer on DLA, the slopes input must be a build-time constant.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IParametricReLULayer : public ILayer
{
protected:
    apiv::VParametricReLULayer* mImpl;
    virtual ~IParametricReLULayer() noexcept = default;
};

//! \enum InterpolationMode
//!
//! \brief Enumerates various modes of interpolation
//!
//!
enum class InterpolationMode : int32_t
{
    kNEAREST = 0, //!< ND (0 < N <= 8) nearest neighbor resizing.
    kLINEAR = 1,  //!< Supports linear (1D), bilinear (2D), and trilinear (3D) interpolation
    kCUBIC = 2    //!< Supports bicubic (2D) interpolation
};

//! \deprecated Deprecated in TensorRT 8.5. Superseded by InterpolationMode.
using ResizeMode = InterpolationMode;

namespace impl
{
//!
//! Maximum number of elements in InterpolationMode enum.
//!
//! \see InterpolationMode
//!
template <>
struct EnumMaxImpl<InterpolationMode>
{
    static constexpr int32_t kVALUE = 3;
};
} // namespace impl

//!
//! \enum ResizeCoordinateTransformation
//!
//! \brief The resize coordinate transformation function.
//!
//! \see IResizeLayer::setCoordinateTransformation()
//!
enum class ResizeCoordinateTransformation : int32_t
{
    //! Think of each value in the tensor as a unit volume, and the coordinate is a point inside this volume.
    //! The coordinate point is drawn as a star `(*)` in the below diagram, and multiple values range has a length.
    //! Define `x_origin` as the coordinate of axis x in the input tensor, `x_resized` as the coordinate of axis x in
    //! the output tensor, `length_origin` as length of the input tensor in axis x, and `length_resize` as length of the
    //! output tensor in axis x.
    //!
    //!     |<--------------length---------->|
    //!     |    0     |    1     |    2     |    3     |
    //!     *          *          *          *
    //!
    //!     x_origin = x_resized * (length_origin - 1) / (length_resize - 1)
    //!
    kALIGN_CORNERS = 0,

    //!     |<--------------length--------------------->|
    //!     |    0     |    1     |    2     |    3     |
    //!     *          *          *          *
    //!
    //!     x_origin = x_resized * (length_origin / length_resize)
    //!
    kASYMMETRIC = 1,

    //!     |<--------------length--------------------->|
    //!     |    0     |    1     |    2     |    3     |
    //!          *          *          *          *
    //!
    //!     x_origin = (x_resized + 0.5) * (length_origin / length_resize) - 0.5
    //!
    kHALF_PIXEL = 2,
};

namespace impl
{
//!
//! Maximum number of elements in ResizeCoordinateTransformation enum.
//!
//! \see ResizeCoordinateTransformation
//!
template <>
struct EnumMaxImpl<ResizeCoordinateTransformation>
{
    static constexpr int32_t kVALUE = 3;
};
} // namespace impl

//!
//! \enum ResizeSelector
//!
//! \brief The coordinate selector when resize to single pixel output.
//!
//! \see IResizeLayer::setSelectorForSinglePixel()
//!
enum class ResizeSelector : int32_t
{
    //! Use formula to map the original index.
    kFORMULA = 0,

    //! Select the upper left pixel.
    kUPPER = 1,
};

namespace impl
{
//!
//! Maximum number of elements in ResizeSelector enum.
//!
//! \see ResizeSelector
//!
template <>
struct EnumMaxImpl<ResizeSelector>
{
    static constexpr int32_t kVALUE = 2;
};
} // namespace impl

//!
//! \enum ResizeRoundMode
//!
//! \brief The rounding mode for nearest neighbor resize.
//!
//! \see IResizeLayer::setNearestRounding()
//!
enum class ResizeRoundMode : int32_t
{
    //! Round half up.
    kHALF_UP = 0,

    //! Round half down.
    kHALF_DOWN = 1,

    //! Round to floor.
    kFLOOR = 2,

    //! Round to ceil.
    kCEIL = 3,
};

namespace impl
{
//!
//! Maximum number of elements in ResizeRoundMode enum.
//!
//! \see ResizeRoundMode
//!
template <>
struct EnumMaxImpl<ResizeRoundMode>
{
    static constexpr int32_t kVALUE = 4;
};
} // namespace impl

//! \class IResizeLayer
//!
//! \brief A resize layer in a network definition.
//!
//! Resize layer can be used for resizing a N-D tensor.
//!
//! Resize layer currently supports the following configurations:
//!     -   ResizeMode::kNEAREST - resizes innermost `m` dimensions of N-D, where 0 < m <= min(8, N) and N > 0
//!     -   ResizeMode::kLINEAR - resizes innermost `m` dimensions of N-D, where 0 < m <= min(3, N) and N > 0
//!
//! Default resize mode is ResizeMode::kNEAREST.
//!
//! The coordinates in the output tensor are mapped to coordinates in the input tensor using a function set by calling
//! setCoordinateTransformation(). The default for all ResizeMode settings (nearest, linear, bilinear, etc.) is
//! ResizeCoordinateTransformation::kASYMMETRIC.
//!
//! The resize layer provides two ways to resize tensor dimensions.
//!     -   Set output dimensions directly. It can be done for static as well as dynamic resize layer.
//!         Static resize layer requires output dimensions to be known at build-time.
//!         Dynamic resize layer requires output dimensions to be set as one of the input tensors.
//!     -   Set scales for resize. Each output dimension is calculated as floor(input dimension * scale).
//!         Only static resize layer allows setting scales where the scales are known at build-time.
//!
//! If executing this layer on DLA, the following combinations of parameters are supported:
//!
//! - In kNEAREST mode:
//!     * (ResizeCoordinateTransformation::kASYMMETRIC, ResizeSelector::kFORMULA, ResizeRoundMode::kFLOOR)
//!     * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kFORMULA, ResizeRoundMode::kHALF_DOWN)
//!     * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kFORMULA, ResizeRoundMode::kHALF_UP)
//!
//! - In kLINEAR mode:
//!     * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kFORMULA)
//!     * (ResizeCoordinateTransformation::kHALF_PIXEL, ResizeSelector::kUPPER)
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IResizeLayer : public ILayer
{
public:
    //!
    //! \brief Set the output dimensions.
    //!
    //! \param dimensions The output dimensions. Number of output dimensions must be the same as the number of input
    //! dimensions.
    //!
    //! If executing this layer on DLA, setOutputDimensions() is not supported.
    //!
    //! If there is a second input, i.e. resize layer is dynamic,
    //! calling setOutputDimensions() is an error and does not update the
    //! dimensions.
    //!
    //! Output dimensions can be specified directly, or via scale factors relative to input dimensions.
    //! Scales for resize can be provided using setScales().
    //!
    //! \see setScales
    //! \see getOutputDimensions
    //!
    void setOutputDimensions(Dims dimensions) noexcept
    {
        return mImpl->setOutputDimensions(dimensions);
    }

    //!
    //! \brief Get the output dimensions.
    //!
    //! \return The output dimensions.
    //!
    Dims getOutputDimensions() const noexcept
    {
        return mImpl->getOutputDimensions();
    }

    //!
    //! \brief Set the resize scales.
    //!
    //! \param scales An array of resize scales.
    //! \param nbScales Number of scales. Number of scales must be equal to the number of input dimensions.
    //!
    //! If executing this layer on DLA, there are three restrictions:
    //! 1) nbScales has to be exactly 4.
    //! 2) the first two elements in scales need to be exactly 1 (for unchanged batch and channel dimensions).
    //! 3) The last two elements in scales, representing the scale values along height and width dimensions,
    //! respectively, need to be integer values in the range of [1, 32] for kNEAREST mode and [1, 4] for kLINEAR.
    //! Example of DLA-supported scales: {1, 1, 2, 2}.
    //!
    //! If there is a second input, i.e. resize layer is dynamic,
    //! calling setScales() is an error and does not update the scales.
    //!
    //! Output dimensions are calculated as follows:
    //! outputDims[i] = floor(inputDims[i] * scales[i])
    //!
    //! Output dimensions can be specified directly, or via scale factors relative to input dimensions.
    //! Output dimensions can be provided directly using setOutputDimensions().
    //!
    //! \see setOutputDimensions
    //! \see getScales
    //!
    void setScales(float const* scales, int32_t nbScales) noexcept
    {
        mImpl->setScales(scales, nbScales);
    }

    //!
    //! \brief Copies resize scales to scales[0, ..., nbScales-1], where nbScales is the number of scales that were set.
    //!
    //! \param size The number of scales to get. If size != nbScales, no scales will be copied.
    //!
    //! \param scales Pointer to where to copy the scales. Scales will be copied only if
    //!               size == nbScales and scales != nullptr.
    //!
    //! In case the size is not known consider using size = 0 and scales = nullptr. This method will return
    //! the number of resize scales.
    //!
    //! \return The number of resize scales i.e. nbScales if scales were set.
    //!         Return -1 in case no scales were set or resize layer is used in dynamic mode.
    //!
    int32_t getScales(int32_t size, float* scales) const noexcept
    {
        return mImpl->getScales(size, scales);
    }

    //!
    //! \brief Set resize mode for an input tensor.
    //!
    //! Supported resize modes are Nearest Neighbor and Linear.
    //!
    //! \see ResizeMode
    //!
    void setResizeMode(ResizeMode resizeMode) noexcept
    {
        mImpl->setResizeMode(resizeMode);
    }

    //!
    //! \brief Get resize mode for an input tensor.
    //!
    //! \return The resize mode.
    //!
    ResizeMode getResizeMode() const noexcept
    {
        return mImpl->getResizeMode();
    }

    //!
    //! \brief Set whether to align corners while resizing.
    //!
    //! If true, the centers of the 4 corner pixels of both input and output
    //! tensors are aligned i.e. preserves the values of corner
    //! pixels.
    //!
    //! Default: false.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by IResizeLayer::setCoordinateTransformation().
    //!
    TRT_DEPRECATED void setAlignCorners(bool alignCorners) noexcept
    {
        mImpl->setAlignCorners(alignCorners);
    }

    //!
    //! \brief True if align corners has been set.
    //!
    //! \return True if align corners has been set, false otherwise.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by IResizeLayer::getCoordinateTransformation().
    //!
    TRT_DEPRECATED bool getAlignCorners() const noexcept
    {
        return mImpl->getAlignCorners();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor.
    //!
    //! Sets the input tensor for the given index. The index must be 0 for a static resize layer.
    //! A static resize layer is converted to a dynamic resize layer by calling setInput with an index 1.
    //! A dynamic resize layer cannot be converted back to a static resize layer.
    //!
    //! For a dynamic resize layer, the values 0 and 1 are valid.
    //! The indices in the dynamic case are as follows:
    //!
    //! - 0: Execution tensor to be resized.
    //! - 1: The output dimensions, as a 1D Int32 shape tensor.
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    //!
    using ILayer::setInput;

    //!
    //! \brief Set coordinate transformation function.
    //!
    //! The function maps a coordinate in the output tensor to a coordinate in the input tensor.
    //!
    //! Default function is ResizeCoordinateTransformation::kASYMMETRIC.
    //!
    //! \see ResizeCoordinateTransformation
    //!
    void setCoordinateTransformation(ResizeCoordinateTransformation coordTransform) noexcept
    {
        mImpl->setCoordinateTransformation(coordTransform);
    }

    //!
    //! \brief Get coordinate transformation function.
    //!
    //! \return The coordinate transformation function.
    //!
    ResizeCoordinateTransformation getCoordinateTransformation() const noexcept
    {
        return mImpl->getCoordinateTransformation();
    }

    //!
    //! \brief Set coordinate selector function when resized to single pixel.
    //!
    //! When resize to single pixel image, use this function to decide how to map the coordinate in the original
    //! image.
    //!
    //! Default is ResizeSelector::kFORMULA.
    //!
    //! \see ResizeSelector
    //!
    void setSelectorForSinglePixel(ResizeSelector selector) noexcept
    {
        mImpl->setSelectorForSinglePixel(selector);
    }

    //!
    //! \brief Get the coordinate selector function when resized to single pixel.
    //!
    //! \return The selector function.
    //!
    ResizeSelector getSelectorForSinglePixel() const noexcept
    {
        return mImpl->getSelectorForSinglePixel();
    }

    //!
    //! \brief Set rounding mode for nearest neighbor resize.
    //!
    //! This value is used for nearest neighbor interpolation rounding. It is applied after coordinate transformation.
    //!
    //! Default is kFLOOR.
    //!
    //! \see ResizeRoundMode
    //!
    void setNearestRounding(ResizeRoundMode value) noexcept
    {
        mImpl->setNearestRounding(value);
    }

    //!
    //! \brief Get rounding mode for nearest neighbor resize.
    //!
    //! \return The rounding mode.
    //!
    ResizeRoundMode getNearestRounding() const noexcept
    {
        return mImpl->getNearestRounding();
    }

    //!
    //! \brief Set the coefficient 'A' used in cubic interpolation.
    //!
    //! Cubic uses the coefficient 'A' to calculate the weight of input pixels:
    //!
    //! <pre>
    //! x := The relative distance between the sampled pixels and the input coordinates.
    //!
    //! weight(x) := for |x| <= 1, ((A + 2) * x - (A + 3)) * x * x + 1,
    //!              for 1 < |x| < 2, ((A * x - 5 * A) * x + 8 * A) * x - 4 * A,
    //!              others 0;
    //! </pre>
    //!
    //! This attribute is valid only if "resize mode" is "cubic".
    //!
    //! The default value is -0.75.
    //!
    void setCubicCoeff(float A) noexcept
    {
        mImpl->setCubicCoeff(A);
    }

    //!
    //! \brief Get the coefficient 'A' used in cubic interpolation.
    //!
    //! \see setCubicCoeff()
    //!
    float getCubicCoeff() const noexcept
    {
        return mImpl->getCubicCoeff();
    }

    //!
    //! \brief Set the state for excluding outside pixels.
    //!
    //! If set to true, the weight of sampling locations outside the input tensor will be set to false, and the weight
    //! will be renormalized so that their sum is 1.0.
    //!
    //! The default value is false.
    //!
    void setExcludeOutside(bool excludeFlag) noexcept
    {
        mImpl->setExcludeOutside(excludeFlag);
    }

    //!
    //! \brief Get the state for excluding outside pixels.
    //!
    //! \see setExcludeOutside()
    //!
    bool getExcludeOutside() const noexcept
    {
        return mImpl->getExcludeOutside();
    }

protected:
    virtual ~IResizeLayer() noexcept = default;
    apiv::VResizeLayer* mImpl;
};

//! Enum that describes kinds of loop outputs.
enum class LoopOutput : int32_t
{
    //! Output value is value of tensor for last iteration.
    kLAST_VALUE = 0,

    //! Output value is concatenation of values of tensor for each iteration, in forward order.
    kCONCATENATE = 1,

    //! Output value is concatenation of values of tensor for each iteration, in reverse order.
    kREVERSE = 2
};

//!
//! Maximum number of elements in LoopOutput enum.
//!
//! \see DataType
//!
template <>
constexpr inline int32_t EnumMax<LoopOutput>() noexcept
{
    return 3;
}

//! Enum that describes kinds of trip limits.
enum class TripLimit : int32_t
{

    kCOUNT = 0, //!< Tensor is scalar of type kINT32 that contains the trip count.
    kWHILE = 1  //!< Tensor is a scalar of type kBOOL. Loop terminates when value is false.
};

//!
//! Maximum number of elements in TripLimit enum.
//!
//! \see DataType
//!
template <>
constexpr inline int32_t EnumMax<TripLimit>() noexcept
{
    return 2;
}

class ILoop;

class ILoopBoundaryLayer : public ILayer
{
public:
    //! Return pointer to ILoop associated with this boundary layer.
    ILoop* getLoop() const noexcept
    {
        return mBoundary->getLoop();
    }

protected:
    virtual ~ILoopBoundaryLayer() noexcept = default;
    apiv::VLoopBoundaryLayer* mBoundary;
};

//!
//! This is a base class for Conditional boundary layers.
//!
//! Boundary layers are used to demarcate the boundaries of Conditionals.
//!
class IIfConditionalBoundaryLayer : public ILayer
{
public:
    //! Return pointer to the IIfConditional associated with this boundary layer.
    IIfConditional* getConditional() const noexcept
    {
        return mBoundary->getConditional();
    }

protected:
    virtual ~IIfConditionalBoundaryLayer() noexcept = default;
    apiv::VConditionalBoundaryLayer* mBoundary;
};

//!
//! This layer represents a condition input to an IIfConditional.
//!
class IConditionLayer : public IIfConditionalBoundaryLayer
{
public:
protected:
    virtual ~IConditionLayer() noexcept = default;
    apiv::VConditionLayer* mImpl;
};

//!
//! This layer represents an output of an IIfConditional.
//!
//! An IIfConditionalOutputLayer has exactly one output.
//!
class IIfConditionalOutputLayer : public IIfConditionalBoundaryLayer
{
public:
protected:
    virtual ~IIfConditionalOutputLayer() noexcept = default;
    apiv::VConditionalOutputLayer* mImpl;
};

//!
//! This layer represents an input to an IIfConditional.
//!
class IIfConditionalInputLayer : public IIfConditionalBoundaryLayer
{
public:
protected:
    virtual ~IIfConditionalInputLayer() noexcept = default;
    apiv::VConditionalInputLayer* mImpl;
};

//!
//! Helper for constructing conditionally-executed subgraphs.
//!
//! An If-conditional conditionally executes part of the network according
//! to the following pseudo-code:
//!
//! If condition is true then:
//!     output = trueSubgraph(trueInputs);
//! Else
//!     output = falseSubgraph(falseInputs);
//! Emit output
//!
//! Condition is a 0D boolean tensor (representing a scalar).
//! trueSubgraph represents a network subgraph that is executed when condition is evaluated to True.
//! falseSubgraph represents a network subgraph that is executed when condition is evaluated to False.
//!
//! The following constraints apply to If-conditionals:
//! - Both the trueSubgraph and falseSubgraph must be defined.
//! - The number of output tensors in both subgraphs is the same.
//! - The type and shape of each output tensor from true/false subgraphs are the same.
//!
class IIfConditional : public INoCopy
{
public:
    //!
    //! \brief Set the condition tensor for this If-Conditional construct.
    //!
    //! \param condition The condition tensor that will determine which subgraph to execute.
    //!
    //! \p condition tensor must be a 0D execution tensor (scalar) with type DataType::kBOOL.
    //!
    //! \see IConditionLayer
    //!
    IConditionLayer* setCondition(ITensor& condition) noexcept
    {
        return mImpl->setCondition(condition);
    }

    //!
    //! \brief Add an If-conditional output.
    //!
    //! \param trueSubgraphOutput The output of the subgraph executed when the conditional evaluates to true.
    //! \param falseSubgraphOutput The output of the subgraph executed when the conditional evaluates to false.
    //!
    //! Each output layer of an IIfConditional represents a single output of either the true-subgraph or the
    //! false-subgraph of an IIfConditional, depending on which subgraph was executed.
    //!
    //! \see IIfConditionalOutputLayer
    //!
    IIfConditionalOutputLayer* addOutput(ITensor& trueSubgraphOutput, ITensor& falseSubgraphOutput) noexcept
    {
        return mImpl->addOutput(trueSubgraphOutput, falseSubgraphOutput);
    }

    //!
    //! \brief Add an If-conditional input.
    //!
    //! \param input An input to the conditional that can be used by either or both of the conditional’s subgraphs.
    //!
    //! \see IIfConditionalInputLayer
    //!
    IIfConditionalInputLayer* addInput(ITensor& input) noexcept
    {
        return mImpl->addInput(input);
    }

    //!
    //! \brief Set the name of the conditional.
    //!
    //! The name is used in error diagnostics.
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        mImpl->setName(name);
    }

    //!
    //! \brief Return the name of the conditional.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

protected:
    virtual ~IIfConditional() noexcept = default;
    apiv::VIfConditional* mImpl;
};


class IRecurrenceLayer : public ILoopBoundaryLayer
{
public:
    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //
    //! Sets the input tensor for the given index.
    //!
    //! For a recurrence layer, the values 0 and 1 are valid.
    //! The indices are as follows:
    //!
    //! - 0: The initial value of the output tensor. The value must come from outside the loop.
    //! - 1: The next value of the output tensor. The value usually comes from inside the loop, and must have the same
    //! dimensions as input 0.
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    //!
    using ILayer::setInput;

protected:
    virtual ~IRecurrenceLayer() noexcept = default;
    apiv::VRecurrenceLayer* mImpl;
};

//!
//! An ILoopOutputLayer is the sole way to get output from a loop.
//!
//! The first input tensor must be defined inside the loop; the output tensor is outside the loop.
//! The second input tensor, if present, must be defined outside the loop.
//!
//! If getLoopOutput() is kLAST_VALUE, a single input must be provided,
//! and that input must from a IRecurrenceLayer in the same loop.
//!
//! If getLoopOutput() is kCONCATENATE or kREVERSE, a second input must be provided.
//! The second input must be a 0D shape tensor, defined before the loop commences,
//! that specifies the concatenation length of the output.
//!
//! The output tensor has j more dimensions than the input tensor, where
//! j == 0 if getLoopOutput() is kLAST_VALUE
//! j == 1 if getLoopOutput() is kCONCATENATE or kREVERSE.
//!
class ILoopOutputLayer : public ILoopBoundaryLayer
{
public:
    LoopOutput getLoopOutput() const noexcept
    {
        return mImpl->getLoopOutput();
    }

    //!
    //! \brief Set where to insert the contenation axis. Ignored if getLoopOutput() is kLAST_VALUE.
    //!
    //! For example, if the input tensor has dimensions [b,c,d],
    //! and getLoopOutput() is  kCONCATENATE, the output has four dimensions.
    //! Let a be the value of the second input.
    //! setAxis(0) causes the output to have dimensions [a,b,c,d].
    //! setAxis(1) causes the output to have dimensions [b,a,c,d].
    //! setAxis(2) causes the output to have dimensions [b,c,a,d].
    //! setAxis(3) causes the output to have dimensions [b,c,d,a].
    //! Default is axis is 0.
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

    //! Get axis being concatenated over.
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //
    //! Sets the input tensor for the given index. The index must be 0 for a kLAST_VALUE loop output layer.
    //! Loop output layer is converted to a kCONCATENATE or kREVERSE loop output layer by calling setInput with an
    //! index 1. A kCONCATENATE or kREVERSE loop output layer cannot be converted back to a kLAST_VALUE loop output
    //! layer.
    //!
    //! For a kCONCATENATE or kREVERSE loop output layer, the values 0 and 1 are valid.
    //! The indices in the kCONCATENATE or kREVERSE cases are as follows:
    //!
    //! - 0: Contribution to the output tensor.  The contribution must come from inside the loop.
    //! - 1: The concatenation length scalar value, must come from outside the loop, as a 0D Int32 shape tensor.
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    //!
    using ILayer::setInput;

protected:
    virtual ~ILoopOutputLayer() noexcept = default;
    apiv::VLoopOutputLayer* mImpl;
};

class ITripLimitLayer : public ILoopBoundaryLayer
{
public:
    TripLimit getTripLimit() const noexcept
    {
        return mImpl->getTripLimit();
    }

protected:
    virtual ~ITripLimitLayer() noexcept = default;
    apiv::VTripLimitLayer* mImpl;
};

class IIteratorLayer : public ILoopBoundaryLayer
{
public:
    //! Set axis to iterate over.
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

    //! Get axis being iterated over.
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }

    //! For reverse=false, the layer is equivalent to addGather(tensor, I, 0) where I is a
    //! scalar tensor containing the loop iteration number.
    //! For reverse=true, the layer is equivalent to addGather(tensor, M-1-I, 0) where M is the trip count
    //! computed from TripLimits of kind kCOUNT.
    //! The default is reverse=false.
    void setReverse(bool reverse) noexcept
    {
        mImpl->setReverse(reverse);
    }

    //! True if and only if reversing input.
    bool getReverse() const noexcept
    {
        return mImpl->getReverse();
    }

protected:
    virtual ~IIteratorLayer() noexcept = default;
    apiv::VIteratorLayer* mImpl;
};

//!
//! Helper for creating a recurrent subgraph.
//!
//! An ILoop cannot be added to an INetworkDefinition where hasImplicitBatchDimensions() returns true.
//!
class ILoop : public INoCopy
{
public:
    //!
    //! \brief Create a recurrence layer for this loop with initialValue as its first input.
    //!
    //! IRecurrenceLayer requires exactly two inputs.  The 2nd input must be added, via method
    //! IRecurrenceLayer::setInput(1,...) before an Engine can be built.
    //!
    IRecurrenceLayer* addRecurrence(ITensor& initialValue) noexcept
    {
        return mImpl->addRecurrence(initialValue);
    }

    //!
    //! \brief Add a trip-count limiter, based on the given tensor.
    //!
    //! There may be at most one kCOUNT and one kWHILE limiter for a loop.
    //! When both trip limits exist, the loop exits when the
    //! count is reached or condition is falsified.
    //! It is an error to not add at least one trip limiter.
    //!
    //! For kCOUNT, the input tensor must be available before the loop starts.
    //!
    //! For kWHILE, the input tensor must be the output of a subgraph that contains
    //! only layers that are not ITripLimitLayer, IIteratorLayer or ILoopOutputLayer.
    //! Any IRecurrenceLayers in the subgraph must belong to the same loop as the
    //! ITripLimitLayer.  A trivial example of this rule is that the input to the kWHILE
    //! is the output of an IRecurrenceLayer for the same loop.
    //!
    ITripLimitLayer* addTripLimit(ITensor& tensor, TripLimit limit) noexcept
    {
        return mImpl->addTripLimit(tensor, limit);
    }

    //!
    //! \brief Return layer that subscripts tensor by loop iteration.
    //!
    //! For reverse=false, this is equivalent to addGather(tensor, I, 0) where I is a
    //! scalar tensor containing the loop iteration number.
    //! For reverse=true, this is equivalent to addGather(tensor, M-1-I, 0) where M is the trip count
    //! computed from TripLimits of kind kCOUNT.
    //!
    IIteratorLayer* addIterator(ITensor& tensor, int32_t axis = 0, bool reverse = false) noexcept
    {
        return mImpl->addIterator(tensor, axis, reverse);
    }

    //! \brief Make an output for this loop, based on the given tensor.
    //!
    //! axis is the axis for concatenation (if using outputKind of kCONCATENATE or kREVERSE).
    //!
    //! If outputKind is kCONCATENATE or kREVERSE, a second input specifying the
    //! concatenation dimension must be added via method ILoopOutputLayer::setInput.
    //!
    ILoopOutputLayer* addLoopOutput(ITensor& tensor, LoopOutput outputKind, int32_t axis = 0) noexcept
    {
        return mImpl->addLoopOutput(tensor, outputKind, axis);
    }

    //!
    //! \brief Set the name of the loop.
    //!
    //! The name is used in error diagnostics.
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        mImpl->setName(name);
    }

    //!
    //! \brief Return the name of the loop.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

protected:
    virtual ~ILoop() noexcept = default;
    apiv::VLoop* mImpl;
};

//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISelectLayer : public ILayer
{
protected:
    virtual ~ISelectLayer() noexcept = default;
    apiv::VSelectLayer* mImpl;
};

//! \class IAssertionLayer
//!
//! \brief An assertion layer in a network
//!
//! The layer has a single input and no output. The input must be a boolean shape tensor.
//! If any element of the input is provably false at build time, the network is rejected.
//! If any element of the input is false at runtime for the supplied runtime dimensions,
//! an error occurs, much the same as if any other runtime error (e.g. using IShuffleLayer
//! to change the volume of a tensor) is handled.
//!
//! Asserting equality of input dimensions may help the optimizer.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IAssertionLayer : public ILayer
{
public:
    //!
    //! \brief Set the message to print if the assertion fails.
    //!
    //! The name is used in error diagnostics.
    //! This method copies the message string.
    //!
    //! \see getMessage()
    //!
    void setMessage(char const* message) noexcept
    {
        mImpl->setMessage(message);
    }

    //!
    //! \brief Return the assertion message.
    //!
    //! \see setMessage()
    //!
    char const* getMessage() const noexcept
    {
        return mImpl->getMessage();
    }

protected:
    virtual ~IAssertionLayer() noexcept = default;

    apiv::VAssertionLayer* mImpl;
};

//!
//! \enum FillOperation
//!
//! \brief Enumerates the tensor fill operations that may performed by a fill layer.
//!
//! \see IFillLayer
//!
enum class FillOperation : int32_t
{
    kLINSPACE = 0,       //!< Generate evenly spaced numbers over a specified interval.
    kRANDOM_UNIFORM = 1, //!< Generate a tensor with random values drawn from a uniform distribution.
    kRANDOM_NORMAL = 2   //!< Generate a tensor with random values drawn from a normal distribution.
};

//!
//! Maximum number of elements in FillOperation enum.
//!
//! \see FillOperation
//!
template <>
constexpr inline int32_t EnumMax<FillOperation>() noexcept
{
    return 3;
}

//!
//! \brief Generate an output tensor with specified mode.
//!
//! The fill layer has two variants, static and dynamic. Static fill specifies its parameters
//! at layer creation time via Dims and the get/set accessor functions of the IFillLayer.
//! Dynamic fill specifies one or more of its parameters as ITensors, by using ILayer::setInput to add
//! a corresponding input.  The corresponding static parameter is used if an input is missing or null.
//!
//! The shape of the output is specified by the parameter \p Dimension, or if non-null and present,
//! the first input, which must be a 1D Int32 shape tensor. Thus an application can determine if the
//! IFillLayer has a dynamic output shape based on whether it has a non-null first input.
//!
//! Alpha and Beta are treated differently based on the Fill Operation specified. See details in
//! IFillLayer::setAlpha(), IFillLayer::setBeta(), and IFillLayer::setInput().
//!
//! A fill layer can produce a shape tensor if the following restrictions are met:
//!
//! * The FillOperation is kLINSPACE.
//! * The output is an Int32 or Float tensor within the volume limit of a shape tensor.
//! * There is at most one input, and if so, that input is input 0.
//! * If input 0 exists, the length of the output tensor must be computable by constant folding.
//!
//! \see FillOperation
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
class IFillLayer : public ILayer
{
public:
    //!
    //! \brief Set the output tensor's dimensions.
    //!
    //! \param dimensions The output tensor's dimensions.
    //!
    //! If the first input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getDimensions
    //
    void setDimensions(Dims dimensions) noexcept
    {
        mImpl->setDimensions(dimensions);
    }

    //!
    //! \brief Get the output tensor's dimensions.
    //!
    //! \return The output tensor's dimensions, or an invalid Dims structure.
    //!
    //! If the first input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setDimensions
    //!
    Dims getDimensions() const noexcept
    {
        return mImpl->getDimensions();
    }

    //!
    //! \brief Set the fill operation for the layer.
    //!
    //! \see getOperation(), FillOperation
    //!
    void setOperation(FillOperation op) noexcept
    {
        mImpl->setOperation(op);
    }

    //!
    //! \brief Get the fill operation for the layer.
    //!
    //! \see setOperation(), FillOperation
    //!
    FillOperation getOperation() const noexcept
    {
        return mImpl->getOperation();
    }

    //!
    //! \brief Set the alpha parameter.
    //!
    //! \param alpha has different meanings for each operator:
    //!
    //! Operation          | Usage
    //! kLINSPACE          | the start value, defaults to 0.0;
    //! kRANDOM_UNIFORM    | the minimum value, defaults to 0.0;
    //! kRANDOM_NORMAL     | the mean of the normal distribution, default is 0.0;
    //!
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getAlpha
    //
    void setAlpha(double alpha) noexcept
    {
        mImpl->setAlpha(alpha);
    }

    //!
    //! \brief Get the value of alpha parameter.
    //!
    //! \return A double value of alpha.
    //!
    //! If the second input is present and non-null,
    //! this function returns -1.0.
    //!
    //! \see setAlpha
    //!
    double getAlpha() const noexcept
    {
        return mImpl->getAlpha();
    }

    //!
    //! \brief Set the beta parameter.
    //!
    //! \param beta has different meanings for each operator:
    //!
    //! Operation          | Usage
    //! kLINSPACE          | the delta value, defaults to 1.0;
    //! kRANDOM_UNIFORM    | the maximal value, defaults to 1.0;
    //! kRANDOM_NORMAL     | the standard deviation of the normal distribution, default is 1.0;
    //!
    //! If a third input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getBeta
    //!
    void setBeta(double beta) noexcept
    {
        mImpl->setBeta(beta);
    }

    //!
    //! \brief Get the value of beta parameter.
    //!
    //! \return A double value of beta.
    //!
    //! If the third input is present and non-null,
    //! this function returns -1.0.
    //!
    //! \see setBeta
    //!
    double getBeta() const noexcept
    {
        return mImpl->getBeta();
    }

    //!
    //! \brief replace an input of this layer with a specific tensor.
    //!
    //! \param index the index of the input to set.
    //! \param tensor the new input tensor
    //!
    //! Indices for kLINSPACE are described as:
    //!
    //! - 0: Shape tensor, represents the output tensor's dimensions.
    //! - 1: Start, a scalar, represents the start value.
    //! - 2: Delta, a 1D tensor, length equals to shape tensor's nbDims, represents the delta value for each dimension.
    //!
    //! Indices for kRANDOM_UNIFORM are described as:
    //!
    //! - 0: Shape tensor, represents the output tensor's dimensions.
    //! - 1: Minimum, a scalar, represents the minimum random value.
    //! - 2: Maximum, a scalar, represents the maximal random value.
    //!
    //! Indices for kRANDOM_NORMAL are described as:
    //!
    //! - 0: Shape tensor, represents the output tensor's dimensions.
    //! - 1: Mean, a scalar, represents the mean of the normal distribution,.
    //! - 2: Scale, a scalar, represents the standard deviation of the normal distribution.
    //!
    //! Using the corresponding setter resets the input to null.
    //!
    //! If either inputs 1 or 2, is non-null, then both must be non-null and have the same data type.
    //!
    //! If this function is called for an index greater or equal to getNbInputs(),
    //! then afterwards getNbInputs() returns index + 1, and any missing intervening
    //! inputs are set to null.
    //!
    using ILayer::setInput;

protected:
    virtual ~IFillLayer() noexcept = default;
    apiv::VFillLayer* mImpl;
};

//!
//! \class IQuantizeLayer
//!
//! \brief A Quantize layer in a network definition.
//!
//! This layer accepts a floating-point data input tensor, and uses the scale and zeroPt inputs to
//! quantize the data to an 8-bit signed integer according to:
//! \p output = clamp(round(\p input / \p scale) + \p zeroPt)
//!
//! Rounding type is rounding-to-nearest ties-to-even (https://en.wikipedia.org/wiki/Rounding#Round_half_to_even).
//! Clamping is in the range [-128, 127].
//!
//! The first input (index 0) is the tensor to be quantized.
//! The second (index 1) and third (index 2) are the scale and zero point respectively.
//! Each of \p scale and \p zeroPt must be either a scalar, or a 1D tensor.
//!
//! The \p zeroPt tensor is optional, and if not set, will be assumed to be zero.  Its data type must be
//! DataType::kINT8. \p zeroPt must only contain zero-valued coefficients, because only symmetric quantization is
//! supported.
//! The \p scale value must be either a scalar for per-tensor quantization, or a 1D tensor for per-channel
//! quantization. All \p scale coefficients must have positive values.  The size of the 1-D \p scale tensor must match
//! the size of the quantization axis. The size of the \p scale must match the size of the \p zeroPt.
//!
//! The subgraph which terminates with the \p scale tensor must be a build-time constant.  The same restrictions apply
//! to the \p zeroPt.
//! The output type, if constrained, must be constrained to DataType::kINT8. The input type, if constrained, must be
//! constrained to DataType::kFLOAT or DataType::kHALF.
//! The output size is the same as the input size. The quantization axis is in reference to the input tensor's
//! dimensions.
//!
//! IQuantizeLayer only supports DataType::kFLOAT precision and will default to this precision during instantiation.
//! IQuantizeLayer only supports DataType::kINT8 output.
//!
//! As an example of the operation of this layer, imagine a 4D NCHW activation input which can be quantized using a
//! single scale coefficient (referred to as per-tensor quantization):
//!     For each n in N:
//!         For each c in C:
//!             For each h in H:
//!                 For each w in W:
//!                     output[n,c,h,w] = clamp(round(\p input[n,c,h,w] / \p scale) + \p zeroPt)
//!
//! Per-channel quantization is supported only for weight inputs. Thus, Activations cannot be quantized per-channel.
//! As an example of per-channel operation, imagine a 4D KCRS weights input and K (dimension 0) as the quantization
//! axis. The scale is an array of coefficients, and must have the same size as the quantization axis.
//!     For each k in K:
//!         For each c in C:
//!             For each r in R:
//!                 For each s in S:
//!                     output[k,c,r,s] = clamp(round(\p input[k,c,r,s] / \p scale[k]) + \p zeroPt[k])
//!
//! \note Only symmetric quantization is supported.
//! \note Currently the only allowed build-time constant \p scale and \zeroPt subgraphs are:
//! 1. Constant -> Quantize
//! 2. Constant -> Cast -> Quantize
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IQuantizeLayer : public ILayer
{
public:
    //!
    //! \brief Get the quantization axis.
    //!
    //! \return axis parameter set by setAxis().
    //! The return value is the index of the quantization axis in the input tensor's dimensions.
    //! A value of -1 indicates per-tensor quantization.
    //! The default value is -1.
    //!
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }
    //!
    //! \brief Set the quantization axis.
    //!
    //! Set the index of the quantization axis (with reference to the input tensor's dimensions).
    //! The axis must be a valid axis if the scale tensor has more than one coefficient.
    //! The axis value will be ignored if the scale tensor has exactly one coefficient (per-tensor quantization).
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

protected:
    virtual ~IQuantizeLayer() noexcept = default;
    apiv::VQuantizeLayer* mImpl;
};

//!
//! \class IDequantizeLayer
//!
//! \brief A Dequantize layer in a network definition.
//!
//! This layer accepts a signed 8-bit integer input tensor, and uses the configured scale and zeroPt inputs to
//! dequantize the input according to:
//! \p output = (\p input - \p zeroPt) * \p scale
//!
//! The first input (index 0) is the tensor to be quantized.
//! The second (index 1) and third (index 2) are the scale and zero point respectively.
//! Each of \p scale and \p zeroPt must be either a scalar, or a 1D tensor.
//!
//! The \p zeroPt tensor is optional, and if not set, will be assumed to be zero.  Its data type must be
//! DataType::kINT8. \p zeroPt must only contain zero-valued coefficients, because only symmetric quantization is
//! supported.
//! The \p scale value must be either a scalar for per-tensor quantization, or a 1D tensor for per-channel
//! quantization. All \p scale coefficients must have positive values.  The size of the 1-D \p scale tensor must match
//! the size of the quantization axis. The size of the \p scale must match the size of the \p zeroPt.
//!
//! The subgraph which terminates with the \p scale tensor must be a build-time constant.  The same restrictions apply
//! to the \p zeroPt.
//! The output type, if constrained, must be constrained to DataType::kFLOAT or DataType::kHALF. The input type, if
//! constrained, must be constrained to DataType::kINT8. The output size is the same as the input size. The quantization
//! axis is in reference to the input tensor's dimensions.
//!
//! IDequantizeLayer only supports DataType::kINT8 precision and will default to this precision during instantiation.
//! IDequantizeLayer only supports DataType::kFLOAT or DataType::kHALF output.
//!
//! As an example of the operation of this layer, imagine a 4D NCHW activation input which can be quantized using a
//! single scale coefficient (referred to as per-tensor quantization):
//!     For each n in N:
//!         For each c in C:
//!             For each h in H:
//!                 For each w in W:
//!                     output[n,c,h,w] = (\p input[n,c,h,w] - \p zeroPt) * \p scale
//!
//! Per-channel dequantization is supported only for input that is rooted at an IConstantLayer (i.e. weights).
//! Activations cannot be quantized per-channel. As an example of per-channel operation, imagine a 4D KCRS weights input
//! and K (dimension 0) as the quantization axis. The scale is an array of coefficients, which is the same size as the
//! quantization axis.
//!     For each k in K:
//!         For each c in C:
//!             For each r in R:
//!                 For each s in S:
//!                     output[k,c,r,s] = (\p input[k,c,r,s] - \p zeroPt[k]) * \p scale[k]
//!
//! \note Only symmetric quantization is supported.
//! \note Currently the only allowed build-time constant \p scale and \zeroPt subgraphs are:
//! 1. Constant -> Quantize
//! 2. Constant -> Cast -> Quantize
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IDequantizeLayer : public ILayer
{
public:
    //!
    //! \brief Get the quantization axis.
    //!
    //! \return axis parameter set by setAxis().
    //! The return value is the index of the quantization axis in the input tensor's dimensions.
    //! A value of -1 indicates per-tensor quantization.
    //! The default value is -1.
    //!
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }
    //!
    //! \brief Set the quantization axis.
    //!
    //! Set the index of the quantization axis (with reference to the input tensor's dimensions).
    //! The axis must be a valid axis if the scale tensor has more than one coefficient.
    //! The axis value will be ignored if the scale tensor has exactly one coefficient (per-tensor quantization).
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

protected:
    virtual ~IDequantizeLayer() noexcept = default;
    apiv::VDequantizeLayer* mImpl;
};

//! \class IEinsumLayer
//!
//! \brief An Einsum layer in a network
//!
//! This layer implements a summation over the elements of the inputs along dimensions specified by the equation
//! parameter, based on the Einstein summation convention.
//! The layer can have one or more inputs of rank >= 0. All the inputs must have type DataType::kFLOAT
//! or DataType::kHALF, not necessarily the same. There is one output of type DataType::kFLOAT.
//! The shape of the output tensor is determined by the equation.
//!
//! The equation specifies ASCII lower-case letters for each dimension in the inputs in the same order as the
//! dimensions, separated by comma for each input. The dimensions labeled with the same subscript must match or be
//! broadcastable. Repeated subscript labels in one input take the diagonal. Repeating a label across multiple inputs
//! means that those axes will be multiplied. Omitting a label from the output means values along those axes will be
//! summed. In implicit mode, the indices which appear once in the expression will be part of the output in increasing
//! alphabetical order. In explicit mode, the output can be controlled by specifying output subscript labels by adding
//! an arrow (‘->’) followed by subscripts for the output.
//! For example, “ij,jk->ik” is equivalent to “ij,jk”.
//! Ellipsis (‘...’) can be used in place of subscripts to broadcast the dimensions.
//! See the TensorRT Developer Guide for more details on equation syntax.
//!
//! Many common operations can be expressed using the Einsum equation.
//! For example:
//! Matrix Transpose:             ij->ji
//! Sum:                          ij->
//! Matrix-Matrix Multiplication: ik,kj->ij
//! Dot Product:                  i,i->
//! Matrix-Vector Multiplication: ik,k->i
//! Batch Matrix Multiplication:  ijk,ikl->ijl
//! Batch Diagonal:               ...ii->...i
//!
//! \note TensorRT does not support ellipsis, diagonal operations or more than two inputs for Einsum.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IEinsumLayer : public ILayer
{
public:
    //!
    //! \brief Set the equation.
    //! The equation is a comma-separated list of subscript labels, where each label refers to a
    //! dimension of the corresponding tensor.
    //!
    //! \return true if the equation was syntactically valid and set successfully, false otherwise.
    //!
    //! \see setEquation()
    //!
    bool setEquation(char const* equation) noexcept
    {
        return mImpl->setEquation(equation);
    }

    //!
    //! \brief Return the equation.
    //!
    //! \see setEquation()
    //!
    char const* getEquation() const noexcept
    {
        return mImpl->getEquation();
    }

protected:
    virtual ~IEinsumLayer() noexcept = default;
    apiv::VEinsumLayer* mImpl;
};

//!
//! \brief Control form of IScatterLayer
//!
//! \see IScatterLayer
//!
enum class ScatterMode : int32_t
{
    kELEMENT = 0, //!< Similar to ONNX ScatterElements
    kND = 1,      //!< Similar to ONNX ScatterND
};

//!
//! Maximum number of elements in ScatterMode enum.
//!
//! \see ScatterMode
//!
template <>
constexpr inline int32_t EnumMax<ScatterMode>() noexcept
{
    return 2;
}

//!
//! \class IScatterLayer
//!
//! \brief A scatter layer in a network definition. Supports several kinds of scattering.
//!
//! The Scatter layer has three input tensors: Data, Indices, and Updates, one output tensor
//! Output, and a scatter mode. When kELEMENT mode is used an optional axis parameter is available.
//! * Data is a tensor of rank r >= 1 that stores the values to be duplicated in Output.
//! * Indices is a tensor of rank q that determines which locations in Output to write new
//!   values to. Constraints on the rank of q depend on the mode:
//!       ScatterMode::kND: q >= 1
//!       ScatterMode::kELEMENT: q must be the same as r
//! * Updates is atensor of rank s >=1 that provides the data
//!   to write to Output specified by its corresponding location in Index. Constraints the rank of Updates depend on the
//!   mode:
//!       ScatterMode::kND: s = r + q - shape(Indices)[-1] - 1
//!       Scattermode::kELEMENT: s = q = r
//! * Output is a tensor with the same dimensions as Data that stores the resulting values of the
//!   transformation. It must not be a shape tensor.
//! The types of Data, Update, and Output shall be the same, and Indices shall be DataType::kINT32.
//!
//! The output is computed by copying the data, and then updating elements of it based on indices.
//! How Indices are interpreted depends upon the ScatterMode.
//!
//! ScatterMode::kND
//!
//!     The indices are interpreted as a tensor of rank q-1 of indexing tuples.
//!     The axis parameter is ignored.
//!
//!     Given that data dims are {d_0,...,d_{r-1}} and indices dims are {i_0,...,i_{q-1}},
//!     define k = indices[q-1], it follows that updates dims are {i_0,...,i_{q-2},d_k,...,d_{r-1}}
//!     The updating can be computed by:
//!         foreach slice in indices[i_0,...i_{q-2}]
//!             output[indices[slice]] = updates[slice]
//!
//! ScatterMode::kELEMENT
//!
//!     Here "axis" denotes the result of getAxis().
//!
//!     For each element X of indices:
//!         Let J denote a sequence for the subscripts of X
//!         Let K = sequence J with element [axis] replaced by X
//!         output[K] = updates[J]
//!
//!     For example, if indices has dimensions [N,C,H,W] and axis is 2, then the updates happen as:
//!
//!         for n in [0,n)
//!             for c in [0,n)
//!                 for h in [0,n)
//!                     for w in [0,n)
//!                         output[n,c,indices[n,c,h,w],w] = updates[n,c,h,w]]
//!
//! Writes to the same output element cause undefined behavior.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IScatterLayer : public ILayer
{
public:
    //!
    //! \brief Set the scatter mode.
    //!
    //! \see getMode()
    //!
    void setMode(ScatterMode mode) noexcept
    {
        mImpl->setMode(mode);
    }

    //!
    //! \brief Get the scatter mode.
    //!
    //! \see setMode()
    //!
    ScatterMode getMode() const noexcept
    {
        return mImpl->getMode();
    }

    //!
    //! \brief Set the axis used by ScatterMode::kELEMENTS.
    //!
    //! The axis defaults to 0.
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

    //!
    //! \brief Get the axis.
    //!
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }

protected:
    apiv::VScatterLayer* mImpl;
    virtual ~IScatterLayer() noexcept = default;
}; // class IScatterLayer

//!
//! \class IOneHotLayer
//!
//! \brief A OneHot layer in a network definition.
//!
//! The OneHot layer has three input tensors: Indices, Values, and Depth, one output tensor:
//! Output, and an axis attribute.
//! * Indices is an Int32 tensor that determines which locations in Output to set as on_value.
//! * Values is a two-element (rank=1) tensor that consists of [off_value, on_value]
//! * Depth is an Int32 shape tensor of rank 0, which contains the depth (number of classes) of the one-hot encoding.
//!   The depth tensor must be a build-time constant, and its value should be positive.
//! * Output is a tensor with rank = rank(indices)+1, where the added dimension contains the one-hot encoding.
//!   The data types of Output is equal to the Values data type.
//! * Axis is a scaler specifying to which dimension of the output one-hot encoding is added.
//!   Axis defaults to -1, that is the new dimension in the output is its final dimension.
//!   Valid range for axis is -rank(indices)-1 <= axis <= rank(indices).
//!
//! The output is computed by copying off_values to all output elements, then setting on_value on the indices
//! specified by the indices tensor.
//! when axis = 0:
//! output[indices[i, j, k], i, j, k] = on_value for all i, j, k and off_value otherwise.
//!
//! when axis = -1:
//! output[i, j, k, indices[i, j, k]] = on_value for all i, j, k and off_value otherwise.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IOneHotLayer : public ILayer
{
public:
    //!
    //! \brief Set the axis parameter.
    //!
    //! \see IOneHotLayer
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

    //!
    //! \brief Get the value of the axis parameter.
    //!
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }

protected:
    apiv::VOneHotLayer* mImpl;
};

//! \class IGridSampleLayer
//!
//! \brief A GridSample layer in a network definition.
//!
//! This layer uses an input tensor and a grid tensor to produce an interpolated output tensor.
//! The input and grid tensors must be shape tensors of rank 4. The only supported SampleMode
//! values are SampleMode::kCLAMP, SampleMode::kFILL, and SampleMode::kREFLECT.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IGridSampleLayer : public ILayer
{
public:
    //!
    //! \brief Set the grid sample interpolation mode.
    //!
    //! \see getInterpolationMode()
    //!
    void setInterpolationMode(InterpolationMode mode) noexcept
    {
        mImpl->setInterpolationMode(mode);
    }

    //!
    //! \brief Get the grid sample interpolation mode.
    //!
    //! \see setInterpolationMode()
    //!
    //! \return The value specified by setInterpolationMode, or InterpolationMode::kLINEAR otherwise.
    //!
    InterpolationMode getInterpolationMode() const noexcept
    {
        return mImpl->getInterpolationMode();
    }

    //!
    //! \brief Set the align corners mode.
    //!
    //! \see getAlignCorners()
    //!
    void setAlignCorners(bool alignCorners) noexcept
    {
        mImpl->setAlignCorners(alignCorners);
    }

    //!
    //! \brief Get the align corners mode.
    //!
    //! \see setAlignCorners()
    //!
    //! \return The value specified by setAlignCorners(), or false otherwise.
    //!
    bool getAlignCorners() const noexcept
    {
        return mImpl->getAlignCorners();
    }

    //!
    //! \brief Set the sample mode.
    //!
    //! \see getSampleMode()
    //!
    //! \return true if layer's sample mode was set to mode, false otherwise.
    //!
    bool setSampleMode(SampleMode mode) noexcept
    {
        return mImpl->setSampleMode(mode);
    }

    //!
    //! \brief Get the sample mode.
    //!
    //! \see setSampleMode()
    //!
    //! \returns the value specified by a successful call to setSampleMode(), or SampleMode::kFILL otherwise.
    //!
    SampleMode getSampleMode() const noexcept
    {
        return mImpl->getSampleMode();
    }

protected:
    apiv::VGridSampleLayer* mImpl;
    virtual ~IGridSampleLayer() noexcept = default;
}; // class IGridSampleLayer

//!
//! \brief Representation of bounding box data used for the Boxes input tensor in INMSLayer
//!
//! \see INMSLayer
//!
enum class BoundingBoxFormat : int32_t
{
    //! (x1, y1, x2, y2) where (x1, y1) and (x2, y2) are any pair of diagonal corners
    kCORNER_PAIRS = 0,
    //! (x_center, y_center, width, height) where (x_center, y_center) is the center point of the box
    kCENTER_SIZES = 1
};

//!
//! Maximum number of elements in BoundingBoxFormat enum.
//!
//! \see BoundingBoxFormat
//!
template <>
constexpr inline int32_t EnumMax<BoundingBoxFormat>() noexcept
{
    return 2;
}

//!
//! \class INMSLayer
//!
//! \brief A non-maximum suppression layer in a network definition.
//!
//! The NMS algorithm iterates through a set of bounding boxes and their confidence scores, in decreasing
//! order of score. Boxes are selected if their score is above a given threshold, and their
//! intersection-over-union (IoU) with previously selected boxes is less than or equal to a given threshold.
//! This layer implements NMS per batch item and per class.
//!
//! For each batch item, the ordering of candidate bounding boxes with the same score is unspecified.
//!
//! The layer has the following inputs, in order of input index:
//!
//! * Boxes contains the input bounding boxes. It is a linear tensor of type kFLOAT or kHALF. It has
//!   shape [batchSize, numInputBoundingBoxes, numClasses, 4] if the boxes are per class, or
//!   [batchSize, numInputBoundingBoxes, 4] if the same boxes are to be used for each class.
//! * Scores contains the per-box scores. It is a linear tensor of the same type as Boxes. It has shape
//!   [batchSize, numInputBoundingBoxes, numClasses].
//! * MaxOutputBoxesPerClass is the maximum number of output boxes per batch item per class.
//!   It is a scalar (0D tensor) of type kINT32.
//! * IoUThreshold is the maximum IoU for selected boxes. It is a scalar (0D tensor) of type kFLOAT in the range
//!   [0.0f, 1.0f]. It is an optional input with default 0.0f.
//! * ScoreThreshold is the value that a box score must exceed in order to be selected. It is a scalar (0D tensor) of type kFLOAT. It is an optional
//!   input with default 0.0f.
//!
//! The layer has the following outputs, in order of output index:
//!
//! * SelectedIndices contains the indices of the selected boxes. It is a linear tensor of type kINT32. It has shape
//!   [NumOutputBoxes, 3]. Each row contains a (batchIndex, classIndex, boxIndex) tuple.
//!   The output boxes are sorted in order of increasing batchIndex and then in order of decreasing score within each batchIndex.
//!   For each batchIndex, the ordering of output boxes with the same score is unspecified.
//!   If MaxOutputBoxesPerClass is a constant input, the maximum number of output boxes is
//!   batchSize * numClasses * min(numInputBoundingBoxes, MaxOutputBoxesPerClass).
//!   Otherwise, the maximum number of output boxes is batchSize * numClasses * numInputBoundingBoxes.
//!   The maximum number of output boxes is used to determine the upper-bound on allocated memory for this output tensor.
//! * NumOutputBoxes is the number of output boxes in SelectedIndices. It is a scalar (0D tensor) of type kINT32.
//!
//! \warning There is a hardware-dependent limit K such that only the K highest scoring boxes in each batch item
//! will be considered for selection. The value of K is 2000 for SM 5.3 and 6.2 devices, and 5000 otherwise.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class INMSLayer : public ILayer
{
public:
    //!
    //! \brief Set the bounding box format parameter for the layer.
    //!
    //! The default value for the bounding box format parameter is kCORNER_PAIRS.
    //!
    //! \see BoundingBoxFormat
    //!
    //! \see getBoundingBoxFormat()
    //!
    void setBoundingBoxFormat(BoundingBoxFormat fmt) noexcept
    {
        mImpl->setBoundingBoxFormat(fmt);
    }

    //!
    //! \brief Get the bounding box format parameter for the layer.
    //!
    //! \see BoundingBoxFormat
    //!
    //! \see setBoundingBoxFormat()
    //!
    BoundingBoxFormat getBoundingBoxFormat() const noexcept
    {
        return mImpl->getBoundingBoxFormat();
    }

    //!
    //! \brief Set the TopK box limit parameter for the layer.
    //!
    //! The TopK box limit is the maximum number of filtered boxes considered for selection per batch item.
    //! The default value for the TopK box limit parameter is 2000 for SM 5.3 and 6.2 devices, and 5000 otherwise.
    //! The TopK box limit must be less than or equal to {2000 for SM 5.3 and 6.2 devices, 5000 otherwise}.
    //!
    //! \see getTopKBoxLimit()
    //!
    void setTopKBoxLimit(int32_t limit) noexcept
    {
        mImpl->setTopKBoxLimit(limit);
    }

    //!
    //! \brief Get the TopK box limit parameter for the layer.
    //!
    //! \see setTopKBoxLimit()
    //!
    int32_t getTopKBoxLimit() const noexcept
    {
        return mImpl->getTopKBoxLimit();
    }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! The indices are as follows:
    //!
    //! - 0: The required Boxes tensor.
    //! - 1: The required Scores tensor.
    //! - 2: The required MaxOutputBoxesPerClass tensor.
    //! - 3: The optional IoUThreshold tensor.
    //! - 4: The optional ScoreThreshold tensor.
    //!
    //! If this function is called for an index greater or equal to getNbInputs(),
    //! then afterwards getNbInputs() returns index + 1, and any missing intervening
    //! inputs are set to null. Note that only optional inputs can be missing.
    //!
    using ILayer::setInput;

protected:
    apiv::VNMSLayer* mImpl;
    virtual ~INMSLayer() noexcept = default;
}; // class INMSLayer

//!
//! \class INetworkDefinition
//!
//! \brief A network definition for input to the builder.
//!
//! A network definition defines the structure of the network, and combined with a IBuilderConfig, is built
//! into an engine using an IBuilder. An INetworkDefinition can either have an implicit batch dimensions, specified
//! at runtime, or all dimensions explicit, full dims mode, in the network definition. The former mode, i.e. the
//! implicit batch size mode, has been deprecated. The function hasImplicitBatchDimension() can be used to query the
//! mode of the network.
//!
//! A network with implicit batch dimensions returns the dimensions of a layer without the implicit dimension,
//! and instead the batch is specified at execute/enqueue time. If the network has all dimensions specified, then
//! the first dimension follows elementwise broadcast rules: if it is 1 for some inputs and is some value N for all
//! other inputs, then the first dimension of each output is N, and the inputs with 1 for the first dimension are
//! broadcast. Having divergent batch sizes across inputs to a layer is not supported.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class INetworkDefinition : public INoCopy
{
public:
    virtual ~INetworkDefinition() noexcept = default;

    //!
    //! \brief Add an input tensor to the network.
    //!
    //! The name of the input tensor is used to find the index into the buffer array for an engine built from
    //! the network. The volume must be less than 2^31 elements.
    //!
    //! For networks with an implicit batch dimension, this volume includes the batch dimension with its length set
    //! to the maximum batch size. For networks with all explicit dimensions and with wildcard dimensions, the volume
    //! is based on the maxima specified by an IOptimizationProfile.Dimensions are normally non-negative integers. The
    //! exception is that in networks with all explicit dimensions, -1 can be used as a wildcard for a dimension to
    //! be specified at runtime. Input tensors with such a wildcard must have a corresponding entry in the
    //! IOptimizationProfiles indicating the permitted extrema, and the input dimensions must be set by
    //! IExecutionContext::setBindingDimensions. Different IExecutionContext instances can have different dimensions.
    //! Wildcard dimensions are only supported for EngineCapability::kSTANDARD. They are not
    //! supported in safety contexts. DLA does not support Wildcard dimensions.
    //!
    //! Tensor dimensions are specified independent of format.  For example, if a
    //! tensor is formatted in "NHWC" or a vectorized format, the dimensions are
    //! still specified in the order{N, C, H, W}. For 2D images with a channel
    //! dimension, the last three dimensions are always {C,H,W}. For 3D images
    //! with a channel dimension, the last four dimensions are always {C,D,H,W}.
    //!
    //! \param name The name of the tensor.
    //! \param type The type of the data held in the tensor.
    //! \param dimensions The dimensions of the tensor.
    //!
    //! \warning It is an error to specify a wildcard value on a dimension that is determined by trained parameters.
    //!
    //! \warning If run on DLA with explicit dimensions, only leading dimension can be a wildcard. And provided profile
    //! must have same minimum, optimum, and maximum dimensions.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see ITensor
    //!
    //! \return The new tensor or nullptr if there is an error.
    //!
    ITensor* addInput(char const* name, DataType type, Dims dimensions) noexcept
    {
        return mImpl->addInput(name, type, dimensions);
    }

    //!
    //! \brief Mark a tensor as a network output.
    //!
    //! \param tensor The tensor to mark as an output tensor.
    //!
    //! \warning It is an error to mark a network input as an output.
    //! \warning It is an error to mark a tensor inside an ILoop or an
    //!          IIfConditional as an output.
    //!
    void markOutput(ITensor& tensor) noexcept
    {
        mImpl->markOutput(tensor);
    }

    //!
    //! \brief Add a convolution layer to the network.
    //!
    //! \param input The input tensor to the convolution.
    //! \param nbOutputMaps The number of output feature maps for the convolution.
    //! \param kernelSize The HW-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The bias weights for the convolution. Weights{} represents no bias.
    //!
    //! \see IConvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new convolution layer, or nullptr if it could not be created.
    //!
    //! \deprecated Superseded by addConvolutionNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED IConvolutionLayer* addConvolution(
        ITensor& input, int32_t nbOutputMaps, DimsHW kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
    {
        return mImpl->addConvolution(input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
    }

    //!
    //! \brief Add a fully connected layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputs The number of outputs of the layer.
    //! \param kernelWeights The kernel weights for the fully connected layer.
    //! \param biasWeights The bias weights for the fully connected layer. Weights{} represents no bias.
    //!
    //! \see IFullyConnectedLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new fully connected layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 8.4. Superseded by addMatrixMultiply().
    //!
    TRT_DEPRECATED IFullyConnectedLayer* addFullyConnected(
        ITensor& input, int32_t nbOutputs, Weights kernelWeights, Weights biasWeights) noexcept
    {
        return mImpl->addFullyConnected(input, nbOutputs, kernelWeights, biasWeights);
    }

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
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new activation layer, or nullptr if it could not be created.
    //!
    IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept
    {
        return mImpl->addActivation(input, type);
    }

    //!
    //! \brief Add a pooling layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of pooling to apply.
    //! \param windowSize The size of the pooling window.
    //!
    //! \see IPoolingLayer PoolingType
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new pooling layer, or nullptr if it could not be created.
    //!
    //! \deprecated Superseded by addPoolingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED IPoolingLayer* addPooling(ITensor& input, PoolingType type, DimsHW windowSize) noexcept
    {
        return mImpl->addPooling(input, type, windowSize);
    }

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
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new LRN layer, or nullptr if it could not be created.
    //!
    ILRNLayer* addLRN(ITensor& input, int32_t window, float alpha, float beta, float k) noexcept
    {
        return mImpl->addLRN(input, window, alpha, beta, k);
    }

    //!
    //! \brief Add a Scale layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!              This tensor is required to have a minimum of 3 dimensions in implicit batch mode
    //!              and a minimum of 4 dimensions in explicit batch mode.
    //! \param mode The scaling mode.
    //! \param shift The shift value.
    //! \param scale The scale value.
    //! \param power The power value.
    //!
    //! If the weights are available, then the size of weights are dependent on the ScaleMode.
    //! For ::kUNIFORM, the number of weights equals 1.
    //! For ::kCHANNEL, the number of weights equals the channel dimension.
    //! For ::kELEMENTWISE, the number of weights equals the product of the last three dimensions of the input.
    //!
    //! \see addScaleNd
    //! \see IScaleLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new Scale layer, or nullptr if it could not be created.
    //!
    IScaleLayer* addScale(ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) noexcept
    {
        return mImpl->addScale(input, mode, shift, scale, power);
    }

    //!
    //! \brief Add a SoftMax layer to the network.
    //!
    //! \see ISoftMaxLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new SoftMax layer, or nullptr if it could not be created.
    //!
    ISoftMaxLayer* addSoftMax(ITensor& input) noexcept
    {
        return mImpl->addSoftMax(input);
    }

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
    //! \warning All tensors must have the same dimensions except along the concatenation axis.
    //!
    IConcatenationLayer* addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept
    {
        return mImpl->addConcatenation(inputs, nbInputs);
    }

    //!
    //! \brief Add a deconvolution layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputMaps The number of output feature maps.
    //! \param kernelSize The HW-dimensions of the deconvolution kernel.
    //! \param kernelWeights The kernel weights for the deconvolution.
    //! \param biasWeights The bias weights for the deconvolution. Weights{} represents no bias.
    //!
    //! \see IDeconvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new deconvolution layer, or nullptr if it could not be created.
    //!
    //! \deprecated Superseded by addDeconvolutionNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED IDeconvolutionLayer* addDeconvolution(
        ITensor& input, int32_t nbOutputMaps, DimsHW kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
    {
        return mImpl->addDeconvolution(input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
    }

    //!
    //! \brief Add an elementwise layer to the network.
    //!
    //! \param input1 The first input tensor to the layer.
    //! \param input2 The second input tensor to the layer.
    //! \param op The binary operation that the layer applies.
    //!
    //! The input tensors must have the same rank and compatible type.
    //! Two types are compatible if they are the same type or are both in the set {kFLOAT, kHALF}.
    //! For each dimension, their lengths must match, or one of them must be one.
    //! In the latter case, the tensor is broadcast along that axis.
    //!
    //! The output tensor has the same rank as the inputs.
    //! For each dimension, its length is the maximum of the lengths of the
    //! corresponding input dimension.
    //!
    //! The inputs are shape tensors if the output is a shape tensor.
    //!
    //! \see IElementWiseLayer
    //!
    //! \return The new elementwise layer, or nullptr if it could not be created.
    //!
    IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept
    {
        return mImpl->addElementWise(input1, input2, op);
    }

    //!
    //! \brief Add a unary layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The operation to apply.
    //!
    //! \see IUnaryLayer
    //!
    //! Generally the input must have a floating-point type (or kINT8 as a quantized float),
    //! except for the following operations:
    //! * kSIGN accepts a floating-point or Int32 tensor.
    //! * kNOT requires a Bool tensor.
    //!
    //! The input is a shape tensor if the output is a shape tensor.
    //!
    //! \return The new unary layer, or nullptr if it could not be created
    //!
    IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) noexcept
    {
        return mImpl->addUnary(input, operation);
    }

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
    //! \deprecated Superseded by addPaddingNd. Deprecated prior to TensorRT 8.0 and will be removed in 9.0
    //!
    TRT_DEPRECATED IPaddingLayer* addPadding(ITensor& input, DimsHW prePadding, DimsHW postPadding) noexcept
    {
        return mImpl->addPadding(input, prePadding, postPadding);
    }

    //!
    //! \brief Add a shuffle layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShuffleLayer
    //!
    //! \return The new shuffle layer, or nullptr if it could not be created.
    //!
    IShuffleLayer* addShuffle(ITensor& input) noexcept
    {
        return mImpl->addShuffle(input);
    }

    //!
    //! \brief Add a OneHot layer to the network.
    //!
    //! \param indices - tensor containing indices where on_value should be set.
    //! \param values - a 2-element tensor, consisting of [off_value, on_value].
    //! \param depth - tensor containing the width of the added one-hot dimension.
    //! \param axis - the axis to add the one-hot encoding to.
    //!
    //! \see IOneHotLayer
    //!
    //! \return The new OneHot layer, or nullptr if it could not be created.
    //!
    IOneHotLayer* addOneHot(ITensor& indices, ITensor& values, ITensor& depth, int32_t axis) noexcept
    {
        return mImpl->addOneHot(indices, values, depth, axis);
    }

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! \return The number of layers in the network.
    //!
    //! \see getLayer()
    //!
    int32_t getNbLayers() const noexcept
    {
        return mImpl->getNbLayers();
    }

    //!
    //! \brief Get the layer specified by the given index.
    //!
    //! \param index The index of the layer.
    //!
    //! \return The layer, or nullptr if the index is out of range.
    //!
    //! \see getNbLayers()
    //!
    ILayer* getLayer(int32_t index) const noexcept
    {
        return mImpl->getLayer(index);
    }

    //!
    //! \brief Get the number of inputs in the network.
    //!
    //! \return The number of inputs in the network.
    //!
    //! \see getInput()
    //!
    int32_t getNbInputs() const noexcept
    {
        return mImpl->getNbInputs();
    }

    //!
    //! \brief Get the input tensor specified by the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range.
    //!
    //! \note adding inputs invalidates indexing here
    //!
    //! \see getNbInputs()
    //!
    ITensor* getInput(int32_t index) const noexcept
    {
        return mImpl->getInput(index);
    }

    //!
    //! \brief Get the number of outputs in the network.
    //!
    //! The outputs include those marked by markOutput or markOutputForShapes.
    //!
    //! \return The number of outputs in the network.
    //!
    //! \see getOutput()
    //!
    int32_t getNbOutputs() const noexcept
    {
        return mImpl->getNbOutputs();
    }

    //!
    //! \brief Get the output tensor specified by the given index.
    //!
    //! \param index The index of the output tensor.
    //!
    //! \return The output tensor, or nullptr if the index is out of range.
    //!
    //! \note adding inputs invalidates indexing here
    //!
    //! \see getNbOutputs()
    //!
    ITensor* getOutput(int32_t index) const noexcept
    {
        return mImpl->getOutput(index);
    }

    //!
    //! \brief Destroy this INetworkDefinition object.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by `delete`.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

    //!
    //! \brief Add a reduce layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The reduction operation to perform.
    //! \param reduceAxes The reduction dimensions.
    //!        The bit in position i of bitmask reduceAxes corresponds to explicit dimension i if result.
    //!        E.g., the least significant bit corresponds to the first explicit dimension and the next to least
    //!        significant bit corresponds to the second explicit dimension.
    //!
    //! \param keepDimensions The boolean that specifies whether or not to keep the reduced dimensions in the
    //! output of the layer.
    //!
    //! The reduce layer works by performing an operation specified by \p operation to reduce the tensor \p input
    //! across the axes specified by \p reduceAxes.
    //!
    //! \see IReduceLayer
    //!
    //! \warning If output is an Int32 shape tensor, ReduceOperation::kAVG is unsupported.
    //!
    //! \return The new reduce layer, or nullptr if it could not be created.
    //!
    IReduceLayer* addReduce(
        ITensor& input, ReduceOperation operation, uint32_t reduceAxes, bool keepDimensions) noexcept
    {
        return mImpl->addReduce(input, operation, reduceAxes, keepDimensions);
    }

    //!
    //! \brief Add a TopK layer to the network.
    //!
    //! The TopK layer has two outputs of the same dimensions. The first contains data values,
    //! the second contains index positions for the values. Output values are sorted, largest first
    //! for operation kMAX and smallest first for operation kMIN.
    //!
    //! Currently only values of K up to 3840 are supported.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \param op Operation to perform.
    //!
    //! \param k Number of elements to keep.
    //!
    //! \param reduceAxes The reduction dimensions.
    //!        The bit in position i of bitmask reduceAxes corresponds to explicit dimension i of the result.
    //!        E.g., the least significant bit corresponds to the first explicit dimension and the next to least
    //!        significant bit corresponds to the second explicit dimension.
    //!
    //!        Currently reduceAxes must specify exactly one dimension, and it must be one of the last four dimensions.
    //!
    //! \see ITopKLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new TopK layer, or nullptr if it could not be created.
    //!
    ITopKLayer* addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept
    {
        return mImpl->addTopK(input, op, k, reduceAxes);
    }

    //!
    //! \brief Add gather with mode GatherMode::kDEFAULT and specified axis and nbElementWiseDims=0.
    //!
    //! \param data The tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param axis The axis in the data tensor to gather on.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it could not be created.
    //!
    IGatherLayer* addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept
    {
        return mImpl->addGather(data, indices, axis);
    }

    //!
    //! \brief Add gather with specified mode, axis=0 and nbElementWiseDims=0.
    //!
    //! \param data The tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param mode The gather mode.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it could not be created.
    //!
    IGatherLayer* addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) noexcept
    {
        return mImpl->addGatherV2(data, indices, mode);
    }

    //!
    //! \brief Add a RaggedSoftMax layer to the network.
    //!
    //! \param input The ZxS input tensor.
    //! \param bounds The Zx1 bounds tensor.
    //!
    //! \see IRaggedSoftMaxLayer
    //!
    //! \warning The bounds tensor cannot have the last dimension be the wildcard character.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new RaggedSoftMax layer, or nullptr if it could not be created.
    //!
    IRaggedSoftMaxLayer* addRaggedSoftMax(ITensor& input, ITensor& bounds) noexcept
    {
        return mImpl->addRaggedSoftMax(input, bounds);
    }

    //!
    //! \brief Add a MatrixMultiply layer to the network.
    //!
    //! \param input0 The first input tensor (commonly A).
    //! \param op0 The operation to apply to input0.
    //! \param input1 The second input tensor (commonly B).
    //! \param op1 The operation to apply to input1.
    //!
    //! The inputs are shape tensors if the output is a shape tensor.
    //!
    //! \see IMatrixMultiplyLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new matrix multiply layer, or nullptr if it could not be created.
    //!
    IMatrixMultiplyLayer* addMatrixMultiply(
        ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) noexcept
    {
        return mImpl->addMatrixMultiply(input0, op0, input1, op1);
    }

    //!
    //! \brief Add a nonzero layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see INonZeroLayer
    //!
    //! \return The new nonzero layer, or nullptr if it could be created.
    //!
    INonZeroLayer* addNonZero(ITensor& input) noexcept
    {
        return mImpl->addNonZero(input);
    }

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
    //! Otherwise the output is a tensor of real values and the output type will be
    //! follow TensorRT's normal precision rules.
    //!
    //! If tensors in the network have an implicit batch dimension, the constant
    //! is broadcast over that dimension.
    //!
    //! If a wildcard dimension is used, the volume of the runtime dimensions must equal
    //! the number of weights specified.
    //!
    //! \warning DataType::kUINT8 not supported.
    //!
    IConstantLayer* addConstant(Dims dimensions, Weights weights) noexcept
    {
        return mImpl->addConstant(dimensions, weights);
    }

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
    //! \deprecated Deprecated prior to TensorRT 8.0 and will be removed in 9.0. Superseded by
    //! INetworkDefinition::addLoop().
    //!
    //! \warning RNN inputs do not support wildcard dimensions or explicit batch size networks.
    //! \warning Int32 tensors are not valid input tensors, only for sequence lengths.
    //!
    //! \return The new RNN layer, or nullptr if it could not be created.
    //!
    TRT_DEPRECATED IRNNv2Layer* addRNNv2(
        ITensor& input, int32_t layerCount, int32_t hiddenSize, int32_t maxSeqLen, RNNOperation op) noexcept
    {
        return mImpl->addRNNv2(input, layerCount, hiddenSize, maxSeqLen, op);
    }

    //!
    //! \brief Add an identity layer.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IIdentityLayer
    //!
    //! \return The new identity layer, or nullptr if it could not be created.
    //!
    IIdentityLayer* addIdentity(ITensor& input) noexcept
    {
        return mImpl->addIdentity(input);
    }

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
    void removeTensor(ITensor& tensor) noexcept
    {
        mImpl->removeTensor(tensor);
    }

    //!
    //! \brief unmark a tensor as a network output.
    //!
    //! \param tensor The tensor to unmark as an output tensor.
    //!
    //! see markOutput()
    //!
    void unmarkOutput(ITensor& tensor) noexcept
    {
        mImpl->unmarkOutput(tensor);
    }

    //!
    //! \brief Add a plugin layer to the network using the IPluginV2 interface.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginV2Layer
    //!
    //! \warning Dimension wildcard are only supported with IPluginV2DynamicExt or IPluginV2IOExt plugins.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    IPluginV2Layer* addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept
    {
        return mImpl->addPluginV2(inputs, nbInputs, plugin);
    }

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
    ISliceLayer* addSlice(ITensor& input, Dims start, Dims size, Dims stride) noexcept
    {
        return mImpl->addSlice(input, start, size, stride);
    }

    //!
    //! \brief Sets the name of the network.
    //!
    //! \param name The name to assign to this network.
    //!
    //! Set the name of the network so that it can be associated with a built
    //! engine. The \p name must be a null-terminated C-style string.
    //! TensorRT makes no use of this string except storing it as part of the engine
    //! so that it may be retrieved at runtime.
    //! A name unique to the builder will be generated by default.
    //!
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see INetworkDefinition::getName(), ISafeCudaEngine::getName()
    //!
    //! \return none
    //!
    void setName(char const* name) noexcept
    {
        mImpl->setName(name);
    }

    //!
    //! \brief Returns the name associated with the network.
    //!
    //! The memory pointed to by getName() is owned by the INetworkDefinition object.
    //!
    //! \see INetworkDefinition::setName()
    //!
    //! \return A null-terminated C-style string representing the name of the network.
    //!
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Add a shape layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShapeLayer
    //!
    //! \warning addShape is only supported when hasImplicitBatchDimensions is false.
    //!
    //! \warning input to addShape cannot contain wildcard dimension values.
    //!
    //! \return The new shape layer, or nullptr if it could not be created.
    //!
    IShapeLayer* addShape(ITensor& input) noexcept
    {
        return mImpl->addShape(input);
    }

    //!
    //! \brief Query whether the network was created with an implicit batch dimension.
    //!
    //! \return True if tensors have implicit batch dimension, false otherwise.
    //!
    //! This is a network-wide property. Either all tensors in the network
    //! have an implicit batch dimension or none of them do.
    //!
    //! hasImplicitBatchDimension() is true if and only if this INetworkDefinition
    //! was created with createNetworkV2() without NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag.
    //!
    //! \see createNetworkV2
    //!
    bool hasImplicitBatchDimension() const noexcept
    {
        return mImpl->hasImplicitBatchDimension();
    }

    //!
    //! \brief Enable tensor's value to be computed by IExecutionContext::getShapeBinding.
    //!
    //! \return True if successful, false if tensor is already marked as an output.
    //!
    //! The tensor must be of type DataType::kINT32 and have no more than one dimension.
    //!
    //! \warning The tensor must have dimensions that can be determined to be constants at build time.
    //!
    //! \warning It is an error to mark a network input as a shape output.
    //!
    //! \see isShapeBinding(), getShapeBinding()
    //!
    bool markOutputForShapes(ITensor& tensor) noexcept
    {
        return mImpl->markOutputForShapes(tensor);
    }

    //!
    //! \brief Undo markOutputForShapes.
    //!
    //! \warning inputs to addShape cannot contain wildcard dimension values.
    //!
    //! \return True if successful, false if tensor is not marked as an output.
    //!
    bool unmarkOutputForShapes(ITensor& tensor) noexcept
    {
        return mImpl->unmarkOutputForShapes(tensor);
    }

    //!
    //! \brief Add a parametric ReLU layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param slope The slope tensor to the layer. This tensor should be unidirectionally broadcastable
    //!        to the input tensor.
    //!
    //! \see IParametricReLULayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new parametric ReLU layer, or nullptr if it could not be created.
    //!
    IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept
    {
        return mImpl->addParametricReLU(input, slope);
    }

    //!
    //! \brief Add a multi-dimension convolution layer to the network.
    //!
    //! \param input The input tensor to the convolution.
    //! \param nbOutputMaps The number of output feature maps for the convolution.
    //! \param kernelSize The multi-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The bias weights for the convolution. Weights{} represents no bias.
    //!
    //! \see IConvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning Only 2D or 3D convolution is supported.
    //!
    //! \return The new convolution layer, or nullptr if it could not be created.
    //!
    IConvolutionLayer* addConvolutionNd(
        ITensor& input, int32_t nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
    {
        return mImpl->addConvolutionNd(input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
    }

    //!
    //! \brief Add a multi-dimension pooling layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of pooling to apply.
    //! \param windowSize The size of the pooling window.
    //!
    //! \see IPoolingLayer PoolingType
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning Only 2D or 3D pooling is supported.
    //!
    //! \return The new pooling layer, or nullptr if it could not be created.
    //!
    IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims windowSize) noexcept
    {
        return mImpl->addPoolingNd(input, type, windowSize);
    }

    //!
    //! \brief Add a multi-dimension deconvolution layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputMaps The number of output feature maps.
    //! \param kernelSize The multi-dimensions of the deconvolution kernel.
    //! \param kernelWeights The kernel weights for the deconvolution.
    //! \param biasWeights The bias weights for the deconvolution. Weights{} represents no bias.
    //!
    //! \see IDeconvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning Only 2D or 3D deconvolution is supported.
    //
    //! \return The new deconvolution layer, or nullptr if it could not be created.
    //!
    IDeconvolutionLayer* addDeconvolutionNd(
        ITensor& input, int32_t nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
    {
        return mImpl->addDeconvolutionNd(input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
    }

    //!
    //! \brief Add a multi-dimension scale layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param mode The scaling mode.
    //! \param shift The shift value.
    //! \param scale The scale value.
    //! \param power The power value.
    //! \param channelAxis The channel axis.
    //!
    //! If the weights are available, then the size of weights are dependent on the ScaleMode.
    //! For ::kUNIFORM, the number of weights equals 1.
    //! For ::kCHANNEL, the number of weights equals the channel dimension.
    //! For ::kELEMENTWISE, the number of weights equals the product of all input dimensions at channelAxis and beyond.
    //!
    //! For example, if the inputs dimensions are [A,B,C,D,E,F], and channelAxis=2:
    //! For ::kUNIFORM, the number of weights is equal to 1.
    //! For ::kCHANNEL, the number of weights is C.
    //! For ::kELEMENTWISE, the number of weights is C*D*E*F.
    //!
    //! channelAxis can also be set explicitly using setChannelAxis().
    //!
    //! \see IScaleLayer
    //! \see setChannelAxis()
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning Only 2D or 3D scale is supported.
    //!
    //! \return The new Scale layer, or nullptr if it could not be created.
    //!
    IScaleLayer* addScaleNd(
        ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power, int32_t channelAxis) noexcept
    {
        return mImpl->addScaleNd(input, mode, shift, scale, power, channelAxis);
    }

    //! \brief Add a resize layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IResizeLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new resize layer, or nullptr if it could not be created.
    //!
    IResizeLayer* addResize(ITensor& input) noexcept
    {
        return mImpl->addResize(input);
    }

    //!
    //! \brief True if network is an explicit precision network
    //!
    //! \deprecated Deprecated in TensorRT 8.0.
    //!
    //! \see createNetworkV2
    //!
    //! \return True if network has explicit precision, false otherwise.
    //!
    TRT_DEPRECATED bool hasExplicitPrecision() const noexcept
    {
        return mImpl->hasExplicitPrecision();
    }

    //!
    //! \brief Add a loop to the network.
    //!
    //! An ILoop provides a way to specify a recurrent subgraph.
    //!
    //! \return Pointer to ILoop that can be used to add loop boundary layers for the loop,
    //!         or nullptr if network has an implicit batch dimension or this version
    //!         of TensorRT does not support loops.
    //!
    //! The network must not have an implicit batch dimension.
    //!
    ILoop* addLoop() noexcept
    {
        return mImpl->addLoop();
    }

    //! \brief Add a select layer to the network.
    //!
    //! \param condition The condition tensor to the layer. Must have type DataType::kBOOL.
    //! \param thenInput The "then" input tensor to the layer.
    //! \param elseInput The "else" input tensor to the layer.
    //!
    //! All three input tensors must have the same rank, and along each axis
    //! must have the same length or a length of one. If the length is one, the tensor
    //! is broadcast along that axis. The output tensor has the dimensions of the inputs AFTER
    //! the broadcast rule is applied. For example, given:
    //!
    //!    dimensions of condition:  [1,1,5,9]
    //!    dimensions of thenInput:  [1,1,5,9]
    //!    dimensions of elseInput:  [1,3,1,9]
    //!
    //! the output dimensions are [1,3,5,9], and the output contents are defined by:
    //!
    //!      output[0,i,j,k] = condition[0,0,j,k] ? thenInput[0,0,j,k] : elseInput[0,i,0,k]
    //!
    //! The output dimensions are not necessarily the max of the input dimensions if any input
    //! is an empty tensor. For example, if in the preceding example, 5 is changed to 0:
    //!
    //!    dimensions of condition:  [1,1,0,9]
    //!    dimensions of thenInput:  [1,1,0,9]
    //!    dimensions of elseInput:  [1,3,1,9]
    //!
    //! then the output dimensions are [1,3,0,9].
    //!
    //! The network must not have an implicit batch dimension.
    //!
    //! The inputs are shape tensors if the output is a shape tensor.
    //!
    //! \see ISelectLayer
    //!
    //! \return The new select layer, or nullptr if it could not be created.
    ISelectLayer* addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept
    {
        return mImpl->addSelect(condition, thenInput, elseInput);
    }

    //!
    //! \brief Add an assertion layer to the network.
    //!
    //! \param condition The input tensor to the layer.
    //! \param message A message to print if the assertion fails.
    //!
    //! \see IAssertionLayer
    //!
    //! \return The new assertion layer, or nullptr if it could not be created.
    //!
    //! The input tensor must be a boolean shape tensor.
    //!
    IAssertionLayer* addAssertion(ITensor& condition, char const* message) noexcept
    {
        return mImpl->addAssertion(condition, message);
    }

    //! \brief Add a fill layer to the network.
    //!
    //! \param dimensions The output tensor dimensions.
    //! \param op The fill operation that the layer applies.
    //!
    //! \warning For FillOperation::kLINSPACE, dimensions.nbDims must be 1.
    //!
    //! This layer is non-deterministic across subsequent calls as the same inputs will produce different
    //! output tensors if \p op is either FillOperation::kRANDOM_UNIFORM or FillOperation::kRANDOM_NORMAL
    //! due to random state being shared across calls. The output tensors generated are determinstic when
    //! starting from the same initial state.
    //!
    //! The network must not have an implicit batch dimension.
    //!
    //! \see IFillLayer
    //!
    //! \return The new fill layer, or nullptr if it could not be created.
    //!
    IFillLayer* addFill(Dims dimensions, FillOperation op) noexcept
    {
        return mImpl->addFill(dimensions, op);
    }

    //! \brief Add a padding layer to the network. Only 2D padding is currently supported.
    //!
    //! \param input The input tensor to the layer.
    //! \param prePadding The padding to apply to the start of the tensor.
    //! \param postPadding The padding to apply to the end of the tensor.
    //!
    //! \see IPaddingLayer
    //!
    //! \return The new padding layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by addSlice().
    //!
    TRT_DEPRECATED IPaddingLayer* addPaddingNd(ITensor& input, Dims prePadding, Dims postPadding) noexcept
    {
        return mImpl->addPaddingNd(input, prePadding, postPadding);
    }

    //! \brief Associate a name with all current uses of the given weights.
    //!
    //! The name must be set after the Weights are used in the network.
    //! Lookup is associative. The name applies to all Weights with matching
    //! type, value pointer, and count. If Weights with a matching value
    //! pointer, but different type or count exists in the network, an
    //! error message is issued, the name is rejected, and return false.
    //! If the name has already been used for other weights,
    //! return false. A nullptr causes the weights to become unnamed,
    //! i.e. clears any previous name.
    //!
    //! \param weights The weights to be named.
    //! \param name The name to associate with the weights.
    //!
    //! \return true on success.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setWeightsName(Weights weights, char const* name) noexcept
    {
        return mImpl->setWeightsName(weights, name);
    }

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
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class.
    //! A nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Add a dequantization layer to the network.
    //!
    //! \param input The input tensor to be quantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IDequantizeLayer
    //!
    //! \p input tensor data type must be DataType::kFLOAT.
    //! \p scale tensor data type must be DataType::kFLOAT. The subgraph which terminates with the \p scale tensor must
    //! be a build-time constant.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale) noexcept
    {
        return mImpl->addDequantize(input, scale);
    }

    //!
    //! \brief Add a Scatter layer to the network with specified mode and axis=0.
    //!
    //! \param input The input tensor to be updated with additional values.
    //! \param indices indices of the elements to be updated.
    //! \param updates values to be used for updates.
    //!
    //! \see IScatterLayer
    //!
    //! \p input tensor data type must be DataType::kFLOAT.
    //! \p indices tensor data type must be DataType::kINT32.
    //! \p updates tensor data type must be DataType::kFLOAT.
    //!
    //! \return The new Scatter layer, or nullptr if it could not be created.
    //!
    IScatterLayer* addScatter(ITensor& data, ITensor& indices, ITensor& updates, ScatterMode mode) noexcept
    {
        return mImpl->addScatter(data, indices, updates, mode);
    }

    //!
    //! \brief Add a quantization layer to the network.
    //!
    //! \param input The input tensor to be quantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IQuantizeLayer
    //!
    //! \p input tensor data type must be DataType::kFLOAT.
    //! \p scale tensor data type must be DataType::kFLOAT. The subgraph which terminates with the \p scale tensor must
    //! be a build-time constant.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale) noexcept
    {
        return mImpl->addQuantize(input, scale);
    }

    //!
    //! \brief Add an If-conditional layer to the network.
    //!
    //! An IIfConditional provides a way to conditionally execute parts of the network.
    //!
    //! \see IIfConditional
    //!
    //! \return The new conditional layer, or nullptr if network has an implicit batch dimension
    //!         or this version of TensorRT does not support conditional execution.
    //!
    IIfConditional* addIfConditional() noexcept
    {
        return mImpl->addIfConditional();
    }

    //! \brief Add an Einsum layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param equation The equation of the layer
    //! \see IEinsumLayer
    //!
    //! \return The new Einsum layer, or nullptr if it could not be created.
    //!
    IEinsumLayer* addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept
    {
        return mImpl->addEinsum(inputs, nbInputs, equation);
    }

    //! \brief Add a GridSample layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param grid The grid tensor to the layer.
    //! \see IGridSampleLayer
    //!
    //! Creates a GridSample layer with a InterpolationMode::kLINEAR, unaligned corners,
    //! and SampleMode::kFILL for 4d-shape input tensors.
    //!
    //! \return The new GridSample layer, or nullptr if it could not be created.
    //!
    IGridSampleLayer* addGridSample(ITensor& input, ITensor& grid) noexcept
    {
        return mImpl->addGridSample(input, grid);
    }

    //!
    //! \brief Add a non-maximum suppression layer to the network.
    //!
    //! \param boxes The input boxes tensor to the layer.
    //!
    //! \param scores The input scores tensor to the layer.
    //!
    //! \param maxOutputBoxesPerClass The input maxOutputBoxesPerClass tensor to the layer.
    //!
    //! \see INMSLayer
    //!
    //! \return The new NMS layer, or nullptr if it could not be created.
    //!
    INMSLayer* addNMS(ITensor& boxes, ITensor& scores, ITensor& maxOutputBoxesPerClass) noexcept
    {
        return mImpl->addNMS(boxes, scores, maxOutputBoxesPerClass);
    }

protected:
    apiv::VNetworkDefinition* mImpl;
};

//!
//! enum CalibrationAlgoType
//!
//! \brief Version of calibration algorithm to use.
//!
enum class CalibrationAlgoType : int32_t
{
    kLEGACY_CALIBRATION = 0,
    kENTROPY_CALIBRATION = 1,
    kENTROPY_CALIBRATION_2 = 2,
    kMINMAX_CALIBRATION = 3,
};

//!
//! Maximum number of elements in CalibrationAlgoType enum.
//!
//! \see DataType
//!
template <>
constexpr inline int32_t EnumMax<CalibrationAlgoType>() noexcept
{
    return 4;
}

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
    virtual int32_t getBatchSize() const noexcept = 0;

    //!
    //! \brief Get a batch of input for calibration.
    //!
    //! The batch size of the input must match the batch size returned by getBatchSize().
    //!
    //! \param bindings An array of pointers to device memory that must be updated to point to device memory
    //! containing each network input data.
    //! \param names The names of the network input for each pointer in the binding array.
    //! \param nbBindings The number of pointers in the bindings array.
    //! \return False if there are no more batches for calibration.
    //!
    //! \see getBatchSize()
    //!
    virtual bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept = 0;

    //!
    //! \brief Load a calibration cache.
    //!
    //! Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on
    //! subsequent builds of the network. The cache includes the regression cutoff and quantile values used to generate
    //! it, and will not be used if these do not batch the settings of the current calibrator. However, the network
    //! should also be recalibrated if its structure changes, or the input data set changes, and it is the
    //! responsibility of the application to ensure this.
    //!
    //! \param length The length of the cached data, that should be set by the called function. If there is no data,
    //! this should be zero.
    //!
    //! \return A pointer to the cache, or nullptr if there is no data.
    //!
    virtual void const* readCalibrationCache(std::size_t& length) noexcept = 0;

    //!
    //! \brief Save a calibration cache.
    //!
    //! \param ptr A pointer to the data to cache.
    //! \param length The length in bytes of the data to cache.
    //!
    //! \see readCalibrationCache()
    //!
    virtual void writeCalibrationCache(void const* ptr, std::size_t length) noexcept = 0;

    //!
    //! \brief Get the algorithm used by this calibrator.
    //!
    //! \return The algorithm used by the calibrator.
    //!
    virtual CalibrationAlgoType getAlgorithm() noexcept = 0;

    virtual ~IInt8Calibrator() noexcept = default;
};

//!
//! Entropy calibrator. This is the Legacy Entropy calibrator. It is less complicated than the legacy calibrator and
//! produces better results.
//!
class IInt8EntropyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the entropy calibrator.
    //!
    CalibrationAlgoType getAlgorithm() noexcept override
    {
        return CalibrationAlgoType::kENTROPY_CALIBRATION;
    }

    virtual ~IInt8EntropyCalibrator() noexcept = default;
};

//!
//! Entropy calibrator 2. This is the preferred calibrator. This is the required calibrator for DLA, as it supports per
//! activation tensor scaling.
//!
class IInt8EntropyCalibrator2 : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the entropy calibrator 2.
    //!
    CalibrationAlgoType getAlgorithm() noexcept override
    {
        return CalibrationAlgoType::kENTROPY_CALIBRATION_2;
    }

    virtual ~IInt8EntropyCalibrator2() noexcept = default;
};

//!
//! MinMax Calibrator. It supports per activation tensor scaling.
//!
class IInt8MinMaxCalibrator : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the MinMax Calibrator.
    //!
    CalibrationAlgoType getAlgorithm() noexcept override
    {
        return CalibrationAlgoType::kMINMAX_CALIBRATION;
    }

    virtual ~IInt8MinMaxCalibrator() noexcept = default;
};

//!
//! Legacy calibrator left for backward compatibility with TensorRT 2.0. This calibrator requires user parameterization,
//! and is provided as a fallback option if the other calibrators yield poor results.
//!
class IInt8LegacyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the legacy calibrator.
    //!
    CalibrationAlgoType getAlgorithm() noexcept override
    {
        return CalibrationAlgoType::kLEGACY_CALIBRATION;
    }

    //!
    //! \brief The quantile (between 0 and 1) that will be used to select the region maximum when the quantile method
    //! is in use.
    //!
    //! See the user guide for more details on how the quantile is used.
    //!
    virtual double getQuantile() const noexcept = 0;

    //!
    //! \brief The fraction (between 0 and 1) of the maximum used to define the regression cutoff when using regression
    //! to determine the region maximum.
    //!
    //! See the user guide for more details on how the regression cutoff is used
    //!
    virtual double getRegressionCutoff() const noexcept = 0;

    //!
    //! \brief Load a histogram.
    //!
    //! Histogram generation is potentially expensive, so it can be useful to generate the histograms once, then use
    //! them when exploring the space of calibrations. The histograms should be regenerated if the network structure
    //! changes, or the input data set changes, and it is the responsibility of the application to ensure this.
    //!
    //! \param length The length of the cached data, that should be set by the called function. If there is no data,
    //! this should be zero.
    //!
    //! \return A pointer to the cache, or nullptr if there is no data.
    //!
    virtual void const* readHistogramCache(std::size_t& length) noexcept = 0;

    //!
    //! \brief Save a histogram cache.
    //!
    //! \param ptr A pointer to the data to cache.
    //! \param length The length in bytes of the data to cache.
    //!
    //! \see readHistogramCache()
    //!
    virtual void writeHistogramCache(void const* ptr, std::size_t length) noexcept = 0;

    virtual ~IInt8LegacyCalibrator() noexcept = default;
};

//!
//! \class IAlgorithmIOInfo
//!
//! \brief Carries information about input or output of the algorithm.
//!        IAlgorithmIOInfo for all the input and output along with IAlgorithmVariant denotes the variation of algorithm
//!        and can be used to select or reproduce an algorithm using IAlgorithmSelector::selectAlgorithms().
//! \see IAlgorithmVariant, IAlgorithm, IAlgorithmSelector::selectAlgorithms()
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IAlgorithmIOInfo : public INoCopy
{
public:
    //!
    //! \brief Return TensorFormat of the input/output of algorithm.
    //!
    TensorFormat getTensorFormat() const noexcept
    {
        return mImpl->getTensorFormat();
    }

    //!
    //! \brief Return DataType of the input/output of algorithm.
    //!
    DataType getDataType() const noexcept
    {
        return mImpl->getDataType();
    }

    //!
    //! \brief Return strides of the input/output tensor of algorithm.
    //!
    Dims getStrides() const noexcept
    {
        return mImpl->getStrides();
    }

protected:
    virtual ~IAlgorithmIOInfo() noexcept = default;
    apiv::VAlgorithmIOInfo* mImpl;
};

//!
//! \class IAlgorithmVariant
//!
//! \brief provides a unique 128-bit identifier, which along with the input and output information
//!        denotes the variation of algorithm and can be used to select or reproduce an algorithm,
//!        using IAlgorithmSelector::selectAlgorithms()
//! \see IAlgorithmIOInfo, IAlgorithm, IAlgorithmSelector::selectAlgorithms()
//! \note A single implementation can have multiple tactics.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IAlgorithmVariant : public INoCopy
{
public:
    //!
    //! \brief Return implementation of the algorithm.
    //!
    int64_t getImplementation() const noexcept
    {
        return mImpl->getImplementation();
    }

    //!
    //! \brief Return tactic of the algorithm.
    //!
    int64_t getTactic() const noexcept
    {
        return mImpl->getTactic();
    }

protected:
    virtual ~IAlgorithmVariant() noexcept = default;
    apiv::VAlgorithmVariant* mImpl;
};

//!
//! \class IAlgorithmContext
//!
//! \brief Describes the context and requirements, that could be fulfilled by one or more instances of IAlgorithm.
//! \see IAlgorithm
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IAlgorithmContext : public INoCopy
{
public:
    //!
    //! \brief Return name of the algorithm node.
    //! This is a unique identifier for the IAlgorithmContext.
    //!
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for input or output tensor.
    //! \param index Index of the input or output of the algorithm. Incremental numbers assigned to indices of inputs
    //!              and the outputs.
    //! \param select Which of the minimum, optimum, or maximum dimensions to be queried.
    //!
    Dims getDimensions(int32_t index, OptProfileSelector select) const noexcept
    {
        return mImpl->getDimensions(index, select);
    }

    //!
    //! \brief Return number of inputs of the algorithm.
    //!
    int32_t getNbInputs() const noexcept
    {
        return mImpl->getNbInputs();
    }

    //!
    //! \brief Return number of outputs of the algorithm.
    //!
    int32_t getNbOutputs() const noexcept
    {
        return mImpl->getNbOutputs();
    }

protected:
    virtual ~IAlgorithmContext() noexcept = default;
    apiv::VAlgorithmContext* mImpl;
};

//!
//! \class IAlgorithm
//! \brief Describes a variation of execution of a layer.
//!        An algorithm is represented by IAlgorithmVariant and the IAlgorithmIOInfo for each of its inputs and outputs.
//!        An algorithm can be selected or reproduced using AlgorithmSelector::selectAlgorithms()."
//! \see IAlgorithmIOInfo, IAlgorithmVariant, IAlgorithmSelector::selectAlgorithms()
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IAlgorithm : public INoCopy
{
public:
    //!
    //! \brief Returns the format of an Algorithm input or output. Algorithm inputs are incrementally numbered first,
    //!        followed by algorithm outputs.
    //! \param index Index of the input or output of the algorithm. Incremental numbers assigned to indices of inputs
    //!              and the outputs.
    //!
    //! \return a reference to IAlgorithmIOInfo specified by index or the first algorithm if index is out of range.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by IAlgorithm::getAlgorithmIOInfoByIndex().
    //!
    TRT_DEPRECATED IAlgorithmIOInfo const& getAlgorithmIOInfo(int32_t index) const noexcept
    {
        return mImpl->getAlgorithmIOInfo(index);
    }

    //!
    //! \brief Returns the algorithm variant.
    //!
    IAlgorithmVariant const& getAlgorithmVariant() const noexcept
    {
        return mImpl->getAlgorithmVariant();
    }

    //!
    //! \brief The time in milliseconds to execute the algorithm.
    //!
    float getTimingMSec() const noexcept
    {
        return mImpl->getTimingMSec();
    }

    //!
    //! \brief The size of the GPU temporary memory in bytes which the algorithm uses at execution time.
    //!
    std::size_t getWorkspaceSize() const noexcept
    {
        return mImpl->getWorkspaceSize();
    }

    //!
    //! \brief Returns the format of an Algorithm input or output. Algorithm inputs are incrementally numbered first,
    //!        followed by algorithm outputs.
    //! \param index Index of the input or output of the algorithm. Incremental numbers assigned to indices of inputs
    //!              and the outputs.
    //!
    //! \return a pointer to a IAlgorithmIOInfo interface or nullptr if index is out of range.
    //!
    IAlgorithmIOInfo const* getAlgorithmIOInfoByIndex(int32_t index) const noexcept
    {
        return mImpl->getAlgorithmIOInfoByIndex(index);
    }

protected:
    virtual ~IAlgorithm() noexcept = default;
    apiv::VAlgorithm* mImpl;
}; // IAlgorithm

//!
//! \class IAlgorithmSelector
//!
//! \brief Interface implemented by application for selecting and reporting algorithms of a layer provided by the
//!        builder.
//! \note A layer in context of algorithm selection may be different from ILayer in INetworkDefiniton.
//!       For example, an algorithm might be implementing a conglomeration of multiple ILayers in INetworkDefinition.
//!
class IAlgorithmSelector
{
public:
    //!
    //! \brief Select Algorithms for a layer from the given list of algorithm choices.
    //!
    //! \return The number of choices selected from [0, nbChoices-1].
    //! \param context The context for which the algorithm choices are valid.
    //! \param choices The list of algorithm choices to select for implementation of this layer.
    //! \param nbChoices Number of algorithm choices.
    //! \param selection The user writes indices of selected choices in to selection buffer which is of size nbChoices.
    //!
    //! \note TensorRT uses its default algorithm selection to choose from the list provided.
    //!       If return value is 0, TensorRT's default algorithm selection is used unless
    //!       BuilderFlag::kREJECT_EMPTY_ALGORITHMS (or the deprecated BuilderFlag::kSTRICT_TYPES) is set.
    //!       The list of choices is valid only for this specific algorithm context.
    //!
    virtual int32_t selectAlgorithms(IAlgorithmContext const& context, IAlgorithm const* const* choices,
        int32_t nbChoices, int32_t* selection) noexcept = 0;
    //!
    //! \brief Called by TensorRT to report choices it made.
    //!
    //! \note For a given optimization profile, this call comes after all calls to selectAlgorithms.
    //! algoChoices[i] is the choice that TensorRT made for algoContexts[i], for i in [0, nbAlgorithms-1]
    //!
    //! \param algoContexts The list of all algorithm contexts.
    //! \param algoChoices The list of algorithm choices made by TensorRT
    //! \param nbAlgorithms The size of algoContexts as well as algoChoices.
    //!
    virtual void reportAlgorithms(IAlgorithmContext const* const* algoContexts, IAlgorithm const* const* algoChoices,
        int32_t nbAlgorithms) noexcept = 0;

    virtual ~IAlgorithmSelector() noexcept = default;
};

//!
//! \brief Represents one or more QuantizationFlag values using binary OR
//! operations.
//!
//! \see IBuilderConfig::getQuantizationFlags(), IBuilderConfig::setQuantizationFlags()
//!
using QuantizationFlags = uint32_t;

//!
//! \enum QuantizationFlag
//!
//! \brief List of valid flags for quantizing the network to int8
//!
//! \see IBuilderConfig::setQuantizationFlag(), IBuilderConfig::getQuantizationFlag()
//!
enum class QuantizationFlag : int32_t
{
    //! Run int8 calibration pass before layer fusion. Only valid for IInt8LegacyCalibrator and
    //! IInt8EntropyCalibrator. The builder always runs the int8 calibration pass before layer fusion for
    //! IInt8MinMaxCalibrator and IInt8EntropyCalibrator2. Disabled by default.
    kCALIBRATE_BEFORE_FUSION = 0
};

//!
//! Maximum number of quantization flags in QuantizationFlag enum.
//!
//! \see QuantizationFlag
//!
template <>
constexpr inline int32_t EnumMax<QuantizationFlag>() noexcept
{
    return 1;
}

//!
//! \brief Represents one or more QuantizationFlag values using binary OR
//! operations, e.g., 1U << BuilderFlag::kFP16 | 1U << BuilderFlag::kDEBUG.
//!
//! \see IBuilderConfig::getFlags(), ITensor::setFlags(),
//!
using BuilderFlags = uint32_t;

//!
//! \enum BuilderFlag
//!
//! \brief List of valid modes that the builder can enable when creating an engine from a network definition.
//!
//! \see IBuilderConfig::setFlag(), IBuilderConfig::getFlag()
//!
enum class BuilderFlag : int32_t
{
    kFP16 = 0,         //!< Enable FP16 layer selection, with FP32 fallback.
    kINT8 = 1,         //!< Enable Int8 layer selection, with FP32 fallback with FP16 fallback if kFP16 also specified.
    kDEBUG = 2,        //!< Enable debugging of layers via synchronizing after every layer.
    kGPU_FALLBACK = 3, //!< Enable layers marked to execute on GPU if layer cannot execute on DLA.

    //! Legacy flag with effect similar to setting all of these three flags:
    //!
    //! * kPREFER_PRECISION_CONSTRAINTS
    //! * kDIRECT_IO
    //! * kREJECT_EMPTY_ALGORITHMS
    //!
    //! except that if the direct I/O requirement cannot be met and kDIRECT_IO was not explicitly set,
    //! instead of the build failing, the build falls back as if kDIRECT_IO was not set.
    //!
    //! \deprecated Deprecated in TensorRT 8.2.
    //!
    kSTRICT_TYPES TRT_DEPRECATED_ENUM = 4,

    kREFIT = 5,                //!< Enable building a refittable engine.
    kDISABLE_TIMING_CACHE = 6, //!< Disable reuse of timing information across identical layers.

    //! Allow (but not require) computations on tensors of type DataType::kFLOAT to use TF32.
    //! TF32 computes inner products by rounding the inputs to 10-bit mantissas before
    //! multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.
    kTF32 = 7,

    //! Allow the builder to examine weights and use optimized functions when weights have suitable sparsity.
    kSPARSE_WEIGHTS = 8,

    //! Change the allowed parameters in the EngineCapability::kSTANDARD flow to
    //! match the restrictions that EngineCapability::kSAFETY check against for DeviceType::kGPU
    //! and EngineCapability::kDLA_STANDALONE check against the DeviceType::kDLA case. This flag
    //! is forced to true if EngineCapability::kSAFETY at build time if it is unset.
    //!
    //! This flag is only supported in NVIDIA Drive(R) products.
    kSAFETY_SCOPE = 9,

    //! Require that layers execute in specified precisions. Build fails otherwise.
    kOBEY_PRECISION_CONSTRAINTS = 10,

    //! Prefer that layers execute in specified precisions.
    //! Fall back (with warning) to another precision if build would otherwise fail.
    kPREFER_PRECISION_CONSTRAINTS = 11,

    //! Require that no reformats be inserted between a layer and a network I/O tensor
    //! for which ITensor::setAllowedFormats was called.
    //! Build fails if a reformat is required for functional correctness.
    kDIRECT_IO = 12,

    //! Fail if IAlgorithmSelector::selectAlgorithms returns an empty set of algorithms.
    kREJECT_EMPTY_ALGORITHMS = 13,

    //! Enable heuristic-based tactic selection for shorter engine generation time. The engine may not
    //! be as performant as when built with a profiling-based builder.
    //!
    //! This flag is only supported by NVIDIA Ampere and later GPUs.
    kENABLE_TACTIC_HEURISTIC = 14
};

//!
//! Maximum number of builder flags in BuilderFlag enum.
//!
//! \see BuilderFlag
//!
template <>
constexpr inline int32_t EnumMax<BuilderFlag>() noexcept
{
    return 15;
}

//!
//! \class ITimingCache
//!
//! \brief Class to handle tactic timing info collected from builder.
//!
//! The timing cache is created or initialized by IBuilderConfig. It can be shared across builder instances
//! to accelerate the builder wallclock time.
//!
//! \see IBuilderConfig
//!
class ITimingCache : public INoCopy
{
public:
    virtual ~ITimingCache() noexcept = default;

    //!
    //! \brief Serialize a timing cache to IHostMemory object.
    //!
    //! This function allows serialization of current timing cache.
    //!
    //! \return A pointer to a IHostMemory object that contains a serialized timing cache.
    //!
    //! \see IHostMemory
    //!
    nvinfer1::IHostMemory* serialize() const noexcept
    {
        return mImpl->serialize();
    }

    //!
    //! \brief Combine input timing cache into local instance.
    //!
    //! This function allows combining entries in the input timing cache to local cache object.
    //!
    //! \param inputCache The input timing cache.
    //! \param ignoreMismatch Whether or not to allow cache verification header mismatch.
    //!
    //! \return True if combined successfully, false otherwise.
    //!
    //! Append entries in input cache to local cache. Conflicting entries will be skipped
    //! The input cache must be generated by a TensorRT build of exact same version, otherwise
    //! combine will be skipped and return false.
    //! ignoreMismatch must be set to true if combining a timing cache created from a
    //! different device.
    //!
    //! \warning Combining caches generated from devices with different device properties may
    //!          lead to functional/performance bugs!
    //!
    bool combine(ITimingCache const& inputCache, bool ignoreMismatch) noexcept
    {
        return mImpl->combine(inputCache, ignoreMismatch);
    }

    //!
    //! \brief Empty the timing cache
    //!
    //! \return True if reset successfully, false otherwise.
    //!
    bool reset() noexcept
    {
        return mImpl->reset();
    }

protected:
    apiv::VTimingCache* mImpl;
};

//!
//! \enum MemoryPoolType
//!
//! \brief The type for memory pools used by TensorRT.
//!
//! \see IBuilderConfig::setMemoryPoolLimit, IBuilderConfig::getMemoryPoolLimit
//!
enum class MemoryPoolType : int32_t
{
    //!
    //! kWORKSPACE is used by TensorRT to store intermediate buffers within an operation.
    //! This is equivalent to the deprecated IBuilderConfig::setMaxWorkspaceSize and overrides that value.
    //! This defaults to max device memory. Set to a smaller value to restrict tactics that use over the
    //! threshold en masse. For more targeted removal of tactics use the IAlgorithmSelector
    //! interface.
    //!
    kWORKSPACE = 0,

    //!
    //! kDLA_MANAGED_SRAM is a fast software managed RAM used by DLA to communicate within a layer.
    //! The size of this pool must be at least 4 KiB and must be a power of 2.
    //! This defaults to 1 MiB.
    //! Orin has capacity of 1 MiB per core, and Xavier shares 4 MiB across all of its accelerator cores.
    //!
    kDLA_MANAGED_SRAM = 1,

    //!
    //! kDLA_LOCAL_DRAM is host RAM used by DLA to share intermediate tensor data across operations.
    //! The size of this pool must be at least 4 KiB and must be a power of 2.
    //! This defaults to 1 GiB.
    //!
    kDLA_LOCAL_DRAM = 2,

    //!
    //! kDLA_GLOBAL_DRAM is host RAM used by DLA to store weights and metadata for execution.
    //! The size of this pool must be at least 4 KiB and must be a power of 2.
    //! This defaults to 512 MiB.
    //!
    kDLA_GLOBAL_DRAM = 3,
};

//!
//! Maximum number of memory pool types in the MemoryPoolType enum.
//!
//! \see MemoryPoolType
//!
template <>
constexpr inline int32_t EnumMax<MemoryPoolType>() noexcept
{
    return 4;
}

//!
//! \enum PreviewFeature
//!
//! \brief Define preview features
//!
//! Preview Features have been fully tested but are not yet as stable as other features in TensorRT.
//! They are provided as opt-in features for at least one release.
//!
enum class PreviewFeature : int32_t
{
    //!
    //! Optimize runtime dimensions with TensorRT's DL Compiler.
    //! Potentially reduces run time and decreases device memory usage and engine size.
    //! Models most likely to benefit from enabling kFASTER_DYNAMIC_SHAPES_0805 are transformer-based models,
    //! and models containing dynamic control flows.
    //!
    kFASTER_DYNAMIC_SHAPES_0805 = 0,

    //!
    //! Disable usage of cuDNN/cuBLAS/cuBLASLt tactics in the TensorRT core library.
    //!
    //! When the flag is enabled, TensorRT core will not use these tactics even if they are specified in
    //! \ref IBuilderConfig::setTacticSources(), but cudnnContext and cublasContext handles will still be passed to
    //! plugins via \ref IPluginV2::attachToContext() if the appropriate tactic sources are set.
    //!
    //! This allows users to experiment with disabling external library tactics without having to modify their
    //! application's plugins to support nullptr handles.
    //!
    //! The default value for this flag is off.
    //!
    //! \see TacticSource
    //!
    kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805 = 1,
};
namespace impl
{
//!
//! Maximum number of elements in PreviewFeature enum.
//!
//! \see PreviewFeature
//!
template <>
struct EnumMaxImpl<PreviewFeature>
{
    static constexpr int32_t kVALUE = 2;
};
} // namespace impl

//!
//! \class IBuilderConfig
//!
//! \brief Holds properties for configuring a builder to produce an engine.
//!
//! \see BuilderFlags
//!
class IBuilderConfig : public INoCopy
{
public:
    virtual ~IBuilderConfig() noexcept = default;

    //!
    //! \brief Set the number of minimization iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter
    //! controls the number of iterations used in minimization. The builder may sometimes run layers for more
    //! iterations to improve timing accuracy if this parameter is set to a small value and the runtime of the
    //! layer is short.
    //!
    //! \see getMinTimingIterations()
    //!
    //! \deprecated Deprecated in TensorRT 8.4. Superseded by setAvgTimingIterations().
    //!
    TRT_DEPRECATED virtual void setMinTimingIterations(int32_t minTiming) noexcept
    {
        mImpl->setMinTimingIterations(minTiming);
    }

    //!
    //! \brief Query the number of minimization iterations.
    //!
    //! By default the minimum number of iterations is 1.
    //!
    //! \see setMinTimingIterations()
    //!
    //! \deprecated Deprecated in TensorRT 8.4. Superseded by getAvgTimingIterations().
    //!
    TRT_DEPRECATED virtual int32_t getMinTimingIterations() const noexcept
    {
        return mImpl->getMinTimingIterations();
    }

    //!
    //! \brief Set the number of averaging iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter
    //! controls the number of iterations used in averaging.
    //!
    //! \see getAvgTimingIterations()
    //!
    virtual void setAvgTimingIterations(int32_t avgTiming) noexcept
    {
        mImpl->setAvgTimingIterations(avgTiming);
    }

    //!
    //! \brief Query the number of averaging iterations.
    //!
    //! By default the number of averaging iterations is 1.
    //!
    //! \see setAvgTimingIterations()
    //!
    int32_t getAvgTimingIterations() const noexcept
    {
        return mImpl->getAvgTimingIterations();
    }

    //!
    //! \brief Configure the builder to target specified EngineCapability flow.
    //!
    //! The flow means a sequence of API calls that allow an application to set up a runtime, engine,
    //! and execution context in order to run inference.
    //!
    //! The supported flows are specified in the EngineCapability enum.
    //!
    void setEngineCapability(EngineCapability capability) noexcept
    {
        mImpl->setEngineCapability(capability);
    }

    //!
    //! \brief Query EngineCapability flow configured for the builder.
    //!
    //! By default it returns EngineCapability::kSTANDARD.
    //!
    //! \see setEngineCapability()
    //!
    EngineCapability getEngineCapability() const noexcept
    {
        return mImpl->getEngineCapability();
    }

    //!
    //! \brief Set Int8 Calibration interface.
    //!
    //! The calibrator is to minimize the information loss during the INT8 quantization process.
    //!
    void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept
    {
        mImpl->setInt8Calibrator(calibrator);
    }

    //!
    //! \brief Get Int8 Calibration interface.
    //!
    IInt8Calibrator* getInt8Calibrator() const noexcept
    {
        return mImpl->getInt8Calibrator();
    }

    //!
    //! \brief Set the maximum workspace size.
    //!
    //! \param workspaceSize The maximum GPU temporary memory which the engine can use at execution time.
    //!
    //! \see getMaxWorkspaceSize()
    //!
    //! \deprecated Deprecated in TensorRT 8.3. Superseded by IBuilderConfig::setMemoryPoolLimit() with
    //! MemoryPoolType::kWORKSPACE.
    //!
    TRT_DEPRECATED void setMaxWorkspaceSize(std::size_t workspaceSize) noexcept
    {
        mImpl->setMaxWorkspaceSize(workspaceSize);
    }

    //!
    //! \brief Get the maximum workspace size.
    //!
    //! By default the workspace size is the size of total global memory in the device.
    //!
    //! \return The maximum workspace size.
    //!
    //! \see setMaxWorkspaceSize()
    //!
    //! \deprecated Deprecated in TensorRT 8.3. Superseded by IBuilderConfig::getMemoryPoolLimit() with
    //! MemoryPoolType::kWORKSPACE.
    //!
    TRT_DEPRECATED std::size_t getMaxWorkspaceSize() const noexcept
    {
        return mImpl->getMaxWorkspaceSize();
    }

    //!
    //! \brief Set the build mode flags to turn on builder options for this network.
    //!
    //! The flags are listed in the BuilderFlags enum.
    //! The flags set configuration options to build the network.
    //!
    //! \param builderFlags The build option for an engine.
    //!
    //! \note This function will override the previous set flags, rather than bitwise ORing the new flag.
    //!
    //! \see getFlags()
    //!
    void setFlags(BuilderFlags builderFlags) noexcept
    {
        mImpl->setFlags(builderFlags);
    }

    //!
    //! \brief Get the build mode flags for this builder config. Defaults to 0.
    //!
    //! \return The build options as a bitmask.
    //!
    //! \see setFlags()
    //!
    BuilderFlags getFlags() const noexcept
    {
        return mImpl->getFlags();
    }

    //!
    //! \brief clear a single build mode flag.
    //!
    //! clears the builder mode flag from the enabled flags.
    //!
    //! \see setFlags()
    //!
    void clearFlag(BuilderFlag builderFlag) noexcept
    {
        mImpl->clearFlag(builderFlag);
    }

    //!
    //! \brief Set a single build mode flag.
    //!
    //! Add the input builder mode flag to the already enabled flags.
    //!
    //! \see setFlags()
    //!
    void setFlag(BuilderFlag builderFlag) noexcept
    {
        mImpl->setFlag(builderFlag);
    }

    //!
    //! \brief Returns true if the build mode flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    bool getFlag(BuilderFlag builderFlag) const noexcept
    {
        return mImpl->getFlag(builderFlag);
    }

    //!
    //! \brief Set the device that this layer must execute on.
    //! \param deviceType that this layer must execute on.
    //! If DeviceType is not set or is reset, TensorRT will use the default DeviceType set in the builder.
    //!
    //! \note The device type for a layer must be compatible with the safety flow (if specified).
    //! For example a layer cannot be marked for DLA execution while the builder is configured for kSAFE_GPU.
    //!
    //! \see getDeviceType()
    //!
    void setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept
    {
        mImpl->setDeviceType(layer, deviceType);
    }

    //!
    //! \brief Get the device that this layer executes on.
    //! \return Returns DeviceType of the layer.
    //!
    DeviceType getDeviceType(ILayer const* layer) const noexcept
    {
        return mImpl->getDeviceType(layer);
    }

    //!
    //! \brief whether the DeviceType has been explicitly set for this layer
    //! \return true if device type is not default
    //! \see setDeviceType() getDeviceType() resetDeviceType()
    //!
    bool isDeviceTypeSet(ILayer const* layer) const noexcept
    {
        return mImpl->isDeviceTypeSet(layer);
    }

    //!
    //! \brief reset the DeviceType for this layer
    //!
    //! \see setDeviceType() getDeviceType() isDeviceTypeSet()
    //!
    void resetDeviceType(ILayer const* layer) noexcept
    {
        mImpl->resetDeviceType(layer);
    }

    //!
    //! \brief Checks if a layer can run on DLA.
    //! \return status true if the layer can on DLA else returns false.
    //!
    bool canRunOnDLA(ILayer const* layer) const noexcept
    {
        return mImpl->canRunOnDLA(layer);
    }

    //!
    //! \brief Sets the DLA core used by the network. Defaults to -1.
    //! \param dlaCore The DLA core to execute the engine on, in the range [0,getNbDlaCores()).
    //!
    //! This function is used to specify which DLA core to use via indexing, if multiple DLA cores are available.
    //!
    //! \warning if getNbDLACores() returns 0, then this function does nothing.
    //!
    //! \see IRuntime::setDLACore() getDLACore()
    //!
    void setDLACore(int32_t dlaCore) noexcept
    {
        mImpl->setDLACore(dlaCore);
    }

    //!
    //! \brief Get the DLA core that the engine executes on.
    //! \return assigned DLA core or -1 for DLA not present or unset.
    //!
    int32_t getDLACore() const noexcept
    {
        return mImpl->getDLACore();
    }

    //!
    //! \brief Sets the default DeviceType to be used by the builder. It ensures that all the layers that can run on
    //! this device will run on it, unless setDeviceType is used to override the default DeviceType for a layer.
    //! \see getDefaultDeviceType()
    //!
    void setDefaultDeviceType(DeviceType deviceType) noexcept
    {
        mImpl->setDefaultDeviceType(deviceType);
    }

    //!
    //! \brief Get the default DeviceType which was set by setDefaultDeviceType.
    //!
    //! By default it returns DeviceType::kGPU.
    //!
    DeviceType getDefaultDeviceType() const noexcept
    {
        return mImpl->getDefaultDeviceType();
    }

    //!
    //! \brief Resets the builder configuration to defaults.
    //!
    //! Useful for initializing a builder config object to its original state.
    //!
    void reset() noexcept
    {
        mImpl->reset();
    }

    //!
    //! \brief Delete this IBuilderConfig.
    //!
    //! De-allocates any internally allocated memory.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by `delete`.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

    //!
    //! \brief Set the cuda stream that is used to profile this network.
    //!
    //! \param stream The cuda stream used for profiling by the builder.
    //!
    //! \see getProfileStream()
    //!
    void setProfileStream(const cudaStream_t stream) noexcept
    {
        return mImpl->setProfileStream(stream);
    }

    //!
    //! \brief Get the cuda stream that is used to profile this network.
    //!
    //! \return The cuda stream set by setProfileStream, nullptr if setProfileStream has not been called.
    //!
    //! \see setProfileStream()
    //!
    cudaStream_t getProfileStream() const noexcept
    {
        return mImpl->getProfileStream();
    }

    //!
    //! \brief Add an optimization profile.
    //!
    //! This function must be called at least once if the network has dynamic or shape input tensors.
    //! This function may be called at most once when building a refittable engine, as more than
    //! a single optimization profile are not supported for refittable engines.
    //!
    //! \param profile The new optimization profile, which must satisfy profile->isValid() == true
    //! \return The index of the optimization profile (starting from 0) if the input is valid, or -1 if the input is
    //!         not valid.
    //!
    int32_t addOptimizationProfile(IOptimizationProfile const* profile) noexcept
    {
        return mImpl->addOptimizationProfile(profile);
    }

    //!
    //! \brief Get number of optimization profiles.
    //!
    //! This is one higher than the index of the last optimization profile that has be defined (or
    //! zero, if none has been defined yet).
    //!
    //! \return The number of the optimization profiles.
    //!
    int32_t getNbOptimizationProfiles() const noexcept
    {
        return mImpl->getNbOptimizationProfiles();
    }

    //!
    //! \brief Set verbosity level of layer information exposed in NVTX annotations and IEngineInspector.
    //!
    //! Control how much layer information will be exposed in NVTX annotations and IEngineInspector.
    //!
    //! \see ProfilingVerbosity, getProfilingVerbosity(), IEngineInspector
    //!
    void setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept
    {
        mImpl->setProfilingVerbosity(verbosity);
    }

    //!
    //! \brief Get verbosity level of layer information exposed in NVTX annotations and IEngineInspector.
    //!
    //! Get the current setting of verbosity level of layer information exposed in
    //! NVTX annotations and IEngineInspector. Default value is ProfilingVerbosity::kLAYER_NAMES_ONLY.
    //!
    //! \see ProfilingVerbosity, setProfilingVerbosity(), IEngineInspector
    //!
    ProfilingVerbosity getProfilingVerbosity() const noexcept
    {
        return mImpl->getProfilingVerbosity();
    }

    //!
    //! \brief Set Algorithm Selector.
    //!
    //! \param selector The algorithm selector to be set in the build config.
    void setAlgorithmSelector(IAlgorithmSelector* selector) noexcept
    {
        mImpl->setAlgorithmSelector(selector);
    }

    //!
    //! \brief Get Algorithm Selector.
    //!
    IAlgorithmSelector* getAlgorithmSelector() const noexcept
    {
        return mImpl->getAlgorithmSelector();
    }

    //!
    //! \brief Add a calibration profile.
    //!
    //! Calibration optimization profile must be set if int8 calibration is used to set scales for a network with
    //! runtime dimensions.
    //!
    //! \param profile The new calibration profile, which must satisfy profile->isValid() == true or be nullptr.
    //! MIN and MAX values will be overwritten by kOPT.
    //! \return True if the calibration profile was set correctly.
    //!
    bool setCalibrationProfile(IOptimizationProfile const* profile) noexcept
    {
        return mImpl->setCalibrationProfile(profile);
    }

    //!
    //! \brief Get the current calibration profile.
    //!
    //! \return A pointer to the current calibration profile or nullptr if calibration profile is unset.
    //!
    IOptimizationProfile const* getCalibrationProfile() noexcept
    {
        return mImpl->getCalibrationProfile();
    }

    //!
    //! \brief Set the quantization flags.
    //!
    //! The flags are listed in the QuantizationFlag enum.
    //! The flags set configuration options to quantize the network in int8.
    //!
    //! \param flags The quantization flags.
    //!
    //! \note This function will override the previous set flags, rather than bitwise ORing the new flag.
    //!
    //! \see getQuantizationFlags()
    //!
    void setQuantizationFlags(QuantizationFlags flags) noexcept
    {
        mImpl->setQuantizationFlags(flags);
    }

    //!
    //! \brief Get the quantization flags.
    //!
    //! \return The quantization flags as a bitmask.
    //!
    //! \see setQuantizationFlag()
    //!
    QuantizationFlags getQuantizationFlags() const noexcept
    {
        return mImpl->getQuantizationFlags();
    }

    //!
    //! \brief clear a quantization flag.
    //!
    //! Clears the quantization flag from the enabled quantization flags.
    //!
    //! \see setQuantizationFlags()
    //!
    void clearQuantizationFlag(QuantizationFlag flag) noexcept
    {
        mImpl->clearQuantizationFlag(flag);
    }

    //!
    //! \brief Set a single quantization flag.
    //!
    //! Add the input quantization flag to the already enabled quantization flags.
    //!
    //! \see setQuantizationFlags()
    //!
    void setQuantizationFlag(QuantizationFlag flag) noexcept
    {
        mImpl->setQuantizationFlag(flag);
    }

    //!
    //! \brief Returns true if the quantization flag is set.
    //!
    //! \see getQuantizationFlags()
    //!
    //! \return True if quantization flag is set, false if unset.
    //!
    bool getQuantizationFlag(QuantizationFlag flag) const noexcept
    {
        return mImpl->getQuantizationFlag(flag);
    }

    //!
    //! \brief Set tactic sources.
    //!
    //! This bitset controls which tactic sources TensorRT is allowed to use for tactic
    //! selection.
    //!
    //! Multiple tactic sources may be combined with a bitwise OR operation. For example,
    //! to enable cublas and cublasLt as tactic sources, use a value of:
    //!
    //! 1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U <<
    //! static_cast<uint32_t>(TacticSource::kCUBLAS_LT)
    //!
    //! \see getTacticSources
    //!
    //! \return true if the tactic sources in the build configuration were updated.
    //!         The tactic sources in the build configuration will not be updated if the provided value is invalid.
    //!
    bool setTacticSources(TacticSources tacticSources) noexcept
    {
        return mImpl->setTacticSources(tacticSources);
    }

    //!
    //! \brief Get tactic sources.
    //!
    //! Get the tactic sources currently set in the engine build
    //! configuration.
    //!
    //! \see setTacticSources
    //!
    //! \return tactic sources
    //!
    TacticSources getTacticSources() const noexcept
    {
        return mImpl->getTacticSources();
    }

    //!
    //! \brief Create timing cache
    //!
    //! Create ITimingCache instance from serialized raw data. The created timing cache doesn’t belong to
    //! a specific IBuilderConfig. It can be shared by multiple builder instances. Call setTimingCache()
    //! before launching a builder to attach cache to builder instance.
    //!
    //! \param blob A pointer to the raw data that contains serialized timing cache
    //! \param size The size in bytes of the serialized timing cache. Size 0 means create a new cache from scratch
    //!
    //! \see setTimingCache
    //!
    //! \return the pointer to ITimingCache created
    //!
    nvinfer1::ITimingCache* createTimingCache(void const* blob, std::size_t size) const noexcept
    {
        return mImpl->createTimingCache(blob, size);
    }

    //!
    //! \brief Attach a timing cache to IBuilderConfig
    //!
    //! The timing cache has verification header to make sure the provided cache can be used in current environment.
    //! A failure will be reported if the CUDA device property in the provided cache is different from current
    //! environment. ignoreMismatch = true skips strict verification and allows loading cache created from a different
    //! device.
    //!
    //! The cache must not be destroyed until after the engine is built.
    //!
    //! \param cache the timing cache to be used
    //! \param ignoreMismatch whether or not allow using a cache that contains different CUDA device property
    //!
    //! \return true if set successfully, false otherwise
    //!
    //! \warning Using cache generated from devices with different CUDA device properties may lead to
    //!          functional/performance bugs.
    //!
    bool setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept
    {
        return mImpl->setTimingCache(cache, ignoreMismatch);
    }

    //!
    //! \brief Get the pointer to the timing cache from current IBuilderConfig
    //!
    //! \return pointer to the timing cache used in current IBuilderConfig
    //!
    nvinfer1::ITimingCache const* getTimingCache() const noexcept
    {
        return mImpl->getTimingCache();
    }

    //!
    //! \brief Set the memory size for the memory pool.
    //!
    //! TensorRT layers access different memory pools depending on the operation.
    //! This function sets in the IBuilderConfig the size limit, specified by \p poolSize,
    //! for the corresponding memory pool, specified by \p pool.
    //! TensorRT will build a plan file that is constrained by these limits or report
    //! which constraint caused the failure.
    //!
    //! If the size of the pool, specified by \p poolSize, fails to meet the size requirements
    //! for the pool, this function does nothing and emits the recoverable error,
    //! ErrorCode::kINVALID_ARGUMENT, to the registered IErrorRecorder.
    //!
    //! If the size of the pool is larger than the maximum possible value for the
    //! configuration, this function does nothing and emits ErrorCode::kUNSUPPORTED_STATE.
    //!
    //! If the pool does not exist on the requested device type when building
    //! the network, a warning is emitted to the logger, and the memory pool
    //! value is ignored.
    //!
    //! Refer to MemoryPoolType to see the size requirements for each pool.
    //!
    //! \param pool The memory pool to limit the available memory for.
    //! \param poolSize The size of the pool in bytes.
    //!
    //! \see getMemoryPoolLimit, MemoryPoolType
    //!
    void setMemoryPoolLimit(MemoryPoolType pool, std::size_t poolSize) noexcept
    {
        mImpl->setMemoryPoolLimit(pool, poolSize);
    }

    //!
    //! \brief Get the memory size limit of the memory pool.
    //!
    //! Retrieve the memory size limit of the corresponding pool in bytes.
    //! If setMemoryPoolLimit for the pool has not been called, this returns the default
    //! value used by TensorRT. This default value is not necessarily the maximum possible
    //! value for that configuration.
    //!
    //! \param pool The memory pool to get the limit for.
    //!
    //! \returns The size of the memory limit, in bytes, for the corresponding pool.
    //!
    //! \see setMemoryPoolLimit
    //!
    std::size_t getMemoryPoolLimit(MemoryPoolType pool) const noexcept
    {
        return mImpl->getMemoryPoolLimit(pool);
    }

    //!
    //! \brief Enable or disable a specific preview feature
    //!
    //! Allows enabling or disabling experimental features, which are not enabled by default in the
    //! current release.
    //!
    //! Refer to PreviewFeature for additional information, and a list of the available features.
    //!
    //! \param feature the feature to enable / disable
    //! \param enable true for enable, false for disable
    //!
    //! \see PreviewFeature, getPreviewFeature
    //!
    void setPreviewFeature(PreviewFeature feature, bool enable) noexcept
    {
        mImpl->setPreviewFeature(feature, enable);
    }

    //!
    //! \brief Get status of preview feature
    //!
    //! \param feature the feature to query
    //!
    //! \returns true if the \p feature is enabled, false otherwise
    //!
    //! \see PreviewFeature, setPreviewFeature
    //!
    bool getPreviewFeature(PreviewFeature feature) const noexcept
    {
        return mImpl->getPreviewFeature(feature);
    }

protected:
    apiv::VBuilderConfig* mImpl;
};

//! \brief Represents one or more NetworkDefinitionCreationFlag flags
//! using binary OR operations.
//!  e.g., 1U << NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
//!
//! \see IBuilder::createNetworkV2
//!
using NetworkDefinitionCreationFlags = uint32_t;

//! \enum NetworkDefinitionCreationFlag
//!
//! \brief List of immutable network properties expressed at network creation time.
//! NetworkDefinitionCreationFlag is used with createNetworkV2() to specify immutable properties of the network.
//! Creating a network without NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag has been deprecated.
//!
//! \see IBuilder::createNetworkV2
//!
enum class NetworkDefinitionCreationFlag : int32_t
{
    //! Mark the network to be an explicit batch network.
    //! Dynamic shape support requires that the kEXPLICIT_BATCH flag is set.
    //! With dynamic shapes, any of the input dimensions can vary at run-time,
    //! and there are no implicit dimensions in the network specification.
    //! Varying dimensions are specified by using the wildcard dimension value -1.
    kEXPLICIT_BATCH = 0,

    //! Deprecated. This flag has no effect now, but is only kept for backward compatability.
    //!
    kEXPLICIT_PRECISION TRT_DEPRECATED_ENUM = 1,
};

//!
//! Maximum number of elements in NetworkDefinitionCreationFlag enum.
//!
//! \see NetworkDefinitionCreationFlag
//!
template <>
constexpr inline int32_t EnumMax<NetworkDefinitionCreationFlag>() noexcept
{
    return 2;
}

//!
//! \class IBuilder
//!
//! \brief Builds an engine from a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IBuilder : public INoCopy
{
public:
    virtual ~IBuilder() noexcept = default;

    //!
    //! \brief Set the maximum batch size. This has no effect for networks created with explicit batch dimension mode.
    //!
    //! \param batchSize The maximum batch size which can be used at execution time, and also the batch size for which
    //! the engine will be optimized.
    //!
    //! \deprecated Deprecated in TensorRT 8.4.
    //!
    //! \see getMaxBatchSize()
    //!
    TRT_DEPRECATED void setMaxBatchSize(int32_t batchSize) noexcept
    {
        mImpl->setMaxBatchSize(batchSize);
    }

    //!
    //! \brief Get the maximum batch size.
    //!
    //! \return The maximum batch size.
    //!
    //! \deprecated Deprecated in TensorRT 8.4.
    //!
    //! \see setMaxBatchSize()
    //! \see getMaxDLABatchSize()
    //!
    TRT_DEPRECATED int32_t getMaxBatchSize() const noexcept
    {
        return mImpl->getMaxBatchSize();
    }

    //!
    //! \brief Determine whether the platform has fast native fp16.
    //!
    bool platformHasFastFp16() const noexcept
    {
        return mImpl->platformHasFastFp16();
    }

    //!
    //! \brief Determine whether the platform has fast native int8.
    //!
    bool platformHasFastInt8() const noexcept
    {
        return mImpl->platformHasFastInt8();
    }

    //!
    //! \brief Destroy this object.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by `delete`.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

    //!
    //! \brief Get the maximum batch size DLA can support.
    //! For any tensor the total volume of index dimensions combined(dimensions other than CHW) with the requested
    //! batch size should not exceed the value returned by this function.
    //!
    //! \warning getMaxDLABatchSize does not work with dynamic shapes.
    //!
    int32_t getMaxDLABatchSize() const noexcept
    {
        return mImpl->getMaxDLABatchSize();
    }

    //!
    //! \brief Return the number of DLA engines available to this builder.
    //!
    int32_t getNbDLACores() const noexcept
    {
        return mImpl->getNbDLACores();
    }

    //!
    //! \brief Set the GPU allocator.
    //! \param allocator Set the GPU allocator to be used by the builder. All GPU memory acquired will use this
    //! allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! \note This allocator will be passed to any engines created via the builder; thus the lifetime of the allocator
    //! must span the lifetime of those engines as
    //! well as that of the builder. If nullptr is passed, the default allocator will be used.
    //!
    void setGpuAllocator(IGpuAllocator* allocator) noexcept
    {
        mImpl->setGpuAllocator(allocator);
    }

    //!
    //! \brief Create a builder configuration object.
    //!
    //! \see IBuilderConfig
    //!
    nvinfer1::IBuilderConfig* createBuilderConfig() noexcept
    {
        return mImpl->createBuilderConfig();
    }

    //!
    //! \brief Builds an engine for the given INetworkDefinition and given IBuilderConfig.
    //!
    //! It enables the builder to build multiple engines based on the same network definition, but with different
    //! builder configurations.
    //!
    //! \note This function will synchronize the cuda stream returned by \p config.getProfileStream() before returning.
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by IBuilder::buildSerializedNetwork().
    //!
    TRT_DEPRECATED nvinfer1::ICudaEngine* buildEngineWithConfig(
        INetworkDefinition& network, IBuilderConfig& config) noexcept
    {
        return mImpl->buildEngineWithConfig(network, config);
    }

    //! \brief Create a network definition object
    //!
    //! Creates a network definition object with immutable properties specified using the flags parameter.
    //! CreateNetworkV2 supports dynamic shapes and explicit batch dimensions when used with
    //! NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag.
    //! Creating a network without NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag has been deprecated.
    //!
    //! \param flags Bitset of NetworkDefinitionCreationFlags specifying network properties combined with bitwise OR.
    //!             e.g., 1U << NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    //!
    //! \see INetworkDefinition, NetworkDefinitionCreationFlags
    //!
    nvinfer1::INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept
    {
        return mImpl->createNetworkV2(flags);
    }

    //! \brief Create a new optimization profile.
    //!
    //! If the network has any dynamic input tensors, the appropriate calls to setDimensions() must be made.
    //! Likewise, if there are any shape input tensors, the appropriate calls to setShapeValues() are required.
    //! The builder retains ownership of the created optimization profile and returns a raw pointer, i.e. the users
    //! must not attempt to delete the returned pointer.
    //!
    //! \see IOptimizationProfile
    //!
    nvinfer1::IOptimizationProfile* createOptimizationProfile() noexcept
    {
        return mImpl->createOptimizationProfile();
    }

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
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class.
    //! A nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Resets the builder state to default values.
    //!
    void reset() noexcept
    {
        mImpl->reset();
    }

    //!
    //! \brief Determine whether the platform has TF32 support.
    //!
    bool platformHasTf32() const noexcept
    {
        return mImpl->platformHasTf32();
    }

    //!
    //! \brief Builds and serializes a network for the given INetworkDefinition and IBuilderConfig.
    //!
    //! This function allows building and serialization of a network without creating an engine.
    //!
    //! \param network Network definition.
    //! \param config Builder configuration.
    //!
    //! \return A pointer to a IHostMemory object that contains a serialized network.
    //!
    //! \note This function will synchronize the cuda stream returned by \p config.getProfileStream() before returning.
    //!
    //! \see INetworkDefinition, IBuilderConfig, IHostMemory
    //!
    nvinfer1::IHostMemory* buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept
    {
        return mImpl->buildSerializedNetwork(network, config);
    }

    //!
    //! \brief Checks that a network is within the scope of the IBuilderConfig settings.
    //!
    //! \param network The network definition to check for configuration compliance.
    //! \param config The configuration of the builder to use when checking \p network.
    //!
    //! Given an INetworkDefinition, \p network, and an IBuilderConfig, \p config, check if
    //! the network falls within the constraints of the builder configuration based on the
    //! EngineCapability, BuilderFlag, and DeviceType. If the network is within the constraints,
    //! then the function returns true, and false if a violation occurs. This function reports
    //! the conditions that are violated to the registered ErrorRecorder.
    //!
    //! \return True if network is within the scope of the restrictions specified by the builder config,
    //! false otherwise.
    //!
    //! \note This function will synchronize the cuda stream returned by \p config.getProfileStream() before returning.
    //!
    //! This function is only supported in NVIDIA Drive(R) products.
    //!
    bool isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept
    {
        return mImpl->isNetworkSupported(network, config);
    }

    //!
    //! \brief get the logger with which the builder was created
    //!
    //! \return the logger
    //!
    ILogger* getLogger() const noexcept
    {
        return mImpl->getLogger();
    }

    //!
    //! \brief Set the maximum number of threads.
    //! \param maxThreads The maximum number of threads that can be used by the builder.
    //! \return True if successful, false otherwise.
    //!
    //! The default value is 1 and includes the current thread.
    //! A value greater than 1 permits TensorRT to use multi-threaded algorithms.
    //! A value less than 1 triggers a kINVALID_ARGUMENT error.
    //!
    bool setMaxThreads(int32_t maxThreads) noexcept
    {
        return mImpl->setMaxThreads(maxThreads);
    }

    //!
    //! \brief get the maximum number of threads that can be used by the builder.
    //!
    //! Retrieves the maximum number of threads that can be used by the builder.
    //!
    //! \return The maximum number of threads that can be used by the builder.
    //!
    //! \see setMaxThreads()
    //!
    int32_t getMaxThreads() const noexcept
    {
        return mImpl->getMaxThreads();
    }

protected:
    apiv::VBuilder* mImpl;
};

} // namespace nvinfer1

//!
//! Internal C entry point for creating IBuilder.
//! @private
//!
extern "C" TENSORRTAPI void* createInferBuilder_INTERNAL(void* logger, int32_t version) noexcept;

namespace nvinfer1
{
namespace
{

//!
//! \brief Create an instance of an IBuilder class.
//!
//! \param logger The logging class for the builder.
//!
//! unnamed namespace avoids linkage surprises when linking objects built with different versions of this header.
//!
inline IBuilder* createInferBuilder(ILogger& logger) noexcept
{
    return static_cast<IBuilder*>(createInferBuilder_INTERNAL(&logger, NV_TENSORRT_VERSION));
}

} // namespace

//!
//! \brief Return the plugin registry for the given capability or nullptr if no registry exists
//!
//! Engine capabilities EngineCapability::kSTANDARD and EngineCapability::kSAFETY have distinct plugin registries.
//! Use IPluginRegistry::registerCreator from the registry to register plugins.
//! Plugins registered in a registry associated with a specific engine capability are only available when
//! building engines with that engine capability.
//!
//! There is no plugin registry for EngineCapability::kDLA_STANDALONE.
//!
extern "C" TENSORRTAPI nvinfer1::IPluginRegistry* getBuilderPluginRegistry(
    nvinfer1::EngineCapability capability) noexcept;

} // namespace nvinfer1

#endif // NV_INFER_H
