/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    kCAST = 1,                //!< Cast layer
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
    kIDENTITY = 20,           //!< Identity layer.
    kPLUGIN_V2 = 21,          //!< PluginV2 layer.
    kSLICE = 22,              //!< Slice layer.
    kSHAPE = 23,              //!< Shape layer.
    kPARAMETRIC_RELU = 24,    //!< Parametric ReLU layer.
    kRESIZE = 25,             //!< Resize Layer.
    kTRIP_LIMIT = 26,         //!< Loop Trip limit layer
    kRECURRENCE = 27,         //!< Loop Recurrence layer
    kITERATOR = 28,           //!< Loop Iterator layer
    kLOOP_OUTPUT = 29,        //!< Loop output layer
    kSELECT = 30,             //!< Select layer.
    kFILL = 31,               //!< Fill layer
    kQUANTIZE = 32,           //!< Quantize layer
    kDEQUANTIZE = 33,         //!< Dequantize layer
    kCONDITION = 34,          //!< Condition layer
    kCONDITIONAL_INPUT = 35,  //!< Conditional Input layer
    kCONDITIONAL_OUTPUT = 36, //!< Conditional Output layer
    kSCATTER = 37,            //!< Scatter layer
    kEINSUM = 38,             //!< Einsum layer
    kASSERTION = 39,          //!< Assertion layer
    kONE_HOT = 40,            //!< OneHot layer
    kNON_ZERO = 41,           //!< NonZero layer
    kGRID_SAMPLE = 42,        //!< Grid sample layer
    kNMS = 43,                //!< NMS layer
    kREVERSE_SEQUENCE = 44,   //!< Reverse sequence layer
    kNORMALIZATION = 45,      //!< Normalization layer
    kPLUGIN_V3 = 46,          //!< PluginV3 layer.
    kSQUEEZE = 47,            //!< Squeeze Layer.
    kUNSQUEEZE = 48,          //!< Unsqueeze Layer.
};

//!
//! Maximum number of elements in LayerType enum.
//!
//! \see LayerType
//!
template <>
constexpr inline int32_t EnumMax<LayerType>() noexcept
{
    return 49;
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
    kRELU = 0,              //!< Rectified linear activation.
    kSIGMOID = 1,           //!< Sigmoid activation.
    kTANH = 2,              //!< TanH activation.
    kLEAKY_RELU = 3,        //!< LeakyRelu activation: x>=0 ? x : alpha * x.
    kELU = 4,               //!< Elu activation: x>=0 ? x : alpha * (exp(x) - 1).
    kSELU = 5,              //!< Selu activation: x>0 ? beta * x : beta * (alpha*exp(x) - alpha)
    kSOFTSIGN = 6,          //!< Softsign activation: x / (1+|x|)
    kSOFTPLUS = 7,          //!< Parametric softplus activation: alpha*log(exp(beta*x)+1)
    kCLIP = 8,              //!< Clip activation: max(alpha, min(beta, x))
    kHARD_SIGMOID = 9,      //!< Hard sigmoid activation: max(0, min(1, alpha*x+beta))
    kSCALED_TANH = 10,      //!< Scaled tanh activation: alpha*tanh(beta*x)
    kTHRESHOLDED_RELU = 11, //!< Thresholded ReLU activation: x>alpha ? x : 0
    kGELU_ERF = 12,         //!< GELU erf activation: 0.5 * x * (1 + erf(sqrt(0.5) * x))
    kGELU_TANH = 13         //!< GELU tanh activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (0.044715F * pow(x, 3) + x)))
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
    static constexpr int32_t kVALUE = 14;
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
    //! Each input and output tensor must have a unique name.
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
    void setDimensions(Dims const& dimensions) noexcept
    {
        mImpl->setDimensions(dimensions);
    }

    //!
    //! \brief Get the dimensions of a tensor.
    //!
    //! \return The dimensions of the tensor.
    //!
    //! \warning getDimensions() returns a -1 for dimensions that are derived from a wildcard dimension.
    //!
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
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED bool setDynamicRange(float min, float max) noexcept
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
    //! \brief Set whether to enable broadcast of tensor across the implicit batch dimension.
    //!
    //! \warning This method has no effect other than issuing a warning.
    //!
    //! \param broadcastAcrossBatch Whether to broadcast the tensor across the implicit
    //!         batch dimension that was a feature of TensorRT 9.x and prior.
    //!
    //! \see getBroadcastAcrossBatch()
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is not supported since TensorRT 10.0.
    //!
    TRT_DEPRECATED void setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept
    {
        mImpl->setBroadcastAcrossBatch(broadcastAcrossBatch);
    }

    //!
    //! \brief Check if tensor is broadcast across the implicit batch dimension.
    //!
    //! \return Always false since TensorRT 10.0 does not support an implicit batch dimension.
    //!
    //! \see setBroadcastAcrossBatch()
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is not supported since TensorRT 10.0.
    //!
    TRT_DEPRECATED bool getBroadcastAcrossBatch() const noexcept
    {
        return mImpl->getBroadcastAcrossBatch();
    }

    //!
    //! \brief Get the storage location of a tensor.
    //!
    //! \return The location of tensor data.
    //!
    //! \see setLocation()
    //!
    TensorLocation getLocation() const noexcept
    {
        return mImpl->getLocation();
    }

    //!
    //! \brief Set the storage location of a tensor
    //!
    //! \param location the location of tensor data
    //!
    //! Only network input tensors for storing sequence lengths for RNNv2 are supported.
    //! Using host storage for layers that do not support it will generate
    //! errors at build time.
    //!
    //! \see getLocation()
    //!
    //! \deprecated Deprecated in TensorRT 10.0. RNNv2 is not supported and the location must
    //! always be TensorLocation::kDEVICE since TensorRT 10.0.
    //!
    TRT_DEPRECATED void setLocation(TensorLocation location) noexcept
    {
        mImpl->setLocation(location);
    }

    //!
    //! \brief Query whether dynamic range is set.
    //!
    //! \return True if dynamic range is set, false otherwise.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED bool dynamicRangeIsSet() const noexcept
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
    //! \brief Set allowed formats for an input or output tensor. By default all formats are allowed.
    //!        Shape tensors (for which isShapeTensor() returns true) may only have row-major linear format.
    //!
    //! When running network on DLA and the build option kGPU_FALLBACK is not specified, if DLA format(kCHW4 with Int8,
    //! kCHW4 with FP16, kCHW16 with FP16, kCHW32 with Int8) is set, the input format is treated as native DLA format
    //! with line stride requirement. Input/output binding with these format should have correct layout during
    //! inference.
    //!
    //! Tensor formats are determined at build time by TensorRT for tensors not marked as input or output.
    //!
    //! \param formats A bitmask of TensorFormat values that are supported for this tensor.
    //!
    //! \see ITensor::getAllowedFormats()
    //!
    //! \see TensorFormats
    //!
    void setAllowedFormats(TensorFormats formats) noexcept
    {
        mImpl->setAllowedFormats(formats);
    }

    //!
    //! \brief Get a bitmask of TensorFormat values that the tensor supports.
    //!        For a shape tensor, only row-major linear format is allowed.
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
    //! It must have type Int32, Int64, Bool, or Float, and its shape must be determinable at build time.
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
    //! It is possible for a tensor to be both a shape tensor and an execution tensor.
    //!
    //! \return True if tensor is a shape tensor, false otherwise.
    //!
    //! \see INetworkDefinition::markOutputForShapes()
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
    //! (\ref ISliceLayer).
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
    //! \return The indexed output tensor, or nullptr if the index is out of range or the tensor is optional.
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
    //! \brief Set the preferred or required computational precision of this layer in a weakly-typed network.
    //!
    //! Setting the precision directs TensorRT to choose an implementation that runs at this computational precision.
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
    //! Strongly-typed networks reject calls to method setPrecision. In strongly-typed networks, the computation
    //! precision is typically controlled by casting the input tensors to the desired type.
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
    //! \brief Set the output type of this layer in a weakly-typed network.
    //!
    //! Setting the output type constrains TensorRT to choose implementations which generate output data with the
    //! given type. If it is not set, TensorRT will select output type based on layer computational precision. TensorRT
    //! could still choose non-conforming output type based on fastest implementation. To force choosing the requested
    //! output type, set exactly one of the following flags, which differ in what happens if no such implementation
    //! exists:
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
    //! Strongly-typed networks reject calls to method setOutputType. Instead, the output type can be set
    //! only for layers that define method setToType(). Those layers are:
    //!
    //! * ICastLayer
    //! * IDequantizeLayer
    //! * IFillLayer
    //! * IQuantizeLayer
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
    //!
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
    //!
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

    //!
    //! \brief Set the metadata for this layer.
    //!
    //! The metadata is emitted in the JSON returned by IEngineInspector with
    //! ProfilingVerbosity set to kDETAILED.
    //!
    //! \param metadata The per-layer metadata.
    //!
    //! \warning The string name must be null-terminated and be at most 4096 bytes including the terminator.
    //!
    //! \see getMetadata()
    //! \see getLayerInformation()
    //!
    void setMetadata(char const* metadata) noexcept
    {
        mLayer->setMetadata(metadata);
    }

    //!
    //! \brief Get the metadata of the layer.
    //!
    //! \return The metadata as a null-terminated C-style string. If setMetadata() has not been called,
    //!         an empty string "" will be returned as a default value.
    //!
    //! \see setMetadata()
    //!
    char const* getMetadata() const noexcept
    {
        return mLayer->getMetadata();
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
//! There are two padding styles, EXPLICIT and SAME with each style having two variants.
//! The EXPLICIT style determine if the final sampling location is used or not.
//! The SAME style determine if the asymmetry in the padding is on the pre or post padding.
//!
//! \code
//! Shorthand:
//!     I = dimensions of input image.
//!     B = prePadding, before the image data.
//!     A = postPadding, after the image data.
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
//!     - EXPLICIT_ROUND_UP:
//! \code
//!         O = ceil((M - DK) / S) + 1
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
//!     - EXPLICIT_ROUND_UP:
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
enum class PaddingMode : int32_t
{
    kEXPLICIT_ROUND_DOWN = 0, //!< Use explicit padding, rounding output size down.
    kEXPLICIT_ROUND_UP = 1,   //!< Use explicit padding, rounding output size up.
    kSAME_UPPER = 2,          //!< Use SAME padding, with prePadding <= postPadding.
    kSAME_LOWER = 3,          //!< Use SAME padding, with prePadding >= postPadding.
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
    static constexpr int32_t kVALUE = 4;
};
} // namespace impl

//!
//! \class IConvolutionLayer
//!
//! \brief A convolution layer in a network definition.
//!
//! This layer performs a correlation operation between 3 or 4 dimensional filter with a 4 or 5 dimensional tensor to
//! produce another 4 or 5 dimensional tensor.
//!
//! An optional bias argument is supported, which adds a per-channel constant to each value in the output.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConvolutionLayer : public ILayer
{
public:
    //!
    //! \brief Set the number of output maps for the convolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    void setNbOutputMaps(int64_t nbOutputMaps) noexcept
    {
        mImpl->setNbOutputMaps(nbOutputMaps);
    }

    //!
    //! \brief Get the number of output maps for the convolution.
    //!
    //! \see setNbOutputMaps()
    //!
    int64_t getNbOutputMaps() const noexcept
    {
        return mImpl->getNbOutputMaps();
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
    void setNbGroups(int64_t nbGroups) noexcept
    {
        mImpl->setNbGroups(nbGroups);
    }

    //!
    //! \brief Get the number of groups of the convolution.
    //!
    //! \see setNbGroups()
    //!
    int64_t getNbGroups() const noexcept
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
    void setPrePadding(Dims const& padding) noexcept
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
    void setPostPadding(Dims const& padding) noexcept
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
    void setKernelSizeNd(Dims const& kernelSize) noexcept
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
    //! \see getStrideNd()
    //!
    void setStrideNd(Dims const& stride) noexcept
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
    void setPaddingNd(Dims const& padding) noexcept
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
    //! \see getDilationNd()
    //!
    void setDilationNd(Dims const& dilation) noexcept
    {
        mImpl->setDilationNd(dilation);
    }

    //!
    //! \brief Get the multi-dimension dilation of the convolution.
    //!
    //! \see setDilationNd()
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
    //!
    //! \see getKernelWeights(), setKernelWeights(), getBiasWeights(), setBiasWeights()
    //!
    using ILayer::setInput;

protected:
    virtual ~IConvolutionLayer() noexcept = default;
    apiv::VConvolutionLayer* mImpl;
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
    kMAX = 0,              //!< Maximum over elements
    kAVERAGE = 1,          //!< Average over elements. If the tensor is padded, the count includes the padding
    kMAX_AVERAGE_BLEND = 2 //!< Blending between max and average pooling: (1-blendFactor)*maxPool + blendFactor*avgPool
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
    void setPrePadding(Dims const& padding) noexcept
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
    void setPostPadding(Dims const& padding) noexcept
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
    void setWindowSizeNd(Dims const& windowSize) noexcept
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
    //! \see getStrideNd()
    //!
    void setStrideNd(Dims const& stride) noexcept
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
    void setPaddingNd(Dims const& padding) noexcept
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
    void setWindowSize(int64_t windowSize) noexcept
    {
        mImpl->setWindowSize(windowSize);
    }

    //!
    //! \brief Get the LRN window size.
    //!
    //! \see getWindowStride()
    //!
    int64_t getWindowSize() const noexcept
    {
        return mImpl->getWindowSize();
    }

    //!
    //! \brief Set the LRN alpha value.
    //!
    //! The valid range is [-1e20, 1e20].
    //!
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
    //!
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
    //!
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
//! \note The input tensor is required to have at least 4 dimensions.
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
//! The following constraints must be satisfied to execute this layer on DLA:
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
    //! For example, consider an NCHW tensor as input.
    //!
    //! Bit 0 corresponds to the N dimension boolean.
    //! Bit 1 corresponds to the C dimension boolean.
    //! Bit 2 corresponds to the H dimension boolean.
    //! Bit 3 corresponds to the W dimension boolean.
    //! By default, softmax is performed on the axis which is the number of axes minus three. It is 0 if
    //! there are fewer than 3 axes. For example, if the input is NCHW, the default axis is C. If the input
    //! is NHW, then the default axis is N.
    //!
    //! For example, to perform softmax on axis R of a NPQRCHW input, set bit 3.
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
    //! \brief Set the number of output feature maps for the deconvolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    void setNbOutputMaps(int64_t nbOutputMaps) noexcept
    {
        mImpl->setNbOutputMaps(nbOutputMaps);
    }

    //!
    //! \brief Get the number of output feature maps for the deconvolution.
    //!
    //! \see setNbOutputMaps()
    //!
    int64_t getNbOutputMaps() const noexcept
    {
        return mImpl->getNbOutputMaps();
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
    void setNbGroups(int64_t nbGroups) noexcept
    {
        mImpl->setNbGroups(nbGroups);
    }

    //!
    //! \brief Get the number of groups for a deconvolution.
    //!
    //! \see setNbGroups()
    //!
    int64_t getNbGroups() const noexcept
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
    //!
    //! \see getPrePadding()
    //!
    void setPrePadding(Dims const& padding) noexcept
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
    //!
    //! \see getPostPadding()
    //!
    void setPostPadding(Dims const& padding) noexcept
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
    //! \see getKernelSizeNd()
    //!
    void setKernelSizeNd(Dims const& kernelSize) noexcept
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
    //! \see getStrideNd()
    //!
    void setStrideNd(Dims const& stride) noexcept
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
    void setPaddingNd(Dims const& padding) noexcept
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
    //!
    //! \see getKernelWeights(), setKernelWeights(), getBiasWeights(), setBiasWeights()
    //!
    using ILayer::setInput;

    //!
    //! \brief Set the multi-dimension dilation of the deconvolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! \see getDilationNd()
    //!
    void setDilationNd(Dims const& dilation) noexcept
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
//! Operations kAND, kOR, and kXOR must have inputs of DataType::kBOOL.
//!
//! Operation kPOW must have inputs of floating-point type or DataType::kINT8.
//!
//! All other operations must have inputs of floating-point type, DataType::kINT8, DataType::kINT32, or
//! DataType::kINT64.
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
//! The types of Data and Output must be the same, and Indices shall be DataType::kINT32 or DataType::kINT64.
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
    //!
    //! \warning Undefined behavior when used with GatherMode::kND.
    //!
    //! \see setGatherAxis()
    //!
    int32_t getGatherAxis() const noexcept
    {
        return mImpl->getGatherAxis();
    }

    //!
    //! \brief Set the number of leading dimensions of indices tensor to be handled elementwise.
    //!
    //! The gathering of indexing starts from the dimension of data[NbElementWiseDims:].
    //! The NbElementWiseDims must be less than the Rank of the data input.
    //!
    //! \param elementWiseDims number of dims to be handled as elementwise.
    //!
    //! Default: 0
    //!
    //! The value of nbElementWiseDims and GatherMode are checked during network validation:
    //!
    //! GatherMode::kDEFAULT: nbElementWiseDims can be 0 or 1.
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
//! \class IPluginV3Layer
//!
//! \brief Layer type for V3 plugins
//!
//! \see IPluginV3
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPluginV3Layer : public ILayer
{
public:
    //!
    //! \brief Get the plugin for the layer.
    //!
    //! \see IPluginV3
    //!
    IPluginV3& getPlugin() noexcept
    {
        return mImpl->getPlugin();
    }

protected:
    apiv::VPluginV3Layer* mImpl;
    virtual ~IPluginV3Layer() noexcept = default;
};

//!
//! \enum UnaryOperation
//!
//! \brief Enumerates the unary operations that may be performed by a Unary layer.
//!
//! Operations kNOT must have inputs of DataType::kBOOL.
//!
//! Operation kSIGN and kABS must have inputs of floating-point type, DataType::kINT8, DataType::kINT32 or
//! DataType::kINT64.
//!
//! Operation kISINF must have inputs of floating-point type.
//!
//! All other operations must have inputs of floating-point type.
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
    kROUND = 22, //!< Round to nearest even for floating-point data type.
    kISINF = 23, //!< Return true if input value equals +/- infinity for floating-point data type.
    kISNAN = 24, //!< Return true if input value is a NaN for floating-point data type.
};

//!
//! Maximum number of elements in UnaryOperation enum.
//!
//! \see UnaryOperation
//!
template <>
constexpr inline int32_t EnumMax<UnaryOperation>() noexcept
{
    return 25;
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
//! The padding layer adds zero-padding at the start and end of the input tensor. It supports padding
//! only the last two dimensions. Applying negative padding results in cropping of the input.
//!
//! To pad across any subset of dimensions, use ISliceLayer with SampleMode::kFILL.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPaddingLayer : public ILayer
{
public:
    //!
    //! \brief Set the padding that is applied at the start of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount.
    //!
    //! \warning Only 2 dimensional padding is currently supported.
    //!
    //! \see getPrePaddingNd
    //!
    void setPrePaddingNd(Dims const& padding) noexcept
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
    void setPostPaddingNd(Dims const& padding) noexcept
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

//!
//! \struct Permutation
//!
//! \brief Represents a permutation of dimensions.
//!
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
//! The layer has an optional second input. If present, it must be a 1D tensor of type Int32 or Int64,
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
    //! Avoid using -1 if the input can have zero volume and any of the other
    //! reshape dimensions can be zero (after resolving special treatment of 0),
    //! because the solution for the -1 becomes indeterminate and TensorRT will report an error.
    //!
    //! The product of the new dimensions must be equal to the product of the old.
    //!
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    void setReshapeDimensions(Dims const& dimensions) noexcept
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
    //! - 1: The dimensions for the reshape operation, as a 1D tensor of type Int32 or Int64.
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
    kWRAP = 1,                                     //!< Coordinates wrap around periodically.
    kCLAMP = 2,                                    //!< Out of bounds indices are clamped to bounds.
    kFILL = 3,                                     //!< Use fill input value when coordinates are out of bounds.
    kREFLECT = 4, //!< Coordinates reflect. The axis of reflection is the middle of the perimeter pixel and the
                  //!< reflections are repeated indefinitely within the padded regions. Repeats values for a single
                  //!< pixel and throws error for zero pixels.
};

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
//! Static slice layers can also optionally specify axes through the get/set accessor functions of the ISliceLayer.
//! Dynamic slice specifies one or more of start, size, stride, or axes as ITensors, by using ILayer::setInput to add
//! a second, third, fourth, or sixth input respectively. The corresponding Dims are used if an input
//! is missing or null.
//!
//! An application can determine if the ISliceLayer has a dynamic output shape based on whether
//! the size or axes input is present and non-null.
//!
//! The slice layer selects for each dimension a start location from within the input tensor, and
//! copies elements to the output tensor using the specified stride across the input tensor.
//! Start, size, and stride tensors must be 1D tensors of type Int32 or Int64 if not specified via Dims.
//!
//! An example of using slice on a tensor:
//! input = {{0, 2, 4}, {1, 3, 5}}
//! start = {1, 0}
//! size = {1, 2}
//! stride = {1, 2}
//! output = {{1, 5}}
//!
//! If axes are provided then starts, ends, and strides must have the same length as axes
//! and specifies a subset of dimensions to slice. If axes are not provided, starts, ends, and strides
//! must be of the same length as the rank of the input tensor.
//!
//! An example of using slice on a tensor with axes specified:
//! input = {{0, 2, 4}, {1, 3, 5}}
//! start = {1}
//! size = {2}
//! stride = {1}
//! axes = {1}
//! output = {{2, 4}, {3, 5}}
//!
//! When the sampleMode is kCLAMP or kREFLECT, for each input dimension, if its size is 0 then the corresponding output
//! dimension must be 0 too.
//!
//! When the sampleMode is kFILL, the fifth input to the slice layer is used to determine the value to fill in out-of-bound
//! indices. It is an error to specify the fifth input in any other sampleMode.
//!
//! A slice layer can produce a shape tensor if the following conditions are met:
//!
//! * start, size, and stride are build time constants, either as static Dims or as constant input tensors.
//! * axes, if provided, are build time constants, either as static Dims or as a constant input tensor.
//! * The number of elements in the output tensor does not exceed 2 * Dims::MAX_DIMS.
//!
//! The input tensor is a shape tensor if the output is a shape tensor.
//!
//! The following constraints must be satisfied to execute this layer on DLA:
//! * start, size, and stride are build time constants, either as static Dims or as constant input tensors.
//! * axes, if provided, are build time constants, either as static Dims or as a constant input tensor.
//! * sampleMode is kDEFAULT, kWRAP, or kFILL.
//! * Strides are 1 for all dimensions.
//! * Slicing is not performed on the first dimension.
//! * The input tensor has four dimensions.
//! * For kFILL sliceMode, the fill value input is a scalar output of an IConstantLayer with value 0 that is not
//!   consumed by any other layer.
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
    void setStart(Dims const& start) noexcept
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
    void setSize(Dims const& size) noexcept
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
    void setStride(Dims const& stride) noexcept
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
    void setMode(SampleMode mode) noexcept
    {
        mImpl->setMode(mode);
    }

    //!
    //! \brief Get the slice mode.
    //!
    //! \see setMode()
    //!
    SampleMode getMode() const noexcept
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
    //! - 1: The start tensor to begin slicing, as a 1D tensor of type Int32 or Int64.
    //! - 2: The size tensor of the resulting slice, as a 1D tensor of type Int32 or Int64.
    //! - 3: The stride of the slicing operation, as a 1D tensor of type Int32 or Int64.
    //! - 4: Value for the kFILL slice mode. The fill value data type should either be the same
    //!      or be implicitly convertible to the input data type.
    //!      Implicit data type conversion is supported among kFLOAT, kHALF, kINT8, and kFP8 data types.
    //!      This input is disallowed for other modes.
    //! - 5: The axes tensor indicating the corresponding axes that start, size, and stride
    //!      should apply to, as a 1D tensor or type Int32 or Int64. Negative values for axes
    //!      indicate indexing from the back of the input tensor. Values must be unique and be
    //!      within the interval of [-rank(input), rank(input)-1].
    //!
    //! Using the corresponding setter resets the input to null.
    //!
    //! If this function is called with a value greater than 0, then the function getNbInputs() changes
    //! from returning 1 to index + 1.
    //!
    using ILayer::setInput;

    //!
    //! \brief Set the axes for this ISliceLayer.
    //!
    //! \param axes The axes on which the starts, ends, and strides parameters of the slice apply to.
    //!
    //! If a sixth input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getAxes
    //!
    void setAxes(Dims const& axes) noexcept
    {
        mImpl->setAxes(axes);
    }

    //!
    //! \brief Get the axes for this ISliceLayer.
    //!
    //! \return The axes on which the starts, ends, and strides parameters of this slice apply to.
    //!
    //! If the sixth input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setAxes
    //!
    Dims getAxes() const noexcept
    {
        return mImpl->getAxes();
    }

protected:
    apiv::VSliceLayer* mImpl;
    virtual ~ISliceLayer() noexcept = default;
};

//! \class IShapeLayer
//!
//! \brief Layer type for getting shape of a tensor.
//!
//! This layer sets the output to a 1D tensor of type Int64 with the dimensions of the input tensor.
//!
//! For example, if the input is a four-dimensional tensor (of any type) with
//! dimensions [2,3,5,7], the output tensor is a one-dimensional Int64 tensor
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
//! This layer can accept both static and dynamic k. Static k can be set through the addTopK() API function,
//! or accessed using the getK() and setK() functions after layer creation. For dynamic k, use the setInput()
//! method to pass in k as a tensor with index 1, which overrides the static k value in calculations.
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
    //! \brief Set the static k value for the layer.
    //!
    //! Currently only values up to 3840 are supported.
    //!
    //! If a second input to this layer has been set, it will be reset to null by this method.
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
    //! This function will return the static k value passed into addTopK(), or the value passed into setK().
    //!
    //! If a second layer input is present and non-null, this function returns -1.
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

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index The index of the input to modify.
    //! \param tensor The new input tensor.
    //!
    //! For a TopK layer, the values 0-1 are valid.
    //! The indices are as follows:
    //!
    //! - 0: Input data tensor.
    //! - 1: A scalar Int32 tensor containing a positive value corresponding to the number of top
    //!      elements to retrieve. Values larger than 3840 will result in a runtime error. If provided,
    //!      this will override the static k value in calculations.
    //!
    using ILayer::setInput;

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
    kNONE = 0,

    //! Like kNONE, but transpose the matrix dimensions.
    kTRANSPOSE = 1,

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
    kVECTOR = 2,
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
    //!
    //! \param index Input tensor number (0 or 1).
    //! \param op New operation.
    //!
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
//! within TensorRT 8.x releases. TensorRT 10.0 onwards it is an error if the network output tensor type is incompatible
//! with the layer output type. E.g., implicit conversion from kFLOAT to kINT32 is not allowed, Use
//! setOutputType(DataType::kINT32) to explict convert kFLOAT to kINT32.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IIdentityLayer : public ILayer
{
protected:
    apiv::VIdentityLayer* mImpl;
    virtual ~IIdentityLayer() noexcept = default;
};

//! \class ICastLayer
//!
//! \brief A cast layer in a network.
//!
//! This layer casts a given tensor to the datatype specified by \p toType.
//!
class ICastLayer : public ILayer
{
public:
    //!
    //! \brief Set cast layer output type.
    //!
    //! \param toType The DataType of the output tensor.
    //!
    //! Set the output type of the cast layer.
    //!
    void setToType(DataType toType) noexcept
    {
        mImpl->setToType(toType);
    }

    //!
    //! \brief Return cast layer output type.
    //!
    //! \return toType parameter set during layer creation or by setToType().
    //! The return value is the output type of the cast layer.
    //!
    DataType getToType() const noexcept
    {
        return mImpl->getToType();
    }

protected:
    apiv::VCastLayer* mImpl;
    virtual ~ICastLayer() noexcept = default;
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
    //! The output type is weights.type. If the network is weakly typed and the weights have a real type,
    //! the output type might be different per TensorRT's type conversion rules.
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
    void setDimensions(Dims const& dimensions) noexcept
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
//!     -   InterpolationMode::kNEAREST - resizes last `m` dimensions of N-D, where 0 < m <= min(8, N) and N > 0
//!     -   InterpolationMode::kLINEAR - resizes last `m` dimensions of N-D, where 0 < m <= min(3, N) and N > 0
//!
//! Default resize mode is InterpolationMode::kNEAREST.
//!
//! The coordinates in the output tensor are mapped to coordinates in the input tensor using a function set by calling
//! setCoordinateTransformation(). The default for all InterpolationMode settings (nearest, linear, bilinear, etc.) is
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
    void setOutputDimensions(Dims const& dimensions) noexcept
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
    //! \see InterpolationMode
    //!
    void setResizeMode(InterpolationMode interpolationMode) noexcept
    {
        mImpl->setResizeMode(interpolationMode);
    }

    //!
    //! \brief Get resize mode for an input tensor.
    //!
    //! \return The resize mode.
    //!
    InterpolationMode getResizeMode() const noexcept
    {
        return mImpl->getResizeMode();
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
    //! - 1: The output dimensions, as a 1D tensor of type Int32 or Int64.
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

//!
//! \enum Enum that describes kinds of loop outputs.
//!
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

//!
//! \enum Enum that describes kinds of trip limits.
//!
enum class TripLimit : int32_t
{

    kCOUNT = 0, //!< Tensor is a scalar of type kINT32 or kINT64 that contains the trip count.
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

//!
//! \class ILoopBoundaryLayer
//!
//! \brief This is a base class for Loop boundary layers.
//!
//! The loop boundary layers are used to define loops within a network, enabling the implementation
//! of recurrences. The boundary layers for a loop are created by class ILoop.
//!
//! There are four kinds of boundary layers.
//! * ITripLimitLayer: controls the number of loop iterations.
//! * IIterationLayer: iterates over an input tensor.
//! * IRecurrenceLayer: returns an initial value or value from the previous loop iteration.
//! * ILoopOutputLayer: generates an output tensor from the loop iterations.
class ILoopBoundaryLayer : public ILayer
{
public:
    //!
    //! \brief Get a pointer to ILoop associated with this boundary layer.
    //!
    ILoop* getLoop() const noexcept
    {
        return mBoundary->getLoop();
    }

protected:
    virtual ~ILoopBoundaryLayer() noexcept = default;
    apiv::VLoopBoundaryLayer* mBoundary;
};

//!
//! \class IIfConditionalBoundaryLayer
//!
//! \brief This is a base class for Conditional boundary layers.
//!
//! Boundary layers are used to demarcate the boundaries of Conditionals.
//!
class IIfConditionalBoundaryLayer : public ILayer
{
public:
    //!
    //! \brief Get a pointer to the IIfConditional associated with this boundary layer.
    //!
    IIfConditional* getConditional() const noexcept
    {
        return mBoundary->getConditional();
    }

protected:
    virtual ~IIfConditionalBoundaryLayer() noexcept = default;
    apiv::VConditionalBoundaryLayer* mBoundary;
};

//!
//! \class IConditionLayer
//!
//! \brief This layer represents a condition input to an IIfConditional.
//!
class IConditionLayer : public IIfConditionalBoundaryLayer
{
public:
protected:
    virtual ~IConditionLayer() noexcept = default;
    apiv::VConditionLayer* mImpl;
};

//!
//! \class IIfConditionalOutputLayer
//!
//! \brief This layer represents an output of an IIfConditional.
//!
//! An IIfConditionalOutputLayer has two inputs and one output.
//!
//! \see IIfConditional::addOutput
//!
class IIfConditionalOutputLayer : public IIfConditionalBoundaryLayer
{
public:
protected:
    virtual ~IIfConditionalOutputLayer() noexcept = default;
    apiv::VConditionalOutputLayer* mImpl;
};

//!
//! \class IIfConditionalInputLayer
//!
//! \brief This layer represents an input to an IIfConditional.
//!
class IIfConditionalInputLayer : public IIfConditionalBoundaryLayer
{
public:
protected:
    virtual ~IIfConditionalInputLayer() noexcept = default;
    apiv::VConditionalInputLayer* mImpl;
};

//!
//! \class IIfConditional
//!
//! \brief Helper for constructing conditionally-executed subgraphs.
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
//! trueSubgraph represents a network subgraph that is executed when condition evaluates to True.
//! falseSubgraph represents a network subgraph that is executed when condition evaluates to False.
//!
//! The following constraints apply to If-conditionals:
//! - Both the trueSubgraph and falseSubgraph must be defined.
//! - The number of output tensors in both subgraphs is the same.
//! - Corresponding output tensors from the true/false subgraphs have the same type and shape.
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
    //! The shapes of the two tensors must be equal unless the condition is a build-time constant.
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

//!
//! \class IRecurrenceLayer
//!
//! \brief A recurrence layer in a network definition.
//!
//! The recurrence layer allows a loop iteration to compute a result from a value computed in the previous iteration.
//!
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
//! \class ILoopOutputLayer
//!
//! \brief An ILoopOutputLayer is the sole way to get output from a loop.
//!
//! The first input tensor must be defined inside the loop; the output tensor is outside the loop.
//! The second input tensor, if present, must be defined outside the loop.
//!
//! If getLoopOutput() is kLAST_VALUE, a single input must be provided,
//! and that input must be from an IRecurrenceLayer in the same loop.
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
    //!
    //! \brief Get which kind a loop output has.
    //!
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

    //!
    //! \brief Get axis being concatenated over.
    //!
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
    //! - 1: The concatenation length scalar value, must come from outside the loop, as a 0D shape tensor of type Int32 or Int64.
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    //!
    using ILayer::setInput;

protected:
    virtual ~ILoopOutputLayer() noexcept = default;
    apiv::VLoopOutputLayer* mImpl;
};

//!
//! \class ITripLimitLayer
//!
//! \brief A layer that represents a trip-count limiter.
//!
//! The trip limit layer sets the execution condition for loops, using kCOUNT to define the number of iterations or
//! kWHILE for a conditional loop. A loop can have one of each kind of limit, in which case the loop exits when
//! the trip count is reached or the condition becomes false.
//!
//! See INetworkDefinition::addTripLimit().
//!
class ITripLimitLayer : public ILoopBoundaryLayer
{
public:
    //!
    //! \brief Get a trip limiter type.
    //!
    TripLimit getTripLimit() const noexcept
    {
        return mImpl->getTripLimit();
    }

protected:
    virtual ~ITripLimitLayer() noexcept = default;
    apiv::VTripLimitLayer* mImpl;
};

//!
//! \class IIteratorLayer
//!
//! \brief A layer to do iterations.
//!
//! The iterator layer iterates over a tensor along the given axis and in the given direction.
//! It enables each loop iteration to inspect a different slice of the tensor.
//!
//! \see ILoop::addIterator()
//!
class IIteratorLayer : public ILoopBoundaryLayer
{
public:
    //!
    //! \brief Set axis to iterate over.
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

    //!
    //! \brief Get axis being iterated over.
    //!
    int32_t getAxis() const noexcept
    {
        return mImpl->getAxis();
    }

    //!
    //! \brief Set iteration order to be reverse.
    //!
    //! For reverse=false, the layer is equivalent to addGather(tensor, I, 0) where I is a
    //! scalar tensor containing the loop iteration number.
    //! For reverse=true, the layer is equivalent to addGather(tensor, M-1-I, 0) where M is the trip count
    //! computed from TripLimits of kind kCOUNT.
    //! The default is reverse=false.
    //!
    void setReverse(bool reverse) noexcept
    {
        mImpl->setReverse(reverse);
    }

    //!
    //! \brief Check if the iteration order is reverse.
    //!
    //! \return True if and only if reversing input.
    //!
    bool getReverse() const noexcept
    {
        return mImpl->getReverse();
    }

protected:
    virtual ~IIteratorLayer() noexcept = default;
    apiv::VIteratorLayer* mImpl;
};

//!
//! \class ILoop
//!
//! \brief Helper for creating a recurrent subgraph.
//!
//! An ILoop defines a loop within a network. It supports the implementation of recurrences,
//! which are crucial for iterative computations, such as RNNs for natural language processing and
//! time-series analysis.
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

    //!
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
//! \class ISelectLayer
//!
//! \brief Select elements from two data tensors based on a condition tensor.
//!
//! The select layer makes elementwise selections from two data tensors based on a condition tensor,
//! behaving similarly to the numpy.where function with three parameters.
//! The three input tensors must share the same rank. Multidirectional broadcasting is supported.
//! The output tensor has the dimensions of the inputs AFTER applying the broadcast rule.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISelectLayer : public ILayer
{
protected:
    virtual ~ISelectLayer() noexcept = default;
    apiv::VSelectLayer* mImpl;
};

//!
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
    //! Compute each value via an affine function of its indices.
    //! For example, suppose the parameters for the IFillLayer are:
    //!
    //! * Dimensions = [3,4]
    //! * Alpha = 1
    //! * Beta = [100,10]
    //!
    //! Element [i,j] of the output is Alpha + Beta[0]*i + Beta[1]*j.
    //! Thus the output matrix is:
    //!
    //!      1  11  21  31
    //!    101 111 121 131
    //!    201 211 221 231
    //!
    //! A static beta b is implicitly a 1D tensor, i.e. Beta = [b].
    kLINSPACE = 0,

    //! Randomly draw values from a uniform distribution.
    kRANDOM_UNIFORM = 1,

    //! Randomly draw values from a normal distribution.
    kRANDOM_NORMAL = 2
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
//! \class IFillLayer
//!
//! \brief Generate a tensor according to a specified mode.
//!
//! The fill layer generates a tensor with values that are drawn from a random distribution
//! or an affine function of their indices, as specified by the FillMode.
//!
//! When an IFillLayer is initially added to a network, all of its parameters are static.
//! Each parameter may be changed to dynamic by setting a corresponding input.
//! A parameter is considered dynamic even if that input is the output of an IConstantLayer.
//! The inputs for each parameter are:
//!
//! - 0: Dimensions
//! - 1: Alpha
//! - 2: Beta
//!
//! The parameter Dimensions describes the shape of the output. If the Dimensions input is provided,
//! it must be a 1D tensor of type Int32 or Int64 whose length is computable by constant folding.
//!
//! The meanings of Alpha and Beta depend on the mode, as described in IFillLayer::setAlpha(),
//! IFillLayer::setBeta(), and IFillLayer::setInput(). Parameters Alpha and Beta must both be static
//! or both be dynamic.
//!
//! An IFillLayer can produce a shape tensor if the following restrictions are met:
//!
//! * The FillOperation is kLINSPACE.
//! * The output has type Int32, Int64, or Float.
//! * The volume of the output is within the volume limit imposed on shape tensors.
//! * If input 0 exists, the values of input 0 must be computable by constant folding.
//!
//! \see FillOperation
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
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
    void setDimensions(Dims const& dimensions) noexcept
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
    //! If input 1 exists, it is reset to null by this method.
    //!
    //! \see getAlpha, setAlphaInt64
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
    //! If input 2 exists, it is reset to null by this method.
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
    //! \see setBeta, setBetaInt64
    //!
    double getBeta() const noexcept
    {
        return mImpl->getBeta();
    }

    //!
    //! \brief Replace an input of this layer with a specific tensor.
    //!
    //! \param index the index of the input to set.
    //! \param tensor the new input tensor
    //!
    //! The three inputs correspond to these setters of IFillLayer:
    //!
    //! - 0: setDimensions
    //! - 1: setAlpha
    //! - 2: setBeta
    //!
    //! The following descriptions give more intuitive names for the inputs.
    //!
    //! Indices for kLINSPACE are:
    //!
    //! - 0: Shape, a 1D shape tensor, specifies the output tensor's dimensions.
    //! - 1: Start, a scalar, specifies the start value.
    //! - 2: Delta, a 1D tensor, specifies the delta value for each dimension.
    //!
    //! Indices for kRANDOM_UNIFORM are:
    //!
    //! - 0: Shape, a 1D shape tensor, specifies the output tensor's dimensions.
    //! - 1: Minimum, a scalar, specifies the minimum random value.
    //! - 2: Maximum, a scalar, specifies the maximal random value.
    //!
    //! Indices for kRANDOM_NORMAL are:
    //!
    //! - 0: Shape, a 1D shape tensor, specifies the output tensor's dimensions.
    //! - 1: Mean, a scalar, specifies the mean of the normal distribution,.
    //! - 2: Scale, a scalar, specifies the standard deviation of the normal distribution.
    //!
    //! Using the corresponding setter resets the input to null.
    //!
    //! If either inputs 1 or 2 is non-null, then both must be non-null and have the same data type.
    //!
    //! If this function is called for an index greater or equal to getNbInputs(),
    //! then afterwards getNbInputs() returns index + 1, and any missing intervening
    //! inputs are set to null.
    //!
    using ILayer::setInput;

    //!
    //! \brief Set the alpha parameter with int64 datatype.
    //!
    //! \param alpha has different meanings for each operator:
    //!
    //! Operation          | Usage
    //! kLINSPACE          | the start value, defaults to 0;
    //! kRANDOM_UNIFORM    | the minimum value, defaults to 0;
    //! kRANDOM_NORMAL     | the mean of the normal distribution, default is 0;
    //!
    //! If a third input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getAlphaInt64
    //
    void setAlphaInt64(int64_t alpha) noexcept
    {
        mImpl->setAlphaInt64(alpha);
    }

    //!
    //! \brief Get the value of alpha parameter with int64 datatype.
    //!
    //! \return A int64 value of alpha.
    //!
    //! If the second input is present and non-null,
    //! this function returns -1.
    //!
    //! \see setAlphaInt64
    //!
    int64_t getAlphaInt64() const noexcept
    {
        return mImpl->getAlphaInt64();
    }

    //!
    //! \brief Set the beta parameter with int64 datatype.
    //!
    //! \param beta has different meanings for each operator:
    //!
    //! Operation          | Usage
    //! kLINSPACE          | the delta value, defaults to 1;
    //! kRANDOM_UNIFORM    | the maximal value, defaults to 1;
    //! kRANDOM_NORMAL     | the standard deviation of the normal distribution, default is 1;
    //!
    //! If a third input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getBetaInt64
    //!
    void setBetaInt64(int64_t beta) noexcept
    {
        mImpl->setBetaInt64(beta);
    }

    //!
    //! \brief Get the value of beta parameter with int64 datatype.
    //!
    //! \return A int64 value of beta.
    //!
    //! If the third input is present and non-null,
    //! this function returns -1.0.
    //!
    //! \see setBetaInt64
    //!
    int64_t getBetaInt64() const noexcept
    {
        return mImpl->getBetaInt64();
    }

    //!
    //! \brief Return true if alpha/beta have type int64, false if they have type double.
    //!
    bool isAlphaBetaInt64() const noexcept
    {
        return mImpl->isAlphaBetaInt64();
    }

    //!
    //! \brief Set the fill layer output type.
    //!
    //! \param toType The DataType of the output tensor.
    //!
    //! Set the output type of the fill layer. Valid values are DataType::kFLOAT, DataType::kINT32,
    //! and DataType::kINT64.
    //! If the network is strongly typed, setToType must be used to set the output type, and use of setOutputType
    //! is an error. Otherwise, types passed to setOutputType and setToType must be the same.
    //!
    //! \see NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
    //!
    void setToType(DataType toType) noexcept
    {
        mImpl->setToType(toType);
    }

    //!
    //! \brief Get the fill layer output type.
    //!
    //! \return toType parameter set during layer creation or by setToType().
    //! The return value is the output type of the fill layer.
    //! The default value is DataType::kFLOAT.
    //!
    DataType getToType() const noexcept
    {
        return mImpl->getToType();
    }

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
//! quantize the data according to:
//! \p output = clamp(round(\p input / \p scale) + \p zeroPt)
//!
//! Rounding type is rounding-to-nearest ties-to-even (https://en.wikipedia.org/wiki/Rounding#Round_half_to_even).
//! Clamping range according to data type:
//! - FP8: [-448, 448]
//! - INT4: [-8, 7]
//! - INT8: [-128, 127]
//!
//! The first input (index 0) is the tensor to be quantized.
//! The second (index 1) and third (index 2) are the scale and zero point respectively.
//! \p scale and \p zeroPt should have identical dimensions, and rank lower or equal to 2.
//!
//! The \p zeroPt tensor is optional, and if not set, will be assumed to be zero. Its data type must match the
//! output data type. \p zeroPt must only contain zero-valued coefficients, because only symmetric quantization is
//! supported.
//! The \p scale value must be a scalar for per-tensor quantization, a 1D tensor for per-channel quantization, or the
//! same rank as the input tensor for block quantization (supported for DataType::kINT4 only). All \p scale
//! coefficients must have positive values. The size of the 1D \p scale tensor must match the size of the quantization
//! axis. For block quantization, the shape of \p scale tensor must match the shape of the input, except for one
//! dimension (the last or second to last dimension) in which blocking occurs.
//! The size of \p zeroPt must match the size of \p scale.
//!
//! The subgraph which terminates with the \p scale tensor must be a build-time constant. The same restrictions apply
//! to the \p zeroPt.
//! The output type, if constrained, must be constrained to DataType::kINT8, DataType::kFP8 or DataType::kINT4. The
//! input type, if constrained, must be constrained to DataType::kFLOAT, DataType::kHALF, or DataType::kBF16. The
//! output size is the same as the input size. The quantization axis is in reference to the input tensor's dimensions.
//!
//! IQuantizeLayer supports DataType::kFLOAT, DataType::kHALF, or DataType::kBF16 precision and will default to
//! DataType::kFLOAT precision during instantiation. For strongly typed networks, \p input data type must match the
//! \p scale data type.
//!
//! IQuantizeLayer supports DataType::kINT8, DataType::kFP8, or DataType::kINT4 output.
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
//! Block quantization is supported only for weight inputs of DataType::kINT4. As an example of blocked
//! operation, imagine a 2D RS weights input, R (dimension 0) as the blocking axis and B as the block size.
//! The scale is a 2D array of coefficients, with dimensions (R//B, S).
//!     For each r in R:
//!         For each s in S:
//!             output[r,s] = clamp(round(\p input[r,s] / \p scale[r//B, s]) + \p zeroPt[r//B, s])
//!
//! \note Only symmetric quantization is supported.
//! \note Currently the only allowed build-time constant \p scale and \p zeroPt subgraphs are:
//! 1. Constant -> Quantize
//! 2. Constant -> Cast -> Quantize
//!
//! \note The input tensor for this layer must not be a scalar.
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
    //! The axis value is used only for per-axis (per-channel) quantization.
    //!
    void setAxis(int32_t axis) noexcept
    {
        mImpl->setAxis(axis);
    }

    //!
    //! \brief Set the Quantize layer output type.
    //!
    //! \param toType The DataType of the output tensor.
    //!
    //! Set the output type of the quantize layer. Valid values are DataType::kINT8 and DataType::kFP8.
    //! If the network is strongly typed, setToType must be used to set the output type, and use of setOutputType
    //! is an error. Otherwise, types passed to setOutputType and setToType must be the same.
    //!
    //! \see NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
    //!
    void setToType(DataType toType) noexcept
    {
        mImpl->setToType(toType);
    }

    //!
    //! \brief Return the Quantize layer output type.
    //!
    //! \return toType parameter set during layer creation or by setToType().
    //! The return value is the output type of the quantize layer.
    //! The default value is DataType::kINT8.
    //!
    DataType getToType() const noexcept
    {
        return mImpl->getToType();
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
//! This layer accepts a quantized type input tensor, and uses the configured scale and zeroPt inputs to
//! dequantize the input according to:
//! \p output = (\p input - \p zeroPt) * \p scale
//!
//! The first input (index 0) is the tensor to be quantized.
//! The second (index 1) and third (index 2) are the scale and zero point respectively.
//! \p scale and \p zeroPt should have identical dimensions, and rank lower or equal to 2.
//!
//! The \p zeroPt tensor is optional, and if not set, will be assumed to be zero. Its data type must be identical to
//! the input's data type. \p zeroPt must only contain zero-valued coefficients, because only symmetric quantization is
//! supported.
//! The \p scale value must be a scalar for per-tensor quantization, a 1D tensor for per-channel quantization, or the
//! same rank as the input tensor for block quantization (supported for DataType::kINT4 only). All \p scale
//! coefficients must have positive values. The size of the 1D \p scale tensor must match the size of the quantization
//! axis. For block quantization, the shape of \p scale tensor must match the shape of the input, except for one
//! dimension (the last or second to last dimension) in which blocking occurs.
//! The size of \p zeroPt must match the size of \p scale.
//!
//! The subgraph which terminates with the \p scale tensor must be a build-time constant.  The same restrictions apply
//! to the \p zeroPt.
//! The output type, if constrained, must be constrained to DataType::kFLOAT, DataType::kHALF, or DataType::kBF16. The
//! input type, if constrained, must be constrained to DataType::kINT8, DataType::kFP8 or DataType::kINT4. The output
//! size is the same as the input size. The quantization axis is in reference to the input tensor's dimensions.
//!
//! IDequantizeLayer supports DataType::kINT8, DataType::kFP8 or DataType::kINT4 precision and will default to
//! DataType::kINT8 precision during instantiation. For strongly typed networks, \p input data type must be same as
//! \p zeroPt data type.
//!
//! IDequantizeLayer supports DataType::kFLOAT, DataType::kHALF, or DataType::kBF16 output. For strongly typed
//! networks, \p output data type is inferred from \p scale data type.
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
//! Block dequantization is supported only for input tensors with DataType::kINT4 that are rooted at an
//! IConstantLayer (i.e. weights). As an example of blocked operation, imagine a 2D RS weights input with R
//! (dimension 0) as the blocking axis and B as the block size. The scale is a 2D array of coefficients, with
//! dimensions (R//B, S).
//! For each r in R:
//!     For each s in S:
//!         output[r,s] = (\p input[r,s] - \p zeroPt[r//B, s]) * \p scale[r//B, s]
//!
//! \note Only symmetric quantization is supported.
//! \note Currently the only allowed build-time constant \p scale and \p zeroPt subgraphs are:
//! 1. Constant -> Quantize
//! 2. Constant -> Cast -> Quantize
//!
//! \note The input tensor for this layer must not be a scalar.
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

    //!
    //! \brief Set the Dequantize layer output type.
    //!
    //! \param toType The DataType of the output tensor.
    //!
    //! Set the output type of the dequantize layer. Valid values are DataType::kFLOAT and DataType::kHALF.
    //! If the network is strongly typed, setToType must be used to set the output type, and use of setOutputType
    //! is an error. Otherwise, types passed to setOutputType and setToType must be the same.
    //!
    //! \see NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
    //!
    void setToType(DataType toType) noexcept
    {
        mImpl->setToType(toType);
    }

    //!
    //! \brief Return the Dequantize layer output type.
    //!
    //! \return toType parameter set during layer creation or by setToType().
    //! The return value is the output type of the quantize layer.
    //! The default value is DataType::kFLOAT.
    //!
    DataType getToType() const noexcept
    {
        return mImpl->getToType();
    }

protected:
    virtual ~IDequantizeLayer() noexcept = default;
    apiv::VDequantizeLayer* mImpl;
};


//!
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
//! an arrow ('->') followed by subscripts for the output.
//! For example, "ij,jk->ik" is equivalent to "ij,jk".
//! Ellipsis ('...') can be used in place of subscripts to broadcast the dimensions.
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
//! \enum ScatterMode
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
//!   values to. Constraints on the rank q depend on the mode:
//!       ScatterMode::kND: q >= 1
//!       ScatterMode::kELEMENT: q must be the same as r
//! * Updates is a tensor of rank s >= 1 that provides the data
//!   to write to Output specified by its corresponding location in Indices.
//!   Constraints on the rank of Updates depend on the mode:
//!       ScatterMode::kND: s = r + q - shape(Indices)[-1] - 1
//!       Scattermode::kELEMENT: s = q = r
//! * Output is a tensor with the same dimensions as Data that stores the resulting values of the
//!   transformation. It must not be a shape tensor.
//! The types of Data, Update, and Output shall be the same, and Indices shall be of type DataType::kINT32 or
//! DataType::kINT64.
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
//!         foreach slice in indices[i_0,...,i_{q-2}]
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
//!                         output[n,c,indices[n,c,h,w],w] = updates[n,c,h,w]
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
//! * Depth is an 0D tensor of type Int32 or Int64, which contains the depth (number of classes) of the one-hot encoding.
//!   The depth tensor must be a positive build-time constant.
//! * Output is a tensor with rank = rank(indices)+1, where the added dimension contains the one-hot encoding.
//!   The data types of Output is equal to the Values data type.
//! * Axis is a scalar specifying to which dimension of the output one-hot encoding is added.
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

//!
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
//! \enum BoundingBoxFormat
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
//! Per batch item, boxes are initially sorted by their scores without regard to class. Only boxes up to a maximum of the TopK limit are considered for selection (per batch).
//! During selection, only overlapping boxes of the same class are compared, so that overlapping boxes of different classes do not suppress each other.
//!
//! For each batch item, the ordering of candidate bounding boxes with the same score is unspecified, but the ordering will be consistent across different runs for the same inputs.
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
//! \class IReverseSequenceLayer
//!
//! \brief A ReverseSequence layer in a network definition.
//!
//! This layer performs batch-wise reversal, which slices the input tensor along the axis batchAxis. For the
//! i-th slice, the operation reverses the first N elements, specified by the corresponding i-th value in
//! sequenceLens, along sequenceAxis and keeps the remaining elements unchanged. The output tensor will have
//! the same shape as the input tensor.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IReverseSequenceLayer : public ILayer
{
public:
    //!
    //! \brief Set the batch axis. Default is 1.
    //!
    //! batchAxis should be between zero (inclusive) and the rank of input (exclusive), and different from
    //! sequenceAxis. Otherwise, ErrorCode::kINVALID_ARGUMENT will be triggered.
    //!
    //! \see setBatchAxis()
    //!
    void setBatchAxis(int32_t batchAxis) noexcept
    {
        mImpl->setBatchAxis(batchAxis);
    }

    //!
    //! \brief Return the batch axis. Return 1 if no batch axis was set.
    //!
    //! \see getBatchAxis()
    //!
    int32_t getBatchAxis() const noexcept
    {
        return mImpl->getBatchAxis();
    }

    //!
    //! \brief Set the sequence axis. Default is 0.
    //!
    //! sequenceAxis should be between zero (inclusive) and the rank of input (exclusive), and different from
    //! batchAxis. Otherwise, ErrorCode::kINVALID_ARGUMENT will be triggered.
    //!
    //! \see setSequenceAxis()
    //!
    void setSequenceAxis(int32_t sequenceAxis) noexcept
    {
        mImpl->setSequenceAxis(sequenceAxis);
    }

    //!
    //! \brief Return the sequence axis. Return 0 if no sequence axis was set.
    //!
    //! \see getSequenceAxis()
    //!
    int32_t getSequenceAxis() const noexcept
    {
        return mImpl->getSequenceAxis();
    }

protected:
    apiv::VReverseSequenceLayer* mImpl;
    virtual ~IReverseSequenceLayer() noexcept = default;
}; // class IReverseSequenceLayer

//!
//! \class INormalizationLayer
//!
//! \brief A normalization layer in a network definition.
//!
//! The normalization layer performs the following operation:
//!
//! X - input Tensor
//! Y - output Tensor
//! S - scale Tensor
//! B - bias Tensor
//!
//! Y = (X - Mean(X, axes)) / Sqrt(Variance(X) + epsilon) * S + B
//!
//! Where Mean(X, axes) is a reduction over a set of axes, and Variance(X) = Mean((X - Mean(X, axes)) ^ 2, axes).
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class INormalizationLayer : public ILayer
{
public:
    //!
    //! \brief Set the epsilon value used for the normalization calculation.
    //!
    //! The default value of \p eps is 1e-5F.
    //!
    //! \param eps The epsilon value used for the normalization calculation.
    //!
    void setEpsilon(float eps) noexcept
    {
        return mImpl->setEpsilon(eps);
    }

    //!
    //! \brief Get the epsilon value used for the normalization calculation.
    //!
    //! \return The epsilon value used for the normalization calculation.
    //!
    float getEpsilon() const noexcept
    {
        return mImpl->getEpsilon();
    }

    //!
    //! \brief Set the reduction axes for the normalization calculation.
    //!
    //! \param axesMask The axes used for the normalization calculation.
    //!
    void setAxes(uint32_t axesMask) noexcept
    {
        return mImpl->setAxes(axesMask);
    }

    //!
    //! \brief Get the axes value used for the normalization calculation.
    //!
    //! \return The axes used for the normalization calculation.
    //!
    uint32_t getAxes() const noexcept
    {
        return mImpl->getAxes();
    }

    //!
    //! \brief Set the number of groups used to split the channels in the normalization calculation.
    //!
    //! The input tensor channels are divided into \p nbGroups groups, and normalization is performed per group.
    //! The channel dimension is considered to be the second dimension in a [N, C, H, W, ...] formatted tensor.
    //!
    //! The default \p nbGroups is 1.
    //!
    //! \warning It is an error to set \p nbGroups to a value that does not evenly divide into the number of channels
    //! of the input tensor.
    //!
    //! \warning When \p nbGroups is != 1, it is expected that the provided axesMask will have all bits corresponding
    //! to dimensions after the channel dimension set to 1, with all other bits set to 0.
    //!
    //! \param nbGroups The number of groups to split the channels into for the normalization calculation.
    //!
    void setNbGroups(int64_t nbGroups) noexcept
    {
        return mImpl->setNbGroups(nbGroups);
    }

    //!
    //! \brief Get the number of groups used to split the channels for the normalization calculation.
    //!
    //! \return The number of groups used to split the channel used for the normalization calculation.
    //!
    int64_t getNbGroups() const noexcept
    {
        return mImpl->getNbGroups();
    }

    //!
    //! \brief Set the compute precision of this layer.
    //!
    //! \param type The datatype used for the compute precision of this layer.
    //!
    //! The method is used to avoid overflow errors by controlling the normalization computation in
    //! mixed precision mode. The compute precision defaults to DataType::kFLOAT32.
    //! To override this default, use this method to set the desired compute precision.
    //!
    //! For a weakly typed network:
    //!
    //! * Method setOutputType() can still be called to control the output data type.
    //!
    //! * Method setPrecision() can still be called. The input data is cast to that precision before
    //!   being cast to the compute precision.
    //!
    //! Strongly typed network rejects calls to this method since the compute precision is typically
    //! controlled by casting the input tensors to the desired type.
    //!
    //! Only DataType::kFLOAT32 and DataType::kHALF are valid types for \p type.
    //!
    void setComputePrecision(DataType type) noexcept
    {
        return mImpl->setComputePrecision(type);
    }

    //!
    //! \brief Get the compute precision of this layer.
    //!
    //! \return The datatype used for the compute precision of this layer.
    //!
    DataType getComputePrecision() const noexcept
    {
        return mImpl->getComputePrecision();
    }

protected:
    apiv::VNormalizationLayer* mImpl;
    virtual ~INormalizationLayer() noexcept = default;
};

//!
//! \class ISqueezeLayer
//!
//! \brief Layer that represents a squeeze operation, removing unit dimensions of the input tensor
//! on a set of axes.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISqueezeLayer : public ILayer
{
public:
    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index The index of the input to modify.
    //! \param tensor The new input tensor.
    //!
    //! For a Squeeze layer, the values 0-1 are valid for index.
    //! The indices are as follows:
    //!
    //! - 0: Input data tensor.
    //! - 1: The axes to remove. Must resolvable to a constant Int32 or Int64 1D shape tensor.
    //!
    using ILayer::setInput;

protected:
    apiv::VSqueezeLayer* mImpl;
    virtual ~ISqueezeLayer() noexcept = default;
};

//!
//! \class IUnsqueezeLayer
//!
//! \brief Layer that represents an unsqueeze operation, which reshapes the input tensor by inserting unit-length dimensions at specified axes of the output.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IUnsqueezeLayer : public ILayer
{
public:
    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index The index of the input to modify.
    //! \param tensor The new input tensor.
    //!
    //! For an Unsqueeze layer, the values 0-1 are valid for index.
    //! The indices are as follows:
    //!
    //! - 0: Input data tensor.
    //! - 1: The output axes at which unit-length dimensions are inserted. Must resolvable to a constant Int32 or Int64 1D shape tensor.
    //!
    using ILayer::setInput;

protected:
    apiv::VUnsqueezeLayer* mImpl;
    virtual ~IUnsqueezeLayer() noexcept = default;
};

//!
//! \class INetworkDefinition
//!
//! \brief A network definition for input to the builder.
//!
//! A network definition defines the structure of the network, and combined with a IBuilderConfig, is built
//! into an engine using an IBuilder. An INetworkDefinition can have all dimensions explicit, full dims mode, in the
//! network definition. The former mode, i.e. the implicit batch size mode, has been deprecated.
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
    //! Each input and output tensor must have a unique name.
    //! The volume must be less than 2^31 elements.
    //!
    //! For networks with wildcard dimensions, the volume
    //! is based on the maxima specified by an IOptimizationProfile.Dimensions are normally non-negative integers. The
    //! exception is that in networks with all explicit dimensions, -1 can be used as a wildcard for a dimension to
    //! be specified at runtime. Input tensors with such a wildcard must have a corresponding entry in the
    //! IOptimizationProfiles indicating the permitted extrema, and the input dimensions must be set by
    //! IExecutionContext::setInputShape. Different IExecutionContext instances can have different dimensions.
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
    ITensor* addInput(char const* name, DataType type, Dims const& dimensions) noexcept
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
    //! \brief Mark a tensor as a debug tensor.
    //!
    //! A debug tensor can be optionally emitted at runtime.
    //! Note that tensor names are required to specify debug
    //! tensors at runtime.
    //!
    //! \param tensor Tensor to be marked as debug
    //!
    //! \return True if tensor successfully marked (or was already marked), false otherwise.
    //!
    //! \see unmarkDebug(), IExecutionContext::setDebugListener(), ITensor::setName()
    //!
    bool markDebug(ITensor& tensor) noexcept
    {
        return mImpl->markDebug(tensor);
    }

    //!
    //! \brief Unmark a tensor as a debug tensor.
    //!
    //! Remove the marking of a tensor as a debug tensor.
    //!
    //! \param tensor Tensor to be unmarked as debug.
    //!
    //! \return True if tensor successfully unmarked (or was already unmarked), false otherwise.
    //!
    //! \see markDebug(), IExecutionContext::setDebugListener()
    //!
    bool unmarkDebug(ITensor& tensor) noexcept
    {
        return mImpl->unmarkDebug(tensor);
    }

    //!
    //! \brief Check if a tensor is marked as debug tensor.
    //!
    //! \return true if tensor is marked as debug tensor, false otherwise.
    //!
    bool isDebugTensor(nvinfer1::ITensor const& tensor) const noexcept
    {
        return mImpl->isDebugTensor(tensor);
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
    //!
    //! \warning Int32 and Int64 are valid only for activation type kRELU.
    //!
    //! \return The new activation layer, or nullptr if it could not be created.
    //!
    IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept
    {
        return mImpl->addActivation(input, type);
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
    ILRNLayer* addLRN(ITensor& input, int64_t window, float alpha, float beta, float k) noexcept
    {
        return mImpl->addLRN(input, window, alpha, beta, k);
    }

    //!
    //! \brief Add a Scale layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!              This tensor must have at least 4 dimensions.
    //! \param mode The scaling mode.
    //! \param shift The shift value.
    //! \param scale The scale value.
    //! \param power The power value.
    //!
    //! If the weights are available, then the size of weights are dependent on the ScaleMode.
    //! For ScaleMode::kUNIFORM, the number of weights equals 1.
    //! For ScaleMode::kCHANNEL, the number of weights equals the channel dimension.
    //! For ScaleMode::kELEMENTWISE, the number of weights equals the product of the last three dimensions of the input.
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
    //! \param depth - a shape tensor containing the width of the added one-hot dimension.
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
    //! \brief Add a reduce layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The reduction operation to perform.
    //! \param reduceAxes The reduction dimensions.
    //!        The bit in position i of bitmask reduceAxes corresponds to explicit dimension i if result.
    //!        E.g., the least significant bit corresponds to the first explicit dimension and the next to least
    //!        significant bit corresponds to the second explicit dimension.
    //! \param keepDimensions The boolean that specifies whether or not to keep the reduced dimensions in the
    //! output of the layer.
    //!
    //! The reduce layer works by performing an operation specified by \p operation to reduce the tensor \p input
    //! across the axes specified by \p reduceAxes.
    //!
    //! \see IReduceLayer
    //!
    //! \warning If output is an Int32 or Int64 shape tensor, ReduceOperation::kAVG is unsupported.
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
    //! \param k The number of elements to keep. For dynamic k, use the setInput() method to pass in k as a tensor
    //!        instead, which will override the static k value passed here in calculations.
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
    //! \warning The input and bounds tensors should be 3D tensors.
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
    //! If a wildcard dimension is used, the volume of the runtime dimensions must equal
    //! the number of weights specified.
    //!
    //! \warning DataType::kUINT8 not supported.
    //!
    IConstantLayer* addConstant(Dims const& dimensions, Weights weights) noexcept
    {
        return mImpl->addConstant(dimensions, weights);
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
    //! \brief Add a cast layer.
    //!
    //! \param input The input tensor to the layer.
    //! \param toType The DataType of the output tensor
    //!
    //! \see ICastLayer
    //!
    //! \return The new cast layer, or nullptr if it could not be created.
    //!
    ICastLayer* addCast(ITensor& input, DataType toType) noexcept
    {
        return mImpl->addCast(input, toType);
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
    //! \brief Add a plugin layer implementing the IPluginV3 interface to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param shapeInputs Shape tensor inputs to the layer.
    //! \param nbShapeInputs The number of shape tensor inputs.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginV3Layer
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    IPluginV3Layer* addPluginV3(ITensor* const* inputs, int32_t nbInputs, ITensor* const* shapeInputs,
        int32_t nbShapeInputs, IPluginV3& plugin) noexcept
    {
        return mImpl->addPluginV3(inputs, nbInputs, shapeInputs, nbShapeInputs, plugin);
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
    ISliceLayer* addSlice(ITensor& input, Dims const& start, Dims const& size, Dims const& stride) noexcept
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
    //! \return The new shape layer, or nullptr if it could not be created.
    //!
    IShapeLayer* addShape(ITensor& input) noexcept
    {
        return mImpl->addShape(input);
    }

    //!
    //! \brief Query whether the network was created with an implicit batch dimension.
    //!
    //! \return Always false since TensorRT 10.0 does not support an implicit batch dimension.
    //!
    //! \see createNetworkV2
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is not supported since TensorRT 10.0.
    //!
    TRT_DEPRECATED bool hasImplicitBatchDimension() const noexcept
    {
        return mImpl->hasImplicitBatchDimension();
    }

    //!
    //! \brief Get the network definition creation flags for this network definition object. Defaults to 0.
    //!
    //! \return The network definition creation options as a bitmask.
    //!
    NetworkDefinitionCreationFlags getFlags() const noexcept
    {
        return mImpl->getFlags();
    }

    //!
    //! \brief Returns true if the network definition creation flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    bool getFlag(NetworkDefinitionCreationFlag networkDefinitionCreationFlag) const noexcept
    {
        return mImpl->getFlag(networkDefinitionCreationFlag);
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
    //! \warning Tensors of type Int32, Int64, Bool, or UInt8 are not allowed as inputs.
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
        ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
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
    IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims const& windowSize) noexcept
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
        ITensor& input, int64_t nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
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
    //! For ScaleMode::kUNIFORM, the number of weights equals 1.
    //! For ScaleMode::kCHANNEL, the number of weights equals the channel dimension.
    //! For ScaleMode::kELEMENTWISE, the number of weights equals the product of all input dimensions at channelAxis and
    //! beyond.
    //!
    //! For example, if the inputs dimensions are [A,B,C,D,E,F], and channelAxis=2:
    //! For ScaleMode::kUNIFORM, the number of weights is equal to 1.
    //! For ScaleMode::kCHANNEL, the number of weights is C.
    //! For ScaleMode::kELEMENTWISE, the number of weights is C*D*E*F.
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

    //!
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
    //! \brief Add a loop to the network.
    //!
    //! An ILoop provides a way to specify a recurrent subgraph.
    //!
    //! \return Pointer to ILoop that can be used to add loop-boundary layers for the loop.
    //!
    //! \see ILoop
    //!
    ILoop* addLoop() noexcept
    {
        return mImpl->addLoop();
    }

    //!
    //! \brief Add an if-then-else to the network.
    //!
    //! An IIfConditional provides a way to conditionally execute parts of the network.
    //!
    //! \return Pointer to the IIfConditional that can be used to add conditional-boundary layers
    //!         for the if-then-else.
    //!
    //! \see IIfConditional
    //!
    IIfConditional* addIfConditional() noexcept
    {
        return mImpl->addIfConditional();
    }

    //!
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

    //!
    //! \brief Add a fill layer to the network.
    //!
    //! \param dimensions The output tensor dimensions if input 0 is missing.
    //! \param op The fill operation that the layer applies.
    //!
    //! \warning For FillOperation::kLINSPACE, dimensions.nbDims must be 1 for static start/delta. If delta is provided
    //! as a 1D tensor, the length of delta must match dimensions.nbDims.
    //!
    //! This layer is non-deterministic across subsequent calls as the same inputs will produce different
    //! output tensors if \p op is either FillOperation::kRANDOM_UNIFORM or FillOperation::kRANDOM_NORMAL
    //! due to random state being shared across calls. The output tensors generated are determinstic when
    //! starting from the same initial state.
    //!
    //! \see IFillLayer
    //!
    //! \return The new fill layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 9.0. Superseded by three-argument addFill.
    //!
    TRT_DEPRECATED IFillLayer* addFill(Dims const& dimensions, FillOperation op) noexcept
    {
        return mImpl->addFill(dimensions, op);
    }

    //!
    //! \brief Add a fill layer to the network.
    //!
    //! \param dimensions The output tensor dimensions if input 0 is missing.
    //! \param op The fill operation that the layer applies.
    //! \param outputType Optional output tensor data type, must be DataType::kFLOAT, DataType::kHALF, DataType::kINT32,
    //! or DataType::kINT64. This parameter is only used for static alpha/beta. Future calls to set output type using
    //! setToType or setOutputType must be consistent.
    //!
    //! \warning For FillOperation::kLINSPACE, dimensions.nbDims must be 1 for static start/delta. If delta is provided
    //! as a 1D tensor, the length of delta must match dimensions.nbDims.
    //!
    //! This layer is non-deterministic across subsequent calls as the same inputs will produce different
    //! output tensors if \p op is either FillOperation::kRANDOM_UNIFORM or FillOperation::kRANDOM_NORMAL
    //! due to random state being shared across calls. The output tensors generated are deterministic when
    //! starting from the same initial state.
    //!
    //! \see IFillLayer
    //!
    //! \return The new fill layer, or nullptr if it could not be created.
    //!
    IFillLayer* addFill(Dims const& dimensions, FillOperation op, DataType outputType) noexcept
    {
        return mImpl->addFillV2(dimensions, op, outputType);
    }

    //!
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
    IPaddingLayer* addPaddingNd(ITensor& input, Dims const& prePadding, Dims const& postPadding) noexcept
    {
        return mImpl->addPaddingNd(input, prePadding, postPadding);
    }

    //!
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
    //! \p input tensor data type must be DataType::kINT8/DataType::kFP8.
    //! \p scale tensor data type must be DataType::kFLOAT. The subgraph which terminates with the \p scale tensor must
    //! be a build-time constant.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 9.0. Superseded by three-argument addDequantize.
    //!
    TRT_DEPRECATED IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale) noexcept
    {
        return mImpl->addDequantize(input, scale);
    }

    //!
    //! \brief Add a dequantization layer to the network.
    //!
    //! \param input The input tensor to be dequantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IDequantizeLayer
    //!
    //! \p input tensor data type must be DataType::kINT8/DataType::kFP8/DataType::kINT4.
    //! \p scale tensor data type defaults to DataType::kFLOAT. For strongly typed networks, it must be the same as the
    //! output data type. The subgraph which terminates with the \p scale tensor must be a build-time constant.
    //! \p outputType output tensor data type, default value is DataType::kFLOAT. Future calls to set output type using
    //! setToType or setOutputType must be consistent. For strongly typed networks, it must be the same as the scale data type.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale, DataType outputType) noexcept
    {
        return mImpl->addDequantizeV2(input, scale, outputType);
    }

    //!
    //! \brief Add a Scatter layer to the network with specified mode and axis=0.
    //!
    //! \param data The input tensor to be updated with additional values.
    //! \param indices indices of the elements to be updated.
    //! \param updates values to be used for updates.
    //! \param mode scatter mode.
    //!
    //! \see IScatterLayer
    //!
    //! \p indices tensor data type must be DataType::kINT32.
    //! \p updates tensor data type must be the same as \p data
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
    //! \p input tensor data type must be DataType::kFLOAT/DataType::kHALF.
    //! \p scale tensor data type must be DataType::kFLOAT. The subgraph which terminates with the \p scale tensor must
    //! be a build-time constant.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 9.0. Superseded by three-argument addQuantize.
    //!
    TRT_DEPRECATED IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale) noexcept
    {
        return mImpl->addQuantize(input, scale);
    }

    //!
    //! \brief Add a quantization layer to the network.
    //!
    //! \param input The input tensor to be quantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IQuantizeLayer
    //!
    //! \p input tensor data type must be DataType::kFLOAT/DataType::kHALF/DataType::kBF16.
    //! \p scale tensor data type defaults to DataType::kFLOAT. For strongly typed networks, it must have the same data
    //! type as the input. The subgraph which terminates with the \p scale tensor must be a build-time constant.
    //! \p outputType output tensor data type, must be DataType::kINT8 (default), DataType::kFP8 or DataType::kINT4.
    //! Future calls to set output type using setToType or setOutputType must be consistent.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale, DataType outputType) noexcept
    {
        return mImpl->addQuantizeV2(input, scale, outputType);
    }


    //!
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

    //!
    //! \brief Add a GridSample layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param grid The grid tensor to the layer.
    //!
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

    //!
    //! \brief Add a ReverseSequence layer to the network.
    //!
    //! \param input The input tensor to the layer. Must have rank >= 2.
    //!
    //! \param sequenceLens 1D tensor specifying lengths of sequences to reverse in a batch. The length of the
    //!        sequenceLens tensor must be equal to the size of the dimension in input tensor specified by batchAxis.
    //!
    //! \see IReverseSequenceLayer
    //!
    //! \return The new ReverseSequence layer, or nullptr if it could not be created.
    //!
    IReverseSequenceLayer* addReverseSequence(ITensor& input, ITensor& sequenceLens) noexcept
    {
        return mImpl->addReverseSequence(input, sequenceLens);
    }

    //!
    //! \brief Add a normalization layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param scale The scale tensor used to scale the normalized output.
    //! \param bias The bias tensor used to scale the normalized output.
    //! \param axesMask The axes on which to perform mean calculations.
    //!        The bit in position i of bitmask axesMask corresponds to explicit dimension i of the result.
    //!        E.g., the least significant bit corresponds to the first explicit dimension and the next to least
    //!        significant bit corresponds to the second explicit dimension.
    //!
    //! The normalization layer works by performing normalization of the tensor \p input on the specified \p axesMask.
    //! The result is then scaled by multiplying with \p scale and adding \p bias.
    //!
    //! The shape of \p scale and \p bias are expected the be the same, and must have the same rank and be
    //! unidirectionally broadcastable to the shape of \p input.
    //!
    //! \see INormalizationLayer
    //!
    //! \return The new normalization layer, or nullptr if it could not be created.
    //!
    INormalizationLayer* addNormalization(ITensor& input, ITensor& scale, ITensor& bias, uint32_t axesMask) noexcept
    {
        return mImpl->addNormalization(input, scale, bias, axesMask);
    }

    //!
    //! \brief Return the builder from which this INetworkDefinition was created.
    //!
    //! \see IBuilder::createNetworkV2
    //!
    //! \return the builder
    virtual IBuilder& getBuilder() const noexcept
    {
        return mImpl->getBuilder();
    }

    //!
    //! \brief Mark weights as refittable when the builder flag kREFIT_INDIVIDUAL is set.
    //!
    //! \param name The name of the weights.
    //!
    //! \return True if the weights were successfully marked as refittable, false if the weights do not exist or cannot
    //! be refitted.
    //!
    bool markWeightsRefittable(char const* name) noexcept
    {
        return mImpl->markWeightsRefittable(name);
    }

    //!
    //! \brief Unmark weights as refittable when the builder flag kREFIT_INDIVIDUAL is set.
    //!
    //! \param name The name of the weights.
    //!
    //! \return True if the weights were successfully marked as unrefittable, false if the weights do not exist.
    //!
    bool unmarkWeightsRefittable(char const* name) noexcept
    {
        return mImpl->unmarkWeightsRefittable(name);
    }

    //!
    //! \brief Whether the weight has been marked as refittable.
    //!
    //! \param name The name of the weights to check.
    //!
    //! \return True if the weights are marked as refittable, false if the weights do not exist or are marked as
    //! non-refittable.
    //!
    bool areWeightsMarkedRefittable(char const* name) const noexcept
    {
        return mImpl->areWeightsMarkedRefittable(name);
    }

    //!
    //! \brief Add a squeeze layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param axes The axes to remove unit dimensions on.
    //!
    //! \see ISqueezeLayer
    //!
    //! Axes must be resolvable to a constant Int32 or Int64 1D shape tensor.
    //! Values in axes must be unique and in the range of [-r, r-1], where r is the rank of the input tensor.
    //! For each axis value, the corresponding dimension in the input tensor must be one.
    //!
    //! \return The new Squeeze layer, or nullptr if it could not be created.
    //!
    ISqueezeLayer* addSqueeze(ITensor& input, ITensor& axes) noexcept
    {
        return mImpl->addSqueeze(input, axes);
    }

    //!
    //! \brief Add an unsqueeze layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param axes The axes to add unit dimensions.
    //!
    //! \see IUnsqueezeLauyer
    //!
    //! Axes must be resolvable to a constant Int32 or Int64 shape tensor.
    //! Values in axes must be unique and in the range of [-r_final, r_final-1], where r_final
    //! is the sum of rank(input) and len(axes).
    //!
    //! r_final must be less than Dims::MAX_DIMS.
    //!
    //! \return The new Unsqueeze layer, or nullptr if it could not be created
    //!
    IUnsqueezeLayer* addUnsqueeze(ITensor& input, ITensor& axes) noexcept
    {
        return mImpl->addUnsqueeze(input, axes);
    }

protected:
    apiv::VNetworkDefinition* mImpl;
};

//!
//! \enum CalibrationAlgoType
//!
//! \brief Version of calibration algorithm to use.
//!
//! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
//!
enum class CalibrationAlgoType : int32_t
{
    kLEGACY_CALIBRATION = 0,    //!< Legacy calibration
    kENTROPY_CALIBRATION = 1,   //!< Legacy entropy calibration
    kENTROPY_CALIBRATION_2 = 2, //!< Entropy calibration
    kMINMAX_CALIBRATION = 3,    //!< Minmax calibration
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
//! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
//!
class TRT_DEPRECATED IInt8Calibrator : public IVersionedInterface
{
public:
    //!
    //! \brief Get the batch size used for calibration batches.
    //!
    //! \return The batch size.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch support is removed in TensorRT 10.0.
    //!
    TRT_DEPRECATED virtual int32_t getBatchSize() const noexcept = 0;

    //!
    //! \brief Get a batch of input for calibration.
    //!
    //! The batch size of the input must match the batch size returned by getBatchSize().
    //!
    //! \param bindings An array of pointers to device memory that must be updated to point to device memory
    //! containing each network input data.
    //! \param names The names of the network input for each pointer in the binding array.
    //! \param nbBindings The number of pointers in the bindings array.
    //!
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

    ~IInt8Calibrator() noexcept override = default;
};

namespace v_1_0
{
class TRT_DEPRECATED IInt8EntropyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IInt8EntropyCalibrator", 1, 0};
    }

    //!
    //! Signal that this is the entropy calibrator.
    //!
    CalibrationAlgoType getAlgorithm() noexcept override
    {
        return CalibrationAlgoType::kENTROPY_CALIBRATION;
    }

    ~IInt8EntropyCalibrator() noexcept override = default;
};
} // namespace v_1_0

//!
//! \class IInt8EntropyCalibrator
//!
//! \brief Entropy calibrator.
//!
//! This is the Legacy Entropy calibrator. It is less complicated than the legacy calibrator and
//! produces better results.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use IEntropyCalibrator, not
//!       v_1_0::IEntropyCalibrator
//!
//! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
//!
using IInt8EntropyCalibrator = v_1_0::IInt8EntropyCalibrator;

namespace v_1_0
{
class TRT_DEPRECATED IInt8EntropyCalibrator2 : public IInt8Calibrator
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IInt8EntropyCalibrator2", 1, 0};
    }

    //!
    //! Signal that this is the entropy calibrator 2.
    //!
    CalibrationAlgoType getAlgorithm() noexcept override
    {
        return CalibrationAlgoType::kENTROPY_CALIBRATION_2;
    }

    ~IInt8EntropyCalibrator2() noexcept override = default;
};
} // namespace v_1_0

//!
//! \class IInt8EntropyCalibrator2
//!
//! \brief Entropy calibrator 2.
//!
//! This is the preferred calibrator. This is the required calibrator for DLA, as it supports per
//! activation tensor scaling.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use IEntropyCalibrator2, not
//!        v_1_0::IEntropyCalibrator2
//!
//! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
//!
using IInt8EntropyCalibrator2 = v_1_0::IInt8EntropyCalibrator2;

namespace v_1_0
{
class TRT_DEPRECATED IInt8MinMaxCalibrator : public IInt8Calibrator
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IInt8MinMaxCalibrator", 1, 0};
    }

    //!
    //! Signal that this is the MinMax Calibrator.
    //!
    CalibrationAlgoType getAlgorithm() noexcept override
    {
        return CalibrationAlgoType::kMINMAX_CALIBRATION;
    }

    ~IInt8MinMaxCalibrator() noexcept override = default;
};
} // namespace v_1_0

//!
//! \class IInt8MinMaxCalibrator
//!
//! \brief MinMax Calibrator.
//!
//! It supports per activation tensor scaling.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use IMinMaxCalibrator>, not
//!       v_1_0::IMinMaxCalibrator
//!
//! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
//!
using IInt8MinMaxCalibrator = v_1_0::IInt8MinMaxCalibrator;

namespace v_1_0
{
class TRT_DEPRECATED IInt8LegacyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IInt8Calibrator", 1, 0};
    }

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

    ~IInt8LegacyCalibrator() noexcept override = default;
};
} // namespace v_1_0

//!
//! \class IInt8LegacyCalibrator
//!
//! \brief Legacy calibrator.
//!
//! This calibrator requires user parameterization,
//! and is provided as a fallback option if the other calibrators yield poor results.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use ILegacyCalibrator, not
//!       v_1_0::ILegacyCalibrator
//!
//! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
//!
using IInt8LegacyCalibrator = v_1_0::IInt8LegacyCalibrator;

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
    //! \brief Return DataType of the input/output of algorithm.
    //!
    //! \return the data type.
    //!
    DataType getDataType() const noexcept
    {
        return mImpl->getDataType();
    }

    //!
    //! \brief Return strides of the input/output tensor of algorithm.
    //! For vectorized formats, strides are given in units of vectors.
    //!
    //! \return the strides of the tensor.
    //!
    Dims getStrides() const noexcept
    {
        return mImpl->getStrides();
    }

    //!
    //! \brief Return the index of the vectorized dimension or -1 for non-vectorized formats.
    //!
    //! \return the index of the vectorized dimension.
    //!
    int64_t getVectorizedDim() const noexcept
    {
        return mImpl->getVectorizedDim();
    }

    //!
    //! \brief Return the number of components per element.
    //! This is always 1 for non-vectorized formats.
    //!
    //! \return the number of components per element.
    //!
    int64_t getComponentsPerElement() const noexcept
    {
        return mImpl->getComponentsPerElement();
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
    //!
    //! This is a unique identifier for the IAlgorithmContext.
    //!
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for input or output tensor.
    //!
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
//!
//! \brief Describes a variation of execution of a layer.
//!        An algorithm is represented by IAlgorithmVariant and the IAlgorithmIOInfo for each of its inputs and outputs.
//!        An algorithm can be selected or reproduced using AlgorithmSelector::selectAlgorithms().
//!
//! \see IAlgorithmIOInfo, IAlgorithmVariant, IAlgorithmSelector::selectAlgorithms()
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IAlgorithm : public INoCopy
{
public:
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
    //!
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

namespace v_1_0
{
class IAlgorithmSelector : public IVersionedInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IAlgorithmSelector", 1, 0};
    }
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
    //!       BuilderFlag::kREJECT_EMPTY_ALGORITHMS is set.
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
} // namespace v_1_0

//!
//! \class IAlgorithmSelector
//!
//! \brief Interface implemented by application for selecting and reporting algorithms of a layer provided by the
//!        builder.
//! \note A layer in context of algorithm selection may be different from ILayer in INetworkDefiniton.
//!       For example, an algorithm might be implementing a conglomeration of multiple ILayers in INetworkDefinition.
//! \note To ensure compatibility of source code with future versions of TensorRT, use IAlgorithmSelector, not
//!       v_1_0::IAlgorithmSelector
//!
using IAlgorithmSelector = v_1_0::IAlgorithmSelector;

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
//! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
//!
enum class QuantizationFlag : int32_t
{
    //! Run int8 calibration pass before layer fusion. Only valid for IInt8LegacyCalibrator and
    //! IInt8EntropyCalibrator. The builder always runs the int8 calibration pass before layer fusion for
    //! IInt8MinMaxCalibrator and IInt8EntropyCalibrator2. Disabled by default.
    kCALIBRATE_BEFORE_FUSION TRT_DEPRECATED_ENUM = 0
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
//! \enum RuntimePlatform
//!
//! \brief Describes the intended runtime platform (operating system and CPU architecture) for the execution of the
//!        TensorRT engine. TensorRT provides support for cross-platform engine compatibility when the target runtime
//!        platform is different from the build platform.
//!
//! \note The cross-platform engine will not be able to run on the host platform it was built on.
//!
//! \note When building a cross-platform engine that also requires version forward compatibility,
//!       kEXCLUDE_LEAN_RUNTIME must be set to exclude the target platform lean runtime.
//!
//! \note The cross-platform engine might have performance differences compared to the natively built engine on the
//!       target platform.
//!
//! \see IBuilderConfig::setRuntimePlatform(), IBuilderConfig::getRuntimePlatform()
//!
enum class RuntimePlatform : int32_t
{
    //! No requirement for cross-platform compatibility. The engine constructed by TensorRT can only run on the
    //! identical platform it was built on.
    kSAME_AS_BUILD = 0,

    //! Designates the target platform for engine execution as Windows AMD64 system. Currently this flag can only be
    //! enabled when building engines on Linux AMD64 platforms.
    kWINDOWS_AMD64 = 1,
};

namespace impl
{
//!
//! Maximum number of elements in RuntimePlatform enum.
//!
//! \see RuntimePlatform
//!
template <>
struct EnumMaxImpl<RuntimePlatform>
{
    static constexpr int32_t kVALUE = 2;
};
} // namespace impl

//!
//! \brief Represents one or more BuilderFlag values using binary OR
//! operations, e.g., 1U << BuilderFlag::kFP16 | 1U << BuilderFlag::kDEBUG.
//!
//! \see IBuilderConfig::setFlags(), IBuilderConfig::getFlags()
//!
using BuilderFlags = uint32_t;

//!
//! \enum BuilderFlag
//!
//! \brief List of valid modes that the builder can enable when creating an engine from a network definition.
//!
//! \see IBuilderConfig::setFlags(), IBuilderConfig::getFlags()
//!
enum class BuilderFlag : int32_t
{
    //! Enable FP16 layer selection, with FP32 fallback.
    kFP16 = 0,

    //! Enable Int8 layer selection, with FP32 fallback with FP16 fallback if kFP16 also specified.
    kINT8 = 1,

    //! Enable debugging of layers via synchronizing after every layer.
    kDEBUG = 2,

    //! Enable layers marked to execute on GPU if layer cannot execute on DLA.
    kGPU_FALLBACK = 3,

    //! Enable building a refittable engine.
    kREFIT = 4,

    //! Disable reuse of timing information across identical layers.
    kDISABLE_TIMING_CACHE = 5,

    //! Allow (but not require) computations on tensors of type DataType::kFLOAT to use TF32.
    //! TF32 computes inner products by rounding the inputs to 10-bit mantissas before
    //! multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.
    kTF32 = 6,

    //! Allow the builder to examine weights and use optimized functions when weights have suitable sparsity.
    kSPARSE_WEIGHTS = 7,

    //! Change the allowed parameters in the EngineCapability::kSTANDARD flow to
    //! match the restrictions that EngineCapability::kSAFETY check against for DeviceType::kGPU
    //! and EngineCapability::kDLA_STANDALONE check against the DeviceType::kDLA case. This flag
    //! is forced to true if EngineCapability::kSAFETY at build time if it is unset.
    //!
    //! This flag is only supported in NVIDIA Drive(R) products.
    kSAFETY_SCOPE = 8,

    //! Require that layers execute in specified precisions. Build fails otherwise.
    kOBEY_PRECISION_CONSTRAINTS = 9,

    //! Prefer that layers execute in specified precisions.
    //! Fall back (with warning) to another precision if build would otherwise fail.
    kPREFER_PRECISION_CONSTRAINTS = 10,

    //! Require that no reformats be inserted between a layer and a network I/O tensor
    //! for which ITensor::setAllowedFormats was called.
    //! Build fails if a reformat is required for functional correctness.
    //! \deprecated Deprecated in TensorRT 10.7. Unneeded API.
    kDIRECT_IO TRT_DEPRECATED_ENUM = 11,

    //! Fail if IAlgorithmSelector::selectAlgorithms returns an empty set of algorithms.
    kREJECT_EMPTY_ALGORITHMS = 12,

    //! Restrict to lean runtime operators to provide version forward compatibility
    //! for the plan.
    //!
    //! This flag is only supported by NVIDIA Volta and later GPUs.
    //! This flag is not supported in NVIDIA Drive(R) products.
    kVERSION_COMPATIBLE = 13,

    //! Exclude lean runtime from the plan when version forward compatability is enabled.
    //! By default, this flag is unset, so the lean runtime will be included in the plan.
    //!
    //! If BuilderFlag::kVERSION_COMPATIBLE is not set then the value of this flag will be ignored.
    kEXCLUDE_LEAN_RUNTIME = 14,

    //! Enable plugins with FP8 input/output.
    //!
    //! This flag is not supported with hardware-compatibility mode.
    //!
    //! \see HardwareCompatibilityLevel
    kFP8 = 15,

    //! Emit error when a tactic being timed is not present in the timing cache.
    //! This flag has an effect only when IBuilderConfig has an associated ITimingCache.
    kERROR_ON_TIMING_CACHE_MISS = 16,

    //! Enable DataType::kBF16 layer selection, with FP32 fallback.
    //! This flag is only supported by NVIDIA Ampere and later GPUs.
    kBF16 = 17,

    //! Disable caching of JIT-compilation results during engine build.
    //! By default, JIT-compiled code will be serialized as part of the timing cache, which may significantly increase
    //! the cache size. Setting this flag prevents the code from being serialized. This flag has an effect only when
    //! BuilderFlag::DISABLE_TIMING_CACHE is not set.
    kDISABLE_COMPILATION_CACHE = 18,

    //! Strip the refittable weights from the engine plan file.
    kSTRIP_PLAN = 19,

    //! \deprecated Deprecated in TensorRT 10.0. Superseded by kSTRIP_PLAN.
    kWEIGHTLESS TRT_DEPRECATED_ENUM = kSTRIP_PLAN,

    //! Create a refittable engine under the assumption that the refit weights will be identical to those provided at
    //! build time. The resulting engine will have the same performance as a non-refittable one. All refittable weights
    //! can be refitted through the refit API, but if the refit weights are not identical to the build-time weights,
    //! behavior is undefined. When used alongside 'kSTRIP_PLAN', this flag will result in a small plan file for which
    //! weights are later supplied via refitting. This enables use of a single set of weights with different inference
    //! backends, or with TensorRT plans for multiple GPU architectures.
    kREFIT_IDENTICAL = 20,

    //!
    //! \brief Enable weight streaming for the current engine.
    //!
    //! Weight streaming from the host enables execution of models that do not fit
    //! in GPU memory by allowing TensorRT to intelligently stream network weights
    //! from the CPU DRAM. Please see ICudaEngine::getMinimumWeightStreamingBudget
    //! for the default memory budget when this flag is enabled.
    //!
    //! Enabling this feature changes the behavior of
    //! IRuntime::deserializeCudaEngine to allocate the entire network’s weights
    //! on the CPU DRAM instead of GPU memory. Then,
    //! ICudaEngine::createExecutionContext will determine the optimal split of
    //! weights between the CPU and GPU and place weights accordingly.
    //!
    //! Future TensorRT versions may enable this flag by default.
    //!
    //! \warning Enabling this flag may marginally increase build time.
    //!
    //! \warning Enabling this feature will significantly increase the latency of
    //!          ICudaEngine::createExecutionContext.
    //!
    //! \see IRuntime::deserializeCudaEngine,
    //!      ICudaEngine::getMinimumWeightStreamingBudget,
    //!      ICudaEngine::setWeightStreamingBudget
    //!
    kWEIGHT_STREAMING = 21,

    //! Enable plugins with INT4 input/output.
    kINT4 = 22,

    //! Enable building a refittable engine and provide fine-grained control. This allows
    //! control over which weights are refittable or not using INetworkDefinition::markWeightsRefittable and
    //! INetworkDefinition::unmarkWeightsRefittable. By default, all weights are non-refittable when this flag is
    //! enabled. This flag cannot be used together with kREFIT or kREFIT_IDENTICAL.
    kREFIT_INDIVIDUAL = 23,

    //!  Disable floating-point optimizations: 0*x => 0, x-x => 0, or x/x => 1. These identities are
    //!  not true when x is a NaN or Inf, and thus might hide propagation or generation of NaNs. This flag is typically
    //!  used in combination with kSPARSE_WEIGHTS.
    //!  There are three valid sparsity configurations.
    //!  1. Disable all sparsity. Both kSPARSE_WEIGHTS and kSTRICT_NANS are unset
    //!  2. Enable sparsity only where it does not affect propagation/generation of NaNs. Both kSPARSE_WEIGHTS and
    //!  kSTRICT_NANS are set
    //!  3. Enable all sparsity. kSPARSE_WEIGHTS is set and kSTRICT_NANS is unset
    kSTRICT_NANS = 24,

    //! Enable memory monitor during build time.
    kMONITOR_MEMORY = 25,

};

//!
//! Maximum number of builder flags in BuilderFlag enum.
//!
//! \see BuilderFlag
//!
template <>
constexpr inline int32_t EnumMax<BuilderFlag>() noexcept
{
    return 26;
}

//!
//! \class ITimingCache
//!
//! \brief Class to handle tactic timing info collected from builder.
//!
//! The timing cache is created or initialized by IBuilderConfig. It can be shared across builder instances
//! to reduce the builder wallclock time.
//!
//! \warning It is a known issue that the same timing cache doesn't guarantee stable engine build reproducibility
//!          at optimization level 4 and higher. This issue will be fixed by 2024.
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
    //! This defaults to max device memory. Set to a smaller value to restrict tactics that use over the
    //! threshold en masse. For more targeted removal of tactics use the IAlgorithmSelector
    //! interface.
    //!
    kWORKSPACE = 0,

    //!
    //! kDLA_MANAGED_SRAM is a fast software managed RAM used by DLA to communicate within a layer.
    //! The size of this pool must be at least 4 KiB and must be a power of 2.
    //! This defaults to 1 MiB.
    //! Orin has capacity of 1 MiB per core.
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

    //!
    //! kTACTIC_DRAM is the device DRAM used by the optimizer to
    //! run tactics. On embedded devices, where host and device memory are unified, this includes all host
    //! memory required by TensorRT to build the network up to the point of each memory allocation.
    //! This defaults to 75% of totalGlobalMem as reported by cudaGetDeviceProperties when
    //! cudaGetDeviceProperties.embedded is true, and 100% otherwise.
    //!
    kTACTIC_DRAM = 4,

    //!
    //! kTACTIC_SHARED_MEMORY defines the maximum sum of shared memory reserved by the driver and
    //! used for executing CUDA kernels. Adjust this value to restrict tactics that exceed the
    //! specified threshold en masse. The default value is device max capability. This value must
    //! be less than 1GiB.
    //!
    //! The driver reserved shared memory can be queried from cuDeviceGetAttribute(&reservedShmem,
    //! CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK).
    //!
    //! Updating this flag will override the shared memory limit set by \ref HardwareCompatibilityLevel,
    //! which defaults to 48KiB - reservedShmem.
    //!
    kTACTIC_SHARED_MEMORY = 5,
};

//!
//! Maximum number of memory pool types in the MemoryPoolType enum.
//!
//! \see MemoryPoolType
//!
template <>
constexpr inline int32_t EnumMax<MemoryPoolType>() noexcept
{
    return 6;
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
    //! Allows optimization profiles to be shared across execution contexts.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. The default value for this flag is on and can not be changed.
    //!
    kPROFILE_SHARING_0806 TRT_DEPRECATED_ENUM = 0,

    //!
    //! Allows plugin I/O to be aliased when using IPluginV3OneBuildV2
    //!
    kALIASED_PLUGIN_IO_10_03 = 1
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
//! \enum HardwareCompatibilityLevel
//!
//! \brief Describes requirements of compatibility with GPU architectures other than that of the GPU on which the engine was
//! built.
//!
//! Levels except kNONE are only supported for engines built on NVIDIA Ampere and later GPUs.
//!
//! \warning Note that compatibility with future hardware depends on CUDA forward compatibility support.
//!
enum class HardwareCompatibilityLevel : int32_t
{
    //! Do not require hardware compatibility with GPU architectures other than that of the GPU on which the engine was
    //! built.
    kNONE = 0,

    //! Require that the engine is compatible with Ampere and newer GPUs. This will limit the combined usage of driver
    //! reserved and backend kernel max shared memory to 48KiB, may reduce the number of available tactics for each
    //! layer, and may prevent some fusions from occurring. Thus this can decrease the performance, especially for tf32
    //! models.
    //! This option will disable cuDNN, cuBLAS, and cuBLAS LT as tactic sources.
    //!
    //! The driver reserved shared memory can be queried from cuDeviceGetAttribute(&reservedShmem,
    //! CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK).
    //!
    kAMPERE_PLUS = 1,
};

namespace impl
{
//!
//! Maximum number of elements in HardwareCompatibilityLevel enum.
//!
//! \see HardwareCompatibilityLevel
//!
template <>
struct EnumMaxImpl<HardwareCompatibilityLevel>
{
    static constexpr int32_t kVALUE = 2;
};
} // namespace impl

namespace v_1_0
{
class IProgressMonitor : public IVersionedInterface
{
public:
    IProgressMonitor() = default;
    virtual ~IProgressMonitor() noexcept = default;

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IProgressMonitor", 1, 0};
    }

    //!
    //! \brief Signal that a phase of the optimizer has started.
    //!
    //! \param phaseName The name of this phase for tracking purposes.
    //! \param parentPhase The parent phase that this phase belongs to, or nullptr if there is no parent.
    //! \param nbSteps The number of steps that are involved in this phase.
    //!
    //! The phaseStart function signals to the application that the current phase is beginning, and that it has a
    //! certain number of steps to perform. If \p phaseParent is nullptr, then the phaseStart is beginning an
    //! independent phase, and if \p phaseParent is specified, then the current phase, specified by \p phaseName, is
    //! within the scope of the parent phase. \p nbSteps will always be a positive number. The phaseStart function
    //! implies that the first step is being executed. TensorRT will signal when each step is complete.
    //!
    //! Phase names are human readable English strings which are unique within a single phase hierarchy but which can be
    //! reused once the previous instance has completed. Phase names and their hierarchies may change between versions
    //! of TensorRT.
    //!
    //! \see phaseFinish
    //!
    virtual void phaseStart(char const* phaseName, char const* parentPhase, int32_t nbSteps) noexcept = 0;

    //!
    //! \brief Signal that a step of an optimizer phase has finished.
    //!
    //! \param phaseName The name of the innermost phase being executed.
    //! \param step The step number that was completed.
    //!
    //! The stepComplete function signals to the application that TensorRT has finished the current \p step for the
    //! phase \p phaseName, and will move onto the next step if there is one. The application can return false for
    //! TensorRT to exit the build early. The step value will increase on subsequent calls in the range [0, nbSteps).
    //!
    //! \return true to continue to the next step or false to stop the build.
    //!
    virtual bool stepComplete(char const* phaseName, int32_t step) noexcept = 0;

    //!
    //! \brief Signal that a phase of the optimizer has finished.
    //!
    //! \param phaseName The name of the phase that has finished.
    //!
    //! The phaseFinish function signals to the application that the phase is complete. This function may be called
    //! before all steps in the range [0, nbSteps) have been reported to stepComplete. This scenario can be triggered by
    //! error handling, internal optimizations, or when stepComplete returns false to request cancellation of the build.
    //!
    //! \see phaseStart
    //!
    virtual void phaseFinish(char const* phaseName) noexcept = 0;

}; // class IProgressMonitor
} // namespace v_1_0

//!
//! \class IProgressMonitor
//!
//! \brief Application-implemented progress reporting interface for TensorRT.
//!
//! The IProgressMonitor is a user-defined object that TensorRT uses to report back when an internal algorithm has
//! started or finished a phase to help provide feedback on the progress of the optimizer.
//!
//! The IProgressMonitor will trigger its start function when a phase is entered and will trigger its finish function
//! when that phase is exited. Each phase consists of one or more steps. When each step is completed, the stepComplete
//! function is triggered. This will allow an application using the builder to communicate progress relative to when the
//! optimization step is expected to complete.
//!
//! The implementation of IProgressMonitor must be thread-safe so that it can be called from multiple internal threads.
//! The lifetime of the IProgressMonitor must exceed the lifetime of all TensorRT objects that use it.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use IProgressMonitor, not
//!       v_1_0::IProgressMonitor
//!
using IProgressMonitor = v_1_0::IProgressMonitor;

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
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept
    {
        mImpl->setInt8Calibrator(calibrator);
    }

    //!
    //! \brief Get Int8 Calibration interface.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED IInt8Calibrator* getInt8Calibrator() const noexcept
    {
        return mImpl->getInt8Calibrator();
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
    //!
    //! \param layer which layer to execute.
    //! \param deviceType that this layer must execute on.
    //! If DeviceType is not set or is reset, TensorRT will use the default DeviceType set in the builder.
    //!
    //! \note The device type for a layer must be compatible with the safety flow (if specified).
    //! For example a layer cannot be marked for DLA execution while the builder is configured for kSAFETY.
    //!
    //! \see getDeviceType()
    //!
    void setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept
    {
        mImpl->setDeviceType(layer, deviceType);
    }

    //!
    //! \brief Get the device that this layer executes on.
    //!
    //! \return Returns DeviceType of the layer.
    //!
    DeviceType getDeviceType(ILayer const* layer) const noexcept
    {
        return mImpl->getDeviceType(layer);
    }

    //!
    //! \brief whether the DeviceType has been explicitly set for this layer
    //!
    //! \return true if device type is not default
    //!
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
    //!
    //! \return status true if the layer can on DLA else returns false.
    //!
    bool canRunOnDLA(ILayer const* layer) const noexcept
    {
        return mImpl->canRunOnDLA(layer);
    }

    //!
    //! \brief Sets the DLA core used by the network. Defaults to -1.
    //!
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
    //!
    //! \return assigned DLA core or -1 for DLA not present or unset.
    //!
    int32_t getDLACore() const noexcept
    {
        return mImpl->getDLACore();
    }

    //!
    //! \brief Sets the default DeviceType to be used by the builder. It ensures that all the layers that can run on
    //! this device will run on it, unless setDeviceType is used to override the default DeviceType for a layer.
    //!
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
    //! \brief Set the CUDA stream that is used to profile this network.
    //!
    //! \param stream The CUDA stream used for profiling by the builder.
    //!
    //! \see getProfileStream()
    //!
    void setProfileStream(const cudaStream_t stream) noexcept
    {
        return mImpl->setProfileStream(stream);
    }

    //!
    //! \brief Get the CUDA stream that is used to profile this network.
    //!
    //! \return The CUDA stream set by setProfileStream, nullptr if setProfileStream has not been called.
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
    //!
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
    //!
    //! \return True if the calibration profile was set correctly.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED bool setCalibrationProfile(IOptimizationProfile const* profile) noexcept
    {
        return mImpl->setCalibrationProfile(profile);
    }

    //!
    //! \brief Get the current calibration profile.
    //!
    //! \return A pointer to the current calibration profile or nullptr if calibration profile is unset.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED IOptimizationProfile const* getCalibrationProfile() noexcept
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
    //! \see setTacticSources()
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

    //!
    //! \brief Set builder optimization level
    //!
    //! Set the builder optimization level. Setting a higher optimization
    //! level allows the optimizer to spend more time searching for optimization opportunities. The
    //! resulting engine may have better performance compared to an engine built with a lower optimization level.
    //!
    //! The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level,
    //! which is currently 5. Setting it to greater than the maximum level results in behavior identical to the
    //! maximum level.
    //!
    //! Below are the descriptions about each builder optimization level:
    //!
    //! - Level 0: This enables the fastest compilation by disabling dynamic kernel generation and selecting the first
    //!   tactic that succeeds in execution. This will also not respect a timing cache.
    //! - Level 1: Available tactics are sorted by heuristics, but only the top are tested to select the best. If a
    //!   dynamic kernel is generated its compile optimization is low.
    //! - Level 2: Available tactics are sorted by heuristics, but only the fastest tactics are tested to select the
    //!   best.
    //! - Level 3: Apply heuristics to see if a static precompiled kernel is applicable or if a new one has to be
    //!   compiled dynamically.
    //! - Level 4: Always compiles a dynamic kernel.
    //! - Level 5: Always compiles a dynamic kernel and compares it to static kernels.
    //!
    //! \param level The optimization level to set to. Must be non-negative.
    //!
    //! \see getBuilderOptimizationLevel
    //!
    void setBuilderOptimizationLevel(int32_t level) noexcept
    {
        mImpl->setBuilderOptimizationLevel(level);
    }

    //!
    //! \brief Get builder optimization level
    //!
    //! \returns the current builder optimization level
    //!
    //! \see setBuilderOptimizationLevel
    //!
    int32_t getBuilderOptimizationLevel() noexcept
    {
        return mImpl->getBuilderOptimizationLevel();
    }

    //!
    //! \brief Set the hardware compatibility level.
    //!
    //! Hardware compatibility allows an engine to run on GPU
    //! architectures other than that of the GPU where the engine was
    //! built.
    //!
    //! The default hardware compatibility level is HardwareCompatibilityLevel::kNONE.
    //!
    //! \param hardwareCompatibilityLevel The level of hardware
    //!        compatibility.
    //!
    void setHardwareCompatibilityLevel(HardwareCompatibilityLevel hardwareCompatibilityLevel) noexcept
    {
        mImpl->setHardwareCompatibilityLevel(hardwareCompatibilityLevel);
    }

    //!
    //! \brief Get the hardware compatibility level.
    //!
    //! \return hardwareCompatibilityLevel The level of hardware
    //!        compatibility.
    //!
    //! \see setHardwareCompatiblityLevel()
    //!
    HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept
    {
        return mImpl->getHardwareCompatibilityLevel();
    }

    //!
    //! \brief Set the plugin libraries to be serialized with version-compatible engines.
    //!
    //! Each entry in the list of libraries must be unique.
    //!
    //! \param paths The paths of plugin libraries.
    //! \param nbPaths The number of paths.
    //!
    void setPluginsToSerialize(char const* const* paths, int32_t nbPaths) noexcept
    {
        mImpl->setPluginsToSerialize(paths, nbPaths);
    }

    //!
    //! \brief Get the plugin library path to be serialized with version-compatible engines.
    //!
    //! \param index Index of the plugin library path in the list.  Should be in the range `[0,
    //! getNbPluginsToSerialize())`.
    //!
    //! \return The path to the plugin library.
    //!
    char const* getPluginToSerialize(int32_t index) const noexcept
    {
        return mImpl->getPluginToSerialize(index);
    }

    //!
    //! \brief Get the number of plugin library paths to be serialized with version-compatible engines.
    //!
    //! \return The number of paths.
    //!
    int32_t getNbPluginsToSerialize() const noexcept
    {
        return mImpl->getNbPluginsToSerialize();
    }

    //!
    //! \brief Set the maximum number of auxiliary streams that TRT is allowed to use.
    //!
    //! If the network contains operators that can run in parallel, TRT can execute them using auxiliary streams
    //! in addition to the one provided to the IExecutionContext::enqueueV3() call.
    //!
    //! The default maximum number of auxiliary streams is determined by the heuristics in TensorRT on whether enabling
    //! multi-stream would improve the performance. This behavior can be overridden by calling this API to set the
    //! maximum number of auxiliary streams explicitly. Set this to 0 to enforce single-stream inference.
    //!
    //! The resulting engine may use fewer auxiliary streams than the maximum if the network does not contain enough
    //! parallelism or if TensorRT determines that using more auxiliary streams does not help improve the performance.
    //!
    //! \note Allowing more auxiliary streams does not always give better performance since there will be
    //! synchronizations overhead between streams. Using CUDA graphs at runtime can help reduce the overhead caused by
    //! cross-stream synchronizations.
    //!
    //! \note Using more auxiliary leads to more memory usage at runtime since some activation memory blocks will not
    //! be able to be reused.
    //!
    //! \param nbStreams The maximum number of auxiliary streams that TRT is allowed to use.
    //!
    //! \see getMaxAuxStreams(), ICudaEngine::getNbAuxStreams(), IExecutionContext::setAuxStreams()
    //!
    void setMaxAuxStreams(int32_t nbStreams) noexcept
    {
        mImpl->setMaxAuxStreams(nbStreams);
    }

    //!
    //! \brief Get the maximum number of auxiliary streams that TRT is allowed to use.
    //!
    //! \see setMaxAuxStreams()
    //!
    int32_t getMaxAuxStreams() const noexcept
    {
        return mImpl->getMaxAuxStreams();
    }

    //!
    //! \brief Sets the progress monitor for building a network.
    //!
    //! \param monitor The progress monitor to assign to the IBuilderConfig.
    //!
    //! The progress monitor signals to the application when different phases of
    //! the compiler are being executed. Setting to nullptr unsets the monitor so
    //! that the application is not signaled.
    //!
    //! \see IBuilderConfig::getProgressMonitor
    //!
    void setProgressMonitor(IProgressMonitor* monitor) noexcept
    {
        return mImpl->setProgressMonitor(monitor);
    }

    //!
    //! \return The progress monitor set by the application or nullptr.
    //!
    //! \see IBuilderConfig::setProgressMonitor
    //!
    IProgressMonitor* getProgressMonitor() const noexcept
    {
        return mImpl->getProgressMonitor();
    }

    //!
    //! \brief Set the target platform for runtime execution.
    //!
    //! Cross-platform compatibility allows an engine to be built and executed on different platforms.
    //!
    //! The default cross-platform target is RuntimePlatform::kSAME_AS_BUILD.
    //!
    //! \param runtimePlatform The target platform for runtime execution.
    //!
    //! \see IBuilderConfig::getRuntimePlatform()
    //!
    void setRuntimePlatform(RuntimePlatform runtimePlatform) noexcept
    {
        mImpl->setRuntimePlatform(runtimePlatform);
    }

    //!
    //! \brief Get the target platform for runtime execution.
    //!
    //! \return The target platform for runtime execution.
    //!
    //! \see IBuilderConfig::setRuntimePlatform()
    //!
    RuntimePlatform getRuntimePlatform() const noexcept
    {
        return mImpl->getRuntimePlatform();
    }

    //!
    //! \brief Set the maximum number of tactics to time when there is a choice of tactics.
    //!
    //! This function controls the number of tactics timed when there are multiple tactics to choose from.
    //!
    //! \see getMaxNbTactics()
    //!
    void setMaxNbTactics(int32_t maxNbTactics) noexcept
    {
        mImpl->setMaxNbTactics(maxNbTactics);
    }

    //!
    //! \brief Query the maximum number of tactics timed when there is a choice.
    //!
    //! By default the value is -1, indicating TensorRT can determine the number of tactics based on its own heuristic.
    //!
    //! \see setMaxNbTactics()
    //!
    int32_t getMaxNbTactics() const noexcept
    {
        return mImpl->getMaxNbTactics();
    }

protected:
    apiv::VBuilderConfig* mImpl;
};

//!
//! \brief Represents one or more NetworkDefinitionCreationFlag flags
//! using binary OR operations.
//!  e.g., 1U << NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
//!
//! \see IBuilder::createNetworkV2
//!
using NetworkDefinitionCreationFlags = uint32_t;

//!
//! \enum NetworkDefinitionCreationFlag
//!
//! \brief List of immutable network properties expressed at network creation time.
//! NetworkDefinitionCreationFlag is used with createNetworkV2() to specify immutable properties of the network.
//!
//! \see IBuilder::createNetworkV2
//!
enum class NetworkDefinitionCreationFlag : int32_t
{
    //! Ignored because networks are always "explicit batch" in TensorRT 10.0.
    //!
    //! \deprecated Deprecated in TensorRT 10.0.
    kEXPLICIT_BATCH TRT_DEPRECATED_ENUM = 0,

    //! Mark the network to be strongly typed.
    //! Every tensor in the network has a data type defined in the network following only type inference rules and the
    //! inputs/operator annotations. Setting layer precision and layer output types is not allowed, and the network
    //! output types will be inferred based on the input types and the type inference rules.
    kSTRONGLY_TYPED = 1,
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
    //! \brief Determine whether the platform has fast native fp16.
    //!
    //! \deprecated Deprecated in TensorRT 10.5. Please query data type support from CUDA directly.
    //!
    TRT_DEPRECATED bool platformHasFastFp16() const noexcept
    {
        return mImpl->platformHasFastFp16();
    }

    //!
    //! \brief Determine whether the platform has fast native int8.
    //!
    //! \deprecated Deprecated in TensorRT 10.5. Please query data type support from CUDA directly.
    //!
    TRT_DEPRECATED bool platformHasFastInt8() const noexcept
    {
        return mImpl->platformHasFastInt8();
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
    //!
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
    //! \brief Create a network definition object
    //!
    //! Creates a network definition object with immutable properties specified using the flags parameter.
    //!
    //! createNetworkV2 supports creating network with properties from NetworkDefinitionCreationFlags.
    //!
    //! CreateNetworkV2 supports dynamic shapes and explicit batch dimensions by default.
    //!
    //! createNetworkV2 with NetworkDefinitionCreationFlag::kSTRONGLY_TYPED flag supports creating a strongly typed plan
    //! where tensor data types are inferred from network input types and operator type specification.
    //!
    //! \param flags Bitset of NetworkDefinitionCreationFlags specifying network properties combined with bitwise OR.
    //!             e.g., 1U << NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
    //!
    //! \see INetworkDefinition, NetworkDefinitionCreationFlags
    //!
    nvinfer1::INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept
    {
        return mImpl->createNetworkV2(flags);
    }

    //!
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
    //!
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
    //! \deprecated Deprecated in TensorRT 10.5. Please query data type support from CUDA directly.
    //!
    TRT_DEPRECATED bool platformHasTf32() const noexcept
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
    //! \note This function will synchronize the CUDA stream returned by \p config.getProfileStream() before returning.
    //!
    //! \see INetworkDefinition, IBuilderConfig, IHostMemory
    //!
    nvinfer1::IHostMemory* buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept
    {
        return mImpl->buildSerializedNetwork(network, config);
    }

    //!
    //! \brief Builds a network for the given INetworkDefinition and IBuilderConfig.
    //!
    //! \param network Network definition.
    //! \param config Builder configuration.
    //!
    //! \return A pointer to a ICudaEngine object that contains an engine.
    //!
    //! \note This function will synchronize the CUDA stream returned by \p config.getProfileStream() before returning.
    //!
    //! \note This function does not support \p BuilderFlag::kVERSION_COMPATIBLE.
    //! Please use \p buildSerializedNetwork to get a version compatible engine.
    //!
    //! \see INetworkDefinition, IBuilderConfig, ICudaEngine
    //!
    nvinfer1::ICudaEngine* buildEngineWithConfig(INetworkDefinition& network, IBuilderConfig& config) noexcept
    {
        return mImpl->buildEngineWithConfig(network, config);
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
    //! \note This function will synchronize the CUDA stream returned by \p config.getProfileStream() before returning.
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
    //!
    //! \param maxThreads The maximum number of threads that can be used by the builder.
    //!
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

    //!
    //! \brief get the local plugin registry that can be used by the builder.
    //!
    //! \return The local plugin registry that can be used by the builder.
    //!
    IPluginRegistry& getPluginRegistry() noexcept
    {
        return mImpl->getPluginRegistry();
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
//! \brief Return the plugin registry for building a Standard engine, or nullptr if no registry exists.
//!
//! Also return nullptr if the input argument is not EngineCapability::kSTANDARD.
//! Engine capabilities EngineCapability::kSTANDARD and EngineCapability::kSAFETY have distinct plugin registries.
//! When building a Safety engine, use nvinfer1::getBuilderSafePluginRegistry().
//! Use IPluginRegistry::registerCreator from the registry to register plugins.
//! Plugins registered in a registry associated with a specific engine capability are only available when
//! building engines with that engine capability.
//!
//! There is no plugin registry for EngineCapability::kDLA_STANDALONE.
//!
extern "C" TENSORRTAPI nvinfer1::IPluginRegistry* getBuilderPluginRegistry(
    nvinfer1::EngineCapability capability) noexcept;

namespace safe
{
//! Forward declaration
class IPluginRegistry;
} // namespace safe

//!
//! \brief Return the plugin registry for building a Safety engine, or nullptr if no registry exists.
//!
//! Also return nullptr if the input argument is not EngineCapability::kSAFETY.
//! When building a Standard engine, use nvinfer1::getBuilderPluginRegistry().
//! Use safe::IPluginRegistry::registerCreator from the registry to register plugins.
//!
extern "C" TENSORRTAPI nvinfer1::safe::IPluginRegistry* getBuilderSafePluginRegistry(
    nvinfer1::EngineCapability capability) noexcept;

} // namespace nvinfer1

#endif // NV_INFER_H
