/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_ONNX_PARSER_H
#define NV_ONNX_PARSER_H

#include "NvInfer.h"
#include <stddef.h>

//!
//! \file NvOnnxParser.h
//!
//! This is the API for the ONNX Parser
//!

#define NV_ONNX_PARSER_MAJOR 0
#define NV_ONNX_PARSER_MINOR 2
#define NV_ONNX_PARSER_PATCH 0

static constexpr int32_t NV_ONNX_PARSER_VERSION
    = ((NV_ONNX_PARSER_MAJOR * 10000) + (NV_ONNX_PARSER_MINOR * 100) + NV_ONNX_PARSER_PATCH);

//!
//! \namespace nvonnxparser
//!
//! \brief The TensorRT ONNX parser API namespace
//!
namespace nvonnxparser
{

//! \return the numerical value of the highest-valued enumerator for type T.
//! It must be specialized for each enum type that uses it.
template <typename T>
constexpr int32_t EnumMax() noexcept = delete;

//!
//! \enum ErrorCode
//!
//! \brief The type of error that the parser or refitter may return
//!
enum class ErrorCode : int
{
    kSUCCESS = 0,
    kINTERNAL_ERROR = 1,
    kMEM_ALLOC_FAILED = 2,
    kMODEL_DESERIALIZE_FAILED = 3,
    kINVALID_VALUE = 4,
    kINVALID_GRAPH = 5,
    kINVALID_NODE = 6,
    kUNSUPPORTED_GRAPH = 7,
    kUNSUPPORTED_NODE = 8,
    kUNSUPPORTED_NODE_ATTR = 9,
    kUNSUPPORTED_NODE_INPUT = 10,
    kUNSUPPORTED_NODE_DATATYPE = 11,
    kUNSUPPORTED_NODE_DYNAMIC = 12,
    kUNSUPPORTED_NODE_SHAPE = 13,
    kREFIT_FAILED = 14
};

//! Specialization. See `nvonnxparser::EnumMax()` for details.
template <>
constexpr int32_t EnumMax<ErrorCode>() noexcept
{
    return 14;
}

//!
//! \brief Represents one or more OnnxParserFlag values using binary OR
//! operations, e.g., 1U << OnnxParserFlag::kNATIVE_INSTANCENORM
//!
//! \see IParser::setFlags() and IParser::getFlags()
//!
using OnnxParserFlags
    = uint32_t;

enum class OnnxParserFlag : int32_t
{
    //! Parse the ONNX model into the INetworkDefinition with the intention of using TensorRT's native layer
    //! implementation over the plugin implementation for InstanceNormalization nodes.
    //! This flag is required when building version-compatible or hardware-compatible engines.
    //! This flag is set to be ON by default.
    kNATIVE_INSTANCENORM = 0,
    //! Enable UINT8 as a quantization data type and asymmetric quantization with non-zero zero-point values
    //! in Quantize and Dequantize nodes. This flag is set to be OFF by default.
    //! The resulting engine must be built targeting DLA version >= 3.16.
    kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA = 1,
    //! Parse the ONNX model with per-node validation for DLA. If the model is not fully supported by DLA, then
    //! parsing will fail. If this flag is set, isSubGraphSupported() will also return capability in the context of DLA
    //! support. When this flag is set, a valid IBuilderConfig must be provided to the parser via setBuilderConfig().
    // This flag is set to be OFF by default.
    kREPORT_CAPABILITY_DLA = 2,
    //! Allow a loaded plugin with the same name as an ONNX operator type to override the default ONNX implementation,
    //! even if the plugin namespace attribute is not set.
    //! Useful for custom plugins that replace standard ONNX operators, such as alternative implementations for better
    //! performance. This flag is set to be OFF by default.
    kENABLE_PLUGIN_OVERRIDE = 3,
    //! Opportunistically rewrite or modify layers to make them more amenable to running on DLA.
    kADJUST_FOR_DLA = 4
};

//! Specialization. See `nvonnxparser::EnumMax()` for details.
template <>
constexpr int32_t EnumMax<OnnxParserFlag>() noexcept
{
    return 4;
}

//!
//! \class IParserError
//!
//! \brief an object containing information about an error
//!
class IParserError
{
public:
    //!
    //!\brief the error code.
    //!
    virtual ErrorCode code() const = 0;
    //!
    //!\brief description of the error.
    //!
    virtual char const* desc() const = 0;
    //!
    //!\brief source file in which the error occurred.
    //!
    virtual char const* file() const = 0;
    //!
    //!\brief source line at which the error occurred.
    //!
    virtual int line() const = 0;
    //!
    //!\brief source function in which the error occurred.
    //!
    virtual char const* func() const = 0;
    //!
    //!\brief index of the ONNX model node in which the error occurred.
    //!
    virtual int node() const = 0;
    //!
    //!\brief name of the node in which the error occurred.
    //!
    virtual char const* nodeName() const = 0;
    //!
    //!\brief name of the node operation in which the error occurred.
    //!
    virtual char const* nodeOperator() const = 0;
    //!
    //!\brief A list of the local function names, from the top level down, constituting the current
    //!             stack trace in which the error occurred. A top-level node that is not inside any
    //!             local function would return a nullptr.
    //!
    virtual char const* const* localFunctionStack() const = 0;
    //!
    //!\brief The size of the stack of local functions at the point where the error occurred.
    //!             A top-level node that is not inside any local function would correspond to
    //              a stack size of 0.
    //!
    virtual int32_t localFunctionStackSize() const = 0;

protected:
    virtual ~IParserError() {}
};

class IRefitterObserver;

//!
//! \class IParser
//!
//! \brief an object for parsing ONNX models into a TensorRT network definition
//!
//! \warning If the ONNX model has a graph output with the same name as a graph input,
//!          the output will be renamed by prepending "__".
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IParser
{
public:
    //!
    //! \brief Parse a serialized ONNX model into the TensorRT network.
    //!         This method has very limited diagnostics. If parsing the serialized model
    //!         fails for any reason (e.g. unsupported IR version, unsupported opset, etc.)
    //!         it the user responsibility to intercept and report the error.
    //!         To obtain a better diagnostic, use the parseFromFile method below.
    //!
    //! \param serialized_onnx_model Pointer to the serialized ONNX model. Can be freed after this function returns.
    //! \param serialized_onnx_model_size Size of the serialized ONNX model
    //!        in bytes
    //! \param model_path Absolute path to the model file for loading external weights if required
    //! \return true if the model was parsed successfully
    //! \see getNbErrors() getError()
    //!
    virtual bool parse(
        void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* model_path = nullptr) noexcept
        = 0;

    //!
    //! \brief Parse an onnx model file, which can be a binary protobuf or a text onnx model
    //!         calls parse method inside.
    //!
    //! \param onnxModelFile name
    //! \param verbosity Level
    //!
    //! \return true if the model was parsed successfully
    //!
    //!
    virtual bool parseFromFile(char const* onnxModelFile, int verbosity) noexcept = 0;

    //!
    //!\brief Returns whether the specified operator may be supported by the
    //!         parser.
    //!
    //! Note that a result of true does not guarantee that the operator will be
    //! supported in all cases (i.e., this function may return false-positives).
    //!
    //! \param op_name The name of the ONNX operator to check for support
    //!
    virtual bool supportsOperator(const char* op_name) const noexcept = 0;

    //!
    //!\brief Get the number of errors that occurred during prior calls to
    //!         \p parse
    //!
    //! \see getError() clearErrors() IParserError
    //!
    virtual int getNbErrors() const noexcept = 0;

    //!
    //!\brief Get an error that occurred during prior calls to \p parse
    //!
    //! \see getNbErrors() clearErrors() IParserError
    //!
    virtual IParserError const* getError(int index) const noexcept = 0;

    //!
    //!\brief Clear errors from prior calls to \p parse
    //!
    //! \see getNbErrors() getError() IParserError
    //!
    virtual void clearErrors() noexcept = 0;

    virtual ~IParser() noexcept = default;

    //!
    //! \brief Query the plugin libraries needed to implement operations used by the parser in a version-compatible
    //! engine.
    //!
    //! This provides a list of plugin libraries on the filesystem needed to implement operations
    //! in the parsed network.  If you are building a version-compatible engine using this network,
    //! provide this list to IBuilderConfig::setPluginsToSerialize to serialize these plugins along
    //! with the version-compatible engine, or, if you want to ship these plugin libraries externally
    //! to the engine, ensure that IPluginRegistry::loadLibrary is used to load these libraries in the
    //! appropriate runtime before deserializing the corresponding engine.
    //!
    //! \param[out] nbPluginLibs Returns the number of plugin libraries in the array, or -1 if there was an error.
    //! \return Array of `nbPluginLibs` C-strings describing plugin library paths on the filesystem if nbPluginLibs > 0,
    //! or nullptr otherwise.  This array is owned by the IParser, and the pointers in the array are only valid until
    //! the next call to parse() or parseFromFile().
    //!
    virtual char const* const* getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept = 0;

    //!
    //! \brief Set the parser flags.
    //!
    //! The flags are listed in the OnnxParserFlag enum.
    //!
    //! \param OnnxParserFlags The flags used when parsing an ONNX model.
    //!
    //! \note This function will override the previous set flags, rather than bitwise ORing the new flag.
    //!
    //! \see getFlags()
    //!
    virtual void setFlags(OnnxParserFlags onnxParserFlags) noexcept = 0;

    //!
    //! \brief Get the parser flags. Defaults to 0.
    //!
    //! \return The parser flags as a bitmask.
    //!
    //! \see setFlags()
    //!
    virtual OnnxParserFlags getFlags() const noexcept = 0;

    //!
    //! \brief clear a parser flag.
    //!
    //! clears the parser flag from the enabled flags.
    //!
    //! \see setFlags()
    //!
    virtual void clearFlag(OnnxParserFlag onnxParserFlag) noexcept = 0;

    //!
    //! \brief Set a single parser flag.
    //!
    //! Add the input parser flag to the already enabled flags.
    //!
    //! \see setFlags()
    //!
    virtual void setFlag(OnnxParserFlag onnxParserFlag) noexcept = 0;

    //!
    //! \brief Returns true if the parser flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    virtual bool getFlag(OnnxParserFlag onnxParserFlag) const noexcept = 0;

    //!
    //!\brief Return the i-th output ITensor object for the ONNX layer "name".
    //!
    //! Return the i-th output ITensor object for the ONNX layer "name".
    //! If "name" is not found or i is out of range, return nullptr.
    //! In the case of multiple nodes sharing the same name this function will return
    //! the output tensors of the first instance of the node in the ONNX graph.
    //!
    //! \param name The name of the ONNX layer.
    //!
    //! \param i The index of the output. i must be in range [0, layer.num_outputs).
    //!
    virtual nvinfer1::ITensor const* getLayerOutputTensor(char const* name, int64_t i) noexcept = 0;

    //!
    //! \brief Check whether TensorRT supports a particular ONNX model.
    //!            If the function returns True, one can proceed to engine building
    //!            without having to call \p parse or \p parseFromFile.
    //!            Results can be queried through \p getNbSubgraphs, \p isSubgraphSupported,
    //!            \p getSubgraphNodes.
    //!
    //! \param serializedOnnxModel Pointer to the serialized ONNX model. Can be freed after this function returns.
    //! \param serializedOnnxModelSize Size of the serialized ONNX model in bytes
    //! \param modelPath Absolute path to the model file for loading external weights if required
    //! \return true if the model is supported
    //!
    virtual bool supportsModelV2(
        void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath = nullptr) noexcept = 0;

    //!
    //! \brief Get the number of subgraphs. Calling this function before calling \p supportsModelV2 results in undefined
    //! behavior.
    //!
    //!
    //! \return Number of subgraphs.
    //!
    virtual int64_t getNbSubgraphs() noexcept = 0;

    //!
    //! \brief Returns whether the subgraph is supported. Calling this function before calling \p supportsModelV2
    //! results in undefined behavior.
    //!
    //!
    //! \param index Index of the subgraph.
    //! \return Whether the subgraph is supported.
    //!
    virtual bool isSubgraphSupported(int64_t const index) noexcept = 0;

    //!
    //! \brief Get the nodes of the specified subgraph. Calling this function before calling \p supportsModelV2 results
    //! in undefined behavior.
    //!
    //!
    //! \param index Index of the subgraph.
    //! \param subgraphLength Returns the length of the subgraph as reference.
    //!
    //! \return Pointer to the subgraph nodes array. This pointer is owned by the Parser.
    //!
    virtual int64_t* getSubgraphNodes(int64_t const index, int64_t& subgraphLength) noexcept = 0;

    //!
    //! \brief Load a serialized ONNX model into the parser. Unlike the parse() or parseFromFile()
    //! functions, this function does not immediately convert the model into a TensorRT
    //! INetworkDefinition. Using this function allows users to provide their own initializers for the ONNX model
    //! through the loadInitializer() function.
    //!
    //! Only one model can be loaded at a time. Subsequent calls to loadModelProto() will result in an error.
    //!
    //! To begin the conversion of the model into a TensorRT INetworkDefinition, use parseModelProto().
    //!
    //! \param serializedOnnxModel Pointer to the serialized ONNX model. Can be freed after this function returns.
    //! \param serializedOnnxModelSize Size of the serialized ONNX model in bytes.
    //! \param modelPath Absolute path to the model file for loading external weights if required.
    //! \return true if the model was loaded successfully
    //! \see getNbErrors() getError()
    //!
    virtual bool loadModelProto(
        void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath = nullptr) noexcept = 0;

    //!
    //! \brief Prompt the ONNX parser to load an initializer with user-provided binary data.
    //! The lifetime of the data must exceed the lifetime of the parser.
    //!
    //! All user-provided initializers must be provided prior to calling refitModelProto().
    //!
    //! This function can be called multiple times to specify the names of multiple initializers.
    //!
    //! Calling this function with an initializer previously specified will overwrite the previous instance.
    //!
    //!
    //! This function will return false if initializer validation fails. Possible validation errors are:
    //! * This function was called prior to loadModelProto().
    //! * The requested initializer was not found in the model.
    //! * The size of the data provided is different from the corresponding initializer in the model.
    //!
    //! \param name Name of the initializer.
    //! \param data Binary data containing the values of the initializer.
    //! \param size Size of the initializer in bytes.
    //! \return true if the initializer was loaded successfully
    //! \see loadModelProto()
    //!
    virtual bool loadInitializer(char const* name, void const* data, size_t size) noexcept = 0;

    //! \brief Begin the parsing and conversion process of the loaded ONNX model into a TensorRT INetworkDefinition.
    //!
    //! \return true if conversion was successful
    //! \see getNbErrors() getError() loadModelProto() loadModelProtoFromFile()
    //!
    virtual bool parseModelProto() noexcept = 0;

    //!
    //! \brief Set the BuilderConfig for the parser.
    //!
    //! \return true if the IBuilderConfig was set successfully, false otherwise.
    //!
    virtual bool setBuilderConfig(const nvinfer1::IBuilderConfig* const builderConfig) noexcept = 0;

    //!
    //! \brief Set or clear an optional observer notified once per refittable weight during parsing.
    //!
    //! When attached, the parser emits one RefitRecord per network weight it names via
    //! INetworkDefinition::setWeightsName, at the moment the weight is created during
    //! parse/parseModelProto. This is the same record schema emitted by
    //! IParserRefitter::setRefitObserver during refit, produced without building or
    //! deserializing an engine. Because the network has not been built yet, the records are a
    //! candidate superset of the built engine's refittable weights: the builder may fold or
    //! absorb some of them. Consumers replaying the records against an engine must skip
    //! records whose trtName the engine's nvinfer1::IRefitter does not report in
    //! getAllWeights, and rely on refitCudaEngine's missing-weights check for coverage.
    //!
    //! TensorRT-RTX only: When an observer is attached to the parser associated with a BuilderConfig that sets
    //! kREFIT_INDIVIDUAL and kSTRIP_PLAN, the parser will provide placeholder weights for the listed initializers
    //! of the following ONNX nodes:
    //!  * Conv kernels and biases
    //!  * Gemm operands and bias
    //!  * MatMul operands
    //!  * DequantizeLinear scales
    //!  * DequantizeLinear FP8 or FP4 data
    //!  * TRT_FP8DequantizeLinear/TRT_MXFP8DequantizeLinear FP8 data
    //! A placeholder weight is represented by nvinfer1::Weights with the same type and count as the original
    //! initializer, but with nullptr values. All placeholder weights are expected to be refit prior to inference.
    //!
    //! A placeholder weight will not be produced under the following circumstances:
    //!  * If the initializer is also a graph input or output
    //!  * If the initializer is referenced by a nested graph
    //!
    //! Placeholders will not be created when importing a model whose producer name is "TensorRT". Networks
    //! containing placeholders must be built with IBuilder::buildSerializedNetwork().
    //!
    //! Records are only valid if the subsequent parse call returns true.
    //!
    //! May be called any time before parse / parseModelProto. Pass nullptr to detach. The
    //! observer must outlive the parse call, or be detached before destruction. Ownership
    //! remains with the caller.
    //!
    //! \see IRefitterObserver IParserRefitter::setRefitObserver
    //!
    virtual void setRefitObserver(IRefitterObserver* observer) noexcept = 0;
};

//!
//! \enum RefitTransformKind
//!
//! \brief Identifies how a refittable engine weight is produced from one or more ONNX initializers
//!        or node attributes.
//!
//! Emitted by IParserRefitter through IRefitterObserver to describe how each refittable engine
//! weight is sourced. A consumer can record these descriptions at build time and replay them at
//! engine-load time to refit directly via nvinfer1::IRefitter::setNamedWeights, without invoking
//! the ONNX parser again.
//!
enum class RefitTransformKind : int32_t
{
    //! Source is one ONNX initializer; refit data equals the initializer data verbatim.
    kIDENTITY = 0,

    //! Source is one ONNX initializer of DOUBLE type; refit data is the FLOAT cast.
    kDOUBLE_TO_FLOAT = 1,

    //! Source is the four scale/bias/mean/variance initializers of a BatchNormalization node.
    //! Refit data is combinedScale[i] = scale[i] / sqrt(variance[i] + epsilon).
    kBATCH_NORM_FOLD_SCALE = 2,

    //! Source is the four scale/bias/mean/variance initializers of a BatchNormalization node.
    //! Refit data is combinedBias[i] = bias[i] - mean[i] * combinedScale[i].
    kBATCH_NORM_FOLD_BIAS = 3,

    //! Source is the value attribute of a Constant node. The bytes are carried in
    //! RefitRecord::fixedData since they are not available as a separate ONNX initializer.
    kCONSTANT_NODE = 4,

    //! Source is the value attribute of a ConstantOfShape node (defaulting to 0.0 if absent).
    //! The bytes are carried in RefitRecord::fixedData.
    kCONSTANT_OF_SHAPE = 5,
};

//! Specialization. See `nvonnxparser::EnumMax()` for details.
template <>
constexpr int32_t EnumMax<RefitTransformKind>() noexcept
{
    return 5;
}

//!
//! \struct RefitRecord
//!
//! \brief One refittable-weight description emitted by IRefitterObserver.
//!
//! All pointers in this struct are owned by the parser and are valid only for the duration of the
//! IRefitterObserver::onRefittableWeight() call. Implementations that need to retain string or
//! buffer contents beyond the call must copy them.
//!
struct RefitRecord
{
    //! Name to pass to nvinfer1::IRefitter::setNamedWeights for this weight. Always non-null.
    char const* trtName;

    //! What transformation produces the refit data from the sources.
    RefitTransformKind kind;

    //! ONNX TensorProto::DataType of the source data **before** any transformation. For kIDENTITY
    //! this matches the parser-produced weight's ONNX type. For kDOUBLE_TO_FLOAT this is DOUBLE
    //! (the post-cast result type is given by \p trtDtype). For kBATCH_NORM_FOLD_* and the
    //! kCONSTANT* kinds the source and result ONNX types coincide.
    int32_t onnxDtype;

    //! TensorRT data type of the post-transform refit data. Provided so consumers can call
    //! nvinfer1::IRefitter::setNamedWeights directly without re-implementing the
    //! ONNX-to-TRT dtype mapping.
    nvinfer1::DataType trtDtype;

    //! Element count of the refit data.
    int64_t count;

    //! Number of source ONNX names supplied below. At most 4.
    int32_t nbSources;

    //! Names of the source ONNX initializers (or output tensor names for Constant nodes).
    //! Array of length nbSources; each entry is null-terminated. Owned by the parser.
    char const* const* sourceOnnxNames;

    //! For kBATCH_NORM_FOLD_SCALE / kBATCH_NORM_FOLD_BIAS: the epsilon used in the fold formula.
    //! For other kinds: unspecified, do not consume.
    float epsilon;

    //! For kCONSTANT_NODE and kCONSTANT_OF_SHAPE: pointer to the build-time-resolved weight
    //! bytes the parser would write into the engine. The data has element type \p onnxDtype and
    //! \p count elements; total length is \p fixedDataSize bytes. The consumer should copy this
    //! data so it can be replayed at refit time. For other kinds: nullptr.
    void const* fixedData;

    //! Length of \p fixedData in bytes, or 0 when \p fixedData is null.
    size_t fixedDataSize;
};

//!
//! \class IRefitterObserver
//!
//! \brief Observer interface invoked by IParserRefitter once per refittable engine weight.
//!
//! Attach via IParserRefitter::setRefitObserver. The intended use is to build a self-describing
//! refit table at build time that can drive nvinfer1::IRefitter::setNamedWeights directly at
//! engine-load time, eliminating the runtime dependency on the original ONNX model structure.
//!
class IRefitterObserver
{
public:
    virtual ~IRefitterObserver() noexcept = default;

    //!
    //! \brief Called once per refittable engine weight as the parser identifies it.
    //!
    //! Invoked during IParserRefitter::refitModelProto, refitFromBytes, or refitFromFile, in the
    //! parser's natural traversal order over the ONNX graph. Implementations must not call back
    //! into the parser or refitter from this method, and must not retain pointers from \p record
    //! beyond the call.
    //!
    virtual void onRefittableWeight(RefitRecord const& record) noexcept = 0;
};

//!
//! \class IParserRefitter
//!
//! \brief An interface designed to refit weights from an ONNX model.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IParserRefitter
{
public:
    //!
    //! \brief Load a serialized ONNX model from memory and perform weight refit.
    //!
    //! \param serializedOnnxModel Pointer to the serialized ONNX model
    //! \param serializedOnnxModelSize Size of the serialized ONNX model
    //!        in bytes
    //! \param modelPath Absolute path to the model file for loading external weights if required
    //! \return true if all the weights in the engine were refit successfully.
    //!
    //! The serialized ONNX model must be identical to the one used to generate the engine
    //! that will be refit.
    //!
    virtual bool refitFromBytes(
        void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath = nullptr) noexcept
        = 0;

    //!
    //! \brief Load and parse a ONNX model from disk and perform weight refit.
    //!
    //! \param onnxModelFile Path to the ONNX model to load from disk.
    //!
    //! \return true if the model was loaded successfully, and if all the weights in the engine were refit successfully.
    //!
    //! The provided ONNX model must be identical to the one used to generate the engine
    //! that will be refit.
    //!
    virtual bool refitFromFile(char const* onnxModelFile) noexcept = 0;

    //!
    //!\brief Get the number of errors that occurred during prior calls to \p refitFromBytes or \p refitFromFile
    //!
    //! \see getError() IParserError
    //!
    virtual int32_t getNbErrors() const noexcept = 0;

    //!
    //!\brief Get an error that occurred during prior calls to \p refitFromBytes or \p refitFromFile
    //!
    //! \see getNbErrors() IParserError
    //!
    virtual IParserError const* getError(int32_t index) const noexcept = 0;

    //!
    //!\brief Clear errors from prior calls to \p refitFromBytes or \p refitFromFile
    //!
    //! \see getNbErrors() getError() IParserError
    //!
    virtual void clearErrors() = 0;

    virtual ~IParserRefitter() noexcept = default;

    //!
    //! \brief Load a serialized ONNX model into the parser. Unlike the refit(), or refitFromFile()
    //! functions, this function does not immediately begin the refit process. Using this function
    //! allows users to provide their own initializers for the ONNX model through the loadInitializer() function.
    //!
    //! Only one model can be loaded at a time. Subsequent calls to loadModelProto() will result in an error.
    //!
    //! To begin the refit process, use refitModelProto().
    //!
    //! \param serializedOnnxModel Pointer to the serialized ONNX model. Can be freed after this function returns.
    //! \param serializedOnnxModelSize Size of the serialized ONNX model in bytes.
    //! \param modelPath Absolute path to the model file for loading external weights if required.
    //! \return true if the model was loaded successfully
    //! \see getNbErrors() getError()
    //!
    virtual bool loadModelProto(
        void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath = nullptr) noexcept = 0;

    //!
    //! \brief Prompt the ONNX refitter to load an initializer with user-provided binary data.
    //! The lifetime of the data must exceed the lifetime of the refitter.
    //!
    //! All user-provided initializers must be provided prior to calling refitModelProto().
    //!
    //! This function can be called multiple times to specify the names of multiple initializers.
    //!
    //! Calling this function with an initializer previously specified will overwrite the previous instance.
    //!
    //! This function will return false if initializer validation fails. Possible validation errors are:
    //! * This function was called prior to loadModelProto()
    //! * The requested initializer was not found in the model.
    //! * The size of the data provided is different from the corresponding initializer in the model.
    //!
    //! \param name Name of the initializer.
    //! \param data Binary data containing the values of the initializer.
    //! \param size Size of the initializer in bytes.
    //! \return true if the initializer was loaded successfully
    //! \see loadModelProto()
    //!
    virtual bool loadInitializer(char const* name, void const* data, size_t size) noexcept = 0;

    //! \brief Begin the refit process from the loaded ONNX model.
    //!
    //! \return true if refit was successful
    //! \see getNbErrors() getError() loadModelProto()
    //!
    virtual bool refitModelProto() noexcept = 0;

    //!
    //! \brief Attach an observer that receives one callback per refittable engine weight.
    //!
    //! May be called any time before refitModelProto / refitFromBytes / refitFromFile. Pass
    //! nullptr to detach. The observer must outlive the refit call, or be detached before
    //! destruction.
    //!
    //! \see IRefitterObserver
    //!
    virtual void setRefitObserver(IRefitterObserver* observer) noexcept = 0;
};

} // namespace nvonnxparser

extern "C" TENSORRTAPI void* createNvOnnxParser_INTERNAL(void* network, void* logger, int version) noexcept;
extern "C" TENSORRTAPI void* createNvOnnxParserRefitter_INTERNAL(
    void* refitter, void* logger, int32_t version) noexcept;
extern "C" TENSORRTAPI int getNvOnnxParserVersion() noexcept;

namespace nvonnxparser
{

namespace
{

//!
//! \brief Create a new parser object
//!
//! \param network The network definition that the parser will write to
//! \param logger The logger to use
//! \return a new parser object or NULL if an error occurred
//!
//! Any input dimensions that are constant should not be changed after parsing,
//! because correctness of the translation may rely on those constants.
//! Changing a dynamic input dimension, i.e. one that translates to -1 in
//! TensorRT, to a constant is okay if the constant is consistent with the model.
//! Each instance of the parser is designed to only parse one ONNX model once.
//!
//! \see IParser
//!
inline IParser* createParser(nvinfer1::INetworkDefinition& network, nvinfer1::ILogger& logger) noexcept
{
    return static_cast<IParser*>(createNvOnnxParser_INTERNAL(&network, &logger, NV_ONNX_PARSER_VERSION));
}

//!
//! \brief Create a new ONNX refitter object
//!
//! \param refitter The Refitter object used to refit the model
//! \param logger The logger to use
//! \return a new ParserRefitter object or NULL if an error occurred
//!
//! \see IParserRefitter
//!
inline IParserRefitter* createParserRefitter(nvinfer1::IRefitter& refitter, nvinfer1::ILogger& logger) noexcept
{
    return static_cast<IParserRefitter*>(
        createNvOnnxParserRefitter_INTERNAL(&refitter, &logger, NV_ONNX_PARSER_VERSION));
}

} // namespace

} // namespace nvonnxparser

#endif // NV_ONNX_PARSER_H
