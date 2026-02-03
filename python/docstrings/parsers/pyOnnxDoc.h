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

// Docstrings for the pyOnnx parser bindings.
#pragma once

namespace tensorrt
{
namespace OnnxParserDoc
{
constexpr const char* descr = R"trtdoc(
    This class is used for parsing ONNX models into a TensorRT network definition

    :ivar num_errors: :class:`int` The number of errors that occurred during prior calls to :func:`parse`
)trtdoc";

constexpr const char* init = R"trtdoc(
    :arg network: The network definition to which the parser will write.
    :arg logger: The logger to use.
)trtdoc";

constexpr const char* parse = R"trtdoc(
    Parse a serialized ONNX model into the TensorRT network.

    :arg model: The serialized ONNX model.
    :arg path: The path to the model file. Only required if the model has externally stored weights.

    :returns: true if the model was parsed successfully
)trtdoc";

constexpr const char* parse_with_weight_descriptors = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.13. See load_initializers.

    Parse a serialized ONNX model into the TensorRT network with consideration of user provided weights.

    :arg model: The serialized ONNX model.

    :returns: true if the model was parsed successfully
)trtdoc";

constexpr const char* parse_from_file = R"trtdoc(
    Parse an ONNX model from file into a TensorRT network.

    :arg model: The path to an ONNX model.

    :returns: true if the model was parsed successfully
)trtdoc";

constexpr const char* supports_model = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.1. See supports_model_v2.

    Check whether TensorRT supports a particular ONNX model.

    :arg model: The serialized ONNX model.
    :arg path: The path to the model file. Only required if the model has externally stored weights.

    :returns: Tuple[bool, List[Tuple[NodeIndices, bool]]]
        The first element of the tuple indicates whether the model is supported.
        The second indicates subgraphs (by node index) in the model and whether they are supported.
)trtdoc";

constexpr const char* supports_model_v2 = R"trtdoc(
    Check whether TensorRT supports a particular ONNX model.
    Query each subgraph with num_subgraphs, is_subgraph_supported, get_subgraph_nodes.

    :arg model: The serialized ONNX model.
    :arg path: The path to the model file. Only required if the model has externally stored weights.
    :returns: true if the model is supported
)trtdoc";

constexpr const char* num_subgraphs = R"trtdoc(
    Get the number of subgraphs. Calling before \p supportsModelV2 is an undefined behavior. Will return 0 by default.

    :returns: Number of subgraphs
)trtdoc";

constexpr const char* is_subgraph_supported = R"trtdoc(
    Returns whether the subgraph is supported. Calling before \p supportsModelV2 is an undefined behavior.
    Will return false by default.

    :arg index: Index of the subgraph to be checked.
    :returns: true if subgraph is supported
)trtdoc";

constexpr const char* get_subgraph_nodes = R"trtdoc(
    Get the nodes of the specified subgraph. Calling before \p supportsModelV2 is an undefined behavior.
    Will return an empty list by default.

    :arg index: Index of the subgraph.
    :returns: List[int]
        A list of node indices in the subgraph.
)trtdoc";

constexpr const char* supports_operator = R"trtdoc(
    Returns whether the specified operator may be supported by the parser.
    Note that a result of true does not guarantee that the operator will be supported in all cases (i.e., this function may return false-positives).

    :arg op_name:  The name of the ONNX operator to check for support
)trtdoc";

constexpr const char* get_error = R"trtdoc(
    Get an error that occurred during prior calls to :func:`parse`

    :arg index: Index of the error
)trtdoc";

constexpr const char* clear_errors = R"trtdoc(
    Clear errors from prior calls to :func:`parse`
)trtdoc";

constexpr const char* clear_flag = R"trtdoc(
    Clears the parser flag from the enabled flags.

    :arg flag: The flag to clear.
)trtdoc";

constexpr const char* get_flag = R"trtdoc(
    Check if a build mode flag is set.

    :arg flag: The flag to check.

    :returns: A `bool` indicating whether the flag is set.
)trtdoc";

constexpr const char* set_flag = R"trtdoc(
    Add the input parser flag to the already enabled flags.

    :arg flag: The flag to set.
)trtdoc";

constexpr const char* get_layer_output_tensor = R"trtdoc(
    Get the i-th output ITensor object for the ONNX layer "name".

   In the case of multiple nodes sharing the same name this function will return
   the output tensors of the first instance of the node in the ONNX graph.

    :arg name: The name of the ONNX layer.

    :arg i: The index of the output.

    :returns: The output tensor or None if the layer was not found or an invalid index was provided.
)trtdoc";

constexpr const char* get_used_vc_plugin_libraries = R"trtdoc(
    Query the plugin libraries needed to implement operations used by the parser in a version-compatible engine.

    This provides a list of plugin libraries on the filesystem needed to implement operations
    in the parsed network.  If you are building a version-compatible engine using this network,
    provide this list to IBuilderConfig.set_plugins_to_serialize() to serialize these plugins along
    with the version-compatible engine, or, if you want to ship these plugin libraries externally
    to the engine, ensure that IPluginRegistry.load_library() is used to load these libraries in the
    appropriate runtime before deserializing the corresponding engine.

    :returns: List[str] List of plugin libraries found by the parser.

    :raises: :class:`RuntimeError` if an internal error occurred when trying to fetch the list of plugin libraries.
)trtdoc";

constexpr const char* load_model_proto = R"trtdoc(
    Load a serialized ONNX model into the parser. Unlike the parse(), parse_from_file(), or parse_with_weight_descriptors()
    functions, this function does not immediately convert the model into a TensorRT INetworkDefinition. Using this function
    allows users to provide their own initializers for the ONNX model through the load_initializer() function.

    Only one model can be loaded at a time. Subsequent calls to load_model_proto() will result in an error.

    To begin the conversion of the model into a TensorRT INetworkDefinition, use parse_model_proto().

    :arg model: The serialized ONNX model.
    :arg path: The path to the model file. Only required if the model has externally stored weights.

    :returns: true if the model was loaded successfully

)trtdoc";

constexpr const char* load_initializer = R"trtdoc(
    Prompt the ONNX parser to load an initializer with user-provided binary data.
    The lifetime of the data must exceed the lifetime of the parser.

    All user-provided initializers must be provided prior to calling parse_model_proto().

    This function can be called multiple times to specify the names of multiple initializers.

    Calling this function with an initializer previously specified will overwrite the previous instance.

    This function will return false if initializer validation fails. Possible validation errors are:
     * This function was called prior to load_model_proto().
     * The requested initializer was not found in the model.
     * The size of the data provided is different from the corresponding initializer in the model.

    :arg name: name of the initializer.
    :arg data: binary data of the initializer.
    :arg size: the size of the binary data.

    :returns: true if the initializer was successfully loaded

)trtdoc";

constexpr const char* parse_model_proto = R"trtdoc(

    Begin the parsing and conversion process of the loaded ONNX model into a TensorRT INetworkDefinition.

    :returns: true if the model was parsed successfully.

)trtdoc";

constexpr const char* set_builder_config = R"trtdoc(
    Set the BuilderConfig for the parser.

    :arg builder_config: The BuilderConfig to set.

    :returns: true if the BuilderConfig was set successfully, false otherwise.
)trtdoc";

} // namespace OnnxParserDoc

namespace OnnxParserRefitterDoc
{
constexpr const char* descr = R"trtdoc(
    This is an interface designed to refit weights from an ONNX model.
)trtdoc";

constexpr const char* init = R"trtdoc(
    :arg refitter: The Refitter object used to refit the model.
    :arg logger: The logger to use.
)trtdoc";

constexpr const char* refit_from_bytes = R"trtdoc(
    Load a serialized ONNX model from memory and perform weight refit.

    :arg model: The serialized ONNX model.
    :arg path: The path to the model file. Only required if the model has externally stored weights.

    :returns: true if all the weights in the engine were refit successfully.
)trtdoc";

constexpr const char* refit_from_file = R"trtdoc(
    Load and parse a ONNX model from disk and perform weight refit.

    :arg model: The path to an ONNX model.

    :returns: true if the model was loaded successfully, and if all the weights in the engine were refit successfully.
)trtdoc";

constexpr const char* get_error = R"trtdoc(
    Get an error that occurred during prior calls to :func:`refitFromBytes` or :func:`refitFromFile`.

    :arg index: Index of the error
)trtdoc";

constexpr const char* clear_errors = R"trtdoc(
    Clear errors from prior calls to :func:`refitFromBytes` or :func:`refitFromFile`.
)trtdoc";

constexpr const char* load_model_proto = R"trtdoc(
    Load a serialized ONNX model into the refitter. Unlike the refit() or refit_from_file()
    functions, this function does not immediately begin the refit process. Using this function
    allows users to provide their own initializers for the ONNX model through the load_initializer() function.

    Only one model can be loaded at a time. Subsequent calls to load_model_proto() will result in an error.

    To begin the refit process, use refit_model_proto().

    :arg model: The serialized ONNX model.
    :arg path: The path to the model file. Only required if the model has externally stored weights.

    :returns: true if the model was loaded successfully.

)trtdoc";

constexpr const char* load_initializer = R"trtdoc(
    Prompt the ONNX refitter to load an initializer with user-provided binary data.
    The lifetime of the data must exceed the lifetime of the refitter.

    All user-provided initializers must be provided prior to calling refit_model_proto().

    This function can be called multiple times to specify the names of multiple initializers.

    Calling this function with an initializer previously specified will overwrite the previous instance.

    This function will return false if initializer validation fails. Possible validation errors are:
     * This function was called prior to load_model_proto().
     * The requested initializer was not found in the model.
     * The size of the data provided is different from the corresponding initializer in the model.

    :arg name: name of the initializer.
    :arg data: binary data of the initializer.
    :arg size: the size of the binary data.

    :returns: true if the initializer was successfully loaded.

)trtdoc";

constexpr const char* refit_model_proto = R"trtdoc(

    Begin the refit process from the loaded ONNX model.

    :returns: true if the model was refit successfully.

)trtdoc";

} // namespace OnnxParserRefitterDoc

namespace ErrorCodeDoc
{
constexpr const char* descr = R"trtdoc(
    The type of parser error
)trtdoc";
} // namespace ErrorCodeDoc

namespace OnnxParserFlagDoc
{
constexpr const char* descr = R"trtdoc(
    Flags that control how an ONNX model gets parsed.
)trtdoc";
constexpr const char* NATIVE_INSTANCENORM = R"trtdoc(
   Parse the ONNX model into the INetworkDefinition with the intention of using TensorRT's native layer implementation over the plugin implementation for InstanceNormalization nodes.
   This flag is required when building version-compatible or hardware-compatible engines.
   This flag is ON by default.
)trtdoc";
constexpr const char* ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA = R"trtdoc(
    Enable UINT8 as a quantization data type and asymmetric quantization with non-zero zero-point values in Quantize and Dequantize nodes.
    The resulting engine must be built targeting DLA version >= 3.16.
    This flag is OFF by default.
 )trtdoc";
constexpr const char* REPORT_CAPABILITY_DLA = R"trtdoc(
    Parse the ONNX model with per-node validation for DLA. If this flag is set, is_subgraph_supported() will
    also return capability in the context of DLA support.
    When this flag is set, a valid BuilderConfig must be provided to the parser via set_builder_config().
    This flag is OFF by default.
 )trtdoc";
constexpr const char* ENABLE_PLUGIN_OVERRIDE = R"trtdoc(
    Allow a loaded plugin with the same name as an ONNX operator type to override the default ONNX implementation,
    even if the plugin namespace attribute is not set.
    This flag is useful for custom plugins that are intended to replace standard ONNX operators, for example to provide
    alternative implementations or improved performance.
    This flag is OFF by default.
)trtdoc";
} // namespace OnnxParserFlagDoc

namespace ParserErrorDoc
{
constexpr const char* descr = R"trtdoc(
    An object containing information about an error
)trtdoc";

constexpr const char* code = R"trtdoc(
    :returns: The error code
)trtdoc";

constexpr const char* desc = R"trtdoc(
    :returns: Description of the error
)trtdoc";

constexpr const char* file = R"trtdoc(
    :returns: Source file in which the error occurred
)trtdoc";

constexpr const char* line = R"trtdoc(
    :returns: Source line at which the error occurred
)trtdoc";

constexpr const char* func = R"trtdoc(
    :returns: Source function in which the error occurred
)trtdoc";

constexpr const char* node = R"trtdoc(
    :returns: Index of the Onnx model node in which the error occurred
)trtdoc";

constexpr const char* node_name = R"trtdoc(
    :returns: Name of the node in the model in which the error occurred
)trtdoc";

constexpr const char* node_operator = R"trtdoc(
    :returns: Name of the node operation in the model in which the error occurred
)trtdoc";

constexpr const char* local_function_stack = R"trtdoc(
    :returns: Current stack trace of local functions in which the error occurred
)trtdoc";

constexpr const char* local_function_stack_size = R"trtdoc(
    :returns: Size of the current stack trace of local functions in which the error occurred
)trtdoc";

} // namespace ParserErrorDoc

constexpr const char* get_nv_onnx_parser_version = R"trtdoc(
:returns: The Onnx Parser version
)trtdoc";

} // namespace tensorrt
