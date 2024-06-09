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
   The flag is ON by default.
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
:returns: The Onnx version
)trtdoc";

namespace IOnnxPluginFactoryDoc
{
constexpr const char* descr = R"trtdoc(
    This plugin factory handles deserialization of the plugins that are built
    into the ONNX parser. Engines with legacy plugin layers built using the ONNX parser
    must use this plugin factory during deserialization.
)trtdoc";

constexpr const char* init = R"trtdoc(
    :arg logger: The logger to use.
)trtdoc";
} // namespace IOnnxPluginFactoryDoc

} // namespace tensorrt
