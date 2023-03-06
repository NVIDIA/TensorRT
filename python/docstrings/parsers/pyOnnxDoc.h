/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Docstrings for the pyCaffe parser bindings.
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
    Check whether TensorRT supports a particular ONNX model.

    :arg model: The serialized ONNX model.
    :arg path: The path to the model file. Only required if the model has externally stored weights.

    :returns: Tuple[bool, List[Tuple[NodeIndices, bool]]]
        The first element of the tuple indicates whether the model is supported.
        The second indicates subgraphs (by node index) in the model and whether they are supported.
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
constexpr const char* VERSION_COMPATIBLE = R"trtdoc(
   Parse the ONNX model into the INetworkDefinition with the intention of building a version-compatible engine in TensorRT 8.6.
   This flag is planned to be deprecated in TensorRT 8.7, and removed in TensorRT 9.0.
   This will choose TensorRT's native InstanceNormalization implementation over the plugin implementation.
   There may be performance degradations when this flag is enabled.
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
