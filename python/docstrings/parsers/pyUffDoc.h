/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Docstrings for the pyUffBindings
namespace tensorrt
{

namespace UffInputOrderDoc
{
constexpr const char* descr = R"trtdoc(
    The different possible supported input orders.
)trtdoc";

} // namespace UffInputOrderDoc

namespace FieldTypeDoc
{
constexpr const char* descr = R"trtdoc(
    The possible field types for the custom layer.
)trtdoc";

} // namespace FieldTypeDoc

namespace FieldMapDoc
{
constexpr const char* descr = R"trtdoc(
    This is a class containing an array of field params used as a layer parameter for plugin layers. The node fields are passed by the parser to the API through the plugin constructor. The implementation of the plugin should parse the contents of the :class:`FieldMap` as part of the plugin constructor.

    :ivar name: :class:`str` field param
    :ivar data: :class:`capsule` field param
    :ivar type: :class:`FieldType` field param
    :ivar length: :class:`int` field param
)trtdoc";

} // namespace FieldMapDoc

namespace FieldCollectionDoc
{
constexpr const char* descr = R"trtdoc(
    This class contains an array of :class:`FieldMap` s.

    :ivar num_fields: :class:`int` The number of :class:`FieldMap` s.
    :ivar fields: :class:`capsule` The array of :class:`FieldMap` s.
)trtdoc";

} // namespace FieldCollectionDoc

namespace UffParserDoc
{

constexpr const char* descr = R"trtdoc(
    This class is used for parsing models described using the UFF format.

    :ivar uff_required_version_major: :class:`int` Version Major of the UFF.
    :ivar uff_required_version_minor: :class:`int` Version Minor of the UFF.
    :ivar uff_required_version_patch: :class:`int` Version Patch of the UFF.
    :ivar plugin_namespace: :class:`str` The namespace used to lookup and create plugins in the network.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
)trtdoc";

constexpr const char* register_input = R"trtdoc(
    Register an input name of a UFF network with the associated Dimensions.

    :arg name: Input name.
    :arg shape: Input shape.
    :arg order: Input order on which the framework input was originally.

    :returns: True if the name registers without error.
)trtdoc";

constexpr const char* register_output = R"trtdoc(
    Register an output name of a UFF network.

    :arg output_name: Output name.

    :returns: True if the name registers without error.
)trtdoc";

constexpr const char* parse = R"trtdoc(
    Parse a UFF file.

    :arg file:  File name of the UFF file.
    :arg network: Network in which the :class:`UffParser` will fill the layers.
    :arg weights_type:  The type on which the weights will be transformed in.

    :returns: True if the UFF file is parsed without error.
)trtdoc";

constexpr const char* parse_buffer = R"trtdoc(
    Parse a UFF buffer - useful if the file is already live in memory.

    :arg buffer:  The UFF buffer.
    :arg network: Network in which the UFFParser will fill the layers.
    :arg weights_type: The type on which the weights will be transformed in.

    :returns: True if the UFF buffer is parsed without error.
)trtdoc";
} // namespace UffParserDoc

} // namespace tensorrt
