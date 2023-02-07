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

// Docstrings for the pyCaffe parser bindings.
#pragma once

namespace tensorrt
{
namespace ICaffeParserDoc
{
constexpr const char* descr = R"trtdoc(
    This class is used for parsing Caffe models. It allows users to export models trained using Caffe to TRT.

    :ivar plugin_factory_v2: :class:`ICaffePluginFactoryV2` The ICaffePluginFactory used to create the user defined plugins.
    :ivar plugin_namespace: :class:`str` The namespace used to lookup and create plugins in the network.
    :ivar protobuf_buffer_size: :class:`int` The buffer size for the parsing and storage of the learned model.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
)trtdoc";

constexpr const char* parse = R"trtdoc(
    Parse a prototxt file and a binaryproto Caffe model to extract network definition and weights associated with the network, respectively.

    :arg deploy: The plain text, prototxt file used to define the network definition.
    :arg model:  The binaryproto Caffe model that contains the weights associated with the network.
    :arg network: Network in which the CaffeParser will fill the layers.
    :arg dtype: The type to which the weights will be transformed.

    :returns: An :class:`IBlobNameToTensor` object that contains the extracted data.
)trtdoc";

constexpr const char* parse_buffer = R"trtdoc(
    Parse a prototxt file and a binaryproto Caffe model to extract network definition and weights associated with the network, respectively.

    :arg deploy_buffer: The memory buffer containing the plain text deploy prototxt used to define the network definition.
    :arg model_buffer: The binaryproto Caffe memory buffer that contains the weights associated with the network.
    :arg network: Network in which the CaffeParser will fill the layers.
    :arg dtype: The type to which the weights will be transformed.

    :returns: An :class:`IBlobNameToTensor` object that contains the extracted data.
)trtdoc";

constexpr const char* parse_binary_proto = R"trtdoc(
    Parse and extract data stored in binaryproto file. The binaryproto file contains data stored in a binary blob. :func:`parse_binary_proto` converts it to an :class:`numpy.ndarray` object.

    :arg filename:  Path to file containing binary proto.

    :returns: :class:`numpy.ndarray` An array that contains the extracted data.
)trtdoc";

} // namespace ICaffeParserDoc

namespace IBlobNameToTensorDoc
{
constexpr const char* descr = R"trtdoc(
    This class is used to store and query :class:`ITensor` s after they have been extracted from a Caffe model using the :class:`CaffeParser` .
)trtdoc";

constexpr const char* find = R"trtdoc(
    Given a blob name, this function returns an :class:`ITensor` object.

    :arg name: Caffe blob name for which the user wants the corresponding :class:`ITensor` .

    :returns: A :class:`ITensor` object corresponding to the queried name. If no such :class:`ITensor` exists, then an empty object is returned.
)trtdoc";
} // namespace IBlobNameToTensorDoc

namespace ICaffePluginFactoryV2Doc
{
constexpr const char* descr = R"trtdoc(
    Plugin factory used to configure plugins.
)trtdoc";

constexpr const char* is_plugin_v2 = R"trtdoc(
    A user implemented function that determines if a layer configuration is provided by an :class:`IPluginV2` .

    :arg layer_name: Name of the layer which the user wishes to validate.

    :returns: True if the the layer configuration is provided by an :class:`IPluginV2` .
)trtdoc";

constexpr const char* create_plugin = R"trtdoc(
    Creates a plugin.

        :arg layer_name: Name of layer associated with the plugin.
        :arg weights: Weights used for the layer.

    :returns: The newly created :class:`IPluginV2` .
)trtdoc";
} // namespace ICaffePluginFactoryV2Doc

namespace FreeFunctionsDoc
{
constexpr const char* shutdown_protobuf_library = R"trtdoc(
    Shuts down protocol buffers library.
)trtdoc";
}
} // namespace tensorrt
