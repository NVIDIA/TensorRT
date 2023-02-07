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

#pragma once

namespace tensorrt
{
namespace IPluginV2Doc
{
constexpr const char* descr = R"trtdoc(
    Plugin class for user-implemented layers.

    Plugins are a mechanism for applications to implement custom layers. When
    combined with IPluginCreator it provides a mechanism to register plugins and
    look up the Plugin Registry during de-serialization.


    :ivar num_outputs: :class:`int` The number of outputs from the layer. This is used by the implementations of :class:`INetworkDefinition` and :class:`Builder` . In particular, it is called prior to any call to :func:`initialize` .
    :ivar tensorrt_version: :class:`int` The API version with which this plugin was built.
    :ivar plugin_type: :class:`str` The plugin type. Should match the plugin name returned by the corresponding plugin creator
    :ivar plugin_version: :class:`str` The plugin version. Should match the plugin version returned by the corresponding plugin creator.
    :ivar plugin_namespace: :class:`str` The namespace that this plugin object belongs to. Ideally, all plugin objects from the same plugin library should have the same namespace.
    :ivar serialization_size: :class:`int` The size of the serialization buffer required.
)trtdoc";

constexpr const char* get_output_shape = R"trtdoc(
    Get the dimension of an output tensor.

    :arg index: The index of the output tensor.
    :arg input_shapes: The shapes of the input tensors.

    This function is called by the implementations of :class:`INetworkDefinition` and :class:`Builder` . In particular, it is called prior to any call to :func:`initialize` .
)trtdoc";

constexpr const char* supports_format = R"trtdoc(
    Check format support.

    This function is called by the implementations of :class:`INetworkDefinition` , :class:`Builder` , and :class:`ICudaEngine` . In particular, it is called when creating an engine and when deserializing an engine.

    :arg dtype: Data type requested.
    :arg format: TensorFormat requested.

    :returns: True if the plugin supports the type-format combination.
)trtdoc";

constexpr const char* configure_with_format = R"trtdoc(
    Configure the layer.

    This function is called by the :class:`Builder` prior to :func:`initialize` . It provides an opportunity for the layer to make algorithm choices on the basis of its weights, dimensions, and maximum batch size.

    The dimensions passed here do not include the outermost batch size (i.e. for 2D image networks, they will be 3D CHW dimensions).

    :arg input_shapes: The shapes of the input tensors.
    :arg output_shapes: The shapes of the output tensors.
    :arg dtype: The data type selected for the engine.
    :arg format: The format selected for the engine.
    :arg max_batch_size: The maximum batch size.
)trtdoc";

constexpr const char* initialize = R"trtdoc(
    Initialize the layer for execution. This is called when the engine is created.

    :returns: 0 for success, else non-zero (which will cause engine termination).
)trtdoc";

constexpr const char* terminate = R"trtdoc(
    Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
)trtdoc";

constexpr const char* get_workspace_size = R"trtdoc(
    Find the workspace size required by the layer.

    This function is called during engine startup, after :func:`initialize` . The workspace size returned should be sufficient for any batch size up to the maximum.

    :arg max_batch_size: :class:`int` The maximum possible batch size during inference.

    :returns: The workspace size.
)trtdoc";

constexpr const char* execute_async = R"trtdoc(
    Execute the layer asynchronously.

    :arg batch_size: The number of inputs in the batch.
    :arg inputs: The memory for the input tensors.
    :arg outputs: The memory for the output tensors.
    :arg workspace: Workspace for execution.
    :arg stream_handle: The stream in which to execute the kernels.

    :returns: 0 for success, else non-zero (which will cause engine termination).
)trtdoc";

constexpr const char* serialize = R"trtdoc(
    Serialize the plugin.
)trtdoc";

constexpr const char* destroy = R"trtdoc(
    Destroy the plugin object. This will be called when the :class:`INetworkDefinition` , :class:`Builder` or :class:`ICudaEngine` is destroyed.
)trtdoc";

constexpr const char* clone = R"trtdoc(
    Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with these parameters.
)trtdoc";
} // namespace IPluginV2Doc

namespace IPluginV2ExtDoc
{
constexpr const char* descr = R"trtdoc(
    Plugin class for user-implemented layers.

    Plugins are a mechanism for applications to implement custom layers. This interface provides additional capabilities to the IPluginV2 interface by supporting different output data types.

    :ivar tensorrt_version: :class:`int` The API version with which this plugin was built.
)trtdoc";

constexpr const char* get_output_data_type = R"trtdoc(

    Return the DataType of the plugin output at the requested index.
    The default behavior should be to return the type of the first input, or DataType::kFLOAT if the layer has no inputs.
    The returned data type must have a format that is supported by the plugin.

    :arg index: Index of the output for which Data type is requested.
    :arg input_types: Data types of the inputs.

    :returns: DataType of the plugin output at the requested index.
)trtdoc";

constexpr const char* configure_plugin = R"trtdoc(
    Configure the layer.

    This function is called by the :class:`Builder` prior to :func:`initialize` . It provides an opportunity for the layer to make algorithm choices on the basis of its weights, dimensions, and maximum batch size.

    The dimensions passed here do not include the outermost batch size (i.e. for 2D image networks, they will be 3D CHW dimensions).

    :arg input_shapes: The shapes of the input tensors.
    :arg output_shapes: The shapes of the output tensors.
    :arg input_types: The data types of the input tensors.
    :arg output_types: The data types of the output tensors.
    :arg input_is_broadcasted: Whether an input is broadcasted across the batch.
    :arg output_is_broadcasted: Whether an output is broadcasted across the batch.
    :arg format: The format selected for floating-point inputs and outputs of the engine.
    :arg max_batch_size: The maximum batch size.
)trtdoc";

constexpr const char* clone = R"trtdoc(
    Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.

    If the source plugin is pre-configured with configure_plugin(), the returned object should also be pre-configured. The returned object should allow attach_to_context() with a new execution context.
    Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object (e.g. via ref-counting) to avoid duplication.
)trtdoc";

constexpr const char* attach_to_context = R"trtdoc(
    Attach the plugin object to an execution context and grant the plugin the access to some context resource.

    :arg cudnn The cudnn context handle of the execution context
    :arg cublas The cublas context handle of the execution context
    :arg allocator The allocator used by the execution context

    This function is called automatically for each plugin when a new execution context is created. If the plugin needs per-context resource, it can be allocated here. The plugin can also get context-owned CUDNN and CUBLAS context here.
)trtdoc";

constexpr const char* detach_from_context = R"trtdoc(
    Detach the plugin object from its execution context.

    This function is called automatically for each plugin when a execution context is destroyed. If the plugin owns per-context resource, it can be released here.
)trtdoc";
} // namespace IPluginV2ExtDoc

namespace PluginFieldTypeDoc
{
constexpr const char* descr = R"trtdoc(
    The possible field types for custom layer.
)trtdoc";
} // namespace PluginFieldTypeDoc

namespace PluginFieldDoc
{
constexpr const char* descr = R"trtdoc(
    Contains plugin attribute field names and associated data.
    This information can be parsed to decode necessary plugin metadata

    :ivar name: :class:`str` Plugin field attribute name.
    :ivar data: :class:`buffer` Plugin field attribute data.
    :ivar type: :class:`PluginFieldType` Plugin field attribute type.
    :ivar size: :class:`int` Number of data entries in the Plugin attribute.
)trtdoc";
} // namespace PluginFieldDoc

namespace PluginFieldCollectionDoc
{
constexpr const char* descr = R"trtdoc(
    Contains plugin attribute field names and associated data.
    This information can be parsed to decode necessary plugin metadata

    :ivar num_fields: :class:`int`  Number of :class:`PluginField` entries.
    :ivar fields: :class:`list` PluginField entries.
)trtdoc";
} // namespace PluginFieldCollectionDoc

namespace IPluginCreatorDoc
{
constexpr const char* descr = R"trtdoc(
    Plugin creator class for user implemented layers

    :ivar tensorrt_version: :class:`int`  Number of :class:`PluginField` entries.
    :ivar name: :class:`str` Plugin name.
    :ivar plugin_version: :class:`str` Plugin version.
    :ivar field_names: :class:`list` List of fields that needs to be passed to :func:`create_plugin` .
    :ivar plugin_namespace: :class:`str` The namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator.
)trtdoc";

constexpr const char* create_plugin = R"trtdoc(
    Creates a new plugin.

    :arg name: The name of the plugin.
    :arg field_collection: The :class:`PluginFieldCollection` for this plugin.

    :returns: :class:`IPluginV2` or :class:`None` on failure.
)trtdoc";

constexpr const char* deserialize_plugin = R"trtdoc(
    Creates a plugin object from a serialized plugin.

    :arg name: Name of the plugin.
    :arg serialized_plugin: A buffer containing a serialized plugin.

    :returns: A new :class:`IPluginV2`
)trtdoc";
} // namespace IPluginCreatorDoc

namespace IPluginRegistryDoc
{
constexpr const char* descr = R"trtdoc(
    Registers plugin creators.

    :ivar plugin_creator_list: All the registered plugin creators.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
)trtdoc";

constexpr const char* register_creator = R"trtdoc(
    Register a plugin creator.

    :arg creator: The IPluginCreator instance.
    :arg plugin_namespace: The namespace of the plugin creator.

    :returns: False if one with the same type is already registered.
)trtdoc";

constexpr const char* deregister_creator = R"trtdoc(
    Deregister a previously registered plugin creator.

    Since there may be a desire to limit the number of plugins,
    this function provides a mechanism for removing plugin creators registered in TensorRT.
    The plugin creator that is specified by ``creator`` is removed from TensorRT and no longer tracked.

    :arg creator: The IPluginCreator instance.

    :returns: ``True`` if the plugin creator was deregistered, ``False`` if it was not found in the registry
            or otherwise could not be deregistered.
)trtdoc";

constexpr const char* get_plugin_creator = R"trtdoc(
    Return plugin creator based on type and version

    :arg type: The type of the plugin.
    :arg version: The version of the plugin.
    :arg plugin_namespace: The namespace of the plugin.

    :returns: An :class:`IPluginCreator` .
)trtdoc";
} // namespace IPluginRegistryDoc

namespace FreeFunctionsDoc
{
constexpr const char* get_plugin_registry = R"trtdoc(
    Return the plugin registry for standard runtime
)trtdoc";

constexpr const char* get_builder_plugin_registry = R"trtdoc(
    Return the plugin registry used for building engines for the specified runtime
)trtdoc";

constexpr const char* init_libnvinfer_plugins = R"trtdoc(
    Initialize and register all the existing TensorRT plugins to the :class:`IPluginRegistry` with an optional namespace.
    The plugin library author should ensure that this function name is unique to the library.
    This function should be called once before accessing the Plugin Registry.

    :arg logger: Logger to print plugin registration information.
    :arg namespace: Namespace used to register all the plugins in this library.
)trtdoc";
} // namespace FreeFunctionsDoc

} // namespace tensorrt
