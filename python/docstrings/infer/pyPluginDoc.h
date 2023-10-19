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

    .. warning::
        This API only applies when called on a C++ plugin from a Python program.

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
    The default behavior should be to return the type of the first input, or `DataType::kFLOAT` if the layer has no inputs.
    The returned data type must have a format that is supported by the plugin.

    :arg index: Index of the output for which data type is requested.
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

    If the source plugin is pre-configured with `configure_plugin()`, the returned object should also be pre-configured. The returned object should allow attach_to_context() with a new execution context.
    Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object (e.g. via ref-counting) to avoid duplication.
)trtdoc";

constexpr const char* attach_to_context = R"trtdoc(
    Attach the plugin object to an execution context and grant the plugin the access to some context resource.

    :arg cudnn: The cudnn context handle of the execution context
    :arg cublas: The cublas context handle of the execution context
    :arg allocator: The allocator used by the execution context

    This function is called automatically for each plugin when a new execution context is created. If the plugin needs per-context resource, it can be allocated here. The plugin can also get context-owned CUDNN and CUBLAS context here.
)trtdoc";

constexpr const char* detach_from_context = R"trtdoc(
    Detach the plugin object from its execution context.

    This function is called automatically for each plugin when a execution context is destroyed. If the plugin owns per-context resource, it can be released here.
)trtdoc";
} // namespace IPluginV2ExtDoc


namespace IPluginV2DynamicExtDoc
{
constexpr const char* descr = R"trtdoc(
    Plugin class for user-implemented layers.

    Plugins are a mechanism for applications to implement custom layers.

    Similar to `IPluginV2Ext` (including capability to support different output data types), but with support for dynamic shapes.

    This class is made available for the purpose of implementing `IPluginV2DynamicExt` plugins with Python. Inherited
    Python->C++ bindings from `IPluginV2` and `IPluginV2Ext` will continue to work on C++-based `IPluginV2DynamicExt` plugins. 

    .. note::
        Every attribute except `tensorrt_version` must be explicitly initialized on Python-based plugins. Except `plugin_namespace`,
        these attributes will be read-only when accessed through a C++-based plugin.

    :ivar num_outputs: :class:`int` The number of outputs from the plugin. This is used by the implementations of :class:`INetworkDefinition` and :class:`Builder`. In particular, it is called prior to any call to :func:`initialize`.
    :ivar tensorrt_version: :class:`int` [READ ONLY] The API version with which this plugin was built.
    :ivar plugin_type: :class:`str` The plugin type. Should match the plugin name returned by the corresponding plugin creator.
    :ivar plugin_version: :class:`str` The plugin version. Should match the plugin version returned by the corresponding plugin creator.
    :ivar plugin_namespace: :class:`str` The namespace that this plugin object belongs to. Ideally, all plugin objects from the same plugin library should have the same namespace.
    :ivar serialization_size: :class:`int` [READ ONLY] The size of the serialization buffer required.
)trtdoc";

constexpr const char* initialize = R"trtdoc(
    Initialize the plugin for execution. This is called when the engine is created.

    .. note::
        When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `pass`. 

    .. warning::
        In contrast to the C++ API for `initialize()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
        if an error occurs. 

    .. warning::
        This `initialize()` method is not available to be called from Python on C++-based plugins.
        
)trtdoc";

constexpr const char* terminate = R"trtdoc(
    Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.

    .. note::
        When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `pass`. 

)trtdoc";

constexpr const char* get_output_dimensions = R"trtdoc(

    Get expressions for computing dimensions of an output tensor from dimensions of the input tensors.

    This function is called by the implementations of `IBuilder` during analysis of the network.

    .. warning::
        This `get_output_dimensions()` method is not available to be called from Python on C++-based plugins 

    :arg output_index:	The index of the output tensor
    :arg inputs:	Expressions for dimensions of the input tensors
    :arg expr_builder:	Object for generating new expressions

    :returns: Expression for the output dimensions at the given `output_index`.
)trtdoc";

constexpr const char* get_output_data_type = R"trtdoc(

    Return the `DataType` of the plugin output at the requested index.
    The default behavior should be to return the type of the first input, or `DataType::kFLOAT` if the layer has no inputs.
    The returned data type must have a format that is supported by the plugin.

    :arg index: Index of the output for which the data type is requested.
    :arg input_types: Data types of the inputs.

    :returns: `DataType` of the plugin output at the requested `index`.
)trtdoc";

constexpr const char* configure_plugin = R"trtdoc(
    Configure the plugin.

    This function can be called multiple times in both the build and execution phases. The build phase happens before `initialize()` is called and only occurs during creation of an engine by `IBuilder`. The execution phase happens after `initialize()` is called and occurs during both creation of an engine by `IBuilder` and execution of an engine by `IExecutionContext`.

    Build phase: `configure_plugin()` is called when a plugin is being prepared for profiling but not for any specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of input and output formats, along with the bound of possible dimensions. The min and max value of the `DynamicPluginTensorDesc` correspond to the `kMIN` and `kMAX` value of the current optimization profile that the plugin is being profiled for, with the `desc.dims` field corresponding to the dimensions of plugin specified at network creation. Wildcard dimensions will exist during this phase in the `desc.dims` field.

    Execution phase: `configure_plugin()` is called when a plugin is being prepared for executing the plugin for specific dimensions. This provides an opportunity for the plugin to change algorithmic choices based on the explicit input dimensions stored in `desc.dims` field.

    .. warning::
        This `configure_plugin()` method is not available to be called from Python on C++-based plugins 

    :arg in: The input tensors attributes that are used for configuration.
    :arg out: The output tensors attributes that are used for configuration.
)trtdoc";

constexpr const char* supports_format_combination = R"trtdoc(
    Return true if plugin supports the format and datatype for the input/output indexed by pos.

    For this method, inputs are indexed from `[0, num_inputs-1]` and outputs are indexed from `[num_inputs, (num_inputs + num_outputs - 1)]`. `pos` is an index into `in_ou`t, where `0 <= pos < (num_inputs + num_outputs - 1)`.

    TensorRT invokes this method to query if the input/output tensor indexed by `pos` supports the format and datatype specified by `in_out[pos].format` and `in_out[pos].type`. The override shall return true if that format and datatype at `in_out[pos]` are supported by the plugin. It is undefined behavior to examine the format or datatype or any tensor that is indexed by a number greater than `pos`.

    .. warning::
        This `supports_format_combination()` method is not available to be called from Python on C++-based plugins

    :arg pos: The input or output tensor index being queried.
    :arg in_out: The combined input and output tensor descriptions.
    :arg num_inputs: The number of inputs.

    :returns: boolean indicating whether the format combination is supported or not.

)trtdoc";

constexpr const char* get_workspace_size = R"trtdoc(
    Return the workspace size (in bytes) required by the plugin.

    This function is called after the plugin is configured, and possibly during execution. The result should be a sufficient workspace size to deal with inputs and outputs of the given size or any smaller problem.

    .. note::
        When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `return 0`. 

    .. warning::
        This `get_workspace_size()` method is not available to be called from Python on C++-based plugins 

    :arg input_desc: How to interpret the memory for the input tensors.
    :arg output_desc: How to interpret the memory for the output tensors.

    :returns: The workspace size (in bytes).
)trtdoc";

constexpr const char* destroy = R"trtdoc(
    Destroy the plugin object. This will be called when the :class:`INetworkDefinition` , :class:`Builder` or :class:`ICudaEngine` is destroyed.

    .. note::
        When implementing a Python-based plugin, implementing this method is optional. The default behavior is a `pass`. 

)trtdoc";

constexpr const char* enqueue = R"trtdoc(
    Execute the layer.

    `inputs` and `outputs` contains pointers to the corresponding input and output device buffers as their `intptr_t` casts. `stream` also represents an `intptr_t` cast of the CUDA stream in which enqueue should be executed.
    
    .. warning::
        Since input, output, and workspace buffers are created and owned by TRT, care must be taken when writing to them from the Python side.

    .. warning::
        In contrast to the C++ API for `enqueue()`, this method must not return an error code. The expected behavior is to throw an appropriate exception.
        if an error occurs. 

    .. warning::
        This `enqueue()` method is not available to be called from Python on C++-based plugins.

    :arg input_desc:	how to interpret the memory for the input tensors.
    :arg output_desc:	how to interpret the memory for the output tensors.
    :arg inputs:	The memory for the input tensors.
    :arg outputs:   The memory for the output tensors.
    :arg workspace: Workspace for execution.
    :arg stream:	The stream in which to execute the kernels.

)trtdoc";

constexpr const char* clone = R"trtdoc(
    Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.

    If the source plugin is pre-configured with `configure_plugin()`, the returned object should also be pre-configured. 
    Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object to avoid duplication.
)trtdoc";

constexpr const char* get_serialization_size = R"trtdoc(
    Return the serialization size (in bytes) required by the plugin.

    .. note::
        When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `return len(serialize())`. 

)trtdoc";

constexpr const char* serialize = R"trtdoc(
    Serialize the plugin.

    .. warning::
        This API only applies when implementing a Python-based plugin.

    :returns: A bytes object containing the serialized representation of the plugin.

)trtdoc";


} // namespace IPluginV2DynamicExtDoc

namespace PluginFieldTypeDoc
{
constexpr const char* descr = R"trtdoc(
    The possible field types for custom layer.
)trtdoc";
} // namespace PluginFieldTypeDoc

namespace PluginTensorDescDoc
{
constexpr const char* descr = R"trtdoc(
    Fields that a plugin might see for an input or output.

    `scale` is only valid when the `type` is `DataType.INT8`. TensorRT will set the value to -1.0 if it is invalid.

    :ivar dims: :class:`Dims` 	Dimensions.
    :ivar format: :class:`TensorFormat` Tensor format.
    :ivar type: :class:`DataType` Type.
    :ivar scale: :class:`float` Scale for INT8 data type.
)trtdoc";
} // namespace PluginTensorDescDoc

namespace DynamicPluginTensorDescDoc
{
constexpr const char* descr = R"trtdoc(
    Summarizes tensors that a plugin might see for an input or output.

    :ivar desc: :class:`PluginTensorDesc` Information required to interpret a pointer to tensor data, except that desc.dims has -1 in place of any runtime dimension..
    :ivar min: :class:`Dims` 	Lower bounds on tensor's dimensions.
    :ivar max: :class:`Dims` 	Upper bounds on tensor's dimensions.
)trtdoc";
} // namespace DynamicPluginTensorDescDoc

namespace DimsExprsDoc
{
constexpr const char* descr = R"trtdoc(
    Analog of class `Dims` with expressions (`IDimensionExpr`) instead of constants for the dimensions.

    Behaves like a Python iterable and lists or tuples of `IDimensionExpr` can be used to construct it.
)trtdoc";
} // namespace DimsExprsDoc

namespace IDimensionExprDoc
{
constexpr const char* descr = R"trtdoc(
    An `IDimensionExpr` represents an integer expression constructed from constants, input dimensions, and binary operations.
    
    These expressions are can be used in overrides of `IPluginV2DynamicExt::get_output_dimensions()` to define output dimensions in terms of input dimensions.
)trtdoc";

constexpr const char* is_constant = R"trtdoc(
    Return true if expression is a build-time constant

)trtdoc";

constexpr const char* get_constant_value = R"trtdoc(
    If `is_constant()`, returns value of the constant. If not `is_constant()`, return int32 minimum.

)trtdoc";
} // namespace IDimensionExprDoc

namespace IExprBuilderDoc
{
constexpr const char* descr = R"trtdoc(
    Object for constructing `IDimensionExpr`.

    There is no public way to construct an `IExprBuilder`. It appears as an argument to method `IPluginV2DynamicExt::get_output_dimensions()`. Overrides of that method can use that `IExprBuilder` argument to construct expressions that define output dimensions in terms of input dimensions.

    Clients should assume that any values constructed by the `IExprBuilder` are destroyed after `IPluginV2DynamicExt::get_output_dimensions()` returns.
)trtdoc";

constexpr const char* constant = R"trtdoc(
    Return pointer to `IDimensionExpr` for given value.

)trtdoc";

constexpr const char* operation = R"trtdoc(
    Return pointer to `IDimensionExpr` that represents the given operation applied to first and second. Returns nullptr if op is not a valid `DimensionOperation`.

)trtdoc";
} // namespace IExprBuilderDoc

namespace DimensionOperationDoc
{
constexpr const char* descr = R"trtdoc(
    An operation on two `IDimensionExpr` s, which represent integer expressions used in dimension computations.

    For example, given two `IDimensionExpr` s `x` and `y` and an `IExprBuilder` `eb`, `eb.operation(DimensionOperation.SUM, x, y)` creates a representation of `x + y`.
)trtdoc";
} // namespace DimensionOperationDoc

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

    .. warning::
        This API only applies when called on a C++ plugin from a Python program.

    `serialized_plugin` will contain a Python bytes object containing the serialized representation of the plugin.

    :arg name: Name of the plugin.
    :arg serialized_plugin: A buffer containing a serialized plugin.

    :returns: A new :class:`IPluginV2`
)trtdoc";

constexpr const char* deserialize_plugin_python = R"trtdoc(
    Creates a plugin object from a serialized plugin.

    .. warning::
        This API only applies when implementing a Python-based plugin.

    `serialized_plugin` contains a serialized representation of the plugin.

    :arg name: Name of the plugin.
    :arg serialized_plugin: A string containing a serialized plugin.

    :returns: A new :class:`IPluginV2`
)trtdoc";
} // namespace IPluginCreatorDoc

namespace IPluginRegistryDoc
{
constexpr const char* descr = R"trtdoc(
    Registers plugin creators.

    :ivar plugin_creator_list: All the registered plugin creators.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    :ivar parent_search_enabled: bool variable indicating whether parent search is enabled. Default is True.
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

constexpr const char* load_library = R"trtdoc(
    Load and register a shared library of plugins.

    :arg: plugin_path: the plugin library path.

    :returns: The loaded plugin library handle. The call will fail and return None if any of the plugins are already registered.
)trtdoc";

constexpr const char* deserialize_library = R"trtdoc(
    Load and register a shared library of plugins from a memory buffer.

    :arg: serialized_plugin_library: a pointer to a plugin buffer to deserialize.

    :returns: The loaded plugin library handle. The call will fail and return None if any of the plugins are already registered.
)trtdoc";

constexpr const char* deregister_library = R"trtdoc(
    Deregister plugins associated with a library. Any resources acquired when the library was loaded will be released.

    :arg: handle: the plugin library handle to deregister.
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
