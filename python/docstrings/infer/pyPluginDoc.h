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

namespace IPluginV3Doc
{
constexpr const char* ipluginv3_descr = R"trtdoc(
    Plugin class for the V3 generation of user-implemented layers.

    IPluginV3 acts as a wrapper around the plugin capability interfaces that define the actual behavior of the plugin.

    This class is made available for the purpose of implementing `IPluginV3` plugins with Python.

    .. note::
        Every attribute must be explicitly initialized on Python-based plugins.
        These attributes will be read-only when accessed through a C++-based plugin.

)trtdoc";

constexpr const char* iplugincapability_descr = R"trtdoc(
    Base class for plugin capability interfaces

    IPluginCapability represents a split in TensorRT V3 plugins to sub-objects that expose different types of capabilites a plugin may have,
    as opposed to a single interface which defines all capabilities and behaviors of a plugin.
)trtdoc";

constexpr const char* ipluginv3onecore_descr = R"trtdoc(
    A plugin capability interface that enables the core capability (PluginCapabilityType.CORE).

    .. note::
        Every attribute must be explicitly initialized on Python-based plugins.
        These attributes will be read-only when accessed through a C++-based plugin.

    :ivar plugin_name: :class:`str` The plugin name. Should match the plugin name returned by the corresponding plugin creator.
    :ivar plugin_version: :class:`str` The plugin version. Should match the plugin version returned by the corresponding plugin creator.
    :ivar plugin_namespace: :class:`str` The namespace that this plugin object belongs to. Ideally, all plugin objects from the same plugin library should have the same namespace.
)trtdoc";

constexpr const char* ipluginv3onebuild_descr = R"trtdoc(
    A plugin capability interface that enables the build capability (PluginCapabilityType.BUILD).

    Exposes methods that allow the expression of the build time properties and behavior of a plugin.

    .. note::
        Every attribute must be explicitly initialized on Python-based plugins.
        These attributes will be read-only when accessed through a C++-based plugin.

    :ivar num_outputs: :class:`int` The number of outputs from the plugin. This is used by the implementations of :class:`INetworkDefinition` and :class:`Builder`.
    :ivar format_combination_limit: :class:`int` The maximum number of format combinations that the plugin supports.
    :ivar metadata_string: :class:`str` The metadata string for the plugin.
    :ivar timing_cache_id: :class:`str` The timing cache ID for the plugin.

)trtdoc";

constexpr const char* ipluginv3onebuildv2_descr = R"trtdoc(
    A plugin capability interface that extends IPluginV3OneBuild by providing I/O aliasing functionality.
)trtdoc";

constexpr const char* ipluginv3oneruntime_descr = R"trtdoc(
    A plugin capability interface that enables the runtime capability (PluginCapabilityType.RUNTIME).

    Exposes methods that allow the expression of the runtime properties and behavior of a plugin.
)trtdoc";

constexpr const char* get_output_shapes = R"trtdoc(

    Get expressions for computing shapes of an output tensor from shapes of the input tensors.

    This function is called by the implementations of `IBuilder` during analysis of the network.

    .. warning::
        This get_output_shapes() method is not available to be called from Python on C++-based plugins

    :arg inputs:	Expressions for shapes of the input tensors
    :arg shape_inputs:	Expressions for shapes of the shape inputs
    :arg expr_builder:	Object for generating new expressions

    :returns: Expressions for the output shapes.
)trtdoc";

constexpr const char* get_output_data_types = R"trtdoc(

    Return `DataType` s of the plugin outputs.

    Provide `DataType.FLOAT` s if the layer has no inputs. The data type for any size tensor outputs must be
    `DataType.INT32`. The returned data types must each have a format that is supported by the plugin.

    :arg input_types: Data types of the inputs.

    :returns: `DataType` of the plugin output at the requested `index`.
)trtdoc";

constexpr const char* configure_plugin = R"trtdoc(
    Configure the plugin.

    This function can be called multiple times in the build phase during creation of an engine by IBuilder.

    Build phase: `configure_plugin()` is called when a plugin is being prepared for profiling but not for any specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of input and output formats, along with the bound of possible dimensions. The min, opt and max value of the
    `DynamicPluginTensorDesc` correspond to the `MIN`, `OPT` and `MAX` value of the current profile that the plugin is
    being profiled for, with the desc.dims field corresponding to the dimensions of plugin specified at network
    creation. Wildcard dimensions may exist during this phase in the desc.dims field.

    .. warning::
        In contrast to the C++ API for `configurePlugin()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
        if an error occurs.

    .. warning::
        This `configure_plugin()` method is not available to be called from Python on C++-based plugins

    :arg in: The input tensors attributes that are used for configuration.
    :arg out: The output tensors attributes that are used for configuration.
)trtdoc";

constexpr const char* on_shape_change = R"trtdoc(
    Called when a plugin is being prepared for execution for specific dimensions. This could happen multiple times in the execution phase, both during creation of an engine by IBuilder and execution of an
    engine by IExecutionContext.

     * IBuilder will call this function once per profile, with `in` resolved to the values specified by the kOPT field of the current profile.
     * IExecutionContext will call this during the next subsequent instance of enqueue_v2() or execute_v3() if: (1) The optimization profile is changed (2). An input binding is changed.

    .. warning::
        In contrast to the C++ API for `onShapeChange()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
        if an error occurs.

    .. warning::
        This `on_shape_change()` method is not available to be called from Python on C++-based plugins

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
    Perform any cleanup or resource release(s) needed before plugin object is destroyed. This will be called when the :class:`INetworkDefinition` , :class:`Builder` or :class:`ICudaEngine` is destroyed.

    .. note::
        There is no direct equivalent to this method in the C++ API.

    .. note::
        Implementing this method is optional. The default behavior is a `pass`.

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

constexpr const char* get_capability_interface = R"trtdoc(
    Return a plugin object implementing the specified PluginCapabilityType.

    .. note::
        IPluginV3 objects added for the build phase (through add_plugin_v3()) must return valid objects for PluginCapabilityType.CORE, PluginCapabilityType.BUILD and PluginCapabilityType.RUNTIME.

    .. note::
        IPluginV3 objects added for the runtime phase must return valid objects for PluginCapabilityType.CORE and PluginCapabilityType.RUNTIME.
)trtdoc";

constexpr const char* clone = R"trtdoc(
    Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.

    If the source plugin is pre-configured with `configure_plugin()`, the returned object should also be pre-configured.
    Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object to avoid duplication.
)trtdoc";

constexpr const char* get_fields_to_serialize = R"trtdoc(
    Return the plugin fields which should be serialized.

    .. note::
        The set of plugin fields returned does not necessarily need to match that advertised through get_field_names() of the corresponding plugin creator.

    .. warning::
        This `get_fields_to_serialize()` method is not available to be called from Python on C++-based plugins.

)trtdoc";

constexpr const char* set_tactic = R"trtdoc(
    Set the tactic to be used in the subsequent call to enqueue().

    If no custom tactics were advertised, this will have a value of 0, which is designated as the default tactic.

    .. warning::
        In contrast to the C++ API for `setTactic()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
        if an error occurs.

    .. warning::
        This `set_tactic()` method is not available to be called from Python on C++-based plugins.

)trtdoc";

constexpr const char* get_valid_tactics = R"trtdoc(
    Return any custom tactics that the plugin intends to use.

    .. note::
        The provided tactic values must be unique and positive

    .. warning::
        This `get_valid_tactics()` method is not available to be called from Python on C++-based plugins.

)trtdoc";

constexpr const char* get_aliased_input = R"trtdoc(
    Communicates to TensorRT that the output at the specified output index is aliased to the input at the returned index

    Enables read-modify-write behavior in plugins. TensorRT may insert copies to facilitate this capability.

    .. note::
        A given plugin input can only be aliased to a single plugin output.

    .. note::
        This API will only be called and have an effect when PreviewFeature.ALIASED_PLUGIN_IO_10_03 is turned on.

    .. warning::
        If an input is not shallow copyable, a copy inserted by TensorRT may not work as intended. Therefore, using this feature with tensors requiring deep copies is not supported.

    .. warning::
        If a given tensor is requested to be aliased by two different plugins, this may result in divergent copies of the tensor after writes from each plugin. e.g. In the below example, t1 and t2 could be divergent.

           +-----+            +--------+
        +->|Copy +--> t* ---->|Plugin0 +--> t1
        |  +-----+            +--------+
        t
        |  +-----+            +--------+
        +->|Copy +--> t** --->|Plugin1 +--> t2
           +-----+            +--------+

    :returns: An integer denoting the index of the input which is aliased to the output at output_index. Returning -1 indicates that the output is not aliased to any input. Otherwise, the valid range for return value is [0, nbInputs - 1].

)trtdoc";

constexpr const char* attach_to_context = R"trtdoc(
    Clone the plugin, attach the cloned plugin object to a execution context and grant the cloned plugin access to some context resources.

    This function is called automatically for each plugin when a new execution context is created.

    The plugin may use resources provided by the resource_context until the plugin is deleted by TensorRT.

    :arg resource_context: A resource context that exposes methods to get access to execution context specific resources. A different resource context is guaranteed for each different execution context to which the plugin is attached.

    .. note::
        This method should clone the entire IPluginV3 object, not just the runtime interface

)trtdoc";

} // namespace IPluginV3Doc

namespace PluginFieldTypeDoc
{
constexpr const char* descr = R"trtdoc(
    The possible field types for custom layer.
)trtdoc";
} // namespace PluginFieldTypeDoc

namespace IPluginResourceDoc
{
constexpr const char* descr = R"trtdoc(
    Interface for plugins to define custom resources that could be shared through the plugin registry
)trtdoc";

constexpr const char* release = R"trtdoc(
    This will only be called for IPluginResource objects that were produced from IPluginResource::clone().

    The IPluginResource object on which release() is called must still be in a clone-able state
    after release() returns.

)trtdoc";

constexpr const char* clone = R"trtdoc(
    Resource initialization (if any) may be skipped for non-cloned objects since only clones will be
    registered by TensorRT.

)trtdoc";

} // namespace IPluginResourceDoc

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
    Return true if expression is a build-time constant.
)trtdoc";

constexpr const char* get_constant_value = R"trtdoc(
    Get the value of the constant.

    If is_constant(), returns value of the constant.
    Else, return int64 minimum.
)trtdoc";

constexpr const char* is_size_tensor = R"trtdoc(
    Return true if this denotes the value of a size tensor.
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
    Return a IDimensionExpr for the given value.
)trtdoc";

constexpr const char* operation = R"trtdoc(
    Return a IDimensionExpr that represents the given operation applied to first and second.
    Returns None if op is not a valid DimensionOperation.
)trtdoc";

constexpr const char* declare_size_tensor = R"trtdoc(
    Declare a size tensor at the given output index, with the specified auto-tuning formula and upper bound.

    A size tensor allows a plugin to have output dimensions that cannot be computed solely from input dimensions.
    For example, suppose a plugin implements the equivalent of INonZeroLayer for 2D input. The plugin can
    have one output for the indices of non-zero elements, and a second output containing the number of non-zero
    elements. Suppose the input has size [M,N] and has K non-zero elements. The plugin can write K to the second
    output. When telling TensorRT that the first output has shape [2,K], plugin uses IExprBuilder.constant() and
    IExprBuilder.declare_size_tensor(1,...) to create the IDimensionExpr that respectively denote 2 and K.

    TensorRT also needs to know the value of K to use for auto-tuning and an upper bound on K so that it can
    allocate memory for the output tensor. In the example, suppose typically half of the plugin's input elements
    are non-zero, and all the elements might be nonzero. then using M*N/2 might be a good expression for the opt
    parameter, and M*N for the upper bound. IDimensionsExpr for these expressions can be constructed from
    IDimensionsExpr for the input dimensions.
)trtdoc";
} // namespace IExprBuilderDoc

namespace DimensionOperationDoc
{
constexpr const char* descr = R"trtdoc(
    An operation on two IDimensionExprs, which represent integer expressions used in dimension computations.

    For example, given two IDimensionExprs x and y and an IExprBuilder eb, eb.operation(DimensionOperation.SUM, x, y) creates a representation of x + y.
)trtdoc";
} // namespace DimensionOperationDoc

namespace PluginCapabilityTypeDoc
{
constexpr const char* descr = R"trtdoc(
    Enumerates the different capability types a IPluginV3 object may have.
)trtdoc";
} // namespace PluginCapabilityTypeDoc

namespace TensorRTPhaseDoc
{
constexpr const char* descr = R"trtdoc(
    Indicates a phase of operation of TensorRT
)trtdoc";
} // namespace TensorRTPhaseDoc

namespace IPluginResourceContextDoc
{
constexpr const char* descr = R"trtdoc(
    Interface for plugins to access per context resources provided by TensorRT

    There is no public way to construct an IPluginResourceContext. It appears as an argument to trt.IPluginV3OneRuntime.attach_to_context().
)trtdoc";
} // namespace IPluginResourceContextDoc

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

    The collection behaves like a Python iterable.
)trtdoc";
} // namespace PluginFieldCollectionDoc

namespace IPluginCreatorInterfaceDoc
{
constexpr const char* descr = R"trtdoc(
    Base class for for plugin sub-interfaces.
)trtdoc";
} // namespace IPluginCreatorInterfaceDoc

namespace IPluginCreatorDoc
{
constexpr const char* descr = R"trtdoc(
    Plugin creator class for user implemented layers

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

namespace IPluginCreatorV3OneDoc
{
constexpr const char* descr = R"trtdoc(
    Plugin creator class for user implemented layers

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
} // namespace IPluginCreatorV3OneDoc

namespace IPluginRegistryDoc
{
constexpr const char* descr = R"trtdoc(
    Registers plugin creators.

    :ivar plugin_creator_list: [DEPRECATED] Deprecated in TensorRT 10.0. List of IPluginV2-descendent plugin creators in current registry.
    :ivar all_creators: List of all registered plugin creators of current registry.
    :ivar all_creators_recursive: List of all registered plugin creators of current registry and its parents (if :attr:`parent_search_enabled` is True).
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    :ivar parent_search_enabled: bool variable indicating whether parent search is enabled. Default is True.
)trtdoc";

constexpr const char* register_creator_iplugincreator = R"trtdoc(
    Register a plugin creator implementing IPluginCreator.

    :arg creator: The IPluginCreator instance.
    :arg plugin_namespace: The namespace of the plugin creator.

    :returns: False if any plugin creator with the same name, version and namespace is already registered.
)trtdoc";

constexpr const char* register_creator = R"trtdoc(
    Register a plugin creator.

    :arg creator: The plugin creator instance.
    :arg plugin_namespace: The namespace of the plugin creator.

    :returns: False if any plugin creator with the same name, version and namespace is already registered..
)trtdoc";

constexpr const char* deregister_creator_iplugincreator = R"trtdoc(
    Deregister a previously registered plugin creator inheriting from IPluginCreator.

    Since there may be a desire to limit the number of plugins,
    this function provides a mechanism for removing plugin creators registered in TensorRT.
    The plugin creator that is specified by ``creator`` is removed from TensorRT and no longer tracked.

    :arg creator: The IPluginCreator instance.

    :returns: ``True`` if the plugin creator was deregistered, ``False`` if it was not found in the registry
            or otherwise could not be deregistered.
)trtdoc";

constexpr const char* deregister_creator = R"trtdoc(
    Deregister a previously registered plugin creator.

    Since there may be a desire to limit the number of plugins,
    this function provides a mechanism for removing plugin creators registered in TensorRT.
    The plugin creator that is specified by ``creator`` is removed from TensorRT and no longer tracked.

    :arg creator: The plugin creator instance.

    :returns: ``True`` if the plugin creator was deregistered, ``False`` if it was not found in the registry
            or otherwise could not be deregistered.
)trtdoc";

constexpr const char* get_plugin_creator = R"trtdoc(
    Return plugin creator based on type, version and namespace

    .. warning::
        Returns None if a plugin creator with matching name, version, and namespace is found, but is not a
        descendent of IPluginCreator

    :arg type: The type of the plugin.
    :arg version: The version of the plugin.
    :arg plugin_namespace: The namespace of the plugin.

    :returns: An :class:`IPluginCreator` .
)trtdoc";

constexpr const char* get_creator = R"trtdoc(
    Return plugin creator based on type, version and namespace

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

constexpr const char* acquire_plugin_resource = R"trtdoc(
    Get a handle to a plugin resource registered against the provided key.

    :arg: key: Key for identifying the resource.
    :arg: resource: A plugin resource object. The object will only need to be valid until this method returns, as only a clone of this object will be registered by TRT. Cannot be null.
)trtdoc";

constexpr const char* release_plugin_resource = R"trtdoc(
    Decrement reference count for the resource with this key. If reference count goes to zero after decrement, release() will be invoked on the resource,
    and the key will be deregistered.

    :arg: key: Key that was used to register the resource.
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

namespace PluginCreatorVersionDoc
{
constexpr char const* descr = R"trtdoc(
    Enum to identify version of the plugin creator.
)trtdoc";
} // namespace PluginCreatorVersionDoc

} // namespace tensorrt
