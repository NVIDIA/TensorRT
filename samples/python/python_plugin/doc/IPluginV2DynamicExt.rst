
IPluginV2DynamicExt
*******************

**class tensorrt.IPluginV2DynamicExt(*args, **kwargs)**

   Plugin class for user-implemented layers.

   Plugins are a mechanism for applications to implement custom
   layers.

   Similar to *IPluginV2Ext* (including capability to support
   different output data types), but with support for dynamic shapes.

   This class is made available for the purpose of implementing
   *IPluginV2DynamicExt* plugins with Python. Inherited Python->C++
   bindings from *IPluginV2* and *IPluginV2Ext* will continue to work
   on C++-based *IPluginV2DynamicExt* plugins.

   Note: Every attribute except *tensorrt_version* must be explicitly
      initialized on Python-based plugins. Except *plugin_namespace*,
      these attributes will be read-only when accessed through a
      C++-based plugin.

   :Variables:
      *  **num_outputs** – ``int`` The number of outputs from the
         plugin. This is used by the implementations of
         `INetworkDefinition
         <../Graph/Network.rst#tensorrt.INetworkDefinition>`_ and
         `Builder <../Core/Builder.rst#tensorrt.Builder>`_. In
         particular, it is called prior to any call to `initialize()
         <#tensorrt.IPluginV2DynamicExt.initialize>`_.

      *  **tensorrt_version** – ``int`` [READ ONLY] The API version
         with which this plugin was built.

      *  **plugin_type** – ``str`` The plugin type. Should match the
         plugin name returned by the corresponding plugin creator.

      *  **plugin_version** – ``str`` The plugin version. Should match
         the plugin version returned by the corresponding plugin
         creator.

      *  **plugin_namespace** – ``str`` The namespace that this plugin
         object belongs to. Ideally, all plugin objects from the same
         plugin library should have the same namespace.

      *  **serialization_size** – ``int`` [READ ONLY] The size of the
         serialization buffer required.

   Overloaded function.

   1. __init__(self: tensorrt.tensorrt.IPluginV2DynamicExt) -> None

   2. __init__(self: tensorrt.tensorrt.IPluginV2DynamicExt, arg0:
      tensorrt.tensorrt.IPluginV2DynamicExt) -> None

   **clone(self: tensorrt.tensorrt.IPluginV2DynamicExt) ->
   `tensorrt.tensorrt.IPluginV2DynamicExt
   <#tensorrt.IPluginV2DynamicExt>`_**

      Clone the plugin object. This copies over internal plugin
      parameters as well and returns a new plugin object with these
      parameters.

      If the source plugin is pre-configured with
      *configure_plugin()*, the returned object should also be
      pre-configured. Cloned plugin objects can share the same
      per-engine immutable resource (e.g. weights) with the source
      object to avoid duplication.

   **configure_plugin(self: tensorrt.tensorrt.IPluginV2DynamicExt,
   pos: List[tensorrt.tensorrt.DynamicPluginTensorDesc], in_out:
   List[tensorrt.tensorrt.DynamicPluginTensorDesc]) -> None**

      Configure the plugin.

      This function can be called multiple times in both the build and
      execution phases. The build phase happens before *initialize()*
      is called and only occurs during creation of an engine by
      *IBuilder*. The execution phase happens after *initialize()* is
      called and occurs during both creation of an engine by
      *IBuilder* and execution of an engine by *IExecutionContext*.

      Build phase: *configure_plugin()* is called when a plugin is
      being prepared for profiling but not for any specific input
      size. This provides an opportunity for the plugin to make
      algorithmic choices on the basis of input and output formats,
      along with the bound of possible dimensions. The min and max
      value of the *DynamicPluginTensorDesc* correspond to the *kMIN*
      and *kMAX* value of the current optimization profile that the
      plugin is being profiled for, with the *desc.dims* field
      corresponding to the dimensions of plugin specified at network
      creation. Wildcard dimensions will exist during this phase in
      the *desc.dims* field.

      Execution phase: *configure_plugin()* is called when a plugin is
      being prepared for executing the plugin for specific dimensions.
      This provides an opportunity for the plugin to change
      algorithmic choices based on the explicit input dimensions
      stored in *desc.dims* field.

      Warning: This *configure_plugin()* method is not available to be
         called from Python on C++-based plugins

      :Parameters:
         *  **in** – The input tensors attributes that are used for
            configuration.

         *  **out** – The output tensors attributes that are used for
            configuration.

   **destroy(self: tensorrt.tensorrt.IPluginV2DynamicExt) -> None**

      Destroy the plugin object. This will be called when the
      `INetworkDefinition
      <../Graph/Network.rst#tensorrt.INetworkDefinition>`_ , `Builder
      <../Core/Builder.rst#tensorrt.Builder>`_ or `ICudaEngine
      <../Core/Engine.rst#tensorrt.ICudaEngine>`_ is destroyed.

      Note: When implementing a Python-based plugin, implementing this
         method is optional. The default behavior is a *pass*.

   **enqueue(self: tensorrt.tensorrt.IPluginV2DynamicExt, input_desc:
   List[tensorrt.tensorrt.PluginTensorDesc], output_desc:
   List[tensorrt.tensorrt.PluginTensorDesc], inputs: List[int],
   outputs: List[int], workspace: int, stream: int) -> None**

      Execute the layer.

      *inputs* and *outputs* contains pointers to the corresponding
      input and output device buffers as their *intptr_t* casts.
      *stream* also represents an *intptr_t* cast of the CUDA stream
      in which enqueue should be executed.

      Warning: Since input, output, and workspace buffers are created and
         owned by TRT, care must be taken when writing to them from
         the Python side.

      Warning: In contrast to the C++ API for *enqueue()*, this method must
         not return an error code. The expected behavior is to throw
         an appropriate exception. if an error occurs.

      Warning: This *enqueue()* method is not available to be called from
         Python on C++-based plugins.

      :Parameters:
         *  **input_desc** – how to interpret the memory for the input
            tensors.

         *  **output_desc** – how to interpret the memory for the
            output tensors.

         *  **inputs** – The memory for the input tensors.

         *  **outputs** – The memory for the output tensors.

         *  **workspace** – Workspace for execution.

         *  **stream** – The stream in which to execute the kernels.

   **get_output_datatype(self: tensorrt.tensorrt.IPluginV2DynamicExt,
   index: int, input_types: List[tensorrt.tensorrt.DataType]) ->
   tensorrt.tensorrt.DataType**

      Return the *DataType* of the plugin output at the requested
      index. The default behavior should be to return the type of the
      first input, or *DataType::kFLOAT* if the layer has no inputs.
      The returned data type must have a format that is supported by
      the plugin.

      :Parameters:
         *  **index** – Index of the output for which the data type is
            requested.

         *  **input_types** – Data types of the inputs.

      :Returns:
         *DataType* of the plugin output at the requested *index*.

   **get_output_dimensions(self:
   tensorrt.tensorrt.IPluginV2DynamicExt, output_index: int, inputs:
   List[tensorrt.tensorrt.DimsExprs], expr_builder:
   tensorrt.tensorrt.IExprBuilder) -> `tensorrt.tensorrt.DimsExprs
   <#tensorrt.DimsExprs>`_**

      Get expressions for computing dimensions of an output tensor
      from dimensions of the input tensors.

      This function is called by the implementations of *IBuilder*
      during analysis of the network.

      Warning: This *get_output_dimensions()* method is not available to be
         called from Python on C++-based plugins

      :Parameters:
         *  **output_index** – The index of the output tensor

         *  **inputs** – Expressions for dimensions of the input
            tensors

         *  **expr_builder** – Object for generating new expressions

      :Returns:
         Expression for the output dimensions at the given
         *output_index*.

   **get_serialization_size(self:
   tensorrt.tensorrt.IPluginV2DynamicExt) -> int**

      Return the serialization size (in bytes) required by the plugin.

      Note: When implementing a Python-based plugin, implementing this
         method is optional. The default behavior is equivalent to
         *return len(serialize())*.

   **get_workspace_size(self: tensorrt.tensorrt.IPluginV2DynamicExt,
   in: List[tensorrt.tensorrt.PluginTensorDesc], out:
   List[tensorrt.tensorrt.PluginTensorDesc]) -> int**

      Return the workspace size (in bytes) required by the plugin.

      This function is called after the plugin is configured, and
      possibly during execution. The result should be a sufficient
      workspace size to deal with inputs and outputs of the given size
      or any smaller problem.

      Note: When implementing a Python-based plugin, implementing this
         method is optional. The default behavior is equivalent to
         *return 0*.

      Warning: This *get_workspace_size()* method is not available to be
         called from Python on C++-based plugins

      :Parameters:
         *  **input_desc** – How to interpret the memory for the input
            tensors.

         *  **output_desc** – How to interpret the memory for the
            output tensors.

      :Returns:
         The workspace size (in bytes).

   **initialize(self: tensorrt.tensorrt.IPluginV2DynamicExt) -> int**

      Initialize the plugin for execution. This is called when the
      engine is created.

      Note: When implementing a Python-based plugin, implementing this
         method is optional. The default behavior is equivalent to
         *pass*.

      Warning: In contrast to the C++ API for *initialize()*, this method
         must not return an error code. The expected behavior is to
         throw an appropriate exception if an error occurs.

      Warning: This *initialize()* method is not available to be called from
         Python on C++-based plugins.

   **serialize(self: tensorrt.tensorrt.IPluginV2DynamicExt) -> bytes**

      Serialize the plugin.

      Warning: This API only applies when implementing a Python-based
         plugin.

      :Returns:
         A bytes object containing the serialized representation of
         the plugin.

   **supports_format_combination(self:
   tensorrt.tensorrt.IPluginV2DynamicExt, pos: int, in_out:
   List[tensorrt.tensorrt.PluginTensorDesc], num_inputs: int) ->
   bool**

      Return true if plugin supports the format and datatype for the
      input/output indexed by pos.

      For this method, inputs are indexed from *[0, num_inputs-1]* and
      outputs are indexed from *[num_inputs, (num_inputs + num_outputs
      - 1)]*. *pos* is an index into *in_ou`t, where `0 <= pos <
      (num_inputs + num_outputs - 1)*.

      TensorRT invokes this method to query if the input/output tensor
      indexed by *pos* supports the format and datatype specified by
      *in_out[pos].format* and *in_out[pos].type*. The override shall
      return true if that format and datatype at *in_out[pos]* are
      supported by the plugin. It is undefined behavior to examine the
      format or datatype or any tensor that is indexed by a number
      greater than *pos*.

      Warning: This *supports_format_combination()* method is not available
         to be called from Python on C++-based plugins

      :Parameters:
         *  **pos** – The input or output tensor index being queried.

         *  **in_out** – The combined input and output tensor
            descriptions.

         *  **num_inputs** – The number of inputs.

      :Returns:
         boolean indicating whether the format combination is
         supported or not.

   **terminate(self: tensorrt.tensorrt.IPluginV2DynamicExt) -> None**

      Release resources acquired during plugin layer initialization.
      This is called when the engine is destroyed.

      Note: When implementing a Python-based plugin, implementing this
         method is optional. The default behavior is equivalent to
         *pass*.

``tensorrt.PluginTensorDesc``

   Fields that a plugin might see for an input or output.

   *scale* is only valid when the *type* is *DataType.INT8*. TensorRT
   will set the value to -1.0 if it is invalid.

   :Variables:
      *  **dims** – `Dims
         <../FoundationalTypes/Dims.rst#tensorrt.Dims>`_   Dimensions.

      *  **format** – `TensorFormat
         <../Graph/LayerBase.rst#tensorrt.TensorFormat>`_ Tensor
         format.

      *  **type** – `DataType
         <../FoundationalTypes/DataType.rst#tensorrt.DataType>`_ Type.

      *  **scale** – ``float`` Scale for INT8 data type.

**class tensorrt.DynamicPluginTensorDesc(self:
tensorrt.tensorrt.DynamicPluginTensorDesc) -> None**

   Summarizes tensors that a plugin might see for an input or output.

   :Variables:
      *  **desc** – `PluginTensorDesc <#tensorrt.PluginTensorDesc>`_
         Information required to interpret a pointer to tensor data,
         except that desc.dims has -1 in place of any runtime
         dimension..

      *  **min** – `Dims
         <../FoundationalTypes/Dims.rst#tensorrt.Dims>`_    Lower
         bounds on tensor’s dimensions.

      *  **max** – `Dims
         <../FoundationalTypes/Dims.rst#tensorrt.Dims>`_    Upper
         bounds on tensor’s dimensions.

**class tensorrt.IDimensionExpr**

   An *IDimensionExpr* represents an integer expression constructed
   from constants, input dimensions, and binary operations.

   These expressions are can be used in overrides of
   *IPluginV2DynamicExt::get_output_dimensions()* to define output
   dimensions in terms of input dimensions.

   **getConstantValue(self: tensorrt.tensorrt.IDimensionExpr) -> int**

      If *is_constant()*, returns value of the constant. If not
      *is_constant()*, return int32 minimum.

   **isConstant(self: tensorrt.tensorrt.IDimensionExpr) -> bool**

      Return true if expression is a build-time constant

**class tensorrt.DimsExprs(*args, **kwargs)**

   Analog of class *Dims* with expressions (*IDimensionExpr*) instead
   of constants for the dimensions.

   Behaves like a Python iterable and lists or tuples of
   *IDimensionExpr* can be used to construct it.

   Overloaded function.

   1. __init__(self: tensorrt.tensorrt.DimsExprs) -> None

   2. __init__(self: tensorrt.tensorrt.DimsExprs, arg0:
      List[tensorrt.tensorrt.IDimensionExpr]) -> None

**class tensorrt.IExprBuilder(self: tensorrt.tensorrt.IExprBuilder) ->
None**

   Object for constructing *IDimensionExpr*.

   There is no public way to construct an *IExprBuilder*. It appears
   as an argument to method
   *IPluginV2DynamicExt::get_output_dimensions()*. Overrides of that
   method can use that *IExprBuilder* argument to construct
   expressions that define output dimensions in terms of input
   dimensions.

   Clients should assume that any values constructed by the
   *IExprBuilder* are destroyed after
   *IPluginV2DynamicExt::get_output_dimensions()* returns.

   **constant(self: tensorrt.tensorrt.IExprBuilder, value: int) ->
   `tensorrt.tensorrt.IDimensionExpr <#tensorrt.IDimensionExpr>`_**

      Return pointer to *IDimensionExpr* for given value.

   **operation(self: tensorrt.tensorrt.IExprBuilder, op:
   nvinfer1::DimensionOperation, first:
   tensorrt.tensorrt.IDimensionExpr, second:
   tensorrt.tensorrt.IDimensionExpr) ->
   `tensorrt.tensorrt.IDimensionExpr <#tensorrt.IDimensionExpr>`_**

      Return pointer to *IDimensionExpr* that represents the given
      operation applied to first and second. Returns nullptr if op is
      not a valid *DimensionOperation*.

**class tensorrt.DimensionOperation(self:
tensorrt.tensorrt.DimensionOperation, value: int) -> None**

      An operation on two *IDimensionExpr* s, which represent integer
      expressions used in dimension computations.

      For example, given two *IDimensionExpr* s *x* and *y* and an
      *IExprBuilder* *eb*, *eb.operation(DimensionOperation.SUM, x,
      y)* creates a representation of *x + y*.

   Members:

      SUM

      PROD

      MAX

      MIN

      SUB

      EQUAL

      LESS

      FLOOR_DIV

      CEIL_DIV

   ``property name``
