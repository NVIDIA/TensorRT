
IPluginRegistry
***************

**class tensorrt.IPluginRegistry**

   Registers plugin creators.

   :Variables:
      *  **plugin_creator_list** – All the registered plugin creators.

      *  **error_recorder** – `IErrorRecorder
         <../Core/ErrorRecorder.rst#tensorrt.IErrorRecorder>`_
         Application-implemented error reporting interface for
         TensorRT objects.

      *  **parent_search_enabled** – bool variable indicating whether
         parent search is enabled. Default is True.

   **deregister_creator(self: tensorrt.tensorrt.IPluginRegistry,
   creator: tensorrt.tensorrt.IPluginCreator) -> bool**

      Deregister a previously registered plugin creator.

      Since there may be a desire to limit the number of plugins, this
      function provides a mechanism for removing plugin creators
      registered in TensorRT. The plugin creator that is specified by
      ``creator`` is removed from TensorRT and no longer tracked.

      :Parameters:
         **creator** – The IPluginCreator instance.

      :Returns:
         ``True`` if the plugin creator was deregistered, ``False`` if
         it was not found in the registry or otherwise could not be
         deregistered.

   **deregister_library(self: tensorrt.tensorrt.IPluginRegistry,
   handle: capsule) -> None**

      Deregister plugins associated with a library. Any resources
      acquired when the library was loaded will be released.

      :Arg:
         handle: the plugin library handle to deregister.

   **get_plugin_creator(self: tensorrt.tensorrt.IPluginRegistry, type:
   str, version: str, plugin_namespace: str = '') ->
   `tensorrt.tensorrt.IPluginCreator
   <IPluginCreator.rst#tensorrt.IPluginCreator>`_**

      Return plugin creator based on type and version

      :Parameters:
         *  **type** – The type of the plugin.

         *  **version** – The version of the plugin.

         *  **plugin_namespace** – The namespace of the plugin.

      :Returns:
         An `IPluginCreator
         <IPluginCreator.rst#tensorrt.IPluginCreator>`_ .

   **load_library(self: tensorrt.tensorrt.IPluginRegistry,
   plugin_path: str) -> capsule**

      Load and register a shared library of plugins.

      :Arg:
         plugin_path: the plugin library path.

      :Returns:
         The loaded plugin library handle. The call will fail and
         return None if any of the plugins are already registered.

   **register_creator(self: tensorrt.tensorrt.IPluginRegistry,
   creator: tensorrt.tensorrt.IPluginCreator, plugin_namespace: str =
   '') -> bool**

      Register a plugin creator.

      :Parameters:
         *  **creator** – The IPluginCreator instance.

         *  **plugin_namespace** – The namespace of the plugin
            creator.

      :Returns:
         False if one with the same type is already registered.

**tensorrt.get_plugin_registry() -> `tensorrt.tensorrt.IPluginRegistry
<#tensorrt.IPluginRegistry>`_**

   Return the plugin registry for standard runtime

**tensorrt.init_libnvinfer_plugins(logger: capsule, namespace: str) ->
bool**

   Initialize and register all the existing TensorRT plugins to the
   `IPluginRegistry <#tensorrt.IPluginRegistry>`_ with an optional
   namespace. The plugin library author should ensure that this
   function name is unique to the library. This function should be
   called once before accessing the Plugin Registry.

   :Parameters:
      *  **logger** – Logger to print plugin registration information.

      *  **namespace** – Namespace used to register all the plugins in
         this library.

**tensorrt.get_builder_plugin_registry(arg0:
nvinfer1::EngineCapability) -> `tensorrt.tensorrt.IPluginRegistry
<#tensorrt.IPluginRegistry>`_**

   Return the plugin registry used for building engines for the
   specified runtime
