
IPluginCreator
**************

``tensorrt.PluginFieldType``

   The possible field types for custom layer.

   Members:

      FLOAT16

      FLOAT32

      FLOAT64

      INT8

      INT16

      INT32

      CHAR

      DIMS

      UNKNOWN

      BF16

      INT64

      FP8

**class tensorrt.PluginField(*args, **kwargs)**

   Contains plugin attribute field names and associated data. This
   information can be parsed to decode necessary plugin metadata

   :Variables:
      *  **name** – ``str`` Plugin field attribute name.

      *  **data** – ``buffer`` Plugin field attribute data.

      *  **type** – `PluginFieldType <#tensorrt.PluginFieldType>`_
         Plugin field attribute type.

      *  **size** – ``int`` Number of data entries in the Plugin
         attribute.

   Overloaded function.

   1. __init__(self: tensorrt.tensorrt.PluginField, name:
      tensorrt.tensorrt.FallbackString = ‘’) -> None

   2. __init__(self: tensorrt.tensorrt.PluginField, name:
      tensorrt.tensorrt.FallbackString, data: buffer, type:
      tensorrt.tensorrt.PluginFieldType = <PluginFieldType.UNKNOWN:
      8>) -> None

**class tensorrt.PluginFieldCollection(*args, **kwargs)**

   Overloaded function.

   1. __init__(self: tensorrt.tensorrt.PluginFieldCollection) -> None

   2. __init__(self: tensorrt.tensorrt.PluginFieldCollection, arg0:
      tensorrt.tensorrt.PluginFieldCollection) -> None

   Copy constructor

   1. __init__(self: tensorrt.tensorrt.PluginFieldCollection, arg0:
      Iterable) -> None

   **append(self: tensorrt.tensorrt.PluginFieldCollection, x:
   nvinfer1::PluginField) -> None**

      Add an item to the end of the list

   **clear(self: tensorrt.tensorrt.PluginFieldCollection) -> None**

      Clear the contents

   **extend(*args, **kwargs)**

      Overloaded function.

      1. extend(self: tensorrt.tensorrt.PluginFieldCollection, L:
         tensorrt.tensorrt.PluginFieldCollection) -> None

      Extend the list by appending all the items in the given list

      1. extend(self: tensorrt.tensorrt.PluginFieldCollection, L:
         Iterable) -> None

      Extend the list by appending all the items in the given list

   **insert(self: tensorrt.tensorrt.PluginFieldCollection, i: int, x:
   nvinfer1::PluginField) -> None**

      Insert an item at a given position.

   **pop(*args, **kwargs)**

      Overloaded function.

      1. pop(self: tensorrt.tensorrt.PluginFieldCollection) ->
         nvinfer1::PluginField

      Remove and return the last item

      1. pop(self: tensorrt.tensorrt.PluginFieldCollection, i: int) ->
         nvinfer1::PluginField

      Remove and return the item at index ``i``

**class tensorrt.IPluginCreator(self:
tensorrt.tensorrt.IPluginCreator) -> None**

   Plugin creator class for user implemented layers

   :Variables:
      *  **tensorrt_version** – ``int``  Number of `PluginField
         <#tensorrt.PluginField>`_ entries.

      *  **name** – ``str`` Plugin name.

      *  **plugin_version** – ``str`` Plugin version.

      *  **field_names** – ``list`` List of fields that needs to be
         passed to `create_plugin()
         <#tensorrt.IPluginCreator.create_plugin>`_ .

      *  **plugin_namespace** – ``str`` The namespace of the plugin
         creator based on the plugin library it belongs to. This can
         be set while registering the plugin creator.

   **create_plugin(self: tensorrt.tensorrt.IPluginCreator, name: str,
   field_collection: tensorrt.tensorrt.PluginFieldCollection_) ->
   tensorrt.tensorrt.IPluginV2**

      Creates a new plugin.

      :Parameters:
         *  **name** – The name of the plugin.

         *  **field_collection** – The `PluginFieldCollection
            <#tensorrt.PluginFieldCollection>`_ for this plugin.

      :Returns:
         ``IPluginV2`` or ``None`` on failure.

   **deserialize_plugin(*args, **kwargs)**

      Overloaded function.

      1. deserialize_plugin(self: tensorrt.tensorrt.IPluginCreator,
         name: str, serialized_plugin: buffer) ->
         tensorrt.tensorrt.IPluginV2

            Creates a plugin object from a serialized plugin.

            Warning: This API only applies when called on a C++ plugin from
               a Python program.

            *serialized_plugin* will contain a Python bytes object
            containing the serialized representation of the plugin.

            :arg name:
               Name of the plugin.

            :arg serialized_plugin:
               A buffer containing a serialized plugin.

            :returns:
               A new ``IPluginV2``

      2. deserialize_plugin(self:
         tensorrt.tensorrt.IPluginV2DynamicExt, name: str,
         serialized_plugin: bytes) ->
         tensorrt.tensorrt.IPluginV2DynamicExt

            Creates a plugin object from a serialized plugin.

            Warning: This API only applies when implementing a Python-based
               plugin.

            *serialized_plugin* contains a serialized representation
            of the plugin.

            :arg name:
               Name of the plugin.

            :arg serialized_plugin:
               A string containing a serialized plugin.

            :returns:
               A new ``IPluginV2``
