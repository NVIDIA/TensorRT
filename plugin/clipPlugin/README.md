# Clip Plugin

**Table Of Contents**
- [Description](#description)
- [Structure](#structure)
    - [Parameters](#parameters)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

The Clip plugin restricts the values of a tensor to the range `[clipMin, clipMax]`. Values below this range are set to `clipMin`, values above this range are set to `clipMax`, and values within this range are unchanged.

The Clip plugin can be used to implement the ReLU6 activation function, which clips the input tensor values to the range `[0, 6]`.


## Structure

The Clip plugin takes one input. The input can have any shape.

The Clip plugin generates one output. The output will have the same shape as the input.

The Clip plugin supports the data types `float32`, `float16`, and `int32`. The data types `int8`, `uint8`, and `bool` are not supported.

This plugin works for network with graph node named `Clip_TRT`. This is also the plugin name that should be used when getting the `ClipPluginCreator` from the Plugin Registry.


### Parameters

The Clip plugin has plugin creator class `ClipPluginCreator` and plugin class `ClipPlugin`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`float32` |`clipMin`                 |The lower bound of the range.
|`float32` |`clipMax`                 |The upper bound of the range.


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

September 2022
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
