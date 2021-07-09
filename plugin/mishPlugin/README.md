# mishPlugin

**Table Of Contents**
- [mishPlugin](#mishplugin)
  - [Description](#description)
    - [Structure](#structure)
  - [Parameters](#parameters)
  - [Additional resources](#additional-resources)
  - [License](#license)
  - [Changelog](#changelog)
  - [Known issues](#known-issues)


## Description

This plugin applies the Mish activation `x * tanh(ln(1+e^x))`.


### Structure

This plugin takes one input and generates one output. The input data is going to be activated with Mish. It has a shape of [N, C, H, W], where N is the batch size, C is the number of channels, H is the height, W is the width. The dimension of the output is exactly the same as the input.


## Parameters

`mishPlugin` has plugin factory class `MishPluginFactory` and plugin class `MishPlugin`.


## Additional resources

-   [Paper](https://arxiv.org/abs/1908.08681)
-   [Implementation](https://github.com/digantamisra98/Mish)
-   [CUDA Version](https://github.com/thomasbrandon/mish-cuda)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

May 2020
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
