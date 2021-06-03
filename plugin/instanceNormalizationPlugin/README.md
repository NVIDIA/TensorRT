# InstanceNormalizationPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `InstanceNormalizePlugin` is used for the InstanceNormalization layer, which is generally used in deep learning models that perform image generation. This plugin is based off the [ONNX opset 6 definition](https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization), and is used in any ONNX model that uses this operation.

Specifically, given an array of values `x = [x_0, x_1, ..., x_n]` , a scale factor, a bias factor, and an epislon,  the InstanceNormalization of x is  `scale * (x-mean) / sqrt(variance + epsilon) + bias` where the mean and variance are computed per instance per channel.
  
### Structure

This plugin takes one input and generates one output. The first input is the data from the last layer that is going to be normalized. It has a shape of `[N, C, H, W]`, where `N` is the batch size, `C` is the number of channels, `H` is the height, `W` is the width. 

The dimensions of the output are exactly the same as the input.

## Parameters

This plugin consists of the plugin creator class `InstanceNormalizationPluginCreator` and the plugin class `InstanceNormalizationPlugin`. To create the plugin instance, the following parameters are used:

| Type       | Parameter                | Description
|------------|--------------------------|--------------------------------------------------------
|`float`     |`epsilon`                 |A small number to prevent being divided by zero during normalization.
|`Weights *` |`scale`                   |A pointer to weights which contains information about scale factors for normalization. The definition of `Weights` can be found in the [NvInfer.h](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_8h_source.html) header.
|`Weights *` |`bias`                    |A pointer to weights which contains information about the bias values for normalization. The definition of `Weights` can be found in the [NvInfer.h](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_8h_source.html) header.
|`int`       |`relu`                    |A value used to enable leaky relu activation
|`float`     |`alpha`                   |A small negative slope for the leaky relu activation 


## Additional resources

The following resources provide a deeper understanding of the `InstanceNormalizationPlugin` plugin:

**Networks**
- [ONNX Operator Definition](https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization)    
- [Instance Normalization Paper](https://arxiv.org/abs/1607.08022)    

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

September 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
