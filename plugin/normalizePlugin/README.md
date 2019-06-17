# normalizePlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `normalizePlugin`  is used for the L2 normalization layer, which is generally used in deep learning models such as ParseNet and SSD during TensorRT inference. This plugin is included in TensorRT and used in [sampleSSD](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#sample_ssd) to run SSD.

Specifically, given an array of values `x = [x_0, x_1, ..., x_n]` and a scale factor, the L2 norm `||x||_2 = sqrt(sum({x_0}^2, {x_1}^2, ..., {x_n}^2)` is calculated and each element in the array is divided by the L2 norm and multiplied by the scale factor.
  
### Structure

This plugin takes one input and generates one output. The input is the data from the last layer that is going to be normalized. It has a shape of `[N, C, H, W]`, where `N` is the batch size, `C` is the number of channels, `H` is the height, `W` is the width. The dimension of the output is exactly the same as the input.


## Parameters

This plugin consists of the plugin creator class `NormalizePluginCreator` and the plugin class `Normalize`. To create the plugin instance, the following parameters are used:

| Type       | Parameter                | Description
|------------|--------------------------|--------------------------------------------------------
|`bool`      |`acrossSpatial`           |When `acrossSpatial = true`, the normalization is conducted for each sample from the batch independently, meaning across spatial locations only. This is somewhat similar to instance normalization. When `acrossSpatial = false`, the normalization is conducted across all samples in the batch and spatial locations. This is somewhat similar to batch normalization.
|`bool`      |`channelShared`           |When `channelShared = true`, a single scale factor is used for all channels. When `channelShared = false`, several scale factors are provided and are used for each channel independently.
|`float`     |`eps`                     |A small number to prevent being divided by zero during normalization.
|`Weights *` |`weights`                 |A pointer to weights which contains information about scale factors for normalization. The definition of `Weights` can be found in the [NvInfer.h](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_8h_source.html) header. The number of values in weights is `1` if `channelShared = false`, otherwise the number of values in weights is the number of channels in the input tensor.
|`int`       |`nbWeights`               |The number of weights sent to the plugin. This value has to be `1`.


## Additional resources

The following resources provide a deeper understanding of the `normalizePlugin` plugin:

**Networks**
- [ParseNet Paper](https://arxiv.org/abs/1506.04579)    
- [SSD Paper](https://arxiv.org/abs/1512.02325)    

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.