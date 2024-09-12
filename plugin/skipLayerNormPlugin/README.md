# skipLayerNormPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

Adds a residual tensor, applies layer normalization, i.e., transforms the mean and standard deviation to beta and gamma respectively.
Optionally, adds a bias vector before layer-normalization.


### Structure

The `skipLayerNormPlugin` takes two inputs; `input` and `skip`.

`input`
For V1, V2, V5, V6, input is a tensor with shape `[S, B, E, 1, 1]` where `S` is the sequence length, `B` is the batch size, `E` is the hidden size, and the last two dimensions are of size 1.
For V3 and V4, input is a tensor with shape `[1, E, S', 1]` where `S'` is the accumulated sequence length, `E` is the hidden size, and the first and last dimensions are of size 1.

`skip`
skip has the same input dimensions as the input.
The purpose of this input is to introduce skip (aka. residual) connections to previously computed tensors.


The `skipLayerNormPlugin` generates the following output:

`output`
output is a tensor with the same shape as the input.


## Parameters

`skipLayerNormPlugin` has plugin creator class `SkipLayerNormPluginDynamicCreator` and plugin class `CustomSkipLayerNormPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               |  Version                | Description
|----------|-----------------------------------------|-------------------------|-------------------------------------------------------------------
|`int`     |`type_id`                                |  1, 2, 5, 6             |Integer encoding the DataType (0: FP32, 1: FP16, 2: INT8)
|`int`     |`ld`                                     |  1, 5                   |The leading dimension of the input tensor, corresponding to the hidden size, denoted by `E` above.
|`Weights` |`beta`                                   |  1, 2, 3, 4, 5, 6, 7, 8 |The mean to normalize to. Shape: `[1, 1, E]`
|`Weights` |`gamma`                                  |  1, 2, 3, 4, 5, 6, 7, 8 |The standard deviation to normalize to. Shape: `[1, 1, E]`
|`Weights` |`bias`                                   |  1, 2, 5, 6             |An optional bias vector to add before normalization. Shape: `[1, 1, E]`


## Additional resources

-   [LayerNorm](https://arxiv.org/abs/1607.06450)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

July 2024
Add v5, v6, v7 and v8 plugins that duplicate the behavior of v1, v3, v3 and v4 plugins respectively, but implement the `IPluginV3` interface instead of the deprecated `IPluginV2DynamicExt` interface.

February  2024
Add epsilon to avoid divide by zero.

October  2020
Add V2 plugin that supports variable sequence length.
Add v3 plugin that supports int8 interleaved variable sequence length.

November 2019
This is the first release of this `README.md` file.

## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
