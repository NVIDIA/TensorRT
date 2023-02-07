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
Optionally can adds a bias vector before layer-normalization.


### Structure

The `skipLayerNormPlugin` takes two inputs; `input` and `skip`.

`input`
input is a tensor with shape `[S, B, E]` where `B` is the batch size and `E` is the hidden size.

`skip`
skip is a tensor with shape `[S, B, E]` where `B` is the batch size and `E` is the hidden size.
The purpose of this input is to introduce skip (aka. residual) connections to previously computed tensors.


The `skipLayerNormPlugin` generates the following output:

`output`
output is a tensor with shape `[S, B, E]` where `B` is the batch size.


## Parameters

`skipLayerNormPlugin` has plugin creator class `SkipLayerNormPluginDynamicCreator` and plugin class `CustomSkipLayerNormPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               |  Version   | Description
|----------|-----------------------------------------|------------|-------------------------------------------------------------------
|`int`     |`type_id`                                |  1, 2      |Integer encoding the DataType (0: FP32, 1: FP16, 2: INT8)
|`int`     |`ld`                                     |  1         |The leading dimension of the input tensor, corresponding to the hidden size, denoted by `E` above.
|`Weights` |`beta`                                   |  1, 2, 3   |The mean to normalize to. Shape: `[1, 1, E]`
|`Weights` |`gamma`                                  |  1, 2, 3   |The standard deviation to normalize to. Shape: `[1, 1, E]`
|`Weights` |`bias`                                   |  1, 2      |An optional bias vector to add before normalization. Shape: `[1, 1, E]`


## Additional resources

-   [LayerNorm](https://arxiv.org/abs/1607.06450)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

October  2020  
Add V2 plugin that supports variable sequence length.  
Add v3 plugin that supports int8 interleaved variable sequence length.

November 2019  
This is the first release of this `README.md` file.

## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
