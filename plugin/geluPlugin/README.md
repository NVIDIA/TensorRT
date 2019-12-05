# geluPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

This plugin applies the Gelu activation `x * Phi(x)`, where Phi is the Gaussian cdf, approximated by: `0.5 * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x^3)))`.
Optionally adds a bias vector before the activation.


### Structure

The `geluPlugin` takes one input; `input`.

`input`
input is a tensor with shape `[S, B, E]` where `B` is the batch size.


The `geluPlugin` generates the following output:

`output`
output is a tensor with shape `[S, B, E]` where `B` is the batch size.


## Parameters

`geluPlugin` has plugin creator class `GeluPluginDynamicCreator` and plugin class `CustomGeluPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               | Description
|----------|-----------------------------------------|-------------------------------------------------------------------
|`int`     |`type_id`                                |Integer encoding the DataType (0: FP32, 1: FP16)
|`Weights` |`bias`                                   |Optional bias parameter. Shape `[1, 1, E]`


## Additional resources

-   [GELU](https://arxiv.org/abs/1606.08415)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

November 2019
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
