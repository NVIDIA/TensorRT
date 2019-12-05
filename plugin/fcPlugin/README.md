# fcPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

Performs a matrix multiplication similar to the FullyConnected Layer in TensorRT, but without bias. The main difference is that the weights are not transposed.
Always dispatches to cuBLAS. At engine build time, the plugin runs a search over the parameters of the available algorithms to find the fastest one available.


### Structure

The `fcPlugin` takes one input; `input`.

`input`
input is a tensor with shape `[N, B, K, 1, 1]` where `B` is the batch size.


The `fcPlugin` generates the following output:

`output`
output is a tensor with shape `[N, B, out_dims, 1, 1]` where `B` is the batch size, and `out_dims` is specified as a plugin parameter.

The trailing singleton dimensions in the input and output are added for compatibility with the default TRT FC layer.

## Parameters

`fcPlugin` has plugin creator class `FCPluginDynamicCreator` and plugin class `CustomFCPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               | Description
|----------|-----------------------------------------|-------------------------------------------------------------------
|`int`     |`out_dims`                               |Integer specifying the length of the third dimension of the output.
|`int`     |`type_id`                                |Integer encoding the DataType (0: FP32, 1: FP16)
|`Weights` |`W`                                      |The weights to multiply with. Shape: `[K, out_dims]`


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

November 2019
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
