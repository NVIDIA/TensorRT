# scatterElements

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The scatterElements plugin implements the scatter operation described in (https://github.com/rusty1s/pytorch_scatter), in compliance with the [ONNX specification for ScatterElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements)

Note: ScatterElements with reduce="none" is implemented in TRT core, not this plugin.

### Structure

This plugin has the 2 versions. The latest is plugin creator class `ScatterElementsPluginV3Creator` and the plugin class `ScatterElementsPluginV3` which extends `IPluginV3`. (name: `ScatterElements`, version: 2)
The legacy plugin that will be deprecated, is plugin creator class `ScatterElementsPluginV2Creator` and the plugin class `ScatterElementsPluginV2`, which extends `IPluginV2DynamicExt` (name: `ScatterElements`, version: 1).

The `ScatterElements` plugin consumes the following inputs:

1. `data` - T: Tensor of rank r >= 1.
2. `indices` - Tind: Tensor of int64 indices, of r >= 1 (same rank as input). All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.
3. `updates` - T: Tensor of rank r >=1 (same rank and shape as indices)

The `ScatterElements` plugin produces the following output:

1. `output` - T: Tensor, same shape as `data`.

## Parameters

The `ScatterElements` plugin has the following parameters:

| Type             | Parameter                       | Description
|------------------|---------------------------------|--------------------------------------------------------
|`int`             |`axis`                           | Which axis to scatter on. Default is 0. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
|`char`            |`reduction`                      | Type of reduction to apply: add, mul, max, min. ‘add’: reduction using the addition operation. ‘mul’: reduction using the multiplication operation.‘max’: reduction using the maximum operation.‘min’: reduction using the minimum operation.


The following resources provide a deeper understanding of the `scatterElements` plugin:

- [pytorch_scatter: original implementation and docs](https://github.com/rusty1s/pytorch_scatter)
- [ONNX specification for ScatterElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

## Changelog

- July 2024: Version 2 of the plugin migrated to `IPluginV3` interface design. The legacy plugin (version 1) using `IPluginV2DynamicExt` interface is deprecated.
- Oct 2023: This is the first release of this `README.md` file.

## Known issues

- Types T=BFLOAT16 and T=INT8 are currently not supported.
- ONNX spec allows Tind=int32 : only INT64 is supported by this plugin
