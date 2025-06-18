# GroupNormalizationPlugin [DEPRECATED]

**This plugin is deprecated since TensorRT 10.12 and will be removed in a future release. No alternatives are planned to be provided.**

**Table Of Contents**
- [Description](#description)
    * [Non-support for Blackwell and later platforms](#non-support-for-blackwell-and-later-platforms)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional Resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known Issues](#known-issues)

## Description

The `GroupNormalizationPlugin` implements Group Normalization, which divides channels into groups and computes normalization statistics within each group. This is particularly useful for vision models where batch sizes may be small.

### Non-support for Blackwell and later platforms

As of TensorRT 10.7, usage of this plugin is not supported on Blackwell or later platforms.
This plugin can be replaced by TensorRT's native `INormalizationLayer`([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_normalization_layer.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Normalization.html)).


**Note:** This plugin remains supported on pre-Blackwell platforms.

### Structure

The plugin takes three inputs:
1. Input tensor with shape `[N, C, H, W]` (batch, channels, height, width), where `C` must be divisible by `num_groups`. (See [Parameters](#parameters) for more details on `num_groups`)
2. Scale parameters (per-channel, shape `[C]`)
3. Bias parameters (per-channel, shape `[C]`)

It produces one output with the same dimensions as the input. The normalization is computed as:
```
group_mean = mean(input, group)
group_var = variance(input, group)
output = gamma (input - group_mean) / sqrt(group_var + epsilon) + beta
```

Key differences from Instance Normalization:
- Normalizes across channel groups rather than individual channels
- More stable for small batch sizes
- Groups channels to capture cross-channel dependencies

## Parameters

| Parameter    | Type    | Description |
|--------------|---------|-------------|
| `epsilon`    | float   | Small value added to variance for numerical stability (default: 1e-5) |
| `num_groups` | int32   | Number of groups to split channels into; must evenly divide C |

## Additional Resources

- **Original Paper**: [Group Normalization](https://arxiv.org/abs/1803.08494)
- **ONNX Operator**: [GroupNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GroupNormalization)
- **TensorRT Documentation**: [INormalizationLayer](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_normalization_layer.html)
- [Master README](../README.md) - Back to main documentation

## License
For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html).

## Changelog
- **May 2025**: Add deprecation note.

- **Feb 2025**: Initial release of this README, Deprecation and non-support notice added.

## Known Issues
- Limited to FP32 precision (native implementation supports mixed precision)
- No NHWC layout support (native implementation supports multiple layouts)
- Batch size must be known during network creation
