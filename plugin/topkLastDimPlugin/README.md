# TopkLastDim

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `TopkLastDim` plugin computes the top-`k` largest or smallest elements of an input tensor along a specified axis, following the [ONNX specification for TopK](https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK) for value selection, ordering, and the meaning of the `axis` / `k` / `largest` parameters. The plugin emits `int32` indices rather than the `int64` indices required by the ONNX spec — see [Known issues](#known-issues).

The plugin is built around the AIR (Adaptive Iterative Radix) sort kernel ported from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/topkLastDim.cu), which operates on the last dimension of a 2D `[numRows, rowLength]` view. The plugin handles arbitrary input rank and arbitrary axis by:

- **Fast path** (`axis == last dim`): the kernel is invoked directly on the input with no copies.
- **General path** (any other axis): the input is transposed so the target axis is moved to the last position, the kernel runs on the resulting 2D view, and the output values and indices are transposed back so the top-`k` dimension sits at the original axis.

### Structure

The `TopkLastDim` plugin consumes the following input:

1. `input` - T: A tensor of arbitrary rank. The plugin computes top-`k` along the axis specified by the `axis` attribute. T can be `float32`, `float16`, `int32`, or `bfloat16`.

The `TopkLastDim` plugin produces the following outputs:

1. `values` - T: A tensor with the same shape as `input` except that the size along `axis` is replaced by `k`. Contains the top-`k` values along `axis`, in descending order when `is_largest == 1` and ascending order when `is_largest == 0`.
2. `indices` - `int32`: A tensor with the same shape as `values`. Contains the indices, along `axis` of `input`, of the corresponding entries in `values`. Note: ONNX TopK specifies `int64` indices; this plugin emits `int32`.

This plugin has the plugin creator class `TopkLastDimPluginCreator` and the plugin class `TopkLastDimPlugin` which extends `IPluginV3`.

## Parameters

The `TopkLastDim` plugin has the following parameters:

| Type    | Parameter     | Description
|---------|---------------|--------------------------------------------------------
| `int32` | `type_id`     | Data type of the input tensor (and of the `values` output). Allowed values follow `nvinfer1::DataType`: `0` (kFLOAT), `1` (kHALF), `3` (kINT32), `7` (kBF16). Required.
| `int32` | `k`           | Number of top elements to return along `axis`. Must be a positive integer not greater than the size of `input` along `axis`. Required.
| `int32` | `is_largest`  | If `1`, return the `k` largest elements (sorted descending). If `0`, return the `k` smallest elements (sorted ascending). Required.
| `int32` | `axis`        | Axis along which to compute top-`k`. Negative values count from the back (e.g., `-1` means the last dimension). Optional; default is `-1`.

## Additional resources

The following resources provide a deeper understanding of the `TopkLastDim` plugin:

- [ONNX specification for TopK](https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK)
- [AIR TopK kernel in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/topkLastDim.cu)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html).

## Changelog

May 2026: This is the first release of this `README.md` file.

## Known issues

- ONNX TopK specifies indices of type `int64`; this plugin emits `int32` indices to match the upstream TRT-LLM kernel's public entry point (which fixes `IdxT = int32_t`). Inputs along `axis` longer than `2^31 - 1` are therefore not addressable.
