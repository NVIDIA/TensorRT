# TensorRT Safety Runtime Command-Line Wrapper: trtexec_safe

**Table Of Contents**
- [Description](#description)
- [Building `trtexec_safe`](#building-trtexec_safe)
- [Using `trtexec_safe`](#using-trtexec_safe)
- [Tool command line arguments](#tool-command-line-arguments)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

`trtexec_safe` is a performance testing tool for TensorRT safety runtime. `trtexec_safe` can load a serialized engine file and test the performance of running inference with TensorRT safety runtime. `trtexec_safe` provides many options, like iterations, average runs, streams, for performance timing.

**Note:** `trtexec_safe` only provides the functionality of doing inference with a pre-built engine file.  The engine should be built with `EngineCapability::kSAFETY`. `trtexec_safe` is not safety-certified, therefore, it could produce AUTOSAR violations in the source code.

## Building `trtexec_safe`

Compile the sample by following the build instructions in the [TensorRT README](https://github.com/NVIDIA/TensorRT/). This will build the `trtexec_safe` binary.

The options that control compiling against safety runtime or standard runtime are summarized in the following table.

| Compile Option                  | Default |Description|
| ------------------------------- | ------- |---------- |
|TRT_SAFETY_INFERENCE_ONLY        | OFF     |When enabled, build the infer part only, skip compiling the builder part.|

## Using `trtexec_safe`

1. `trtexec_safe` requires a prebuilt safe TensorRT engine as input. Use `trtexec` to build a safe TensorRT engine.
    ```
    trtexec --safe --saveEngine=/path/to/safe/engine --model=/path/to/model
    ```
  
    where `--safe` indicates `trtexec` to build a safety certified TensorRT engine and `--saveEngine` specifies the path to save the output engine. For other details of `trtexec`, see `<TensorRT root directory>/samples/trtexec/README.md`.


2. Use `trtexec_safe` to load the engine and run performance testing.

    ```
    trtexec_safe --loadEngine=/path/to/safe/engine
    ```
    `trtexec_safe` will do inference for several iterations and give a final median execution time on both the CPU and GPU. For other options, refer to the [Tool command line arguments](#tool-command-line-arguments) section.

## Tool command line arguments

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

```
trtexec_safe [-h or --help]
```

## Additional resources

The following resources provide more details about `trtexec_safe`:

**Documentation**
- [TensorRT trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)
- [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/tensorrt/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html)
documentation.

# Changelog

June 2020
This is the first release of this `README.md` file.

September 2020
This sample was updated to fit TensorRt API changes:
IRuntime::validateSerializedEngine is removed.

March 2022
This sample was updated for DriveOS 6.0 and later releases.

Dec. 2025
This sample was updated to use the CMake-based build system.

# Known issues

There are no known issues in this sample.

