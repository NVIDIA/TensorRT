# Using The CuDLA API To Run A TensorRT Engine


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
   * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
   * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleCudla, uses an API to construct a network of a single ElementWise layer and builds the engine. The engine runs in DLA standalone mode using cuDLA runtime. In order to do that, the sample uses cuDLA APIs to do engine conversion and cuDLA runtime preparation, as well as inference.

## How does this sample work?

After the construction of a network, the module with cuDLA is loaded from the network data. The input and output tensors are then allocated and registered with cuDLA. When the input tensors are copied from CPU to GPU, the cuDLA task can be submitted and executed. Then we wait for stream operations to finish and bring output buffer to CPU to be verified for correctness.

Specifically:
-   The single-layered network is built by TensorRT.
-   `cudlaCreateDevice` is called to create DLA device.
-   `cudlaModuleLoadFromMemory` is called to load the engine memory for DLA use.
-   `cudaMalloc` and `cudlaMemRegister` are called to first allocate memory on GPU, then let the CUDA pointer be registered with the DLA.
-   `cudlaModuleGetAttributes` is called to get module attributes from the loaded module.
-   `cudlaSubmitTask` is called to submit the inference task.


### TensorRT API layers and ops

In this sample, the [ElementWise](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#elementwise-layer) layer is used. For more information, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

## Prerequisites

- **Platform**: This sample can only be built and run on aarch64 platforms with DLA hardware (Jetson or DRIVE). It is not supported on x86.
- **cuDLA library**: The `cudla` library must be available in your CUDA toolkit installation.
- **CMake flag**: The sample requires `-DTRT_BUILD_ENABLE_DLA=ON` to be included in the build.

If built without the DLA flag enabled, this sample will print the following error message:
```
DLA is not enabled, please compile with ENABLE_DLA=1
```
and quit.

## Running the sample

1.  Compile this sample using CMake with the DLA flag enabled:
	```
	cd <TensorRT root directory>
	mkdir -p build && cd build
	cmake .. -DTRT_BUILD_ENABLE_DLA=ON
	make sample_cudla
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample to perform inference on DLA.
    `./sample_cudla`

3. Verify that the sample ran successfully. If the sample runs successfully you should see an output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_cudla # ./sample_cudla
	[I] [TRT]
	[I] [TRT] --------------- Layers running on DLA:
	[I] [TRT] [DlaLayer] {ForeignNode[(Unnamed Layer* 0) [ElementWise]]},
	[I] [TRT] --------------- Layers running on GPU:
	[I] [TRT]
	…(omit messages)
	&&&& PASSED TensorRT.sample_cudla
	```

	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `./sample_cudla -h` command line option.


## Additional resources

The following resources provide a deeper understanding of sampleCudla.

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)
- [Developer Guide for cuDLA APIs](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#cudla-intro)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

June 2022
This is the first release of the `README.md` file.


## Known issues

There are no known issues with this tool.
