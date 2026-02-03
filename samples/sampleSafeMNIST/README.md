# “Hello World” For TensorRT Safety


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleSafeMNIST, consists of two parts; build and infer. The build part of this sample demonstrates how to use the builder `IBuilderConfig::setEngineCapability()` flag for safety. The inference part of this sample demonstrates how to use the safe graph.

The build part builds a safe version of a TensorRT engine and saves it into a binary file, then the infer part loads the prebuilt safe engine and performs inference on an input image.

## How does this sample work?

This sample uses an ONNX model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

Specifically, this sample:
-   Build (sample_mnist_safe_build):
-   Performs the basic setup and initialization of TensorRT
-   [Imports a trained ONNX model using ONNX parser](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html#importing-a-model-using-the-onnx-parser)
-   Preprocesses the input and stores the result in a managed buffer
-   [Builds a safe engine](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html#building-an-engine)
-   Infer (sample_mnist_safe_infer):
-   Create a safe graph for setting up tensors and executing inference on a built network.

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely that in the image.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT API: Layers](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Activation.html)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Convolution.html)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.


## Running the sample

1. Download the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md) to read images from the ubyte file. The images need to be saved into `.pgm` format and renamed as `<label>.pgm`.

2. Put the images into the `data/mnist` directory together with the existing ONNX network `safe_mnist.onnx`.

3. Compile the sample by following the build instructions in the [TensorRT README](https://github.com/NVIDIA/TensorRT/). This will build the sample binaries, including `sample_mnist_safe_build` and `sample_mnist_safe_infer`.

4. The compile options are summarized in the following table.

	| Compile Option                  | Default |Description|
	| ------------------------------- | ------- |---------- |
	|TRT_SAFETY_INFERENCE_ONLY        | OFF     |When enabled, build the infer part only, skip compiling the builder part.|

5.  Run the sample to build a TensorRT safe engine.
    ```

    ./sample_mnist_safe_build [--datadir=/path/to/data/dir/] [--remoteAutoTuningConfig=<config>]

    ```
	This sample generates `safe_mnist.engine`, which is a binary file that contains the serialized engine data.

	This sample reads ONNX model to build the network:
	- `safe_mnist.onnx` - The ONNX model that contains the network design.

	**Note:** By default, this sample expects these files to be in either the `data/samples/mnist/` or `data/mnist/` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path/` as a command line argument.

6. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_mnist_safe_build # ./sample_mnist_safe_build
	[I] Building a GPU inference engine for MNIST
	[I] [TRT] Detected 1 input and 1 output network tensors.
	&&&& PASSED TensorRT.sample_mnist_safe_build # ./sample_mnist_safe_build
	```
	This output shows that the sample ran successfully; `PASSED`.

7. Run the sample to perform inference on the digit:
	`./sample_mnist_safe_infer`

	**Note:** This sample expects `./sample_mnist_safe_build` has been run to generate a safe engine file. It loads input image from `data/samples/mnist` directory, and walks back 10 directories to locate the image.

8. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 3:
	```
	&&&& RUNNING TensorRT.sample_mnist_safe_infer # ./sample_mnist_safe_infer
	[I] Running a GPU inference engine for MNIST
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@#-:.-=@@@@@@@@@@@@@@
	@@@@@%= . *@@@@@@@@@@@@@@@@@
	@@@@% .:+%%% *@@@@@@@@@@@@@@
	@@@@+=#@@@@@# @@@@@@@@@@@@@@
	@@@@@@@@@@@% @@@@@@@@@@@@@@@
	@@@@@@@@@@@: *@@@@@@@@@@@@@@
	@@@@@@@@@@- .@@@@@@@@@@@@@@@
	@@@@@@@@@: #@@@@@@@@@@@@@@@@
	@@@@@@@@: +*%#@@@@@@@@@@@@@@
	@@@@@@@% :+*@@@@@@@@@@@@@@@@
	@@@@@@@@#*+--.:: +@@@@@@@@@@
	@@@@@@@@@@@@@@@@#=:. +@@@@@@
	@@@@@@@@@@@@@@@@@@@@ .@@@@@@
	@@@@@@@@@@@@@@@@@@@@#. #@@@@
	@@@@@@@@@@@@@@@@@@@@# @@@@@@
	@@@@@@@@@%@@@@@@@@@@- +@@@@@
	@@@@@@@@#-@@@@@@@@*. =@@@@@@
	@@@@@@@@ .+%%%%+=. =@@@@@@@@
	@@@@@@@@ =@@@@@@@@@@@@@@@@@@
	@@@@@@@@*=: :--*@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	[I]  Prob 0  0.0000 Class 0:
	[I]  Prob 1  0.0000 Class 1:
	[I]  Prob 2  0.0000 Class 2:
	[I]  Prob 3  1.0000 Class 3: **********
	[I]  Prob 4  0.0000 Class 4:
	[I]  Prob 5  0.0000 Class 5:
	[I]  Prob 6  0.0000 Class 6:
	[I]  Prob 7  0.0000 Class 7:
	[I]  Prob 8  0.0000 Class 8:
	[I]  Prob 9  0.0000 Class 9:

	&&&& PASSED TensorRT.sample_safe_mnist_infer # ./sample_mnist_safe_infer
	```

	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `./sample_mnist_safe_build [-h or --help]` command.

**Note:** This sample supports long flags (e.g., `--help`, `--verbose`) and limited short flags (`-h`, `-v`, `-d`). Only explicitly whitelisted short flags are supported to avoid conflicts with negative numbers.

```
Usage: ./sample_mnist_safe_build [-h or --help] [--datadir=<path to data directory>]
--help          Display help information
--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--threads       Specify the number of independent threads to drive engines. Default is 1.
--verbose       Use verbose logging.
--saveEngine    Save the serialized engine to the file, the default is to use (safe_mnist.engine).
--remoteAutoTuningConfig  Set the remote auto tuning config. Format: protocol://username[:password]@hostname[:port]?param1=value1&param2=value2
                Example: ssh://user:pass@192.0.2.100:22?remote_exec_path=/opt/tensorrt/bin&remote_lib_path=/opt/tensorrt/lib
```

#### When to use remoteAutoTuningConfig

The `--remoteAutoTuningConfig` parameter is designed for **cross-platform development scenarios** where you need to:

**Primary Use Case - Cross-Platform Building:**
- **Build on Host Platform**: Compile and build TensorRT engines on a development machine (e.g., Linux x86_64)
- **Auto-tune on Target Platform**: Perform kernel auto-tuning on the actual deployment target (e.g., QNX aarch64)

**Typical Scenarios:**
- **QNX Development**: Building engines on Linux development machines but deploying on QNX automotive platforms

**Important Technical Limitation:**
- **QNX Safety Devices**: QNX safety platforms do **NOT** support engine building operations. All engine construction must be performed on development platforms (Linux/QNX standard), making remote auto-tuning essential for safety deployments.


## Additional resources

The following resources provide a deeper understanding about sampleSafeMNIST.

**Dataset**
- [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

**Documentation**
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

Jun. 2019
This is the first release of the `README.md` file and sample.

Dec. 2019
Switch the sample to use ONNX model, and update the content of `README.md`.

Jun. 2020
This sample was updated to fit TensorRT API changes since version 6.3. Please see [TensorRT API](http://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html).

Sep. 2020
This sample was updated to fit TensorRT API changes since version 6.4.

Mar. 2022
This sample was updated for DriveOS 6.0 and later releases.

Jun. 2023
This sample was updated to remove deprecated APIs of ICudaEngine and IExecutionContext.

Jan. 2024
Update static linking description

Feb. 2025
This sample was updated for TRT 10.x and later releases.

Jul. 2025
This sample was updated for the TRT 10.13.1 safety release.

Dec. 2025
This sample was updated to use the CMake-based build system.

## Known issues

There are no known issues in this sample.
