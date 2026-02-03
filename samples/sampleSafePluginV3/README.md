# Plugin Sample For TensorRT Safety


**Table Of Contents**
- [Description](#description)
- [Workflow for safety plugin deployment](#workflow-for-safety-plugin-deployment)
- [How does this sample work?](#how-does-this-sample-work)
  * [Register safe plugin creator](#register-safe-plugin-creator)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
  * [Sample `--help` options](#sample---help-options)
    * [When to use remoteAutoTuningConfig](#when-to-use-remoteautotuningconfig)
- [Shared plugin library and trtexec](#shared-plugin-library-and-trtexec)
  * [Creation of safety plugin library](#creation-of-safety-plugin-library)
  * [Using the plugin library with trtexec and trtexec\_safe](#using-the-plugin-library-with-trtexec-and-trtexec_safe)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleSafePluginV3, consists of two parts: build and infer. The build part of this sample demonstrates how to use the builder in safety for networks that include plugins. The inference part of this sample demonstrates how to use the safe runtime and engine for networks that include plugins.

The build part builds a safe version of a TensorRT engine and saves it into a binary file, then the infer part loads the prebuilt safe engine and performs inference on an input image. The infer part uses the safety header proxy, with the following build steps demonstrating how to build it against the safe runtime for deployment and development.

## Workflow for safety plugin deployment

The deployment of the safety plugin on QNX safety involves a series of steps designed to ensure reliability. The recommended workflow is as follows:

-   Build safety plugin for QNX standard platform and proxy plugin for X86 platform.
-   Build safety engine from X86 platform through remote auto-tuning on QNX safety platform.
-   Run safety engine with safety plugin on QNX standard platform with proxy runtime to perform necessary testing.
-   Build safety plugin for QNX safety platform.
-   Deploy safety engine on QNX safety platform with safety plugin.

## How does this sample work?

This sample uses an ONNX model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

Specifically, this sample:
-   Registers a Max Pooling plugin
-   Performs the basic setup and initialization of TensorRT using the ONNX parser
-   Imports a trained ONNX model using the ONNX parser
-   Preprocesses the input and stores the result in a managed buffer
-   Builds a safe engine
-   Serializes and deserializes the engine
-   Uses the engine to perform inference on an input image

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely that in the image.

### Register safe plugin creator

`ISafePluginCreatorV3One` will be used by TensorRT's builder for engine creation. You would need to register the safe plugin creator into the safePluginRegistry for engine generation and validation. The current sample registers MaxPoolCreator in namespace "" to parse and verify model `mnist_safe_plugin.onnx`.
```
  auto safePluginRegistry = nvinfer2::safe::getSafePluginRegistry(g_recorder);
  safePluginRegistry->registerCreator(maxPoolPluginCreator, "", g_recorder);
```

## Prerequisites
1. Preparing sample data

See [Preparing sample data](../README.md#preparing-sample-data) in the main samples README.

## Running the sample

1. Verify that the MNIST images are in `$TRT_DATADIR/mnist` and the ONNX network `mnist_safe_plugin.onnx` is in `$TRT_DATADIR/safe_plugin`.

2. Compile the sample by following the build instructions in the [TensorRT README](https://github.com/NVIDIA/TensorRT/). This will build the sample binaries, including `sample_plugin_safe_build` and `sample_plugin_safe_infer`.

3. The options that control compiling against safety runtime are summarized in the following table.
	| Compile Option                  | Default |Description|
	| ------------------------------- | ------- |---------- |
	|TRT_SAFETY_INFERENCE_ONLY        | OFF     |When enabled, build the infer part only, skip compiling the builder part.|


4.  Run the sample to build a TensorRT safe engine.
	```
	./sample_plugin_safe_build [--datadir=/path/to/data/dir/] [--remoteAutoTuningConfig=<config>]
	```

	This sample generates `safe_plugin.engine`, which is a binary file that contains the serialized engine data.

	This sample reads ONNX model to build the network:

        - `mnist_safe_plugin.onnx` - The ONNX model that contains the network design with maxPoolPlugin, version 1, namespace ""

	**Note:** By default, this sample expects these files to be in either the `data/samples/safe_plugin/` or `data/safe_plugin/` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path/` as a command line argument.

5. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    &&&& RUNNING TensorRT.sample_safe_plugin_build [TensorRT v101200] [b5] # sample_plugin_safe_build --datadir=data/samples/safe_plugin/
    [04/23/2025-08:07:23] [I] Building a GPU inference engine for MNIST with plugins
    [04/23/2025-08:07:24] [I] [TRT] [MemUsageChange] Init CUDA: CPU +26, GPU +0, now: CPU 35, GPU 422 (MiB)
    [04/23/2025-08:07:36] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +1639, GPU +8, now: CPU 1874, GPU 430 (MiB)
    [04/23/2025-08:07:36] [I] [TRT] ----------------------------------------------------------------
    [04/23/2025-08:07:36] [I] [TRT] Input filename:   data/samples/safe_plugin/mnist_safe_plugin.onnx
    [04/23/2025-08:07:36] [I] [TRT] ONNX IR version:  0.0.7
    [04/23/2025-08:07:36] [I] [TRT] Opset version:    8
    [04/23/2025-08:07:36] [I] [TRT] Producer name:
    [04/23/2025-08:07:36] [I] [TRT] Producer version:
    [04/23/2025-08:07:36] [I] [TRT] Domain:
    [04/23/2025-08:07:36] [I] [TRT] Model version:    0
    [04/23/2025-08:07:36] [I] [TRT] Doc string:
    [04/23/2025-08:07:36] [I] [TRT] ----------------------------------------------------------------
    [04/23/2025-08:07:36] [I] [TRT] Searching for plugin: MaxPoolPlugin, plugin_version: 1, plugin_namespace:
    [04/23/2025-08:07:36] [W] [TRT] onnxOpImporters.cpp:6641: Attribute pType not found in plugin node! Ensure that the plugin creator has a default value defined or the engine may fail to build.
    [04/23/2025-08:07:36] [I] [TRT] Successfully created plugin: MaxPoolPlugin
    [04/23/2025-08:07:40] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
    [04/23/2025-08:07:40] [I] [TRT] Compiler backend is used during engine build.
    [04/23/2025-08:07:57] [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [04/23/2025-08:07:57] [I] [TRT] Total Host Persistent Memory: 0 bytes
    [04/23/2025-08:07:57] [I] [TRT] Total Device Persistent Memory: 0 bytes
    [04/23/2025-08:07:57] [I] [TRT] Max Scratch Memory: 0 bytes
    [04/23/2025-08:07:57] [I] [TRT] Total Activation Memory: 0 bytes
    [04/23/2025-08:07:57] [I] [TRT] Total Weights Memory: 0 bytes
    [04/23/2025-08:07:57] [I] [TRT] Engine generation completed in 16.9161 seconds.
    [04/23/2025-08:07:57] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 1 MiB
    &&&& PASSED TensorRT.sample_safe_plugin_build [TensorRT v101200] [b5] # sample_plugin_safe_build --datadir=data/samples/safe_plugin/
    ```

	This output shows that the sample ran successfully: `PASSED`.

6. Run the sample to perform inference on the digit:
	`./sample_plugin_safe_infer`

	**Note:** This sample expects `./sample_plugin_safe_build` has been run to generate a safe engine file. It loads input image from `data/samples/mnist` directory, and walks back 10 directories to locate the image.

7. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 0:
    ```
    Input:
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@+  :@@@@@@@@
    @@@@@@@@@@@@@@%= :. --%@@@@@
    @@@@@@@@@@@@@%. -@= - :@@@@@
    @@@@@@@@@@@@@: -@@#%@@ #@@@@
    @@@@@@@@@@@@: #@@@@@@@-#@@@@
    @@@@@@@@@@@= #@@@@@@@@=%@@@@
    @@@@@@@@@@= #@@@@@@@@@:@@@@@
    @@@@@@@@@+ -@@@@@@@@@%.@@@@@
    @@@@@@@@@::@@@@@@@@@@+-@@@@@
    @@@@@@@@-.%@@@@@@@@@@.*@@@@@
    @@@@@@@@ *@@@@@@@@@@@ *@@@@@
    @@@@@@@% %@@@@@@@@@%.-@@@@@@
    @@@@@@@:*@@@@@@@@@+. %@@@@@@
    @@@@@@# @@@@@@@@@# .*@@@@@@@
    @@@@@@# @@@@@@@@=  +@@@@@@@@
    @@@@@@# @@@@@@%. .+@@@@@@@@@
    @@@@@@# @@@@@*. -%@@@@@@@@@@
    @@@@@@# ---    =@@@@@@@@@@@@
    @@@@@@#      *%@@@@@@@@@@@@@
    @@@@@@@%: -=%@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    Set address of Input3 on device at 7fcb26612600
    Set address of Output on device at 7fcb26613400
    Output:
    Prob 0  0.9998 Class 0: **********
    Prob 1  0.0000 Class 1:
    Prob 2  0.0000 Class 2:
    Prob 3  0.0000 Class 3:
    Prob 4  0.0000 Class 4:
    Prob 5  0.0000 Class 5:
    Prob 6  0.0002 Class 6:
    Prob 7  0.0000 Class 7:
    Prob 8  0.0000 Class 8:
    Prob 9  0.0000 Class 9:
    &&&& PASSED TensorRT.sample_plugin_safe_infer # sample_plugin_safe_infer
    ```
	This output shows that the sample ran successfully: `PASSED`.


### Sample `--help` options

For builder, to see the full list of available options and their descriptions, use the `./sample_plugin_safe_build [-h or --help]` command.

**Note:** This sample supports long flags (e.g., `--help`, `--verbose`) and limited short flags (`-h`, `-v`, `-d`). Only explicitly whitelisted short flags are supported to avoid conflicts with negative numbers.

    Usage: ./sample_plugin_safe_build [-h or --help] [-d or --datadir=<path to data directory>]
    --help or -h    Display help information
    --datadir or -d Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/safe_plugin/, data/safe_plugin/)
    --saveEngine    Save the serialized engine to the file, the default is to use safe_plugin.engine.
    --remoteAutoTuningConfig  Set the remote auto tuning config. Format: protocol://username[:password]@hostname[:port]?param1=value1&param2=value2
                    Example: ssh://user:pass@192.0.2.100:22?remote_exec_path=/opt/tensorrt/bin&remote_lib_path=/opt/tensorrt/lib

Rather than passing in their actual password on the command line, a user may instead use the password
string "PROMPT" (without quotes) and they will be interactivly prompted for their password.
This avoids having the password visible or stored in the shell history. This functionality is only
supported on Linux x86 -- the remote autotuning flow runs autotuning on an x86 Linux host
which communicates over SSH to the aarch64 QNX machine.

NOTE: The current implementation only supports username/password authentication. However, adding SSH 
key support would be feasible given that the code is built around SSH.

For inference, to see the full list of available options and their descriptions, use the `./sample_plugin_safe_infer [-h or --help]` command.

    Usage: ./sample_plugin_safe_infer [-h or --help] [-d or --datadir=<path to data directory>]
    --help          Display help information
    --loadEngine    Load the serialized engine to the file, the default is to use safe_plugin.engine.

#### When to use remoteAutoTuningConfig

The `--remoteAutoTuningConfig` parameter is designed for **cross-platform development scenarios** where you need to:

**Primary Use Case - Cross-Platform Building:**
- **Build on Host Platform**: Compile and build TensorRT engines on a development machine (e.g., Linux x86_64)
- **Auto-tune on Target Platform**: Perform kernel auto-tuning on the actual deployment target (e.g., QNX aarch64)

**Typical Scenarios:**
- **QNX Development**: Building engines on Linux development machines but deploying on QNX automotive platforms

**Important Technical Limitation:**
- **QNX Safety Devices**: QNX safety platforms do **NOT** support engine building operations. All engine construction must be performed on development platforms (Linux/QNX standard), making remote auto-tuning essential for safety deployments.

## Shared plugin library and trtexec

Safety plugins must be built into shared libraries and dynamically loaded in applications. As shown in this sample, users may declare and register a plugin creator from their application directly, or define a plugin creator registering function within their plugin libraries themselves. In other words, we present a method that supports the manual registration of safety plugins from within a shared plugin library. Safety plugin libraries adhering to this protocol can be utilized with both trtexec and trtexec_safe.

### Creation of safety plugin library

After implementing the plugin creator, create an API entry function `getSafetyPluginCreator` to pull in the plugin creator. The entry function takes `pluginNamespace` and `pluginName` as arguments. If the input plugin namespace and name are valid, the function shall construct a `pluginCreator` instance, and return a pointer to the instance.
```
extern "C" __attribute__((visibility("default"))) nvinfer2::safe::IPluginCreatorInterface* getSafetyPluginCreator(
    char const* pluginNamespace, char const* pluginName)
```

Please refer to `maxPoolPluginCreatorInterface.cpp` and `maxPoolPluginRuntimeCreatorInterface.cpp` for sample implementations.

### Using the plugin library with trtexec and trtexec_safe

Both `trtexec` and `trtexec_safe` support the `--safetyPlugins` argument, where you specify safety plugins to load from a safety plugin library that implements the manual-registration protocol. The sample plugin and its creator interface will be built into the shared libraries libsample_safe_plugin_v3.so(Linux x86_64 platform) for BUILD and RUNTIME capability and libsample_safe_plugin_v3_safe.so(QNX-safe platform) for RUNTIME only capability.

Sample command to build, validate, and save the safety engine on Linux x86_64 platform.
```
trtexec --onnx=$TRT_DATADIR/safe_plugin/mnist_safe_plugin.onnx --safe --skipInference --fp16 --consistency --safetyPlugins=libsample_safe_plugin_v3.so[::MaxPoolPlugin] --saveEngine=sample.engine
```
Sample command to run the pre-built engine on the QNX-safe platform.
```
trtexec_safe --loadEngine=sample.engine --safetyPlugins=libsample_safe_plugin_v3_safe.so[::MaxPoolPlugin]
```

## Additional resources

The following resources provide a deeper understanding about sampleSafePluginV3.

**Dataset**
- Sample data available in [TensorRT GitHub Releases](https://github.com/NVIDIA/TensorRT/releases).

**Documentation**
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

 - Apr. 2025
   - This is the first release of the `README.md` file and sample.

Jul. 2025
This sample was updated for the TRT 10.13.1 safety release.

Dec. 2025
This sample was updated to use the CMake-based build system.

## Known issues

There are no known issues in this sample.
