# TensorRT Command-Line Wrapper: trtexec

**Table Of Contents**
- [TensorRT Command-Line Wrapper: trtexec](#tensorrt-command-line-wrapper-trtexec)
  - [Description](#description)
  - [Building `trtexec`](#building-trtexec)
  - [Using `trtexec`](#using-trtexec)
    - [Example 1: Profiling a custom layer](#example-1-profiling-a-custom-layer)
    - [Example 2: Running a network on DLA](#example-2-running-a-network-on-dla)
    - [Example 3: Running an ONNX model with full dimensions and dynamic shapes](#example-3-running-an-onnx-model-with-full-dimensions-and-dynamic-shapes)
    - [Example 4: Collecting and printing a timing trace](#example-4-collecting-and-printing-a-timing-trace)
    - [Example 5: Tune throughput with multi-streaming](#example-5-tune-throughput-with-multi-streaming)
    - [Example 6: Create a strongly typed plan file](#example-6-create-a-strongly-typed-plan-file)
  - [Tool command line arguments](#tool-command-line-arguments)
  - [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

Included in the `samples` directory is a command line wrapper tool, called `trtexec`. `trtexec` is a tool to quickly utilize TensorRT without having to develop your own application. The `trtexec` tool has two main purposes:
-   It’s useful for benchmarking networks on random or user-provided input data.
-   It’s useful for generating serialized engines from models.

**Benchmarking network** - If you have a model saved as an ONNX file, you can use the `trtexec` tool to test the performance of running inference on your network using TensorRT. The `trtexec` tool has many options for specifying inputs and outputs, iterations for performance timing, precision allowed, and other options.

**Serialized engine generation** - If you generate a saved serialized engine file, you can pull it into another application that runs inference. For example, you can use the [TensorRT Laboratory](https://github.com/NVIDIA/tensorrt-laboratory) to run the engine with multiple execution contexts from multiple threads in a fully pipelined asynchronous way to test parallel inference performance. Also, in INT8 mode, random weights are used, meaning trtexec does not provide calibration capability.

**Using custom input data** - By default trtexec will run inference with randomly generated inputs. To provide custom inputs for an inference run, trtexec expects a binary file containing the data for each input tensor. It is recommended that this binary file be generated through `numpy`. For example, to create custom data of all ones to an ONNX model with one input named `data` with shape `(1,3,244,244)` and type `FLOAT`:

```
import numpy as np
data = np.ones((1,3,244,244), dtype=np.float32)
data.tofile("data.bin")
```

This binary file can be be loaded by trtexec during inference by using the `--loadInputs` flag:

```
./trtexec --onnx=model.onnx --loadInputs="data":data.bin
```

## Building `trtexec`

`trtexec` can be used to build engines, using different TensorRT features (see command line arguments), and run inference. `trtexec` also measures and reports execution time and can be used to understand performance and possibly locate bottlenecks.

Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

## Using `trtexec`

`trtexec` can build engines from models in ONNX format.

### Example 1: Profiling a custom layer

You can profile a custom layer using the `IPluginRegistry` for the plugins and `trtexec`. You’ll need to first register the plugin with `IPluginRegistry`.

If you are using TensorRT shipped plugins, you should load the `libnvinfer_plugin.so` file, as these plugins are pre-registered.

If you have your own plugin, then it has to be registered explicitly. The following macro can be used to register the plugin creator `YourPluginCreator` with the `IPluginRegistry`.
`REGISTER_TENSORRT_PLUGIN(YourPluginCreator);`

### Example 2: Running a network on DLA

To run the MNIST network on NVIDIA DLA (Deep Learning Accelerator) using `trtexec` in FP16 mode, issue:
```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=1 --fp16 --allowGPUFallback
```
To run the MNIST network on DLA using `trtexec` in INT8 mode, issue:
```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=1 --int8 --allowGPUFallback
```
To run the MNIST network on DLA using `trtexec`, issue:
```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=0 --fp16 --allowGPUFallback
```

For more information about DLA, see [Working With DLA](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#dla_topic).

### Example 3: Running an ONNX model with full dimensions and dynamic shapes

To run an ONNX model in full-dimensions mode with static input shapes:

```
./trtexec --onnx=model.onnx
```

The following examples assumes an ONNX model with one dynamic input with name `input` and dimensions `[-1, 3, 244, 244]`

To run an ONNX model in full-dimensions mode with an given input shape:

```
./trtexec --onnx=model.onnx --shapes=input:32x3x244x244
```

To benchmark your ONNX model with a range of possible input shapes:

```
./trtexec --onnx=model.onnx --minShapes=input:1x3x244x244 --optShapes=input:16x3x244x244 --maxShapes=input:32x3x244x244 --shapes=input:5x3x244x244
```

### Example 4: Collecting and printing a timing trace

When running, `trtexec` prints the measured performance, but can also export the measurement trace to a json file:
```
./trtexec --onnx=data/mnist/mnist.onnx --exportTimes=trace.json
```
Once the trace is stored in a file, it can be printed using the `tracer.py` utility. This tool prints timestamps and duration of input, compute, and output, in different forms:
```
./tracer.py trace.json
```
Similarly, profiles can also be printed and stored in a json file. The utility `profiler.py` can be used to read and print the profile from a json file.

### Example 5: Tune throughput with multi-streaming

Tuning throughput may require running multiple concurrent streams of execution. This is the case for example when the latency achieved is well within the desired
threshold, and we can increase the throughput, even at the expense of some latency. For example, saving engines with different precisions and assume that both
execute within 2ms, the latency threshold:
```
trtexec --onnx=resnet50.onnx --saveEngine=g1.trt --int8 --skipInference
trtexec --onnx=resnet50.onnx --saveEngine=g2.trt --best --skipInference
```
Now, the saved engines can be tried to find the combination precision/streams below 2 ms that maximizes the throughput:
```
trtexec --loadEngine=g1.trt --streams=2
trtexec --loadEngine=g1.trt --streams=3
trtexec --loadEngine=g1.trt --streams=4
trtexec --loadEngine=g2.trt --streams=2
```

### Example 6: Create a strongly typed plan file
This flag will create a network with the `NetworkDefinitionCreationFlag::kSTRONGLY_TYPED` flag where tensor data types are inferred from network input types
and operator type specification.  Use of specific builder precision flags such as `--int8` or `--best` with this option is not allowed.
```
./trtexec --onnx=model.onnx --stronglyTyped
```

## Tool command line arguments

To see the full list of available options and their descriptions, issue the `./trtexec --help` command.

**Note:** Specifying the `--safe` parameter turns the safety mode switch `ON`. By default, the `--safe` parameter is not specified; the safety mode switch is `OFF`. The layers and parameters that are contained within the `--safe` subset are restricted if the switch is set to `ON`. The switch is used for prototyping the safety restricted flows until the TensorRT safety runtime is made available. This parameter is required when loading or saving safe engines with the standard TensorRT package. For more information, see the [Working With Automotive Safety section in the TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#working_auto_safety).

## Additional resources

The following resources provide more details about `trtexec`:

**Documentation**
- [NVIDIA trtexec](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#trtexec)
- [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

# Changelog

April 2019
This is the first release of this `README.md` file.

# Known issues

There are no known issues in this sample.
