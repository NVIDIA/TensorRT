# Performing Inference In INT8 Precision


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Configuring the builder to use INT8 without the INT8 calibrator](#configuring-the-builder-to-use-int8-without-the-int8-calibrator)
	* [Configuring the network to use custom dynamic ranges and set per-layer precision](#configuring-the-network-to-use-custom-dynamic-ranges-and-set-per-layer-precision)
	* [Building the engine](#building-the-engine)
	* [Running the engine](#running-the-engine)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Preparing sample data](#preparing-sample-data)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Models other than ResNet-50 with custom configuration](#models-other-than-resnet-50-with-custom-configuration)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

This sample, sampleINT8API, performs INT8 inference without using the INT8 calibrator; using the user provided per activation tensor dynamic range. INT8 inference is available only on GPUs with compute capability 6.1 or newer and supports Image Classification ONNX models such as ResNet-50, VGG19, and MobileNet.

Specifically, this sample demonstrates how to:
-   Use `nvinfer1::ITensor::setDynamicRange` to set per-tensor dynamic range
-   Use `nvinfer1::ILayer::setPrecision` to set computation precision of a layer
-   Use `nvinfer1::ILayer::setOutputType` to set output tensor data type of a layer
-   Perform INT8 inference without using INT8 calibration

## How does this sample work?

In order to perform INT8 inference, you need to provide TensorRT with the dynamic range for each network tensor, including network input and output tensor. One way to choose the dynamic range is to use the TensorRT INT8 calibrator. But if you don't want to go that route (for example, let’s say you used quantization-aware training or you just want to use the min and max tensor values seen during training), you can skip the INT8 calibration and set custom per-network tensor dynamic ranges. This sample implements INT8 inference for the ONNX ResNet-50 model using per-network tensor dynamic ranges specified in an input file.

This sample uses the [ONNX ResNet-50 model](https://github.com/onnx/models/tree/master/vision/classification/resnet/model).

Specifically, this sample performs the following steps:
- [Configuring the builder to use INT8 without the INT8 calibrator](#configuring-the-builder-to-use-int8-without-the-int8-calibrator)
- [Configuring the network to use custom dynamic ranges and set per-layer precision](#configuring-the-network-to-use-custom-dynamic-ranges-and-set-per-layer-precision)
- [Building the engine](#building-the-engine)
- [Running the engine](#running-the-engine)

### Configuring the builder to use INT8 without the INT8 calibrator

1.  Ensure that INT8 inference is supported on the platform:
	`if (!builder->platformHasFastInt8()) return false;`

2.  Enable INT8 mode by setting the builder flag:
	`builder->setInt8Mode(true);`

	You can choose not to provide the INT8 calibrator.
	`builder->setInt8Calibrator(nullptr);`

	If you want to provide the calibrator, manual dynamic range will override calibration generate dynamic range/scale. See sampleINT8 on how to setup INT8 calibrator.

3.  Optionally and for debugging purposes, the following flag configures the builder to choose type conforming layer implementation, if one exists.

	For example, in the case of `DataType::kINT8`, types are requested by `setInt8Mode(true)`. Setting this flag ensures that only the conformant layer implementation (with `kINT8` input and output types), are chosen even if a high performance non-conformat implementation is available. If no conformant layer exists, TensorRT will choose a non-conformant layer if available regardless of the setting for this flag.

	`builder->setStrictTypeConstraints(true);`

### Configuring the network to use custom dynamic ranges and set per-layer precision

1.  Iterate through the network to set the per activation tensor dynamic range.
	```
	readPerTensorDynamicRangeValue() // This function populates dictionary with keys=tensor_names, values=floating point dynamic range.
	```

2.  Set the dynamic range for network inputs:
	```
	string input_name = network->getInput(i)->getName();
	network->getInput(i)->setDynamicRange(-tensorMap.at(input_name), tensorMap.at(input_name));
	```

3.  Set the dynamic range for per layer tensors:
	```
	string tensor_name = network->getLayer(i)->getOutput(j)->getName(); network->getLayer(i)->getOutput(j)->setDynamicRange(-tensorMap.at(name), tensorMap.at(name));
	```

4.  Optional: This sample also showcases using layer precision APIs. Using these APIs, you can selectively choose to run the layer with user configurable precision and type constraints. It may not result in optimal inference performance, but can be helpful while debugging mixed precision inference.

	Iterate through the network to per layer precision:
	```
	auto layer = network->getLayer(i);
	layer->setPrecision(nvinfer1::DataType::kINT8);
	```

	This gives the layer’s inputs and outputs a preferred type (for example, `DataType::kINT8`). You can choose a different preferred type for an input or output of a layer using:
	```
	for (int j=0; j<layer->getNbOutputs(); ++j) {
	layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
	}
	```

	Using layer precision APIs with `builder->setStrictTypeConstraints(true)` set, ensures that the requested layer precisions are obeyed by the builder irrespective of the performance. If no implementation is available with request precision constraints, the builder will choose the fastest implementation irrespective of precision and type constraints. For more information on using mixed precision APIs, see [Setting The Layer Precision Using C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#set_layer_mp_c).

### Building the engine

After we configure the builder with INT8 mode and calibrator, we can build the engine similar to any FP32 engine.

`ICudaEngine* engine = builder->buildCudaEngine(*network);`

### Running the engine

After the engine has been built, it can be used just like an FP32 engine. For example, inputs and outputs remain in 32-bit floating point.

1.  Create an execution context and CUDA stream for the execution of this inference.
	```
	auto context = mEngine->createExecutionContext();
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	```

2.  Copy the data from the host input buffers to the device input buffers.
    `buffers.copyInputToDeviceAsync(stream);`

3.  Enqueue the inference work and perform actual inference.
	```
	context->enqueue(batchSize, buffers.getDeviceBindings().data(), input_stream, nullptr))
	```

4.  Copy data from the device output buffers to the host output buffers.
	`buffers.copyOutputToHostAsync(stream);`

5.  Wait for the work in the stream to complete and release it.
	```
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	```

6.  Check and print the output of the inference.
	`outputCorrect = verifyOutput(buffers);`

### TensorRT API layers and ops

This sample demonstrates how you can enable INT8 inference using the following mixed precision APIs.

[ITensor::SetDynamicRange](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_tensor.html#a956f662b1d2ebe7ba3aba3391aedddf5)
Set dynamic range for the tensor. Currently, only symmetric ranges are supported, therefore, the larger of the absolute values of the provided bounds is used.

[ILayer::SetPrecision](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_layer.html#ac66f1546a28a92c20a76718a6762ea14)
Set the computational precision of this layer. Setting the precision forces TensorRT to choose the implementations which run at this precision. If precision is not set, TensorRT will select the computational precision based on performance considerations and the flags specified to the builder.

[ILayer::SetOutputType](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_layer.html#a85aded4e3ff0867e392602551d5b5dc7)
Set the output type of this layer. Setting the output type forces TensorRT to choose the implementations which generate output data with the given type. If the output type is not set, TensorRT will select the implementation based on performance considerations and the flags specified to the builder.

## Preparing sample data

In addition to the model file and input image, you will need per-tensor dynamic range stored in a text file along with the ImageNet label reference file.

The following required files are included in the package and are located in the `data/int8_api` directory.

`reference_labels.txt`
The ImageNet reference label file.

`resnet50_per_tensor_dynamic_range.txt`
The ResNet-50 per-tensor dynamic ranges file.

`airliner.ppm`
The image to be inferred.

1.  Download the [ONNX ResNet-50 model](https://github.com/onnx/models/tree/master/vision/classification/resnet/model).
    ```bash
    wget https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz -O $TRT_DATADIR/int8_api/resnet50.tar.gz
    ```

2.  Unpackage the model file.
    ```bash
    tar zxvf $TRT_DATADIR/int8_api/resnet50.tar.gz -C $TRT_DATADIR/int8_api/
    ```

3.  Copy `resnet50/model.onnx` to the `data/int8_api/resnet50.onnx` directory.
    ```bash
    mv $TRT_DATADIR/int8_api/resnet50/model.onnx $TRT_DATADIR/int8_api/resnet50.onnx
    ```

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2. Run the sample to perform INT8 inference on a classification network, for example, ResNet-50.

    To run INT8 inference with custom dynamic ranges:
    ```bash
    ./sample_int8_api [--model=model_file] [--ranges=per_tensor_dynamic_range_file] [--image=image_file] [--reference=reference_file] [--data=/path/to/data/dir] [--useDLACore=<int>] [-v or --verbose]
    ```

    For example:
    ```bash
    ./sample_int8_api --model=$TRT_DATADIR/resnet50/ResNet50.onnx --image=$TRT_DATADIR/int8_api/airliner.ppm --reference=$TRT_DATADIR/int8_api/reference_labels.txt --ranges=$TRT_DATADIR/int8_api/resnet50_per_tensor_dynamic_range.txt
    ```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:

	```
	&&&& RUNNING TensorRT.sample_int8_api # ./sample_int8_api
	[I] Please follow README.md to generate missing input files.
	[I] Validating input parameters. Using following input files for inference.
	[I]     Model File: ../../../../../../../../../data/samples/int8_api/resnet50.onnx
	[I]     Image File: ../../../../../../../../../data/samples/int8_api/airliner.ppm
	[I]     Reference File: ../../../../../../../../../data/samples/int8_api/reference_labels.txt
	[I]     Dynamic Range File: ../../../../../../../../../data/samples/int8_api/resnet50_per_tensor_dynamic_range.txt
	[I] Building and running a INT8 GPU inference engine for ../../../../../../../../../data/samples/int8_api/resnet50.onnx
	[I] [TRT] ----------------------------------------------------------------
	[I] [TRT] Input filename:   ../../../../../../../../../data/samples/int8_api/resnet50.onnx
	[I] [TRT] ONNX IR version:  0.0.3
	[I] [TRT] Opset version:    9
	[I] [TRT] Producer name:    onnx-caffe2
	[I] [TRT] Producer version:
	[I] [TRT] Domain:
	[I] [TRT] Model version:    0
	[I] [TRT] Doc string:
	[I] [TRT] ----------------------------------------------------------------
	[I] Setting Per Layer Computation Precision
	[I] Setting Per Tensor Dynamic Range
	[W] [TRT] Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32 or Bool.
	[I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
	[I] [TRT] Detected 1 inputs and 1 output network tensors.
	[I] [TRT] Total Host Persistent Memory: 123728
	[I] [TRT] Total Device Persistent Memory: 0
	[I] [TRT] Total Scratch Memory: 0
	[I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 116 MiB, GPU 4523 MiB
	[I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 3.49361ms to assign 3 blocks to 74 nodes requiring 2408448 bytes.
	[I] [TRT] Total Activation Memory: 2408448
	[I] [TRT] Loaded engine size: 25 MiB
	[I] SampleINT8API result: Detected:
	[I] [1] space shuttle
	[I] [2] airliner
	[I] [3] warplane
	[I] [4] projectile
	[I] [5] wing
	&&&& PASSED TensorRT.sample_int8_api # ./sample_int8_api
	```

	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Models other than ResNet-50 with custom configuration

In order to use this sample with other model files with a custom configuration, perform the following steps:

1.  Download the [Image Classification model files](https://github.com/onnx/models/tree/master/vision/classification) from GitHub.

2.  Create an input image with a PPM extension. Resize it with the dimensions of 224x224x3.

3.  Create a file called `reference_labels.txt`.

	**Note:** Ensure each line corresponds to a single imagenet label. You can download the imagenet 1000 class human readable labels from [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). The reference label file contains only a single label name per line, for example, `0:'tench, Tinca tinca'` is represented as `tench`.

4.  Create a file called `<network_name>_per_tensor_dynamic_ranges.txt`.

	1.  Before you can create the dynamic range file, you need to generate the tensor names by providing the dynamic range for each network tensor.

		This sample provides an option to write names of the network tensors to a file, for example `network_tensors.txt`. This file can then be used to generate the `<network_name>_per_tensor_dynamic_ranges.txt` file in step 4-2 below. To generate the list of network tensors file, perform the following steps:

		i.  Write network tensors to a file:
		```bash
		./sample_int8_api [--model=model_file] [--write_tensors] [--network_tensors_file=network_tensors.txt] [-v or --verbose]
		```

		ii.  Run INT8 inference with user provided dynamic ranges:
		```bash
		./sample_int8_api [--model=model_file] [--ranges=per_tensor_dynamic_range_file] [--image=image_file] [--reference=reference_file] [--data=/path/to/data/dir] [--useDLACore=<int>] [-v or --verbose]
		```

		sampleINT8API needs following files to build the network and run inference:

		`<network>.onnx`
		The model file which contains the network and trained weights.

		`Reference_labels.txt`
		Labels reference file i.e. ground truth ImageNet 1000 class mappings.

		`Per_tensor_dynamic_range.txt`
		Custom per-tensor dynamic range file or you can simply override them by iterating through network layers.

		`Image_to_infer.ppm`
		PPM Image to run inference with.

		**Note:** By default, the sample expects these files to be in either the `data/samples/int8_api/` or `data/int8_api/` directories. The list of default directories can be changed by adding one or more paths with `--data=/new/path` as a command line argument.

	2.  To create the `<network_name>_per_tensor_dynamic_ranges.txt` file, ensure each line corresponds to the tensor name and floating point dynamic range, for example `<tensor_name> : <float dynamic range>`.

		Tensor names generated in the `network_tensors.txt` file (step 4-1) can be used here to represent `<tensor_name>`. The dynamic range can either be obtained from training (by measuring the `min` and `max` value of activation tensors in each epoch) or from using custom post processing techniques (similar to TensorRT calibration). You can also choose to use a dummy per-tensor dynamic range to run the sample.

		**Note:** INT8 inference accuracy may reduce when dummy/random dynamic ranges are provided.

# Additional resources

The following resources provide a deeper understanding how to perform inference in INT8:

**INT8API:**
- [Setting Per-Tensor Dynamic Range Using C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#set_tensor_mp_c)

**Generate per-tensor dynamic range:**
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)
- [Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper](https://arxiv.org/pdf/1806.08342.pdf)
- [8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)

**Models:**
- [ONNX ResNet-50 model](https://github.com/onnx/models/tree/master/vision/classification/resnet/model)
- [Image Classification Model Files](https://github.com/onnx/models/tree/master/vision/classification)

**Blogs:**
- [Why are Eight Bits Enough for Deep Neural Networks?](https://petewarden.com/2015/05/23/why-are-eight-bits-enough-for-deep-neural-networks/)
- [What I’ve learned about Neural Network Quantization](https://petewarden.com/2017/06/22/what-ive-learned-about-neural-network-quantization/)

**Videos:**
- [Inference and Quantization](https://www.youtube.com/watch?v=VsGX9kFXjbs)
- [8-bit Inference with TensorRT Webinar](http://on-demand.gputechconf.com/gtcdc/2017/video/DC7172/)

**Documentation:**
- [Introduction to NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working with TensorRT Using the C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

March 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
