# Working with ONNX models with named input dimensions


**Table Of Contents**
- [Description](#description)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `sampleNamedDimensions`, illustrates how to work with ONNX models with named input dimensions in TensorRT.

ONNX has a notion of named dimension parameters: two network inputs with the same named dimension parameter are considered equal. TensorRT supports this feature by checking that in the optimization profile these dimensions have overlapping intervals and that at runtime they have the same value.

Here, we synthetically create an ONNX model consisting of a single [Concat](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat) layer with two 2D input tensors:
```
input0      input1
    \         /
     \       /
      --------
      |Concat|
      --------
          |
          |
       output
```
Concatenation is performed on the zeroth axis, so only the first dimensions of the input tensors are required to be the same. However, since both inputs have dimension `[n_rows, 8]`, the named dimensions `n_rows` additionally require the zeroth dimensions of the two input tensors to match as well.


## Running the sample

1.  The sample gets compiled when building the TensorRT OSS following the [instructions](https://github.com/NVIDIA/TensorRT). The binary named `sample_named_dimensions` will be created in the output directory.

2.  Generate the ONNX model file by running this command:
	```
	python3 create_model.py
	```
	This will create a file named `concat_layer.onnx`.

3. Run the sample to build and run the engine from the ONNX model.
	```
	./sample_named_dimensions [-h or --help] [-d or --datadir=<path to data directory>]
	```

3.  Verify that the sample has run successfully. If successful you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_named_dimensions [TensorRT v8500] # build/x86_64-gnu/sample_named_dimensions
	[I] [TRT] ----------------------------------------------------------------
	[I] [TRT] Input filename:   ../trt/samples/sampleNamedDimensions/concat_layer.onnx
	[I] [TRT] ONNX IR version:  0.0.7
	[I] [TRT] Opset version:    11
	[I] [TRT] Producer name:
	[I] [TRT] Producer version:
	[I] [TRT] Domain:
	[I] [TRT] Model version:    0
	[I] [TRT] Doc string:
	[I] [TRT] ----------------------------------------------------------------
	[I] Input0:
	-4.17896 4.21201 -8.6982 9.33153 -4.90741 1.1953 9.45208 1.04329
	-5.47509 0.150872 -4.29573 1.72331 3.69642 5.73303 -4.89766 5.00559
	
	[I] Input1:
	9.01907 3.57581 -1.36986 -3.22044 -5.90874 -8.11433 2.38472 -0.0868187
	0.842402 -1.75138 4.55962 -6.38946 -7.73614 -1.26044 -4.23012 4.33806
	
	[I] Output:
	-4.17896 4.21201 -8.6982 9.33153 -4.90741 1.1953 9.45208 1.04329
	-5.47509 0.150872 -4.29573 1.72331 3.69642 5.73303 -4.89766 5.00559
	9.01907 3.57581 -1.36986 -3.22044 -5.90874 -8.11433 2.38472 -0.0868187
	0.842402 -1.75138 4.55962 -6.38946 -7.73614 -1.26044 -4.23012 4.33806
	
	&&&& PASSED TensorRT.sample_named_dimensions [TensorRT v8500] # build/x86_64-gnu/sample_named_dimensions
	```


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding about the named input dimensions feature in the ONNX project:

**ONNX**
- [GitHub: ONNX](https://github.com/onnx/onnx)
- [Github: ONNX-TensorRT Open source parser](https://github.com/onnx/onnx-tensorrt)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

June 2022
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
