# DDS Faster R-CNN Object Detection in TensorRT
## Introduction
The `dds_faster_rcnn` sample demonstrates the usage of [tensorrt.IOutputAllocator](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IOutputAllocator) in TensorRT to execute networks with data-dependent shape (DDS) outputs. In this sample, we showcase an end-to-end workflow for building and running an object detection model [Faster-RCNN](https://arxiv.org/abs/1506.01497).

### What are Data-Dependent Shapes (DDS)?
Data-Dependent Shapes (DDS) refer to shapes of layer outputs in a neural network which depend on the input data to the layer; in other words, it cannot be inferred solely by inspecting the shapes of the layer's input tensors.  An example of this is the output shape of the `INonZeroLayer`, which is determined by the number of non-zero elements in the input tensor.

DDS outputs are common in models that involve dynamic processing, such as object detection, segmentation, and natural language processing.

### What is an `IOutputAllocator`?
An `IOutputAllocator` is an interface in TensorRT that defines a class responsible for dynamically allocating and managing the device memory for output tensors of a TensorRT engine. The class implementing this interface must provide a way to allocate and deallocate memory for output tensors, which can vary in size depending on the input data.

### Why do we need to implement `IOutputAllocator`
In traditional models, the output shapes are typically fixed and known at build time. However, in the case of data-dependent shaped (DDS) outputs, the output size is only known at inference time. This means that the memory allocation for output tensors cannot be determined until the model is actually run with a specific input. To handle this situation, TensorRT provides the `IOutputAllocator` interface, which allows developers to implement a custom memory allocation strategy for DDS outputs. By implementing this interface, developers can ensure that the output tensors are properly allocated and deallocated during inference, avoiding potential memory issues and improving the overall performance of the model.

### How does `IOutputAllocator` work?
To implement the `IOutputAllocator` interface, you need to provide implementations for the following two key methods:

- `reallocate_output_async(self, tensor_name, memory, size, alignment, stream)`: This method is responsible for allocating or reallocating memory for an output tensor. It is called during the inference phase when the output tensor size is known. The method takes in parameters such as the tensor name, current memory address, new size, alignment, and CUDA stream, and returns the new memory address.
- `notify_shape(self, tensor_name, shape)`: This method is used to notify the allocator of a change in the shape of an output tensor. It is typically called after reallocate_output_async() to update the allocator's internal state with the new shape information.
During inference, the TensorRT engine will call these methods to manage the memory allocation for DDS output tensors. The `IOutputAllocator` implementation is responsible for ensuring that the memory allocation is properly handled, taking into account factors such as memory fragmentation, alignment, and performance optimization.

Here is a high-level overview of the workflow:

1. Instantiate the output allocator and attach to TensorRT with [IExecutionContext.set_output_allocator()](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext.set_output_allocator)
1. The TensorRT engine determines that an output tensor needs to be allocated or reallocated.
1. `reallocate_output_async` is called to allocate or reallocate memory for the output tensor.
1. The allocator updates its internal state and returns the new memory address.
1. The TensorRT engine uses the new memory address to store the output tensor data.
1. `notify_shape()` method is called to update the allocator's internal state with the new shape information.

By implementing the `IOutputAllocator` interface, developers can create custom memory allocation strategies that optimize performance, reduce memory fragmentation, and improve the overall efficiency of their model inference.

## Setup
We recommend running these scripts on an environment with TensorRT >= 10.8.0.

Install TensorRT as per the [TensorRT Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html). You will need to make sure the Python bindings for TensorRT are also installed correctly, these are available by installing the `python3-libnvinfer` and `python3-libnvinfer-dev` packages on your TensorRT download.

To simplify TensorRT installation, use an NGC Docker Image, such as:

```bash
docker pull nvcr.io/nvidia/tensorrt:25.01-py3
```

Install all dependencies listed in requirements.txt:

```bash
pip3 install -r requirements.txt
```

## Model Conversion
To start, download the pre-trained Faster R-CNN model in ONNX format using the following command:

```bash
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx
```

With the ONNX model downloaded, run the following command to prepare it for TensorRT engine conversion:

```bash
python3 modify_onnx.py \
    --input ./FasterRCNN-12.onnx \
    --output ./fasterrcnn12_trt.onnx
```

This will create a modified ONNX graph file that is ready for conversion to a TensorRT engine.

## Build TensorRT Engine

To build the TensorRT engine, run the following command:

```bash
python3 build_engine.py \
    --onnx ./fasterrcnn12_trt.onnx \
    --engine ./fasterrcnn12_trt.engine
```

## Inference
To test the built TensorRT engine, download a test image using the following command:

```bash
wget https://onnxruntime.ai/images/demo.jpg
```

Then, run the inference script using the following command:

```
python3 infer.py \
    --engine ./fasterrcnn12_trt.engine \
    --input ./demo.jpg \
    --output ./output_dir \
    --labels labels_coco_80.txt
```
This will perform object detection on the test image and save the output to the specified directory (`output_dir` in this case).

# Changelog

October 2025
Migrate to strongly typed APIs.

August 2025
Removed support for Python versions < 3.10.

February 2025
Initial release
