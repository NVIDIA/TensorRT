# TensorRT Samples

## Contents

### 1. "Hello World" Samples

| Sample | Language | Format | Description |
|---|---|---|---|
| [sampleOnnxMNIST](sampleOnnxMNIST) | C++ | ONNX | “Hello World” For TensorRT With ONNX |
| [network_api_pytorch_mnist](python/network_api_pytorch_mnist) | Python | INetwork | “Hello World” For TensorRT Using Pytorch |

### 2. TensorRT API Samples
| Sample | Language | Format | Description |
|---|---|---|---|
| [sampleAlgorithmSelector](sampleAlgorithmSelector) | C++ | ONNX | Algorithm Selection API usage |
| [sampleCharRNN](sampleCharRNN) | C++ | INetwork | Building An RNN Network Layer By Layer |
| [sampleDynamicReshape](sampleDynamicReshape) | C++ | ONNX | Digit Recognition With Dynamic Shapes In TensorRT |
| [sampleINT8API](sampleINT8API) | C++ | ONNX | Performing Inference In INT8 Precision |
| [sampleNamedDimensions](sampleNamedDimensions) | C++ | ONNX | Working with named input dimensions |
| [sampleOnnxMnistCoordConvAC](sampleOnnxMnistCoordConvAC) | C++ | ONNX | Implementing CoordConv with a custom plugin |
| [sampleIOFormats](sampleIOFormats) | C++ | ONNX | Specifying TensorRT I/O Formats |
| [trtexec](trtexec) | C++ | All | TensorRT Command-Line Wrapper: trtexec |
| [engine_refit_onnx_bidaf](python/engine_refit_onnx_bidaf) | Python | ONNX | refitting an engine built from an ONNX model via parsers. |
| [introductory_parser_samples](python/introductory_parser_samples) | Python | ONNX | Introduction To Importing Models Using TensorRT Parsers |
| [onnx_packnet](python/onnx_packnet) | Python | ONNX | TensorRT Inference Of ONNX Models With Custom Layers |

### 3. Application Samples
| Sample | Language | Format | Description |
|---|---|---|---|
| [detectron2](python/detectron2) | Python | ONNX | Support for Detectron 2 Mask R-CNN R50-FPN 3x model in TensorRT |
| [efficientdet](python/efficientdet) | Python | ONNX | EfficientDet Object Detection with TensorRT |
| [efficientnet](python/efficientnet) | Python | ONNX | EfficientNet V1 and V2 Classification with TensorRT |
| [tensorflow_object_detection_api](python/tensorflow_object_detection_api) | Python | ONNX | TensorFlow Object Detection API Models in TensorRT |
| [yolov3_onnx](python/yolov3_onnx) | Python | ONNX | Object Detection Using YOLOv3 With TensorRT ONNX Backend |

## Known Limitations

  - UFF converter and GraphSurgeon tools are only supported with Tensorflow 1.x
  - For the UFF samples, please use the [NVIDIA tf1 (Tensorflow 1.x)](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running) for running these tests or install Tensorflow 1.x manually.
