# TensorRT Samples

## Contents

### 1. "Hello World" Samples

| Sample | Language | Format | Description |
|---|---|---|---|
| [sampleOnnxMNIST](sampleOnnxMNIST) | C++ | ONNX | “Hello World” For TensorRT With ONNX |
| [sampleMNIST](sampleMNIST) | C++ | Caffe | “Hello World” For TensorRT Using Caffe|
| [end_to_end_tensorflow_mnist](python/end_to_end_tensorflow_mnist) | Python | UFF | “Hello World” For TensorRT Using TensorFlow |
| [network_api_pytorch_mnist](python/network_api_pytorch_mnist) | Python | INetwork | “Hello World” For TensorRT Using Pytorch |

### 2. TensorRT API Samples
| Sample | Language | Format | Description |
|---|---|---|---|
| [sampleAlgorithmSelector](sampleAlgorithmSelector) | C++ | Caffe | Algorithm Selection API usage |
| [sampleCharRNN](sampleCharRNN) | C++ | INetwork | Building An RNN Network Layer By Layer |
| [sampleDynamicReshape](sampleDynamicReshape) | C++ | ONNX | Digit Recognition With Dynamic Shapes In TensorRT |
| [sampleINT8](sampleINT8) | C++ | Caffe | Performing Inference In INT8 Using Custom Calibration |
| [sampleINT8API](sampleINT8API) | C++ | Caffe | Performing Inference In INT8 Precision |
| [sampleMNISTAPI](sampleMNISTAPI) | C++ | INetwork | Building a Simple MNIST Network Layer by Layer |
| [sampleOnnxMnistCoordConvAC](sampleOnnxMnistCoordConvAC) | C++ | ONNX | Implementing CoordConv with a custom plugin |
| [sampleIOFormats](sampleIOFormats) | C++ | Caffe | Specifying TensorRT I/O Formats |
| [sampleUffMNIST](sampleUffMNIST) | C++ | UFF | Import A TensorFlow Model And Run Inference |
| [sampleUffPluginV2Ext](sampleUffPluginV2Ext) | C++ | UFF | Adding A Custom Layer That Supports INT8 I/O To Your Network |
| [trtexec](trtexec) | C++ | All | TensorRT Command-Line Wrapper: trtexec |
| [engine_refit_mnist](python/engine_refit_mnist) | Python | INetwork | Refitting A TensorRT Engine |
| [int8_caffe_mnist](python/int8_caffe_mnist) | Python | Caffe | INT8 Calibration |
| [introductory_parser_samples](python/introductory_parser_samples) | Python | Any | Introduction To Importing Models Using TensorRT Parsers |
| [onnx_packnet](python/onnx_packnet) | Python | ONNX | TensorRT Inference Of ONNX Models With Custom Layers |
| [uff_custom_plugin](python/uff_custom_plugin) | Python | INetwork | Adding A Custom Layer To Your TensorFlow Network In TensorRT |

### 3. Application Samples
| Sample | Language | Format | Description |
|---|---|---|---|
| [sampleFasterRCNN](sampleFasterRCNN) | C++ | Caffe | Object Detection With Faster R-CNN |
| [sampleUffFasterRCNN](sampleUffFasterRCNN) | C++ | UFF | Object Detection With A TensorFlow FasterRCNN Network |
| [sampleGoogleNet](sampleGoogleNet) | C++ | Caffe | Building And Running GoogleNet In TensorRT |
| [sampleSSD](sampleSSD) | C++ | Caffe | Object Detection With SSD |
| [sampleUffSSD](sampleUffSSD) | C++ | UFF | Object Detection With A TensorFlow SSD Network |
| [sampleUffMaskRCNN](sampleUffMaskRCNN) | C++ | UFF | Object Detection And Instance Segmentation With MasK R-CNN Network |
| [efficientdet](python/efficientdet) | Python | ONNX | EfficientDet Object Detection with TensorRT |
| [efficientnet](python/efficientnet) | Python | ONNX | EfficientNet V1 and V2 Classification with TensorRT |
| [tensorflow_object_detection_api](python/tensorflow_object_detection_api) | Python | ONNX | TensorFlow Object Detection API Models in TensorRT |
| [uff_ssd](python/uff_ssd) | Python | UFF | Object Detection with SSD |
| [yolov3_onnx](python/yolov3_onnx) | Python | ONNX | Object Detection Using YOLOv3 With TensorRT ONNX Backend |

## Known Limitations

  - UFF converter and GraphSurgeon tools are only supported with Tensorflow 1.x
  - For the UFF samples, please use the [NVIDIA tf1 (Tensorflow 1.x)](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running) for running these tests or install Tensorflow 1.x manually.
