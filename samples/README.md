# TensorRT Samples

## Contents

| Sample | Language | Format | Description |
|---|---|---|---|
| [sampleAlgorithmSelector](opensource/sampleAlgorithmSelector) | C++ | Caffe | Algorithm Selection API usage |
| [sampleCharRNN](opensource/sampleCharRNN) | C++ | INetwork | Building An RNN Network Layer By Layer |
| [sampleDynamicReshape](opensource/sampleDynamicReshape) | C++ | ONNX | Digit Recognition With Dynamic Shapes In TensorRT |
| [sampleFasterRCNN](opensource/sampleFasterRCNN) | C++ | Caffe | Object Detection With Faster R-CNN |
| [sampleGoogleNet](opensource/sampleGoogleNet) | C++ | Caffe | Building And Running GoogleNet In TensorRT |
| [sampleINT8](opensource/sampleINT8) | C++ | Caffe | Building And Running GoogleNet In TensorRT |
| [sampleINT8API](opensource/sampleINT8API) | C++ | Caffe | Performing Inference In INT8 Precision |
| [sampleMLP](opensource/sampleMLP) | C++ | INetwork | “Hello World” For Multilayer Perceptron (MLP) |
| [sampleMNIST](opensource/sampleMNIST) | C++ | Caffe | “Hello World” For TensorRT |
| [sampleMNISTAPI](opensource/sampleMNISTAPI) | C++ | INetwork | Building a Simple MNIST Network Layer by Layer |
| [sampleNMT](opensource/sampleNMT) | C++ | INetwork | Neural Machine Translation Using A seq2seq Model |
| [sampleOnnxMNIST](opensource/sampleOnnxMNIST) | C++ | ONNX | “Hello World” For TensorRT With ONNX |
| [sampleOnnxMnistCoordConvAC](opensource/sampleOnnxMnistCoordConvAC) | C++ | ONNX | Implementing CoordConv with a custom plugin |
| [sampleReformatFreeIO](opensource/sampleReformatFreeIO) | C++ | Caffe | Specifying I/O Formats Via Reformat-Free-I/O API |
| [sampleSSD](opensource/sampleSSD) | C++ | Caffe | Object Detection With SSD |
| [sampleUffFasterRCNN](opensource/sampleUffFasterRCNN) | C++ | UFF | Object Detection With A TensorFlow FasterRCNN Network |
| [sampleUffMNIST](opensource/sampleUffMNIST) | C++ | UFF | Import A TensorFlow Model And Run Inference |
| [sampleUffMaskRCNN](opensource/sampleUffMaskRCNN) | C++ | UFF | Object Detection And Instance Segmentation With MasK R-CNN Network |
| [sampleUffPluginV2Ext](opensource/sampleUffPluginV2Ext) | C++ | UFF | Adding A Custom Layer That Supports INT8 I/O To Your Network |
| [sampleUffSSD](opensource/sampleUffSSD) | C++ | UFF | Object Detection With A TensorFlow SSD Network |
| [trtexec](opensource/trtexec) | C++ | All | TensorRT Command-Line Wrapper: trtexec |
| [efficientdet](python/efficientdet) | Python | ONNX | EfficientDet Object Detection with TensorRT |
| [efficientnet](python/efficientnet) | Python | ONNX | EfficientNet V1 and V2 Classification with TensorRT |
| [end_to_end_tensorflow_mnist](python/end_to_end_tensorflow_mnist) | Python | UFF | “Hello World” For TensorRT Using TensorFlow |
| [engine_refit_mnist](python/engine_refit_mnist) | Python | INetwork | Refitting A TensorRT Engine |
| [int8_caffe_mnist](python/int8_caffe_mnist) | Python | Caffe | INT8 Calibration |
| [introductory_parser_samples](python/introductory_parser_samples) | Python | Any | Introduction To Importing Models Using TensorRT Parsers |
| [network_api_pytorch_mnist](python/network_api_pytorch_mnist) | Python | INetwork | “Hello World” For TensorRT |
| [onnx_packnet](python/onnx_packnet) | Python | ONNX | TensorRT Inference Of ONNX Models With Custom Layers |
| [uff_custom_plugin](python/uff_custom_plugin) | Python | INetwork | Adding A Custom Layer To Your TensorFlow Network In TensorRT |
| [uff_ssd](python/uff_ssd) | Python | UFF | Object Detection with SSD |
| [yolov3_onnx](python/yolov3_onnx) | Python | ONNX | Object Detection Using YOLOv3 With TensorRT ONNX Backend |


## Known Limitations

  - UFF converter and GraphSurgeon tools are only supported with Tensorflow 1.x
  - For the UFF samples, please use the [NVIDIA tf1 (Tensorflow 1.x)](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running) for running these tests or install Tensorflow 1.x manually.
