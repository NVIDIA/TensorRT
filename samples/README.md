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
| [sampleCharRNN](sampleCharRNN) | C++ | INetwork | Building An RNN Network Layer By Layer |
| [sampleDynamicReshape](sampleDynamicReshape) | C++ | ONNX | Digit Recognition With Dynamic Shapes In TensorRT |
| [sampleEditableTimingCache](sampleEditableTimingCache) | C++ | INetwork | Create a deterministic build using editable timing cache |
| [sampleINT8API](sampleINT8API) | C++ | ONNX | Performing Inference In INT8 Precision |
| [sampleNamedDimensions](sampleNamedDimensions) | C++ | ONNX | Working with named input dimensions |
| [sampleNonZeroPlugin](sampleNonZeroPlugin) | C++ | INetwork | Adding plugin with data-dependent output shapes |
| [sampleOnnxMnistCoordConvAC](sampleOnnxMnistCoordConvAC) | C++ | ONNX | Implementing CoordConv with a custom plugin |
| [sampleIOFormats](sampleIOFormats) | C++ | ONNX | Specifying TensorRT I/O Formats |
| [sampleProgressMonitor](sampleProgressMonitor) | C++ | ONNX | Progress Monitor API usage |
| [trtexec](trtexec) | C++ | All | TensorRT Command-Line Wrapper: trtexec |
| [engine_refit_onnx_bidaf](python/engine_refit_onnx_bidaf) | Python | ONNX | refitting an engine built from an ONNX model via parsers. |
| [introductory_parser_samples](python/introductory_parser_samples) | Python | ONNX | Introduction To Importing Models Using TensorRT Parsers |
| [onnx_packnet](python/onnx_packnet) | Python | ONNX | TensorRT Inference Of ONNX Models With Custom Layers |
| [simpleProgressMonitor](python/simple_progress_monitor) | Python | ONNX | Progress Monitor API usage |
| [python_plugin](python/python_plugin) | Python | INetwork/ONNX | Python-based TRT plugins |
| [non_zero_plugin](python/non_zero_plugin) | Python | INetwork/ONNX | Python-based TRT plugin for NonZero op |

### 3. Application Samples
| Sample | Language | Format | Description |
|---|---|---|---|
| [detectron2](python/detectron2) | Python | ONNX | Support for Detectron 2 Mask R-CNN R50-FPN 3x model in TensorRT |
| [[DEPRECATED] efficientdet](python/efficientdet) | Python | ONNX | EfficientDet Object Detection with TensorRT |
| [[DEPRECATED] tensorflow_object_detection_api](python/tensorflow_object_detection_api) | Python | ONNX | TensorFlow Object Detection API Models in TensorRT |
| [[DEPRECATED] yolov3_onnx](python/yolov3_onnx) | Python | ONNX | Object Detection Using YOLOv3 With TensorRT ONNX Backend |

## Preparing sample data

Many samples require the TensorRT sample data package. If not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers), download and extract it:

1. Download the sample data from [TensorRT GitHub Releases](https://github.com/NVIDIA/TensorRT/releases).

2. Extract and set up the data:
    ```bash
    unzip tensorrt_sample_data_xxx.zip
    mkdir -p /usr/src/tensorrt/data
    cp -r tensorrt_sample_data_*/* /usr/src/tensorrt/data/
    export TRT_DATADIR=/usr/src/tensorrt/data
    ```

After extraction, the data directory structure should be:
```
$TRT_DATADIR/
├── char-rnn/
├── int8_api/
├── mnist/
└── resnet50/
```
