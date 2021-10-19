# EfficientDet Object Detection in TensorRT

![efficientdet](https://drive.google.com/uc?export=view&id=1Le98wETvmKKj0fUKoCFLsld7o8QPJq9C)

These scripts help with conversion and execution of [Google EfficientDet](https://arxiv.org/abs/1911.09070) models with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt). This process is compatible with models trained through either Google AutoML or the TensorFlow Object Detection API.

## Contents
- [Setup](#setup)
- [Model Conversion](#model-conversion)
  * [TensorFlow Saved Model](#tensorflow-saved-model)
  * [Create ONNX Graph](#create-onnx-graph)
  * [Build TensorRT Engine](#build-tensorrt-engine)
- [Inference](#inference)
  * [Inference in Python](#inference-in-python)
  * [Evaluate mAP Metric](#evaluate-map-metric)
  * [TF vs TRT Comparison](#tf-vs-trt-comparison)

## Setup

For best results, we recommend running these scripts on an environment with TensorRT >= 8.0.1 and TensorFlow 2.5.

Install TensorRT as per the [TensorRT Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). You will need to make sure the Python bindings for TensorRT are also installed correctly, these are available by installing the `python3-libnvinfer` and `python3-libnvinfer-dev` packages on your TensorRT download.

Install all dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

You will also need the latest `onnx_graphsurgeon` python module. If not already installed by TensorRT, you can install it manually by running:

```
pip install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

**NOTE:** Please make sure that the `onnx-graphsurgeon` module installed by pip is version >= 0.3.9.

Finally, you may want to clone the EfficientDet code from the [AutoML Repository](https://github.com/google/automl) to use some helper utilities from it:

```
git clone https://github.com/google/automl
```

## Model Conversion

The workflow to convert an EfficientDet model is basically TensorFlow → ONNX → TensorRT, and so parts of this process require TensorFlow to be installed. If you are performing this conversion to run inference on the edge, such as for NVIDIA Jetson devices, it might be easier to do the ONNX conversion on a PC first.

### TensorFlow Saved Model

The starting point of conversion is a TensorFlow saved model. This can be exported from your own trained models, or you can download a pre-trained model. This conversion script is compatible with two types of models:

1. EfficientDet models trained with the [AutoML](https://github.com/google/automl/tree/master/efficientdet) framework. 
2. EfficientDet models trained with the [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) API (TFOD).

#### 1. AutoML Models

You can download one of the pre-trained AutoML saved models from the [EfficientDet TFHub](https://tfhub.dev/s?network-architecture=efficientdet), such as:

```
wget https://storage.googleapis.com/tfhub-modules/tensorflow/efficientdet/d0/1.tar.gz
```

The contents of this package, when extracted, will hold a saved model ready for conversion.

**NOTE:** Some saved models in TFHub may give problems with ONNX conversion. If so, please download the original checkpoint and export the saved model manually as per the instructions below.

Alternatively, if you are training your own model, or if you need to re-export the saved model manually, you will need the training checkpoint (or a pre-trained "ckpt" from the [AutoML Repository](https://github.com/google/automl/tree/master/efficientdet) such as [this](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz)). The checkpoint directory should have a file structure such as this:

```
efficientdet-d0
├── model.data-00000-of-00001
├── model.index
└── model.meta
```

To export a saved model from here, clone and install the [AutoML](https://github.com/google/automl) repository, and run:

```
cd /path/to/automl/efficientdet
python model_inspect.py \
    --runmode saved_model \
    --model_name efficientdet-d0 \
    --ckpt_path /path/to/efficientdet-d0 \
    --saved_model_dir /path/to/saved_model
```

Where the `--model_name` argument is the network name corresponding to this checkpoint, usually between `efficientdet-d0` and `efficientdet-d7x`. The `--ckpt_path` points to the directory holding the checkpoint as described above. The TF saved model will be exported to the path given by `--saved_model_dir`.

> **Custom Image Size:** If your application requires inference at a different image resolution than the training input size, you can re-export the model for the exact size you require. To do so, export a saved model from checkpoint as shown above, but add an extra argument as: `--hparams 'image_size=1920x1280'`

#### 2. TFOD Models

You can download one of the pre-trained TFOD models from the [TF2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), such as:

```
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
```

When extracted, this package holds a directory named `saved_model` which holds the saved model ready for conversion.

However, if you are working with your own trained model, or if you need to re-export the saved model, you can do so from the training checkpoint. The downloaded package above also contains a pre-trained checkpoint. The structure is similar to this:

```
efficientdet_d0_coco17_tpu-32
├── checkpoint
│   ├── ckpt-0.data-00000-of-00001
│   └── ckpt-0.index
├── pipeline.config
└── saved_model
    └── saved_model.pb
```

To (re-)export a saved model from here, clone and install the TFOD API from the [TF Models Repository](https://github.com/tensorflow/models) repository, and run: 

```
cd /path/to/models/research/object_detection
python exporter_main_v2.py \
    --input_type image_tensor \
    --trained_checkpoint_dir /path/to/efficientdet_d0_coco17_tpu-32/checkpoint \
    --pipeline_config_path /path/to/efficientdet_d0_coco17_tpu-32/pipeline.config \
    --output_directory /path/to/export
```

Where `--trained_checkpoint_dir` and `--pipeline_config_path` point to the corresponding paths in the training checkpoint. On the path pointed by `--output_directory` you will then find the newly created saved model in a directory aptly named `saved_model`.

**NOTE:** TFOD EfficientDet models will have a slightly reduced throughput than their AutoML model counterparts. This is due to differences in the graph construction that TFOD makes use of.

### Create ONNX Graph

To generate an ONNX model file, first find the input shape that corresponds to the model you're converting:

| **Model**        | **Input Shape** |
| -----------------|-----------------|
| EfficientDet D0  | N,512,512,3     |
| EfficientDet D1  | N,640,640,3     |
| EfficientDet D2  | N,768,768,3     |
| EfficientDet D3  | N,896,896,3     |
| EfficientDet D4  | N,1024,1024,3   |
| EfficientDet D5  | N,1280,1280,3   |
| EfficientDet D6  | N,1280,1280,3   |
| EfficientDet D7  | N,1536,1536,3   |
| EfficientDet D7x | N,1536,1536,3   |

Where **N** is the batch size you would like to run inference at, such as `8,512,512,3` for a batch size of 8. If you exported the saved model with a custom input image size, you should use that specific shape instead.

The ONNX conversion process supports both `NHWC` and `NCHW` input formats, so if your input source is an `NCHW` data format, you can use the corresponding input shape, i.e. `1,512,512,3` -> `1,3,512,512`.

With the correct input shape selected, and the TF saved model ready to be converted, run:

```
python create_onnx.py \
    --input_shape '1,512,512,3' \
    --saved_model /path/to/saved_model \
    --onnx /path/to/model.onnx
```

This will create the file `model.onnx` which is ready to convert to TensorRT.

The script has a few optional arguments, including:

* `--nms_threshold [...]` allows overriding the default NMS score threshold parameter, as the runtime latency of the NMS plugin is sensitive to this value. It's a good practice to set this value as high as possible, while still fulfilling your application requirements, to reduce inference latency.
* `--legacy_plugins` allows falling back to older plugins on systems where a version lower than TensorRT 8.0.1 is installed. This will result in substantially slower inference times however, but is provided for compatibility.

Optionally, you may wish to visualize the resulting ONNX graph with a tool such as [Netron](https://netron.app/).

![netron](https://drive.google.com/uc?export=view&id=1m9zRbvNtlbftN7P46dtOLPbcwEbz4XwS)

The input to the graph is a `float32` tensor with the selected input shape, containing RGB pixel data in the range of 0 to 255. Normalization, mean subtraction and scaling will be performed inside the EfficientDet graph, so it is not required to further pre-process the input data.

The outputs of the graph are the same as the outputs of the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin. If the ONNX graph was created with `--legacy_plugins` for TensorRT 7 compatibility, the outputs will correspond to those of the [BatchedNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin) plugin instead. 

### Build TensorRT Engine

It is possible to build the TensorRT engine directly with `trtexec` using the ONNX graph generated in the previous step. However, the script `build_engine.py` is provided for convenience, as it has been tailored to EfficientDet engine building and calibration. Run `python build_engine.py --help` for details on available settings.

#### FP16 Precision

To build the TensorRT engine file with FP16 precision, run:

```
python build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision fp16
```

The file `engine.trt` will be created, which can now be used to infer with TensorRT.

For best results, make sure no other processes are using the GPU during engine build, as it may affect the optimal tactic selection process.

#### INT8 Precision

To build and calibrate an engine for INT8 precision, run:

```
python build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision int8 \
    --calib_input /path/to/calibration/images \
    --calib_cache /path/to/calibration.cache
```

Where `--calib_input` points to a directory with several thousands of images. For example, this could be a subset of the training or validation datasets that were used for the model. It's important that this data represents the runtime data distribution relatively well, therefore, the more images that are used for calibration, the better accuracy that will be achieved in INT8 precision. For models trained for the [COCO dataset](https://cocodataset.org/#home), we have found that 5,000 images gives a good result.

The `--calib_cache` controls where the calibration cache file will be written to. This is useful to keep a cached copy of the calibration results. Next time you need to build the engine for the same network, if this file exists, it will skip the calibration step and use the cached values instead.

#### Benchmark Engine

Optionally, you can obtain execution timing information for the built engine by using the `trtexec` utility, as:

```
trtexec \
    --loadEngine=/path/to/engine.trt \
    --useCudaGraph --noDataTransfers \
    --iterations=100 --avgRuns=100
```

If it's not already in your `$PATH`, the `trtexec` binary is usually found in `/usr/src/tensorrt/bin/trtexec`, depending on your TensorRT installation method.

An inference benchmark will run, with GPU Compute latency times printed out to the console. Depending on your environment, you should see something similar to:

```
GPU Compute Time: min = 1.55835 ms, max = 1.91591 ms, mean = 1.58719 ms, median = 1.578 ms, percentile(99%) = 1.90668 ms
```

## Inference

For optimal performance, inference should be done in a C++ application that takes advantage of CUDA Graphs to launch the inference request. Alternatively, the TensorRT engine built with this process can also be executed through either [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) or [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

However, for convenience, a python inference script is provided here for quick testing of the built TensorRT engine.

### Inference in Python

To perform object detection on a set of images with TensorRT, run:

```
python infer.py \
    --engine /paht/to/engine.trt \
    --input /path/to/images \
    --output /path/to/output
```

Where the input path can be either a single image file, or a directory of jpg/png/bmp images.

The detection results will be written out to the specified output directory, consisting of a visualization image, and a tab-separated results file for each input image processed.

![infer](https://drive.google.com/uc?export=view&id=1ZzTHizLx65t_cJcIIflnzXA5yxCYsQz6)

> *This example is generated with a TensorRT engine for the pre-trained AutoML EfficientDet-D0 model re-exported with a custom image size of 1920x1080 as described above. The engine uses an NMS score threshold of 0.4. This is the same [sample image](https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png) and model parameters as used in the AutoML [inference tutorial](https://github.com/google/automl/blob/master/efficientdet/g3doc/street.jpg).*

### Evaluate mAP Metric

Given a validation dataset (such as [COCO val2017 data](http://images.cocodataset.org/zips/val2017.zip)) and ground truth annotations (such as [COCO instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)), you can get the mAP metrics for the built TensorRT engine. This will use the mAP metrics calculation script from the [AutoML](https://github.com/google/automl) repository.

```
python eval_coco.py \
    --engine /path/to/engine.trt \
    --input /path/to/coco/val2017 \
    --annotations /path/to/coco/annotations/instances_val2017.json \
    --automl_path /path/to/automl
```

Where the `--automl_path` argument points to the root of the AutoML repository.

The mAP metric is sensitive to the NMS score threshold used, as using a high threshold will reduce the model recall, resulting in a lower mAP value. Ideally, mAP should be measured with a threshold of 0, but such a low value will impact the runtime latency of the EfficientNMS plugin. It may be a good idea to build separate TensorRT engines for different purposes. That is, one engine with a low threshold (like 0) dedicated for mAP validation, and another engine with your application specific threshold (like 0.4) for deployment. This is why we keep the NMS threshold as a configurable parameter in the `create_onnx.py` script.

### TF vs TRT Comparison

To compare how the TensorRT detections match the original TensorFlow model results, you can run:

```
python compare_tf.py \
    --engine /path/to/engine.trt \
    --saved_model /path/to/saved_model \
    --input /path/to/images \
    --output /path/to/output
```

This script will process the images found in the given input path through both TensorFlow and TensorRT using the corresponding saved model and engine. It will then write to the output path a set of visualization images showing the inference results of both frameworks for visual qualitative comparison.

If you run this on COCO val2017 images, you may also add the parameter `--annotations /path/to/coco/annotations/instances_val2017.json` to further compare against COCO ground truth annotations.

![compare_tf](https://drive.google.com/uc?export=view&id=1zgh_RbYX6RWzu7nKLCcSzy60VPiQROZJ)
