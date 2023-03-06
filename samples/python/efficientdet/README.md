# EfficientDet Object Detection in TensorRT

![efficientdet](https://drive.google.com/uc?export=view&id=1Le98wETvmKKj0fUKoCFLsld7o8QPJq9C)

These scripts help with conversion and execution of [Google EfficientDet](https://arxiv.org/abs/1911.09070) models with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt). This process is compatible with models trained through either Google AutoML or the TensorFlow Object Detection API.

## Contents
- [Changelog](#changelog)
- [Setup](#setup)
- [Model Conversion](#model-conversion)
  * [TensorFlow Saved Model](#tensorflow-saved-model)
  * [Create ONNX Graph](#create-onnx-graph)
  * [Build TensorRT Engine](#build-tensorrt-engine)
- [Inference](#inference)
  * [Inference in Python](#inference-in-python)
  * [Evaluate mAP Metric](#evaluate-map-metric)
  * [TF vs TRT Comparison](#tf-vs-trt-comparison)

## Changelog

- January 2022:
  - Added support for EfficientDet Lite and AdvProp models.
  - Added dynamic batch support.
  - Added mixed precision engine builder.
- July 2021:
  - Initial release.

## Setup

We recommend running these scripts on an environment with TensorRT >= 8.0.1 and TensorFlow >= 2.5.

Install TensorRT as per the [TensorRT Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). You will need to make sure the Python bindings for TensorRT are also installed correctly, these are available by installing the `python3-libnvinfer` and `python3-libnvinfer-dev` packages on your TensorRT download.

To simplify TensorRT and TensorFlow installation, use an [NGC TensorFlow Docker Image](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow), such as:

```bash
docker pull nvcr.io/nvidia/tensorflow:22.01-tf1-py3
```

Install all dependencies listed in `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

You will also need the latest `onnx_graphsurgeon` python module. If not already installed by TensorRT, you can install it manually by running:

```bash
pip3 install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

**NOTE:** Please make sure that the `onnx-graphsurgeon` module installed by pip is version >= 0.3.9.

Finally, you may want to clone the EfficientDet code from the [AutoML Repository](https://github.com/google/automl) to use some helper utilities from it. This exporter has been tested with commit [0b0ba5e](https://github.com/google/automl/tree/0b0ba5ebd0860edd939465fc4152da4ff9f79b44/efficientdet) from December 2021, so it may be a good idea to checkout the repository at that specific commit to avoid possible future incompatibilities:

```bash
git clone https://github.com/google/automl
cd automl
git checkout 0b0ba5e
```

## Model Conversion

The workflow to convert an EfficientDet model is basically TensorFlow → ONNX → TensorRT, and so parts of this process require TensorFlow to be installed. If you are performing this conversion to run inference on the edge, such as for NVIDIA Jetson devices, it might be easier to do the ONNX conversion on a PC first.

### TensorFlow Saved Model

The starting point of conversion is a TensorFlow saved model. This can be exported from your own trained models, or you can download a pre-trained model. This conversion script is compatible with three types of models:

1. EfficientDet models trained with the [AutoML](https://github.com/google/automl/tree/master/efficientdet) framework. Compatible with all "d0-7", "lite0-4" and "AdvProp" variations.
2. EfficientDet models trained with the [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) API (TFOD).
3. EfficientDet models pre-trained on COCO and downloaded from [TFHub](https://tfhub.dev/s?network-architecture=efficientdet).

#### 1. AutoML Models

If you are training your own model, you will need the training checkpoint. You can also download a pre-trained checkpoint from the "ckpt" links on the [AutoML Repository](https://github.com/google/automl/tree/master/efficientdet) README file, such as [this](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz).

This converter is compatible with all *efficientdet-d0* through *efficientdet-d7x*, and *efficientdet-lite0* through *efficientdet-lite4* model variations. This converter also works with the [AdvProp](https://github.com/google/automl/blob/master/efficientdet/Det-AdvProp.md) models. However, AdvProp models are trained with the `scale_range` hparam, which changes the expected input image value range, so you will need to adjust the preprocessor argument when creating the ONNX graph. More details on the corresponding section below.

The checkpoint directory should have a file structure such as this:

```
efficientdet-d0
├── model.data-00000-of-00001
├── model.index
└── model.meta
```

To export a saved model from here, clone and install the [AutoML](https://github.com/google/automl) repository, and run:

```bash
cd /path/to/automl/efficientdet
python3 model_inspect.py \
    --runmode saved_model \
    --model_name efficientdet-d0 \
    --ckpt_path /path/to/efficientdet-d0 \
    --saved_model_dir /path/to/saved_model
```

Where the `--model_name` argument is the network name corresponding to this checkpoint, usually between `efficientdet-d0` and `efficientdet-d7x`. The `--ckpt_path` points to the directory holding the checkpoint as described above. The TF saved model will be exported to the path given by `--saved_model_dir`.

> **Custom Image Size:** If your application requires inference at a different image resolution than the training input size, you can re-export the model for the exact size you require. To do so, export a saved model from checkpoint as shown above, but add an extra argument as: `--hparams 'image_size=1920x1280'`

#### 2. TFOD Models

You can download one of the pre-trained TFOD models from the [TF2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), such as:

```bash
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
```

When extracted, this package holds a directory named `saved_model` which holds the saved model ready for conversion.

However, if you are working with your own trained EfficientDet model from the TensorFlow Object Detection API, or if you need to re-export the saved model, you can do so from the training checkpoint. The downloaded package above also contains a pre-trained checkpoint. The structure is similar to this:

```
efficientdet_d0_coco17_tpu-32
├── checkpoint
│   ├── ckpt-0.data-00000-of-00001
│   └── ckpt-0.index
├── pipeline.config
└── saved_model
    └── saved_model.pb
```

To (re-)export a saved model from here, clone the TFOD API repository from [TF Models Repository](https://github.com/tensorflow/models) repository, and install it following the [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#installation). Then run:

```bash
cd /path/to/models/research/object_detection
python3 exporter_main_v2.py \
    --input_type image_tensor \
    --trained_checkpoint_dir /path/to/efficientdet_d0_coco17_tpu-32/checkpoint \
    --pipeline_config_path /path/to/efficientdet_d0_coco17_tpu-32/pipeline.config \
    --output_directory /path/to/export
```

Where `--trained_checkpoint_dir` and `--pipeline_config_path` point to the corresponding paths in the training checkpoint. On the path pointed by `--output_directory` you will then find the newly created saved model in a directory aptly named `saved_model`.

**NOTE:** TFOD EfficientDet models will have a slightly reduced throughput than their AutoML model counterparts. This is due to differences in the graph construction that TFOD makes use of.

#### 3. TFHub Models

You can download one of the pre-trained AutoML saved models from the [EfficientDet TFHub](https://tfhub.dev/s?network-architecture=efficientdet). Currently, only the efficientdet/d0 - d7 models are compatible with this converter. If you need to work with a pre-trained lite model, please follow the AutoML checkpoint route above.

Download a model from TFHub, such as:

```bash
wget https://storage.googleapis.com/tfhub-modules/tensorflow/efficientdet/d0/1.tar.gz
```

The contents of this package, when extracted, will hold a saved model ready for conversion.

### Create ONNX Graph

To generate an ONNX model file, first find the input size that corresponds to the model you're converting:

| **Model**           | **Input Size** |
| --------------------|----------------|
| efficientdet-d0     | 512,512        |
| efficientdet-d1     | 640,640        |
| efficientdet-d2     | 768,768        |
| efficientdet-d3     | 896,896        |
| efficientdet-d4     | 1024,1024      |
| efficientdet-d5     | 1280,1280      |
| efficientdet-d6     | 1280,1280      |
| efficientdet-d7     | 1536,1536      |
| efficientdet-d7x    | 1536,1536      |
| efficientdet-lite0  | 320,320        |
| efficientdet-lite1  | 384,384        |
| efficientdet-lite2  | 448,448        |
| efficientdet-lite3  | 512,512        |
| efficientdet-lite3x | 640,640        |
| efficientdet-lite4  | 640,640        |

If you've re-exported the model with a custom image size, then of course use that. With the correct input size and the TF saved model ready to be converted, run:

```bash
python3 create_onnx.py \
    --input_size 512,512 \
    --saved_model /path/to/saved_model \
    --onnx /path/to/model.onnx
```

This will create the file `model.onnx` which is ready to convert to TensorRT.

The script has a few optional arguments, including:

* `--input_format [NHWC,NCHW]` allows switching between NHWC (default) and NCHW data format modes. If your data source is in NCHW format, you may want to select this mode to avoid extra transposes.
* `--nms_threshold [...]` allows overriding the default NMS score threshold parameter, as the runtime latency of the NMS plugin is sensitive to this value. It's a good practice to set this value as high as possible, while still fulfilling your application requirements, to reduce inference latency.
* `--preprocessor [imagenet,scale_range]` allows switching between two possible image preprocessing methods. Most EfficientDet models use the `imagenet` method, which this argument defaults to, and corresponds to standard ImageNet mean subtraction and standard deviation normalization. The `scale_range` method instead normalizes the image to a range of [-1,+1]. Please use this method only when converting the **AdvProp** pre-trained checkpoints, as they were created with this preprocessor operation.

Optionally, you may wish to visualize the resulting ONNX graph with a tool such as [Netron](https://netron.app/).

![netron](https://drive.google.com/uc?export=view&id=1m9zRbvNtlbftN7P46dtOLPbcwEbz4XwS)

The input to the graph is a `float32` tensor with the selected input shape, containing RGB pixel data in the range of 0 to 255. Normalization, mean subtraction and scaling will be performed inside the EfficientDet graph, so it is not required to further pre-process the input data.

The outputs of the graph are the same as the outputs of the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin.

### Build TensorRT Engine

It is possible to build the TensorRT engine directly with `trtexec` using the ONNX graph generated in the previous step. You can do so by running:

```bash
trtexec \
    --onnx=/path/to/model.onnx \
    --saveEngine=/path/to/engine.trt \
    --optShapes=input:$INPUT_SHAPE \
    --workspace=1024
```

Where `$INPUT_SHAPE` defines the input spec to build the engine with, e.g. `--optShapes=input:8x512x512x3`. Other common `trtexec` functionality for lower precision modes or other options will also work as expected.

However, the script `build_engine.py` is also provided in this repository for convenience, as it has been tailored to EfficientDet engine building and INT8 calibration. Run `python3 build_engine.py --help` for details on available settings.

#### FP16 Precision

To build the TensorRT engine file with FP16 precision, run:

```bash
python3 build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision fp16
```

The file `engine.trt` will be created, which can now be used to infer with TensorRT.

For best results, make sure no other processes are using the GPU during engine build, as it may affect the optimal tactic selection process.

#### INT8 Precision

To build and calibrate an engine for INT8 precision, run:

```bash
python3 build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision int8 \
    --calib_input /path/to/calibration/images \
    --calib_cache /path/to/calibration.cache
```

Where `--calib_input` points to a directory with several thousands of images. For example, this could be a subset of the training or validation datasets that were used for the model. It's important that this data represents the runtime data distribution relatively well, therefore, the more images that are used for calibration, the better accuracy that will be achieved in INT8 precision. For models trained for the [COCO dataset](https://cocodataset.org/#home), we have found that 5,000 images gives a good result.

The `--calib_cache` is optional, and it controls where the calibration cache file will be written to. This is useful to keep a cached copy of the calibration results. Next time you need to build an int8 engine for the same network, if this file exists, the builder will skip the calibration step and use the cached values instead.

#### Mixed Precision (Experimental)

Mixed precision is a custom mode that pins some key layers to FP16, while the rest of the network is converted at INT8 precision. The purpose of this mode is to balance accuracy and throughput. It's experimental and is given here to show one possible way of balancing achieved accuracy according to an application's latency budget. This mode has been tuned for COCO pre-trained models. For other datasets, you may need to adjust the layers to pin.

Some sample results of using this mode:

| **Model / Precision**   | **Latency** | **COCO mAP** |
| ------------------------|-------------|--------------|
| efficientdet-d0 / fp32  | 3.25 ms     | 0.341        |
| efficientdet-d0 / fp16  | 2.27 ms     | 0.341        |
| efficientdet-d0 / mixed | **1.75 ms** | **0.320**    |
| efficientdet-d0 / int8  | 1.63 ms     | 0.299        |

To use mixed precision mode, follow the same instructions as for building and calibrating an INT8 engine as given above, but using the argument `--precision mixed` instead.

#### Static and Dynamic Batch Size

By default, `build_engine.py` creates a static batch size 1 engine. To build with a different static batch size, set the `--batch_size` argument accordingly:

```bash
python3 build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --batch_size 8
```

You can also build an engine with a dynamic batch size. To do so, select a minimum and maximum batch size, as well as an optimal batch size for which TensorRT will fine tune the engine performance best. These batch sizes should be given via the argument `--dynamic_batch_size MIN,OPT,MAX`, such as:

```bash
python3 build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --dynamic_batch_size 1,16,32
```

#### Benchmark Engine

Optionally, you can obtain execution timing information for the built engine by using the `trtexec` utility, as:

```bash
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

```bash
python3 infer.py \
    --engine /path/to/engine.trt \
    --input /path/to/images \
    --output /path/to/output
```

Where the input path can be either a single image file, or a directory of jpg/png/bmp images.

The detection results will be written out to the specified output directory, consisting of a visualization image, and a tab-separated results file for each input image processed.

![infer](https://drive.google.com/uc?export=view&id=1ZzTHizLx65t_cJcIIflnzXA5yxCYsQz6)

> *This example is generated with a TensorRT engine for the pre-trained AutoML EfficientDet-D0 model re-exported with a custom image size of 1920x1080 as described above. The engine uses an NMS score threshold of 0.4. This is the same [sample image](https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png) and model parameters as used in the AutoML [inference tutorial](https://github.com/google/automl/blob/master/efficientdet/tutorial.ipynb) to produce this [sample TensorFlow inference image](https://github.com/google/automl/blob/master/efficientdet/g3doc/street.jpg).*

### Evaluate mAP Metric

Given a validation dataset (such as [COCO val2017 data](http://images.cocodataset.org/zips/val2017.zip)) and ground truth annotations (such as [COCO instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)), you can get the mAP metrics for the built TensorRT engine. This will use the mAP metrics calculation script from the [AutoML](https://github.com/google/automl) repository.

```bash
python3 eval_coco.py \
    --engine /path/to/engine.trt \
    --input /path/to/coco/val2017 \
    --annotations /path/to/coco/annotations/instances_val2017.json \
    --automl_path /path/to/automl
```

Where the `--automl_path` argument points to the root of the AutoML repository.

The mAP metric is sensitive to the NMS score threshold used, as using a high threshold will reduce the model recall, resulting in a lower mAP value. Ideally, mAP should be measured with a threshold of 0, but such a low value will impact the runtime latency of the EfficientNMS plugin. It may be a good idea to build separate TensorRT engines for different purposes. That is, one engine with a low threshold (like 0) dedicated for mAP validation, and another engine with your application specific threshold (like 0.4) for deployment. This is why we keep the NMS threshold as a configurable parameter in the `create_onnx.py` script.

### TF vs TRT Comparison

To compare how the TensorRT detections match the original TensorFlow model results, you can run:

```bash
python3 compare_tf.py \
    --engine /path/to/engine.trt \
    --saved_model /path/to/saved_model \
    --input /path/to/images \
    --nms_threshold 0.4 \
    --output /path/to/output
```

This script will process the images found in the given input path through both TensorFlow and TensorRT using the corresponding saved model and engine. It will then write to the output path a set of visualization images showing the inference results of both frameworks for visual qualitative comparison.

`--nms_threshold` overrides the score threshold for the NMS operation if it is higher than the threshold in the model/engine. For better visualization, `--nms_threshold 0.4` is used here for filtering out the noisy detections.

If you run this on COCO val2017 images, you may also add the parameter `--annotations /path/to/coco/annotations/instances_val2017.json` to further compare against COCO ground truth annotations.

![compare_tf](https://drive.google.com/uc?export=view&id=1zgh_RbYX6RWzu7nKLCcSzy60VPiQROZJ)
