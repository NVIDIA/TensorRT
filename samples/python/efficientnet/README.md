# EfficientNet V1 and V2 in TensorRT

These scripts help with conversion and execution of Google [EfficientNet V1](https://arxiv.org/abs/1905.11946) and [EfficientNet V2](https://arxiv.org/abs/2104.00298) models with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

## Contents
- [Setup](#setup)
- [Model Conversion](#model-conversion)
  * [TensorFlow Saved Model](#tensorflow-saved-model)
  * [Create ONNX Graph](#create-onnx-graph)
  * [Build TensorRT Engine](#build-tensorrt-engine)
  * [Benchmark TensorRT Engine](#benchmark-tensorrt-engine)
- [Inference](#inference)
  * [Input Preprocessing](#input-preprocessing)
  * [Inference in Python](#inference-in-python)
  * [Validate against Ground Truth](#validate-against-ground-truth)
  * [Compare against TensorFlow](#compare-against-tensorflow)

## Setup

For best results, we recommend running these scripts on an environment with TensorRT >= 8.0.1 and TensorFlow 2.5.

Install TensorRT as per the [TensorRT Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). You will need to make sure the Python bindings for TensorRT are also installed correctly, these are available by installing the `python3-libnvinfer` and `python3-libnvinfer-dev` packages on your TensorRT download.

Make sure all other packages listed in `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

You will also need the latest `onnx_graphsurgeon` python module. If not already installed by TensorRT, you can install it manually by running:

```bash
pip3 install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

## Model Conversion

The workflow to convert an EfficientNet model is basically TensorFlow → ONNX → TensorRT, and so parts of this process require TensorFlow to be installed. If you are performing this conversion to run inference on the edge, such as for NVIDIA Jetson devices, it might be easier to do the ONNX conversion on a PC first.

### TensorFlow Saved Model

The starting point of conversion is a TensorFlow saved model. This can be exported from your own trained models, or you can download a pre-trained model. This conversion script is compatible with two types of models:

1. EfficientNet V1 models trained with the [TensorFlow TPU Models](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) framework.
2. EfficientNet V2 models trained with the [AutoML](https://github.com/google/automl/tree/master/efficientnetv2) framework.

#### 1. EfficientNet V1

You can download one of the pre-trained saved models from the [EfficientNet TFHub](https://tfhub.dev/google/collections/efficientnet), such as:

```bash
wget https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/b0/classification/1.tar.gz
```

The contents of this package, when extracted, will hold a saved model ready for conversion.

Alternatively, if you are training your own model, or if you need to re-export the saved model manually, you will need the training checkpoint (or a pre-trained "ckpt" from the [EfficientNet Repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) such as [this](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b0.tar.gz)).

To export a saved model from the checkpoint, clone and install the [TensorFlow TPU Models](https://github.com/tensorflow/tpu) repository, and run:

```bash
cd /path/to/tpu/models/official/efficientnet
python3 export_model.py \
    --ckpt_dir /path/to/efficientnet-b0 \
    --image_size 224 \
    --model_name efficientnet-b0 \
    --output_tflite /dev/null \
    --noquantize \
    --output_saved_model_dir /path/to/saved_model
```

Adapt `--image_size` and `--model_name` according to the checkpoint model being used. The `--ckpt_dir` argument points to the directory holding the checkpoint as described above. The TF saved model will be exported to the path given by `--output_saved_model_dir`.

#### 2. EfficientNet V2

At the time of this writing, there exist no EfficientNet V2 saved models in TFHub yet. So you will need to download a pre-trained checkpoint, or use your own trained model of course.

To do so, you will need your training checkpoint (or a pre-trained "ckpt" from the [EfficientNet V2 Repository](https://github.com/google/automl/tree/master/efficientnetv2) such as [this](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s.tgz)):

```bash
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s.tgz
```

To export a saved model from here, clone and install the [AutoML](https://github.com/google/automl) repository, and run:

```bash
cd /path/to/automl/efficientnetv2
python3 infer.py \
    --mode tf2bm \
    --model_name efficientnetv2-s \
    --model_dir ../../efficientnetv2-s/ \
    --export_dir ../../efficientnetv2-s/saved_model
```

Where you should adapt `--model_name` to the corresponding model for the checkpoint used. The `--model_dir` argument should point to the downloaded or trained checkpoint as described above. The exported saved model will then be available in the directory pointed by the `--export_dir` argument.

### Create ONNX Graph

To generate an ONNX model file, find the saved model as described above, select a batch size and input size, and run:

```bash
python3 create_onnx.py \
    --saved_model /path/to/saved_model \
    --onnx /path/to/model.onnx \
    --batch_size 1 \
    --input_size 384
```

You may need to adapt the argument `--input_size` to explicitly define the exact input image dimensions to use in the graph. Consult the model definitions in the corresponding training system, to find the expected input size for the model you are working with.

This will create the file `model.onnx` which is ready to convert to TensorRT.

Optionally, you may wish to visualize the resulting ONNX graph with a tool such as [Netron](https://netron.app/).

### Build TensorRT Engine

It is possible to build the TensorRT engine directly with `trtexec` using the ONNX graph generated in the previous step. However, the script `build_engine.py` is provided for convenience, as it has been tailored to EfficientNet engine building and calibration. Run `python3 build_engine.py --help` for details on available settings.

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
    --calib_cache /path/to/calibration.cache \
    --calib_preprocessor V2
```

Where `--calib_input` points to a directory with several thousands of images. For example, this could be a subset of the training or validation datasets that were used for the model. It's important that this data represents the runtime data distribution relatively well, therefore, the more images that are used for calibration, the better accuracy that will be achieved in INT8 precision. For ImageNet networks, we have found that 25,000 images gives a good result.

The `--calib_cache` argument controls where the calibration cache file will be written to. This is useful to keep a cached copy of the calibration results. Next time you need to build the engine for the same network, if this file exists, it will skip the calibration step and use the cached values instead.

Finally, the `--calib_preprocessor` option sets the preprocessing algorithm to apply on calibration images. Please refer to the [Input Preprocessing](#input-preprocessing) section below for more details.

Run `python3 build_engine.py --help` for additional build options.

### Benchmark TensorRT Engine

Optionally, you can obtain execution timing information for the built engine by using the `trtexec` utility, as:

```bash
trtexec \
    --loadEngine=/path/to/engine.trt \
    --useCudaGraph --noDataTransfers \
    --iterations=100 --avgRuns=100
```

If it's not already in your `$PATH`, the `trtexec` binary is usually found in `/usr/src/tensorrt/bin/trtexec`, depending on your TensorRT installation method.

An inference benchmark will run, with GPU Compute latency times printed out to the console. Depending on the version of TensorRT, you should see something similar to:

```
GPU Compute Time: min = 1.79895 ms, max = 1.9209 ms, mean = 1.80589 ms, median = 1.80493 ms, percentile(99%) = 1.81396 ms
```

## Inference

For optimal performance, inference should be done in a C++ application that takes advantage of CUDA Graphs to launch the inference request. Alternatively, the TensorRT engine built with this process can also be executed through either [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) or [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

However, for convenience, a python inference script is provided here for quick testing of the built TensorRT engine.

### Input Preprocessing

An important concept for computer vision models is the preprocessing applied to an image before feeding it to the classifier network. The various EfficientNet models supported by this converter use different preprocessing algorithms.

We have implemented three different preprocessor algorithms, as defined in `image_batcher.py`. They are:

| **Preprocessing** | **Resizing**             | **Normalization** | **Mean Subtract** |
| ----------------- | ------------------------ | ----------------- | ----------------- |
| **V2**            | Bilinear Resize          | [-1 to +1] Range  | No                |
| **V1**            | Bicubic Resize + PadCrop | [0 to +1] Range   | No                |
| **V1MS**          | Bicubic Resize + PadCrop | [0 to +1] Range   | Yes               |

**V2:** This is the preprocessor to be used with all EfficientNet V2 models. EfficientNet V2 does not require mean subtraction, so it is never performed for these models.

**V1:** This is the default preprocessor to be used with most EfficientNet V1 models. EfficientNet V1 normally expects mean subtraction to be applied. However, some TensorFlow saved models, such as those downloaded from TFHub, already perform this operation within the graph itself, so it is not required to do it during preprocessing.

**V1MS:** Depending on the saved model exporter, some EfficientNet V1 models may not have the integrated mean subtraction. This is often the case with models exported from the pre-trained *checkpoints*. For those cases, this preprocessor will apply mean subtraction during preprocessing.

These are the supported values for `--preprocessor` and `--calib_preprocessor` arguments used throughout these scripts. Note that choosing an incorrect preprocessor for a model will considerably impact its accuracy. Please take a moment to choose the correct preprocessor to use before performing inference or validation of a model.

### Inference in Python

To classify a set of images with TensorRT, run:

```bash
python3 infer.py \
    --engine /path/to/engine.trt \
    --input /path/to/images \
    --preprocessor V2
```

Where the input path can be either a single image file, or a directory of jpg/png/bmp images. The classification results will be printed out to the console, one image per line, as:

```
<image path>  <predicted class id>  <confidence score>
```

You can also redirect these results to a file, and optionally set a separator character (such as for CSV file creation):

```bash
python3 infer.py \
    --engine /path/to/engine.trt \
    --input /path/to/ILSVRC2012_img_val \
    --preprocessor V2 \
    --separator ',' > results.csv
```

### Validate against Ground Truth

To validate the TensorRT inference results accuracy against ground truth labels, run:

```bash
python3 eval_gt.py \
    --engine /path/to/engine.trt \
    --annotations /path/to/val.txt \
    --input /path/to/images \
    --preprocessor V2
```

The annotations file is expected to have one line per image, where the first column is the image filename, and the second column is the ground truth class label. For example:

```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
[...]
```

> **NOTE:** The ImageNet pre-trained models follow the label mapping introduced by [Caffe](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh), which indexes labels according to their synset number. The validation file for this format can be downloaded from Caffe's ILSVRC2012 auxiliary package at:
>
> http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
>
> You can use the `val.txt` file bundled in this package for ImageNet evaluation purposes.

Upon a successful run of `EfficientNet V2-S` on the `ILSVRC2012_img_val` [ImageNet](https://www.image-net.org/download.php) dataset, for example, you should see something like:

```
Top-1 Accuracy: 83.710%
Top-5 Accuracy: 96.615%
```

### Compare against TensorFlow

Another method to validate the engine is to compare the TensorRT inference results with what TensorFlow produces, to make sure both frameworks give similar results. For this, run:

```bash
python3 compare_tf.py \
    --engine /path/to/engine.trt \
    --saved_model /path/to/saved_model \
    --input /path/to/images \
    --preprocessor V2
```

This can be performed on any set of images, no ground truth is required. The script executes both the TensorFlow saved model and the TensorRT engine simultaneously on the given input images. It then computes the class prediction similarity and RMSE in confidence scores between both outputs.

Upon a successful run, you should see something like:

```
Matching Top-1 class predictions for 4999 out of 5000 images: 99.98%
RMSE between TensorFlow and TensorRT confidence scores: 0.006
```
