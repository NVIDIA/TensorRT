# About
This folder contains the Quantization-Aware Training (QAT) workflow for [standard networks](#step-1-model-quantization-and-fine-tuning).

The QAT end-to-end workflow (TF2-to-ONNX) consists of the following steps:
- Model quantization using the `quantize_model` function with `NVIDIA` quantization scheme.
- QAT model fine-tuning (saves checkpoints).
- Baseline vs QAT models accuracy comparison.
- QAT model conversion to SavedModel format.
- Conversion of SavedModel to ONNX.
- TensorRT engine building via ONNX file and inference.

# Requirements
## 1. Base requirements
1. Install `tensorflow-quantization` toolkit.
2. Install additional requirements: `pip install -r requirements.txt`.
3. (Optional) Install TensorRT for full workflow support (needed for `infer_engine.py`).

**Note**: For CLI run, please go to the cloned repository's root directory and run `export PYTHONPATH=$PWD`, so that the `examples` folder is available for import.

## 2. Data preparation
### A. Raw data download
We are using the ImageNet 2012 dataset (task 1 - image classification), which requires manual downloads due to terms of access agreements.
Please login/sign-up on [the ImageNet website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and download the "train/validation data".
This is needed for the QAT model fine-tuning, and it is also used to evaluate the Baseline and QAT models.

### B. Conversion to tfrecord
Our workflow supports `tfrecord` format, so please follow the following instructions (modified from [TensorFlow's instructions](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy)) to convert the downloaded `.tar` ImageNet files to the required format:

1. Set `IMAGENET_HOME=/path/to/imagenet/tar/files` in [`data/imagenet_data_setup.sh`](data/imagenet_data_setup.sh).
2. Download [`imagenet_to_gcs.py`](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) to `$IMAGENET_HOME`.
3. Run `./data/imagenet_data_setup.sh`.

# Workflow

## Step 1: Model quantization and fine-tuning
Model quantization, fine-tuning, and conversion to ONNX.

Example models:

| Model         | Task             | Script - QAT Workflow        |
|---------------|------------------|------------------------------|
| ResNet        | Classification   | [resnet](resnet)             |
| EfficientNet  | Classification   | [efficientnet](efficientnet) |
| MobileNet     | Classification   | [mobilenet](mobilenet)       |
| Inception     | Classification   | [inception](inception)       |
> For each model's performance results, please refer to the toolkit's User Guide ("Model Zoo").

## Step 2: TensorRT deployment
Build the TensorRT engine and evaluate its latency and accuracy performances.

#### 2.1. Build TensorRT engine from ONNX
Convert the ONNX model into a TensorRT engine (also obtains latency measurements):

```sh
trtexec --onnx=model_qat.onnx --int8 --saveEngine=model_qat.engine --verbose
```

Arguments:
* `--onnx`: Path to QAT onnx graph.
* `--saveEngine`: Output filename of TensorRT engine.
* `--verbose`: Flag to enable verbose logging.

#### 2.2. TensorRT Inference
Obtain accuracy results on the validation dataset:

```sh
python infer_engine.py --engine=<path_to_trt_engine> --data_dir=<path_to_tfrecord_val_data> -b=<batch_size>
```

Arguments:
- `-e, --engine`: TensorRT engine filename (to load).
- `-m, --model_name`: Name of the model, needed to choose the appropriate input pre-processing. Options={`resnet_v1` (default), `resnet_v2`, `efficientnet_b0`, `efficientnet_b3`, `mobilenet_v1`, `mobilenet_v2`}.
- `-d, --data_dir`: Path to directory of input images in **tfrecord format** (`data["validation"]`).
- `-k, --top_k_value` (default=1): Value of `K` for the top-K predictions used in the accuracy calculation.
- `-b, --batch_size` (default=1): Number of inputs to send in parallel (up to max batch size of engine).
- `--log_file`: Filename to save logs.

Outputs:
- `.log` file: contains the engine's performance accuracy. 

# Additional resources

The following resources provide a deeper understanding about Quantization aware training, TF2ONNX and importing a model into TensorRT using Python.

**Quantization Aware Training**

* <a href="https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/">Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT</a>

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)
- [Quantization Aware Training guide](https://www.tensorflow.org/model_optimization/guide/quantization/training)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**Parsers**

- [TF2ONNX Converter](https://github.com/onnx/tensorflow-onnx)
- [ONNX Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**Documentation**

- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Importing A Model Using A Parser In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)
- [NVIDIA’s TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
