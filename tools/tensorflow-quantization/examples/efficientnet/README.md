## About
This script presents a QAT end-to-end workflow (TF2-to-ONNX) for [EfficientNet](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/efficientnet).

### Contents
[Requirements](#requirements) • [Workflow](#workflow) • [Results](#results)  

## Requirements
1. Install base requirements and prepare data. Please refer to [examples' README](../README.md).

2. Clone the models from Tensorflow model garden:

```
git clone https://github.com/tensorflow/models.git
pushd models && git checkout tags/v2.8.0 && popd
export PYTHONPATH=$PWD/models:$PYTHONPATH
pip install -r models/official/requirements.txt
```

> cd models && git submodule init && git submodule update

3. Download pretrained checkpoints:
   1. B0: https://tfhub.dev/tensorflow/efficientnet/b0/classification/1
   2. B3: https://tfhub.dev/tensorflow/efficientnet/b3/classification/1


## Workflow

### Step 1: Model Quantization and Fine-tuning

* In `run_qat_workflow.py`, please set the `pretrained_ckpt_path` field to the directory of the downloaded checkpoint to start fine-tuning with QAT. All the required hyper-parameters can be set in the `HYPERPARAMS` dictionary.

  Please run the following to quantize, fine-tune, and save the final graph in SavedModel format.

  ```sh
  python run_qat_workflow.py
  ```
  > Update `MODEL_VERSION` to the EfficientNet version you wish to quantize.

### Step 2: Exporting a QAT SavedModel

Once you've fine-tuned the QAT model, export it by running

```sh
python export.py --ckpt <path_to_pretrained_ckpt> --output <saved_model_output_name> --model_version b0
```

This script applies quantization to the model, restores the checkpoint, and exports it in a SavedModel format. This script will generate `eff`  which is a directory containing saved model. We set the overall graph data format to `NCHW` by using `tf.keras.backend.set_image_data_format('channels_first')`. TensorRT expects `NCHW` format for graphs trained with QAT for better optimizations.

Arguments:

* `--ckpt` : Path to fine-tuned QAT checkpoint to be loaded.
* `--output` : Name of output TF saved model.
* `--model_version` : EfficientNet model version, currently supports {`b0`, `b3`}.

### Step 3: Conversion to ONNX

Convert the saved model into ONNX by running

```sh
python -m tf2onnx.convert --saved-model <path_to_saved_model> --output model_qat.onnx  --opset 13
```

By default, tf2onnx uses TF's graph optimizers to performs constant folding after a saved model is loaded.

Arguments:

* `--saved-model` : Name of TF SavedModel
* `--output` : Name of ONNX output graph
* `--opset` : ONNX opset version (opset 13 or higher must be used)

### Step 4: TensorRT Deployment
Please refer to the [examples' README](../README.md).

## Results

This section presents the validation accuracy for the full ImageNet dataset on NVIDIA's A100 GPU and TensorRT 8.4 GA.

### EfficientNet-B0
| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 76.97        | 6.77               |
| PTQ (TensorRT)        | 71.71        | 0.67               |
| **QAT** (TensorRT)    | 75.82        | 0.68               |
> QAT fine-tuning hyper-parameters: `bs64, ep10, lr=0.001, steps_per_epoch=None`

### EfficientNet-B3
| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 81.36        | 10.33              |
| PTQ (TensorRT)        | 78.88        | 1.24               |
| **QAT** (TensorRT)    | 79.48        | 1.23               |
> QAT fine-tuning hyper-parameters: `bs=32, ep20, steps_per_epoch=None, lr=0.0001`

### Notes
- QAT fine-tuning hyper-parameters:
  - Optimizer: `piecewise_sgd`, `lr_schedule=[(1.0, 1), (0.1, 3), (0.01, 6), (0.001, 9), (0.001, 15)]`.
  - Other hyper-parameters are under each model's results table.
- PTQ calibration: `bs=64`
- EfficientNet model quantization:
  - QDQ nodes added in Residual connection (fix added to ResidualQDQCustomCase for `Conv-BN-Activation-Dropout` pattern),
  - Global Average Pooling,
  - Multiply layer in SE block.
