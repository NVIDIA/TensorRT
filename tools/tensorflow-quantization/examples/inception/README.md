## About
This script presents a QAT end-to-end workflow (TF2-to-ONNX) for [Inception models](https://keras.io/api/applications/inceptionv3/) in `tf.keras.applications`.

### Contents
[Requirements](#requirements) • [Workflow](#workflow) • [Results](#results)  

## Requirements
Install base requirements and prepare data. Please refer to [examples' README](../README.md).

## Workflow

### Step 1: Model Quantization and Fine-tuning
> Similar to [ResNet](../resnet): different model and different input pre-processing.

Please run the following to quantize, fine-tune, and save the final graph in SavedModel format (checkpoints are also saved).

```sh
python run_qat_workflow.py
```

### Step 2: Conversion to ONNX
Step 1 already does the conversion from SavedModel to ONNX automatically. For manual steps, please see step 3 in [EfficientNet's README](../efficientnet_b0/README.md).

### Step 3: TensorRT Deployment
Please refer to the [examples' README](../README.md).

## Results
Results obtained on NVIDIA's A100 GPU and TensorRT 8.4.2.4 (GA Update 1).

### Inception-v3

| Model    | TF (%) | TF latency (ms, bs=1) | TRT(%) | TRT latency (ms, bs=1) |
|----------|--------|-----------------------|--------|------------------------|
| Baseline | 77.86  | 9.01                  | 77.86  | 1.39                   |
| PTQ      | -      | -                     | 77.73  | 0.82                   |
| **QAT**  | 78.11  | 101.97                | 78.08  | 0.82                   |

### Notes
- Optimization: MaxPool needs to be quantized to trigger horizontal fusion in Concat layer.
- QAT fine-tuning hyper-params:
  - Optimizer: `piecewise_sgd`, `lr_schedule=[(1.0, 1), (0.1, 2), (0.01, 7)]` (default)
  - Hyper-parameters: `bs=64, ep=10, lr=0.001, steps_per_epoch=500`
- PTQ calibration: `bs=64`.
