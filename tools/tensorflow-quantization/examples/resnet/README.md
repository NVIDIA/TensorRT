## About
This script presents a QAT end-to-end workflow (TF2-to-ONNX) for [ResNet models](https://keras.io/api/applications/resnet/) in `tf.keras.applications`.

### Contents
[Requirements](#requirements) • [Workflow](#workflow) • [Results](#results)  

## Requirements
Install base requirements and prepare data. Please refer to [examples' README](../README.md).

## Workflow

### Step 1: Model Quantization and Fine-tuning

Please run the following to quantize, fine-tune, and save the final graph in SavedModel format (checkpoints are also saved).

```sh
python run_qat_workflow.py
```

### Step 2: Conversion to ONNX
Step 1 already does the conversion from SavedModel to ONNX automatically. For manual steps, please see step 3 in [EfficientNet's README](../efficientnet_b0/README.md).

### Step 3: TensorRT Deployment
Please refer to the [examples' README](../README.md).

## Results
Results obtained on NVIDIA's A100 GPU and TensorRT 8.4 EA.

### ResNet50-v1

| Model    | TF (%)      | TF latency (ms, bs=1) | TRT(%) | TRT latency (ms, bs=1) |
|----------|-------------|-----------------------|--------|------------------------|
| Baseline | 75.05       | 7.95                  | 75.05  | 1.96                   |
| PTQ      | -           | -                     | 74.96  | 0.46                   |
| **QAT**  | 75.11 (ep5) | -                     | 75.12  | 0.45                   |

### ResNet50-v2

| Model    | TF (%)       | TF latency (ms, bs=1) | TRT(%)  | TRT latency (ms, bs=1) |
|----------|--------------|-----------------------|---------|------------------------|
| Baseline | 75.36        | 6.16                  | 75.37   | 2.35                   |
| PTQ      | -            | -                     | 75.48   | 0.57                   |
| **QAT**  | 75.59 (ep5)  | -                     | 75.65   | 0.57                   |

### ResNet101-v1

| Model    | TF (%)       | TF latency (ms, bs=1) | TRT(%) | TRT latency (ms, bs=1) |
|----------|--------------|-----------------------|--------|------------------------|
| Baseline | 76.47        | 15.92                 | 76.48  | 3.84                   |
| PTQ      | -            | -                     | 76.32  | 0.84                   |
| **QAT**  | 76.33 (ep30) | -                     | 76.26  | 0.84                   |

### ResNet101-v2

| Model    | TF (%) | TF latency (ms, bs=1) | TRT(%) | TRT latency (ms, bs=1) |
|----------|--------|-----------------------|--------|------------------------|
| Baseline | 76.89  | 14.13                 | 76.88  | 4.55                   |
| PTQ      | -      | -                     | 76.94  | 1.05                   |
| **QAT**  | 77.20  | -                     | 77.15  | 1.05                   |
> QAT fine-tuning hyper-parameters for ResNet101-v2: `bs=32` (`bs=64` was OOM).

### Notes
- QAT fine-tuning hyper-parameters:
  - Optimizer: `piecewise_sgd`, `lr_schedule=[(1.0,1),(0.1,2),(0.01,7)]` (default)
  - Hyper-parameters: `bs=64, ep=10, lr=0.001`.
  - Added QDQ nodes in Residual connection.
- PTQ calibration: `bs=64`.
