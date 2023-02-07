## About
This script presents a QAT end-to-end workflow (TF2-to-ONNX) for [MobileNet models](https://keras.io/api/applications/mobilenet/) in `tf.keras.applications`.

### Contents
[Requirements](#requirements) • [Workflow](#workflow) • [Results](#results)  

## Requirements
Install base requirements and prepare data. Please refer to [examples' README](../README.md).

## Workflow

### Step 1: Model Quantization and Fine-tuning
> Similar to [ResNet](../resnet): different model and different input pre-processing (`mobilenet`).

Please run the following to quantize, fine-tune, and save the final graph in SavedModel format (checkpoints are also saved).

```sh
python run_qat_workflow.py
```

### Step 2: Conversion to ONNX
Step 1 already does the conversion from SavedModel to ONNX automatically. For manual steps, please see step 3 in [EfficientNet's README](../efficientnet_b0/README.md).

### Step 3: TensorRT Deployment
Please refer to the [examples' README](../README.md).

## Results
Results obtained on NVIDIA's A100 GPU and TensorRT 8.4.10.1.

### MobileNet-v1

| Model    | TF (%)      | TF latency (ms, bs=1) | TRT(%) | TRT latency (ms, bs=1) |
|----------|-------------|-----------------------|--------|------------------------|
| Baseline | 70.60       | 1.99                  | 70.60  | 0.32                   |
| PTQ      | -           | -                     | 69.31  | 0.16                   |
| **QAT**  | 70.51 (ep2) | 50.49                 | 70.43  | 0.16                   |

**Note**: no residual connections exist in MobileNet-v1.

### MobileNet-v2

| Model    | TF (%)      | TF latency (ms, bs=1) | TRT(%)   | TRT latency (ms, bs=1) |
|----------|-------------|-----------------------|----------|------------------------|
| Baseline | 71.77       | 3.71                  | 71.77    | 0.55                   |
| PTQ      | -           | -                     | 70.87    | 0.30                   |
| **QAT**  | 71.68 (ep1) | 74.27                 | 71.62    | 0.30                   |

**Note**: residual connections exist in MobileNet-v2.

### Notes
- QAT fine-tuning hyper-parameters:
  - Optimizer: `piecewise_sgd`, `lr_schedule=[(1.0, 1), (0.1, 2), (0.01, 7)]` (default)
  - Hyper-parameters: `bs=64, ep=10, lr=0.001`
- PTQ calibration: `bs=64`.
- MobileNet-v3 might not show good acceleration in TensorRT due to its architecture (`Conv->BN->((Add->Clip->Mul), ())->Mul`), which is not a kernel fusion in TRT. 
